"""The MTP (multi-token prediction) head shipped with Qwen3.8-27B, as a draft
model. Semantics follow vLLM's Qwen3_5MultiTokenPredictor:

    x = fc(cat(pre_fc_norm_embedding(embed(next_token)),
               pre_fc_norm_hidden(target_hidden)))        # [.., 2*hidden] -> hidden
    x = attention_layer(x, position)                      # one full-attention block with its own KV
    logits = lm_head(norm(x))

where target_hidden is the target model's post-final-norm hidden state at the
position before next_token (the same vector lm_head sampled next_token from),
and the block sits at next_token's position. Chaining feeds the block's own
normed output as the next target_hidden. Embedding and lm_head are the
target's (mtp_use_dedicated_embeddings = false).
"""
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from . import ops
from .config import ModelConfig
from .model import AttnWeights, LayerWeights, attn_forward, mlp_forward
from .quant import Linear
from .state import State


@dataclass
class MTPWeights:
    fc: Linear                        # 2*hidden -> hidden
    layer: LayerWeights               # full-attention block
    norm: torch.Tensor
    pre_norm_emb: torch.Tensor
    pre_norm_hidden: torch.Tensor


def build_mtp(cfg: ModelConfig, mtp: dict, device) -> MTPWeights:
    """mtp: 'mtp.*' name -> CPU bf16 tensor (from load_packed(..., with_mtp=True))."""
    dev = lambda n: mtp[n].to(device)
    p = "mtp.layers.0."
    m = p + "self_attn."
    mixer = AttnWeights(
        qkv=Linear(torch.cat([dev(m + "q_proj.weight"), dev(m + "k_proj.weight"), dev(m + "v_proj.weight")])),
        o=Linear(dev(m + "o_proj.weight")),
        q_norm_w=dev(m + "q_norm.weight"), k_norm_w=dev(m + "k_norm.weight"))
    layer = LayerWeights(
        ln1=dev(p + "input_layernorm.weight"), ln2=dev(p + "post_attention_layernorm.weight"), mixer=mixer,
        gate_up=Linear(torch.cat([dev(p + "mlp.gate_proj.weight"), dev(p + "mlp.up_proj.weight")])),
        down=Linear(dev(p + "mlp.down_proj.weight")))
    return MTPWeights(fc=Linear(dev("mtp.fc.weight")), layer=layer, norm=dev("mtp.norm.weight"),
                      pre_norm_emb=dev("mtp.pre_fc_norm_embedding.weight"),
                      pre_norm_hidden=dev("mtp.pre_fc_norm_hidden.weight"))


class MTPHead:
    """Eager MTP head with its own single-layer KV cache. Positions are the target's
    positions: the block for next_token at position p reads its cache rows < p."""

    def __init__(self, cfg: ModelConfig, w: MTPWeights, embed: torch.Tensor, lm_head, max_len: int, device="cuda"):
        self.cfg = cfg
        self.w = w
        self.embed = embed
        self.lm_head = lm_head
        self.device = torch.device(device)
        self.cfg1 = replace(cfg, layer_types=("full_attention",), n_layers=1)
        self.state = State(self.cfg1, max_len, self.device)
        self.cos, self.sin = ops.rope_table(max_len, cfg.rotary_dim, cfg.rope_theta, self.device)

    def reset(self):
        self.state.reset()

    def set_pos(self, pos: int):
        """Rewind or advance the cache position (rows beyond are simply overwritten)."""
        self.state.pos = pos
        self.state.pos_t.fill_(pos)

    @torch.no_grad()
    def forward(self, next_tokens: torch.Tensor, target_hidden: torch.Tensor):
        """next_tokens [T], target_hidden [T, hidden] (post-final-norm, one position earlier)
        -> (logits [T, vocab], normed hidden [T, hidden]). Block positions are state.pos..;
        advances the cache by T."""
        cfg = self.cfg
        e = ops.rmsnorm(self.embed[next_tokens], self.w.pre_norm_emb, cfg.eps)
        h = ops.rmsnorm(target_hidden, self.w.pre_norm_hidden, cfg.eps)
        x = self.w.fc(torch.cat([e, h], dim=-1))
        lw = self.w.layer
        n = ops.rmsnorm(x, lw.ln1, cfg.eps)
        x = x + attn_forward(n, lw.mixer, cfg, self.state, 0, self.cos, self.sin)
        n = ops.rmsnorm(x, lw.ln2, cfg.eps)
        x = x + mlp_forward(n, lw)
        self.state.advance(next_tokens.shape[0])
        hn = ops.rmsnorm(x, self.w.norm, cfg.eps)
        return self.lm_head(hn), hn
