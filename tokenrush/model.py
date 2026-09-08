"""The text path of Qwen3.8-27B as plain functions over weight containers.

Layers are pure functions of (weights, state, activations); there is no
nn.Module and no autograd. Activations are [T, hidden] with bs=1 implicit.
"""
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn.functional as F

from . import ops
from .config import ModelConfig
from .quant import Linear, QLinear
from .state import State

Lin = Union[Linear, QLinear]


@dataclass
class GDNWeights:
    in_qkvz: Lin           # hidden -> 2*qk_dim + val_dim (conv path) | val_dim (gate z), one GEMV
    in_ba: torch.Tensor    # [2*HV, hidden] bf16: b rows then a rows
    conv_w: torch.Tensor   # [conv_dim, K] bf16
    A: torch.Tensor        # [HV] fp32, = -exp(A_log)
    dt_bias: torch.Tensor  # [HV] fp32
    norm_w: torch.Tensor   # [gdn_v_dim] bf16
    out: Lin               # val_dim -> hidden


@dataclass
class AttnWeights:
    qkv: Lin               # hidden -> n_heads*head_dim*2 (query|gate per head) | kv_heads*head_dim (k) | same (v)
    o: Lin
    q_norm_w: torch.Tensor  # [head_dim]
    k_norm_w: torch.Tensor


@dataclass
class LayerWeights:
    ln1: torch.Tensor
    ln2: torch.Tensor
    mixer: Union[GDNWeights, AttnWeights]
    gate_up: Lin           # hidden -> 2*ffn: gate rows then up rows
    down: Lin


@dataclass
class ModelWeights:
    embed: torch.Tensor    # [vocab, hidden] bf16
    layers: list
    final_norm: torch.Tensor
    lm_head: Lin

    @property
    def nbytes(self):
        n = self.embed.numel() * 2 + self.lm_head.nbytes
        for l in self.layers:
            n += l.gate_up.nbytes + l.down.nbytes
            m = l.mixer
            if isinstance(m, GDNWeights):
                n += m.in_qkvz.nbytes + m.out.nbytes + m.in_ba.numel() * 2 + m.conv_w.numel() * 2
            else:
                n += m.qkv.nbytes + m.o.nbytes
        return n


# ----------------------------------------------------------------- layers


def gdn_forward(x: torch.Tensor, w: GDNWeights, cfg: ModelConfig, state: State, slot: int) -> torch.Tensor:
    T = x.shape[0]
    qkv, z = torch.split(w.in_qkvz(x), [cfg.conv_dim, cfg.gdn_val_dim], dim=-1)   # [T, conv_dim], [T, val_dim]
    b, a = torch.split(F.linear(x, w.in_ba), [cfg.gdn_v_heads, cfg.gdn_v_heads], dim=-1)   # [T, HV] each
    if T == 1:
        qkv = ops.conv_step(qkv, state.conv[slot], w.conv_w)
    else:
        qkv = ops.conv_prefill(qkv, state.conv[slot], w.conv_w)
    q, k, v = torch.split(qkv, [cfg.gdn_qk_dim, cfg.gdn_qk_dim, cfg.gdn_val_dim], dim=-1)
    q = q.view(T, cfg.gdn_k_heads, cfg.gdn_k_dim)
    k = k.view(T, cfg.gdn_k_heads, cfg.gdn_k_dim)
    v = v.view(T, cfg.gdn_v_heads, cfg.gdn_v_dim)
    beta = torch.sigmoid(b)                                      # [T, HV] bf16
    g = w.A * F.softplus(a.float() + w.dt_bias)                  # [T, HV] fp32
    if T == 1:
        o = ops.gdn_step(q[0], k[0], v[0], g[0], beta[0], state.rec[slot])[None]
    else:
        o = ops.gdn_prefill(q, k, v, g, beta, state.rec[slot])
    o = ops.gated_rmsnorm(o.reshape(-1, cfg.gdn_v_dim), w.norm_w, z.reshape(-1, cfg.gdn_v_dim), cfg.eps)
    return w.out(o.view(T, cfg.gdn_val_dim))


def attn_forward(x: torch.Tensor, w: AttnWeights, cfg: ModelConfig, state: State, slot: int,
                 cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    T = x.shape[0]
    pos = state.pos
    hd = cfg.head_dim
    kv_dim = cfg.n_kv_heads * hd
    qg, k, v = torch.split(w.qkv(x), [cfg.n_heads * 2 * hd, kv_dim, kv_dim], dim=-1)
    qg = qg.view(T, cfg.n_heads, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:]
    gate = gate.reshape(T, cfg.n_heads * hd)
    q = ops.rmsnorm(q, w.q_norm_w, cfg.eps).transpose(0, 1)                       # [Hq, T, D]
    k = ops.rmsnorm(k.view(T, cfg.n_kv_heads, hd), w.k_norm_w, cfg.eps).transpose(0, 1)
    v = v.view(T, cfg.n_kv_heads, hd).transpose(0, 1)
    q = ops.apply_rope(q, cos[pos:pos + T], sin[pos:pos + T])
    k = ops.apply_rope(k, cos[pos:pos + T], sin[pos:pos + T])
    state.k[slot, :, pos:pos + T] = k
    state.v[slot, :, pos:pos + T] = v
    K = state.k[slot, :, :pos + T]
    V = state.v[slot, :, :pos + T]
    if T == 1:
        o = ops.attn_decode(q, K, V)
    else:
        o = ops.attn_prefill(q, K, V, pos)
    o = o.transpose(0, 1).reshape(T, cfg.n_heads * hd) * torch.sigmoid(gate)
    return w.o(o)


def mlp_forward(x: torch.Tensor, w: LayerWeights) -> torch.Tensor:
    gate, up = torch.chunk(w.gate_up(x), 2, dim=-1)
    return w.down(F.silu(gate) * up)


# ----------------------------------------------------------------- engine


class Engine:
    def __init__(self, cfg: ModelConfig, weights: ModelWeights, max_len: int, device="cuda"):
        self.cfg = cfg
        self.w = weights
        self.device = torch.device(device)
        self.state = State(cfg, max_len, self.device)
        self.cos, self.sin = ops.rope_table(max_len, cfg.rotary_dim, cfg.rope_theta, self.device)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, all_logits: bool = False) -> torch.Tensor:
        """tokens [T] on device, at positions state.pos .. state.pos+T-1. Advances the
        state by T. Returns logits [1, vocab] for the last token (or [T, vocab])."""
        cfg, st = self.cfg, self.state
        T = tokens.shape[0]
        assert st.pos + T <= st.max_len, "context exceeds the preallocated cache"
        x = self.w.embed[tokens]
        for li, lw in enumerate(self.w.layers):
            h = ops.rmsnorm(x, lw.ln1, cfg.eps)
            if cfg.layer_types[li] == "linear_attention":
                h = gdn_forward(h, lw.mixer, cfg, st, st.gdn_slot[li])
            else:
                h = attn_forward(h, lw.mixer, cfg, st, st.attn_slot[li], self.cos, self.sin)
            x = x + h
            h = ops.rmsnorm(x, lw.ln2, cfg.eps)
            x = x + mlp_forward(h, lw)
        st.pos += T
        if not all_logits:
            x = x[-1:]
        x = ops.rmsnorm(x, self.w.final_norm, cfg.eps)
        return self.w.lm_head(x)

    def prefill(self, tokens: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        logits = None
        for s in range(0, tokens.shape[0], chunk):
            logits = self.forward(tokens[s:s + chunk])
        return logits

    def decode(self, token: torch.Tensor) -> torch.Tensor:
        return self.forward(token.view(1))

    def reset(self):
        self.state.reset()
