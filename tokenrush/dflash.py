"""DFlash2, z-lab's block-diffusion draft for Qwen3.8-27B, on our ops.

One draft forward turns the block [t, mask x 7] into 7 draft tokens, conditioned
on the target's residual stream after layers `target_layer_ids` (5, 19, 33, 47,
61 for this checkpoint): those five vectors per context position are
concatenated, projected by `fc`, normed, and injected as extra keys/values into
every draft layer. The block rows attend bidirectionally among themselves and to
the context rows within a 2048-position window on either side (the draft keeps
its own K/V of context rows; block rows are never cached). Each layer also runs
two grouped dynamic causal convolutions over the block (kernel 2) around its
attention and its MLP, and a rank-256 candidate selector chains the top-16
candidates per position into the draft sequence.

Follows z-lab/dflash/model.py (MIT). Norms here are plain-gain Qwen3 RMSNorm
(weight * x), not Qwen3.5's (1 + weight).
"""
import glob
import json
import os
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .config import ModelConfig
from .quant import DEFAULT_BACKEND, Linear, QLinear, quantize_int4
from .state import State


def rmsnorm_plain(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3RMSNorm: normalize in fp32, cast, multiply by the weight (no +1)."""
    xf = x.float()
    out = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return weight * out


@dataclass
class DFlashLayer:
    ln1: torch.Tensor
    ln2: torch.Tensor
    q: object; k: object; v: object; o: object          # Linear / QLinear
    q_norm_w: torch.Tensor
    k_norm_w: torch.Tensor
    gate_up: object
    down: object
    attn_conv_base: torch.Tensor                        # [2 (prepare/finish), ks, hidden]
    attn_conv_proj: object                              # hidden -> 2*ks*groups
    mlp_conv_base: torch.Tensor
    mlp_conv_proj: object


@dataclass
class DFlashWeights:
    fc: object                                          # 5*hidden -> hidden
    hidden_norm: torch.Tensor
    layers: list
    norm: torch.Tensor
    sel_hidden_proj: object                             # hidden -> rank
    sel_pred_cb: torch.Tensor                           # [vocab, rank]
    sel_succ_cb: torch.Tensor
    # config
    target_layer_ids: tuple
    block_size: int
    mask_token_id: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    conv_ks: int
    conv_group: int
    sel_top_k: int
    window: int
    eps: float
    rope_theta: float


def load_dflash(path: str, device="cuda", int4: bool = False) -> DFlashWeights:
    cfg = json.load(open(os.path.join(path, "config.json")))
    dc = cfg["dflash_config"]
    t = {}
    for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for n in f.keys():
                t[n] = f.get_tensor(n)
    dev = lambda n: t[n].to(device)

    def lin(names, split_k=1):
        w = torch.cat([dev(n) for n in names])
        return QLinear(*quantize_int4(w), backend=DEFAULT_BACKEND, split_k=split_k) if int4 else Linear(w)

    layers = []
    for i in range(cfg["num_hidden_layers"]):
        p = f"layers.{i}."
        layers.append(DFlashLayer(
            ln1=dev(p + "input_layernorm.weight"), ln2=dev(p + "post_attention_layernorm.weight"),
            q=lin([p + "self_attn.q_proj.weight"]), k=lin([p + "self_attn.k_proj.weight"]),
            v=lin([p + "self_attn.v_proj.weight"]), o=lin([p + "self_attn.o_proj.weight"], 4 if int4 else 1),
            q_norm_w=dev(p + "self_attn.q_norm.weight"), k_norm_w=dev(p + "self_attn.k_norm.weight"),
            gate_up=lin([p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"]),
            down=lin([p + "mlp.down_proj.weight"], 4 if int4 else 1),
            attn_conv_base=dev(p + "attention_conv.base_kernel"), attn_conv_proj=lin([p + "attention_conv.kernel_projection.weight"]),
            mlp_conv_base=dev(p + "mlp_conv.base_kernel"), mlp_conv_proj=lin([p + "mlp_conv.kernel_projection.weight"])))
    return DFlashWeights(
        fc=lin(["fc.weight"]), hidden_norm=dev("hidden_norm.weight"), layers=layers, norm=dev("norm.weight"),
        sel_hidden_proj=lin(["candidate_selector.hidden_projection.weight"]),
        sel_pred_cb=dev("candidate_selector.predecessor_codebook"), sel_succ_cb=dev("candidate_selector.successor_codebook"),
        target_layer_ids=tuple(dc["target_layer_ids"]), block_size=int(dc["block_size"]), mask_token_id=int(dc["mask_token_id"]),
        n_heads=cfg["num_attention_heads"], n_kv_heads=cfg["num_key_value_heads"], head_dim=cfg["head_dim"],
        conv_ks=int(dc["conv_kernel_size"]), conv_group=int(dc["conv_group_size"]), sel_top_k=int(dc["selector_top_k"]),
        window=int(cfg["sliding_window"]), eps=cfg["rms_norm_eps"], rope_theta=cfg["rope_parameters"]["rope_theta"])


def grouped_dynamic_conv(h: torch.Tensor, dyn: torch.Tensor, base: torch.Tensor, group: int) -> torch.Tensor:
    """h [L, H]; dyn [L, ks, groups]; base [ks, H]. Causal depthwise conv over the block
    rows with a per-position, per-group dynamic kernel added to a static one."""
    L, H = h.shape
    groups = H // group
    blocks = h.view(L, groups, group)
    out = torch.zeros_like(blocks)
    for off in range(base.shape[0]):
        vals = blocks if off == 0 else F.pad(blocks[:-off], (0, 0, 0, 0, off, 0))
        out = out + base[off].view(1, groups, group).to(h.dtype) * vals
        out = torch.addcmul(out, dyn[:, off].view(L, groups, 1), vals)
    return out.view(L, H)


class DFlashDraft:
    """Eager DFlash2 draft with its own context K/V cache. Positions are the target's."""

    def __init__(self, w: DFlashWeights, embed: torch.Tensor, lm_head, hidden: int, max_len: int, device="cuda",
                 kv_dtype=torch.bfloat16):
        self.w = w
        self.embed = embed
        self.lm_head = lm_head
        self.hidden = hidden
        self.device = torch.device(device)
        self.max_len = max_len
        L = len(w.layers)
        # the draft attends within a window, so its context cache is a ring of ring_size
        # rows (position mod ring_size); ring_size > window + block so a row is never
        # overwritten before it drops out of every query's window
        self.ring = 1
        while self.ring < w.window + w.block_size + 8:
            self.ring *= 2
        self.k = torch.zeros(L, w.n_kv_heads, self.ring, w.head_dim, device=device, dtype=kv_dtype)
        self.v = torch.zeros_like(self.k)
        self.ctx = 0                                   # context rows cached: positions 0..ctx-1
        inv = 1.0 / (w.rope_theta ** (torch.arange(0, w.head_dim, 2, device=device).float() / w.head_dim))
        fr = torch.arange(max_len, device=device).float()[:, None] * inv[None]
        emb = torch.cat([fr, fr], -1)
        self.cos, self.sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)

    def reset(self):
        self.ctx = 0

    @staticmethod
    def _rope(x, cos, sin):                            # x [.., T, D], cos/sin [T, D]
        h = x.shape[-1] // 2
        rot = torch.cat([-x[..., h:], x[..., :h]], -1)
        return x * cos + rot * sin

    @torch.no_grad()
    def forward(self, features: torch.Tensor, block: torch.Tensor):
        """features [n_ctx, 5*hidden]: the target's residual stream after the five layers
        for context positions ctx .. ctx+n_ctx-1 (new since the last call); block [B]:
        the committed token followed by mask tokens, at positions ctx+n_ctx ...
        Returns (draft tokens [B-1], normed block hidden [B, hidden]). Caches the
        context rows' K/V; advances ctx by n_ctx."""
        w = self.w
        n_ctx, B = features.shape[0], block.shape[0]
        p0 = self.ctx + n_ctx                          # first block position
        if n_ctx:
            th = rmsnorm_plain(w.fc(features), w.hidden_norm, w.eps)          # [n_ctx, H]
        else:
            th = torch.empty(0, self.hidden, device=self.device, dtype=torch.bfloat16)
        h = self.embed[block]                                                  # [B, H]
        ctx_pos = torch.arange(self.ctx, p0, device=self.device)
        blk_pos = torch.arange(p0, p0 + B, device=self.device)
        lo = max(0, p0 - (w.window - 1))               # earliest context position row 0 can see
        keys_pos = torch.arange(lo, p0, device=self.device)                    # context keys; rows mask their own bound
        # mask: block row i (position p0+i) sees context key j if (p0+i) - j < window; all block rows
        vis_ctx = (blk_pos[:, None] - keys_pos[None, :]) < w.window            # [B, n_keys]
        mask = torch.cat([vis_ctx, torch.ones(B, B, dtype=torch.bool, device=self.device)], 1)
        for li, lw in enumerate(w.layers):
            n = rmsnorm_plain(h, lw.ln1, w.eps)
            dyn = lw.attn_conv_proj(n).view(B, 2, w.conv_ks, -1)
            n = grouped_dynamic_conv(n, dyn[:, 0], lw.attn_conv_base[0], w.conv_group)
            q = rmsnorm_plain(lw.q(n).view(B, w.n_heads, w.head_dim), lw.q_norm_w, w.eps)
            k_blk = rmsnorm_plain(lw.k(n).view(B, w.n_kv_heads, w.head_dim), lw.k_norm_w, w.eps)
            v_blk = lw.v(n).view(B, w.n_kv_heads, w.head_dim)
            q = self._rope(q.transpose(0, 1), self.cos[blk_pos], self.sin[blk_pos])          # [Hq, B, D]
            k_blk = self._rope(k_blk.transpose(0, 1), self.cos[blk_pos], self.sin[blk_pos])
            if n_ctx:
                k_ctx = rmsnorm_plain(lw.k(th).view(n_ctx, w.n_kv_heads, w.head_dim), lw.k_norm_w, w.eps)
                v_ctx = lw.v(th).view(n_ctx, w.n_kv_heads, w.head_dim)
                k_ctx = self._rope(k_ctx.transpose(0, 1), self.cos[ctx_pos], self.sin[ctx_pos])  # [Hkv, n_ctx, D]
                self.k[li].index_copy_(1, ctx_pos % self.ring, k_ctx.to(self.k.dtype))
                self.v[li].index_copy_(1, ctx_pos % self.ring, v_ctx.transpose(0, 1).to(self.v.dtype))
            krows = keys_pos % self.ring
            K = torch.cat([self.k[li].index_select(1, krows).to(q.dtype), k_blk], 1)          # [Hkv, n_keys+B, D]
            V = torch.cat([self.v[li].index_select(1, krows).to(q.dtype), v_blk.transpose(0, 1)], 1)
            o = F.scaled_dot_product_attention(q[None], K[None], V[None], attn_mask=mask, enable_gqa=True)[0]
            o = lw.o(o.transpose(0, 1).reshape(B, w.n_heads * w.head_dim))
            o = grouped_dynamic_conv(o, dyn[:, 1], lw.attn_conv_base[1], w.conv_group)
            h = h + o
            n = rmsnorm_plain(h, lw.ln2, w.eps)
            dyn = lw.mlp_conv_proj(n).view(B, 2, w.conv_ks, -1)
            n = grouped_dynamic_conv(n, dyn[:, 0], lw.mlp_conv_base[0], w.conv_group)
            g, u = lw.gate_up(n).chunk(2, -1)
            m = lw.down(F.silu(g) * u)
            m = grouped_dynamic_conv(m, dyn[:, 1], lw.mlp_conv_base[1], w.conv_group)
            h = h + m
        self.ctx = p0
        hn = rmsnorm_plain(h, w.norm, w.eps)                                    # [B, H]
        return self.select(hn[1:], block[0]), hn

    def select(self, hid: torch.Tensor, anchor: torch.Tensor, draft_head=None) -> torch.Tensor:
        """Candidate selector: top-k per position by the lm_head, chained by a bilinear
        codebook score with the predecessor token. hid [B-1, H] -> tokens [B-1].
        draft_head: optional (linear over a vocabulary slice, id map) to read less."""
        w = self.w
        if draft_head is None:
            logits = self.lm_head(hid)                                          # [B-1, vocab] bf16
            unary, cands = logits.topk(w.sel_top_k, dim=-1)                     # [B-1, k], bf16 as in the reference
        else:
            head, idmap = draft_head
            unary, cands = head(hid).topk(w.sel_top_k, dim=-1)
            cands = idmap.index_select(0, cands.reshape(-1)).view_as(cands)
        hp = w.sel_hidden_proj(hid)                                             # [B-1, rank]
        pred = anchor.view(1)
        out = []
        for pos in range(hid.shape[0]):
            # bf16 throughout, matching the reference's einsum so near-ties resolve the same way
            score = unary[pos] + ((w.sel_pred_cb.index_select(0, pred) * hp[pos:pos + 1])
                                  @ w.sel_succ_cb.index_select(0, cands[pos]).t())[0]
            j = score.argmax().view(1)
            pred = cands[pos].gather(0, j)                                      # no host read: graph-safe
            out.append(pred)
        return torch.cat(out)


    # ------------------------------------------------ graph-capturable paths

    @torch.no_grad()
    def prime(self, features: torch.Tensor):
        """Cache the context K/V for `features` [n, 5*hidden] at positions ctx.. (the
        prompt), computing nothing for a block. Eager, any n."""
        w = self.w
        n = features.shape[0]
        th = rmsnorm_plain(w.fc(features), w.hidden_norm, w.eps)
        pos = torch.arange(self.ctx, self.ctx + n, device=self.device)
        rows = pos % self.ring
        for li, lw in enumerate(w.layers):
            k = rmsnorm_plain(lw.k(th).view(n, w.n_kv_heads, w.head_dim), lw.k_norm_w, w.eps)
            k = self._rope(k.transpose(0, 1), self.cos[pos], self.sin[pos])
            self.k[li].index_copy_(1, rows, k.to(self.k.dtype))
            self.v[li].index_copy_(1, rows, lw.v(th).view(n, w.n_kv_heads, w.head_dim).transpose(0, 1).to(self.v.dtype))
        self.ctx += n

    def forward_static(self, features: torch.Tensor, n_ctx: torch.Tensor, ctx_t: torch.Tensor, block: torch.Tensor,
                       draft_head=None):
        """Shape-static forward for the graph. features [R, 5*hidden] with the first
        n_ctx (device scalar) rows valid, at positions ctx_t .. ; block [B] at positions
        ctx_t + n_ctx ... All R context rows are written to the cache (the invalid ones
        land beyond the block start and are overwritten before anything reads them); the
        attention reads a fixed window of W keys ending at the block start, masked.
        Returns draft tokens [B-1]. draft_head: the lm_head slice for the selector."""
        w = self.w
        R, B, W = features.shape[0], block.shape[0], w.window
        dev = self.device
        p0 = ctx_t + n_ctx                                                       # [1] device
        th = rmsnorm_plain(w.fc(features), w.hidden_norm, w.eps)                # [R, H]
        h = self.embed[block]
        ar_r = torch.arange(R, device=dev)
        ctx_pos = ctx_t + ar_r                                                   # [R]
        blk_pos = p0 + torch.arange(B, device=dev)                               # [B]
        cos_c, sin_c = self.cos.index_select(0, ctx_pos), self.sin.index_select(0, ctx_pos)
        cos_b, sin_b = self.cos.index_select(0, blk_pos), self.sin.index_select(0, blk_pos)
        from . import fused as fused_k
        pending = None                                  # residual contribution not yet added to h
        for li, lw in enumerate(w.layers):
            h, n = fused_k.add_rmsnorm(h, pending, lw.ln1, w.eps, plain=True)
            dyn = lw.attn_conv_proj(n).view(B, 2, w.conv_ks, -1)
            n = fused_k.grouped_conv(n, dyn[:, 0], lw.attn_conv_base[0], w.conv_group)
            q = rmsnorm_plain(lw.q(n).view(B, w.n_heads, w.head_dim), lw.q_norm_w, w.eps)
            k_ctx = rmsnorm_plain(lw.k(th).view(R, w.n_kv_heads, w.head_dim), lw.k_norm_w, w.eps)
            v_ctx = lw.v(th).view(R, w.n_kv_heads, w.head_dim)
            k_blk = rmsnorm_plain(lw.k(n).view(B, w.n_kv_heads, w.head_dim), lw.k_norm_w, w.eps)
            v_blk = lw.v(n).view(B, w.n_kv_heads, w.head_dim)
            q = self._rope(q.transpose(0, 1), cos_b, sin_b)
            k_ctx = self._rope(k_ctx.transpose(0, 1), cos_c, sin_c)
            k_blk = self._rope(k_blk.transpose(0, 1), cos_b, sin_b)
            self.k[li].index_copy_(1, ctx_pos % self.ring, k_ctx.to(self.k.dtype))
            self.v[li].index_copy_(1, ctx_pos % self.ring, v_ctx.transpose(0, 1).to(self.v.dtype))
            # context keys < p0 within the window (our split kernel, ring-addressed) + the B
            # block keys (bidirectional, a small kernel), merged in the reduce; no torch mask
            o = fused_k.attn_window_block(q.transpose(0, 1).reshape(B, w.n_heads * w.head_dim).contiguous(),
                                          self.k[li], self.v[li], p0, k_blk.contiguous(), v_blk.transpose(0, 1).contiguous(),
                                          w.n_heads, w.n_kv_heads, w.head_dim, W, ring=self.ring)
            o = lw.o(o)
            o = fused_k.grouped_conv(o, dyn[:, 1], lw.attn_conv_base[1], w.conv_group)
            h, n = fused_k.add_rmsnorm(h, o, lw.ln2, w.eps, plain=True)
            dyn = lw.mlp_conv_proj(n).view(B, 2, w.conv_ks, -1)
            n = fused_k.grouped_conv(n, dyn[:, 0], lw.mlp_conv_base[0], w.conv_group)
            g, u = lw.gate_up(n).chunk(2, -1)
            m = lw.down(F.silu(g) * u)
            pending = fused_k.grouped_conv(m, dyn[:, 1], lw.mlp_conv_base[1], w.conv_group)
        _, hn = fused_k.add_rmsnorm(h, pending, w.norm, w.eps, plain=True)
        self.last_block_hidden = hn
        return self.select(hn[1:], block[0], draft_head)
