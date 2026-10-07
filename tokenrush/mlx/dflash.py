"""DFlash2 (z-lab's block-diffusion draft) on the MLX backend.

A port of the eager reference path in tokenrush/dflash.py (DFlashDraft.forward and
select), which follows z-lab/dflash/model.py (MIT). One draft forward turns the
block [t, mask x 7] into 7 drafts, conditioned on the target's residual stream
after layers 5/19/33/47/61; the block rows attend bidirectionally among
themselves and to context rows within a 2048-position window. Norms are plain
Qwen3 RMSNorm (weight * x).

Two simplifications against the CUDA engine, both safe on unified memory: the
context K/V is stored by position over the whole max_len (5 layers x 20 KB per
token) instead of a ring, and the step is an eager MLX graph evaluated once with
the verify step, which plays the role of the CUDA graph.
"""
import glob
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from .model import Engine
from .spec import Sampling
from .weights import Linear, QLinear, _leaves


def rmsnorm_plain(x, w, eps):
    return w * mx.fast.rms_norm(x.astype(mx.float32), None, eps).astype(x.dtype)


def grouped_dynamic_conv(h, dyn, base, group: int):
    """h [L, H]; dyn [L, ks, groups]; base [ks, H]. Causal depthwise conv over the block
    rows with a per-position, per-group dynamic kernel added to a static one."""
    L, H = h.shape
    groups = H // group
    blocks = h.reshape(L, groups, group)
    out = mx.zeros_like(blocks)
    for off in range(base.shape[0]):
        vals = blocks if off == 0 else mx.concatenate([mx.zeros((off, groups, group), h.dtype), blocks[:-off]])
        out = out + base[off].reshape(1, groups, group).astype(h.dtype) * vals
        out = out + dyn[:, off].reshape(L, groups, 1) * vals
    return out.reshape(L, H)


@dataclass
class DFlashLayer:
    ln1: mx.array
    ln2: mx.array
    q: object
    k: object
    v: object
    o: object
    q_norm_w: mx.array
    k_norm_w: mx.array
    gate_up: object
    down: object
    attn_conv_base: mx.array
    attn_conv_proj: object
    mlp_conv_base: mx.array
    mlp_conv_proj: object


class DFlashDraft:
    def __init__(self, path: str, engine: Engine, int4: bool = True, draft_vocab: Optional[mx.array] = None):
        cfg = json.load(open(os.path.join(path, "config.json")))
        dc = cfg["dflash_config"]
        t = {}
        for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            t.update(mx.load(shard))

        def lin(names):
            w = mx.concatenate([t[n] for n in names], axis=0)
            return QLinear(*mx.quantize(w, group_size=128, bits=4)) if int4 else Linear(w)

        self.layers = []
        for i in range(cfg["num_hidden_layers"]):
            p = f"layers.{i}."
            self.layers.append(DFlashLayer(
                ln1=t[p + "input_layernorm.weight"], ln2=t[p + "post_attention_layernorm.weight"],
                q=lin([p + "self_attn.q_proj.weight"]), k=lin([p + "self_attn.k_proj.weight"]),
                v=lin([p + "self_attn.v_proj.weight"]), o=lin([p + "self_attn.o_proj.weight"]),
                q_norm_w=t[p + "self_attn.q_norm.weight"], k_norm_w=t[p + "self_attn.k_norm.weight"],
                gate_up=lin([p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"]), down=lin([p + "mlp.down_proj.weight"]),
                attn_conv_base=t[p + "attention_conv.base_kernel"], attn_conv_proj=lin([p + "attention_conv.kernel_projection.weight"]),
                mlp_conv_base=t[p + "mlp_conv.base_kernel"], mlp_conv_proj=lin([p + "mlp_conv.kernel_projection.weight"])))
        self.fc = lin(["fc.weight"])
        self.hidden_norm = t["hidden_norm.weight"]
        self.norm = t["norm.weight"]
        self.sel_proj = lin(["candidate_selector.hidden_projection.weight"])
        self.pred_cb = t["candidate_selector.predecessor_codebook"]
        self.succ_cb = t["candidate_selector.successor_codebook"]
        self.target_layer_ids = list(dc["target_layer_ids"])
        self.block_size = int(dc["block_size"])
        self.mask_token_id = int(dc["mask_token_id"])
        self.n_heads, self.n_kv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.conv_ks, self.conv_group = int(dc["conv_kernel_size"]), int(dc["conv_group_size"])
        self.top_k, self.window, self.eps = int(dc["selector_top_k"]), int(cfg["sliding_window"]), cfg["rms_norm_eps"]
        self.theta = cfg["rope_parameters"]["rope_theta"]
        del t
        self.engine = engine
        self.embed = engine.w.embed
        self.draft_map = draft_vocab
        self.head = engine.w.lm_head if draft_vocab is None else engine.w.lm_head.rows(draft_vocab)
        engine.feature_layers = self.target_layer_ids
        self.reset()
        mx.eval(_leaves(self.layers), _leaves([self.fc, self.sel_proj, self.head]), self.hidden_norm, self.norm,
                self.pred_cb, self.succ_cb)

    def reset(self):
        """Back to no context. The caches are allocated once: attention reads only rows < ctx."""
        if not hasattr(self, "k"):
            L, ml = len(self.layers), self.engine.max_len
            self.k = [mx.zeros((self.n_kv, ml, self.hd), mx.bfloat16) for _ in range(L)]
            self.v = [mx.zeros((self.n_kv, ml, self.hd), mx.bfloat16) for _ in range(L)]
        self.ctx = 0

    def arrays(self):
        return self.k + self.v

    def _rope(self, x, offset):
        return mx.fast.rope(x, self.hd, traditional=False, base=self.theta, scale=1.0, offset=offset)

    def add_context(self, features):
        """Cache K/V for context rows at positions ctx.. from the target's features [n, 5*hidden]."""
        n = features.shape[0]
        th = rmsnorm_plain(self.fc(features), self.hidden_norm, self.eps)
        for li, lw in enumerate(self.layers):
            k = rmsnorm_plain(lw.k(th).reshape(n, self.n_kv, self.hd), lw.k_norm_w, self.eps).transpose(1, 0, 2)
            v = lw.v(th).reshape(n, self.n_kv, self.hd).transpose(1, 0, 2)
            self.k[li][:, self.ctx:self.ctx + n] = self._rope(k, self.ctx)
            self.v[li][:, self.ctx:self.ctx + n] = v
        self.ctx += n

    def forward(self, features, block):
        """features [n_ctx, 5*hidden] (new context rows) or None; block [B] at positions
        ctx + n_ctx .. -> draft tokens [B-1]."""
        if features is not None and features.shape[0]:
            self.add_context(features)
        p0, B = self.ctx, block.shape[0]
        lo = max(0, p0 - (self.window - 1))
        blk_pos = mx.arange(p0, p0 + B)
        vis = (blk_pos[:, None] - mx.arange(lo, p0)[None, :]) < self.window
        mask = mx.concatenate([vis, mx.ones((B, B), mx.bool_)], axis=1)
        h = self.embed[block]
        for li, lw in enumerate(self.layers):
            n = rmsnorm_plain(h, lw.ln1, self.eps)
            dyn = lw.attn_conv_proj(n).reshape(B, 2, self.conv_ks, -1)
            n = grouped_dynamic_conv(n, dyn[:, 0], lw.attn_conv_base[0], self.conv_group)
            q = rmsnorm_plain(lw.q(n).reshape(B, self.n_heads, self.hd), lw.q_norm_w, self.eps).transpose(1, 0, 2)
            kb = rmsnorm_plain(lw.k(n).reshape(B, self.n_kv, self.hd), lw.k_norm_w, self.eps).transpose(1, 0, 2)
            vb = lw.v(n).reshape(B, self.n_kv, self.hd).transpose(1, 0, 2)
            q, kb = self._rope(q, p0), self._rope(kb, p0)
            K = mx.concatenate([self.k[li][:, lo:p0], kb], axis=1)
            V = mx.concatenate([self.v[li][:, lo:p0], vb], axis=1)
            o = mx.fast.scaled_dot_product_attention(q[None], K[None], V[None], scale=self.hd ** -0.5, mask=mask)[0]
            o = lw.o(o.transpose(1, 0, 2).reshape(B, self.n_heads * self.hd))
            h = h + grouped_dynamic_conv(o, dyn[:, 1], lw.attn_conv_base[1], self.conv_group)
            n = rmsnorm_plain(h, lw.ln2, self.eps)
            dyn = lw.mlp_conv_proj(n).reshape(B, 2, self.conv_ks, -1)
            n = grouped_dynamic_conv(n, dyn[:, 0], lw.mlp_conv_base[0], self.conv_group)
            gu = lw.gate_up(n)
            f = gu.shape[-1] // 2
            m = lw.down((gu[:, :f] * mx.sigmoid(gu[:, :f])) * gu[:, f:])
            h = h + grouped_dynamic_conv(m, dyn[:, 1], lw.mlp_conv_base[1], self.conv_group)
        hn = rmsnorm_plain(h, self.norm, self.eps)
        return self.select(hn[1:], block[0:1])

    def select(self, hid, anchor):
        """Top-16 candidates per position by the lm_head, chained by the bilinear codebook
        score with the predecessor token. hid [B-1, H] -> tokens [B-1]."""
        logits = self.head(hid)
        cands = mx.argpartition(-logits.astype(mx.float32), self.top_k - 1, axis=-1)[:, :self.top_k]
        unary = mx.take_along_axis(logits, cands, -1)
        if self.draft_map is not None:
            cands = self.draft_map[cands]
        hp = self.sel_proj(hid)                                              # [B-1, rank]
        pred, out = anchor, []
        for pos in range(hid.shape[0]):
            score = unary[pos] + ((self.pred_cb[pred] * hp[pos:pos + 1]) @ self.succ_cb[cands[pos]].T)[0]
            pred = cands[pos][mx.argmax(score)][None]
            out.append(pred)
        return mx.concatenate(out)


def generate_dflash(engine: Engine, draft: DFlashDraft, ids: list, max_new: int, stop_ids=(),
                    sampling: Sampling = None, chunk: int = 2048, on_tokens=None):
    """Speculative generation with the DFlash2 block draft. -> (tokens, stats)."""
    sampling = sampling or Sampling()
    pick = None if sampling.temperature <= 0 else sampling
    t0 = time.perf_counter()
    engine.reset()
    draft.reset()
    engine.feature_layers = draft.target_layer_ids        # another mode may have cleared it
    ids_a = mx.array(ids)
    for s in range(0, len(ids), chunk):
        H = engine.forward_hidden(ids_a[s:s + chunk])
        draft.add_context(engine.features)
        mx.eval(H, engine.state.arrays(), draft.arrays())
    tok = (sampling if pick else (lambda l: mx.argmax(l, -1)))(engine.w.lm_head(H[-1:]))
    mx.eval(tok)
    t_prefill = time.perf_counter() - t0
    mask = mx.full((draft.block_size - 1,), draft.mask_token_id, mx.int32)
    new_feats, out, steps = None, [], 0
    t1 = time.perf_counter()
    while len(out) < max_new and engine.state.pos + draft.block_size < engine.max_len:
        drafts = draft.forward(new_feats, mx.concatenate([tok.astype(mx.int32), mask]))
        seq = mx.concatenate([tok.astype(mx.int32), drafts.astype(mx.int32)])
        n, pred, _ = engine.verify(seq, pick)
        feats = engine.features
        mx.eval(n, pred, feats, engine.state.arrays(), draft.arrays())
        n = int(n.item())
        engine.commit(n)
        new = seq[:n + 1].tolist()
        out.extend(new)
        steps += 1
        tok = pred[n:n + 1]
        new_feats = feats[:n + 1]
        if on_tokens:
            on_tokens(new)
        if any(t in stop_ids for t in new):
            out = out[:next(i for i, t in enumerate(out) if t in stop_ids) + 1]
            break
    out = out[:max_new]
    dt = time.perf_counter() - t1
    return out, {"prefill_s": t_prefill, "decode_s": dt, "tok_s": len(out) / dt, "steps": steps,
                 "tokens_per_step": len(out) / max(steps, 1), "ms_per_step": 1e3 * dt / max(steps, 1)}
