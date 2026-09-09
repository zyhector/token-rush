"""GPTQ for the engine's int4 g128 asymmetric packing (Phase 1b, part 2).

    python -m tokenrush.gptq --src /workspace/models/Qwen3.8-27B --dst /workspace/models/Qwen3.8-27B-int4g128-gptq \
        --calib data/quality/calib.npz --mse

Same grid as quant.py (per group of 128 along the input dimension: a bf16
scale, a bf16 minimum, 4-bit codes; dequant = code * scale + mn in bf16),
same packed checkpoint (weights.pack_checkpoint), so every kernel, graph and
test carries over unchanged. What changes is which code each weight gets.

GPTQ (Frantar et al. 2022), the plain version: for each linear, the Hessian
H = X^T X of its input over the calibration set; columns are quantized one
at a time in blocks of 128 and the rounding error of each column is spread
over the not-yet-quantized columns through H^-1 (the OBS update), so what is
minimized is the output error ||W X - Q X||, not the weight error. Layers are
quantized in order and each layer's calibration input is computed through the
already-quantized layers before it (sequential), so later layers absorb part
of the earlier ones' error. Within a layer, the Hessians of all its linears
are collected in one pass with the bf16 weights. Group parameters (scale, mn)
are min/max of the group's current, error-updated weights, rounded to bf16
before the codes are chosen, exactly as quantize_int4 does. No act-order
(our packing has contiguous groups), no MSE scale search: the first version.
"""
import argparse
import json
import math
import os
import time

import torch

from . import ops
from .config import ModelConfig
from .corpus import load_ids
from .model import GDNWeights, attn_forward, gdn_forward, mlp_forward
from .quant import GROUP, Linear
from .state import State
from .weights import HFTensors, build_layer, pack_checkpoint


class Recorder:
    """A Linear that accumulates the Hessian of its input as it is called."""

    def __init__(self, lin: Linear):
        self.lin = lin
        k = lin.weight.shape[1]
        self.H = torch.zeros(k, k, device=lin.weight.device, dtype=torch.float32)
        self.n = 0

    def __call__(self, x):
        xf = x.reshape(-1, x.shape[-1]).float()
        self.H.addmm_(xf.t(), xf)
        self.n += xf.shape[0]
        return self.lin(x)

    def partials(self, x):
        return self(x).float()[None]

    @property
    def shape(self):
        return self.lin.shape

    @property
    def nbytes(self):
        return self.lin.nbytes


def _dequant_col(code, scale, mn):
    """The exact bf16 arithmetic of quant.dequantize_int4, for one column."""
    return (code.to(torch.bfloat16) * scale).to(torch.bfloat16) + mn


def _group_params(W1: torch.Tensor, mse: bool = False):
    """(scale, mn) bf16 for rows x one group of the current weights: the group's min/max
    as quantize_int4 chooses them, or (mse) the range shrunk per row to the factor in
    1.0 .. 0.6 that minimizes sum |w - deq|^2.4 (GPTQ's mse=True search)."""
    mn = W1.amin(1)
    mx = W1.amax(1)
    if not mse:
        scale = (mx - mn) / 15.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale).to(torch.bfloat16)
        return scale, mn.to(torch.bfloat16)
    best = torch.full((W1.shape[0],), float("inf"), device=W1.device)
    best_scale = torch.empty(W1.shape[0], device=W1.device, dtype=torch.bfloat16)
    best_mn = torch.empty_like(best_scale)
    for p in torch.linspace(1.0, 0.6, 33).tolist():
        mn1 = (p * mn).to(torch.bfloat16)
        scale1 = ((p * mx - p * mn) / 15.0)
        scale1 = torch.where(scale1 == 0, torch.ones_like(scale1), scale1).to(torch.bfloat16)
        q = ((W1 - mn1.float()[:, None]) / scale1.float()[:, None]).round().clamp_(0, 15)
        deq = ((q.to(torch.bfloat16) * scale1[:, None]).to(torch.bfloat16) + mn1[:, None]).float()
        err = (deq - W1).abs().pow(2.4).sum(1)
        better = err < best
        best = torch.where(better, err, best)
        best_scale = torch.where(better, scale1, best_scale)
        best_mn = torch.where(better, mn1, best_mn)
    return best_scale, best_mn


@torch.no_grad()
def gptq_quantize(W: torch.Tensor, H: torch.Tensor, group: int = GROUP, damp: float = 0.01, act_order: bool = False,
                  mse: bool = False):
    """W bf16 [out, in], H fp32 [in, in] -> (qweight uint8 [out, in/2], scale bf16 [out, in/group], mn bf16 [...]),
    the packing of quant.quantize_int4, codes chosen by GPTQ.

    act_order: quantize the columns in order of decreasing H diagonal (the most
    influential inputs first, when the most error budget remains) with *static
    groups*: the group parameters are fixed up front from the original weights of
    each contiguous group, so the packing keeps its contiguous groups and no
    permutation is stored."""
    if act_order:
        return _gptq_act_order(W, H, group, damp, mse=mse)
    out, inp = W.shape
    assert inp % group == 0
    W = W.float().clone()
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.diag(H).mean() * torch.eye(inp, device=H.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)          # upper-triangular factor of H^-1
    codes = torch.empty(out, inp, device=W.device, dtype=torch.uint8)
    scales = torch.empty(out, inp // group, device=W.device, dtype=torch.bfloat16)
    mns = torch.empty_like(scales)
    block = group                                            # one group per block, aligned
    for i1 in range(0, inp, block):
        i2 = i1 + block
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        # group parameters from the current (error-updated) weights, in bf16
        scale, mn = _group_params(W1, mse)
        g = i1 // group
        scales[:, g] = scale
        mns[:, g] = mn
        sf, mf = scale.float(), mn.float()
        for i in range(block):
            w = W1[:, i]
            q = ((w - mf) / sf).round().clamp_(0, 15).to(torch.uint8)
            codes[:, i1 + i] = q
            deq = _dequant_col(q, scale, mn).float()
            err = (w - deq) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    q = codes.view(out, inp // 2, 2)
    packed = (q[..., 0] | (q[..., 1] << 4)).contiguous()
    return packed, scales.contiguous(), mns.contiguous()


@torch.no_grad()
def _gptq_act_order(W, H, group, damp, block=128, mse=False):
    out, inp = W.shape
    assert inp % group == 0
    W = W.float().clone()
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    # static groups: parameters of every contiguous group from the original weights
    n_groups = inp // group
    scales = torch.empty(out, n_groups, device=W.device, dtype=torch.bfloat16)
    mns = torch.empty_like(scales)
    for g in range(n_groups):
        scales[:, g], mns[:, g] = _group_params(W[:, g * group:(g + 1) * group], mse)
    perm = torch.argsort(torch.diag(H), descending=True)
    W = W[:, perm]
    H = H[perm][:, perm]
    H += damp * torch.diag(H).mean() * torch.eye(inp, device=H.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    codes_p = torch.empty(out, inp, device=W.device, dtype=torch.uint8)
    gidx = perm // group                                     # group of each permuted column
    for i1 in range(0, inp, block):
        i2 = min(i1 + block, inp)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            g = gidx[i1 + i]
            scale, mn = scales[:, g], mns[:, g]
            w = W1[:, i]
            q = ((w - mn.float()) / scale.float()).round().clamp_(0, 15).to(torch.uint8)
            codes_p[:, i1 + i] = q
            deq = _dequant_col(q, scale, mn).float()
            err = (w - deq) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    codes = torch.empty_like(codes_p)
    codes[:, perm] = codes_p
    q = codes.view(out, inp // 2, 2)
    packed = (q[..., 0] | (q[..., 1] << 4)).contiguous()
    return packed, scales.contiguous(), mns.contiguous()


def layer_forward(x, lw, li, cfg, state, cos, sin):
    """One layer of the eager (non-fused) path on a whole sequence, state reset first."""
    state.reset()
    n = ops.rmsnorm(x, lw.ln1, cfg.eps)
    if cfg.layer_types[li] == "linear_attention":
        h = gdn_forward(n, lw.mixer, cfg, state, state.gdn_slot[li], False)
    else:
        h = attn_forward(n, lw.mixer, cfg, state, state.attn_slot[li], cos, sin, None, False)
    x = x + h
    n = ops.rmsnorm(x, lw.ln2, cfg.eps)
    return x + mlp_forward(n, lw, False)


def _linears(lw, li, cfg):
    """(attribute owner, attribute, checkpoint names of the row blocks) of the quantized linears of a layer."""
    p = f"layers.{li}."
    if isinstance(lw.mixer, GDNWeights):
        m = p + "linear_attn."
        mix = [(lw.mixer, "in_qkvz", [m + "in_proj_qkv.weight", m + "in_proj_z.weight"]),
               (lw.mixer, "out", [m + "out_proj.weight"])]
    else:
        m = p + "self_attn."
        mix = [(lw.mixer, "qkv", [m + "q_proj.weight", m + "k_proj.weight", m + "v_proj.weight"]),
               (lw.mixer, "o", [m + "o_proj.weight"])]
    return mix + [(lw, "gate_up", [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"]),
                  (lw, "down", [p + "mlp.down_proj.weight"])]


@torch.no_grad()
def run(src, dst, calib, group=GROUP, damp=0.01, device="cuda", limit=None, act_order=False, mse=False):
    cfg = ModelConfig.load(src)
    tensors = HFTensors(src)
    ids = calib["ids"]
    if limit:
        ids = ids[:limit]
    N, T = ids.shape
    print(f"GPTQ: {N} sequences x {T} tokens, group {group}, damp {damp}, act_order {act_order}, mse {mse}")
    state = State(cfg, T, device)
    cos, sin = ops.rope_table(T, cfg.rotary_dim, cfg.rope_theta, device)
    embed = tensors["embed_tokens.weight"].to(device)
    xs = [embed[ids[s].to(device)] for s in range(N)]            # residual stream per sequence, [T, hidden] bf16
    del embed
    results = {}                                                   # checkpoint name -> (q, s, m) on CPU
    t_all = time.time()
    for li in range(cfg.n_layers):
        t0 = time.time()
        lw = build_layer(cfg, tensors, li, device, backend=None)
        recs = []
        for owner, attr, names in _linears(lw, li, cfg):
            r = Recorder(getattr(owner, attr))
            setattr(owner, attr, r)
            recs.append((owner, attr, names, r))
        for s in range(N):
            layer_forward(xs[s], lw, li, cfg, state, cos, sin)
        t_h = time.time() - t0
        err = []
        for owner, attr, names, r in recs:
            W = r.lin.weight
            H = r.H / r.n
            q, sc, mn = gptq_quantize(W, H, group, damp, act_order, mse)
            # split the row blocks back into the checkpoint's tensors
            rows = [tensors[n].shape[0] for n in names]
            o = 0
            for n, nr in zip(names, rows):
                results[n] = (q[o:o + nr].cpu(), sc[o:o + nr].cpu(), mn[o:o + nr].cpu())
                o += nr
            from .quant import dequantize_int4
            deq = dequantize_int4(q, sc, mn)
            # output-space relative error on the calibration Hessian, the quantity GPTQ minimizes
            D = (W.float() - deq.float())
            num = (D @ H * D).sum()
            den = (W.float() @ H * W.float()).sum()
            err.append(f"{attr} {math.sqrt(num / den):.4f}")
            setattr(owner, attr, Linear(deq))
            del r.H
        for s in range(N):
            xs[s] = layer_forward(xs[s], lw, li, cfg, state, cos, sin)
        del lw
        print(f"  layer {li:2d} ({cfg.layer_types[li][:6]}): H {t_h:.0f}s, total {time.time() - t0:.0f}s; "
              f"output rel err {' '.join(err)}", flush=True)
    # lm_head on the final-normed hidden states
    t0 = time.time()
    final_norm = tensors["norm.weight"].to(device)
    head = Linear(tensors["lm_head.weight"].to(device))
    r = Recorder(head)
    for s in range(N):
        r(ops.rmsnorm(xs[s], final_norm, cfg.eps))
    q, sc, mn = gptq_quantize(head.weight, r.H / r.n, group, damp, act_order, mse)
    results["lm_head.weight"] = (q.cpu(), sc.cpu(), mn.cpu())
    print(f"  lm_head: {time.time() - t0:.0f}s; all layers {time.time() - t_all:.0f}s")
    del xs, head, r
    torch.cuda.empty_cache()
    meta = {"method": "gptq", "damp": damp, "act_order": act_order, "mse": mse, "calibration": {"sequences": N, "tokens": T, "mix": calib.get("mix"),
                                                             "seed": calib.get("seed")}}
    # the 17 minutes of work, kept until the checkpoint is written (a full disk once lost it)
    os.makedirs(dst, exist_ok=True)
    tmp = os.path.join(dst, "gptq_codes.pt")
    torch.save({"results": results, "meta": meta}, tmp)
    pack_checkpoint(src, dst, group, device=device, quantizer=lambda name, t: results[name], meta_extra=meta)
    os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF bf16 checkpoint")
    ap.add_argument("--dst", required=True, help="packed checkpoint to write")
    ap.add_argument("--calib", required=True, help="bench/quality_calib.py output")
    ap.add_argument("--group", type=int, default=GROUP)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=None, help="calibration sequences to use (debug)")
    ap.add_argument("--act-order", action="store_true", help="quantize columns by decreasing Hessian diagonal, static groups")
    ap.add_argument("--mse", action="store_true", help="per-row range-shrink search for the group parameters")
    ap.add_argument("--from-codes", default=None, help="pack a saved gptq_codes.pt instead of recomputing")
    a = ap.parse_args()
    if a.from_codes:
        saved = torch.load(a.from_codes)
        results = saved["results"]
        pack_checkpoint(a.src, a.dst, a.group, quantizer=lambda name, t: results[name], meta_extra=saved["meta"])
        return
    tensors, meta = load_ids(a.calib)
    run(a.src, a.dst, {**tensors, **meta}, a.group, a.damp, limit=a.limit, act_order=a.act_order, mse=a.mse)


if __name__ == "__main__":
    main()
