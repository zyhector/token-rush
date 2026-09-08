#!/usr/bin/env python3
"""4-bit GEMV shootout at the real bs=1 shapes of Qwen3.8-27B.

Every candidate is a (pack, run) pair. The harness builds a 12-layer set
(9 GDN + 3 attention, every projection a separate matrix, as the engine runs
them) plus lm_head, records all GEMVs into one CUDA graph, and reports the
packed bytes actually read per replay divided by the replay time: GB/s
against the 1701 GB/s wall, and the whole-model decode tok/s that bandwidth
implies at the candidate's bits per weight. lm_head is also timed alone,
since it is the largest single matrix and a different shape class.

Correctness is a relative error against bf16 x @ W_dequant.T, where W_dequant
is the candidate's own dequantization of its packed weights, so a candidate
cannot score by reading fewer bytes than it claims.

    python harness.py [--wall 1701] [--only name,name] [--skip name]

Candidates register with @candidate(name, bpw); files named cand_*.py in
this directory are imported if their dependencies are present.
"""
import argparse
import glob
import importlib
import os
import sys
import time

import torch

HIDDEN, FFN = 5120, 17408
VOCAB = 248320
TEXT_PARAMS = 26.896e9      # body + lm_head, what a decode step reads

GDN_LAYER = [(10240, HIDDEN), (6144, HIDDEN), (HIDDEN, 6144), (FFN, HIDDEN), (FFN, HIDDEN), (HIDDEN, FFN)]
ATT_LAYER = [(12288, HIDDEN), (1024, HIDDEN), (1024, HIDDEN), (HIDDEN, 6144), (FFN, HIDDEN), (FFN, HIDDEN), (HIDDEN, FFN)]
LAYER_SET = [s for layer in ([GDN_LAYER] * 3 + [ATT_LAYER]) * 3 for s in layer]   # (out, in) per matrix
LM_HEAD = (VOCAB, HIDDEN)
# The same layers with q|k|v (and in_proj_qkv|z) and gate|up concatenated into one
# matrix each: fewer, larger GEMVs. What the engine packs once this shootout is done.
GDN_FUSED = [(10240 + 6144, HIDDEN), (HIDDEN, 6144), (2 * FFN, HIDDEN), (HIDDEN, FFN)]
ATT_FUSED = [(12288 + 2048, HIDDEN), (HIDDEN, 6144), (2 * FFN, HIDDEN), (HIDDEN, FFN)]
LAYER_SET_FUSED = [s for layer in ([GDN_FUSED] * 3 + [ATT_FUSED]) * 3 for s in layer]

CANDIDATES = {}


def candidate(name, bpw):
    """Decorate a class with pack(w_bf16) -> obj, run(obj, x) -> y, dequant(obj) -> bf16 [out, in],
    nbytes(obj) -> int. bpw is the nominal bits per weight, used for the whole-model projection."""
    def deco(cls):
        cls.name, cls.bpw = name, bpw
        CANDIDATES[name] = cls
        return cls
    return deco


def best_time(fn, reps=7, iters=10, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / iters)
    return best


def graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def bench(cls, shapes, wall):
    torch.manual_seed(0)
    c = cls()
    packed, xs, refs = [], {}, []
    nbytes = 0
    for out, inp in shapes:
        w = (torch.randn(out, inp, device="cuda") * 0.02).to(torch.bfloat16)
        p = c.pack(w)
        packed.append(p)
        nbytes += c.nbytes(p)
        if inp not in xs:
            xs[inp] = torch.randn(1, inp, device="cuda", dtype=torch.bfloat16)
        wd = c.dequant(p)
        refs.append((xs[inp].float() @ wd.float().T))
        del w, wd

    def step():
        for (out, inp), p in zip(shapes, packed):
            c.run(p, xs[inp])

    err = 0.0
    for (out, inp), p, r in zip(shapes, packed, refs):
        y = c.run(p, xs[inp]).float()
        err = max(err, ((y - r).norm() / r.norm()).item())
    g = graph(step)
    t_eager = best_time(step)
    t_graph = best_time(g.replay)
    del packed, refs, g
    torch.cuda.empty_cache()
    return {"bytes": nbytes, "eager_ms": t_eager * 1e3, "graph_ms": t_graph * 1e3,
            "gbs": nbytes / t_graph / 1e9, "pct": nbytes / t_graph / 1e9 / wall * 100, "err": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall", type=float, default=1701.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--per-shape", action="store_true")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.modules["harness"] = sys.modules[__name__]   # candidates register into this module's registry
    for f in sorted(glob.glob(os.path.join(here, "cand_*.py"))):
        mod = os.path.basename(f)[:-3]
        try:
            importlib.import_module(mod)
        except Exception as e:
            print(f"[skip] {mod}: {type(e).__name__}: {str(e)[:120]}")

    names = [n for n in CANDIDATES if (not a.only or n in a.only.split(",")) and n not in a.skip.split(",")]
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} | torch {torch.__version__} | wall {a.wall:.0f} GB/s | candidates: {', '.join(names)}\n")
    hdr = f"{'candidate':<14} {'bpw':>5} | {'12 layers: GB':>13} {'ms':>7} {'GB/s':>6} {'%wall':>6} | {'lm_head: GB':>11} {'ms':>7} {'GB/s':>6} {'%wall':>6} | {'model tok/s':>11} {'rel err':>8}"
    print(hdr)
    print("-" * len(hdr))
    for n in names:
        cls = CANDIDATES[n]
        if a.per_shape:
            print(n)
            per_shape(cls, a.wall)
            continue
        for label, shapes in (("", LAYER_SET), (" fused", LAYER_SET_FUSED)):
            try:
                L = bench(cls, shapes, a.wall)
                H = bench(cls, [LM_HEAD], a.wall)
            except Exception as e:
                print(f"{n + label:<14} FAILED: {type(e).__name__}: {str(e)[:100]}")
                torch.cuda.empty_cache()
                continue
            # whole-model projection: layer bytes scaled 64/12 plus lm_head, at each part's measured GB/s
            t_model = (L["bytes"] * 64 / 12 / L["gbs"] + H["bytes"] / H["gbs"]) / 1e9
            print(f"{n + label:<14} {cls.bpw:>5.2f} | {L['bytes']/1e9:>13.2f} {L['graph_ms']:>7.2f} {L['gbs']:>6.0f} {L['pct']:>6.1f} | "
                  f"{H['bytes']/1e9:>11.2f} {H['graph_ms']:>7.2f} {H['gbs']:>6.0f} {H['pct']:>6.1f} | {1/t_model:>11.1f} {max(L['err'], H['err']):>8.1e}")



def per_shape(cls, wall, shapes=None):
    """GB/s for each distinct shape on its own, in-graph, for kernel tuning."""
    shapes = shapes or sorted(set(LAYER_SET + LAYER_SET_FUSED + [LM_HEAD]), key=lambda s: s[0] * s[1])
    c = cls()
    for out, inp in shapes:
        w = (torch.randn(out, inp, device="cuda") * 0.02).to(torch.bfloat16)
        p = c.pack(w)
        nb = c.nbytes(p)
        n_copies = max(1, -(-(400 << 20) // nb))          # > 4x the 96 MB L2 in flight
        ps = [p] + [c.pack(w) for _ in range(n_copies - 1)]
        x = torch.randn(1, inp, device="cuda", dtype=torch.bfloat16)

        def step():
            for q in ps:
                c.run(q, x)
        g = graph(step)
        t = best_time(g.replay, iters=10) / n_copies
        print(f"  {out:>7} x {inp:<6} {nb / 1e6:8.1f} MB  {t * 1e6:7.1f} us  {nb / t / 1e9:6.0f} GB/s  {nb / t / 1e9 / wall * 100:5.1f}%   (x{n_copies})")
        del w, p, ps, g
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
