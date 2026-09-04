#!/usr/bin/env python3
"""Speed-of-light for the dominant decode term: bs=1 bf16 GEMVs at the real
per-layer shapes of Qwen3.8-27B, eager and inside one CUDA graph.

Also reports which kernels cuBLAS dispatches for these shapes on this card.
On CUDA 12.8 it fell back to Ampere-lineage "cutlass_80_tensorop_*" GEMMs; on
cu130 it dispatches a native gemvx kernel. Check, do not assume.

One 4-layer block is 3 GDN layers + 1 attention layer; this runs three blocks
(9 GDN + 3 attention, 12 of the model's 64 layers) so the weights fit in VRAM
alongside nothing else. Weight bytes are counted exactly; the result is GB/s
against the measured read bandwidth from check_bandwidth.py.
"""
import sys
import time

import torch

HIDDEN, FFN = 5120, 17408
GDN_QK, GDN_V, GDN_D = 16, 48, 128
ATT_Q, ATT_KV, ATT_D = 24, 4, 256
WALL_GBS = float(sys.argv[1]) if len(sys.argv) > 1 else 1624.0

# (in_features, out_features) for every matmul a layer runs at decode
GDN_LAYER = [
    (HIDDEN, 2 * GDN_QK * GDN_D + 2 * GDN_V * GDN_D),   # in_proj: q, k, v, z
    (HIDDEN, 2 * GDN_V),                                 # in_proj: b, a
    (GDN_V * GDN_D, HIDDEN),                             # out_proj
    (HIDDEN, 2 * FFN),                                   # gate + up
    (FFN, HIDDEN),                                       # down
]
ATT_LAYER = [
    (HIDDEN, 2 * ATT_Q * ATT_D),                         # q_proj incl. gate
    (HIDDEN, ATT_KV * ATT_D),                            # k_proj
    (HIDDEN, ATT_KV * ATT_D),                            # v_proj
    (ATT_Q * ATT_D, HIDDEN),                             # o_proj
    (HIDDEN, 2 * FFN),
    (FFN, HIDDEN),
]
BLOCK = [GDN_LAYER] * 3 + [ATT_LAYER]
N_BLOCKS = 3


def best_time(fn, reps=5, iters=10, warmup=10):
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


def main():
    torch.manual_seed(0)
    shapes = [s for _ in range(N_BLOCKS) for layer in BLOCK for s in layer]
    weights = [torch.empty(o, i, device="cuda", dtype=torch.bfloat16).normal_(std=0.01)
               for i, o in shapes]
    inputs = {i: torch.randn(1, i, device="cuda", dtype=torch.bfloat16) for i, _ in shapes}
    nbytes = sum(w.numel() * 2 for w in weights)
    print(f"{len(weights)} GEMVs over {N_BLOCKS * 4} layers, {nbytes / 1e9:.2f} GB bf16 weights")

    def step():
        for (i, _), w in zip(shapes, weights):
            torch.matmul(inputs[i], w.T)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()

    eager = best_time(step)
    graphed = best_time(g.replay)
    for name, t in (("eager", eager), ("graphed", graphed)):
        gbs = nbytes / t / 1e9
        print(f"  {name:8s} {t * 1e3:6.2f} ms   {gbs:6.0f} GB/s   {gbs / WALL_GBS * 100:5.1f}% of {WALL_GBS:.0f} GB/s wall")

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    names = {}
    for e in prof.key_averages():
        if e.device_type.name == "CUDA" and e.self_device_time_total > 0:
            names[e.key] = names.get(e.key, 0) + e.count
    print("\ncuBLAS kernels dispatched for these shapes:")
    for k, n in sorted(names.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d}x  {k[:110]}")


if __name__ == "__main__":
    main()
