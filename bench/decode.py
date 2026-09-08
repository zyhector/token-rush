#!/usr/bin/env python3
"""Time the engine's decode step and say where the time goes.

    python bench/decode.py --model <packed> [--backend tinygemm|triton|dequant] [--steps 20] [--profile]

Prefills a short prompt, runs warmup steps, then times `steps` decode steps
with the GPU synchronized. --profile prints the top CUDA kernels of one step.
Development numbers: see docs/progress.md.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenrush.model import Engine
from tokenrush.weights import load_packed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="triton")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()

    cfg, w, _ = load_packed(a.model, backend=a.backend)
    eng = Engine(cfg, w, max_len=a.max_len)
    print(f"weights {w.nbytes / 1e9:.2f} GB, cuda allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    prompt = torch.tensor([760, 6511, 314, 9338, 369], device="cuda")   # "The capital of France is"
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = eng.prefill(prompt)
    torch.cuda.synchronize()
    print(f"prefill {len(prompt)} tokens: {(time.perf_counter() - t0) * 1e3:.0f} ms")
    tok = logits[-1].argmax()
    for _ in range(a.warmup):
        tok = eng.decode(tok)[-1].argmax()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.steps):
        tok = eng.decode(tok)[-1].argmax()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / a.steps
    print(f"decode: {dt * 1e3:.2f} ms/step = {1 / dt:.1f} tok/s   ({w.nbytes / dt / 1e9:.0f} GB/s over weight bytes, "
          f"{w.nbytes / dt / 1e9 / 1701 * 100:.1f}% of the wall)")
    print(f"peak cuda {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            tok = eng.decode(tok)[-1].argmax()
            torch.cuda.synchronize()
        rows = [(e.key, e.count, e.self_device_time_total) for e in prof.key_averages()
                if e.self_device_time_total > 0]
        total = sum(r[2] for r in rows)
        print(f"\none decode step, {total / 1e3:.2f} ms of GPU time, top kernels:")
        for k, n, t in sorted(rows, key=lambda r: -r[2])[:15]:
            print(f"  {t / 1e3:7.3f} ms  {n:4d}x  {k[:90]}")
        cpu = sum(e.self_cpu_time_total for e in prof.key_averages())
        print(f"CPU-side time in the step: {cpu / 1e3:.2f} ms")
    if a.backend == "triton":
        from tokenrush.quant import _int4_gemv_kernel
        print("\nautotuner picks (N, K) -> config:")
        for k, c in _int4_gemv_kernel.cache.items():
            print(f"  {k}: {c}")


if __name__ == "__main__":
    main()
