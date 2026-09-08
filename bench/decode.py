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
    ap.add_argument("--eager", action="store_true", help="eager decode instead of graph replay")
    ap.add_argument("--context", type=int, default=0, help="prefill this many random tokens first")
    ap.add_argument("--kv", default="bf16", choices=("bf16", "fp8"), help="KV cache dtype")
    a = ap.parse_args()

    cfg, w, _ = load_packed(a.model, backend=a.backend)
    eng = Engine(cfg, w, max_len=a.max_len, kv_dtype=torch.float8_e4m3fn if a.kv == "fp8" else torch.bfloat16)
    if not a.eager:
        t0 = time.perf_counter()
        eng.capture()
        print(f"captured graphs for buckets {sorted(eng.graphs)} in {time.perf_counter() - t0:.1f}s")
    print(f"weights {w.nbytes / 1e9:.2f} GB, cuda allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    prompt = torch.tensor([760, 6511, 314, 9338, 369], device="cuda")   # "The capital of France is"
    if a.context:
        prompt = torch.cat([torch.randint(1000, 200000, (a.context,), device="cuda"), prompt])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = eng.prefill(prompt)
    torch.cuda.synchronize()
    print(f"prefill {len(prompt)} tokens: {(time.perf_counter() - t0) * 1e3:.0f} ms")
    tok = logits[-1].argmax()
    if a.eager:
        step = lambda t: eng.decode(t)[-1].argmax()
    else:
        eng.tok.copy_(tok.view(1))
        step = lambda t: eng.step()[0]
    for _ in range(a.warmup):
        tok = step(tok)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.steps):
        tok = step(tok)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / a.steps
    read = w.nbytes - w.embed.numel() * w.embed.element_size()   # the embedding table is not streamed
    read += eng.state.pos * eng.state.kv_bytes_per_token   # live KV
    print(f"decode at context {eng.state.pos}: {dt * 1e3:.2f} ms/step = {1 / dt:.1f} tok/s   ({read / 1e9:.2f} GB read per step -> "
          f"{read / dt / 1e9:.0f} GB/s, {read / dt / 1e9 / 1701 * 100:.1f}% of the wall; ceiling {1701e9 / read:.1f} tok/s)")
    print(f"peak cuda {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            tok = step(tok)
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
