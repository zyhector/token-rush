#!/usr/bin/env python3
"""Cost of the graphed verify step as a function of draft length K, on the real
model. The 'verify step costs 1.1x a raw step' assumption in docs/feasibility.md,
measured. Replays each K's graph with fixed (mostly wrong) drafts; time per replay
does not depend on how many are accepted.

    python bench/verify_cost.py --model <packed> [--max-k 4]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenrush.model import Engine  # noqa: E402
from tokenrush.weights import load_packed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-k", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--kv", default="bf16", choices=("bf16", "fp8"))
    a = ap.parse_args()
    cfg, w, _ = load_packed(a.model)
    eng = Engine(cfg, w, max_len=8192, max_spec=a.max_k,
                 kv_dtype=torch.float8_e4m3fn if a.kv == "fp8" else torch.bfloat16)
    t0 = time.perf_counter()
    eng.capture_verify(list(range(a.max_k + 1)))
    print(f"captured verify graphs K=0..{a.max_k} in {time.perf_counter() - t0:.1f}s; "
          f"state {eng.state.nbytes / 1e9:.2f} GB, cuda {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    prompt = torch.tensor([760, 6511, 314, 9338, 369], device="cuda")
    raw = None
    for K in range(a.max_k + 1):
        eng.reset()
        eng.prefill(prompt)
        eng.tok.copy_(eng.logits_last_argmax() if hasattr(eng, "logits_last_argmax") else torch.tensor([279], device="cuda"))
        drafts = torch.full((K,), 279, device="cuda", dtype=torch.long)     # ' the': rarely all accepted
        for _ in range(5):
            eng.verify(drafts)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.steps):
            eng.verify(drafts)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / a.steps
        raw = raw or dt
        print(f"K={K}: {dt * 1e3:6.2f} ms per verify step ({K + 1} tokens) = {dt / raw:.2f}x the K=0 step; "
              f"{(K + 1) / dt:.0f} tok/s if every draft were accepted")


if __name__ == "__main__":
    main()
