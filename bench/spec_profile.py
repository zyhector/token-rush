#!/usr/bin/env python3
"""Kernel-level profile of one speculative step (draft chain + verify) and one raw
step at a given context length, to attribute the spec step's growth with context.

    python bench/spec_profile.py --model <packed> --context 200000 [--k 3]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenrush.model import Engine  # noqa: E402
from tokenrush.mtp import MTPHead, build_mtp  # noqa: E402
from tokenrush.spec import prime_spec  # noqa: E402
from tokenrush.weights import load_packed  # noqa: E402


def prof(fn, label):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    rows = [(e.key, e.count, e.self_device_time_total) for e in p.key_averages() if e.self_device_time_total > 0]
    total = sum(r[2] for r in rows)
    print(f"\n{label}: {total / 1e3:.2f} ms GPU")
    for k, c, t in sorted(rows, key=lambda r: -r[2])[:12]:
        print(f"  {t / 1e3:7.3f} ms {c:4d}x {k[:70]}")
    return {k: t for k, _, t in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", type=int, default=200000)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    cfg, w, mtp_t = load_packed(a.model, with_mtp=True)
    eng = Engine(cfg, w, max_len=262144, max_spec=a.k, kv_dtype=torch.float8_e4m3fn)
    eng.capture()
    mtp = MTPHead(cfg, build_mtp(cfg, mtp_t, "cuda", int4=True), w.embed, w.lm_head, 262144, kv_dtype=torch.float8_e4m3fn)
    eng.attach_mtp(mtp, draft_vocab=torch.arange(131072))
    eng.capture_spec(a.k)
    ids = torch.randint(1000, 200000, (a.context,)).tolist()
    for ctx in (64, a.context):
        t0 = time.time()
        prime_spec(eng, mtp, ids[:ctx])
        print(f"\n===== context {ctx} (primed in {time.time() - t0:.0f}s)")
        raw = prof(lambda: eng.step(), "raw step")
        eng.reset(); mtp.reset(); prime_spec(eng, mtp, ids[:ctx])
        spec = prof(lambda: eng.spec_step(a.k), f"spec step K={a.k}")
        eng.reset(); mtp.reset()


if __name__ == "__main__":
    main()
