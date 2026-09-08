#!/usr/bin/env python3
"""Speculative greedy decoding end to end (eager MTP drafts, graphed verify):
exactness against raw greedy, effective tok/s per prompt family, and a profile
of the K-draft verify step.

    python bench/spec_run.py --model <packed> [--k 3] [--new 200]
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenrush.generate import generate  # noqa: E402
from tokenrush.model import Engine  # noqa: E402
from tokenrush.mtp import MTPHead, build_mtp  # noqa: E402
from tokenrush.spec import generate_spec, generate_spec_graph  # noqa: E402
from tokenrush.weights import load_packed  # noqa: E402
from mtp_accept import PROMPTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--new", type=int, default=200)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--consistent", action="store_true", help="raw on the M-row kernel too: bit-exact vs spec")
    ap.add_argument("--graph", action="store_true", help="draft chain inside the graph")
    ap.add_argument("--mtp-int4", action="store_true", help="quantize the MTP head's projections to int4")
    ap.add_argument("--dynamic", default=None, help="Kmin:Kmax adaptive depth, e.g. 2:4 (implies --graph)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    cfg, w, mtp_t = load_packed(a.model, with_mtp=True)
    kmax = int(a.dynamic.split(":")[1]) if a.dynamic else a.k
    eng = Engine(cfg, w, max_len=4096, max_spec=kmax, consistent=a.consistent)
    eng.capture()
    t0 = time.perf_counter()
    eng.capture_verify([0, a.k])
    print(f"verify graphs captured in {time.perf_counter() - t0:.0f}s")
    mtp = MTPHead(cfg, build_mtp(cfg, mtp_t, "cuda", int4=a.mtp_int4), w.embed, w.lm_head, 4096)
    dyn = tuple(int(v) for v in a.dynamic.split(":")) if a.dynamic else None
    if dyn:
        a.graph = True
        a.k = dyn[1]
    if a.graph:
        eng.attach_mtp(mtp)
        t0 = time.perf_counter()
        for k in (range(dyn[0], dyn[1] + 1) if dyn else [a.k]):
            eng.capture_spec(k)
        print(f"spec graphs (draft chain + verify) captured in {time.perf_counter() - t0:.0f}s")
    stop = set(cfg.eos_ids) | {tok.eos_token_id}
    for name, text in PROMPTS.items():
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False))
        raw, st_raw = generate(eng, tok, ids, a.new, stop, stream=False, temperature=a.temperature, top_p=a.top_p, seed=0)
        if a.graph:
            spec, st_spec = generate_spec_graph(eng, mtp, tok, ids, a.new, stop, K=a.k, stream=False, dynamic=dyn,
                                                temperature=a.temperature, top_p=a.top_p, seed=0)
        else:
            spec, st_spec = generate_spec(eng, mtp, tok, ids, a.new, stop, K=a.k, stream=False)
        n = min(len(raw), len(spec))
        same = next((i for i in range(n) if raw[i] != spec[i]), n)
        print(f"\n{name}: raw {st_raw['decode_tok_s']:.1f} tok/s | spec K={a.k}: {st_spec['decode_tok_s']:.1f} tok/s, "
              f"{st_spec['accepted_per_step']:.2f} tokens/step, {st_spec['ms_per_step']:.1f} ms/step ({'in-graph' if a.graph else 'eager'} drafts) | "
              f"identical for {same}/{n} tokens" + ("" if same == n else "  <-- MISMATCH") + (f" depths {st_spec['depths']}" if st_spec.get('depths') else ""))
        print("   " + repr(tok.decode(spec[:30])))
    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        eng.reset(); eng.prefill(torch.tensor(ids[:16], device="cuda"))
        drafts = torch.full((a.k,), 279, device="cuda", dtype=torch.long)
        if a.graph:
            mtp.reset(); eng.n_accepted.zero_()
            step = lambda: eng.spec_step(a.k)
        else:
            step = lambda: eng.verify(drafts)
        for _ in range(3):
            step()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step(); torch.cuda.synchronize()
        rows = [(e.key, e.count, e.self_device_time_total) for e in prof.key_averages() if e.self_device_time_total > 0]
        print(f"\nK={a.k} {'spec (draft+verify)' if a.graph else 'verify'} step, {sum(r[2] for r in rows) / 1e3:.2f} ms GPU, top kernels:")
        for k, c, t in sorted(rows, key=lambda r: -r[2])[:10]:
            print(f"  {t / 1e3:7.3f} ms {c:4d}x {k[:80]}")
        from tokenrush.quant import _int4_gemm_rows_kernel
        print("rows-kernel autotune picks:")
        for k, c in _int4_gemm_rows_kernel.cache.items():
            print(f"  {k[:3]}: {c}")


if __name__ == "__main__":
    main()
