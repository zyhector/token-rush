#!/usr/bin/env python3
"""Speculative and raw decode vs. context length on real long text (fp8 KV).

    python bench/spec_context.py --model <packed> --text /workspace/data/prose.txt --contexts 0,22000,90000,200000

The prompt is the first N tokens of the text; the continuation is generated
greedily (200 tokens) once raw and once speculatively (dynamic 3:4, int4 MTP).
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenrush.generate import generate  # noqa: E402
from tokenrush.model import Engine  # noqa: E402
from tokenrush.mtp import MTPHead, build_mtp  # noqa: E402
from tokenrush.spec import generate_spec_graph  # noqa: E402
from tokenrush.weights import load_packed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--contexts", default="0,22000,90000,200000")
    ap.add_argument("--new", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=262144)
    ap.add_argument("--mtp-kv", default="fp8", choices=("bf16", "fp8"), help="the MTP head's own cache dtype")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    cfg, w, mtp_t = load_packed(a.model, with_mtp=True)
    eng = Engine(cfg, w, max_len=a.max_len, max_spec=4, kv_dtype=torch.float8_e4m3fn)
    eng.capture()
    mtp = MTPHead(cfg, build_mtp(cfg, mtp_t, "cuda", int4=True), w.embed, w.lm_head, a.max_len,
                  kv_dtype=torch.float8_e4m3fn if a.mtp_kv == "fp8" else torch.bfloat16)
    eng.attach_mtp(mtp)
    for k in (3, 4):
        eng.capture_spec(k)
    print(f"cuda {torch.cuda.memory_allocated() / 1e9:.1f} GB after capture")
    text = open(a.text).read()
    all_ids = tok.encode(text[:6_000_000])
    print(f"{len(all_ids)} tokens available")
    stop = set()   # ignore eos: measure a fixed number of tokens
    for ctx in (int(c) for c in a.contexts.split(",")):
        ctx = max(ctx, 64)
        ids = all_ids[:ctx]
        t0 = time.time()
        raw, st_raw = generate(eng, tok, ids, a.new, stop, stream=False)
        spec, st_spec = generate_spec_graph(eng, mtp, tok, ids, a.new, stop, stream=False, dynamic=(3, 4))
        kv = eng.state.kv_bytes_per_token * ctx / 1e9
        print(f"context {ctx:>7}: raw {st_raw['decode_tok_s']:6.1f} tok/s | spec {st_spec['decode_tok_s']:6.1f} tok/s, "
              f"{st_spec['accepted_per_step']:.2f} tokens/step, {st_spec['ms_per_step']:.1f} ms/step | "
              f"KV read {kv:.1f} GB/step | prefill {st_raw['prefill_s']:.0f}s | {time.time() - t0:.0f}s  "
              f"| {tok.decode(spec[:12])!r}")


if __name__ == "__main__":
    main()
