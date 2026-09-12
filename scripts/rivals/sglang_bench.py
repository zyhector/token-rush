#!/usr/bin/env python3
"""bs=1 decode measurements against a running SGLang server.

Decode tok/s is measured on a streamed request as tokens between the first
and last chunk over the wall time between them, so prefill, prefix-cache hits
and state-cache restores cancel. Greedy, ignore_eos, one request in flight.
Best of `reps`.

Usage:
  sglang_bench.py raw                   # short-context decode
  sglang_bench.py context 0 22000 90000 200000   # decode vs. tokens already in context
  sglang_bench.py spec                  # decode + mean accepted length, real prompts
  sglang_bench.py files a.txt b.txt ... # one streamed request per prompt file (bench/cut_prompts.py), fixed n

Speculation stats come from the response `meta_info`: `spec_verify_ct` is the
number of verify steps, so completion_tokens / spec_verify_ct is the mean
accepted length per step (draft + bonus token).
"""
import argparse
import json
import random
import sys
import time
import urllib.request

URL = "http://127.0.0.1:30000"
PROMPTS = {
    "essay": "Write a detailed essay on the causes and consequences of the fall of the "
             "Western Roman Empire, covering political, economic and military factors.",
    "code": "Write a Python implementation of a thread-safe LRU cache with TTL expiry, "
            "including unit tests.",
    "math": "Solve step by step: A train leaves city A at 60 km/h and another leaves city B, "
            "450 km away, at 90 km/h toward it one hour later. When and where do they meet?",
}


def generate(payload):
    req = urllib.request.Request(f"{URL}/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return time.perf_counter() - t, out


def stream_rate(body, n, ignore_eos=True):
    """Decode tok/s from a streamed request: tokens between the first and the
    last chunk over the wall time between them. Prefill, prefix-cache hits and
    state-cache restores all happen before the first chunk and cancel."""
    payload = {**body, "stream": True,
               "sampling_params": {"max_new_tokens": n, "temperature": 0, "ignore_eos": ignore_eos}}
    req = urllib.request.Request(f"{URL}/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t_first = t_last = None
    n_first = n_last = 0
    meta = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            meta = chunk["meta_info"]
            now = time.perf_counter()
            if t_first is None:
                t_first, n_first = now, meta["completion_tokens"]
            t_last, n_last = now, meta["completion_tokens"]
    return (n_last - n_first) / (t_last - t_first), meta


def decode_rate(body, n, reps):
    """tok/s of decode for a request body (text or input_ids), best of reps."""
    best, meta = 0.0, None
    for _ in range(reps):
        rate, meta = stream_rate(body, n)
        best = max(best, rate)
    return best, meta


def cmd_raw(args):
    for name, text in PROMPTS.items():
        rate, meta = decode_rate({"text": text}, args.n, args.reps)
        print(f"{name:6s} {rate:7.1f} tok/s   (prompt {meta['prompt_tokens']} tok)")


def cmd_context(args):
    random.seed(0)
    for depth in args.depths:
        # random ids from the ordinary-token range: keeps the KV cache real-sized
        # without any chance of hitting special tokens
        ids = [random.randint(1000, 150000) for _ in range(max(depth, 1))]
        rate, meta = decode_rate({"input_ids": ids}, args.n, args.reps)
        print(f"context {depth:>7d}: {rate:7.1f} tok/s   (prompt {meta['prompt_tokens']} tok)")


def cmd_spec(args):
    for name, text in PROMPTS.items():
        # greedy, natural stopping: acceptance is content-dependent, so measure on
        # what the model actually writes
        rate, m = stream_rate({"text": text}, args.n, ignore_eos=False)
        n = m["completion_tokens"]
        verify = m.get("spec_verify_ct", 0)
        acc = n / verify if verify else float("nan")
        print(f"{name:6s} {rate:7.1f} tok/s   {n:4d} tokens in {verify:4d} verify steps"
              f"   mean accepted length {acc:.2f}")


def cmd_files(args):
    """The multi-position protocol of bench/spec_context.py on a server: each file is
    the prompt (N tokens ending at a position), n greedy tokens, one run each; the
    line carries the file name, so the table script can group by context."""
    for path in args.files:
        rate, m = stream_rate({"text": open(path).read()}, args.n, ignore_eos=True)
        n = m["completion_tokens"]
        verify = m.get("spec_verify_ct", 0)
        acc = f"   {n / verify:.2f} per step" if verify else ""
        print(f"file {path}: {rate:7.1f} tok/s   (prompt {m['prompt_tokens']} tok, {n} new){acc}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["raw", "context", "spec", "files"])
    p.add_argument("depths", nargs="*")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    if args.mode == "files":
        args.files = args.depths
    else:
        args.depths = [int(d) for d in args.depths]
    {"raw": cmd_raw, "context": cmd_context, "spec": cmd_spec, "files": cmd_files}[args.mode](args)


if __name__ == "__main__":
    main()
