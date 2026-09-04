#!/usr/bin/env python3
"""bs=1 decode tok/s against a running vLLM (OpenAI-compatible) server.

Same method as sglang_bench.py: streamed request, tokens between first and
last chunk over the wall time between them, greedy, ignore_eos, best of
`reps`. Usage:
  vllm_bench.py raw
  vllm_bench.py context 0 22000 90000 200000   # random-token prompts already in KV
"""
import argparse
import json
import time
import urllib.request

PROMPTS = {
    "essay": "Write a detailed essay on the causes and consequences of the fall of the "
             "Western Roman Empire, covering political, economic and military factors.",
    "code": "Write a Python implementation of a thread-safe LRU cache with TTL expiry, "
            "including unit tests.",
    "math": "Solve step by step: A train leaves city A at 60 km/h and another leaves city B, "
            "450 km away, at 90 km/h toward it one hour later. When and where do they meet?",
}


def stream_rate(url, model, prompt, n):
    payload = {"model": model, "prompt": prompt, "max_tokens": n, "temperature": 0,
               "ignore_eos": True, "stream": True}
    req = urllib.request.Request(f"{url}/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t_first = t_last = None
    count = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data:") or b"[DONE]" in line:
                continue
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            else:
                count += 1
                t_last = now
    return count / (t_last - t_first)


def best_rate(url, model, prompt, n, reps):
    return max(stream_rate(url, model, prompt, n) for _ in range(reps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["raw", "context"])
    p.add_argument("depths", nargs="*", type=int)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    with urllib.request.urlopen(f"{args.url}/v1/models") as r:
        model = json.load(r)["data"][0]["id"]
    if args.mode == "raw":
        for name, text in PROMPTS.items():
            print(f"{name:6s} {best_rate(args.url, model, text, args.n, args.reps):7.1f} tok/s")
    else:
        import random
        random.seed(0)
        for depth in args.depths:
            ids = [random.randint(1000, 150000) for _ in range(max(depth, 1))]
            rate = best_rate(args.url, model, ids, args.n, args.reps)
            print(f"context {depth:>7d}: {rate:7.1f} tok/s")


if __name__ == "__main__":
    main()
