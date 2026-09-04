#!/usr/bin/env python3
"""bs=1 decode tok/s against a running ollama server.

ollama reports eval_count and eval_duration (ns) per response, which is the
decode phase only; tok/s is their ratio. Greedy, fixed token budget, best of
`reps`. Usage: ollama_bench.py <model> [--url http://127.0.0.1:11434]
"""
import argparse
import json
import urllib.request

PROMPTS = {
    "essay": "Write a detailed essay on the causes and consequences of the fall of the "
             "Western Roman Empire, covering political, economic and military factors.",
    "code": "Write a Python implementation of a thread-safe LRU cache with TTL expiry, "
            "including unit tests.",
    "math": "Solve step by step: A train leaves city A at 60 km/h and another leaves city B, "
            "450 km away, at 90 km/h toward it one hour later. When and where do they meet?",
}


def generate(url, model, prompt, n):
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": 0, "num_predict": n}, "keep_alive": "10m"}
    req = urllib.request.Request(f"{url}/api/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return out["eval_count"] / (out["eval_duration"] / 1e9), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--url", default="http://127.0.0.1:11434")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    generate(args.url, args.model, "warm up", 8)
    for name, text in PROMPTS.items():
        best = 0.0
        for _ in range(args.reps):
            rate, out = generate(args.url, args.model, text, args.n)
            best = max(best, rate)
        print(f"{name:6s} {best:7.1f} tok/s   ({out['eval_count']} tokens)")


if __name__ == "__main__":
    main()
