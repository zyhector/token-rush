#!/usr/bin/env python3
"""bs=1 decode measurements for ExLlamaV3, in-process.

Same prompts and method as the server benchmarks: greedy, a fixed number of new
tokens, decode tok/s from the generator's own per-job timing (time_generate /
new_tokens), which excludes prefill. `context` mode prefills random ids of
the given length first, as the other rival sweeps do.

Usage:
  exl3_bench.py <model_dir> raw
  exl3_bench.py <model_dir> context 0 22000 90000 200000
  exl3_bench.py <model_dir> mtp --draft-tokens 1   # the checkpoint's own MTP head as draft
"""
import argparse
import random
import time

import torch
from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import GreedySampler

PROMPTS = {
    "essay": "Write a detailed essay on the causes and consequences of the fall of the "
             "Western Roman Empire, covering political, economic and military factors.",
    "code": "Write a Python implementation of a thread-safe LRU cache with TTL expiry, "
            "including unit tests.",
    "math": "Solve step by step: A train leaves city A at 60 km/h and another leaves city B, "
            "450 km away, at 90 km/h toward it one hour later. When and where do they meet?",
}


def load(model_dir, max_tokens, draft_tokens=0):
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    # max_history reserves recurrent-state history for speculative rollback in the
    # GDN layers; without it the draft path fails at the first verify
    cache = Cache(model, max_num_tokens=max_tokens, max_history=draft_tokens)
    model.load(progressbar=False)
    tokenizer = Tokenizer.from_config(config)
    kw = {}
    if draft_tokens:
        draft = Model.from_config(config, component="mtp")
        draft_cache = Cache(draft, max_num_tokens=max_tokens)   # must exist before load()
        draft.load(progressbar=False)
        kw = dict(draft_model=draft, draft_cache=draft_cache, num_draft_tokens=draft_tokens)
    gen = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=4096, **kw)
    return model, cache, tokenizer, gen


def run(gen, ids, n, reps, **job_kwargs):
    """Best decode tok/s over reps, from the job's own time_generate."""
    best = 0.0
    for _ in range(reps):
        job = Job(input_ids=ids, max_new_tokens=n, min_new_tokens=n,
                  sampler=GreedySampler(), stop_conditions=[], **job_kwargs)
        gen.enqueue(job)
        last = None
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("eos"):
                    last = r
        best = max(best, last["new_tokens"] / last["time_generate"])
        run.last = last
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("mode", choices=["raw", "context", "mtp"])
    p.add_argument("--draft-tokens", type=int, default=1)
    p.add_argument("depths", nargs="*", type=int)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=32768)
    args = p.parse_args()
    max_tokens = max([args.max_tokens] + [d + args.n + 64 for d in args.depths])
    max_tokens = (max_tokens + 255) // 256 * 256          # cache pages are 256 tokens
    model, cache, tokenizer, gen = load(args.model_dir, max_tokens,
                                        args.draft_tokens if args.mode == "mtp" else 0)
    print(f"loaded {args.model_dir}, cache {max_tokens} tokens, "
          f"{torch.cuda.memory_allocated() / 1e9:.1f} GB allocated")
    if args.mode == "mtp" and args.depths:
        # chained MTP vs. tokens already in the cache (random ids, as the raw sweep)
        random.seed(0)
        for depth in args.depths:
            ids = torch.tensor([[random.randint(1000, 150000) for _ in range(max(depth, 1))]])
            rate = run(gen, ids, args.n, args.reps)
            r = run.last
            acc = r.get("accepted_draft_tokens", 0)
            print(f"context {depth:>7d}: {rate:7.1f} tok/s   {1 + acc / max(r['new_tokens'] - acc, 1):.2f} per step")
    elif args.mode in ("raw", "mtp"):
        for name, text in PROMPTS.items():
            ids = tokenizer.encode(text, add_bos=False)
            rate = run(gen, ids, args.n, args.reps)
            extra = ""
            if args.mode == "mtp":
                r = run.last
                acc = r.get("accepted_draft_tokens", 0)
                extra = f"   accepted draft tokens {acc}/{r['new_tokens']} -> {1 + acc / max(r['new_tokens'] - acc, 1):.2f} per step"
            print(f"{name:6s} {rate:7.1f} tok/s{extra}")
    else:
        random.seed(0)
        for depth in args.depths:
            ids = torch.tensor([[random.randint(1000, 150000) for _ in range(max(depth, 1))]])
            print(f"context {depth:>7d}: {run(gen, ids, args.n, args.reps):7.1f} tok/s")


if __name__ == "__main__":
    main()
