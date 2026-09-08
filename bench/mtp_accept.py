#!/usr/bin/env python3
"""Acceptance rate of the shipped MTP head as a chained draft, teacher-forced
against the target's own greedy continuation. Phase 3, step 1.

    python bench/mtp_accept.py --model /workspace/models/Qwen3.8-27B-int4g128 [--new 256] [--depth 3]

For each prompt family the target generates `new` greedy tokens (eager,
recording its post-norm hidden state at every position). Then at every
position the MTP head drafts a chain of `depth` tokens exactly as the engine
will: pair (true next token, target hidden) at depth 1, then (draft token,
MTP's own normed hidden) for deeper steps, each drafted token compared with
the true continuation. Reports P(depth-k draft correct | all shallower
correct) and the mean accepted tokens per verify step (with the bonus token)
that a chain of each depth would give.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenrush.model import Engine  # noqa: E402
from tokenrush.mtp import MTPHead, build_mtp  # noqa: E402
from tokenrush.weights import load_packed  # noqa: E402

PROMPTS = {   # the Phase 0 families (scripts/rivals/vllm_bench.py), chat-formatted, thinking off
    "essay": "Write a detailed essay on the causes and consequences of the fall of the "
             "Western Roman Empire, covering political, economic and military factors.",
    "code": "Write a Python implementation of a thread-safe LRU cache with TTL expiry, "
            "including unit tests.",
    "math": "Solve step by step: A train leaves city A at 60 km/h and another leaves city B, "
            "450 km away, at 90 km/h toward it one hour later. When and where do they meet?",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    cfg, w, mtp_t = load_packed(a.model, with_mtp=True)
    eng = Engine(cfg, w, max_len=a.max_len)
    mtp = MTPHead(cfg, build_mtp(cfg, mtp_t, "cuda"), w.embed, w.lm_head, a.max_len)
    print(f"MTP head: {sum(t.numel() * 2 for t in mtp_t.values()) / 1e9:.2f} GB bf16")

    totals = torch.zeros(a.depth)
    n_total = 0
    for name, text in PROMPTS.items():
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False))
        T = len(ids)
        x = torch.tensor(ids, device="cuda")
        # 1. target: greedy continuation with hidden states at every position
        eng.reset()
        logits = eng.forward(x, all_logits=True)
        hidden = [eng.last_hidden]                                   # [T, hidden]
        toks = list(ids)
        nxt = int(logits[-1].argmax())
        for _ in range(a.new):
            toks.append(nxt)
            logits = eng.decode(torch.tensor([nxt], device="cuda"))
            hidden.append(eng.last_hidden)
            nxt = int(logits[-1].argmax())
        toks.append(nxt)
        H = torch.cat(hidden)                                        # H[p] = hidden at position p
        toks_t = torch.tensor(toks, device="cuda")                   # toks[p+1] = argmax from H[p]
        L = len(toks) - 1

        # 2. MTP over the prompt: pairs (tok[t+1], H[t]) at positions 1..T-1
        mtp.reset()
        mtp.set_pos(1)
        mtp.forward(toks_t[1:T], H[:T - 1])

        # 3. chained drafts at every generated position
        correct = torch.zeros(a.depth)
        n = 0
        t0 = time.time()
        for p in range(T - 1, L - a.depth - 1):
            mtp.set_pos(p + 1)
            tok_in, h_in = toks_t[p + 1:p + 2], H[p:p + 1]
            ok = True
            for d in range(a.depth):
                lg, hn = mtp.forward(tok_in, h_in)
                draft = int(lg[-1].argmax())
                ok = ok and draft == toks[p + 2 + d]
                if ok:
                    correct[d] += 1
                else:
                    break
                tok_in, h_in = torch.tensor([draft], device="cuda"), hn
            n += 1
        dt = time.time() - t0
        rate = correct / n
        acc = 1 + rate.cumsum(0)
        print(f"\n{name}: {T} prompt + {a.new} greedy tokens; {n} draft positions; {dt / n * 1e3:.1f} ms per chained draft (eager)")
        print("  P(draft k correct | shallower correct): " + "  ".join(f"k={k + 1}: {rate[k]:.3f}" for k in range(a.depth)))
        print("  mean accepted tokens per verify step, chain depth d: " + "  ".join(f"d={k + 1}: {acc[k]:.2f}" for k in range(a.depth)))
        print("  target text: " + repr(tok.decode(toks[T:T + 40])))
        totals += correct
        n_total += n
    rate = totals / n_total
    acc = 1 + rate.cumsum(0)
    print("\nall families: " + "  ".join(f"k={k + 1}: {rate[k]:.3f}" for k in range(a.depth))
          + " | accepted/step: " + "  ".join(f"d={k + 1}: {acc[k]:.2f}" for k in range(a.depth)))


if __name__ == "__main__":
    main()
