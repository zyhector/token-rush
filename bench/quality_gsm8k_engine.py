#!/usr/bin/env python3
"""GSM8K through our own engine (the packed int4 checkpoint, graphed greedy
decode), with the same prompt and scoring as bench/quality_hf.py --gsm8k, so
the HF-side row for our quant can be cross-checked against the engine the user
actually runs.

    python bench/quality_gsm8k_engine.py [--model <repo id or packed dir>] --out results/quality/gsm8k_engine.json
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.quality_hf import INSTRUCTION, extract, same_number  # noqa: E402
from tokenrush.generate import generate  # noqa: E402
from tokenrush.model import Engine  # noqa: E402
from tokenrush.weights import DEFAULT_REPO, load_packed, resolve_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_REPO, help="packed checkpoint: a Hub repo id or a local directory")
    ap.add_argument("--no-download", action="store_true", help="fail instead of downloading a missing Hub checkpoint")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--out")
    a = ap.parse_args()
    a.model = resolve_model(a.model, download=not a.no_download)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from tokenrush.quant import DEFAULT_BACKEND
    rows = load_dataset("openai/gsm8k", "main", split="test").select(range(a.n))
    tok = AutoTokenizer.from_pretrained(a.model)
    cfg, w, _ = load_packed(a.model, backend=a.backend or DEFAULT_BACKEND)
    eng = Engine(cfg, w, max_len=4096)
    eng.capture()
    stop = set(cfg.eos_ids) | {tok.eos_token_id}
    records, correct, n_tok, t0 = [], 0, 0, time.time()
    for i, r in enumerate(rows):
        text = tok.apply_chat_template([{"role": "user", "content": INSTRUCTION + r["question"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False)
        out, st = generate(eng, tok, ids, a.max_new, stop, stream=False)
        ans = tok.decode(out, skip_special_tokens=True)
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        pred = extract(ans)
        ok = same_number(pred, gold)
        correct += ok
        n_tok += len(out)
        records.append({"i": i, "gold": gold, "pred": pred, "correct": bool(ok), "tokens": len(out), "text": ans})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(rows)}: acc {correct / (i + 1):.3f}  {n_tok / (time.time() - t0):.0f} tok/s", flush=True)
    acc = correct / len(rows)
    print(f"GSM8K {len(rows)} problems: {correct} correct = {acc:.3f}; {sum(r['pred'] is None for r in records)} unparsed; "
          f"{time.time() - t0:.0f}s")
    if a.out:
        json.dump({"src": f"engine:{a.model}", "backend": a.backend or DEFAULT_BACKEND, "n": len(rows), "correct": correct,
                   "accuracy": acc, "max_new": a.max_new, "instruction": INSTRUCTION, "records": records},
                  open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
