#!/usr/bin/env python3
"""Calibration set for GPTQ: fixed 2048-token windows, disjoint from the
quality-table corpora (bench/quality_corpus.py).

    python bench/quality_calib.py --tok /workspace/models/Qwen3.8-27B --out data/quality/calib.npz

  prose  WikiText-103 *train* (the table uses WikiText-2 *test*), random windows
  code   torch's Python sources outside nn/modules (the table uses nn/modules)
  math   GSM8K train rows 2000 onward (the table's math chunks are the first ~45 rows)
Saved as int32 ids [n, 2048] in the repo (tokenrush/corpus.py) with the mix
and seed: this is the input that chose a checkpoint's codes, and the code
corpus comes from the installed torch, so it cannot be rebuilt bit-for-bit
under another torch version.
"""
import argparse
import glob
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenrush.corpus import save_ids, sha256  # noqa: E402

SEQ = 2048


def windows(ids, n, rng):
    starts = sorted(rng.sample(range(0, len(ids) - SEQ), n))
    return [ids[s:s + SEQ] for s in starts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prose", type=int, default=160)
    ap.add_argument("--code", type=int, default=64)
    ap.add_argument("--math", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    rng = random.Random(a.seed)
    enc = lambda t: tok.encode(t, add_special_tokens=False)

    rows = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"]
    prose = "".join(rows[:400000])                                   # ~25 MB of the 540 MB
    ids = enc(prose)
    print(f"prose: {len(ids)} tokens")
    seqs = windows(ids, a.prose, rng)

    import torch as _t
    root = os.path.dirname(_t.__file__)
    files = sorted(f for f in glob.glob(root + "/**/*.py", recursive=True)
                   if "/nn/modules/" not in f and "/testing/" not in f and "/_inductor/" not in f)
    rng.shuffle(files)
    text, size = [], 0
    for f in files:
        text.append(f"# ==== {os.path.relpath(f, root)} ====\n" + open(f, errors="ignore").read())
        size += os.path.getsize(f)
        if size > 6e6:
            break
    ids = enc("\n".join(text))
    print(f"code: {len(ids)} tokens from {len(text)} files")
    seqs += windows(ids, a.code, rng)

    g = load_dataset("openai/gsm8k", "main", split="train")
    ids = enc("\n\n".join(f"Question: {r['question']}\nAnswer: {r['answer']}" for r in g.select(range(2000, len(g)))))
    print(f"math: {len(ids)} tokens")
    seqs += windows(ids, a.math, rng)

    ids_t = torch.tensor(seqs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_ids(a.out, {"ids": ids_t}, {"seq": SEQ, "mix": [a.prose, a.code, a.math], "seed": a.seed, "tokenizer": a.tok})
    print(f"saved {a.out}: {tuple(ids_t.shape)}, sha256 {sha256(ids_t)}")


if __name__ == "__main__":
    main()
