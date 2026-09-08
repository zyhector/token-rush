#!/usr/bin/env python3
"""Build the draft vocabulary: token ids ordered by frequency over the local prose and
code corpora, with math-ish tokens (digits, operators, LaTeX, units) boosted since the
corpora lack them, then the rest of the vocabulary. The first N ids are the rows of
lm_head the draft chain reads (tokenrush/draft_vocab_64k.pt ships the first 65536).

    python bench/draft_vocab.py --prose /workspace/data/prose.txt --code /workspace/data/code.txt --out tokenrush/draft_vocab_64k.pt
"""
import argparse
import collections
import re

import torch
from transformers import AutoTokenizer

MATHISH = re.compile(r"[0-9]|[=+\-*/^_<>≤≥≠±×÷√∑∫π%$]|\\(frac|sqrt|times|cdot|text|left|right|begin|end|boxed|approx|pm)"
                     r"|km|h\b|km/h|hour|minute|second|meter|speed|distance|time|train|equation|solve|therefore|total|answer|step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/Qwen3.8-27B-int4g128")
    ap.add_argument("--prose", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--n", type=int, default=65536)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    counts = collections.Counter()
    for f, n in ((a.prose, 8_000_000), (a.code, 3_200_000)):
        counts.update(tok.encode(open(f).read()[:n]))
    for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"):
        for i in tok.encode(t):
            counts[i] += 10 ** 9
    n_math = 0
    for tstr, i in tok.get_vocab().items():
        try:
            s = tok.convert_tokens_to_string([tstr])
        except Exception:
            continue
        if MATHISH.search(s) and len(s) <= 12:
            counts[i] += 1000
            n_math += 1
    order = [i for i, _ in counts.most_common()]
    seen = set(order)
    order += [i for i in range(len(tok)) if i not in seen]
    torch.save(torch.tensor(order[:a.n], dtype=torch.int32), a.out)
    print(f"{len(counts)} ids with counts ({n_math} math-ish boosted); saved the first {a.n} to {a.out}")


if __name__ == "__main__":
    main()
