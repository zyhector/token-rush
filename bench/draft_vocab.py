#!/usr/bin/env python3
"""Build corpus-specific draft vocabularies: token ids ordered by frequency over chosen
corpora, math-ish tokens (digits, operators, LaTeX, units) boosted, then the rest of the
vocabulary in id order. The first N ids are the rows of lm_head the draft chain reads.

    python bench/draft_vocab.py --en /workspace/data/prose.txt --code /workspace/data/code.txt \
        --zh /workspace/data/zh.txt --out-dir tokenrush/draft_vocab

Writes en_64k.pt (English prose + code), mix_64k.pt and mix_96k.pt (Chinese + English + code).
Which one to use is a deployment choice measured in docs/progress.md (step 21); the
engine's default is the language-neutral id order ('128k'). A list built without a
language cuts that language's speculation below raw decode.
"""
import argparse
import collections
import os
import re

import torch
from transformers import AutoTokenizer

MATHISH = re.compile(r"[0-9]|[=+\-*/^_<>≤≥≠±×÷√∑∫π%$]|\\(frac|sqrt|times|cdot|text|left|right|begin|end|boxed|approx|pm)"
                     r"|km|h\b|km/h|hour|minute|second|meter|speed|distance|time|train|equation|solve|therefore|total|answer|step")


def order_ids(tok, corpora, weights):
    """corpora: name -> text; weights: name -> per-token weight so a smaller corpus
    is not drowned. Returns all ids, most frequent first."""
    counts = collections.Counter()
    for name, text in corpora.items():
        ids = tok.encode(text)
        w = weights.get(name, 1.0)
        for i, c in collections.Counter(ids).items():
            counts[i] += c * w
        print(f"  {name}: {len(ids)} tokens, {len(set(ids))} distinct, weight {w}")
    for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"):
        for i in tok.encode(t):
            counts[i] += 10 ** 12
    for tstr, i in tok.get_vocab().items():
        try:
            s = tok.convert_tokens_to_string([tstr])
        except Exception:
            continue
        if MATHISH.search(s) and len(s) <= 12:
            counts[i] += 1000
    order = [i for i, _ in counts.most_common()]
    seen = set(order)
    return order + [i for i in range(len(tok)) if i not in seen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/Qwen3.8-27B-int4g128")
    ap.add_argument("--en", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--zh", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    en = open(a.en).read()[:8_000_000]
    code = open(a.code).read()[:3_200_000]
    zh = open(a.zh).read()[:6_000_000]
    os.makedirs(a.out_dir, exist_ok=True)
    print("English + code:")
    o = order_ids(tok, {"en": en, "code": code}, {})
    torch.save(torch.tensor(o[:65536], dtype=torch.int32), os.path.join(a.out_dir, "en_64k.pt"))
    print("Chinese + English + code (Chinese weighted to match English in tokens):")
    o = order_ids(tok, {"zh": zh, "en": en, "code": code}, {"zh": 1.0, "en": 1.0, "code": 1.0})
    for n, name in ((65536, "mix_64k.pt"), (98304, "mix_96k.pt")):
        torch.save(torch.tensor(o[:n], dtype=torch.int32), os.path.join(a.out_dir, name))
    cjk = [i for i in range(len(tok)) if any('一' <= ch <= '鿿' for ch in tok.decode([i]))]
    for name in ("en_64k", "mix_64k", "mix_96k"):
        ids = set(torch.load(os.path.join(a.out_dir, name + ".pt")).tolist())
        print(f"  {name}: CJK tokens covered {sum(1 for i in cjk if i in ids)}/{len(cjk)}")


if __name__ == "__main__":
    main()
