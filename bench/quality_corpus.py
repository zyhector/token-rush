#!/usr/bin/env python3
"""Build the token chunks the quantization-quality table is measured on.

    python bench/quality_corpus.py --tok /workspace/models/Qwen3.8-27B --out data/quality/chunks.npz

Three corpora, tokenized once and cut into contiguous non-overlapping chunks
of 4096 tokens, so every candidate sees exactly the same positions:
  wikitext2  WikiText-2 test (raw), the first 16 chunks = 65536 tokens
  code       torch's Python sources (torch/nn/modules/*.py, sorted), 2 chunks
  math       GSM8K *train* problems with their worked solutions, 2 chunks
             (the accuracy task uses the disjoint test split)
Saved as {name: int32 ids [n_chunks, 4096]} in the repo (tokenrush/corpus.py):
the code corpus is read from the installed torch, so the ids cannot be
rebuilt bit-for-bit under another torch version.
"""
import argparse
import glob
import os

import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenrush.corpus import save_ids, sha256  # noqa: E402

CHUNK = 4096


def wikitext2():
    from datasets import load_dataset
    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    # the rows are the lines of wiki.test.raw, newline included; joining with ""
    # reproduces the file llama.cpp's perplexity tool reads
    assert all(r == "" or r.endswith("\n") for r in rows[:100])
    return "".join(rows)


def code():
    import torch as _t
    root = os.path.join(os.path.dirname(_t.__file__), "nn", "modules")
    parts = []
    for fn in sorted(glob.glob(os.path.join(root, "*.py"))):
        parts.append(f"# ==== torch/nn/modules/{os.path.basename(fn)} ====\n" + open(fn).read())
    return "\n".join(parts)


def math():
    from datasets import load_dataset
    rows = load_dataset("openai/gsm8k", "main", split="train")
    return "\n\n".join(f"Question: {r['question']}\nAnswer: {r['answer']}" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wikitext-chunks", type=int, default=16)
    ap.add_argument("--code-chunks", type=int, default=2)
    ap.add_argument("--math-chunks", type=int, default=2)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    out, meta = {}, {"chunk": CHUNK, "tokenizer": a.tok}
    for name, fn, n in (("wikitext2", wikitext2, a.wikitext_chunks), ("code", code, a.code_chunks),
                        ("math", math, a.math_chunks)):
        text = fn()
        ids = tok.encode(text, add_special_tokens=False)
        need = n * CHUNK
        assert len(ids) >= need, f"{name}: {len(ids)} tokens < {need}"
        out[name] = torch.tensor(ids[:need]).view(n, CHUNK)
        print(f"{name}: {len(text) / 1e6:.2f} M chars, {len(ids)} tokens, using {n} x {CHUNK}, sha256 {sha256(out[name])}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_ids(a.out, out, meta)
    print("saved", a.out)


if __name__ == "__main__":
    main()
