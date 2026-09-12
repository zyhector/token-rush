#!/usr/bin/env python3
"""Cut the prompt files a rival needs to follow the multi-position protocol of
bench/spec_context.py: for every context N and position p, the N tokens of the
text ending at p (HF tokenizer), decoded back to text.

    python bench/cut_prompts.py --text /workspace/data/pg19.txt --contexts 0,8000,... --positions 6 \\
        --new 512 --out-dir /workspace/data/prompts/pg19

Positions are resolved exactly as spec_context.py resolves `--positions <count>`
(same lo/hi), and printed. Files: <out-dir>/N_<ctx>_p_<pos>.txt.
"""
import argparse
import os

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zyhector/Qwen3.8-27B-TokenRush-int4g128")
    ap.add_argument("--text", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--new", type=int, default=512)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    all_ids = tok.encode(open(a.text).read()[:6_000_000])
    contexts = [int(c) for c in a.contexts.split(",")]
    lo, hi = max(contexts) + 64, len(all_ids) - a.new
    if a.positions.isdigit() and int(a.positions) < 1000:              # a count; larger numbers are offsets
        P = int(a.positions)
        positions = [lo + (hi - lo) * i // (P - 1) for i in range(P)] if P > 1 else [lo]
    else:
        positions = [int(p) for p in a.positions.split(",")]
    print(f"{len(all_ids)} tokens; positions: {','.join(map(str, positions))}")
    os.makedirs(a.out_dir, exist_ok=True)
    for ctx in contexts:
        n = max(ctx, 64)
        for p in positions:
            path = os.path.join(a.out_dir, f"N_{ctx}_p_{p}.txt")
            open(path, "w").write(tok.decode(all_ids[p - n:p]))
    print(f"{len(contexts) * len(positions)} files in {a.out_dir}")


if __name__ == "__main__":
    main()
