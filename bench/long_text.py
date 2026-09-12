#!/usr/bin/env python3
"""Build the long-text corpora that bench/spec_context.py and bench/needle.py read.

    python bench/long_text.py --out-dir /workspace/data

prose.txt: WikiText-103 train, the first ~6 MB of non-empty lines.
code.txt: torch's own Python sources (files >= 2 KB, in path order), ~6 MB.
pg19.txt: PG-19 (Project Gutenberg books, the corpus the long-context
speculative-decoding papers use; the `emozilla/pg19` parquet mirror, since
`deepmind/pg19` is a loading script that current `datasets` refuses), whole
books in dataset order up to ~6 MB, skipping the first two (the King James
Bible and the Complete Shakespeare: 10 MB of verse and drama, not prose).
The multi-position sweep (docs/progress.md step 34) reads this one, because
WikiText's article boundaries and `= =` headings are their own noise source.
Phase 4 (docs/progress.md step 32) used the first two files; the 200k rows are
sensitive to what the text is doing at that position, so the *_shift.txt
variants (the same files from 1.5 MB in) are what the attribution probe used.
"""
import argparse
import glob
import os


def load_pg19(chars, skip=2):
    """Whole PG-19 books in dataset order (streamed, so nothing but these is fetched),
    each one's Gutenberg text as published, joined by a blank line."""
    from datasets import load_dataset
    buf, n = [], 0
    for i, r in enumerate(load_dataset("emozilla/pg19", split="train", streaming=True)):
        if i < skip:
            continue
        t = r["text"]
        buf.append(t.strip() + "\n\n"); n += len(t)
        print(f"  + {r.get('short_book_title', '?')[:60]!r}: {len(t) / 1e6:.2f} MB")
        if n > chars:
            break
    return "".join(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/data")
    ap.add_argument("--chars", type=int, default=6_000_000)
    ap.add_argument("--shift", type=int, default=1_500_000)
    ap.add_argument("--only", default=None, choices=("prose", "code", "pg19"), help="build one corpus only")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    from datasets import load_dataset
    if a.only == "pg19":
        pg19 = load_pg19(a.chars)
        open(os.path.join(a.out_dir, "pg19.txt"), "w").write(pg19)
        print(f"pg19: {len(pg19) / 1e6:.1f} MB")
        return
    import torch
    rows = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    buf, n = [], 0
    for r in rows:
        t = r["text"]
        if not t.strip():
            continue
        buf.append(t); n += len(t)
        if n > a.chars:
            break
    prose = "".join(buf)
    buf, n = [], 0
    root = os.path.dirname(torch.__file__)
    for p in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        try:
            s = open(p).read()
        except Exception:
            continue
        if len(s) < 2000:
            continue
        buf.append(f"# ---- {os.path.relpath(p, root)}\n{s}\n"); n += len(s)
        if n > a.chars:
            break
    code = "".join(buf)
    for name, text in (("prose", prose), ("code", code)):
        open(os.path.join(a.out_dir, f"{name}.txt"), "w").write(text)
        open(os.path.join(a.out_dir, f"{name}_shift.txt"), "w").write(text[a.shift:])
        print(f"{name}: {len(text) / 1e6:.1f} MB")
    pg19 = load_pg19(a.chars)
    open(os.path.join(a.out_dir, "pg19.txt"), "w").write(pg19)
    print(f"pg19: {len(pg19) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
