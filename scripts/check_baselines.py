#!/usr/bin/env python3
"""Check the arithmetic in docs/baselines.md: every context-table row must satisfy
ceiling = WALL / bytes and % of wall = tok/s * bytes / WALL.

    python scripts/check_baselines.py [docs/baselines.md] [--wall 1701]

Worth running after any edit to those tables, and especially in Phase 4 when
they are all rewritten on a new machine. A wrong byte count is the error this
project actually made: counting the embedding table, which a decode step does
not stream, put vLLM at 88% of the wall instead of 76% in every document until
2026-09-09 (`docs/progress.md` step 31).
"""
import argparse
import re
import sys

ROW = re.compile(r"^\|\s*(\d+k?)\s*\|\s*([\d.]+)\s*tok/s\s*\|\s*([\d.]+)\s*GB\s*\|\s*([\d.]+)\s*\|\s*\**([\d.]+)%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="docs/baselines.md")
    ap.add_argument("--wall", type=float, default=1701.0, help="measured read bandwidth, GB/s")
    ap.add_argument("--tol", type=float, default=0.6, help="tolerance in tok/s and in points of %")
    a = ap.parse_args()
    section, n, bad = "?", 0, 0
    for i, line in enumerate(open(a.path), 1):
        if line.startswith("## "):
            section = line[3:].strip()
        m = ROW.match(line.strip())
        if not m:
            continue
        n += 1
        _, tps, gb, ceiling, pct = (float(x) if j else x for j, x in enumerate(m.groups()))
        want_ceiling, want_pct = a.wall / gb, tps * gb / a.wall * 100
        if abs(want_ceiling - ceiling) > a.tol or abs(want_pct - pct) > a.tol:
            bad += 1
            print(f"{a.path}:{i} [{section}] {tps} tok/s on {gb} GB: "
                  f"ceiling {ceiling} (expect {want_ceiling:.1f}), {pct}% (expect {want_pct:.1f}%)")
    print(f"{n} context-table rows checked against a {a.wall:.0f} GB/s wall, {bad} inconsistent")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
