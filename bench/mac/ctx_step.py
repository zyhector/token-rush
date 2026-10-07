"""Decode and verify step time against context length, without a long prefill: the
position is set directly (attention reads the same number of bytes whatever the cache
holds). Interleaved over positions so heat affects all alike.

    uv run python bench/mac/ctx_step.py [--pos 512,8192,32768,60000]
"""
import argparse
import time

import mlx.core as mx

from tokenrush.hub import DEFAULT_REPO, resolve_model
from tokenrush.mlx.model import Engine
from tokenrush.mlx.weights import load_packed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="512,8192,32768,60000")
    a = ap.parse_args()
    ps = [int(p) for p in a.pos.split(",")]
    cfg, w, _ = load_packed(resolve_model(DEFAULT_REPO))
    eng = Engine(cfg, w, max_len=max(ps) + 16)
    kv_per_tok = sum(k.nbytes + v.nbytes for k, v in zip(eng.state.k, eng.state.v)) / eng.max_len
    best = {}
    for _ in range(3):
        for p in ps:
            for M in (1, 8):
                eng.state.pos = p
                t0 = time.perf_counter()
                n, pred, h = eng.verify(mx.random.randint(0, 1000, (M,)))
                mx.eval(n, pred, h, eng.state.arrays())
                dt = time.perf_counter() - t0
                best[p, M] = min(best.get((p, M), 1e9), dt)
    wall = 291.2
    for p in ps:
        ceiling = wall / (w.streamed_bytes / 1e9 + kv_per_tok * p / 1e9)
        print(f"pos {p:6d}: 1 row {best[p, 1] * 1e3:6.1f} ms ({1 / best[p, 1]:5.1f} tok/s, ceiling {ceiling:5.1f})   "
              f"8 rows {best[p, 8] * 1e3:6.1f} ms ({best[p, 8] / best[p, 1]:.2f}x)")


if __name__ == "__main__":
    main()
