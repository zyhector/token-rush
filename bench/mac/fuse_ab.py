"""A/B of model.FUSE (compiled elementwise groups) on the real model, interleaved:
pipelined raw decode tok/s and the 8-row verify step.

    uv run python bench/mac/fuse_ab.py
"""
import time

import mlx.core as mx

from tokenrush.hub import DEFAULT_REPO, resolve_model
from tokenrush.mlx import model as M
from tokenrush.mlx.weights import load_packed


def decode(eng, n=48):
    eng.reset()
    tok = mx.argmax(eng.prefill(mx.random.randint(0, 1000, (64,)))[-1:], -1)
    mx.eval(tok)
    nxt = mx.argmax(eng.forward(tok), -1)
    mx.async_eval(nxt, eng.state.arrays())
    t0 = time.perf_counter()
    for _ in range(n):
        nxt = mx.argmax(eng.forward(nxt), -1)
        mx.async_eval(nxt, eng.state.arrays())
    mx.eval(nxt)
    return n / (time.perf_counter() - t0)


def verify8(eng, reps=5):
    base = eng.state.pos
    best = 1e9
    for _ in range(reps):
        eng.state.pos = base
        t0 = time.perf_counter()
        n, p, h = eng.verify(mx.random.randint(0, 1000, (8,)))
        mx.eval(n, p, h, eng.state.arrays())
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main():
    cfg, w, _ = load_packed(resolve_model(DEFAULT_REPO))
    eng = M.Engine(cfg, w, max_len=4096)
    res = {True: [], False: []}
    for rep in range(3):
        for fuse in (False, True):
            M.FUSE = fuse
            decode(eng, 8)
            res[fuse].append((decode(eng), verify8(eng)))
    for fuse in (False, True):
        print(f"FUSE={fuse!s:<5}  decode {max(r[0] for r in res[fuse]):5.2f} tok/s   "
              f"verify M=8 {min(r[1] for r in res[fuse]):5.1f} ms")


if __name__ == "__main__":
    main()
