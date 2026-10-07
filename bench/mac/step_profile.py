"""Where a speculative step's time goes on a Mac: the verify step alone at M = 1..8
rows, and the MTP draft chain alone at depth K, each evaluated by itself.

    uv run python bench/mac/step_profile.py
"""
import time

import mlx.core as mx

from tokenrush.hub import DEFAULT_REPO, resolve_model
from tokenrush.mlx import weights as W
from tokenrush.mlx.model import Engine
from tokenrush.mlx.spec import MTPHead
from tokenrush.mlx.weights import load_packed


def timed(fn, reps=8):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    path = resolve_model(DEFAULT_REPO)
    cfg, w, mtp_t = load_packed(path, with_mtp=True)
    eng = Engine(cfg, w, max_len=4096)
    mtp = MTPHead(eng, mtp_t)
    ids = mx.random.randint(0, 1000, (512,))
    eng.prefill(ids)
    base_pos = eng.state.pos

    def verify(M):
        eng.state.pos = base_pos
        n, pred, h = eng.verify(mx.random.randint(0, 1000, (M,)))
        mx.eval(n, pred, h, eng.state.arrays())

    # the two routings alternate per M, best of 3, so heat and order affect both alike
    for M in range(1, 9):
        best = {}
        for _ in range(3):
            for name, rows in (("mlx", [0, -1]), ("nax", [5, 8])):
                W.NAX_ROWS[:] = rows
                best[name] = min(best.get(name, 1e9), timed(lambda: verify(M), reps=4))
        print(f"verify M={M}: mlx {best['mlx']:6.1f} ms   routed {best['nax']:6.1f} ms", flush=True)
    W.NAX_ROWS[:] = [5, 8]
    h = mx.random.normal((1, cfg.hidden)).astype(mx.bfloat16)

    def chain(K):
        mtp.state.pos = base_pos
        x = mtp.hidden(mx.array([1, 2, 3]), mx.concatenate([h, h, h]))[-1:]
        ds = []
        for i in range(K):
            d = mtp.draft(x)
            ds.append(d)
            if i + 1 < K:
                x = mtp.hidden(d, x)
        mx.eval(ds, mtp.state.arrays())

    print("draft chain ms " + "  ".join(f"K={K}:{timed(lambda: chain(K)):.1f}" for K in range(1, 8)))


if __name__ == "__main__":
    main()
