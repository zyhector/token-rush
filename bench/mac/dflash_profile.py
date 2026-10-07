"""Where a DFlash2 step's time goes: the draft forward alone, the 8-row verify alone,
the two in one graph (as generate_dflash runs them), and the host-side graph build.

    uv run python bench/mac/dflash_profile.py
"""
import time

import mlx.core as mx

from tokenrush.hub import DEFAULT_REPO, DFLASH_REPO, resolve_model
from tokenrush.mlx.dflash import DFlashDraft
from tokenrush.mlx.model import Engine
from tokenrush.mlx.weights import load_packed


def main():
    cfg, w, _ = load_packed(resolve_model(DEFAULT_REPO))
    eng = Engine(cfg, w, max_len=4096)
    d = DFlashDraft(resolve_model(DFLASH_REPO), eng, draft_vocab=mx.arange(131072, dtype=mx.int32))
    ids = mx.random.randint(0, 1000, (512,))
    eng.forward_hidden(ids)
    d.add_context(eng.features)
    mx.eval(eng.state.arrays(), d.arrays())
    base, ctx0 = eng.state.pos, d.ctx
    feats = eng.features[-3:]
    block = mx.concatenate([mx.array([5], mx.int32), mx.full((7,), d.mask_token_id, mx.int32)])

    def run(draft=True, verify=True, build_only=False):
        eng.state.pos, d.ctx = base, ctx0
        outs = []
        t0 = time.perf_counter()
        dr = d.forward(feats, block) if draft else mx.random.randint(0, 1000, (7,))
        if verify:
            n, pred, h = eng.verify(mx.concatenate([block[:1], dr.astype(mx.int32)]))
            outs += [n, pred, h, eng.features] + eng.state.arrays()
        outs += [dr] + d.arrays()
        t1 = time.perf_counter()
        if not build_only:
            mx.eval(outs)
        return t1 - t0, time.perf_counter() - t0

    for _ in range(3):
        run()
    rows = {"draft only": dict(verify=False), "verify only (M=8)": dict(draft=False), "draft + verify": {}}
    for _ in range(2):
        for name, kw in rows.items():
            ts = [run(**kw) for _ in range(6)]
            build = min(t[0] for t in ts) * 1e3
            total = min(t[1] for t in ts) * 1e3
            print(f"{name:<20} {total:6.1f} ms   (graph build on the host {build:4.1f} ms)", flush=True)


if __name__ == "__main__":
    main()
