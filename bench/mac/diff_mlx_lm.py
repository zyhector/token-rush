"""Engine correctness on a Mac: our MLX engine against mlx-lm on the same int4 bytes.

Teacher-forced: our engine prefills a prompt and greedy-decodes N tokens step by
step (the decode path, not just prefill); mlx-lm then scores the whole sequence in
one forward. Per position: argmax agreement, max |logit diff|, KL(mlx-lm || ours).
The two never share a kernel (GDN, attention, norms, rope all differ), so
agreement is evidence about both.

    uv run python bench/mac/diff_mlx_lm.py [--tokens 64]
"""
import argparse
import gc
import time

import mlx.core as mx
from transformers import AutoTokenizer

from tokenrush.mlx.model import Engine
from tokenrush.mlx.weights import load_packed
from tokenrush.hub import DEFAULT_REPO, resolve_model

PROMPTS = [
    "The history of the Roman Empire spans",
    "def quicksort(arr):\n    \"\"\"Sort a list in place.\"\"\"\n",
    "Q: A train travels 120 km in 1.5 hours. What is its average speed?\nA:",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_REPO)
    ap.add_argument("--tokens", type=int, default=48)
    a = ap.parse_args()
    path = resolve_model(a.model)
    tok = AutoTokenizer.from_pretrained(path)

    cfg, w, _ = load_packed(path)
    eng = Engine(cfg, w, max_len=4096)
    runs = []
    for p in PROMPTS:
        ids = tok(p)["input_ids"]
        eng.reset()
        logits = [eng.prefill(mx.array(ids))]
        seq = list(ids)
        t0 = time.perf_counter()
        for _ in range(a.tokens):
            nxt = int(mx.argmax(logits[-1][-1]).item())
            seq.append(nxt)
            logits.append(eng.forward(mx.array([nxt])))
            mx.eval(logits[-1], eng.state.arrays())
        dt = time.perf_counter() - t0
        ours = mx.concatenate([l[-1:] for l in logits[:-1]]).astype(mx.float32)   # predicting seq[len(ids):]
        mx.eval(ours)
        runs.append((ids, seq, ours))
        print(f"ours: {a.tokens} tokens in {dt:.2f}s ({a.tokens / dt:.1f} tok/s eager, eval per step)")
        print("   ", repr(tok.decode(seq[len(ids):])[:160]))
    del eng, w
    gc.collect()
    mx.clear_cache()

    from mlx_lm_ref import load_mlx_lm
    model = load_mlx_lm(path)
    worst = 0
    for ids, seq, ours in runs:
        full = model(mx.array(seq[:-1])[None])[0].astype(mx.float32)               # [len-1, vocab]
        ref = full[len(ids) - 1:]
        agree = (mx.argmax(ref, -1) == mx.argmax(ours, -1)).astype(mx.float32).mean().item()
        lp, lq = ref - mx.logsumexp(ref, -1, keepdims=True), ours - mx.logsumexp(ours, -1, keepdims=True)
        kl = (mx.exp(lp) * (lp - lq)).sum(-1)
        dmax = mx.abs(ref - ours).max().item()
        print(f"top-1 agreement {agree:.3f}  KL mean {kl.mean().item():.2e} max {kl.max().item():.2e}  max|dlogit| {dmax:.3f}")
        worst = max(worst, kl.max().item())
    print("worst KL", f"{worst:.2e}")


if __name__ == "__main__":
    main()
