"""Raw decode on a Mac, ours against mlx-lm on the same int4 bytes, as % of the
measured wall (bench/mac/bandwidth.py; streamed bytes exclude the embedding).

    uv run python bench/mac/decode.py [--tokens 256] [--wall 291.2]
"""
import argparse
import gc
import time

import mlx.core as mx
from transformers import AutoTokenizer

from tokenrush.hub import DEFAULT_REPO, resolve_model
from tokenrush.mlx.model import Engine
from tokenrush.mlx.weights import load_packed

PROMPT = "Write a detailed history of the printing press, from Gutenberg to the present day."


def ours(eng, ids, n, pipelined: bool):
    eng.reset()
    logits = eng.prefill(mx.array(ids))
    tok = mx.argmax(logits[-1:], -1)                       # [1], stays on device
    mx.eval(tok)
    out = []
    t0 = time.perf_counter()
    if pipelined:
        # build step t+1 while the GPU runs step t; the token never visits the host
        logits = eng.forward(tok)
        nxt = mx.argmax(logits, -1)
        mx.async_eval(nxt, eng.state.arrays())
        for _ in range(n - 1):
            cur = nxt
            logits = eng.forward(cur)
            nxt = mx.argmax(logits, -1)
            mx.async_eval(nxt, eng.state.arrays())
            out.append(cur)
        mx.eval(nxt)
        out.append(nxt)
    else:
        for _ in range(n):
            logits = eng.forward(tok)
            tok = mx.argmax(logits, -1)
            mx.eval(tok, eng.state.arrays())
            out.append(tok)
    dt = time.perf_counter() - t0
    return [int(t.item()) for t in out], n / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_REPO)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--wall", type=float, default=291.2, help="GB/s, from bench/mac/bandwidth.py")
    a = ap.parse_args()
    path = resolve_model(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    cfg, w, _ = load_packed(path)
    gb = w.streamed_bytes / 1e9
    ceiling = a.wall / gb
    print(f"{gb:.2f} GB streamed per token; wall {a.wall} GB/s -> ceiling {ceiling:.1f} tok/s")
    eng = Engine(cfg, w, max_len=4096)
    ours(eng, ids, 16, True)                                # warm up kernels
    toks_s, r_sync = ours(eng, ids, a.tokens, False)
    toks_p, r_pipe = ours(eng, ids, a.tokens, True)
    assert toks_s == toks_p, "pipelined decode diverged from synchronous decode"
    for name, r in (("ours, eval per step", r_sync), ("ours, pipelined", r_pipe)):
        print(f"{name:<24} {r:6.2f} tok/s  {100 * r / ceiling:5.1f}% of the wall")
    del eng, w
    gc.collect(); mx.clear_cache()

    from mlx_lm.generate import generate_step
    from mlx_lm_ref import load_mlx_lm
    model = load_mlx_lm(path)
    for rep in range(2):                                    # first pass warms up
        t0 = n = None
        ref = []
        for i, (t, _) in enumerate(generate_step(mx.array(ids), model, max_tokens=a.tokens + 1)):
            if i == 0:
                t0 = time.perf_counter()                    # after prefill + first token
            else:
                ref.append(int(t))
        r = a.tokens / (time.perf_counter() - t0)
    same = sum(x == y for x, y in zip(ref, toks_p))
    print(f"{'mlx-lm ' + __import__('mlx_lm').__version__:<24} {r:6.2f} tok/s  {100 * r / ceiling:5.1f}% of the wall"
          f"   (greedy tokens equal to ours for the first {next((i for i, (x, y) in enumerate(zip(ref, toks_p)) if x != y), len(ref))})")


if __name__ == "__main__":
    main()
