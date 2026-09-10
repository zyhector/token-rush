#!/usr/bin/env python3
"""Needle-in-a-haystack retrieval at long context: is 256k *usable*, not merely loadable.

    python bench/needle.py [--model <repo id or packed dir>] --text /workspace/data/prose.txt --contexts 131072,262144

A passphrase sentence is buried half-way through `context` tokens of real prose
(fp8 KV, chunked prefill); the model is asked for it with the chat template,
thinking off, and answers with raw graphed greedy decode. Prints retrieved /
missed per context with the prefill time, the decode tok/s at that length and
the peak VRAM. Phase 4 runs this at 128k and 256k.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenrush.generate import generate  # noqa: E402
from tokenrush.model import Engine  # noqa: E402
from tokenrush.weights import DEFAULT_REPO, load_packed, resolve_model  # noqa: E402

NEEDLE = " The secret passphrase for the Token Rush vault is amber-falcon-7731. "
QUESTION = "\n\nWhat is the secret passphrase for the Token Rush vault mentioned in the text above? Answer with the passphrase only."
PASS = "amber-falcon-7731"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_REPO, help="packed checkpoint: a Hub repo id or a local directory")
    ap.add_argument("--no-download", action="store_true", help="fail instead of downloading a missing Hub checkpoint")
    ap.add_argument("--text", required=True, help="a long prose file, the haystack")
    ap.add_argument("--contexts", default="131072,262144", help="prompt lengths in tokens (clamped to --max-len)")
    ap.add_argument("--depth", type=float, default=0.5, help="where in the haystack the needle goes")
    ap.add_argument("--max-len", type=int, default=262144)
    ap.add_argument("--new", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=4096)
    a = ap.parse_args()
    a.model = resolve_model(a.model, download=not a.no_download)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    cfg, w, _ = load_packed(a.model)
    eng = Engine(cfg, w, max_len=a.max_len, kv_dtype=torch.float8_e4m3fn)
    eng.capture()
    stop = set(cfg.eos_ids) | {tok.eos_token_id}
    head = tok.encode("<|im_start|>user\n")
    needle = tok.encode(NEEDLE)
    tail = tok.encode(QUESTION + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    hay = tok.encode(open(a.text).read()[:6_000_000])
    print(f"{len(hay)} tokens of haystack available; needle {len(needle)} tokens at depth {a.depth:.0%}")
    for ctx in (int(c) for c in a.contexts.split(",")):
        ctx = min(ctx, a.max_len - a.new - 1)
        n_hay = ctx - len(head) - len(needle) - len(tail)
        d = int(n_hay * a.depth)
        ids = head + hay[:d] + needle + hay[d:n_hay] + tail
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out, st = generate(eng, tok, ids, a.new, stop, stream=False, chunk=a.chunk)
        ans = tok.decode(out, skip_special_tokens=True).strip()
        ok = PASS in ans
        print(f"context {len(ids):>7}: {'RETRIEVED' if ok else 'MISSED'}  {ans!r}  | prefill {st['prefill_s']:.0f}s "
              f"({st['prefill_tok_s']:.0f} tok/s), decode {st['decode_tok_s']:.1f} tok/s, "
              f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
