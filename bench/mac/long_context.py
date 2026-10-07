"""Long context on a Mac: prefill speed, needle retrieval (bench/needle.py's needle and
question), and decode speed at that context, raw and with DFlash2, at a few context
lengths. The haystack is the pre-tokenized WikiText-2 and calibration text in
data/quality. Through Session: the needle question is asked first, then a summary of
the same text, which reuses the prefix and gives a long natural answer to time.

    uv run python bench/mac/long_context.py [--ctx 8192,32768,60000]
"""
import argparse
import time

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from tokenrush.hub import DEFAULT_REPO, DFLASH_REPO, resolve_model
from tokenrush.mlx.dflash import DFlashDraft
from tokenrush.mlx.model import Engine
from tokenrush.mlx.session import Session
from tokenrush.mlx.weights import load_packed

NEEDLE = " The secret passphrase for the Token Rush vault is amber-falcon-7731. "
QUESTION = ("\n\nWhat is the secret passphrase for the Token Rush vault mentioned in the text above? "
            "Answer with the passphrase only.")
SUMMARY = "\n\nSummarize the text above in detail, section by section."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="8192,32768")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--depth", type=float, default=0.5)
    a = ap.parse_args()
    ctxs = [int(c) for c in a.ctx.split(",")]
    path = resolve_model(DEFAULT_REPO)
    tok = AutoTokenizer.from_pretrained(path)
    q = np.load("data/quality/chunks.npz")
    hay = np.concatenate([q["wikitext2"].reshape(-1), np.load("data/quality/calib.npz")["ids"].reshape(-1)]).tolist()
    head = tok.encode("<|im_start|>user\n")
    needle = tok.encode(NEEDLE)
    asst = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    tail_q, tail_s = tok.encode(QUESTION + asst), tok.encode(SUMMARY + asst)
    cfg, w, _ = load_packed(path)
    eng = Engine(cfg, w, max_len=max(ctxs) + a.tokens + 64)
    d = DFlashDraft(resolve_model(DFLASH_REPO), eng, draft_vocab=mx.arange(131072, dtype=mx.int32))
    sess = Session(eng, tok, cfg, dflash=d)
    for L in ctxs:
        n_hay = L - len(head) - len(needle) - len(tail_q)
        cut = int(n_hay * a.depth)
        body = head + hay[:cut] + needle + hay[cut:n_hay]
        row = []
        for mode in ("raw", "dflash"):
            sess.forget()
            list(sess.generate(body, max_new=0, mode=mode))        # prefill once; a snapshot at the text's end
            prefill = sess.stats.prefill_s
            ans = [t for step in sess.generate(body + tail_q, max_new=16, mode=mode) for t in step]
            found = "amber-falcon-7731" in tok.decode(ans)
            list(sess.generate(body + tail_s, max_new=a.tokens, mode=mode))
            st = sess.stats.as_dict()
            row.append(f"{mode} {st['decode_tok_s']:5.1f} tok/s ({st['accepted_per_step']:.2f}/step, "
                       f"reused {st['reused_tokens']}) needle {'found' if found else 'MISSED'}")
        print(f"ctx {len(body):6d}: prefill {len(body) / prefill:5.0f} tok/s ({prefill:.1f} s)  " + "  ".join(row),
              flush=True)

if __name__ == "__main__":
    main()
