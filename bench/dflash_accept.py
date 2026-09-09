#!/usr/bin/env python3
"""Teacher-forced acceptance of z-lab's DFlash2 block-diffusion draft for Qwen3.8-27B,
fed with OUR engine's hidden states (layers 5, 19, 33, 47, 61) and greedy continuation.

    python bench/dflash_accept.py --model <packed> --draft /workspace/models/Qwen3.8-27B-DFlash2 [--new 256]

Per family: the target generates `new` greedy tokens (eager, recording the residual
stream after every layer); then at every position p the draft, holding the context
features of positions < p in its cache, drafts 7 tokens for the block [t_p, mask x 7]
in one forward, and the leading matches against the true continuation are counted.
Reports accepted tokens per verify step for K = 7 and for K = 3, 4 (using only the
first drafts), plus the draft forward's eager cost.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/dflash")
from mtp_accept import PROMPTS  # noqa: E402
from tokenrush.model import Engine  # noqa: E402
from tokenrush.weights import load_packed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--new", type=int, default=256)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from dflash.model import DFlash2DraftModel, _make_cache, _crop_to
    tok = AutoTokenizer.from_pretrained(a.model)
    cfg, w, _ = load_packed(a.model)
    eng = Engine(cfg, w, max_len=4096)
    draft = DFlash2DraftModel.from_pretrained(a.draft, dtype=torch.bfloat16).cuda().eval()
    layer_ids = list(draft.target_layer_ids)
    B = draft.block_size
    mask_id = draft.mask_token_id
    print(f"draft: {sum(p.numel() for p in draft.parameters()) / 1e9:.2f}B params, block {B}, target layers {layer_ids}")
    head = lambda h: w.lm_head(h.reshape(-1, h.shape[-1])).view(*h.shape[:-1], -1)

    def features(trace):
        """trace[i+1] = residual after layer i (trace[0] = embedding) -> [T, 5*hidden]."""
        return torch.cat([trace[l + 1] for l in layer_ids], dim=-1)

    for name, text in PROMPTS.items():
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False))
        T = len(ids)
        x = torch.tensor(ids, device="cuda")
        # 1. target greedy with per-layer residuals at every position
        eng.reset(); eng.trace = []
        logits = eng.forward(x, all_logits=True)
        feats = [features(eng.trace)]                                   # [T, 25600]
        toks = list(ids)
        nxt = int(logits[-1].argmax())
        for _ in range(a.new):
            toks.append(nxt)
            eng.trace = []
            logits = eng.decode(torch.tensor([nxt], device="cuda"))
            feats.append(features(eng.trace))
            nxt = int(logits[-1].argmax())
        toks.append(nxt)
        eng.trace = None
        F = torch.cat(feats)[None]                                      # [1, T+new, 25600]
        L = len(toks) - 1
        # 2. draft, teacher-forced one position at a time
        cache = _make_cache(draft.config)
        pos_all = torch.arange(L + B + 1, device="cuda")[None]
        acc = torch.zeros(B)                                            # acc[k] = #steps with >= k leading matches
        n, t_draft = 0, 0.0
        with torch.no_grad():
            ctx_start = 0
            for p in range(T, L - B):
                th = F[:, ctx_start:p]                                  # new context rows since last step
                block = torch.full((1, B), mask_id, dtype=torch.long, device="cuda")
                block[0, 0] = toks[p]
                noise = torch.nn.functional.embedding(block, w.embed)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                hid = draft(target_hidden=th, noise_embedding=noise, position_ids=pos_all[:, ctx_start:p + B],
                            past_key_values=cache, use_cache=True)[:, 1 - B:, :]
                _crop_to(cache, p)
                drafts, _, _ = draft.propose(hid, block[:, 0], head, 0.0)
                torch.cuda.synchronize(); t_draft += time.perf_counter() - t0
                d = drafts[0].tolist()
                k = 0
                while k < B - 1 and d[k] == toks[p + 1 + k]:
                    k += 1
                acc[:k + 1] += 1                                        # >=0 .. >=k matches
                n += 1
                ctx_start = p
        rate = acc / n
        per_k = {K: 1 + rate[1:K + 1].sum().item() for K in (3, 4, 7)}
        print(f"\n{name}: {n} positions; draft forward {t_draft / n * 1e3:.1f} ms (eager bf16)")
        print("  P(>= k leading drafts correct): " + " ".join(f"k={k}:{rate[k]:.2f}" for k in range(1, B)))
        print("  accepted tokens per verify step: " + " ".join(f"K={K}: {v:.2f}" for K, v in per_k.items()))


if __name__ == "__main__":
    main()
