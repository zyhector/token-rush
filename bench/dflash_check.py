#!/usr/bin/env python3
"""Our DFlash2 forward against z-lab's reference: same weights, same inputs.
(1) random context features and block on a fresh cache, then a second step with new
context rows: draft tokens must be identical and the block hidden close;
(2) the essay acceptance measurement through our forward (compare with
bench/dflash_accept.py's 3.57 at K=7).

    python bench/dflash_check.py --model <packed> --draft /workspace/models/Qwen3.8-27B-DFlash2
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
from tokenrush.dflash import DFlashDraft, load_dflash  # noqa: E402
from tokenrush.model import Engine  # noqa: E402
from tokenrush.weights import load_packed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--int4", action="store_true")
    a = ap.parse_args()
    from dflash.model import DFlash2DraftModel, _make_cache, _crop_to
    cfg, w, _ = load_packed(a.model)
    ref = DFlash2DraftModel.from_pretrained(a.draft, dtype=torch.bfloat16).cuda().eval()
    ours = DFlashDraft(load_dflash(a.draft, int4=a.int4), w.embed, w.lm_head, cfg.hidden, 4096)
    B = ours.w.block_size
    head = lambda h: w.lm_head(h.reshape(-1, h.shape[-1])).view(*h.shape[:-1], -1)
    torch.manual_seed(0)
    rel = lambda x, y: ((x.float() - y.float()).norm() / y.float().norm()).item()

    # (1) two steps on random inputs
    cache = _make_cache(ref.config)
    ours.reset()
    ctx0, ctx1 = 60, 3
    feats = (torch.randn(ctx0 + ctx1, 5 * cfg.hidden, device="cuda") * 0.5).bfloat16()
    pos_all = torch.arange(ctx0 + ctx1 + B + 4, device="cuda")[None]
    start = 0
    with torch.no_grad():
        for step, (s, e) in enumerate(((0, ctx0), (ctx0, ctx0 + ctx1))):
            p = e
            block = torch.full((B,), ours.w.mask_token_id, dtype=torch.long, device="cuda")
            block[0] = 12345 + step
            noise = torch.nn.functional.embedding(block[None], w.embed)
            hid_ref = ref(target_hidden=feats[None, s:e], noise_embedding=noise, position_ids=pos_all[:, s:p + B],
                          past_key_values=cache, use_cache=True)
            _crop_to(cache, p)
            tok_ref, _, _ = ref.propose(hid_ref[:, 1 - B:], block[None, 0], head, 0.0)
            tok_ours, hn_ours = ours.forward(feats[s:e], block)
            print(f"step {step}: hidden rel err {rel(hn_ours, hid_ref[0]):.2e}; drafts ref {tok_ref[0].tolist()} ours {tok_ours.tolist()} "
                  f"{'MATCH' if tok_ref[0].tolist() == tok_ours.tolist() else 'DIFF'}")

    # (2) essay acceptance through our forward
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    eng = Engine(cfg, w, max_len=4096)
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": PROMPTS["essay"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False))
    T = len(ids); x = torch.tensor(ids, device="cuda")
    eng.reset(); eng.trace = []
    logits = eng.forward(x, all_logits=True)
    feats = [torch.cat([eng.trace[l + 1] for l in ours.w.target_layer_ids], -1)]
    toks = list(ids); nxt = int(logits[-1].argmax())
    for _ in range(256):
        toks.append(nxt); eng.trace = []
        logits = eng.decode(torch.tensor([nxt], device="cuda"))
        feats.append(torch.cat([eng.trace[l + 1] for l in ours.w.target_layer_ids], -1))
        nxt = int(logits[-1].argmax())
    toks.append(nxt); eng.trace = None
    Fe = torch.cat(feats); L = len(toks) - 1
    ours.reset(); acc = torch.zeros(B); n = 0; t_d = 0.0; ctx_start = 0
    for p in range(T, L - B):
        block = torch.full((B,), ours.w.mask_token_id, dtype=torch.long, device="cuda"); block[0] = toks[p]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        d, _ = ours.forward(Fe[ctx_start:p], block)
        torch.cuda.synchronize(); t_d += time.perf_counter() - t0
        d = d.tolist(); k = 0
        while k < B - 1 and d[k] == toks[p + 1 + k]:
            k += 1
        acc[:k + 1] += 1; n += 1; ctx_start = p
    rate = acc / n
    print(f"essay through our forward ({'int4' if a.int4 else 'bf16'}): accepted/step K=7 {1 + rate[1:].sum():.2f} "
          f"(reference measurement 3.57), P(k=1) {rate[1]:.2f}; draft forward {t_d / n * 1e3:.1f} ms eager")


if __name__ == "__main__":
    main()
