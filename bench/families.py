#!/usr/bin/env python3
"""Effective tok/s of the speculative decoder on six prompt families (English
essay / code / math, Chinese essay / math, mixed) with either draft.

    python bench/families.py --model <packed> [--draft dflash,mtp] [--new 300] [--backend marlin|triton]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft_vocab_eval import ZH  # noqa: E402
from mtp_accept import PROMPTS  # noqa: E402
from tokenrush.model import Engine  # noqa: E402
from tokenrush.mtp import MTPHead, build_mtp  # noqa: E402
from tokenrush.quant import DEFAULT_BACKEND  # noqa: E402
from tokenrush.spec import generate_dflash, generate_spec_graph  # noqa: E402
from tokenrush.weights import load_packed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dflash-path", default="/workspace/models/Qwen3.8-27B-DFlash2")
    ap.add_argument("--draft", default="dflash,mtp", help="comma list of dflash, mtp")
    ap.add_argument("--new", type=int, default=300)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    drafts = a.draft.split(",")
    backend = a.backend or DEFAULT_BACKEND
    cfg, w, mtp_t = load_packed(a.model, with_mtp="mtp" in drafts, backend=backend)
    prompts = dict(PROMPTS); prompts.update(ZH)
    stop = set(cfg.eos_ids) | {tok.eos_token_id}
    dv = torch.arange(131072)
    ids_by_prompt = {pn: tok.encode(tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                                            add_generation_prompt=True, enable_thinking=False))
                     for pn, t in prompts.items()}
    print(f"backend {backend}, {a.new} new tokens, 128k draft vocab, T={a.temperature} top-p={a.top_p}")
    print("draft   | " + " | ".join(f"{pn:>14}" for pn in prompts) + "   (tok/s, accepted/step, ms/step)")
    for dname in drafts:
        if dname == "dflash":
            from tokenrush.dflash import DFlashDraft, load_dflash
            eng = Engine(cfg, w, max_len=a.max_len, max_spec=7); eng.capture()
            draft = DFlashDraft(load_dflash(a.dflash_path, int4=True), w.embed, w.lm_head, cfg.hidden, a.max_len)
            eng.attach_dflash(draft, draft_vocab=dv)
            eng.capture_spec_dflash()
            run = lambda ids: generate_dflash(eng, draft, tok, ids, a.new, stop, stream=False, temperature=a.temperature, top_p=a.top_p, seed=0)
        else:
            eng = Engine(cfg, w, max_len=a.max_len, max_spec=4); eng.capture()
            draft = MTPHead(cfg, build_mtp(cfg, mtp_t, "cuda", int4=True), w.embed, w.lm_head, a.max_len)
            eng.attach_mtp(draft, draft_vocab=dv)
            for k in (3, 4):
                eng.capture_spec(k)
            run = lambda ids: generate_spec_graph(eng, draft, tok, ids, a.new, stop, stream=False, dynamic=(3, 4), temperature=a.temperature, top_p=a.top_p, seed=0)
        cells = []
        for pn, ids in ids_by_prompt.items():
            out, st = run(ids)
            ms = 1000.0 * st["accepted_per_step"] / st["decode_tok_s"]
            cells.append(f"{st['decode_tok_s']:4.0f} {st['accepted_per_step']:.2f} {ms:4.1f}")
        print(f"{dname:7s} | " + " | ".join(f"{c:>14}" for c in cells))
        del eng, draft
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
