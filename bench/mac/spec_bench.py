"""Speculative decoding on a Mac: effective tok/s per prompt family for raw decode
(K = 0) and the MTP chain at fixed and dynamic depth, on the CUDA engine's Phase 0
prompts (bench/mtp_accept.py, thinking off). Greedy; reports how long each run's
output stays identical to raw greedy (bf16 near-ties may split them later).

    uv run python bench/mac/spec_bench.py [--tokens 256] [--ks 0,1,2,3,4,dyn]
"""
import argparse
import json
import os

import mlx.core as mx
from transformers import AutoTokenizer

from tokenrush.hub import DEFAULT_REPO, DFLASH_REPO, resolve_model
from tokenrush.mlx.dflash import DFlashDraft, generate_dflash
from tokenrush.mlx.model import Engine
from tokenrush.mlx.spec import MTPHead, generate
from tokenrush.mlx.weights import load_packed


def phase0_prompts():
    """PROMPTS from bench/mtp_accept.py, read without importing it (it imports the CUDA engine)."""
    import ast
    src = open(os.path.join(os.path.dirname(__file__), "..", "mtp_accept.py")).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PROMPTS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise LookupError("PROMPTS not found in bench/mtp_accept.py")


PROMPTS = phase0_prompts()


def draft_vocab(name):
    if not name:
        return None
    import torch
    ids = torch.load(os.path.join(os.path.dirname(__file__), "../../tokenrush/draft_vocab", name + ".pt"))
    return mx.array(ids.cpu().numpy().astype("int32"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_REPO)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--ks", default="0,1,2,3,4,dyn")
    ap.add_argument("--draft-vocab", default="en_64k", help="'' for the full lm_head")
    ap.add_argument("--json")
    a = ap.parse_args()
    path = resolve_model(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    cfg, w, mtp_t = load_packed(path, with_mtp=True)
    eng = Engine(cfg, w, max_len=8192)
    mtp = MTPHead(eng, mtp_t, draft_vocab=draft_vocab(a.draft_vocab))
    dfl = DFlashDraft(resolve_model(DFLASH_REPO), eng, draft_vocab=draft_vocab(a.draft_vocab)) if "dflash" in a.ks else None
    stop = set(cfg.eos_ids) | {tok.convert_tokens_to_ids("<|im_end|>")}
    prompts = {k: tok.encode(tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False))
               for k, t in PROMPTS.items()}
    generate(eng, mtp, prompts["essay"], 32, K=3)                      # warm up
    rows, results = [], {}
    for kspec in a.ks.split(","):
        cells = []
        for fam, ids in prompts.items():
            if kspec == "dflash":
                out, st = generate_dflash(eng, dfl, ids, a.tokens)
            else:
                K, dyn = (4, (1, 6)) if kspec == "dyn" else (int(kspec), None)
                out, st = generate(eng, mtp, ids, a.tokens, stop_ids=(), K=K, dynamic=dyn)
            if kspec == "0":
                results[fam] = out
            same = next((i for i, (x, y) in enumerate(zip(out, results.get(fam, out))) if x != y), len(out))
            cells.append(f"{st['tok_s']:6.1f} ({st['tokens_per_step']:.2f}/step, {st['ms_per_step']:.0f} ms, ={same})")
            rows.append({"K": kspec, "family": fam, **{k: v for k, v in st.items() if k != "depths"}, "same_as_raw": same})
        print(f"K={kspec:<4} " + "  ".join(f"{f}: {c}" for f, c in zip(prompts, cells)), flush=True)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
