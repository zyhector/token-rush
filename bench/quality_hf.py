#!/usr/bin/env python3
"""The HF transformers side of the quality table: the downstream task, and the
noise floor of the yardstick.

    # noise floor: HF's own bf16 logits against our bf16 reference on a few chunks
    python bench/quality_hf.py --logits-check --ref /workspace/ref/logits --chunks-per-corpus 1
    # GSM8K, 200 problems, greedy, for the bf16 model or a candidate's weights overlaid on it
    python bench/quality_hf.py --gsm8k --out results/quality/gsm8k_bf16.json
    python bench/quality_hf.py --gsm8k --src packed:/workspace/models/Qwen3.8-27B-int4g128 --out results/quality/gsm8k_int4_rtn.json

The model is Qwen3_5ForCausalLM in bf16 across both GPUs with `fla` blocked
(its fused GDN decode kernel is miscompiled on sm_120, docs/environment.md).
A candidate is applied by copying its dequantized matrices over the HF
parameters, so every candidate runs the identical forward.
"""
import argparse
import json
import os
import re
import sys
import time

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF_DIR = "/workspace/models/Qwen3.8-27B"


def load_model(hf_dir, mem0="30GiB", mem1="26GiB"):
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    tok = AutoTokenizer.from_pretrained(hf_dir)
    t0 = time.time()
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        hf_dir, dtype=torch.bfloat16, device_map="auto", max_memory={0: mem0, 1: mem1, "cpu": "200GiB"},
        output_loading_info=True)
    model.eval()
    devs = sorted({str(p.device) for p in model.parameters()})
    print(f"loaded in {time.time() - t0:.0f}s on {devs}; missing {len(info['missing_keys'])} "
          f"unexpected {len(info['unexpected_keys'])} (mtp/visual expected)")
    assert "cpu" not in devs, "part of the model landed on the CPU: raise the memory caps"
    return tok, model


def overlay(model, src):
    """Copy the source's tensors over the HF parameters that carry the same weight."""
    params = dict(model.named_parameters())
    n = 0
    with torch.no_grad():
        for ours in list(src.names()):
            for cand in ("model." + ours, ours):
                if cand in params:
                    p = params[cand]
                    t = src[ours]
                    assert tuple(t.shape) == tuple(p.shape), (ours, t.shape, p.shape)
                    p.copy_(t.to(p.device, p.dtype))
                    n += 1
                    break
            else:
                if not ours.startswith("mtp."):
                    print("  no HF parameter for", ours)
    print(f"overlaid {n} tensors from {src.label}")
    return n


# ------------------------------------------------------------ logits check


def logits_check(a, tok, model):
    from bench.quality_logits import chunk_metrics, summarize
    from tokenrush.corpus import load_ids
    chunks, _ = load_ids(a.chunks)
    dev0 = model.get_input_embeddings().weight.device
    out = {}
    for corpus in a.corpora.split(","):
        per = [[], [], [], []]
        for ci in range(min(a.chunks_per_corpus, chunks[corpus].shape[0])):
            ids = chunks[corpus][ci]
            t0 = time.time()
            with torch.no_grad():
                logits = model(input_ids=ids[None].to(dev0), use_cache=False).logits[0]
            torch.cuda.synchronize()
            ref = torch.load(os.path.join(a.ref, f"{corpus}_{ci:02d}.pt")).to(logits.device)
            m = chunk_metrics(ref, logits.to(torch.bfloat16), ids.to(logits.device))
            s = summarize(*m)
            print(f"  {corpus} chunk {ci}: HF fwd {time.time() - t0:.0f}s  ppl ours {s['ppl_ref']:.4f} HF {s['ppl']:.4f}  "
                  f"KL(ours||HF) mean {s['kl_mean']:.2e} p99 {s['kl_p99']:.2e} max {s['kl_max']:.2e}  top1 {s['top1_agreement']:.4f}")
            for p, x in zip(per, m):
                p.append(x.cpu())
            del logits, ref
        out[corpus] = summarize(*[torch.cat(p) for p in per])
    if a.out:
        json.dump({"what": "HF bf16 vs our bf16 reference (the yardstick's noise floor)", "per_corpus": out},
                  open(a.out, "w"), indent=1)
        print("saved", a.out)


# ------------------------------------------------------------ GSM8K

INSTRUCTION = "Solve the following math problem step by step. Put the final numeric answer within \\boxed{}.\n\n"
_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def extract(text):
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    cands = re.findall(_NUM, boxed[-1]) if boxed else []
    if not cands:
        cands = re.findall(_NUM, text)
    if not cands:
        return None
    return cands[-1].replace(",", "").rstrip(".")


def same_number(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def gsm8k(a, tok, model):
    from datasets import load_dataset
    rows = load_dataset("openai/gsm8k", "main", split="test").select(range(a.n))
    dev0 = model.get_input_embeddings().weight.device
    tok.padding_side = "left"
    prompts = [tok.apply_chat_template([{"role": "user", "content": INSTRUCTION + r["question"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for r in rows]
    golds = [r["answer"].split("####")[-1].strip().replace(",", "") for r in rows]
    records, correct, n_tok = [], 0, 0
    t0 = time.time()
    for s in range(0, len(prompts), a.batch):
        batch = prompts[s:s + a.batch]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(dev0)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, pad_token_id=tok.pad_token_id)
        for j, g in enumerate(gen):
            new = g[enc["input_ids"].shape[1]:]
            new = new[: (new == tok.eos_token_id).nonzero()[0, 0] + 1] if (new == tok.eos_token_id).any() else new
            text = tok.decode(new, skip_special_tokens=True)
            pred = extract(text)
            ok = same_number(pred, golds[s + j])
            correct += ok
            n_tok += int(new.numel())
            records.append({"i": s + j, "gold": golds[s + j], "pred": pred, "correct": bool(ok),
                            "tokens": int(new.numel()), "text": text})
        done = s + len(batch)
        print(f"  {done}/{len(prompts)}: acc {correct / done:.3f}  {n_tok / (time.time() - t0):.1f} tok/s  "
              f"{(time.time() - t0) / done:.1f} s/problem", flush=True)
    acc = correct / len(prompts)
    print(f"GSM8K {len(prompts)} problems: {correct} correct = {acc:.3f}; "
          f"{sum(r['pred'] is None for r in records)} unparsed; {time.time() - t0:.0f}s")
    if a.out:
        json.dump({"src": a.src or "bf16", "n": len(prompts), "correct": correct, "accuracy": acc,
                   "max_new": a.max_new, "batch": a.batch, "instruction": INSTRUCTION, "records": records},
                  open(a.out, "w"), indent=1)
        print("saved", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default=HF_DIR)
    ap.add_argument("--src", help="candidate tensor source to overlay (bench/quality_sources.py); default bf16")
    ap.add_argument("--logits-check", action="store_true")
    ap.add_argument("--ref", default="/workspace/ref/logits")
    ap.add_argument("--chunks", default="data/quality/chunks.npz")
    ap.add_argument("--chunks-per-corpus", type=int, default=1)
    ap.add_argument("--corpora", default="wikitext2,code,math")
    ap.add_argument("--gsm8k", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--mem0", default="30GiB")
    ap.add_argument("--mem1", default="26GiB")
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.modules["fla"] = None              # HF must not reach fla's fused GDN kernel (miscompiled on sm_120)
    sys.modules["causal_conv1d"] = None
    tok, model = load_model(a.hf, a.mem0, a.mem1)
    if a.src:
        from bench.quality_sources import open_source
        src = open_source(a.src, a.hf, device="cuda:1")
        overlay(model, src)
        del src
        torch.cuda.empty_cache()
    if a.logits_check:
        logits_check(a, tok, model)
    if a.gsm8k:
        gsm8k(a, tok, model)


if __name__ == "__main__":
    main()
