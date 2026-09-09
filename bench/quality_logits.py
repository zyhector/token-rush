#!/usr/bin/env python3
"""Quantization quality: KL to bf16, perplexity, top-1 agreement, through our
own bf16 forward for every candidate (docs/quality_plan.md, Part 1).

    # 1. the reference: bf16 logits of every chunk, once
    python bench/quality_logits.py --src hf:/workspace/models/Qwen3.8-27B --save-ref /workspace/ref/logits
    # 2. each candidate against it
    python bench/quality_logits.py --src packed:/workspace/models/Qwen3.8-27B-int4g128 --ref /workspace/ref/logits \
        --out results/quality/int4_rtn.json

The candidate's weights come from a tensor source (bench/quality_sources.py):
its quantized matrices dequantized to bf16, everything else bf16 from the HF
checkpoint. The forward is the engine's eager prefill path (fla chunk kernel,
SDPA), the one the engine-correctness gate verified against HF, with each
layer built from the source on demand (55.6 GB of bf16 does not fit a card).
Chunks are independent: the state is reset before each one.

Per position i of a chunk: KL(bf16 || candidate) over the full vocabulary in
fp32, top-1 agreement, and the NLL of token i+1 (perplexity). Reported per
corpus and pooled.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenrush.config import ModelConfig  # noqa: E402
from tokenrush.model import Engine, ModelWeights  # noqa: E402
from tokenrush.quant import Linear  # noqa: E402
from tokenrush.weights import build_layer  # noqa: E402
from tokenrush.corpus import load_ids  # noqa: E402
from bench.quality_sources import open_source  # noqa: E402

HF_DIR = "/workspace/models/Qwen3.8-27B"


class SourceEngine(Engine):
    """StreamingBF16Engine over an arbitrary tensor source: each layer's bf16
    weights are built from the source when the forward reaches it."""

    def __init__(self, cfg, src, max_len, device="cuda"):
        self.src = src
        w = ModelWeights(embed=src["embed_tokens.weight"].to(device), layers=[None] * cfg.n_layers,
                         final_norm=src["norm.weight"].to(device), lm_head=Linear(src["lm_head.weight"].to(device)))
        super().__init__(cfg, w, max_len, device)

    def layer(self, li):
        return build_layer(self.cfg, self.src, li, self.device, backend=None)


def chunk_metrics(ref, out, ids, rows=512):
    """ref, out: [T, V] bf16 logits; ids: [T]. -> per-position KL, top-1 agreement, NLL (ref, out)."""
    T = ref.shape[0]
    kl, agree, nll_r, nll_o = [], [], [], []
    for s in range(0, T, rows):
        r = F.log_softmax(ref[s:s + rows].float(), -1)
        o = F.log_softmax(out[s:s + rows].float(), -1)
        kl.append((r.exp() * (r - o)).sum(-1))
        agree.append(r.argmax(-1) == o.argmax(-1))
        e = min(s + rows, T - 1)                                   # positions with a next token
        if s < e:
            tgt = ids[s + 1:e + 1]
            nll_r.append(-r[:e - s].gather(1, tgt[:, None])[:, 0])
            nll_o.append(-o[:e - s].gather(1, tgt[:, None])[:, 0])
    return torch.cat(kl), torch.cat(agree), torch.cat(nll_r), torch.cat(nll_o)


def summarize(kl, agree, nll_r, nll_o):
    kl = kl.float()
    return {"positions": int(kl.numel()),
            "kl_mean": kl.mean().item(), "kl_median": kl.median().item(),
            "kl_p99": kl.quantile(0.99).item(), "kl_max": kl.max().item(),
            "top1_agreement": agree.float().mean().item(),
            "ppl_ref": nll_r.float().mean().exp().item(), "ppl": nll_o.float().mean().exp().item(),
            "ppl_delta": (nll_o.float().mean().exp() - nll_r.float().mean().exp()).item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="tensor source spec, see bench/quality_sources.py")
    ap.add_argument("--hf", default=HF_DIR)
    ap.add_argument("--chunks", default="data/quality/chunks.npz")
    ap.add_argument("--save-ref", help="write this run's logits here (the bf16 reference)")
    ap.add_argument("--ref", help="compare against the logits saved here")
    ap.add_argument("--out", help="json with the metrics")
    ap.add_argument("--corpora", default="wikitext2,code,math")
    ap.add_argument("--limit", type=int, default=None, help="chunks per corpus (debug)")
    ap.add_argument("--head-from", default=None,
                    help="tensor source spec whose lm_head.weight replaces the candidate's (mixing a rival's body "
                         "with our head)")
    ap.add_argument("--keep-bf16", default=None,
                    help="regex over our tensor names: matching tensors come from the bf16 checkpoint instead of the "
                         "candidate (attribution runs, e.g. 'lm_head' or 'layers\\.(0|1|2)\\.')")
    a = ap.parse_args()
    assert bool(a.save_ref) != bool(a.ref), "one of --save-ref / --ref"

    chunks, chunk_meta = load_ids(a.chunks)
    T = chunk_meta["chunk"]
    cfg = ModelConfig.load(a.hf)
    t0 = time.time()
    src = open_source(a.src, a.hf)
    if a.head_from:
        from bench.quality_sources import Overlay
        src = Overlay(src, open_source(a.head_from, a.hf), ["lm_head.weight"])
        print(f"lm_head from {a.head_from}")
    if a.keep_bf16:
        from bench.quality_sources import KeepBF16
        src = KeepBF16(src, a.keep_bf16, a.hf)
        print(f"keeping {len(src.kept)} tensors in bf16 (pattern {a.keep_bf16!r})")
    bpw, params = src.streamed_bpw(a.hf)
    print(f"source {a.src}: {src.label}, {bpw:.3f} bits/weight over the {params / 1e9:.2f}B streamed weights "
          f"({time.time() - t0:.0f}s to open)")
    eng = SourceEngine(cfg, src, max_len=T)
    if a.save_ref:
        os.makedirs(a.save_ref, exist_ok=True)

    results = {"src": a.src, "label": src.label, "streamed_bpw": bpw, "streamed_params": params, "per_corpus": {}}
    if hasattr(src, "types"):
        from collections import Counter
        results["tensor_types"] = dict(Counter(src.types.values()))
    pooled = [[], [], [], []]
    for corpus in a.corpora.split(","):
        ids_all = chunks[corpus]
        n = ids_all.shape[0] if a.limit is None else min(a.limit, ids_all.shape[0])
        per = [[], [], [], []]
        for ci in range(n):
            ids = ids_all[ci].cuda()
            eng.reset()
            t0 = time.time()
            logits = eng.forward(ids, all_logits=True)                      # [T, V] bf16
            torch.cuda.synchronize()
            t_fwd = time.time() - t0
            if a.save_ref:
                torch.save(logits.cpu(), os.path.join(a.save_ref, f"{corpus}_{ci:02d}.pt"))
                ref = logits
            else:
                ref = torch.load(os.path.join(a.ref, f"{corpus}_{ci:02d}.pt")).cuda()
            m = chunk_metrics(ref, logits, ids)
            for p, x in zip(per, m):
                p.append(x.cpu())
            s = summarize(*m)
            print(f"  {corpus} chunk {ci}: fwd {t_fwd:.1f}s  ppl_ref {s['ppl_ref']:.3f} ppl {s['ppl']:.3f}  "
                  f"KL mean {s['kl_mean']:.2e} p99 {s['kl_p99']:.2e}  top1 {s['top1_agreement']:.4f}")
            del logits, ref
        cat = [torch.cat(p) for p in per]
        for pl, c in zip(pooled, cat):
            pl.append(c)
        results["per_corpus"][corpus] = summarize(*cat)
        r = results["per_corpus"][corpus]
        print(f"{corpus}: {r['positions']} positions  ppl {r['ppl_ref']:.3f} -> {r['ppl']:.3f} ({r['ppl_delta']:+.3f})  "
              f"KL mean {r['kl_mean']:.2e} median {r['kl_median']:.2e} p99 {r['kl_p99']:.2e} max {r['kl_max']:.2e}  "
              f"top1 {r['top1_agreement']:.4f}")
    results["pooled"] = summarize(*[torch.cat(p) for p in pooled])
    r = results["pooled"]
    print(f"pooled: {r['positions']} positions  KL mean {r['kl_mean']:.2e} p99 {r['kl_p99']:.2e}  top1 {r['top1_agreement']:.4f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
