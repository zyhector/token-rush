#!/usr/bin/env python3
"""Probe one layer's linears: how much output error each candidate's weights make
on real text, and (for AWQ-smoothed checkpoints) what the smoothing did.

    python bench/quality_layer_probe.py --sources packed:/workspace/models/Qwen3.8-27B-int4g128-gptq-mse,ct:/workspace/models/Qwen3.8-27B-RedHatAI-INT4

This is the tool behind the layer-0 comparison in `docs/quantization.md`
(RTN 0.032 / RedHatAI 0.024 / our GPTQ 0.018 on `in_proj_qkv`). It answers a
question the KL table cannot: *where* a candidate's error is, per matrix,
in the space that matters (`W x`, not `W`).

Restricted to layer 0, and within it to the two matrices whose input really is
`rmsnorm(embed[ids])` — `in_proj_qkv` and `in_proj_z`. That input is
computable without a forward pass, so the probe is exact and costs seconds.
`out_proj` reads the GDN output and the MLP matrices read the post-attention
norm of the residual, so feeding them the mixer input measures nothing about
them; they are available under `--proxy` and labelled as such. Any deeper
layer needs the residual stream carried through the candidate's own quantized
layers, which is what `quality_logits.py` measures end to end anyway.

For a checkpoint whose calibration folded per-channel scales into the layer
norms (AWQ smoothing: llm-compressor, RedHatAI), `--undo-smoothing` divides
the smoothing back out before comparing weights, using that checkpoint's own
`input_layernorm`. Qwen's RMSNorm gain is `1 + w`, so the factor is
`(1 + w_hf) / (1 + w_candidate)`.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenrush import ops  # noqa: E402
from tokenrush.config import ModelConfig  # noqa: E402
from tokenrush.corpus import load_ids  # noqa: E402
from bench.quality_sources import HFSource, open_source  # noqa: E402

# Only these two take rmsnorm(embed[ids]) as their input, so only these two are
# exact here. out_proj reads the GDN output and the MLP matrices read the
# post-attention norm of the residual; both need layer 0's mixer to be run, so
# they are available with --proxy but are a proxy, not a measurement.
MATS = ["linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight"]
PROXY_MATS = ["linear_attn.out_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="/workspace/models/Qwen3.8-27B")
    ap.add_argument("--sources", required=True, help="comma-separated source specs (bench/quality_sources.py)")
    ap.add_argument("--chunks", default="data/quality/chunks.npz")
    ap.add_argument("--eval-chunks", type=int, default=4, help="held-out wikitext chunks for the input")
    ap.add_argument("--mats", default=None, help="comma-separated suffixes (default: the exact two)")
    ap.add_argument("--proxy", action="store_true", help="also probe out_proj and the MLP with the mixer input (a proxy)")
    ap.add_argument("--undo-smoothing", action="store_true")
    a = ap.parse_args()

    cfg = ModelConfig.load(a.hf)
    base = HFSource(a.hf)
    chunks, _ = load_ids(a.chunks)
    ids = chunks["wikitext2"][:a.eval_chunks].cuda().reshape(-1)
    embed = base["embed_tokens.weight"].cuda()
    ln = base["layers.0.input_layernorm.weight"].cuda()
    X = ops.rmsnorm(embed[ids], ln, cfg.eps).float()          # the mixer's real input
    del embed
    print(f"layer 0 mixer input: {tuple(X.shape)} from {a.eval_chunks} held-out WikiText-2 chunks")
    mats = a.mats.split(",") if a.mats else MATS + (PROXY_MATS if a.proxy else [])
    srcs = [(spec, open_source(spec, a.hf)) for spec in a.sources.split(",")]

    for name in mats:
        full = "layers.0." + name
        W = base[full].cuda().float()
        ref = X @ W.T
        print(f"\n{name}  {tuple(W.shape)}" + ("   (PROXY: this matrix does not read the mixer input)" if name in PROXY_MATS else ""))
        for spec, src in srcs:
            if full not in src:
                print(f"  {spec:55s} missing")
                continue
            w = src[full].cuda().float()
            note = ""
            norm_name = "layers.0.post_attention_layernorm.weight" if name.startswith("mlp") else "layers.0.input_layernorm.weight"
            if a.undo_smoothing and norm_name in src.names():
                ref_norm = base[norm_name].cuda().float()
                smooth = (1 + ref_norm) / (1 + src[norm_name].cuda().float())
                if (smooth - 1).abs().max() > 1e-3:
                    w = w / smooth[None, :]
                    note = f"   [smoothing undone, factor {smooth.min():.3f}..{smooth.max():.3f}]"
            werr = ((w - W).norm() / W.norm()).item()
            oerr = ((X @ w.T - ref).norm() / ref.norm()).item()
            print(f"  {spec:55s} weight {werr:.4f}  output {oerr:.4f}{note}")


if __name__ == "__main__":
    main()
