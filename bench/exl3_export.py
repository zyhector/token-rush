#!/usr/bin/env python3
"""Dequantize an ExLlamaV3 checkpoint to bf16 safetensors in our tensor names,
with exllamav3's own reconstruction, for the quality table.

Runs in the exl3 venv (docs/baselines.md), not the project's:

    /workspace/venvs/exl3/bin/python bench/exl3_export.py \
        --src /workspace/models/Qwen3.8-27B-exl3-4.0 --dst /workspace/models/Qwen3.8-27B-exl3-4.0-bf16

Only the EXL3-quantized matrices are written (the norms, embeddings and the
small projections it keeps in bf16/fp16 are taken from the HF checkpoint by
bench/quality_sources.py's overlay); export.json records the bytes each
tensor occupies in the EXL3 format, for the bits-per-weight column.
"""
import argparse
import json
import os
import time

import torch
from safetensors.torch import save_file

HF_PREFIX = "model.language_model."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard-bytes", type=int, default=4 << 30)
    ap.add_argument("--limit", type=int, default=None, help="tensors to export (debug)")
    a = ap.parse_args()
    from exllamav3 import Config, Model
    from exllamav3.modules import Linear
    from exllamav3.modules.quant.exl3 import LinearEXL3

    os.makedirs(a.dst, exist_ok=True)
    t0 = time.time()
    config = Config.from_directory(a.src)
    model = Model.from_config(config)
    model.load(a.device, progressbar=False)
    print(f"loaded {a.src} in {time.time() - t0:.0f}s")

    buf, size, shard, n, meta_bytes, total = {}, 0, 0, 0, {}, 0

    def flush():
        nonlocal buf, size, shard
        if buf:
            save_file(buf, os.path.join(a.dst, f"model-{shard:05d}.safetensors"))
            shard += 1
            buf, size = {}, 0

    for module in model:
        if not isinstance(module, Linear) or not isinstance(module.inner, LinearEXL3):
            continue
        key = module.key
        if key.startswith("model.visual") or key.startswith("mtp."):
            continue
        ours = (key[len(HF_PREFIX):] if key.startswith(HF_PREFIX) else key) + ".weight"
        inner = module.inner
        w = inner.get_weight_tensor()                                     # [in, out] fp16
        w = w.t().contiguous().to(torch.bfloat16).cpu()                   # [out, in]
        nbytes = sum(t.numel() * t.element_size() for t in
                     (inner.trellis, inner.suh, inner.svh) if isinstance(t, torch.Tensor))
        meta_bytes[ours] = nbytes
        buf[ours] = w
        size += w.numel() * 2
        total += w.numel()
        n += 1
        if n % 50 == 0:
            print(f"  {n} tensors, {total / 1e9:.2f}B params, {time.time() - t0:.0f}s")
        if size >= a.shard_bytes:
            flush()
        if a.limit and n >= a.limit:
            break
    flush()
    json.dump({"label": "EXL3 4.00bpw", "source": a.src, "bytes": meta_bytes, "tensors": n, "params": total},
              open(os.path.join(a.dst, "export.json"), "w"), indent=1)
    print(f"exported {n} tensors, {total / 1e9:.2f}B params, {sum(meta_bytes.values()) / 1e9:.2f} GB in EXL3 form, "
          f"{shard} shards, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
