"""Write our int4 checkpoint as a standard mlx-lm directory, for the rivals that load
from disk (vllm-metal, SGLang's MLX backend). The codes, scales and minimums are
written as they are (MLX affine 4-bit, group 128, is our packing byte for byte), so
every engine compared reads exactly the same weights. The MTP head is dropped, as
mlx-lm drops it.

    uv run python bench/mac/export_mlx_lm.py --out /path/to/Qwen3.8-27B-tokenrush-mlx
"""
import argparse
import json
import os
import shutil

import mlx.core as mx

from mlx_lm_ref import mlx_lm_weights
from tokenrush.hub import COPY_FILES, DEFAULT_REPO, resolve_model

SHARD = 4 << 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_REPO)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = resolve_model(a.model)
    os.makedirs(a.out, exist_ok=True)
    for fn in COPY_FILES:
        if os.path.exists(os.path.join(src, fn)):
            shutil.copy(os.path.join(src, fn), os.path.join(a.out, fn))
    cfg = json.load(open(os.path.join(src, "config.json")))
    q = {"group_size": 128, "bits": 4, "mode": "affine"}
    cfg["quantization"] = q
    cfg["quantization_config"] = q
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)

    w = mlx_lm_weights(src)
    shards, cur, size = [], {}, 0
    for k in sorted(w):
        cur[k] = w[k]
        size += w[k].nbytes
        if size >= SHARD:
            shards.append(cur)
            cur, size = {}, 0
    if cur:
        shards.append(cur)
    index = {"metadata": {"total_size": sum(t.nbytes for t in w.values())}, "weight_map": {}}
    for i, sh in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(os.path.join(a.out, name), sh, metadata={"format": "mlx"})
        index["weight_map"].update({k: name for k in sh})
    json.dump(index, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    print(f"wrote {index['metadata']['total_size'] / 1e9:.2f} GB in {len(shards)} shards to {a.out}")


if __name__ == "__main__":
    main()
