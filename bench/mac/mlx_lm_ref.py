"""mlx-lm's Qwen3.5 implementation on our checkpoint's bytes, built in memory.

mlx-lm is the Mac rival and the independent reference for engine correctness:
a second implementation of the same model (its own GDN kernel, its own attention,
its own norms) reading the same int4 codes, so any disagreement beyond bf16
noise is a bug on one side. Because our packing is MLX's affine 4-bit format,
nothing is converted or written to disk: the packed tensors are renamed to HF
names and loaded into mlx-lm's module tree as-is.
"""
import json
import os

import mlx.core as mx
import mlx.nn as nn

from tokenrush.mlx.weights import read_tensors


def hf_name(ours: str) -> str:
    if ours.startswith("lm_head."):
        return "language_model." + ours
    return "language_model.model." + ours


def mlx_lm_weights(path: str) -> dict:
    tensors, _ = read_tensors(path)
    out = {}
    for name, t in tensors.items():
        if name.endswith(".qweight"):
            base = name[:-len(".qweight")]                      # ...proj.weight
            out[hf_name(base)] = t.view(mx.uint32)
        elif name.endswith(".scale"):
            out[hf_name(name[:-len(".weight.scale")] + ".scales")] = t
        elif name.endswith(".mn"):
            out[hf_name(name[:-len(".weight.mn")] + ".biases")] = t
        else:
            out[hf_name(name)] = t
    return out


def load_mlx_lm(path: str):
    """-> (mlx-lm Model with our weights, its make_cache)."""
    from mlx_lm.models import qwen3_5
    config = json.load(open(os.path.join(path, "config.json")))
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(config))
    weights = model.sanitize(mlx_lm_weights(path))             # conv1d layout, norm +1
    nn.quantize(model, group_size=128, bits=4,
                class_predicate=lambda p, m: isinstance(m, nn.Linear) and f"{p}.scales" in weights)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model
