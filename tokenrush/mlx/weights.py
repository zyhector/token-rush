"""The packed checkpoint as MLX arrays, with no conversion pass.

Our int4 packing (tokenrush/quant.py: element 2j in the low nibble of byte j,
element 2j+1 in the high nibble) read as little-endian uint32 puts element i of
each 8-element word at bits 4i..4i+3, which is exactly MLX's affine 4-bit layout.
Dequantization is the same formula on both sides, w = q * scale + mn, so
`scale` is MLX's `scales` and `mn` its `biases`. The published file therefore
loads as-is: a uint8 -> uint32 view per matrix, nothing recomputed, the GPTQ
codes and their quality numbers carried over unchanged.
"""
import glob
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Union

import mlx.core as mx

from ..config import ModelConfig

GROUP = 128
BITS = 4
MTP_PREFIX = "mtp."
from ..hub import PACK_META


def _gpu_generation() -> int:
    """applegpu_g17s -> 17. The neural accelerators that kernels.qmm_nax runs on are g17+ (M5)."""
    import re
    m = re.search(r"g(\d+)", mx.device_info().get("architecture", ""))
    return int(m.group(1)) if m else 0


# row counts routed to kernels.qmm_nax; [0, -1] disables it (and is the default before M5)
NAX_ROWS = [5, 8] if _gpu_generation() >= 17 else [0, -1]


@dataclass
class QLinear:
    """int4 g128 affine weight, y = x @ W.T. w: uint32 [out, in/8]; scales, biases: bf16 [out, in/128]."""
    w: mx.array
    scales: mx.array
    biases: mx.array

    def __call__(self, x: mx.array) -> mx.array:
        # MLX's kernels are best up to 4 rows; 5..8 rows (a deep verify step) read the
        # weights once per row there, so they go to our matmul2d kernel (bench/mac/qmv_rows.py)
        if NAX_ROWS[0] <= x.shape[0] <= NAX_ROWS[1] and x.dtype == mx.bfloat16 and self.w.shape[0] % 128 == 0:
            from .kernels import qmm_nax
            return qmm_nax(x, self.w, self.scales, self.biases)
        return mx.quantized_matmul(x, self.w, self.scales, self.biases, transpose=True, group_size=GROUP, bits=BITS)

    @property
    def shape(self):
        return (self.w.shape[0], self.w.shape[1] * 32 // BITS)

    @property
    def nbytes(self):
        return self.w.nbytes + self.scales.nbytes + self.biases.nbytes

    def rows(self, ids: mx.array) -> "QLinear":
        """The sub-matrix of these output rows (a draft vocabulary's lm_head)."""
        return QLinear(self.w[ids], self.scales[ids], self.biases[ids])

    @staticmethod
    def cat(parts):
        return QLinear(*(mx.concatenate([getattr(p, f) for p in parts], axis=0) for f in ("w", "scales", "biases")))


@dataclass
class Linear:
    """bf16 weight, y = x @ W.T."""
    weight: mx.array

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.T

    @property
    def nbytes(self):
        return self.weight.nbytes


Lin = Union[Linear, QLinear]


@dataclass
class GDNWeights:
    in_qkvz: Lin           # hidden -> conv_dim (q|k|v, through the conv) | val_dim (gate z)
    in_ba: mx.array        # [2*HV, hidden] bf16: b rows then a rows
    conv_w: mx.array       # [conv_dim, K] bf16
    A: mx.array            # [HV] fp32, = -exp(A_log)
    dt_bias: mx.array      # [HV] fp32
    norm_w: mx.array       # [gdn_v_dim] bf16
    out: Lin


@dataclass
class AttnWeights:
    qkv: Lin               # hidden -> n_heads*2*head_dim (query|gate per head) | kv (k) | kv (v)
    o: Lin
    q_norm_w: mx.array
    k_norm_w: mx.array


@dataclass
class LayerWeights:
    ln1: mx.array
    ln2: mx.array
    mixer: Union[GDNWeights, AttnWeights]
    gate_up: Lin           # hidden -> 2*ffn: gate rows then up rows
    down: Lin


@dataclass
class ModelWeights:
    embed: mx.array
    layers: list
    final_norm: mx.array
    lm_head: Lin

    def arrays(self):
        """Every array, for mx.eval and byte counts."""
        out = [self.embed, self.final_norm]
        for o in [self.lm_head] + self.layers:
            out += _leaves(o)
        return out

    @property
    def streamed_bytes(self):
        """Bytes one decode step reads: everything but the embedding table (one row per token)."""
        return sum(a.nbytes for a in self.arrays()) - self.embed.nbytes


def _leaves(o):
    if isinstance(o, mx.array):
        return [o]
    if isinstance(o, (list, tuple)):
        return [a for x in o for a in _leaves(x)]
    if hasattr(o, "__dataclass_fields__"):
        return [a for f in o.__dataclass_fields__ for a in _leaves(getattr(o, f))]
    return []


def read_tensors(path: str, with_mtp: bool = False):
    """name -> mx.array for every tensor in the packed checkpoint (lazily mapped by mx.load)."""
    tensors, mtp = {}, {}
    for shard in sorted(glob.glob(os.path.join(path, "model-*.safetensors"))):
        for name, t in mx.load(shard).items():
            if name.startswith(MTP_PREFIX):
                if with_mtp:
                    mtp[name] = t
            else:
                tensors[name] = t
    return tensors, mtp


def linear(tensors: dict, names) -> Lin:
    """One Linear from one or more checkpoint matrices concatenated along the output dim."""
    names = [names] if isinstance(names, str) else names
    if names[0] + ".qweight" in tensors:
        parts = []
        for n in names:
            q = tensors[n + ".qweight"]
            assert q.dtype == mx.uint8 and q.shape[1] % 4 == 0
            parts.append(QLinear(q.view(mx.uint32), tensors[n + ".scale"], tensors[n + ".mn"]))
        return QLinear.cat(parts) if len(parts) > 1 else parts[0]
    return Linear(mx.concatenate([tensors[n] for n in names], axis=0))


def build_layer(cfg: ModelConfig, tensors: dict, i: int) -> LayerWeights:
    p = f"layers.{i}."
    if cfg.layer_types[i] == "linear_attention":
        m = p + "linear_attn."
        mixer = GDNWeights(
            in_qkvz=linear(tensors, [m + "in_proj_qkv.weight", m + "in_proj_z.weight"]),
            in_ba=mx.concatenate([tensors[m + "in_proj_b.weight"], tensors[m + "in_proj_a.weight"]]),
            conv_w=tensors[m + "conv1d.weight"].squeeze(1),
            A=-mx.exp(tensors[m + "A_log"].astype(mx.float32)),
            dt_bias=tensors[m + "dt_bias"].astype(mx.float32),
            norm_w=tensors[m + "norm.weight"],
            out=linear(tensors, m + "out_proj.weight"))
    else:
        m = p + "self_attn."
        mixer = AttnWeights(
            qkv=linear(tensors, [m + "q_proj.weight", m + "k_proj.weight", m + "v_proj.weight"]),
            o=linear(tensors, m + "o_proj.weight"),
            q_norm_w=tensors[m + "q_norm.weight"], k_norm_w=tensors[m + "k_norm.weight"])
    return LayerWeights(
        ln1=tensors[p + "input_layernorm.weight"], ln2=tensors[p + "post_attention_layernorm.weight"],
        mixer=mixer, gate_up=linear(tensors, [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"]),
        down=linear(tensors, p + "mlp.down_proj.weight"))


def build_weights(cfg: ModelConfig, tensors: dict) -> ModelWeights:
    return ModelWeights(embed=tensors["embed_tokens.weight"],
                        layers=[build_layer(cfg, tensors, i) for i in range(cfg.n_layers)],
                        final_norm=tensors["norm.weight"], lm_head=linear(tensors, "lm_head.weight"))


def load_packed(path: str, with_mtp: bool = False):
    """-> (cfg, ModelWeights, mtp tensors or None). Materializes the weights in unified memory."""
    meta = json.load(open(os.path.join(path, PACK_META)))
    assert meta["format"] == "int4" and meta["group"] == GROUP, meta
    cfg = ModelConfig.load(path)
    t0 = time.time()
    tensors, mtp = read_tensors(path, with_mtp)
    w = build_weights(cfg, tensors)
    del tensors                           # the fused concatenations replace the per-matrix arrays
    mx.eval(w.arrays())
    print(f"loaded {sum(a.nbytes for a in w.arrays()) / 1e9:.2f} GB of weights "
          f"({w.streamed_bytes / 1e9:.2f} GB streamed per token) in {time.time() - t0:.1f}s")
    return cfg, w, (mtp if with_mtp else None)
