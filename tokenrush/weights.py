"""Checkpoint handling: read the HF safetensors, keep the text path, drop the
vision tower, set the MTP head aside, quantize the big projections, and write
a packed checkpoint that loads in seconds. Loading builds the weight
containers in `model.py`."""
import glob
import json
import os
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .config import ModelConfig
from .model import AttnWeights, GDNWeights, LayerWeights, ModelWeights
from .quant import GROUP, Linear, QLinear, quantize_int4

HF_PREFIX = "model.language_model."
VISION_PREFIX = "model.visual."
MTP_PREFIX = "mtp."
PACK_META = "tokenrush.json"
COPY_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "chat_template.jinja")

# Projections that get the 4-bit treatment. Everything else stays bf16:
# embeddings (one row read per token), norms, conv, the tiny in_proj_a/b, and
# the MTP head (kept whole, loaded only when asked for).
QUANT_SUFFIXES = ("in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight",
                  "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                  "gate_proj.weight", "up_proj.weight", "down_proj.weight")


def _rename(name: str):
    """HF tensor name -> ours, or None to drop."""
    if name.startswith(VISION_PREFIX):
        return None
    if name.startswith(HF_PREFIX):
        return name[len(HF_PREFIX):]
    return name  # lm_head.weight, mtp.*


def is_quantized(name: str) -> bool:
    if name.startswith(MTP_PREFIX):
        return False
    return name == "lm_head.weight" or name.endswith(QUANT_SUFFIXES)


def iter_hf_tensors(src: str):
    for shard in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                ours = _rename(name)
                if ours is not None:
                    yield ours, f.get_tensor(name)


def pack_checkpoint(src: str, dst: str, group: int = GROUP, shard_bytes: int = 4 << 30, device="cuda"):
    """bf16 HF checkpoint -> packed int4 checkpoint at dst."""
    os.makedirs(dst, exist_ok=True)
    for fn in COPY_FILES:
        if os.path.exists(os.path.join(src, fn)):
            shutil.copy(os.path.join(src, fn), os.path.join(dst, fn))
    buf, size, shard_idx, quantized, total = {}, 0, 0, [], 0
    t0 = time.time()

    def flush():
        nonlocal buf, size, shard_idx
        if buf:
            save_file(buf, os.path.join(dst, f"model-{shard_idx:05d}.safetensors"))
            shard_idx += 1
            buf, size = {}, 0

    for name, t in iter_hf_tensors(src):
        if is_quantized(name):
            q, s, m = quantize_int4(t.to(device), group)
            outs = {name + ".qweight": q, name + ".scale": s, name + ".mn": m}
            quantized.append(name)
        else:
            outs = {name: t}
        for k, v in outs.items():
            v = v.cpu().contiguous()
            buf[k] = v
            size += v.numel() * v.element_size()
            total += v.numel() * v.element_size()
        if size >= shard_bytes:
            flush()
    flush()
    meta = {"format": "int4", "group": group, "quantized": quantized, "shards": shard_idx,
            "packed_bytes": total, "source": os.path.abspath(src)}
    json.dump(meta, open(os.path.join(dst, PACK_META), "w"), indent=1)
    print(f"packed {total / 1e9:.2f} GB into {shard_idx} shards in {time.time() - t0:.0f}s "
          f"({len(quantized)} tensors quantized)")
    return meta


def is_packed(path: str) -> bool:
    return os.path.exists(os.path.join(path, PACK_META))


def _linear(tensors, name, device):
    if name + ".qweight" in tensors:
        return QLinear(*(tensors[name + s].to(device) for s in (".qweight", ".scale", ".mn")))
    return Linear(tensors[name].to(device))


def build_weights(cfg: ModelConfig, tensors: dict, device) -> ModelWeights:
    """tensors: our names -> CPU tensors (bf16, or the packed triple)."""
    dev = lambda n, dt=None: (tensors[n].to(device) if dt is None else tensors[n].to(device=device, dtype=dt))
    layers = []
    for i, lt in enumerate(cfg.layer_types):
        p = f"layers.{i}."
        if lt == "linear_attention":
            m = p + "linear_attn."
            mixer = GDNWeights(
                in_qkv=_linear(tensors, m + "in_proj_qkv.weight", device),
                in_z=_linear(tensors, m + "in_proj_z.weight", device),
                in_b=dev(m + "in_proj_b.weight"), in_a=dev(m + "in_proj_a.weight"),
                conv_w=dev(m + "conv1d.weight").squeeze(1),
                A=-torch.exp(dev(m + "A_log", torch.float32)),
                dt_bias=dev(m + "dt_bias", torch.float32),
                norm_w=dev(m + "norm.weight"),
                out=_linear(tensors, m + "out_proj.weight", device))
        else:
            m = p + "self_attn."
            mixer = AttnWeights(
                q=_linear(tensors, m + "q_proj.weight", device), k=_linear(tensors, m + "k_proj.weight", device),
                v=_linear(tensors, m + "v_proj.weight", device), o=_linear(tensors, m + "o_proj.weight", device),
                q_norm_w=dev(m + "q_norm.weight"), k_norm_w=dev(m + "k_norm.weight"))
        layers.append(LayerWeights(
            ln1=dev(p + "input_layernorm.weight"), ln2=dev(p + "post_attention_layernorm.weight"),
            mixer=mixer, gate=_linear(tensors, p + "mlp.gate_proj.weight", device),
            up=_linear(tensors, p + "mlp.up_proj.weight", device),
            down=_linear(tensors, p + "mlp.down_proj.weight", device)))
    return ModelWeights(embed=dev("embed_tokens.weight"), layers=layers, final_norm=dev("norm.weight"),
                        lm_head=_linear(tensors, "lm_head.weight", device))


def load_packed(path: str, device="cuda", with_mtp: bool = False):
    """-> (cfg, ModelWeights, mtp tensors or None)."""
    cfg = ModelConfig.load(path)
    t0 = time.time()
    tensors, mtp = {}, {}
    for shard in sorted(glob.glob(os.path.join(path, "model-*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name.startswith(MTP_PREFIX):
                    if with_mtp:
                        mtp[name] = f.get_tensor(name)
                else:
                    tensors[name] = f.get_tensor(name)
    w = build_weights(cfg, tensors, device)
    torch.cuda.synchronize()
    print(f"loaded {w.nbytes / 1e9:.2f} GB of weights in {time.time() - t0:.1f}s")
    return cfg, w, (mtp if with_mtp else None)
