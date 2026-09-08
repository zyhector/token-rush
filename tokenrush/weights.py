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
from .quant import DEFAULT_BACKEND, GROUP, Linear, QLinear, quantize_int4

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


def _linear(tensors, names, device, backend):
    """One Linear from one or more checkpoint matrices concatenated along the output dim."""
    names = [names] if isinstance(names, str) else names
    if names[0] + ".qweight" in tensors:
        parts = [QLinear(*(tensors[n + s].to(device) for s in (".qweight", ".scale", ".mn")), backend="dequant")
                 for n in names]
        return QLinear.cat(parts, backend) if len(parts) > 1 else QLinear(parts[0].qweight, parts[0].scale, parts[0].mn, backend)
    return Linear(torch.cat([tensors[n].to(device) for n in names]))


def build_weights(cfg: ModelConfig, tensors: dict, device, backend=DEFAULT_BACKEND) -> ModelWeights:
    """tensors: our names -> CPU tensors (bf16, or the packed triple)."""
    dev = lambda n, dt=None: (tensors[n].to(device) if dt is None else tensors[n].to(device=device, dtype=dt))
    lin = lambda names: _linear(tensors, names, device, backend)
    layers = []
    for i, lt in enumerate(cfg.layer_types):
        p = f"layers.{i}."
        if lt == "linear_attention":
            m = p + "linear_attn."
            mixer = GDNWeights(
                in_qkvz=lin([m + "in_proj_qkv.weight", m + "in_proj_z.weight"]),
                in_ba=torch.cat([dev(m + "in_proj_b.weight"), dev(m + "in_proj_a.weight")]),
                conv_w=dev(m + "conv1d.weight").squeeze(1),
                A=-torch.exp(dev(m + "A_log", torch.float32)),
                dt_bias=dev(m + "dt_bias", torch.float32),
                norm_w=dev(m + "norm.weight"),
                out=lin(m + "out_proj.weight"))
        else:
            m = p + "self_attn."
            mixer = AttnWeights(
                qkv=lin([m + "q_proj.weight", m + "k_proj.weight", m + "v_proj.weight"]),
                o=lin(m + "o_proj.weight"),
                q_norm_w=dev(m + "q_norm.weight"), k_norm_w=dev(m + "k_norm.weight"))
        layers.append(LayerWeights(
            ln1=dev(p + "input_layernorm.weight"), ln2=dev(p + "post_attention_layernorm.weight"),
            mixer=mixer, gate_up=lin([p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"]),
            down=lin(p + "mlp.down_proj.weight")))
    return ModelWeights(embed=dev("embed_tokens.weight"), layers=layers, final_norm=dev("norm.weight"),
                        lm_head=lin("lm_head.weight"))


def load_packed(path: str, device="cuda", with_mtp: bool = False, backend=DEFAULT_BACKEND):
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
    w = build_weights(cfg, tensors, device, backend)
    torch.cuda.synchronize()
    print(f"loaded {w.nbytes / 1e9:.2f} GB of weights in {time.time() - t0:.1f}s")
    return cfg, w, (mtp if with_mtp else None)
