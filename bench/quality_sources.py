"""Tensor sources for the quantization-quality table: every candidate checkpoint
presented as the same name -> bf16 tensor mapping (our names, as in
tokenrush/weights.py), so all of them run through the engine's bf16 path and
the table measures weights, not engines.

    hf:<dir>        the bf16 HF checkpoint (the reference; also the base every
                    overlay sits on)
    packed:<dir>    our int4 g128 packing, dequantized with tokenrush.quant
    gguf:<file>     a llama.cpp GGUF: its quantized matrices (incl. token_embd
                    and output) dequantized with the gguf package, overlaid on hf
    nvfp4:<dir>     a compressed-tensors NVFP4 checkpoint (QUASAR-QAT): fp4 codes
                    x fp8 block scales / global scale; every other tensor from
                    that checkpoint (it was QAT-trained, so they may differ)
    bf16dir:<dir>   bf16 safetensors with our names (the ExLlamaV3 export from
                    bench/exl3_export.py), overlaid on hf
    ct:<dir>        a compressed-tensors int4 pack-quantized checkpoint (llm-compressor,
                    AutoRound export; e.g. RedHatAI/Qwen3.8-27B-INT4), its own bf16 for
                    the rest

Each source also accounts the bytes its format stores for the matrices the
engine streams per token (the 401 tensors we quantize), so the table can carry
a bits-per-weight column computed the same way for everyone.
"""
import glob
import json
import os

import torch

from tokenrush.quant import dequantize_int4
from tokenrush.weights import HFTensors, HF_PREFIX, PACK_META, is_quantized

# ------------------------------------------------------------ base + overlay


class Source:
    """name -> tensor (any device; build_layer moves it). Subclasses fill `_get`."""
    label = "?"

    def __init__(self, base=None, device="cuda", cache=False):
        self.base = base
        self.device = device
        self.bytes = {}          # our name -> bytes this format stores for it (quantized tensors only)
        self.cache = {} if cache else None    # dequantized bf16 kept in host RAM (for slow CPU dequantizers)

    def names(self):
        raise NotImplementedError

    def __contains__(self, name):
        return name in self.names() or (self.base is not None and name in self.base)

    def __getitem__(self, name):
        if name in self.names():
            if self.cache is None:
                return self._get(name)
            if name not in self.cache:
                self.cache[name] = self._get(name).cpu()
            return self.cache[name]
        return self.base[name]

    def _get(self, name):
        raise NotImplementedError

    def streamed_bpw(self, hf_dir):
        """Bits per weight over the tensors the engine streams (those we quantize),
        from this format's storage bytes; tensors the format keeps in bf16 count 16."""
        cfg_names = [n for n in HFTensors(hf_dir).files if is_quantized(n)]
        hf = HFTensors(hf_dir)
        bits = params = 0
        for n in cfg_names:
            numel = hf.files[n][0].get_slice(hf.files[n][1]).get_shape()
            numel = int(torch.tensor(numel).prod())
            params += numel
            bits += 8 * self.bytes.get(n, 2 * numel)
        return bits / params, params


class HFSource(Source):
    label = "bf16"

    def __init__(self, src, device="cuda"):
        super().__init__(None, device)
        self.t = HFTensors(src)

    def names(self):
        return self.t.files

    def _get(self, name):
        return self.t[name]


class PackedSource(Source):
    """Our packed int4 checkpoint, dequantized on the device."""
    label = "int4 g128 (ours)"

    def __init__(self, path, base, device="cuda"):
        super().__init__(base, device)
        from safetensors import safe_open
        self.meta = json.load(open(os.path.join(path, PACK_META)))
        self.files = {}
        for shard in sorted(glob.glob(os.path.join(path, "model-*.safetensors"))):
            f = safe_open(shard, framework="pt", device="cpu")
            for k in f.keys():
                self.files[k] = f
        self.q = set(self.meta["quantized"])
        for n in self.q:
            self.bytes[n] = sum(self._nbytes(n + s) for s in (".qweight", ".scale", ".mn"))

    def _nbytes(self, k):
        sl = self.files[k].get_slice(k)
        dt = {"U8": 1, "BF16": 2, "F16": 2, "F32": 4}[sl.get_dtype()]
        return int(torch.tensor(sl.get_shape()).prod()) * dt

    def names(self):
        return self.q

    def _get(self, name):
        q, s, m = (self.files[name + k].get_tensor(name + k).to(self.device) for k in (".qweight", ".scale", ".mn"))
        return dequantize_int4(q, s, m)


# ------------------------------------------------------------ GGUF

GGUF_MAP = {                      # our suffix -> gguf suffix
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


# llama.cpp's converter reorders the GDN value heads: its head j is HF's head
# 3*(j % 16) + j // 16 (recovered in docs/progress.md, Phase 1b, by matching
# rows against the bf16 weights). q and k heads keep their order.
GDN_V_HEADS, GDN_K_HEADS, GDN_HEAD_DIM = 48, 16, 128
_GGUF_V_PERM = [3 * (j % 16) + j // 16 for j in range(GDN_V_HEADS)]      # gguf head j -> hf head
_GGUF_V_INV = torch.tensor(sorted(range(GDN_V_HEADS), key=lambda j: _GGUF_V_PERM[j]))   # hf head h -> gguf head


def _unpermute_v_heads(w: torch.Tensor, dim: int, offset: int = 0) -> torch.Tensor:
    """Reorder the 48 value-head blocks of 128 along `dim` (starting at `offset`) from
    llama.cpp's order back to HF's."""
    w = w.movedim(dim, 0)
    head = w[offset:offset + GDN_V_HEADS * GDN_HEAD_DIM].reshape(GDN_V_HEADS, GDN_HEAD_DIM, *w.shape[1:])
    head = head[_GGUF_V_INV.to(w.device)].reshape(GDN_V_HEADS * GDN_HEAD_DIM, *w.shape[1:])
    w = torch.cat([w[:offset], head, w[offset + GDN_V_HEADS * GDN_HEAD_DIM:]])
    return w.movedim(0, dim).contiguous()


def _unpermute_heads(w: torch.Tensor, dim: int) -> torch.Tensor:
    """48 rows/cols, one per value head (in_proj_a / in_proj_b)."""
    w = w.movedim(dim, 0)
    return w[_GGUF_V_INV.to(w.device)].movedim(0, dim).contiguous()


class GGUFSource(Source):
    label = "GGUF"

    def __init__(self, path, base, device="cuda"):
        super().__init__(base, device, cache=True)
        import gguf
        self.gguf = gguf
        self.reader = gguf.GGUFReader(path)
        by_name = {t.name: t for t in self.reader.tensors}
        self.map = {}             # our name -> gguf tensor
        for gname, t in by_name.items():
            if gname == "token_embd.weight":
                ours = "embed_tokens.weight"
            elif gname == "output.weight":
                ours = "lm_head.weight"
            elif gname.startswith("blk."):
                _, i, rest = gname.split(".", 2)
                suffix = next((o for o, g in GGUF_MAP.items() if g == rest), None)
                if suffix is None:
                    continue
                ours = f"layers.{i}.{suffix}"
            else:
                continue
            self.map[ours] = t
            self.bytes[ours] = t.data.nbytes
        self.types = {n: t.tensor_type.name for n, t in self.map.items()}

    def names(self):
        return self.map

    def _get(self, name):
        t = self.map[name]
        arr = self.gguf.quants.dequantize(t.data, t.tensor_type)          # float32, [rows, ne0]
        shape = [int(s) for s in reversed(t.shape)]                       # gguf lists ne0 first
        w = torch.from_numpy(arr.reshape(shape)).to(self.device, torch.bfloat16)
        if ".linear_attn." in name:
            if name.endswith(("in_proj_a.weight", "in_proj_b.weight")):
                w = _unpermute_heads(w, 0)
            elif name.endswith("in_proj_z.weight"):
                w = _unpermute_v_heads(w, 0)
            elif name.endswith("in_proj_qkv.weight"):
                w = _unpermute_v_heads(w, 0, offset=2 * GDN_K_HEADS * GDN_HEAD_DIM)
            elif name.endswith("out_proj.weight"):
                w = _unpermute_v_heads(w, 1)
        return w


# ------------------------------------------------------------ NVFP4 (compressed-tensors)

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [out, in/2] -> float32 codes [out, in]; low nibble = even element."""
    lo = packed & 0xF
    hi = packed >> 4
    codes = torch.stack([lo, hi], -1).view(packed.shape[0], -1).long()
    mag = _E2M1.to(packed.device)[codes & 0x7]
    sign = torch.where(codes & 0x8 != 0, -1.0, 1.0)
    return mag * sign


class NVFP4Source(Source):
    label = "NVFP4"

    def __init__(self, path, device="cuda"):
        super().__init__(None, device)
        from safetensors import safe_open
        self.files = {}
        for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            f = safe_open(shard, framework="pt", device="cpu")
            for k in f.keys():
                self.files[k] = f
        self.q, self.plain = {}, {}
        for k in self.files:
            if k.startswith("model.visual."):
                continue
            ours = k[len(HF_PREFIX):] if k.startswith(HF_PREFIX) else k
            if ours.endswith(".weight_packed"):
                n = ours[:-len("_packed")]
                self.q[n] = k[:-len("_packed")]
                self.bytes[n] = sum(self._nbytes(k[:-len("_packed")] + s) for s in ("_packed", "_scale", "_global_scale"))
            elif ours.endswith((".weight_scale", ".weight_global_scale", ".input_global_scale")):
                continue
            else:
                self.plain[ours] = k
        self._names = set(self.q) | set(self.plain)

    def _nbytes(self, k):
        sl = self.files[k].get_slice(k)
        dt = {"U8": 1, "F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4}[sl.get_dtype()]
        return int(torch.tensor(sl.get_shape()).prod()) * dt

    def names(self):
        return self._names

    def _get(self, name):
        if name in self.plain:
            k = self.plain[name]
            return self.files[k].get_tensor(k)
        k = self.q[name]
        packed = self.files[k + "_packed"].get_tensor(k + "_packed").to(self.device)
        scale = self.files[k + "_scale"].get_tensor(k + "_scale").to(self.device).float()      # e4m3 per 16
        gscale = self.files[k + "_global_scale"].get_tensor(k + "_global_scale").to(self.device).float()
        codes = unpack_fp4(packed)                                                             # [out, in]
        out, inp = codes.shape
        w = codes.view(out, inp // 16, 16) * (scale / gscale)[..., None]
        return w.view(out, inp).to(torch.bfloat16)


# ------------------------------------------------------------ compressed-tensors int4 pack-quantized


class CTInt4Source(Source):
    """A compressed-tensors `pack-quantized` int4 checkpoint (llm-compressor / AutoRound
    export; e.g. RedHatAI/Qwen3.8-27B-INT4): weight_packed int32 [out, in/8] with column
    k in nibble k % 8 (low first), nibble = code + 8 for the signed code in [-8, 7];
    weight_scale [out, in/group]; symmetric unless weight_zero_point is present
    (int32 [out/8, in/group], packed along the output dim, also offset by 8).
    Dequant: w = (nibble - zp) * scale with zp = 8 when symmetric. Tensors the checkpoint
    keeps in bf16 come from it too (lm_head, embeddings, norms)."""
    label = "compressed-tensors int4"

    def __init__(self, path, device="cuda"):
        super().__init__(None, device)
        from safetensors import safe_open
        self.files = {}
        for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            f = safe_open(shard, framework="pt", device="cpu")
            for k in f.keys():
                self.files[k] = f
        cfg = json.load(open(os.path.join(path, "config.json"))).get("quantization_config", {})
        groups = cfg.get("config_groups", {})
        w = next(iter(groups.values()))["weights"] if groups else {}
        self.group = w.get("group_size", 128)
        self.symmetric = w.get("symmetric", True)
        self.label = f"compressed-tensors int4 g{self.group} {'sym' if self.symmetric else 'asym'}"
        self.q, self.plain = {}, {}
        for k in self.files:
            if k.startswith("model.visual."):
                continue
            ours = k[len(HF_PREFIX):] if k.startswith(HF_PREFIX) else k
            if ours.endswith(".weight_packed"):
                n = ours[:-len("_packed")]
                self.q[n] = k[:-len("_packed")]
                self.bytes[n] = sum(self._nbytes(k[:-len("_packed")] + s)
                                    for s in ("_packed", "_scale", "_zero_point") if k[:-len("_packed")] + s in self.files)
            elif ours.endswith((".weight_scale", ".weight_zero_point", ".weight_shape", "k_scale", "v_scale")):
                continue
            else:
                self.plain[ours] = k
        self._names = set(self.q) | set(self.plain)

    def _nbytes(self, k):
        sl = self.files[k].get_slice(k)
        dt = {"U8": 1, "I8": 1, "I32": 4, "BF16": 2, "F16": 2, "F32": 4}[sl.get_dtype()]
        return int(torch.tensor(sl.get_shape()).prod()) * dt

    def names(self):
        return self._names

    @staticmethod
    def unpack_int32(packed: torch.Tensor, dim: int) -> torch.Tensor:
        """int32 packed 8 nibbles per word along `dim` (low nibble first) -> uint8 nibbles."""
        packed = packed.movedim(dim, -1)
        nib = torch.stack([(packed >> (4 * i)) & 0xF for i in range(8)], -1)      # [..., words, 8]
        return nib.reshape(*packed.shape[:-1], -1).to(torch.uint8).movedim(-1, dim)

    def codes(self, name):
        """(codes uint8 [out, in], scale [out, groups], zero-point int [out, groups]) in the
        checkpoint's own convention: w = (codes - zp) * scale."""
        k = self.q[name]
        packed = self.files[k + "_packed"].get_tensor(k + "_packed").to(self.device)
        scale = self.files[k + "_scale"].get_tensor(k + "_scale").to(self.device)
        codes = self.unpack_int32(packed, 1)                                          # [out, in]
        if k + "_zero_point" in self.files:
            zp = self.unpack_int32(self.files[k + "_zero_point"].get_tensor(k + "_zero_point").to(self.device), 0)
            zp = zp.to(torch.int32)                                                  # [out, groups]
        else:
            zp = torch.full_like(scale, 8, dtype=torch.int32)
        return codes, scale, zp

    def _get(self, name):
        if name in self.plain:
            k = self.plain[name]
            return self.files[k].get_tensor(k)
        codes, scale, zp = self.codes(name)
        out, inp = codes.shape
        g = inp // scale.shape[1]
        w = (codes.view(out, -1, g).float() - zp[..., None].float()) * scale[..., None].float()
        return w.view(out, inp).to(torch.bfloat16)


# ------------------------------------------------------------ bf16 export dir (ExLlamaV3)


class BF16DirSource(Source):
    label = "bf16 export"

    def __init__(self, path, base, device="cuda"):
        super().__init__(base, device)
        from safetensors import safe_open
        self.files = {}
        for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            f = safe_open(shard, framework="pt", device="cpu")
            for k in f.keys():
                self.files[k] = f
        meta_fn = os.path.join(path, "export.json")
        if os.path.exists(meta_fn):
            meta = json.load(open(meta_fn))
            self.bytes = {k: int(v) for k, v in meta.get("bytes", {}).items()}
            self.label = meta.get("label", self.label)

    def names(self):
        return self.files

    def _get(self, name):
        return self.files[name].get_tensor(name)


class KeepBF16(Source):
    """A candidate with the tensors matching a regex taken from bf16 instead
    (attribution: how much of the loss is the head, or a layer range)."""

    def __init__(self, inner, pattern, hf_dir, device="cuda"):
        import re
        super().__init__(HFSource(hf_dir, device), device)
        self.inner, self.pat = inner, re.compile(pattern)
        self.kept = [n for n in inner.names() if self.pat.search(n)]
        self.bytes = {n: b for n, b in inner.bytes.items() if not self.pat.search(n)}
        self.label = f"{inner.label} with {pattern!r} in bf16"

    def names(self):
        return self.inner.names()

    def _get(self, name):
        return self.base[name] if self.pat.search(name) else self.inner[name]


class Overlay(Source):
    """`inner` with the named tensors taken from `other` instead (a rival's body with our head)."""

    def __init__(self, inner, other, names, device="cuda"):
        super().__init__(None, device)
        self.inner, self.other, self.over = inner, other, set(names)
        self.bytes = dict(inner.bytes)
        for n in self.over:
            self.bytes.pop(n, None)
            if n in other.bytes:
                self.bytes[n] = other.bytes[n]
        self.label = f"{inner.label} + {other.label} for {sorted(self.over)}"

    def names(self):
        return set(self.inner.names()) | self.over

    def __contains__(self, name):
        return name in self.over or name in self.inner

    def _get(self, name):
        return self.other[name] if name in self.over else self.inner[name]


# ------------------------------------------------------------ factory


def open_source(spec: str, hf_dir: str, device="cuda") -> Source:
    kind, _, path = spec.partition(":")
    base = HFSource(hf_dir, device)
    if kind == "hf":
        return HFSource(path, device)
    if kind == "packed":
        return PackedSource(path, base, device)
    if kind == "gguf":
        return GGUFSource(path, base, device)
    if kind == "nvfp4":
        return NVFP4Source(path, device)
    if kind == "bf16dir":
        return BF16DirSource(path, base, device)
    if kind == "ct":
        return CTInt4Source(path, device)
    raise ValueError(spec)
