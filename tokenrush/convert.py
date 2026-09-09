"""Re-pack a compressed-tensors int4 checkpoint (llm-compressor / AutoRound
export, e.g. RedHatAI/Qwen3.8-27B-INT4) into the engine's int4 g128 packing.

    python -m tokenrush.convert --src /workspace/models/Qwen3.8-27B-RedHatAI-INT4 \
        --hf /workspace/models/Qwen3.8-27B --head /workspace/models/Qwen3.8-27B-int4g128-gptq \
        --dst /workspace/models/Qwen3.8-27B-int4g128-redhat

The codes are the same 4-bit integers; the only arithmetic is the zero point:
their w = (code - zp) * scale becomes our w = code * scale + mn with
mn = -zp * scale (exact in bf16 for the symmetric zp = 8 and a bf16 scale).
Every other tensor comes from the source checkpoint too — its AWQ smoothing
folded per-channel scales into the layer norms and the small projections, so
they are not the original bf16 values — except the MTP head, which the source
does not carry and which comes from the HF checkpoint. lm_head, which these
checkpoints keep in bf16, is taken from a packed checkpoint of ours
(--head; the GPTQ one) or round-to-nearest quantized (--head rtn).
"""
import argparse
import os

import torch

from .quant import GROUP, quantize_int4
from .weights import HFTensors, MTP_PREFIX, PACK_META, is_quantized, write_packed


def to_our_packing(codes: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor):
    """codes uint8 [out, in], scale [out, groups], zp int [out, groups] -> (qweight, scale, mn)."""
    out, inp = codes.shape
    q = codes.view(out, inp // 2, 2)
    packed = (q[..., 0] | (q[..., 1] << 4)).contiguous()
    scale = scale.to(torch.bfloat16)
    mn = (-zp.float() * scale.float()).to(torch.bfloat16)
    return packed, scale.contiguous(), mn.contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="compressed-tensors int4 checkpoint")
    ap.add_argument("--hf", required=True, help="the bf16 HF checkpoint (for the MTP head)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--head", default="rtn", help="packed checkpoint whose lm_head to take, or 'rtn'")
    ap.add_argument("--group", type=int, default=GROUP)
    a = ap.parse_args()
    from bench.quality_sources import CTInt4Source          # the format reader lives with the quality tools
    src = CTInt4Source(a.src, device="cuda")
    hf = HFTensors(a.hf)
    assert src.group == a.group, (src.group, a.group)

    if a.head == "rtn":
        head = quantize_int4(src["lm_head.weight"].cuda(), a.group)
        head_meta = "rtn"
    else:
        from safetensors import safe_open
        import glob
        head = None
        for shard in sorted(glob.glob(os.path.join(a.head, "model-*.safetensors"))):
            with safe_open(shard, framework="pt", device="cpu") as f:
                if "lm_head.weight.qweight" in f.keys():
                    head = tuple(f.get_tensor("lm_head.weight" + s) for s in (".qweight", ".scale", ".mn"))
        assert head is not None, f"no packed lm_head in {a.head}"
        head_meta = os.path.abspath(a.head)

    def items():
        n_q = 0
        for name in sorted(src.names()):
            if name == "lm_head.weight":
                yield name, head
            elif name in src.q:
                assert is_quantized(name), name
                yield name, to_our_packing(*src.codes(name))
                n_q += 1
            else:
                yield name, src[name]
        for name in hf.files:
            if name.startswith(MTP_PREFIX):
                yield name, hf[name]
        print(f"{n_q} matrices re-packed from {a.src}")

    meta = {"source": os.path.abspath(a.src), "method": "converted", "converted_from": src.label,
            "lm_head": head_meta, "mtp_from": os.path.abspath(a.hf)}
    write_packed(a.dst, items(), a.group, a.src, meta)


if __name__ == "__main__":
    main()
