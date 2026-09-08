"""Candidates that need vLLM's compiled kernels: run the harness with
/workspace/venvs/vllm/bin/python. Marlin (GPTQ-style symmetric int4, fp16
scales per group of 128, 4.125 bpw) and NVFP4 (W4A4: fp4 weights and
activations with fp8 block scales per 16, 4.5 bpw, the format vLLM served
Qwen3.8-27B with in Phase 0)."""
import torch

from harness import candidate
from cand_torch import GROUP

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear, marlin_make_empty_g_idx, marlin_make_workspace_new, marlin_permute_scales)
from vllm.scalar_type import scalar_types


@candidate("marlin-int4", 4.125)
class Marlin:
    def pack(self, w):
        N, K = w.shape
        dev = w.device
        wf = w.float().view(N, K // GROUP, GROUP)
        s = (wf.abs().amax(-1) / 7).clamp(min=1e-8)                        # [N, K/g]
        q = (wf / s[..., None]).round().clamp_(-8, 7).add_(8).to(torch.int32).view(N, K)   # 0..15, zero at 8
        s = s.to(torch.bfloat16)
        # GPTQ layout: int32 [K/8, N], 8 codes per word along K
        qk = q.t().contiguous().view(K // 8, 8, N)
        shifts = (torch.arange(8, device=dev, dtype=torch.int32) * 4)[None, :, None]
        gptq = (qk << shifts).sum(1, dtype=torch.int32).contiguous()
        marlin_w = ops.gptq_marlin_repack(gptq, torch.empty(0, dtype=torch.int, device=dev), K, N, 4)
        marlin_s = marlin_permute_scales(s.t().contiguous(), K, N, GROUP)
        empty = marlin_make_empty_g_idx(dev)
        ws = marlin_make_workspace_new(dev)
        return dict(w=marlin_w, s=marlin_s, zp=empty, g=empty, gs=empty, ws=ws, N=N, K=K, q=q, scale=s)

    def dequant(self, p):
        q, s = p["q"], p["scale"]
        N = q.shape[0]
        return ((q.view(N, -1, GROUP).float() - 8) * s.float()[..., None]).view(N, -1).to(torch.bfloat16)

    def run(self, p, x):
        return apply_gptq_marlin_linear(x, p["w"], p["s"], p["zp"], p["g"], p["gs"], p["ws"],
                                        scalar_types.uint4b8, p["N"], p["K"], True)

    def nbytes(self, p):
        return p["w"].numel() * 4 + p["s"].numel() * 2


@candidate("nvfp4-cutlass", 4.5)
class NVFP4:
    """Correctness column is against the *unquantized* bf16 weight (the
    swizzled fp8 block scales are not worth unswizzling here), so it includes
    the fp4 quantization error of both weight and activation."""

    def pack(self, w):
        w_gs = ((448 * 6) / w.float().abs().amax()).to(torch.float32)
        w_fp4, w_sf = ops.scaled_fp4_quant(w, w_gs)
        return dict(w=w_fp4, sf=w_sf, gs=w_gs, ref=w, x_gs=None)

    def dequant(self, p):
        return p["ref"]

    def run(self, p, x):
        if p["x_gs"] is None:
            p["x_gs"] = ((448 * 6) / x.float().abs().amax()).to(torch.float32)
            p["alpha"] = 1.0 / (p["x_gs"] * p["gs"])
        x_fp4, x_sf = ops.scaled_fp4_quant(x, p["x_gs"])
        return ops.cutlass_scaled_fp4_mm(x_fp4, p["w"], x_sf, p["sf"], p["alpha"], torch.bfloat16)

    def nbytes(self, p):
        return p["w"].numel() + p["sf"].numel()
