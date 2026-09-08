"""Candidates that need only torch: the bf16 cuBLAS reference, the engine's
current dequantize-then-matmul, and PyTorch's built-in tinygemm int4."""
import torch
import torch.nn.functional as F

from harness import candidate

GROUP = 128


def _rtn(w, group=GROUP):
    """Asymmetric int4 RTN: codes [out, in] uint8 in 0..15, scale, mn [out, in/group] bf16."""
    out, inp = w.shape
    wf = w.float().view(out, inp // group, group)
    mn, mx = wf.amin(-1), wf.amax(-1)
    scale = torch.where(mx == mn, torch.ones_like(mx), (mx - mn) / 15).to(torch.bfloat16)
    mn = mn.to(torch.bfloat16)
    q = ((wf - mn.float()[..., None]) / scale.float()[..., None]).round().clamp_(0, 15).to(torch.uint8)
    return q.view(out, inp), scale, mn


def _dequant(q, scale, mn):
    out, groups = scale.shape
    w = q.to(torch.bfloat16).view(out, groups, -1)
    return (w * scale[..., None] + mn[..., None]).view(out, -1)


@candidate("bf16-cublas", 16.0)
class BF16:
    def pack(self, w):
        return w.clone()

    def run(self, p, x):
        return F.linear(x, p)

    def dequant(self, p):
        return p

    def nbytes(self, p):
        return p.numel() * 2


@candidate("dequant-mm", 4.25)
class DequantMM:
    """What the engine does today (tokenrush.quant.QLinear)."""

    def pack(self, w):
        q, s, m = _rtn(w)
        packed = q[:, ::2] | (q[:, 1::2] << 4)
        return packed.contiguous(), s, m

    def dequant(self, p):
        packed, s, m = p
        q = torch.stack([packed & 0xF, packed >> 4], -1).view(packed.shape[0], -1)
        return _dequant(q, s, m)

    def run(self, p, x):
        return F.linear(x, self.dequant(p))

    def nbytes(self, p):
        return sum(t.numel() * t.element_size() for t in p)


@candidate("tinygemm-int4", 4.25)
class TinyGemm:
    """torch.ops.aten._weight_int4pack_mm: int4 weight, bf16 scale+zero per group,
    the kernel torchao's int4_weight_only uses. Dequant convention:
    w = (q - 8) * scale + zero, with zero = mn + 8 * scale so the codes are ours."""
    inner_k_tiles = 8

    def pack(self, w):
        q, s, m = _rtn(w)
        out, inp = w.shape
        # tinygemm reads the high nibble as the even element: probed, not documented
        packed = (q[:, 1::2] | (q[:, ::2] << 4)).contiguous()
        wp = torch.ops.aten._convert_weight_to_int4pack(packed, self.inner_k_tiles)
        zero = (m.float() + 8 * s.float()).to(torch.bfloat16)
        # [in/group, out, 2]: scale, zero
        sz = torch.stack([s.t(), zero.t()], -1).contiguous()
        return wp, sz, (out, inp), (s, m, q)

    def dequant(self, p):
        _, _, _, (s, m, q) = p
        return _dequant(q, s, m)

    def run(self, p, x):
        wp, sz, (out, inp), _ = p
        return torch.ops.aten._weight_int4pack_mm(x, wp, GROUP, sz)

    def nbytes(self, p):
        wp, sz, _, _ = p
        return wp.numel() * 4 + sz.numel() * 2
