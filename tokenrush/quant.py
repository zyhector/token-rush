"""Weight-only int4 quantization, group-wise asymmetric round-to-nearest.

The dumbest thing that fits on the card: 4 bits per weight plus a bf16 scale
and a bf16 minimum per group of 128, i.e. 4.25 bits per weight. No
calibration. The GEMV is dequantize-then-matmul, which reads and writes far
more than the 4-bit bytes and is the first thing Phase 1a replaces after the
model runs. Quality is Phase 1b's problem.

Packing: element 2j sits in the low nibble of byte j, element 2j+1 in the high
nibble, along the input dimension.
"""
import torch
import torch.nn.functional as F

GROUP = 128


def quantize_int4(w: torch.Tensor, group: int = GROUP):
    """bf16 [out, in] -> (qweight uint8 [out, in/2], scale bf16 [out, in/group], mn bf16 [out, in/group])."""
    out, inp = w.shape
    assert inp % group == 0 and inp % 2 == 0
    wf = w.float().view(out, inp // group, group)
    mn = wf.amin(-1)
    mx = wf.amax(-1)
    scale = (mx - mn) / 15.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    # Round the group parameters to bf16 first so the codes are chosen against
    # exactly the parameters the dequantizer will use.
    scale = scale.to(torch.bfloat16)
    mn = mn.to(torch.bfloat16)
    q = ((wf - mn.float()[..., None]) / scale.float()[..., None]).round().clamp_(0, 15).to(torch.uint8)
    q = q.view(out, inp // 2, 2)
    packed = q[..., 0] | (q[..., 1] << 4)
    return packed.contiguous(), scale.contiguous(), mn.contiguous()


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [out, in/2] -> uint8 codes [out, in]."""
    out = packed.shape[0]
    lo = packed & 0xF
    hi = packed >> 4
    return torch.stack([lo, hi], dim=-1).view(out, -1)


def dequantize_int4(packed, scale, mn) -> torch.Tensor:
    """-> bf16 [out, in]. In-place arithmetic on one bf16 temporary."""
    out = packed.shape[0]
    groups = scale.shape[1]
    w = unpack_int4(packed).to(torch.bfloat16).view(out, groups, -1)
    w.mul_(scale[..., None]).add_(mn[..., None])
    return w.view(out, -1)


class Linear:
    """bf16 weight, y = x @ W.T."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight

    @property
    def shape(self):
        return tuple(self.weight.shape)

    @property
    def nbytes(self):
        return self.weight.numel() * self.weight.element_size()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class QLinear:
    """int4 group-wise weight; dequantize-then-matmul."""

    def __init__(self, qweight, scale, mn):
        self.qweight, self.scale, self.mn = qweight, scale, mn

    @property
    def shape(self):
        return (self.qweight.shape[0], self.qweight.shape[1] * 2)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.qweight, self.scale, self.mn))

    def dequantize(self) -> torch.Tensor:
        return dequantize_int4(self.qweight, self.scale, self.mn)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantize())
