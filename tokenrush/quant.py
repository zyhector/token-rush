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
import triton
import triton.language as tl

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

    def partials(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).float()

    @property
    def shape(self):
        return tuple(self.weight.shape)

    @property
    def nbytes(self):
        return self.weight.numel() * self.weight.element_size()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


# ------------------------------------------------------------ Triton GEMV
# Our own int4 GEMV over this packing. The dequantization folds into two
# scalars per (row, group): sum_i (q_i*s + m) x_i = s * sum_i q_i x_i + m * sum_i x_i,
# so the inner loop is an integer-times-bf16 dot product.


# The winners of a 108-config sweep over the model's six decode shapes
# (scripts/gemv_shootout, bench/decode.py --profile); a full sweep costs
# 15 s on lm_head alone, this handful costs under a second per shape.
_GEMV_CONFIGS = [
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 1024}, num_warps=4, num_stages=3),    # lm_head 248320x5120
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 1024}, num_warps=4, num_stages=4),    # qkv 14336x5120
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 1024}, num_warps=8, num_stages=2),    # out 5120x6144
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 1024}, num_warps=8, num_stages=3),    # down 5120x17408
    triton.Config({"BLOCK_N": 16, "BLOCK_K": 1024}, num_warps=4, num_stages=4),   # gate_up 34816x5120
    triton.Config({"BLOCK_N": 32, "BLOCK_K": 512}, num_warps=4, num_stages=4),    # in_qkvz 16384x5120
]


@triton.autotune(configs=_GEMV_CONFIGS, key=["N", "K"])
@triton.jit
def _int4_gemv_kernel(x_ptr, w_ptr, s_ptr, m_ptr, y_ptr, N, K,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    KB: tl.constexpr = BLOCK_K // 2
    G: tl.constexpr = BLOCK_K // GROUP_SIZE
    GB: tl.constexpr = GROUP_SIZE // 2
    n_groups = K // GROUP_SIZE
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        kb = k0 // 2 + tl.arange(0, KB)
        xe = tl.load(x_ptr + k0 + 2 * tl.arange(0, KB)).to(tl.float32)
        xo = tl.load(x_ptr + k0 + 2 * tl.arange(0, KB) + 1).to(tl.float32)
        p = tl.load(w_ptr + rows[:, None] * (K // 2) + kb[None, :], mask=row_mask[:, None], other=0)
        prod = (p & 0xF).to(tl.float32) * xe[None, :] + (p >> 4).to(tl.float32) * xo[None, :]
        pg = tl.sum(tl.reshape(prod, [BLOCK_N, G, GB]), axis=2)
        xs = tl.sum(tl.reshape(xe + xo, [G, GB]), axis=1)
        gi = k0 // GROUP_SIZE + tl.arange(0, G)
        sc = tl.load(s_ptr + rows[:, None] * n_groups + gi[None, :], mask=row_mask[:, None], other=0).to(tl.float32)
        mn = tl.load(m_ptr + rows[:, None] * n_groups + gi[None, :], mask=row_mask[:, None], other=0).to(tl.float32)
        acc += tl.sum(pg * sc + xs[None, :] * mn, axis=1)
    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=row_mask)


def int4_gemv(x, packed, scale, mn):
    """x [1, K] bf16 -> [1, N] bf16."""
    N, K = packed.shape[0], packed.shape[1] * 2
    y = torch.empty(1, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _int4_gemv_kernel[grid](x, packed, scale, mn, y, N, K, GROUP_SIZE=K // scale.shape[1])
    return y


# Split-K variant for the narrow shapes (N = 5120: out_proj, down_proj), whose
# N/BLOCK_N programs each stream a long K alone and leave bandwidth idle. The
# K range is cut into SPLIT_K pieces, each program writes an fp32 partial row
# block, and the consumer (fused add+RMSNorm) sums the partials as it reads
# them, so the reduction costs no launch.
_SPLITK_CONFIGS = [
    triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
    for bn in (8, 16) for bk in (512, 1024) for nw in (4, 8) for ns in (2, 3, 4)]


@triton.autotune(configs=_SPLITK_CONFIGS, key=["N", "K", "SPLIT_K"])
@triton.jit
def _int4_gemv_splitk_kernel(x_ptr, w_ptr, s_ptr, m_ptr, y_ptr, N, K,
                             SPLIT_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                             GROUP_SIZE: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    KB: tl.constexpr = BLOCK_K // 2
    G: tl.constexpr = BLOCK_K // GROUP_SIZE
    GB: tl.constexpr = GROUP_SIZE // 2
    n_groups = K // GROUP_SIZE
    n_blocks = K // BLOCK_K
    per = (n_blocks + SPLIT_K - 1) // SPLIT_K
    b0 = pid_k * per
    b1 = tl.minimum(b0 + per, n_blocks)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for blk in range(b0, b1):
        k0 = blk * BLOCK_K
        kb = k0 // 2 + tl.arange(0, KB)
        xe = tl.load(x_ptr + k0 + 2 * tl.arange(0, KB)).to(tl.float32)
        xo = tl.load(x_ptr + k0 + 2 * tl.arange(0, KB) + 1).to(tl.float32)
        pk = tl.load(w_ptr + rows[:, None] * (K // 2) + kb[None, :], mask=row_mask[:, None], other=0)
        prod = (pk & 0xF).to(tl.float32) * xe[None, :] + (pk >> 4).to(tl.float32) * xo[None, :]
        pg = tl.sum(tl.reshape(prod, [BLOCK_N, G, GB]), axis=2)
        xs = tl.sum(tl.reshape(xe + xo, [G, GB]), axis=1)
        gi = k0 // GROUP_SIZE + tl.arange(0, G)
        sc = tl.load(s_ptr + rows[:, None] * n_groups + gi[None, :], mask=row_mask[:, None], other=0).to(tl.float32)
        mn = tl.load(m_ptr + rows[:, None] * n_groups + gi[None, :], mask=row_mask[:, None], other=0).to(tl.float32)
        acc += tl.sum(pg * sc + xs[None, :] * mn, axis=1)
    tl.store(y_ptr + pid_k * N + rows, acc, mask=row_mask)


def int4_gemv_splitk(x, packed, scale, mn, split_k: int):
    """x [1, K] bf16 -> fp32 partials [split_k, N]; sum over dim 0 is the GEMV."""
    N, K = packed.shape[0], packed.shape[1] * 2
    y = torch.empty(split_k, N, device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), split_k)
    _int4_gemv_splitk_kernel[grid](x, packed, scale, mn, y, N, K, SPLIT_K=split_k, GROUP_SIZE=K // scale.shape[1])
    return y


# --------------------------------------------------------------- QLinear

BACKENDS = ("triton", "tinygemm", "dequant")
DEFAULT_BACKEND = "triton"


class QLinear:
    """int4 group-wise weight with a choice of GEMV backend.

    triton:   our own kernel above. 88% of the wall on the shootout's fused
              layer set, 92% inside the real model; prefill (T > 1) dequantizes
              the same nibbles and uses cuBLAS. The default, and the kernel
              Phase 2 fuses further.
    tinygemm: torch's built-in _weight_int4pack_mm, 2.5% faster in the
              shootout but its packed layout is opaque, so it holds the weight
              in that layout only and serves prefill from it too, 6x slower
              than dequant+GEMM at T >= 512. Kept as a reference backend.
    dequant:  dequantize-then-matmul at every T; the reference."""

    def __init__(self, qweight, scale, mn, backend=DEFAULT_BACKEND, split_k: int = 1):
        assert backend in BACKENDS, backend
        self.backend = backend
        self.split_k = split_k        # > 1: partials(x) uses the split-K kernel (triton backend only)
        self.shape_ = (qweight.shape[0], qweight.shape[1] * 2)
        self.group = self.shape_[1] // scale.shape[1]
        if backend == "tinygemm":
            # tinygemm reads the high nibble as the even element (probed, not
            # documented) and dequantizes as (q - 8) * scale + zero.
            swapped = ((qweight & 0xF) << 4) | (qweight >> 4)
            self.tg_w = torch.ops.aten._convert_weight_to_int4pack(swapped.contiguous(), 8)
            zero = (mn.float() + 8 * scale.float()).to(torch.bfloat16)
            self.tg_sz = torch.stack([scale.t(), zero.t()], -1).contiguous()   # [K/g, N, 2]
            self.qweight = self.scale = self.mn = None
        else:
            self.qweight, self.scale, self.mn = qweight, scale, mn

    @staticmethod
    def cat(parts, backend=DEFAULT_BACKEND):
        """Concatenate along the output dimension: rows are independent in this packing.
        parts must be 'dequant' or 'triton' backed (they keep the nibbles)."""
        return QLinear(torch.cat([p.qweight for p in parts]), torch.cat([p.scale for p in parts]),
                       torch.cat([p.mn for p in parts]), backend)

    @property
    def shape(self):
        return self.shape_

    @property
    def nbytes(self):
        ts = (self.tg_w, self.tg_sz) if self.backend == "tinygemm" else (self.qweight, self.scale, self.mn)
        return sum(t.numel() * t.element_size() for t in ts)

    def dequantize(self) -> torch.Tensor:
        assert self.qweight is not None, "tinygemm backend does not keep the nibbles"
        return dequantize_int4(self.qweight, self.scale, self.mn)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "tinygemm":
            return torch.ops.aten._weight_int4pack_mm(x, self.tg_w, self.group, self.tg_sz)
        if x.shape[0] != 1 or self.backend == "dequant":
            return F.linear(x, self.dequantize())
        if self.split_k > 1:
            return self.partials(x).sum(0, keepdim=True).to(x.dtype)
        return int4_gemv(x, self.qweight, self.scale, self.mn)

    def partials(self, x: torch.Tensor) -> torch.Tensor:
        """x [1, K] -> [S, N] fp32 whose sum over S is the GEMV (S == 1 unless split-K)."""
        if self.backend == "triton" and self.split_k > 1 and x.shape[0] == 1:
            return int4_gemv_splitk(x, self.qweight, self.scale, self.mn, self.split_k)
        return self(x).float()
