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
        return self(x).float()[None]                      # [1, T, N]

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
# Re-picked with an L2-proof sweep (cycling >400 MB of weight copies, since the
# autotuner's own timing keeps a 30-90 MB matrix resident in the 96 MB L2).
_GEMV_CONFIGS = [
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 1024}, num_warps=4, num_stages=3),    # lm_head 248320x5120 (99%)
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 512}, num_warps=2, num_stages=2),     # qkv 14336x5120 (87%)
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 256}, num_warps=2, num_stages=3),     # in_qkvz 16384x5120 (87%)
    triton.Config({"BLOCK_N": 16, "BLOCK_K": 1024}, num_warps=4, num_stages=2),   # gate_up 34816x5120 (93.5%)
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
_SPLITK_CONFIGS = [                                                              # L2-proof sweep picks
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 512}, num_warps=4, num_stages=2),     # out/o 5120x6144 (80.5%)
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 512}, num_warps=4, num_stages=3),     # down 5120x17408 (91%)
]


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


# M-row variants for the speculative verify step (M = K+1 tokens, M <= 8): the
# weights are read once and dotted with M input rows. Same packing, same
# per-group folding; the accumulator is [M, BLOCK_N].
MAX_ROWS = 8


# Picks of a 36-config sweep at the model's shapes for M = 2..5 (bench/spec_run.py
# --profile); a full sweep on lm_head costs seconds per M and stalls generation.
_ROWS_CONFIGS = [                                                                 # L2-proof sweep at M=4
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 512}, num_warps=4, num_stages=2),    # in_qkvz 16384 (79%)
    triton.Config({"BLOCK_N": 32, "BLOCK_K": 256}, num_warps=4, num_stages=3),    # qkv 14336 (81%)
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=4, num_stages=2),   # gate_up 34816 (84%)
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 256}, num_warps=4, num_stages=3),    # lm_head (95%)
    triton.Config({"BLOCK_N": 32, "BLOCK_K": 256}, num_warps=4, num_stages=2),    # out/o 5120x6144 split-K (76%)
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=8, num_stages=2),   # down 5120x17408 split-K (82%)
]


@triton.autotune(configs=_ROWS_CONFIGS, key=["N", "K", "M"])
@triton.jit
def _int4_gemm_rows_kernel(x_ptr, w_ptr, s_ptr, m_ptr, y_ptr, N, K,
                           M: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                           GROUP_SIZE: tl.constexpr):
    """y[s, m, n] partials (SPLIT_K > 1, fp32) or y[m, n] (SPLIT_K == 1, bf16).
    Each weight block is dequantized once to bf16 and multiplied against all M
    input rows (padded to 16) on the tensor cores, so the per-row cost is a dot,
    not another pass over the packed bytes."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    KB: tl.constexpr = BLOCK_K // 2
    G: tl.constexpr = BLOCK_K // GROUP_SIZE
    n_groups = K // GROUP_SIZE
    n_blocks = (K + BLOCK_K - 1) // BLOCK_K
    per = (n_blocks + SPLIT_K - 1) // SPLIT_K
    b0 = pid_k * per
    b1 = tl.minimum(b0 + per, n_blocks)
    MP: tl.constexpr = 16                                 # tl.dot needs 16 rows
    mi = tl.arange(0, MP)
    mmask = mi < M
    kk = tl.arange(0, BLOCK_K)
    acc = tl.zeros([MP, BLOCK_N], dtype=tl.float32)
    for blk in range(b0, b1):
        k0 = blk * BLOCK_K
        kb = k0 // 2 + tl.arange(0, KB)
        kmask = kb < K // 2
        pk = tl.load(w_ptr + rows[:, None] * (K // 2) + kb[None, :], mask=row_mask[:, None] & kmask[None, :], other=0)   # [BLOCK_N, KB]
        codes = tl.reshape(tl.join(pk & 0xF, pk >> 4), [BLOCK_N, BLOCK_K]).to(tl.float32)    # element order 2j, 2j+1
        gi = k0 // GROUP_SIZE + tl.arange(0, G)
        gm = row_mask[:, None] & (gi[None, :] < n_groups)
        sc = tl.load(s_ptr + rows[:, None] * n_groups + gi[None, :], mask=gm, other=0).to(tl.float32)   # [BLOCK_N, G]
        mn = tl.load(m_ptr + rows[:, None] * n_groups + gi[None, :], mask=gm, other=0).to(tl.float32)
        wq = tl.reshape(codes, [BLOCK_N, G, GROUP_SIZE]) * sc[:, :, None] + mn[:, :, None]
        wb = tl.reshape(wq, [BLOCK_N, BLOCK_K]).to(tl.bfloat16)                                # dequantized block
        x = tl.load(x_ptr + mi[:, None] * K + k0 + kk[None, :], mask=mmask[:, None] & ((k0 + kk) < K)[None, :], other=0.0)   # [MP, BLOCK_K] bf16
        acc += tl.dot(x, tl.trans(wb))
    omask = mmask[:, None] & row_mask[None, :]
    if SPLIT_K == 1:
        tl.store(y_ptr + mi[:, None] * N + rows[None, :], acc.to(y_ptr.dtype.element_ty), mask=omask)
    else:
        tl.store(y_ptr + (pid_k * M + mi[:, None]) * N + rows[None, :], acc, mask=omask)


def int4_gemm_rows(x, packed, scale, mn, split_k: int = 1):
    """x [M, K] bf16, 2 <= M <= MAX_ROWS -> [M, N] bf16 (split_k == 1) or fp32 partials [split_k, M, N]."""
    M, K = x.shape
    N = packed.shape[0]
    assert 1 <= M <= MAX_ROWS
    if split_k == 1:
        y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    else:
        y = torch.empty(split_k, M, N, device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), split_k)
    _int4_gemm_rows_kernel[grid](x, packed, scale, mn, y, N, K, M=M, SPLIT_K=split_k, GROUP_SIZE=K // scale.shape[1])
    return y


# --------------------------------------------------------------- QLinear

BACKENDS = ("marlin", "triton", "tinygemm", "dequant")


def _default_backend():
    from . import marlin as _marlin
    return "marlin" if _marlin.available() else "triton"


DEFAULT_BACKEND = _default_backend()
MARLIN_ROWS = 16          # the Marlin kernel's row limit (one 16-row tile)


ROWS_FOR_ONE = False      # use the M-row (tensor-core) kernel for T == 1 as well, so single-token
                          # and multi-token steps share bit-identical numerics


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
    dequant:  dequantize-then-matmul at every T; the reference.
    marlin:   the Phase 3c port of Marlin (marlin.py): tensor-core int4 GEMM in
              Marlin's weight layout, one kernel for T <= 16 at the same bandwidth
              for every T (the Triton M-row kernel loses 15-20% by T = 8), and
              bit-identical rows whatever T is. The default when it builds. With
              split_k > 1 the kernel writes lock-free fp32 partial slots instead of
              chaining block reductions. Shapes must be multiples of 128 in both
              dimensions (all of the model's are); others fall back to triton."""

    def __init__(self, qweight, scale, mn, backend=DEFAULT_BACKEND, split_k: int = 1):
        assert backend in BACKENDS, backend
        self.shape_ = (qweight.shape[0], qweight.shape[1] * 2)
        self.group = self.shape_[1] // scale.shape[1]
        if backend == "marlin" and not (self.shape_[0] % 128 == 0 and self.shape_[1] % 128 == 0 and self.group == 128):
            backend = "triton"
        self.backend = backend
        self.split_k = split_k        # > 1: partials(x) uses the split-K kernel (triton / marlin backends)
        if backend == "marlin":
            from . import marlin as _marlin
            self.B, self.ms, self.mm = _marlin.pack(qweight, scale, mn)
            _marlin.workspaces(self.shape_[0], qweight.device)     # sized before any graph capture
            self.qweight = self.scale = self.mn = None
        elif backend == "tinygemm":
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
        if self.backend == "tinygemm":
            ts = (self.tg_w, self.tg_sz)
        elif self.backend == "marlin":
            ts = (self.B, self.ms, self.mm)
        else:
            ts = (self.qweight, self.scale, self.mn)
        return sum(t.numel() * t.element_size() for t in ts)

    def nibbles(self):
        """(qweight, scale, mn) in the engine's packing (unpacked from Marlin's if needed)."""
        if self.backend == "marlin":
            from . import marlin as _marlin
            return _marlin.unpack(self.B, self.ms, self.mm)
        assert self.qweight is not None, "tinygemm backend does not keep the nibbles"
        return self.qweight, self.scale, self.mn

    def rows(self, ids: torch.Tensor):
        """A QLinear over the output rows `ids` (the truncated draft vocabulary)."""
        return QLinear(*(t.index_select(0, ids) for t in self.nibbles()), backend=self.backend, split_k=self.split_k)

    def dequantize(self) -> torch.Tensor:
        return dequantize_int4(*self.nibbles())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "tinygemm":
            return torch.ops.aten._weight_int4pack_mm(x, self.tg_w, self.group, self.tg_sz)
        T = x.shape[0]
        if self.backend == "marlin":
            if T > MARLIN_ROWS:
                return F.linear(x, self.dequantize())
            from . import marlin as _marlin
            if self.split_k > 1:
                return _marlin.mul_partial(x, self.B, self.ms, self.mm).sum(0).to(x.dtype)
            return _marlin.mul(x, self.B, self.ms, self.mm)
        if self.backend == "dequant" or T > MAX_ROWS:
            return F.linear(x, self.dequantize())
        if self.split_k > 1:
            return self.partials(x).sum(0).to(x.dtype)
        if T == 1 and not ROWS_FOR_ONE:
            return int4_gemv(x, self.qweight, self.scale, self.mn)
        return int4_gemm_rows(x, self.qweight, self.scale, self.mn)

    def partials(self, x: torch.Tensor) -> torch.Tensor:
        """x [T, K] -> [S, T, N] fp32 whose sum over S is the product (S == 1 unless split-K)."""
        T = x.shape[0]
        if self.backend == "marlin" and self.split_k > 1 and T <= MARLIN_ROWS:
            from . import marlin as _marlin
            return _marlin.mul_partial(x, self.B, self.ms, self.mm)
        if self.backend == "triton" and self.split_k > 1 and T <= MAX_ROWS:
            if T == 1 and not ROWS_FOR_ONE:
                return int4_gemv_splitk(x, self.qweight, self.scale, self.mn, self.split_k)[:, None, :]
            return int4_gemm_rows(x, self.qweight, self.scale, self.mn, self.split_k)
        return self(x).float()[None]
