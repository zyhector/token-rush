"""Our own Triton int4 GEMV over the engine's packing (tokenrush.quant):
byte j of a row holds elements 2j (low nibble) and 2j+1 (high nibble), a
bf16 scale and minimum per group of 128 along the input.

The dequantization folds into two scalars per (row, group):
    sum_i (q_i * s + m) * x_i  =  s * sum_i q_i x_i  +  m * sum_i x_i
so the inner loop is an int-times-bf16 dot product and the group parameters
touch the accumulator once per group."""
import torch
import triton
import triton.language as tl

from harness import candidate
from cand_torch import _rtn, _dequant, GROUP


@triton.autotune(
    configs=[triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
             for bn in (8, 16, 32, 64) for bk in (256, 512, 1024) for nw in (2, 4, 8) for ns in (2, 3, 4)
             if bn * bk // 2 >= 2048],
    key=["N", "K"],
)
@triton.jit
def _int4_gemv_kernel(x_ptr, w_ptr, s_ptr, m_ptr, y_ptr, N, K,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    KB: tl.constexpr = BLOCK_K // 2            # packed bytes per row per iteration
    G: tl.constexpr = BLOCK_K // GROUP_SIZE    # groups per iteration
    GB: tl.constexpr = GROUP_SIZE // 2         # bytes per group
    n_groups = K // GROUP_SIZE
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        kb = k0 // 2 + tl.arange(0, KB)                                   # byte offsets
        xe = tl.load(x_ptr + k0 + 2 * tl.arange(0, KB)).to(tl.float32)      # even elements
        xo = tl.load(x_ptr + k0 + 2 * tl.arange(0, KB) + 1).to(tl.float32)  # odd elements
        p = tl.load(w_ptr + rows[:, None] * (K // 2) + kb[None, :], mask=row_mask[:, None], other=0)
        lo = (p & 0xF).to(tl.float32)
        hi = (p >> 4).to(tl.float32)
        prod = lo * xe[None, :] + hi * xo[None, :]                          # [BLOCK_N, KB]
        pg = tl.sum(tl.reshape(prod, [BLOCK_N, G, GB]), axis=2)             # [BLOCK_N, G]
        xs = tl.sum(tl.reshape(xe + xo, [G, GB]), axis=1)                   # [G]
        gi = k0 // GROUP_SIZE + tl.arange(0, G)
        s = tl.load(s_ptr + rows[:, None] * n_groups + gi[None, :], mask=row_mask[:, None], other=0).to(tl.float32)
        m = tl.load(m_ptr + rows[:, None] * n_groups + gi[None, :], mask=row_mask[:, None], other=0).to(tl.float32)
        acc += tl.sum(pg * s + xs[None, :] * m, axis=1)
    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=row_mask)


def int4_gemv(x, packed, scale, mn):
    """x [1, K] bf16; packed [N, K/2] uint8; scale, mn [N, K/GROUP] bf16 -> [1, N] bf16."""
    N, K = packed.shape[0], packed.shape[1] * 2
    y = torch.empty(1, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _int4_gemv_kernel[grid](x, packed, scale, mn, y, N, K, GROUP_SIZE=GROUP)
    return y


@candidate("triton-int4", 4.25)
class TritonInt4:
    def pack(self, w):
        q, s, m = _rtn(w)
        packed = (q[:, ::2] | (q[:, 1::2] << 4)).contiguous()
        return packed, s, m

    def dequant(self, p):
        packed, s, m = p
        q = torch.stack([packed & 0xF, packed >> 4], -1).view(packed.shape[0], -1)
        return _dequant(q, s, m)

    def run(self, p, x):
        return int4_gemv(x, *p)

    def nbytes(self, p):
        return sum(t.numel() * t.element_size() for t in p)
