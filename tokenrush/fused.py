"""Phase 2 fused kernels for the decode step. Each replaces a chain of small
torch ops in `ops.py` / `model.py` with one launch, and is differential-tested
against that chain (tests/test_fused.py), which is itself verified against HF.

  add_rmsnorm      : x + h, then RMSNorm with the (1 + w) gain          (11 kernels -> 1)
  gdn_step_fused   : conv step, beta/g gating, delta-rule state update,
                     gated RMSNorm and output gate for one GDN layer    (22 kernels -> 1 or 2)
  silu_mul         : silu(gate) * up                                    (2 -> 1)

The GDN conv state is a 4-column ring per channel, column = position mod 4,
so a program can write the new input into its column while other programs
still read the three older ones (nobody reads the column being written).
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------- add + rmsnorm


@triton.jit
def _add_rmsnorm_kernel(x_ptr, h_ptr, w_ptr, xo_ptr, y_ptr, N, eps, HAS_H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)
    if HAS_H:
        x = x + tl.load(h_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)
        x = x.to(tl.bfloat16)                       # the residual stream is bf16
        tl.store(xo_ptr + row * N + offs, x, mask=m)
        x = x.to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    y = x * tl.math.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
    y = y * (1.0 + w)
    tl.store(y_ptr + row * N + offs, y.to(y_ptr.dtype.element_ty), mask=m)


def add_rmsnorm(x: torch.Tensor, h, weight: torch.Tensor, eps: float):
    """x [T, N] bf16, h [T, N] bf16 or None -> (x + h as bf16, rmsnorm(x + h)) ; with h None -> (x, rmsnorm(x))."""
    T, N = x.shape
    y = torch.empty_like(x)
    xo = torch.empty_like(x) if h is not None else x
    _add_rmsnorm_kernel[(T,)](x, h if h is not None else x, weight, xo, y, N, eps,
                              HAS_H=h is not None, BLOCK=triton.next_power_of_2(N), num_warps=8)
    return xo, y


# ------------------------------------------------------------- silu * mul


@triton.jit
def _silu_mul_kernel(g_ptr, u_ptr, y_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
    y = g / (1.0 + tl.exp(-g)) * u
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=m)


def silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """gate_up [T, 2F] -> silu(gate) * up [T, F]. Rows are contiguous so a row's
    gate and up halves are two strided views."""
    T, F2 = gate_up.shape
    F = F2 // 2
    y = torch.empty(T, F, device=gate_up.device, dtype=gate_up.dtype)
    if T != 1:
        g, u = gate_up.chunk(2, -1)
        return torch.nn.functional.silu(g) * u
    _silu_mul_kernel[(triton.cdiv(F, 2048),)](gate_up, gate_up[:, F:], y, F, BLOCK=2048)
    return y


# ------------------------------------------------------- fused GDN step


@triton.jit
def _softplus(x):
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


@triton.jit
def _conv_ring(x_ptr, ring_ptr, w_ptr, ch, c0, c1, c2, c3):
    """Causal conv (kernel 4) + silu for the channels `ch` of the new input at x_ptr,
    reading the three older inputs from ring columns c0..c2 and writing the new one
    into c3. Returns fp32."""
    xn = tl.load(x_ptr + ch).to(tl.float32)
    s0 = tl.load(ring_ptr + ch * 4 + c0).to(tl.float32)
    s1 = tl.load(ring_ptr + ch * 4 + c1).to(tl.float32)
    s2 = tl.load(ring_ptr + ch * 4 + c2).to(tl.float32)
    w0 = tl.load(w_ptr + ch * 4 + 0).to(tl.float32)
    w1 = tl.load(w_ptr + ch * 4 + 1).to(tl.float32)
    w2 = tl.load(w_ptr + ch * 4 + 2).to(tl.float32)
    w3 = tl.load(w_ptr + ch * 4 + 3).to(tl.float32)
    y = s0 * w0 + s1 * w1 + s2 * w2 + xn * w3
    tl.store(ring_ptr + ch * 4 + c3, xn.to(ring_ptr.dtype.element_ty))
    y = y.to(tl.bfloat16).to(tl.float32)            # HF's conv returns bf16; silu is applied to that
    y = y / (1.0 + tl.exp(-y))
    return y.to(tl.bfloat16).to(tl.float32)         # and its silu output is bf16 too


@triton.jit
def _gdn_step_fused_kernel(qkvz_ptr, ba_ptr, ring_ptr, convw_ptr, A_ptr, dtb_ptr, rec_ptr, o_ptr, y_ptr,
                           normw_ptr, pos_ptr, scale, eps,
                           H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                           BV: tl.constexpr, QK_DIM: tl.constexpr, VAL_DIM: tl.constexpr):
    """One program per (V head, BV-column block). qkvz is the in_proj output
    [2*QK_DIM + 2*VAL_DIM]: q | k | v | z. ba is [2*HV]: b | a. When BV == V the
    program owns a whole head and also applies the gated norm and output gate,
    writing y; otherwise it writes the raw head output o for a second kernel."""
    pid = tl.program_id(0)
    NV: tl.constexpr = V // BV
    i_v = pid % NV
    i_hv = pid // NV
    i_h = i_hv // (HV // H)
    pos = tl.load(pos_ptr)
    c3 = pos % 4
    c0 = (pos + 1) % 4
    c1 = (pos + 2) % 4
    c2 = (pos + 3) % 4
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    q = _conv_ring(qkvz_ptr, ring_ptr, convw_ptr, i_h * K + o_k, c0, c1, c2, c3)
    k = _conv_ring(qkvz_ptr, ring_ptr, convw_ptr, QK_DIM + i_h * K + o_k, c0, c1, c2, c3)
    v = _conv_ring(qkvz_ptr, ring_ptr, convw_ptr, 2 * QK_DIM + i_hv * V + o_v, c0, c1, c2, c3)
    b = tl.load(ba_ptr + i_hv).to(tl.float32)
    a = tl.load(ba_ptr + HV + i_hv).to(tl.float32)
    beta = (1.0 / (1.0 + tl.exp(-b))).to(tl.bfloat16).to(tl.float32)   # HF: b.sigmoid() in bf16
    g = tl.load(A_ptr + i_hv) * _softplus(a + tl.load(dtb_ptr + i_hv))
    q = q / tl.sqrt(tl.sum(q * q) + 1e-6) * scale
    k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
    p_h = rec_ptr + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
    h = tl.load(p_h) * tl.exp(g)
    v = beta * (v - tl.sum(h * k[:, None], 0))
    h += k[:, None] * v[None, :]
    tl.store(p_h, h)
    o = tl.sum(h * q[:, None], 0)                                   # [BV] fp32
    if BV == V:
        ob = o.to(tl.bfloat16).to(tl.float32)                       # the ops path rounds o to bf16 here
        var = tl.sum(ob * ob) / V
        n = (ob * tl.math.rsqrt(var + eps)).to(tl.bfloat16)
        w = tl.load(normw_ptr + o_v)
        n = (w * n).to(tl.float32)
        z = tl.load(qkvz_ptr + 2 * QK_DIM + VAL_DIM + i_hv * V + o_v).to(tl.float32)
        n = n * (z / (1.0 + tl.exp(-z)))
        tl.store(y_ptr + i_hv * V + o_v, n.to(y_ptr.dtype.element_ty))
    else:
        tl.store(o_ptr + i_hv * V + o_v, o.to(o_ptr.dtype.element_ty))


@triton.jit
def _gated_norm_kernel(o_ptr, z_ptr, w_ptr, y_ptr, eps, V: tl.constexpr):
    """Gated RMSNorm + silu gate for one V head, matching ops.gated_rmsnorm's dtype order."""
    i_hv = tl.program_id(0)
    o_v = tl.arange(0, V)
    o = tl.load(o_ptr + i_hv * V + o_v).to(tl.float32)
    var = tl.sum(o * o) / V
    n = (o * tl.math.rsqrt(var + eps)).to(tl.bfloat16)
    w = tl.load(w_ptr + o_v)
    n = (w * n).to(tl.float32)
    z = tl.load(z_ptr + i_hv * V + o_v).to(tl.float32)
    n = n * (z / (1.0 + tl.exp(-z)))
    tl.store(y_ptr + i_hv * V + o_v, n.to(y_ptr.dtype.element_ty))


def gdn_step_fused(qkvz, ba, ring, conv_w, A, dt_bias, rec, norm_w, pos_t, cfg, eps, BV=128):
    """One GDN decode step: qkvz [1, 2*qk+2*val] bf16 (in_proj output), ba [1, 2*HV]
    -> y [1, val_dim] bf16, ready for out_proj. Updates ring and rec in place."""
    H, HV, K, V = cfg.gdn_k_heads, cfg.gdn_v_heads, cfg.gdn_k_dim, cfg.gdn_v_dim
    QK, VAL = cfg.gdn_qk_dim, cfg.gdn_val_dim
    y = torch.empty(1, VAL, device=qkvz.device, dtype=qkvz.dtype)
    o = y if BV == V else torch.empty_like(y)
    _gdn_step_fused_kernel[(HV * (V // BV),)](
        qkvz, ba, ring, conv_w, A, dt_bias, rec, o, y, norm_w, pos_t, K ** -0.5, eps,
        H=H, HV=HV, K=K, V=V, BV=BV, QK_DIM=QK, VAL_DIM=VAL, num_warps=4 if BV == V else 1)
    if BV != V:
        _gated_norm_kernel[(HV,)](o, qkvz[:, 2 * QK + VAL:], norm_w, y, eps, V=V, num_warps=1)
    return y
