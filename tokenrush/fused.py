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
def _add_rmsnorm_kernel(x_ptr, h_ptr, w_ptr, xo_ptr, y_ptr, N, eps, HAS_H: tl.constexpr, NSPLIT: tl.constexpr,
                        BLOCK: tl.constexpr):
    """h may be NSPLIT fp32 partial rows (split-K GEMV output): they are summed and
    rounded to bf16 first, which is the value a plain GEMV would have produced."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < N
    T = tl.num_programs(0)
    x = tl.load(x_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)
    if HAS_H:
        h = tl.load(h_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)                 # split 0
        for sidx in tl.static_range(1, NSPLIT):
            h += tl.load(h_ptr + (sidx * T + row) * N + offs, mask=m, other=0.0).to(tl.float32)
        x = x + h.to(tl.bfloat16).to(tl.float32)
        x = x.to(tl.bfloat16)                       # the residual stream is bf16
        tl.store(xo_ptr + row * N + offs, x, mask=m)
        x = x.to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    y = x * tl.math.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
    y = y * (1.0 + w)
    tl.store(y_ptr + row * N + offs, y.to(y_ptr.dtype.element_ty), mask=m)


def add_rmsnorm(x: torch.Tensor, h, weight: torch.Tensor, eps: float):
    """x [T, N] bf16; h None, [T, N] bf16, or [S, T, N] fp32 partials ->
    (x + h as bf16, rmsnorm(x + h)); with h None -> (x, rmsnorm(x))."""
    T, N = x.shape
    y = torch.empty_like(x)
    xo = torch.empty_like(x) if h is not None else x
    nsplit = 1 if h is None or h.dim() == 2 else h.shape[0]
    _add_rmsnorm_kernel[(T,)](x, h if h is not None else x, weight, xo, y, N, eps,
                              HAS_H=h is not None, NSPLIT=nsplit, BLOCK=triton.next_power_of_2(N), num_warps=8)
    return xo, y


# ------------------------------------------------------------- silu * mul


@triton.jit
def _silu_mul_kernel(gu_ptr, y_ptr, F, BLOCK: tl.constexpr):
    """Row t of gate_up is [gate (F) | up (F)]; one program per (row, block)."""
    t = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < F
    g = tl.load(gu_ptr + t * 2 * F + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + t * 2 * F + F + offs, mask=m, other=0.0).to(tl.float32)
    y = g / (1.0 + tl.exp(-g)) * u
    tl.store(y_ptr + t * F + offs, y.to(y_ptr.dtype.element_ty), mask=m)


def silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """gate_up [T, 2F] -> silu(gate) * up [T, F], one launch."""
    T, F2 = gate_up.shape
    F = F2 // 2
    if T > 8:
        g, u = gate_up.chunk(2, -1)
        return torch.nn.functional.silu(g) * u
    y = torch.empty(T, F, device=gate_up.device, dtype=gate_up.dtype)
    _silu_mul_kernel[(triton.cdiv(F, 2048), T)](gate_up.contiguous(), y, F, BLOCK=2048)
    return y


# ------------------------------------------------------- fused GDN step


@triton.jit
def _softplus(x):
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


@triton.jit
def _conv_ring(x_ptr, ring_ptr, w_ptr, ch, c0, c1, c2, c3, R: tl.constexpr):
    """Causal conv (kernel 4) + silu for the channels `ch` of the new input at x_ptr,
    reading the three older inputs from ring columns c0..c2 and writing the new one
    into c3 (R columns per channel). Returns fp32."""
    xn = tl.load(x_ptr + ch).to(tl.float32)
    s0 = tl.load(ring_ptr + ch * R + c0).to(tl.float32)
    s1 = tl.load(ring_ptr + ch * R + c1).to(tl.float32)
    s2 = tl.load(ring_ptr + ch * R + c2).to(tl.float32)
    w0 = tl.load(w_ptr + ch * 4 + 0).to(tl.float32)
    w1 = tl.load(w_ptr + ch * 4 + 1).to(tl.float32)
    w2 = tl.load(w_ptr + ch * 4 + 2).to(tl.float32)
    w3 = tl.load(w_ptr + ch * 4 + 3).to(tl.float32)
    y = s0 * w0 + s1 * w1 + s2 * w2 + xn * w3
    tl.store(ring_ptr + ch * R + c3, xn.to(ring_ptr.dtype.element_ty))
    y = y.to(tl.bfloat16).to(tl.float32)            # HF's conv returns bf16; silu is applied to that
    y = y / (1.0 + tl.exp(-y))
    return y.to(tl.bfloat16).to(tl.float32)         # and its silu output is bf16 too


@triton.jit
def _gdn_step_fused_kernel(qkvz_ptr, x_ptr, ba_w_ptr, ring_ptr, convw_ptr, A_ptr, dtb_ptr, rec_ptr, slot_ptr,
                           o_ptr, y_ptr, normw_ptr, pos_ptr, scale, eps, rec_slot_stride,
                           M: tl.constexpr, R: tl.constexpr,
                           H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                           BV: tl.constexpr, QK_DIM: tl.constexpr, VAL_DIM: tl.constexpr,
                           HIDDEN: tl.constexpr, BLOCK_H: tl.constexpr):
    """M consecutive tokens (M == 1 is plain decode) for one (V head, BV-column
    block). qkvz is the in_proj output [M, 2*QK_DIM + 2*VAL_DIM] (q | k | v | z per
    row), x the layer input [M, HIDDEN] (for the b/a gate dot products, fp32
    accumulate, one rounding to bf16 as cuBLAS did). The recurrent state is read
    from slot *slot_ptr of rec_ptr (slot stride rec_slot_stride) and the state
    after token i is written to slot i, so a verify step can commit any prefix by
    naming its slot. The ring column for token i is (pos + i) mod R. When
    BV == V the program owns a whole head and also applies the gated norm and
    output gate, writing y; otherwise it writes the raw head output o."""
    pid = tl.program_id(0)
    NV: tl.constexpr = V // BV
    i_v = pid % NV
    i_hv = pid // NV
    i_h = i_hv // (HV // H)
    pos = tl.load(pos_ptr)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    QKVZ: tl.constexpr = 2 * QK_DIM + 2 * VAL_DIM
    p_tile = i_hv * K * V + o_k[:, None] * V + o_v[None, :]
    h = tl.load(rec_ptr + tl.load(slot_ptr) * rec_slot_stride + p_tile)
    # b and a gate projections for all M tokens in one vectorized pass (they do not
    # depend on the recurrence), fp32 accumulate, one rounding to bf16 as cuBLAS did
    MP: tl.constexpr = 8
    mi = tl.arange(0, MP)
    mmask = mi < M
    b_all = tl.zeros([MP], dtype=tl.float32)
    a_all = tl.zeros([MP], dtype=tl.float32)
    for h0 in tl.static_range(0, HIDDEN, BLOCK_H):
        hh = h0 + tl.arange(0, BLOCK_H)
        xb = tl.load(x_ptr + mi[:, None] * HIDDEN + hh[None, :], mask=mmask[:, None], other=0.0).to(tl.float32)   # [MP, BLOCK_H]
        wb = tl.load(ba_w_ptr + i_hv * HIDDEN + hh).to(tl.float32)
        wa = tl.load(ba_w_ptr + (HV + i_hv) * HIDDEN + hh).to(tl.float32)
        b_all += tl.sum(xb * wb[None, :], axis=1)
        a_all += tl.sum(xb * wa[None, :], axis=1)
    b_all = b_all.to(tl.bfloat16).to(tl.float32)
    a_all = a_all.to(tl.bfloat16).to(tl.float32)
    A_h = tl.load(A_ptr + i_hv)
    dtb_h = tl.load(dtb_ptr + i_hv)
    for i in tl.static_range(M):
        c3 = (pos + i) % R
        c0 = (pos + i + R - 3) % R
        c1 = (pos + i + R - 2) % R
        c2 = (pos + i + R - 1) % R
        row = qkvz_ptr + i * QKVZ
        q = _conv_ring(row, ring_ptr, convw_ptr, i_h * K + o_k, c0, c1, c2, c3, R)
        k = _conv_ring(row, ring_ptr, convw_ptr, QK_DIM + i_h * K + o_k, c0, c1, c2, c3, R)
        v = _conv_ring(row, ring_ptr, convw_ptr, 2 * QK_DIM + i_hv * V + o_v, c0, c1, c2, c3, R)
        b = tl.sum(tl.where(mi == i, b_all, 0.0))
        a = tl.sum(tl.where(mi == i, a_all, 0.0))
        beta = (1.0 / (1.0 + tl.exp(-b))).to(tl.bfloat16).to(tl.float32)   # HF: b.sigmoid() in bf16
        g = A_h * _softplus(a + dtb_h)
        q = q / tl.sqrt(tl.sum(q * q) + 1e-6) * scale
        k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        h = h * tl.exp(g)
        v = beta * (v - tl.sum(h * k[:, None], 0))
        h += k[:, None] * v[None, :]
        tl.store(rec_ptr + i * rec_slot_stride + p_tile, h)
        o = tl.sum(h * q[:, None], 0)                                   # [BV] fp32
        if BV == V:
            ob = o.to(tl.bfloat16).to(tl.float32)                       # the ops path rounds o to bf16 here
            var = tl.sum(ob * ob) / V
            n = (ob * tl.math.rsqrt(var + eps)).to(tl.bfloat16)
            w = tl.load(normw_ptr + o_v)
            n = (w * n).to(tl.float32)
            z = tl.load(row + 2 * QK_DIM + VAL_DIM + i_hv * V + o_v).to(tl.float32)
            n = n * (z / (1.0 + tl.exp(-z)))
            tl.store(y_ptr + i * VAL_DIM + i_hv * V + o_v, n.to(y_ptr.dtype.element_ty))
        else:
            tl.store(o_ptr + i * VAL_DIM + i_hv * V + o_v, o.to(o_ptr.dtype.element_ty))


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


def gdn_step_fused(qkvz, x, ba_w, ring, conv_w, A, dt_bias, rec, slot, norm_w, pos_t, cfg, eps, BV=128):
    """M-token GDN step (M = qkvz.shape[0] <= 8): qkvz [M, 2*qk+2*val] bf16 (in_proj
    output), x [M, hidden] the layer input, ba_w [2*HV, hidden], rec [n_slots, HV, K, V]
    with the committed state in slot *slot -> y [M, val_dim] bf16, ready for out_proj.
    Updates the ring in place and writes rec slots 0..M-1."""
    M = qkvz.shape[0]
    H, HV, K, V = cfg.gdn_k_heads, cfg.gdn_v_heads, cfg.gdn_k_dim, cfg.gdn_v_dim
    QK, VAL = cfg.gdn_qk_dim, cfg.gdn_val_dim
    assert rec.shape[0] >= M, "not enough recurrent-state slots for this many tokens"
    y = torch.empty(M, VAL, device=qkvz.device, dtype=qkvz.dtype)
    o = y if BV == V else torch.empty_like(y)
    _gdn_step_fused_kernel[(HV * (V // BV),)](
        qkvz, x, ba_w, ring, conv_w, A, dt_bias, rec, slot, o, y, norm_w, pos_t, K ** -0.5, eps, rec.stride(0),
        M=M, R=ring.shape[1], H=H, HV=HV, K=K, V=V, BV=BV, QK_DIM=QK, VAL_DIM=VAL, HIDDEN=cfg.hidden, BLOCK_H=1024,
        num_warps=4 if BV == V else 1)
    if BV != V:
        for i in range(M):
            _gated_norm_kernel[(HV,)](o[i], qkvz[i, 2 * QK + VAL:], norm_w, y[i], eps, V=V, num_warps=1)
    return y


# ------------------------------------------------------ attention: prep


FP8_MAX = 448.0            # float8_e4m3fn


@triton.jit
def _attn_prep_kernel(qkv_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, pos_ptr, k_cache, v_cache, ks_ptr, vs_ptr, q_out, eps,
                      HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, RD: tl.constexpr, MAXLEN: tl.constexpr,
                      FP8: tl.constexpr):
    """One program per head. Heads 0..HQ-1: RMSNorm + rope the query, write q_out.
    Heads HQ..HQ+HKV-1: RMSNorm + rope the key and write it and the value into the
    cache at position pos. qkv is the fused projection [HQ*2D (q|gate per head) | HKV*D | HKV*D].
    Rounding mirrors ops.rmsnorm / ops.apply_rope in bf16."""
    pid = tl.program_id(0)
    i = tl.program_id(1)                                          # token index within the step
    pos = tl.load(pos_ptr) + i
    QKV: tl.constexpr = HQ * 2 * D + 2 * HKV * D
    d = tl.arange(0, D)
    HALF: tl.constexpr = RD // 2
    r = tl.arange(0, HALF)
    if pid < HQ:
        src = qkv_ptr + i * QKV + pid * 2 * D
        w_ptr = qn_ptr
    else:
        src = qkv_ptr + i * QKV + HQ * 2 * D + (pid - HQ) * D
        w_ptr = kn_ptr
    x = tl.load(src + d).to(tl.float32)
    var = tl.sum(x * x) / D
    x = x * tl.math.rsqrt(var + eps) * (1.0 + tl.load(w_ptr + d).to(tl.float32))
    x = x.to(tl.bfloat16)
    # rope on the first RD dims: [x1 | x2] -> [x1*c - x2*s | x2*c + x1*s], each product rounded to bf16
    x1 = tl.load(src + r).to(tl.float32)                         # reload the two halves as vectors
    x2 = tl.load(src + HALF + r).to(tl.float32)
    inv = tl.math.rsqrt(var + eps)
    w1 = 1.0 + tl.load(w_ptr + r).to(tl.float32)
    w2 = 1.0 + tl.load(w_ptr + HALF + r).to(tl.float32)
    x1 = (x1 * inv * w1).to(tl.bfloat16).to(tl.float32)
    x2 = (x2 * inv * w2).to(tl.bfloat16).to(tl.float32)
    c = tl.load(cos_ptr + pos * RD + r).to(tl.float32)           # cos == cos[HALF + r]
    s = tl.load(sin_ptr + pos * RD + r).to(tl.float32)
    o1 = ((x1 * c).to(tl.bfloat16).to(tl.float32) + (-x2 * s).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    o2 = ((x2 * c).to(tl.bfloat16).to(tl.float32) + (x1 * s).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    if pid < HQ:
        qo = q_out + (i * HQ + pid) * D
        tl.store(qo + d, x)                                       # dims RD.. keep the normed value
        tl.store(qo + r, o1)
        tl.store(qo + HALF + r, o2)
    else:
        j = pid - HQ
        koff = (j * MAXLEN + pos) * D
        if FP8:
            # one scale per (head, position): amax over the roped vector / 448
            xr = tl.where(d >= RD, x.to(tl.float32), 0.0)
            amax = tl.maximum(tl.max(tl.abs(xr)), tl.maximum(tl.max(tl.abs(o1.to(tl.float32))), tl.max(tl.abs(o2.to(tl.float32)))))
            sc = tl.maximum(amax, 1e-6) / 448.0
            tl.store(ks_ptr + j * MAXLEN + pos, sc)
            tl.store(k_cache + koff + d, (x.to(tl.float32) / sc).to(k_cache.dtype.element_ty))
            tl.store(k_cache + koff + r, (o1.to(tl.float32) / sc).to(k_cache.dtype.element_ty))
            tl.store(k_cache + koff + HALF + r, (o2.to(tl.float32) / sc).to(k_cache.dtype.element_ty))
        else:
            tl.store(k_cache + koff + d, x)
            tl.store(k_cache + koff + r, o1)
            tl.store(k_cache + koff + HALF + r, o2)
        v = tl.load(qkv_ptr + i * QKV + HQ * 2 * D + HKV * D + j * D + d)
        if FP8:
            vf = v.to(tl.float32)
            sv = tl.maximum(tl.max(tl.abs(vf)), 1e-6) / 448.0
            tl.store(vs_ptr + j * MAXLEN + pos, sv)
            tl.store(v_cache + koff + d, (vf / sv).to(v_cache.dtype.element_ty))
        else:
            tl.store(v_cache + koff + d, v)


def attn_prep(qkv, q_norm_w, k_norm_w, cos, sin, pos_t, k_cache, v_cache, cfg, k_scale=None, v_scale=None):
    """qkv [1, HQ*2D + 2*HKV*D] -> q [HQ*D] normed + roped; writes K, V (and their fp8
    scales when the cache is fp8) at pos_t."""
    HQ, HKV, D = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    M = qkv.shape[0]
    fp8 = k_cache.dtype == torch.float8_e4m3fn
    q = torch.empty(M, HQ * D, device=qkv.device, dtype=qkv.dtype)
    _attn_prep_kernel[(HQ + HKV, M)](qkv, q_norm_w, k_norm_w, cos, sin, pos_t, k_cache, v_cache,
                                     k_scale if fp8 else q, v_scale if fp8 else q, q, cfg.eps,
                                     HQ=HQ, HKV=HKV, D=D, RD=cfg.rotary_dim, MAXLEN=k_cache.shape[1], FP8=fp8, num_warps=1)
    return q


def kv_write_prefill(k, v, state, slot, pos):
    """Prefill-side cache write. k, v [Hkv, T, D] bf16 -> cache rows pos..pos+T-1, quantized
    to fp8 with per-(head, position) scales when the cache is fp8 (the same convention as
    the prep kernel)."""
    T = k.shape[1]
    if not state.fp8:
        state.k[slot, :, pos:pos + T] = k
        state.v[slot, :, pos:pos + T] = v
        return
    for src, cache, scale in ((k, state.k, state.k_scale), (v, state.v, state.v_scale)):
        f = src.float()
        sc = f.abs().amax(-1).clamp(min=1e-6) / FP8_MAX                     # [Hkv, T]
        cache[slot, :, pos:pos + T] = (f / sc[..., None]).to(torch.float8_e4m3fn)
        scale[slot, :, pos:pos + T] = sc


# ------------------------------------------- attention: flash-decoding


@triton.jit
def _attn_split_kernel(q_ptr, k_cache, v_cache, ks_ptr, vs_ptr, pos_ptr, m_ptr, l_ptr, acc_ptr, scale,
                       HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, MAXLEN: tl.constexpr,
                       NSPLIT: tl.constexpr, BLOCK_N: tl.constexpr, ROWS: tl.constexpr, FP8: tl.constexpr,
                       M: tl.constexpr):
    """One program per (kv head, split). The G = HQ // HKV query heads of the kv
    head are padded to ROWS rows for tensor-core dots. Keys 0..pos are live;
    the split's block range is derived from pos, so the grid is static."""
    pid = tl.program_id(0)
    j = pid // NSPLIT
    s = pid % NSPLIT
    G: tl.constexpr = HQ // HKV
    pos = tl.load(pos_ptr)
    L = pos + M                                                   # live keys: the cache plus the M new ones
    n_blocks = (L + BLOCK_N - 1) // BLOCK_N
    per_split = (n_blocks + NSPLIT - 1) // NSPLIT
    b0 = s * per_split
    b1 = tl.minimum(b0 + per_split, n_blocks)
    rows = tl.arange(0, ROWS)
    d = tl.arange(0, D)
    qi = rows // G                                                # query index of each row
    hq = j * G + rows % G                                         # query head of each row
    rmask = rows < G * M
    q = tl.load(q_ptr + (qi * HQ + hq)[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0)   # [ROWS, D] bf16
    qlimit = pos + qi                                             # row sees keys <= pos + query index
    m_i = tl.full([ROWS], float("-inf"), tl.float32)
    l_i = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, D], tl.float32)
    kb = k_cache + j * MAXLEN * D
    vb = v_cache + j * MAXLEN * D
    for blk in range(b0, b1):
        kidx = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        kmask = kidx < L
        k = tl.load(kb + kidx[:, None] * D + d[None, :], mask=kmask[:, None], other=0.0)            # [BLOCK_N, D]
        if FP8:
            ksc = tl.load(ks_ptr + j * MAXLEN + kidx, mask=kmask, other=0.0)
            k = (k.to(tl.float32) * ksc[:, None]).to(tl.bfloat16)
        sc = tl.dot(q, tl.trans(k)) * scale                                                          # [ROWS, BLOCK_N] fp32
        sc = tl.where(kmask[None, :] & (kidx[None, :] <= qlimit[:, None]), sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(sc - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vb + kidx[:, None] * D + d[None, :], mask=kmask[:, None], other=0.0)
        if FP8:
            vsc = tl.load(vs_ptr + j * MAXLEN + kidx, mask=kmask, other=0.0)
            p = p * vsc[None, :]                                    # fold the row scale into the probabilities
            v = v.to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    h = qi * HQ + hq                                              # (query, head) slot in the partial buffers
    tl.store(m_ptr + h * NSPLIT + s, m_i, mask=rmask)
    tl.store(l_ptr + h * NSPLIT + s, l_i, mask=rmask)
    tl.store(acc_ptr + (h * NSPLIT + s)[:, None] * D + d[None, :], acc, mask=rmask[:, None])


@triton.jit
def _attn_reduce_kernel(m_ptr, l_ptr, acc_ptr, qkv_ptr, out_ptr,
                        D: tl.constexpr, NSPLIT: tl.constexpr, HQ: tl.constexpr, QKV: tl.constexpr):
    """One program per query head: combine the splits, normalize, apply the
    sigmoid output gate (HF: attn_output * sigmoid(gate), in bf16)."""
    h = tl.program_id(0)                                          # = query * HQ + head
    s = tl.arange(0, NSPLIT)
    d = tl.arange(0, D)
    m = tl.load(m_ptr + h * NSPLIT + s)
    l = tl.load(l_ptr + h * NSPLIT + s)
    m_max = tl.max(m, 0)
    w = tl.exp(m - m_max)                                         # -inf splits -> 0
    l_tot = tl.sum(w * l, 0)
    acc = tl.load(acc_ptr + (h * NSPLIT + s)[:, None] * D + d[None, :])   # [NSPLIT, D]
    o = tl.sum(acc * w[:, None], 0) / l_tot
    o = o.to(tl.bfloat16).to(tl.float32)
    i = h // HQ
    hh = h % HQ
    gate = tl.load(qkv_ptr + i * QKV + hh * 2 * D + D + d).to(tl.float32)
    g = (1.0 / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + h * D + d, (o * g).to(out_ptr.dtype.element_ty))


def attn_decode_fused(q, qkv, k_cache, v_cache, pos_t, cfg, k_scale=None, v_scale=None, NSPLIT=32, BLOCK_N=32):
    """q [M, HQ*D] (from attn_prep) for the M new tokens at pos.., qkv [M, ..] (for the gates)
    -> gated attention output [M, HQ*D]; query i sees cache rows <= pos + i."""
    HQ, HKV, D = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    M = q.shape[0]
    G = HQ // HKV
    rows = max(16, triton.next_power_of_2(G * M))
    dev = q.device
    fp8 = k_cache.dtype == torch.float8_e4m3fn
    m = torch.empty(M * HQ, NSPLIT, device=dev, dtype=torch.float32)
    l = torch.empty_like(m)
    acc = torch.empty(M * HQ, NSPLIT, D, device=dev, dtype=torch.float32)
    out = torch.empty(M, HQ * D, device=dev, dtype=q.dtype)
    _attn_split_kernel[(HKV * NSPLIT,)](q, k_cache, v_cache, k_scale if fp8 else m, v_scale if fp8 else m, pos_t,
                                        m, l, acc, D ** -0.5,
                                        HQ=HQ, HKV=HKV, D=D, MAXLEN=k_cache.shape[1], NSPLIT=NSPLIT,
                                        BLOCK_N=BLOCK_N, ROWS=rows, FP8=fp8, M=M, num_warps=4, num_stages=3)
    _attn_reduce_kernel[(M * HQ,)](m, l, acc, qkv, out, D=D, NSPLIT=NSPLIT, HQ=HQ, QKV=qkv.shape[1], num_warps=1)
    return out


# ------------------------------------------------- attention: prefill


@triton.jit
def _attn_prefill_kernel(q_ptr, k_cache, v_cache, ks_ptr, vs_ptr, o_ptr, pos, T, scale,
                         HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, MAXLEN: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, FP8: tl.constexpr):
    """Causal flash attention for a prefill chunk of T queries at positions pos..pos+T-1
    over the cache rows 0..pos+T-1. q, o are [T, HQ, D]; one program per (query block,
    query head); the kv head is the query head's group."""
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    j = h // (HQ // HKV)
    m = pid_m * BM + tl.arange(0, BM)
    qm = m < T
    d = tl.arange(0, D)
    q = tl.load(q_ptr + (m[:, None] * HQ + h) * D + d[None, :], mask=qm[:, None], other=0.0)      # [BM, D]
    qpos = pos + m
    hi = pos + tl.minimum(pid_m * BM + BM, T)                       # keys strictly below hi can be seen by this block
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    kb = k_cache + j * MAXLEN * D
    vb = v_cache + j * MAXLEN * D
    for n0 in range(0, hi, BN):
        kidx = n0 + tl.arange(0, BN)
        kmask = kidx < hi
        k = tl.load(kb + kidx[:, None] * D + d[None, :], mask=kmask[:, None], other=0.0)
        if FP8:
            ksc = tl.load(ks_ptr + j * MAXLEN + kidx, mask=kmask, other=0.0)
            k = (k.to(tl.float32) * ksc[:, None]).to(tl.bfloat16)
        sc = tl.dot(q, tl.trans(k)) * scale                                                         # [BM, BN]
        allowed = (kidx[None, :] <= qpos[:, None]) & kmask[None, :]
        sc = tl.where(allowed, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(sc - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vb + kidx[:, None] * D + d[None, :], mask=kmask[:, None], other=0.0)
        if FP8:
            vsc = tl.load(vs_ptr + j * MAXLEN + kidx, mask=kmask, other=0.0)
            p = p * vsc[None, :]
            v = v.to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(o_ptr + (m[:, None] * HQ + h) * D + d[None, :], o.to(o_ptr.dtype.element_ty), mask=qm[:, None])


def attn_prefill_fused(q, k_cache, v_cache, pos, cfg, k_scale=None, v_scale=None, BM=64, BN=32):
    """q [T, HQ, D] bf16 (normed, roped) for positions pos.. ; cache holds rows 0..pos+T-1
    -> o [T, HQ, D] bf16."""
    T, HQ, D = q.shape
    fp8 = k_cache.dtype == torch.float8_e4m3fn
    o = torch.empty_like(q)
    dummy = o
    _attn_prefill_kernel[(triton.cdiv(T, BM), HQ)](
        q, k_cache, v_cache, k_scale if fp8 else dummy, v_scale if fp8 else dummy, o, pos, T, D ** -0.5,
        HQ=HQ, HKV=cfg.n_kv_heads, D=D, MAXLEN=k_cache.shape[1], BM=BM, BN=BN, FP8=fp8, num_warps=8, num_stages=2)
    return o
