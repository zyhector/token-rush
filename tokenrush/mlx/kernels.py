"""Metal kernels the MLX backend owns.

The gated delta rule, one kernel for every case. The CUDA engine keeps K+1
snapshots of the recurrent state so a verify step can commit whichever prefix
was accepted (48 layers x 8 x 3.1 MB written per step); on unified memory at a
fraction of the 5090's bandwidth that write is not free. Here a verify step
writes no state at all: it returns its outputs and the caller keeps the rows'
(k, v, g, beta), a few KB per layer. The next call replays the first n_pend of
those rows into the state before its own rows, so every step reads the state
once and writes it once, whatever was accepted.

State layout is [HV, V, K] fp32 (K, the key dim, contiguous) so the 32 lanes of a
SIMD group read one state row as 128 contiguous floats.
"""
import mlx.core as mx

_GDN_SOURCE = """
    // grid (32, V, HV), threadgroup (32, TG_V, 1): one SIMD group per state row (hv, dv),
    // each lane holds NK = K/32 consecutive keys of it.
    const uint lane = thread_position_in_threadgroup.x;
    const uint dv = thread_position_in_grid.y;
    const uint hv = thread_position_in_grid.z;
    const uint h = hv / (HV / H);
    constexpr int NK = K / 32;
    const uint row = (hv * V + dv) * K + lane * NK;

    float s[NK];
    for (int i = 0; i < NK; ++i) s[i] = state_in[row + i];

    // 1. replay the committed rows of the previous verify step: state only
    const int np = n_pend[0];
    for (int t = 0; t < np; ++t) {
        const float decay = metal::exp(pg[t * HV + hv]);
        const float b = pbeta[t * HV + hv];
        const uint kb = (t * H + h) * K + lane * NK;
        float kv = 0.0f;
        for (int i = 0; i < NK; ++i) { s[i] *= decay; kv += s[i] * pk[kb + i]; }
        kv = simd_sum(kv);
        const float delta = (static_cast<float>(pv[(t * HV + hv) * V + dv]) - kv) * b;
        for (int i = 0; i < NK; ++i) s[i] += pk[kb + i] * delta;
    }

    // 2. this call's rows: outputs, and the state only if they are committed
    const int T = q_shape[0];
    float r[NK];
    for (int i = 0; i < NK; ++i) r[i] = s[i];
    for (int t = 0; t < T; ++t) {
        const float decay = metal::exp(g[t * HV + hv]);
        const float b = beta[t * HV + hv];
        const uint kb = (t * H + h) * K + lane * NK;
        float kv = 0.0f;
        for (int i = 0; i < NK; ++i) { r[i] *= decay; kv += r[i] * k[kb + i]; }
        kv = simd_sum(kv);
        const float delta = (static_cast<float>(v[(t * HV + hv) * V + dv]) - kv) * b;
        float out = 0.0f;
        for (int i = 0; i < NK; ++i) { r[i] += k[kb + i] * delta; out += r[i] * q[kb + i]; }
        out = simd_sum(out);
        if (lane == 0) o[(t * HV + hv) * V + dv] = static_cast<OutT>(out);
    }
    if (COMMIT) {
        for (int i = 0; i < NK; ++i) state_out[row + i] = r[i];
    } else {
        for (int i = 0; i < NK; ++i) state_out[row + i] = s[i];
    }
"""

_gdn_kernel = mx.fast.metal_kernel(
    name="tokenrush_gdn",
    input_names=["q", "k", "v", "g", "beta", "state_in", "pk", "pv", "pg", "pbeta", "n_pend"],
    output_names=["o", "state_out"],
    source=_GDN_SOURCE,
)


def gdn(q, k, v, g, beta, state, pend=None, commit: bool = True, tg_v: int = 4):
    """The gated delta rule over T rows, from `state`.

    q, k: [T, H, K] fp32, already L2-normalized (q also scaled by K**-0.5).
    v: [T, HV, V] (bf16); g: [T, HV] fp32 log-decay; beta: [T, HV] fp32.
    state: [HV, V, K] fp32.
    pend: None or (pk, pv, pg, pbeta, n_pend): rows of an earlier verify call, of
          which the first n_pend (an int32 [1] array, on device) are replayed into the
          state before this call's rows.
    commit: whether this call's rows update the state (prefill, raw decode) or only
            produce outputs (a verify step; the caller keeps the rows as the next pend).
    -> (o [T, HV, V] in v's dtype, new state [HV, V, K] fp32)
    """
    T, H, K = q.shape
    _, HV, V = v.shape
    assert K % 32 == 0 and V % tg_v == 0 and HV % H == 0
    if pend is None:
        pk = mx.zeros((1, H, K), mx.float32)
        pv = mx.zeros((1, HV, V), v.dtype)
        pg = mx.zeros((1, HV), mx.float32)
        pb = mx.zeros((1, HV), mx.float32)
        n_pend = mx.zeros((1,), mx.int32)
    else:
        pk, pv, pg, pb, n_pend = pend
    o, s = _gdn_kernel(
        inputs=[q, k, v, g, beta, state, pk, pv, pg, pb, n_pend],
        template=[("OutT", v.dtype), ("H", H), ("HV", HV), ("K", K), ("V", V), ("COMMIT", int(commit))],
        grid=(32, V, HV),
        threadgroup=(32, tg_v, 1),
        output_shapes=[(T, HV, V), (HV, V, K)],
        output_dtypes=[v.dtype, mx.float32],
    )
    return o, s


def gdn_reference(q, k, v, g, beta, state):
    """The same recurrence in plain MLX ops, fp32, row by row. state [HV, V, K]."""
    T, H, K = q.shape
    HV = v.shape[1]
    rep = HV // H
    s = state
    outs = []
    for t in range(T):
        qt = mx.repeat(q[t], rep, axis=0)                     # [HV, K]
        kt = mx.repeat(k[t], rep, axis=0)
        s = s * mx.exp(g[t])[:, None, None]
        kv = (s * kt[:, None, :]).sum(-1)                     # [HV, V]
        delta = (v[t].astype(mx.float32) - kv) * beta[t][:, None]
        s = s + delta[:, :, None] * kt[:, None, :]
        outs.append((s * qt[:, None, :]).sum(-1))
    return mx.stack(outs).astype(v.dtype), s


# ------------------------------------------------------ M-row int4 GEMV
# mx.quantized_matmul reads the weights once per row for 2 <= M < 16 (measured:
# M = 8 costs 2.8-3.8x M = 1 at our shapes, bench/mac/bandwidth.py), which is
# exactly the verify step's range. This kernel dequantizes each weight once and
# applies it to all M rows: the Mac counterpart of the CUDA engine's Marlin port.
# Layout is MLX's affine 4-bit (= ours byte for byte), group 128.

_QMV_SOURCE = """
    // grid (32*NSG, N/(RPS*NSG)), threadgroup (32*NSG, 1): each SIMD group owns RPS
    // consecutive output rows; each lane walks K in strides of 32*VPT values.
    const uint lane = thread_position_in_threadgroup.x % 32;
    const uint sg = thread_position_in_threadgroup.x / 32;
    const uint row0 = (threadgroup_position_in_grid.y * NSG + sg) * RPS;
    constexpr int KW = K / 8;          // uint32 words per weight row
    constexpr int G = K / 128;         // groups per row
    constexpr int WPT = VPT / 8;       // words per lane per stride

    float acc[M][RPS];
    for (int m = 0; m < M; ++m) for (int r = 0; r < RPS; ++r) acc[m][r] = 0.0f;

    for (int k0 = lane * VPT; k0 < K; k0 += 32 * VPT) {
        // the RPS rows' codes for this stride, dequantized once (scale/bias applied per row below)
        const int g = k0 / 128;
        float qf[RPS][VPT];
        float sc[RPS], bi[RPS];
        for (int r = 0; r < RPS; ++r) {
            const uint row = row0 + r;
            sc[r] = static_cast<float>(scales[row * G + g]);
            bi[r] = static_cast<float>(biases[row * G + g]);
            for (int j = 0; j < WPT; ++j) {
                const uint32_t wv = w[row * KW + k0 / 8 + j];
                for (int i = 0; i < 8; ++i) qf[r][j * 8 + i] = static_cast<float>((wv >> (4 * i)) & 0xF);
            }
        }
        // then every activation row against them: only one x row lives in registers
        for (int m = 0; m < M; ++m) {
            float xv[VPT];
            float xs = 0.0f;
            for (int i = 0; i < VPT; ++i) { xv[i] = static_cast<float>(x[m * K + k0 + i]); xs += xv[i]; }
            for (int r = 0; r < RPS; ++r) {
                float d = 0.0f;
                for (int i = 0; i < VPT; ++i) d += xv[i] * qf[r][i];
                acc[m][r] += sc[r] * d + bi[r] * xs;
            }
        }
    }
    for (int m = 0; m < M; ++m)
        for (int r = 0; r < RPS; ++r) {
            const float v = simd_sum(acc[m][r]);
            if (lane == 0) y[m * N + row0 + r] = static_cast<T>(v);
        }
"""

_qmv_kernel = mx.fast.metal_kernel(
    name="tokenrush_qmv_rows",
    input_names=["x", "w", "scales", "biases"],
    output_names=["y"],
    source=_QMV_SOURCE,
)


def qmv_rows(x, w, scales, biases, rps: int = 4, nsg: int = 2, vpt: int = 8):
    """y = x @ dequant(W).T for M = x.shape[0] rows, each weight read once.
    x [M, K] bf16; w uint32 [N, K/8]; scales, biases [N, K/128]."""
    M, K = x.shape
    N = w.shape[0]
    assert K % (32 * vpt) == 0 and N % (rps * nsg) == 0 and vpt % 8 == 0, (M, K, N)
    return _qmv_kernel(
        inputs=[x, w, scales, biases],
        template=[("T", x.dtype), ("M", M), ("N", N), ("K", K), ("RPS", rps), ("NSG", nsg), ("VPT", vpt)],
        grid=(32 * nsg, N // (rps * nsg), 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]


# ------------------------------------------- M-row int4 GEMM on simdgroup matrices
# The scalar kernel above is ALU-bound at M = 8 (8 FMA per weight at the wall needs
# more fp32 throughput than the GPU has). Here each SIMD group dequantizes a
# 64 (k) x 8 (n) weight tile into threadgroup memory once and multiplies it with
# the 8 x 64 activation tile on simdgroup_float8x8 matrices; M <= 8 rows ride
# along in one 8-row tile (zero-padded).

_QMM8_HEADER = "#include <metal_simdgroup_matrix>\n"

_QMM8_SOURCE = """
    // C^T [8 n x 8 m] = W tile [8 n x 8 k] * x^T [8 k x 8 m], accumulated over K.
    // In Apple's 8x8 SIMD-group layout a lane holds elements (fm, fn) and (fm, fn+1):
    // for the W operand that is two adjacent 4-bit codes of one weight row, one byte,
    // dequantized straight into the lane's registers. No threadgroup memory, no barrier.
    // grid (32*NSG, N/(8*NT*NSG)), threadgroup (32*NSG, 1); xf [8, K] fp32, rows >= M zero.
    const uint lane = thread_position_in_threadgroup.x % 32;
    const uint sg = thread_position_in_threadgroup.x / 32;
    const uint nb = (threadgroup_position_in_grid.y * NSG + sg) * 8 * NT;
    constexpr int KW = K / 8;
    constexpr int G = K / 128;
    const uint qid = lane / 4;
    const uint fm = (qid & 4) + (lane / 2) % 4;        // row of the lane's elements
    const uint fn = (qid & 2) * 2 + (lane % 2) * 2;    // first of its two columns

    simdgroup_float8x8 C[NT];
    for (int t = 0; t < NT; ++t) C[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0 = 0; k0 < K; k0 += 128) {
        float sc[NT], bi[NT];
        for (int t = 0; t < NT; ++t) {
            const uint row = nb + t * 8 + fm;
            sc[t] = static_cast<float>(scales[row * G + k0 / 128]);
            bi[t] = static_cast<float>(biases[row * G + k0 / 128]);
        }
        for (int kk = 0; kk < 128; kk += 8) {
            simdgroup_float8x8 X;
            simdgroup_load(X, xf + k0 + kk, K, ulong2(0, 0), true);    // x^T tile [k x m]
            for (int t = 0; t < NT; ++t) {
                const uint wv = w[(nb + t * 8 + fm) * KW + (k0 + kk) / 8];
                simdgroup_float8x8 Wt;
                thread auto& e = Wt.thread_elements();
                e[0] = sc[t] * static_cast<float>((wv >> (4 * fn)) & 0xF) + bi[t];
                e[1] = sc[t] * static_cast<float>((wv >> (4 * (fn + 1))) & 0xF) + bi[t];
                simdgroup_multiply_accumulate(C[t], Wt, X, C[t]);
            }
        }
    }
    for (int t = 0; t < NT; ++t)
        simdgroup_store(C[t], y + nb + t * 8, N, ulong2(0, 0), true);   // y [8 m, N]
"""

_qmm8_kernel = mx.fast.metal_kernel(
    name="tokenrush_qmm8",
    input_names=["xf", "w", "scales", "biases"],
    output_names=["y"],
    source=_QMM8_SOURCE,
    header=_QMM8_HEADER,
)


def qmm8(x, w, scales, biases, nsg: int = 4, nt: int = 2):
    """y = x @ dequant(W).T for M <= 8 rows on simdgroup matrices. x [M, K] -> [M, N] in x's dtype."""
    M, K = x.shape
    N = w.shape[0]
    assert M <= 8 and K % 128 == 0 and N % (8 * nt * nsg) == 0, (M, K, N)
    xf = x.astype(mx.float32)
    if M < 8:
        xf = mx.concatenate([xf, mx.zeros((8 - M, K), mx.float32)])
    y = _qmm8_kernel(
        inputs=[xf, w, scales, biases],
        template=[("N", N), ("K", K), ("NSG", nsg), ("NT", nt)],
        grid=(32 * nsg, N // (8 * nt * nsg), 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(8, N)],
        output_dtypes=[mx.float32],
    )[0]
    return y[:M].astype(x.dtype)


# --------------------------------- M-row int4 GEMM on the M5 GPU's neural accelerators
# Same identity as above, on Metal 4's matmul2d (the path MLX's own NAX kernels use,
# mlx/backend/metal/kernels/steel/gemm/nax.h): C [16 m x 32 n] += x [16 m x 16 k] *
# W^T, with W supplied as [n x k] (transpose_b). In the 16x16 fragment layout a lane
# holds rows fm and fm+8, columns fn..fn+3: for W that is four adjacent 4-bit codes
# of one weight row, half a word, dequantized to bf16 in registers. Rows m >= M are
# zero. SPLIT > 1 cuts K so narrow-N matrices (5120 outputs, 17408 inputs) fill the GPU;
# the partial sums are added afterwards.

_NAX_HEADER = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp;\n"

_NAX_SOURCE = """
    const uint lane = thread_position_in_threadgroup.x % 32;
    const uint sg = thread_position_in_threadgroup.x / 32;
    const uint n0 = (threadgroup_position_in_grid.y * NSG + sg) * 32;
    const uint kz = threadgroup_position_in_grid.z;
    constexpr int KW = K / 8;
    constexpr int G = K / 128;
    constexpr int KS = K / SPLIT;
    const uint qid = lane >> 2;
    const uint fm = (qid & 4) | ((lane >> 1) & 3);
    const uint fn = ((qid & 2) | (lane & 1)) * 4;

    constexpr auto desc = tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true, tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc, execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>, metal::remove_addrspace_t<decltype(ct_b)>, float>();
    for (int i = 0; i < 16; ++i) ct_c[i] = 0.0f;
    for (int i = 4; i < 8; ++i) ct_a[i] = bfloat(0.0f);              // rows fm + 8 >= 8 > M

    const uint rows[4] = {n0 + fm, n0 + fm + 8, n0 + 16 + fm, n0 + 24 + fm};
    const bool live = fm < M;
    for (int k0 = kz * KS; k0 < (int)(kz + 1) * KS; k0 += 128) {
        float sc[4], bi[4];
        for (int t = 0; t < 4; ++t) {
            sc[t] = static_cast<float>(scales[rows[t] * G + k0 / 128]);
            bi[t] = static_cast<float>(biases[rows[t] * G + k0 / 128]);
        }
        for (int kk = 0; kk < 128; kk += 16) {
            const int k = k0 + kk + fn;
            for (int j = 0; j < 4; ++j) ct_a[j] = live ? x[fm * K + k + j] : bfloat(0.0f);
            const uint sh = 4 * (k % 8);
            for (int t = 0; t < 4; ++t) {
                const uint wv = w[rows[t] * KW + k / 8] >> sh;
                for (int j = 0; j < 4; ++j)
                    ct_b[t * 4 + j] = static_cast<bfloat>(sc[t] * static_cast<float>((wv >> (4 * j)) & 0xF) + bi[t]);
            }
            op.run(ct_a, ct_b, ct_c);
        }
    }
    if (live) {
        for (int j = 0; j < 4; ++j) {
            y[(kz * M + fm) * N + n0 + fn + j] = ct_c[j];
            y[(kz * M + fm) * N + n0 + 16 + fn + j] = ct_c[8 + j];
        }
    }
"""

_nax_kernel = mx.fast.metal_kernel(
    name="tokenrush_qmm_nax",
    input_names=["x", "w", "scales", "biases"],
    output_names=["y"],
    source=_NAX_SOURCE,
    header=_NAX_HEADER,
)


def nax_split(N: int, K: int, target: int = 2048) -> int:
    """The smallest K split (dividing K into 128-multiples) that gives >= target SIMD groups."""
    split = 1
    while N // 32 * split < target and K % (128 * split * 2) == 0:
        split *= 2
    return split


def qmm_nax(x, w, scales, biases, nsg: int = 4, split: int = 0):
    """y = x @ dequant(W).T for M <= 8 rows of bf16 x on matmul2d. -> [M, N] bf16.
    split 0 picks the K split from the shape (nax_split)."""
    M, K = x.shape
    N = w.shape[0]
    split = split or nax_split(N, K)
    assert M <= 8 and x.dtype == mx.bfloat16 and N % (32 * nsg) == 0 and K % (128 * split) == 0, (M, K, N)
    y = _nax_kernel(
        inputs=[x, w, scales, biases],
        template=[("M", M), ("N", N), ("K", K), ("NSG", nsg), ("SPLIT", split)],
        grid=(32 * nsg, N // (32 * nsg), split),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(split, M, N)],
        output_dtypes=[mx.float32],
    )[0]
    return (y[0] if split == 1 else y.sum(0)).astype(x.dtype)
