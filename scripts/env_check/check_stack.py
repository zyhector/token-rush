#!/usr/bin/env python3
"""Validate the Python stack on this card, then measure the GDN launch tax.

Three things the project depends on and cannot take for granted after a
toolkit or machine change:

  1. Triton emits working sm_120 code.
  2. CUDA graph capture and replay work.
  3. fla's fused_recurrent_gated_delta_rule runs at the bs=1 decode shape
     (B=1, T=1, 16 QK heads, 48 V heads, head_dim 128) with FP32 state, and
     agrees with a plain PyTorch reference.

Then the one Phase 2 number that is toolkit-dependent: a 48-layer GDN decode
chain, eager vs. replayed from one CUDA graph. The layer is a stand-in (the
same tensors replayed 48 times, so weights and state stay cache-warm); the
reliable output is the launch tax, not the graphed absolute.
"""
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

B, T = 1, 1
H_QK, H_V, D = 16, 48, 128     # Qwen3.8-27B GDN layer
HIDDEN, CONV_K = 5120, 4
N_LAYERS = 48


def ok(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        raise SystemExit(1)


# ---------------------------------------------------------------- 1. Triton
@triton.jit
def _axpy(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    x = tl.load(x_ptr + off, mask=m)
    y = tl.load(y_ptr + off, mask=m)
    tl.store(o_ptr + off, 2 * x + y, mask=m)


def check_triton():
    n = 1 << 20
    x = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    y = torch.randn_like(x)
    o = torch.empty_like(x)
    _axpy[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
    ok("triton sm_120 codegen", torch.allclose(o, 2 * x + y),
       f"triton {triton.__version__}")


# ------------------------------------------------------------ 2. CUDA graph
def check_cuda_graph():
    x = torch.randn(1, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.01
    out = torch.empty_like(x)

    def step():
        out.copy_(F.silu(x @ w.T) + x)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    x.copy_(torch.randn_like(x))
    g.replay()
    torch.cuda.synchronize()
    ref = F.silu(x @ w.T) + x
    ok("CUDA graph capture + replay", torch.allclose(out, ref))


# --------------------------------------------------------- 3. fla GDN step
def gdn_reference(q, k, v, g, beta, state):
    """One recurrent gated-delta step, FP32, per V head (QK heads shared)."""
    q = q.float(); k = k.float(); v = v.float()
    rep = H_V // H_QK
    q = q.repeat_interleave(rep, dim=2); k = k.repeat_interleave(rep, dim=2)
    q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
    q = q * D ** -0.5
    S = state.clone()                                   # [B, HV, K, V]
    kv = k[:, 0]                                         # [B, HV, K]
    S = S * torch.exp(g[:, 0].float())[..., None, None]
    pred = torch.einsum("bhk,bhkv->bhv", kv, S)
    delta = (v[:, 0].float() - pred) * beta[:, 0].float()[..., None]
    S = S + torch.einsum("bhk,bhv->bhkv", kv, delta)
    o = torch.einsum("bhk,bhkv->bhv", q[:, 0], S)
    return o[:, None], S


@triton.jit
def _gdn_step_kernel(q, k, v, g, beta, h0, o, ht, scale,
                     H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                     BV: tl.constexpr):
    """One gated-delta-rule step for one V head, BV columns of the state per
    program. The textbook recurrence, nothing clever: it exists so the launch
    tax can be measured with a single-kernel state update, and as a second
    opinion on whether Triton can compile this recurrence on this card."""
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v, i_hv = pid % NV, pid // NV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    b_q = tl.load(q + i_h * K + o_k).to(tl.float32)
    b_k = tl.load(k + i_h * K + o_k).to(tl.float32)
    b_v = tl.load(v + i_hv * V + o_v).to(tl.float32)
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6) * scale
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_g = tl.load(g + i_hv).to(tl.float32)
    b_beta = tl.load(beta + i_hv).to(tl.float32)
    p_h = i_hv * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(h0 + p_h).to(tl.float32) * tl.exp(b_g)
    b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
    b_h += b_k[:, None] * b_v[None, :]
    tl.store(o + i_hv * V + o_v, tl.sum(b_h * b_q[:, None], 0).to(o.dtype.element_ty))
    tl.store(ht + p_h, b_h)


def triton_recurrent_step(q, k, v, g, beta, initial_state, output_final_state=True,
                          use_qk_l2norm_in_kernel=True, BV=8):
    o = torch.empty_like(v)
    ht = torch.empty_like(initial_state)
    _gdn_step_kernel[(H_V * (D // BV),)](
        q, k, v, g, beta, initial_state, o, ht, D ** -0.5,
        H=H_QK, HV=H_V, K=D, V=D, BV=BV, num_warps=1)
    return o, ht


def _gdn_inputs():
    torch.manual_seed(0)
    q = torch.randn(B, T, H_QK, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, T, H_QK, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, T, H_V, D, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(B, T, H_V, device="cuda", dtype=torch.float32)
    beta = torch.rand(B, T, H_V, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(B, H_V, D, D, device="cuda", dtype=torch.float32) * 0.1
    return q, k, v, g, beta, state


def _differential(step, n_calls=6):
    """Worst-case error of `step` against the FP32 reference over repeated
    calls on the same inputs. Repeated, because a miscompiled kernel can pass
    once and fail on the next call with the same data."""
    q, k, v, g, beta, state = _gdn_inputs()
    o_ref, s_ref = gdn_reference(q, k, v, g, beta, state)
    worst_o = worst_s = 0.0
    for _ in range(n_calls):
        o, s = step(q, k, v, g, beta, initial_state=state, output_final_state=True,
                    use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize()
        eo = (o.float() - o_ref).abs().max().item()
        es = (s - s_ref).abs().max().item()
        worst_o = float("nan") if eo != eo else max(worst_o, eo)
        worst_s = float("nan") if es != es else max(worst_s, es)
    passed = worst_o < 5e-2 and worst_s < 1e-2
    return passed, f"max|dO|={worst_o:.2e} max|dS|={worst_s:.2e} over {n_calls} calls"


def check_fla():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    import fla
    q, k, v, g, beta, state = _gdn_inputs()
    o, s_new = fused_recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state=state, output_final_state=True,
        use_qk_l2norm_in_kernel=True)
    ok("fla fused_recurrent_gated_delta_rule runs at decode shape",
       o.shape == (B, T, H_V, D) and s_new.shape == state.shape and s_new.dtype == torch.float32,
       f"fla {fla.__version__}, out {tuple(o.shape)} {o.dtype}, state {tuple(s_new.shape)} {s_new.dtype}")

    # The fused single-step kernel is the one a decode loop wants, but it is
    # not to be trusted blindly: it is a Triton kernel on a young target.
    passed, detail = _differential(fused_recurrent_gated_delta_rule)
    print(f"  [{'OK' if passed else 'FAIL'}] fla fused_recurrent matches FP32 reference  ({detail})")
    if not passed:
        print("        -> fused_recurrent is miscompiled on this stack: whole heads come back NaN,"
              " deterministically per call sequence. Use chunk_gated_delta_rule for the"
              " reference decode path until a Triton fix lands.")
    passed_chunk, detail = _differential(chunk_gated_delta_rule)
    ok("fla chunk_gated_delta_rule (T=1) matches FP32 reference", passed_chunk, detail)
    passed_own, detail = _differential(triton_recurrent_step)
    ok("minimal Triton step in this file matches FP32 reference", passed_own, detail)
    if passed:
        return "fla fused_recurrent", fused_recurrent_gated_delta_rule
    return "own minimal Triton kernel, fla fused_recurrent being broken here", triton_recurrent_step


def torch_recurrent_step(q, k, v, g, beta, initial_state, output_final_state=True,
                         use_qk_l2norm_in_kernel=True):
    """The same recurrence as gdn_reference, but written as the handful of
    torch ops a decode loop would launch: usable as a timing stand-in when
    the fused kernel is not."""
    rep = H_V // H_QK
    q = F.normalize(q.float(), dim=-1).repeat_interleave(rep, dim=2)[:, 0] * D ** -0.5
    k = F.normalize(k.float(), dim=-1).repeat_interleave(rep, dim=2)[:, 0]
    S = initial_state * torch.exp(g[:, 0])[..., None, None]
    delta = (v[:, 0].float() - torch.einsum("bhk,bhkv->bhv", k, S)) * beta[:, 0].float()[..., None]
    S = S + k[..., None] * delta[..., None, :]
    o = torch.einsum("bhk,bhkv->bhv", q, S)
    return o[:, None].to(v.dtype), S


# --------------------------------------------------- 4. GDN launch tax
def bench_launch_tax(step_name, fused_step):
    """Everything a GDN layer does at decode *except* the big projections
    (in_proj / out_proj are ordinary GEMVs, measured by check_gemv_sol.py and
    bandwidth-bound). One layer's tensors are reused for all 48 layers so the
    3 MB state stays cache-warm: the launch tax is what this measures."""
    torch.manual_seed(0)
    dt = torch.bfloat16
    n_qkv = 2 * H_QK * D + H_V * D
    qkvz = torch.randn(B, n_qkv + H_V * D, device="cuda", dtype=dt)
    ba = torch.randn(B, 2 * H_V, device="cuda", dtype=dt)
    conv_w = torch.randn(n_qkv, CONV_K, device="cuda", dtype=dt) * 0.1
    conv_state = torch.zeros(B, n_qkv, CONV_K, device="cuda", dtype=dt)
    a_log = torch.randn(H_V, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(H_V, device="cuda", dtype=torch.float32)
    norm_w = torch.ones(D, device="cuda", dtype=dt)
    state = torch.zeros(B, H_V, D, D, device="cuda", dtype=torch.float32)
    out = torch.empty(B, H_V * D, device="cuda", dtype=dt)

    def layer():
        # conv1d step, gating, delta-rule state update, gated RMSNorm, output gate
        qkv, z = qkvz.split([n_qkv, H_V * D], dim=-1)
        conv_state.copy_(torch.cat([conv_state[:, :, 1:], qkv[:, :, None]], dim=-1))
        qkv = F.silu((conv_state * conv_w).sum(-1))
        q, k, v = qkv.split([H_QK * D, H_QK * D, H_V * D], dim=-1)
        b, a = ba.float().split([H_V, H_V], dim=-1)
        beta = torch.sigmoid(b)
        g = -torch.exp(a_log) * F.softplus(a + dt_bias)
        o, s = fused_step(q.view(B, T, H_QK, D), k.view(B, T, H_QK, D),
                          v.view(B, T, H_V, D), g[:, None], beta[:, None].to(dt),
                          initial_state=state, output_final_state=True,
                          use_qk_l2norm_in_kernel=True)
        state.copy_(s)
        o = o.view(B, H_V, D).float()
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * norm_w
        out.copy_((o * F.silu(z.view(B, H_V, D).float())).view(B, -1))

    def chain():
        for _ in range(N_LAYERS):
            layer()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            chain()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        chain()

    def timeit(fn, reps=5, iters=20):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(reps):
            t = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t) / iters)
        return best * 1e3

    eager = timeit(chain)
    graphed = timeit(g.replay)
    print(f"\n{N_LAYERS}-layer GDN decode chain, projections excluded (state update: {step_name}):")
    print(f"  eager   {eager:6.2f} ms/token")
    print(f"  graphed {graphed:6.2f} ms/token")
    print(f"  launch tax removed {eager - graphed:5.2f} ms  "
          f"= {(eager - graphed) / 10.0 * 100:.0f}% of a 10 ms budget for 100 tok/s")


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} | sm_{p.major}{p.minor} | torch {torch.__version__} | cuda {torch.version.cuda}\n")
    check_triton()
    check_cuda_graph()
    step_name, step = check_fla()
    bench_launch_tax(step_name, step)


if __name__ == "__main__":
    main()
