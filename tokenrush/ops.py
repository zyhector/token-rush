"""The building blocks: norms, rotary, the GDN conv and recurrence, attention.

Every function here mirrors the HF `transformers` Qwen3_5 implementation
(`modeling_qwen3_5.py`) closely enough for the engine-correctness gate, and is
the reference that Phase 2's fused kernels are differential-tested against.
bs=1 everywhere: activations are [T, dim], never [B, T, dim].
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ------------------------------------------------------------------ norms


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3.5 RMSNorm: the stored weight is an offset, the gain is (1 + weight)."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.float())).to(x.dtype)


def gated_rmsnorm(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float) -> torch.Tensor:
    """GDN output norm: plain gain (no +1), then multiplied by silu(gate). Matches
    Qwen3_5RMSNormGated including its dtype order (normalize in fp32, cast, gain
    in bf16, gate in fp32, cast)."""
    xf = x.float()
    h = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    h = weight * h.to(x.dtype)
    h = h * F.silu(gate.float())
    return h.to(x.dtype)


# ----------------------------------------------------------------- rotary


def rope_table(max_pos: int, rotary_dim: int, theta: float, device, dtype=torch.bfloat16):
    """cos/sin tables [max_pos, rotary_dim] in the activation dtype, as HF does."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    pos = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = pos[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [..., T, head_dim], cos/sin [T, rotary_dim]. Rotates the first rotary_dim only."""
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    xr = xr * cos + _rotate_half(xr) * sin
    return torch.cat([xr, xp], dim=-1)


# ------------------------------------------------------- GDN causal conv1d
# conv_state is a ring of recent pre-conv inputs per channel, [conv_dim, R],
# column = position mod R. Columns (pos - 3 .. pos) mod R are the window for
# position pos; unwritten columns are zero, which is the causal padding.


def conv_prefill(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor, pos: int):
    """x [T, conv_dim] at positions pos..pos+T-1 -> silu(conv(x)) [T, conv_dim]; updates the ring."""
    T = x.shape[0]
    R = conv_state.shape[1]
    xt = x.t()                                                     # [conv_dim, T]
    cols = torch.tensor([(pos - 3 + j) % R for j in range(3)], device=x.device)
    prev = conv_state.index_select(1, cols)                        # [conv_dim, 3]
    if pos < 3:
        prev = prev.clone()
        prev[:, :3 - pos] = 0
    window = torch.cat([prev, xt], dim=-1)
    y = F.conv1d(window[None].to(weight.dtype), weight[:, None, :], groups=weight.shape[0])[0]
    for t in range(max(0, T - R), T):
        conv_state[:, (pos + t) % R] = xt[:, t]
    return F.silu(y).t().to(x.dtype)


def conv_step(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor, pos_t: torch.Tensor):
    """x [1, conv_dim] at device position pos_t -> [1, conv_dim]; updates the ring. Graph-safe."""
    R = conv_state.shape[1]
    cols = (pos_t + torch.arange(R - 3, R + 1, device=x.device)) % R   # (pos-3 .. pos) mod R
    window = conv_state.index_select(1, cols)                      # [conv_dim, 4]
    window[:, 3] = x[0]
    # fp32 products, one rounding: bit-identical to the cuDNN conv1d HF runs
    y = (window.float() * weight.float()).sum(-1).to(weight.dtype)
    conv_state.index_copy_(1, cols[3:], x.t())
    return F.silu(y)[None].to(x.dtype)


# ------------------------------------------------------ GDN recurrence step


@triton.jit
def _gdn_step_kernel(q, k, v, g, beta, h0, o, ht, scale,
                     H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                     BV: tl.constexpr):
    """One gated-delta-rule step for one V head and BV state columns per program.
    q, k: [H, K]; v: [HV, V]; g, beta: [HV]; h0/ht: [HV, K, V] fp32 (may alias)."""
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


def gdn_step(q, k, v, g, beta, state, BV: int = 8):
    """One decode step of the gated delta rule, state updated in place.
    q, k [H, K] bf16; v [HV, V] bf16; g [HV] fp32; beta [HV]; state [HV, K, V] fp32.
    Returns o [HV, V] in v's dtype. q and k are L2-normalized in the kernel."""
    H, K = q.shape
    HV, V = v.shape
    o = torch.empty_like(v)
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    _gdn_step_kernel[(HV * (V // BV),)](
        q, k, v, g, beta, state, o, state, K ** -0.5,
        H=H, HV=HV, K=K, V=V, BV=BV, num_warps=1)
    return o


def gdn_step_reference(q, k, v, g, beta, state):
    """The same step in plain fp32 torch. Returns (o, new_state); does not touch state."""
    H, K = q.shape
    HV, V = v.shape
    rep = HV // H
    qf = F.normalize(q.float(), dim=-1, eps=1e-6).repeat_interleave(rep, 0) * K ** -0.5
    kf = F.normalize(k.float(), dim=-1, eps=1e-6).repeat_interleave(rep, 0)
    S = state * torch.exp(g.float())[:, None, None]
    delta = (v.float() - torch.einsum("hk,hkv->hv", kf, S)) * beta.float()[:, None]
    S = S + kf[:, :, None] * delta[:, None, :]
    o = torch.einsum("hk,hkv->hv", qf, S)
    return o.to(v.dtype), S


def gdn_prefill(q, k, v, g, beta, state):
    """T tokens through fla's chunked kernel. q, k [T, H, K]; v [T, HV, V]; g [T, HV] fp32;
    beta [T, HV]; state [HV, K, V] fp32, updated in place. Returns o [T, HV, V]."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    o, s = chunk_gated_delta_rule(
        q[None], k[None], v[None], g[None], beta[None],
        initial_state=state[None], output_final_state=True, use_qk_l2norm_in_kernel=True)
    state.copy_(s[0])
    return o[0]


# -------------------------------------------------------------- attention


def attn_decode(q, K, V):
    """q [Hq, 1, D]; K, V [Hkv, S, D] -> [Hq, 1, D]. Every cached key is visible."""
    return F.scaled_dot_product_attention(q[None], K[None], V[None], enable_gqa=True)[0]


def attn_decode_bucket(q, K, V, pos_t):
    """q [Hq, 1, D]; K, V [Hkv, S, D] where S is a fixed bucket >= pos+1; keys beyond
    pos_t (a device tensor) are masked. Shape-static, so it captures into a graph."""
    S = K.shape[1]
    mask = torch.arange(S, device=q.device)[None, :] <= pos_t[:, None]        # [1, S]
    return F.scaled_dot_product_attention(q[None], K[None], V[None], attn_mask=mask, enable_gqa=True)[0]


def attn_prefill(q, K, V, pos: int, block: int = 1024):
    """q [Hq, T, D] for positions pos..pos+T-1; K, V [Hkv, pos+T, D].
    Causal over the whole cache: query i sees keys <= pos+i. Queries go through in
    blocks so the boolean mask never exceeds block x S."""
    Hq, T, D = q.shape
    S = K.shape[1]
    assert S == pos + T
    out = torch.empty_like(q)
    keys = torch.arange(S, device=q.device)
    for s0 in range(0, T, block):
        s1 = min(T, s0 + block)
        qi = q[:, s0:s1]
        mask = keys[None, :] <= (pos + torch.arange(s0, s1, device=q.device))[:, None]
        out[:, s0:s1] = F.scaled_dot_product_attention(
            qi[None], K[None, :, :s1 + pos], V[None, :, :s1 + pos], attn_mask=mask[:, :s1 + pos],
            enable_gqa=True)[0]
    return out
