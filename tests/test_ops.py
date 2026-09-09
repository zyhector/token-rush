"""Differential and structural tests for the engine's building blocks.

None of these need the real weights. They check (1) every op against a plain
torch reference or against HF's own module, (2) that decode step by step,
prefill in one chunk, and prefill in several chunks all agree with each other.
The second kind is what catches state-management bugs, and it is the cheap
half of the engine-correctness gate."""
import math

import pytest
import torch
import torch.nn.functional as F

from tokenrush import ops
from tokenrush.config import ModelConfig
from tokenrush.model import AttnWeights, Engine, GDNWeights, LayerWeights, ModelWeights
from tokenrush.quant import Linear, QLinear, dequantize_int4, quantize_int4, unpack_int4

torch.manual_seed(0)
DEV = "cuda"
BF = torch.bfloat16

# Real Qwen3.8-27B head geometry, few layers, small vocab.
CFG = ModelConfig(hidden=5120, n_layers=4, layer_types=("linear_attention",) * 3 + ("full_attention",),
                  vocab=1024, ffn=1024, n_heads=24, n_kv_heads=4, head_dim=256, rotary_dim=64,
                  rope_theta=1e7, gdn_k_heads=16, gdn_v_heads=48, gdn_k_dim=128, gdn_v_dim=128, conv_k=4,
                  eps=1e-6, max_pos=4096, eos_ids=(0,))


def rnd(*shape, std=0.02, dtype=BF):
    return (torch.randn(*shape, device=DEV) * std).to(dtype)


# ------------------------------------------------------------------ norms


def test_rmsnorm_matches_hf():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    m = Qwen3_5RMSNorm(256, eps=1e-6).to(DEV, BF)
    with torch.no_grad():
        m.weight.copy_(rnd(256, std=0.5))
    x = rnd(7, 24, 256, std=1.0)
    torch.testing.assert_close(ops.rmsnorm(x, m.weight, 1e-6), m(x), rtol=0, atol=0)


def test_gated_rmsnorm_matches_hf():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated
    m = Qwen3_5RMSNormGated(128, eps=1e-6).to(DEV, BF)
    with torch.no_grad():
        m.weight.copy_(rnd(128, std=0.5) + 1)
    x, z = rnd(48, 128, std=1.0), rnd(48, 128, std=1.0)
    torch.testing.assert_close(ops.gated_rmsnorm(x, m.weight, z, 1e-6), m(x, z), rtol=0, atol=0)


# ----------------------------------------------------------------- rotary


def test_rope_matches_hf():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb
    hc = Qwen3_5TextConfig(hidden_size=5120, num_attention_heads=24, num_key_value_heads=4, head_dim=256,
                           rope_theta=1e7, partial_rotary_factor=0.25,
                           rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                                            "mrope_section": [11, 11, 10], "mrope_interleaved": True},
                           max_position_embeddings=4096)
    rot = Qwen3_5TextRotaryEmbedding(hc).to(DEV)
    T, pos0 = 5, 1000
    q, k = rnd(1, 24, T, 256, std=1.0), rnd(1, 4, T, 256, std=1.0)
    pid = torch.arange(pos0, pos0 + T, device=DEV)[None]
    if hasattr(rot, "recomposition_frequencies"):        # transformers >= 5.17: mRoPE wants (3, bs, T) position ids
        pid = pid[None].expand(3, 1, T)
    cos, sin = rot(q, pid)
    q_hf, k_hf = apply_rotary_pos_emb(q, k, cos, sin)
    cos_t, sin_t = ops.rope_table(4096, 64, 1e7, DEV)
    torch.testing.assert_close(ops.apply_rope(q[0], cos_t[pos0:pos0 + T], sin_t[pos0:pos0 + T]), q_hf[0], rtol=0, atol=0)
    torch.testing.assert_close(ops.apply_rope(k[0], cos_t[pos0:pos0 + T], sin_t[pos0:pos0 + T]), k_hf[0], rtol=0, atol=0)


# ------------------------------------------------------------- GDN pieces


def test_conv_step_matches_prefill():
    cd, K, T = CFG.conv_dim, CFG.conv_k, 6
    w = rnd(cd, K, std=0.3)
    x = rnd(T, cd, std=1.0)
    s1 = torch.zeros(cd, K, device=DEV, dtype=BF)
    s2 = torch.zeros_like(s1)
    pos = lambda p: torch.tensor([p], device=DEV)
    y_pre = ops.conv_prefill(x, s1, w, 0)
    y_step = torch.cat([ops.conv_step(x[t:t + 1], s2, w, pos(t)) for t in range(T)])
    torch.testing.assert_close(y_pre, y_step, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(s1, s2, rtol=0, atol=0)
    # a second prefill continues from the ring the steps left behind, at an arbitrary phase
    x2 = rnd(3, cd, std=1.0)
    torch.testing.assert_close(ops.conv_prefill(x2, s1, w, T),
                               torch.cat([ops.conv_step(x2[t:t + 1], s2, w, pos(T + t)) for t in range(3)]), rtol=1e-2, atol=1e-2)
    # against a plain causal conv over the whole sequence
    xa = torch.cat([x, x2])
    ref = F.silu(F.conv1d(xa.t()[None], w[:, None, :], padding=3, groups=cd)[0, :, :T + 3]).t()
    s3 = torch.zeros_like(s1)
    torch.testing.assert_close(ops.conv_prefill(xa, s3, w, 0), ref, rtol=1e-2, atol=1e-2)


def _gdn_inputs(T=1):
    q = rnd(T, 16, 128, std=1.0)
    k = rnd(T, 16, 128, std=1.0)
    v = rnd(T, 48, 128, std=1.0)
    g = -torch.rand(T, 48, device=DEV)
    beta = torch.rand(T, 48, device=DEV).to(BF)
    state = torch.randn(48, 128, 128, device=DEV) * 0.1
    return q, k, v, g, beta, state


def test_gdn_step_matches_reference_repeatedly():
    q, k, v, g, beta, state = _gdn_inputs()
    o_ref, s_ref = ops.gdn_step_reference(q[0], k[0], v[0], g[0], beta[0], state)
    for _ in range(6):
        s = state.clone()
        o = ops.gdn_step(q[0], k[0], v[0], g[0], beta[0], s)
        torch.cuda.synchronize()
        assert torch.isfinite(o).all() and torch.isfinite(s).all()
        torch.testing.assert_close(o.float(), o_ref.float(), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(s, s_ref, rtol=1e-4, atol=1e-5)


def test_gdn_prefill_matches_steps():
    T = 8
    q, k, v, g, beta, state = _gdn_inputs(T)
    s1, s2 = state.clone(), state.clone()
    o_pre = ops.gdn_prefill(q, k, v, g, beta, s1)
    o_step = torch.stack([ops.gdn_step(q[t], k[t], v[t], g[t], beta[t], s2) for t in range(T)])
    torch.testing.assert_close(o_pre.float(), o_step.float(), rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(s1, s2, rtol=1e-2, atol=1e-2)


# -------------------------------------------------------------- attention


def test_attn_prefill_matches_decode():
    Hq, Hkv, D, pos, T = 24, 4, 256, 37, 9
    q = rnd(Hq, T, D, std=1.0)
    K = rnd(Hkv, pos + T, D, std=1.0)
    V = rnd(Hkv, pos + T, D, std=1.0)
    o_pre = ops.attn_prefill(q, K, V, pos, block=4)
    o_dec = torch.cat([ops.attn_decode(q[:, t:t + 1], K[:, :pos + t + 1], V[:, :pos + t + 1]) for t in range(T)], 1)
    torch.testing.assert_close(o_pre, o_dec, rtol=2e-2, atol=2e-2)


# ------------------------------------------------------------------ quant


def test_int4_pack_roundtrip():
    w = rnd(64, 512, std=1.0)
    q, s, m = quantize_int4(w)
    assert q.shape == (64, 256) and s.shape == (64, 4) and m.shape == (64, 4)
    codes = unpack_int4(q)
    assert codes.max() <= 15
    wq = dequantize_int4(q, s, m)
    step = (w.float().view(64, 4, 128).amax(-1) - w.float().view(64, 4, 128).amin(-1)) / 15
    err = (wq.float() - w.float()).abs().view(64, 4, 128).amax(-1)
    assert (err <= step * 0.5 + 0.02 * w.float().abs().max()).all()   # half a step plus bf16 rounding
    # the dequant backend and the dequantized bf16 Linear agree exactly; the Triton
    # rows path agrees to accumulation-order rounding
    x = rnd(3, 512, std=1.0)
    torch.testing.assert_close(QLinear(q, s, m, backend="dequant")(x), Linear(wq)(x), rtol=0, atol=0)
    y = QLinear(q, s, m, backend="triton")(x)
    assert ((y.float() - Linear(wq)(x).float()).norm() / Linear(wq)(x).float().norm()).item() < 1e-2


# --------------------------------------------------- engine, structurally


def random_weights(cfg: ModelConfig, backend=None) -> ModelWeights:
    """backend None -> bf16 Linear; else QLinear with that GEMV backend (split-K on the
    hidden-sized outputs, as the loader does)."""
    def lin(out, inp, std=0.02):
        w = rnd(out, inp, std=std)
        if not backend:
            return Linear(w)
        return QLinear(*quantize_int4(w), backend=backend, split_k=4 if (out == cfg.hidden and backend == "triton") else 1)

    layers = []
    for lt in cfg.layer_types:
        if lt == "linear_attention":
            mixer = GDNWeights(
                in_qkvz=lin(cfg.conv_dim + cfg.gdn_val_dim, cfg.hidden),
                in_ba=rnd(2 * cfg.gdn_v_heads, cfg.hidden),
                conv_w=rnd(cfg.conv_dim, cfg.conv_k, std=0.3),
                A=-torch.exp(torch.randn(cfg.gdn_v_heads, device=DEV)),
                dt_bias=torch.randn(cfg.gdn_v_heads, device=DEV),
                norm_w=rnd(cfg.gdn_v_dim, std=0.5) + 1, out=lin(cfg.hidden, cfg.gdn_val_dim))
        else:
            mixer = AttnWeights(
                qkv=lin(cfg.n_heads * cfg.head_dim * 2 + 2 * cfg.n_kv_heads * cfg.head_dim, cfg.hidden),
                o=lin(cfg.hidden, cfg.n_heads * cfg.head_dim),
                q_norm_w=rnd(cfg.head_dim, std=0.5), k_norm_w=rnd(cfg.head_dim, std=0.5))
        layers.append(LayerWeights(ln1=rnd(cfg.hidden, std=0.5), ln2=rnd(cfg.hidden, std=0.5), mixer=mixer,
                                   gate_up=lin(2 * cfg.ffn, cfg.hidden), down=lin(cfg.hidden, cfg.ffn)))
    return ModelWeights(embed=rnd(cfg.vocab, cfg.hidden, std=1.0), layers=layers,
                        final_norm=rnd(cfg.hidden, std=0.5), lm_head=lin(cfg.vocab, cfg.hidden))


@pytest.mark.parametrize("backend", ["tinygemm", "triton"])
def test_int4_backends_match_dequant(backend):
    """The int4 GEMV kernels agree with dequantize-then-matmul on real shapes."""
    for out, inp in ((5120, 6144), (34816, 5120), (1024, 5120)):
        q, s, m = quantize_int4(rnd(out, inp))
        x = rnd(1, inp, std=1.0)
        ref = QLinear(q, s, m, backend="dequant")(x).float()
        y = QLinear(q, s, m, backend=backend)(x).float()
        assert ((y - ref).norm() / ref.norm()).item() < 1e-2, (backend, out, inp)


def test_qlinear_cat_matches_separate():
    a, b = rnd(1024, 512), rnd(256, 512)
    qa, qb = QLinear(*quantize_int4(a), backend="dequant"), QLinear(*quantize_int4(b), backend="dequant")
    x = rnd(1, 512, std=1.0)
    torch.testing.assert_close(QLinear.cat([qa, qb], "dequant")(x), torch.cat([qa(x), qb(x)], -1), rtol=0, atol=0)


@pytest.mark.parametrize("quant", [None, "dequant", "triton", "tinygemm"])
def test_engine_prefill_chunks_and_decode_agree(quant):
    w = random_weights(CFG, quant)
    eng = Engine(CFG, w, max_len=256, fused=False)
    toks = torch.randint(0, CFG.vocab, (23,), device=DEV)

    eng.reset()
    full = eng.forward(toks, all_logits=True).float()                      # one chunk
    eng.reset()
    chunked = torch.cat([eng.forward(toks[:10], all_logits=True), eng.forward(toks[10:17], all_logits=True),
                         eng.forward(toks[17:], all_logits=True)]).float()  # three chunks
    eng.reset()
    eng.forward(toks[:4])
    stepped = torch.cat([eng.decode(toks[t]) for t in range(4, 23)]).float()  # prefix, then token by token

    assert eng.state.pos == 23
    assert full.std() > 0.1 and torch.isfinite(full).all()
    # The three paths run different kernels (SDPA tiles by sequence length, fla's
    # chunk kernel accumulates in bf16, the step kernel in fp32), so they agree to
    # bf16 noise, about 2% in norm, and no better. A state bug shows up as error
    # that jumps after a chunk boundary, so that is what is checked.
    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()
    assert rel(chunked, full) < 0.05, rel(chunked, full)
    assert rel(stepped, full[4:]) < 0.05, rel(stepped, full[4:])
    row = ((chunked - full).norm(dim=-1) / full.norm(dim=-1))
    assert row[10:].mean() < 2 * row[:10].mean(), row.tolist()
    row = ((stepped - full[4:]).norm(dim=-1) / full[4:].norm(dim=-1))
    assert row[8:].mean() < 2 * row[:8].mean(), row.tolist()


def test_graph_replay_matches_eager_bucketed_step():
    """A captured decode step reproduces the eager bucketed step bit for bit, and
    the bucketed step agrees with the sliced eager decode to bf16 noise."""
    w = random_weights(CFG, "triton")
    eng = Engine(CFG, w, max_len=256, fused=False)
    eng.capture(buckets=[64, 256])
    toks = torch.randint(0, CFG.vocab, (40,), device=DEV)

    # eager reference: sliced decode
    eng.reset()
    eng.forward(toks[:30])
    ref = torch.cat([eng.decode(toks[t]) for t in range(30, 40)]).float()

    # graph replay, fed the same tokens (not its own argmax)
    eng.reset()
    eng.forward(toks[:30])
    outs = []
    for t in range(30, 40):
        eng.tok.copy_(toks[t:t + 1])
        eng.step()
        outs.append(eng.logits.clone())
    got = torch.cat(outs).float()
    assert eng.state.pos == 40 and int(eng.state.pos_t) == 40
    assert eng.bucket(30) == 64 and eng.bucket(64) == 256

    # eager execution of the very same bucketed function, for the bit-exact check
    eng.reset()
    eng.forward(toks[:30])
    eag = []
    for t in range(30, 40):
        eng.tok.copy_(toks[t:t + 1])
        with torch.no_grad():
            eng._graph_step(eng.bucket(eng.state.pos))
        eng.state.pos += 1
        eag.append(eng.logits.clone())
    eag = torch.cat(eag).float()

    torch.testing.assert_close(got, eag, rtol=0, atol=0)
    assert ((got - ref).norm() / ref.norm()).item() < 0.05
