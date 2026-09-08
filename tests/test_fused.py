"""Differential tests for the Phase 2 fused kernels against the ops they replace
(which are verified against HF by bench/gate_engine.py)."""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_ops import CFG, DEV, BF, rnd, random_weights  # noqa: E402

from tokenrush import fused, ops  # noqa: E402
from tokenrush.model import Engine, gdn_forward  # noqa: E402

torch.manual_seed(0)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_add_rmsnorm():
    x, h, w = rnd(3, 5120, std=1.0), rnd(3, 5120, std=0.3), rnd(5120, std=0.5)
    xo, y = fused.add_rmsnorm(x, h, w, 1e-6)
    torch.testing.assert_close(xo, x + h, rtol=0, atol=0)
    torch.testing.assert_close(y, ops.rmsnorm(x + h, w, 1e-6), rtol=1e-2, atol=1e-2)
    xo, y = fused.add_rmsnorm(x, None, w, 1e-6)
    assert xo is x
    torch.testing.assert_close(y, ops.rmsnorm(x, w, 1e-6), rtol=1e-2, atol=1e-2)


def test_silu_mul():
    gu = rnd(1, 2 * 17408, std=1.0)
    g, u = gu.chunk(2, -1)
    torch.testing.assert_close(fused.silu_mul(gu), F.silu(g) * u, rtol=1e-2, atol=1e-2)


def _gdn_layer_inputs():
    from tokenrush.model import GDNWeights
    from tokenrush.quant import Linear
    cfg = CFG
    w = GDNWeights(
        in_qkvz=Linear(rnd(cfg.conv_dim + cfg.gdn_val_dim, cfg.hidden)), in_ba=rnd(2 * cfg.gdn_v_heads, cfg.hidden),
        conv_w=rnd(cfg.conv_dim, cfg.conv_k, std=0.3), A=-torch.exp(torch.randn(cfg.gdn_v_heads, device=DEV)),
        dt_bias=torch.randn(cfg.gdn_v_heads, device=DEV), norm_w=rnd(cfg.gdn_v_dim, std=0.5) + 1,
        out=Linear(rnd(cfg.hidden, cfg.gdn_val_dim)))
    return cfg, w


def test_gdn_step_fused_matches_ops_over_steps():
    """Prefill 5 tokens through the ops path, then 6 decode steps: fused vs ops, same
    inputs, comparing outputs and both states after every step (ring phase wraps)."""
    from tokenrush.state import State
    cfg, w = _gdn_layer_inputs()
    xs = rnd(11, cfg.hidden, std=1.0)
    for BV in (128, 32):
        s_ops, s_fused = State(cfg, 64, DEV), State(cfg, 64, DEV)
        gdn_forward(xs[:5], w, cfg, s_ops, 0)
        gdn_forward(xs[:5], w, cfg, s_fused, 0)
        s_ops.advance(5); s_fused.advance(5)
        torch.testing.assert_close(s_ops.conv, s_fused.conv, rtol=0, atol=0)
        for t in range(5, 11):
            y_ops = gdn_forward(xs[t:t + 1], w, cfg, s_ops, 0, fused=False)
            qkvz = w.in_qkvz(xs[t:t + 1])
            y_f = w.out(fused.gdn_step_fused(qkvz, xs[t:t + 1], w.in_ba, s_fused.conv[0], w.conv_w, w.A, w.dt_bias,
                                             s_fused.rec[0], w.norm_w, s_fused.pos_t, cfg, cfg.eps, BV=BV))
            s_ops.advance(1); s_fused.advance(1)
            assert torch.isfinite(y_f).all()
            assert rel(y_f, y_ops) < 2e-2, (BV, t, rel(y_f, y_ops))
            torch.testing.assert_close(s_fused.conv, s_ops.conv, rtol=0, atol=0)
            assert rel(s_fused.rec, s_ops.rec) < 1e-3, (BV, t, rel(s_fused.rec, s_ops.rec))


def test_engine_fused_matches_unfused_and_graphs():
    w = random_weights(CFG, "triton")
    toks = torch.randint(0, CFG.vocab, (30,), device=DEV)
    outs = {}
    for fused_flag in (False, True):
        eng = Engine(CFG, w, max_len=256, fused=fused_flag)
        eng.reset()
        eng.forward(toks[:20])
        outs[fused_flag] = torch.cat([eng.decode(toks[t]) for t in range(20, 30)]).float()
    # fused vs unfused differ by kernel rounding order (flash-decoding vs SDPA, split-K
    # summation, norm reduction order): ~1% on random-weight logits
    assert rel(outs[True], outs[False]) < 2e-2, rel(outs[True], outs[False])
    # and the fused step captures and replays (one graph: the live length is read on device)
    eng = Engine(CFG, w, max_len=256, fused=True)
    eng.capture()
    assert list(eng.graphs) == [256]
    eng.reset(); eng.forward(toks[:20])
    got = []
    for t in range(20, 30):
        eng.tok.copy_(toks[t:t + 1]); eng.step(); got.append(eng.logits.clone())
    got = torch.cat(got).float()
    assert rel(got, outs[True]) < 2e-2


def test_attention_fused_matches_ops_over_steps():
    """Prefill through the ops path, then decode steps at lengths that cross split and
    block boundaries: fused prep + flash-decoding vs the ops attention, and the cache
    rows they write."""
    from tokenrush.model import AttnWeights, attn_forward
    from tokenrush.quant import Linear
    from tokenrush.state import State
    from tokenrush import ops as _ops
    cfg = CFG
    w = AttnWeights(qkv=Linear(rnd(cfg.n_heads * cfg.head_dim * 2 + 2 * cfg.n_kv_heads * cfg.head_dim, cfg.hidden)),
                    o=Linear(rnd(cfg.hidden, cfg.n_heads * cfg.head_dim)),
                    q_norm_w=rnd(cfg.head_dim, std=0.5), k_norm_w=rnd(cfg.head_dim, std=0.5))
    max_len = 4096
    cos, sin = _ops.rope_table(max_len, cfg.rotary_dim, cfg.rope_theta, DEV)
    for T0 in (5, 2100):            # short (1 split active) and long (all 32 splits, partial last block)
        xs = rnd(T0 + 6, cfg.hidden, std=1.0)
        s_ops, s_f = State(cfg, max_len, DEV), State(cfg, max_len, DEV)
        attn_forward(xs[:T0], w, cfg, s_ops, 0, cos, sin)
        attn_forward(xs[:T0], w, cfg, s_f, 0, cos, sin)
        s_ops.advance(T0); s_f.advance(T0)
        for t in range(T0, T0 + 6):
            y_ops = attn_forward(xs[t:t + 1], w, cfg, s_ops, 0, cos, sin, fused=False)
            y_f = attn_forward(xs[t:t + 1], w, cfg, s_f, 0, cos, sin, fused=True)
            s_ops.advance(1); s_f.advance(1)
            assert torch.isfinite(y_f).all()
            assert rel(y_f, y_ops) < 2e-2, (T0, t, rel(y_f, y_ops))
            kr, vr = s_ops.k[0, :, :t + 1], s_ops.v[0, :, :t + 1]
            assert rel(s_f.k[0, :, :t + 1], kr) < 5e-3 and torch.equal(s_f.v[0, :, :t + 1], vr), (T0, t)


def test_splitk_gemv_and_partial_norm():
    from tokenrush.quant import QLinear, quantize_int4
    x = rnd(1, 17408, std=1.0)
    q, s, m = quantize_int4(rnd(5120, 17408))
    ref = QLinear(q, s, m, backend="dequant")(x).float()
    for sk in (2, 4, 8):
        ql = QLinear(q, s, m, backend="triton", split_k=sk)
        parts = ql.partials(x)
        assert parts.shape == (sk, 5120)
        assert rel(parts.sum(0), ref) < 1e-2
        assert rel(ql(x), ref) < 1e-2
        # add_rmsnorm over the partials == add_rmsnorm over the summed bf16 value
        xr, w = rnd(1, 5120, std=1.0), rnd(5120, std=0.5)
        xo1, y1 = fused.add_rmsnorm(xr, parts, w, 1e-6)
        xo2, y2 = fused.add_rmsnorm(xr, parts.sum(0, keepdim=True).to(torch.bfloat16), w, 1e-6)
        torch.testing.assert_close(xo1, xo2, rtol=0, atol=0)
        torch.testing.assert_close(y1, y2, rtol=0, atol=0)


def test_fused_final_norm_sees_all_partials():
    """Regression: with split-K the last MLP output is S partial rows; the final
    norm must consume their sum, not the last row. The trace records the full
    residual after every layer, so lm_head(rmsnorm(trace[-1])) is the truth."""
    from tokenrush import ops as _ops
    w = random_weights(CFG, "triton")
    assert w.layers[0].down.split_k > 1
    eng = Engine(CFG, w, max_len=64, fused=True)
    eng.reset()
    eng.trace = []
    tok = torch.randint(0, CFG.vocab, (1,), device=DEV)
    logits = eng.decode(tok).float()
    ref = w.lm_head(_ops.rmsnorm(eng.trace[-1], w.final_norm, CFG.eps)).float()
    assert rel(logits, ref) < 5e-3, rel(logits, ref)
