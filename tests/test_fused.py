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
                                             s_fused.rec[:, 0], s_fused.slot, w.norm_w, s_fused.pos_t, cfg, cfg.eps, BV=BV))
            s_ops.advance(1); s_fused.advance(1)
            assert torch.isfinite(y_f).all()
            assert rel(y_f, y_ops) < 2e-2, (BV, t, rel(y_f, y_ops))
            torch.testing.assert_close(s_fused.conv, s_ops.conv, rtol=0, atol=0)
            assert rel(s_fused.rec, s_ops.rec) < 1e-3, (BV, t, rel(s_fused.rec, s_ops.rec))


def test_gdn_multi_token_step_matches_sequential():
    """An M-token fused step equals M single-token steps: outputs, ring, and the state
    snapshot in slot i equals the sequential state after token i."""
    from tokenrush.state import State
    cfg, w = _gdn_layer_inputs()
    M = 5
    xs = rnd(7 + M, cfg.hidden, std=1.0)
    s_seq, s_multi = State(cfg, 64, DEV, n_slots=M), State(cfg, 64, DEV, n_slots=M)
    for st in (s_seq, s_multi):
        gdn_forward(xs[:7], w, cfg, st, 0)
        st.advance(7)
    ys, snaps = [], []
    for t in range(7, 7 + M):
        ys.append(gdn_forward(xs[t:t + 1], w, cfg, s_seq, 0, fused=True).sum(0))
        s_seq.slot.zero_(); s_seq.slot_h = 0                       # single-token steps leave the state in slot 0
        snaps.append(s_seq.rec[0].clone())
        s_seq.advance(1)
    y_multi = gdn_forward(xs[7:7 + M], w, cfg, s_multi, 0, fused=True).sum(0)   # [M, hidden]
    for i in range(M):
        assert rel(y_multi[i], ys[i]) < 1e-2, (i, rel(y_multi[i], ys[i]))
        assert rel(s_multi.rec[i], snaps[i]) < 1e-3, (i, rel(s_multi.rec[i], snaps[i]))   # M-row vs 1-row GEMV rounding
    assert rel(s_multi.conv, s_seq.conv) < 1e-3                                             # ring holds the GEMV output


def test_attention_multi_query_matches_sequential():
    from tokenrush.model import AttnWeights, attn_forward
    from tokenrush.quant import Linear
    from tokenrush.state import State
    from tokenrush import ops as _ops
    cfg = CFG
    w = AttnWeights(qkv=Linear(rnd(cfg.n_heads * cfg.head_dim * 2 + 2 * cfg.n_kv_heads * cfg.head_dim, cfg.hidden)),
                    o=Linear(rnd(cfg.hidden, cfg.n_heads * cfg.head_dim)),
                    q_norm_w=rnd(cfg.head_dim, std=0.5), k_norm_w=rnd(cfg.head_dim, std=0.5))
    max_len = 2048
    cos, sin = _ops.rope_table(max_len, cfg.rotary_dim, cfg.rope_theta, DEV)
    for T0 in (5, 1100):
        M = 5
        xs = rnd(T0 + M, cfg.hidden, std=1.0)
        s_seq, s_multi = State(cfg, max_len, DEV), State(cfg, max_len, DEV)
        for st in (s_seq, s_multi):
            attn_forward(xs[:T0], w, cfg, st, 0, cos, sin, fused=True)
            st.advance(T0)
        ys = []
        for t in range(T0, T0 + M):
            ys.append(attn_forward(xs[t:t + 1], w, cfg, s_seq, 0, cos, sin, fused=True).sum(0))
            s_seq.advance(1)
        y_multi = attn_forward(xs[T0:T0 + M], w, cfg, s_multi, 0, cos, sin, fused=True).sum(0)
        for i in range(M):
            assert rel(y_multi[i], ys[i]) < 2e-2, (T0, i, rel(y_multi[i], ys[i]))
        assert rel(s_multi.k[0, :, :T0 + M], s_seq.k[0, :, :T0 + M]) < 1e-3   # M-row vs 1-row GEMV rounding


def test_rows_gemv_matches_dequant():
    from tokenrush.quant import QLinear, quantize_int4
    for out, inp, sk in ((5120, 6144, 4), (34816, 5120, 1), (1024, 5120, 1)):
        q, s, m = quantize_int4(rnd(out, inp))
        x = rnd(5, inp, std=1.0)
        ref = QLinear(q, s, m, backend="dequant")(x).float()
        ql = QLinear(q, s, m, backend="triton", split_k=sk)
        assert rel(ql(x), ref) < 1e-2, (out, inp)
        parts = ql.partials(x)
        assert parts.shape[1:] == (5, out) and rel(parts.sum(0), ref) < 1e-2


def test_verify_step_matches_sequential_decode_and_commits():
    """A K-draft verify step: its K+1 logits equal K+1 sequential decode steps' logits
    (teacher-forced on the same tokens), the accepted count is the leading-match
    count, and after the commit the engine continues exactly as the sequential one."""
    w = random_weights(CFG, "triton")
    K = 3
    eng = Engine(CFG, w, max_len=256, fused=True, max_spec=K)
    eng.capture()
    eng.capture_verify([0, K])
    ref = Engine(CFG, w, max_len=256, fused=True)
    ref.capture()
    toks = torch.randint(0, CFG.vocab, (40,), device=DEV)
    # reference: teacher-forced sequential greedy logits after a 20-token prefix
    ref.reset(); ref.forward(toks[:20])
    seq_logits, seq_argmax = [], []
    for t in range(20, 20 + K + 1):
        ref.tok.copy_(toks[t:t + 1]); ref.step()
        seq_logits.append(ref.logits.clone()); seq_argmax.append(int(ref.logits.argmax()))
    # verify step on the same K+1 tokens: drafts = toks[21:21+K]; n = leading matches of argmaxes
    eng.reset(); eng.forward(toks[:20]); eng.tok.copy_(toks[20:21])
    n = eng.verify(toks[21:21 + K])
    expect = 0
    for i in range(K):
        if seq_argmax[i] == int(toks[21 + i]):
            expect += 1
        else:
            break
    assert n == expect
    got = eng.spec_logits[:K + 1].float()
    for i in range(K + 1):
        assert rel(got[i], seq_logits[i].float()) < 2e-2, (i, rel(got[i], seq_logits[i].float()))
    assert int(eng.tok) == seq_argmax[n] and eng.state.pos == 20 + n + 1 and int(eng.state.pos_t) == eng.state.pos
    # continue: a plain K=0 verify step after the commit equals the sequential engine's next step
    ref.reset(); ref.forward(toks[:20 + n + 1]); ref.tok.copy_(torch.tensor([seq_argmax[n]], device=DEV)); ref.step()
    eng.verify(torch.empty(0, dtype=torch.long, device=DEV))
    assert rel(eng.spec_logits[0].float(), ref.logits.float()) < 2e-2
    assert int(eng.tok) == int(ref.tok)


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
        assert parts.shape == (sk, 1, 5120)
        assert rel(parts.sum(0), ref) < 1e-2
        assert rel(ql(x), ref) < 1e-2
        # add_rmsnorm over the partials == add_rmsnorm over the summed bf16 value
        xr, w = rnd(1, 5120, std=1.0), rnd(5120, std=0.5)
        xo1, y1 = fused.add_rmsnorm(xr, parts, w, 1e-6)
        xo2, y2 = fused.add_rmsnorm(xr, parts.sum(0).to(torch.bfloat16), w, 1e-6)
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


def test_prefill_kernel_matches_sdpa_bf16():
    """The Triton prefill attention (bf16 mode) against ops.attn_prefill (SDPA with the
    causal-over-cache mask), for a chunk that starts mid-cache and is not a multiple of
    the block size."""
    from tokenrush import ops as _ops
    HQ, HKV, D, pos, T = 24, 4, 256, 37, 150
    max_len = 512
    K = rnd(HKV, max_len, D, std=1.0); V = rnd(HKV, max_len, D, std=1.0)
    q = rnd(HQ, T, D, std=1.0)
    ref = _ops.attn_prefill(q, K[:, :pos + T], V[:, :pos + T], pos)                      # [HQ, T, D]
    got = fused.attn_prefill_fused(q.transpose(0, 1).contiguous(), K, V, pos, CFG).transpose(0, 1)
    assert rel(got, ref) < 2e-2, rel(got, ref)


def test_fp8_cache_roundtrip_and_attention():
    """fp8 write (prefill path and prep kernel) then decode: fp8 vs bf16 cache on the same
    layer, error within what e4m3 (3 mantissa bits) allows."""
    from tokenrush.model import AttnWeights, attn_forward
    from tokenrush.quant import Linear
    from tokenrush.state import State
    from tokenrush import ops as _ops
    cfg = CFG
    w = AttnWeights(qkv=Linear(rnd(cfg.n_heads * cfg.head_dim * 2 + 2 * cfg.n_kv_heads * cfg.head_dim, cfg.hidden)),
                    o=Linear(rnd(cfg.hidden, cfg.n_heads * cfg.head_dim)),
                    q_norm_w=rnd(cfg.head_dim, std=0.5), k_norm_w=rnd(cfg.head_dim, std=0.5))
    max_len = 1024
    cos, sin = _ops.rope_table(max_len, cfg.rotary_dim, cfg.rope_theta, DEV)
    T0 = 300
    xs = rnd(T0 + 5, cfg.hidden, std=1.0)
    s16, s8 = State(cfg, max_len, DEV), State(cfg, max_len, DEV, kv_dtype=torch.float8_e4m3fn)
    y16 = attn_forward(xs[:T0], w, cfg, s16, 0, cos, sin, fused=True)
    y8 = attn_forward(xs[:T0], w, cfg, s8, 0, cos, sin, fused=True)
    s16.advance(T0); s8.advance(T0)
    assert rel(y8, y16) < 5e-2, rel(y8, y16)                       # prefill through the fp8 cache
    k8 = s8.k[0, :, :T0].float() * s8.k_scale[0, :, :T0, None]
    assert rel(k8, s16.k[0, :, :T0]) < 3e-2                        # the cache itself
    for t in range(T0, T0 + 5):
        y16 = attn_forward(xs[t:t + 1], w, cfg, s16, 0, cos, sin, fused=True)
        y8 = attn_forward(xs[t:t + 1], w, cfg, s8, 0, cos, sin, fused=True)
        s16.advance(1); s8.advance(1)
        assert rel(y8, y16) < 5e-2, (t, rel(y8, y16))
    # the prep kernel's fp8 rows agree with the prefill-path quantization convention
    k8 = s8.k[0, :, T0:T0 + 5].float() * s8.k_scale[0, :, T0:T0 + 5, None]
    assert rel(k8, s16.k[0, :, T0:T0 + 5]) < 3e-2


def test_engine_fp8_end_to_end():
    w = random_weights(CFG, "triton")
    toks = torch.randint(0, CFG.vocab, (30,), device=DEV)
    outs = {}
    for dt in (torch.bfloat16, torch.float8_e4m3fn):
        eng = Engine(CFG, w, max_len=256, fused=True, kv_dtype=dt)
        eng.capture()
        eng.reset(); eng.forward(toks[:20])
        got = []
        for t in range(20, 30):
            eng.tok.copy_(toks[t:t + 1]); eng.step(); got.append(eng.logits.clone())
        outs[dt] = torch.cat(got).float()
    assert rel(outs[torch.float8_e4m3fn], outs[torch.bfloat16]) < 5e-2


def test_plain_norm_and_grouped_conv():
    from tokenrush.dflash import rmsnorm_plain, grouped_dynamic_conv
    x, h, w = rnd(3, 5120, std=1.0), rnd(3, 5120, std=0.3), rnd(5120, std=0.5) + 1
    xo, y = fused.add_rmsnorm(x, h, w, 1e-6, plain=True)
    torch.testing.assert_close(xo, x + h, rtol=0, atol=0)
    torch.testing.assert_close(y, rmsnorm_plain(x + h, w, 1e-6), rtol=1e-2, atol=1e-2)
    L, H, G, KS = 8, 5120, 16, 2
    hh = rnd(L, H, std=1.0)
    proj = rnd(L, 2 * KS * (H // G), std=0.3)                  # the kernel_projection output layout
    dyn = proj.view(L, 2, KS, H // G)
    base = rnd(2, KS, H, std=0.3)
    for j in range(2):
        ref = grouped_dynamic_conv(hh, dyn[:, j], base[j], G)
        got = fused.grouped_conv(hh, dyn[:, j], base[j], G)
        assert ((got.float() - ref.float()).norm() / ref.float().norm()).item() < 1e-2


def test_attn_window_block_matches_sdpa():
    """The draft's attention (context keys < pos within a window, plus B bidirectional
    block keys, ungated) against masked SDPA, at a position inside and beyond the window."""
    HQ, HKV, D, B, W = 32, 8, 128, 8, 2048
    max_len = 4096
    K = rnd(HKV, max_len, D, std=1.0); V = rnd(HKV, max_len, D, std=1.0)
    for pos in (60, 3000):
        q = rnd(B, HQ * D, std=1.0)
        kb, vb = rnd(HKV, B, D, std=1.0), rnd(HKV, B, D, std=1.0)
        got = fused.attn_window_block(q, K, V, torch.tensor([pos], device=DEV), kb, vb, HQ, HKV, D, W)
        lo = max(0, pos - (W - 1))                     # row 0's bound; each row masks its own
        keys = torch.arange(lo, pos, device=DEV)
        blk = pos + torch.arange(B, device=DEV)
        vis = (blk[:, None] - keys[None, :]) < W
        mask = torch.cat([vis, torch.ones(B, B, dtype=torch.bool, device=DEV)], 1)
        Kf = torch.cat([K[:, lo:pos], kb], 1); Vf = torch.cat([V[:, lo:pos], vb], 1)
        ref = F.scaled_dot_product_attention(q.view(B, HQ, D).transpose(0, 1)[None], Kf[None], Vf[None],
                                             attn_mask=mask, enable_gqa=True)[0].transpose(0, 1).reshape(B, HQ * D)
        assert rel(got, ref) < 2e-2, (pos, rel(got, ref))


def test_attn_window_block_ring():
    """Ring-addressed cache: rows hold position mod ring; at a position past the ring's
    size the kernel must still see exactly the window."""
    HQ, HKV, D, B, W, R = 32, 8, 128, 8, 2048, 4096
    pos = 9000
    Kfull = rnd(HKV, pos, D, std=1.0); Vfull = rnd(HKV, pos, D, std=1.0)
    Kring = torch.zeros(HKV, R, D, device=DEV, dtype=BF); Vring = torch.zeros_like(Kring)
    live = torch.arange(pos - R + 8, pos, device=DEV)            # the last R-8 positions are what a ring holds
    Kring[:, live % R] = Kfull[:, live]; Vring[:, live % R] = Vfull[:, live]
    q = rnd(B, HQ * D, std=1.0); kb, vb = rnd(HKV, B, D, std=1.0), rnd(HKV, B, D, std=1.0)
    got = fused.attn_window_block(q, Kring, Vring, torch.tensor([pos], device=DEV), kb, vb, HQ, HKV, D, W, ring=R)
    lo = pos - (W - 1); keys = torch.arange(lo, pos, device=DEV); blk = pos + torch.arange(B, device=DEV)
    mask = torch.cat([(blk[:, None] - keys[None, :]) < W, torch.ones(B, B, dtype=torch.bool, device=DEV)], 1)
    Kf = torch.cat([Kfull[:, lo:pos], kb], 1); Vf = torch.cat([Vfull[:, lo:pos], vb], 1)
    ref = F.scaled_dot_product_attention(q.view(B, HQ, D).transpose(0, 1)[None], Kf[None], Vf[None],
                                         attn_mask=mask, enable_gqa=True)[0].transpose(0, 1).reshape(B, HQ * D)
    assert rel(got, ref) < 2e-2, rel(got, ref)
