"""The MLX backend's kernels against plain-op references, and the engine's
consistency on a small random model (no checkpoint needed).

    uv run pytest tests/mac -q
"""
import pytest

mx = pytest.importorskip("mlx.core")

from tokenrush.config import ModelConfig
from tokenrush.mlx import kernels
from tokenrush.mlx.model import Engine
from tokenrush.mlx.weights import AttnWeights, GDNWeights, LayerWeights, ModelWeights, QLinear

H, HV, K, V = 2, 6, 128, 64


def rows(T, seed):
    mx.random.seed(seed)
    q = kernels_l2(mx.random.normal((T, H, K))) * K ** -0.5
    k = kernels_l2(mx.random.normal((T, H, K)))
    v = mx.random.normal((T, HV, V)).astype(mx.bfloat16)
    g = -mx.random.uniform(0.01, 0.5, (T, HV))
    beta = mx.random.uniform(0.0, 1.0, (T, HV))
    return q, k, v, g, beta


def kernels_l2(x):
    return x * mx.rsqrt((x * x).sum(-1, keepdims=True) + 1e-12)


def close(a, b, tol):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    err = mx.abs(a - b).max().item()
    scale = mx.abs(b).max().item() + 1e-6
    assert err <= tol * scale, f"max err {err} vs scale {scale}"


@pytest.mark.parametrize("T", [1, 5, 37])
def test_gdn_matches_reference(T):
    q, k, v, g, beta = rows(T, T)
    s0 = mx.random.normal((HV, V, K)) * 0.1
    o, s = kernels.gdn(q, k, v, g, beta, s0)
    o_ref, s_ref = kernels.gdn_reference(q, k, v, g, beta, s0)
    close(o, o_ref, 2e-2)          # o is bf16
    close(s, s_ref, 1e-5)


@pytest.mark.parametrize("n", [0, 1, 4])
def test_gdn_deferred_commit(n):
    """A verify call (commit=False) leaves the state alone; the next call replaying its
    first n+1 rows equals processing those rows and then its own in sequence."""
    s0 = mx.random.normal((HV, V, K)) * 0.1
    pq, pk, pv, pg, pb = rows(5, 100)
    o_v, s_v = kernels.gdn(pq, pk, pv, pg, pb, s0, commit=False)
    close(s_v, s0, 0)
    o_ref, _ = kernels.gdn_reference(pq, pk, pv, pg, pb, s0)
    close(o_v, o_ref, 2e-2)
    q, k, v, g, beta = rows(3, 200)
    n_pend = mx.array([n + 1], mx.int32)
    o, s = kernels.gdn(q, k, v, g, beta, s0, pend=(pk, pv, pg, pb, n_pend))
    _, s_mid = kernels.gdn_reference(pq[:n + 1], pk[:n + 1], pv[:n + 1], pg[:n + 1], pb[:n + 1], s0)
    o_ref, s_ref = kernels.gdn_reference(q, k, v, g, beta, s_mid)
    close(o, o_ref, 2e-2)
    close(s, s_ref, 1e-5)


# ------------------------------------------------------------ a small model

def tiny_cfg():
    return ModelConfig(hidden=256, n_layers=4, layer_types=("linear_attention",) * 3 + ("full_attention",),
                       vocab=512, ffn=384, n_heads=4, n_kv_heads=2, head_dim=128, rotary_dim=32,
                       rope_theta=1e7, gdn_k_heads=2, gdn_v_heads=4, gdn_k_dim=128, gdn_v_dim=64,
                       conv_k=4, eps=1e-6, max_pos=4096, eos_ids=(0,))


def qlin(out, inp, std=0.05):
    w, s, b = mx.quantize((mx.random.normal((out, inp)) * std).astype(mx.bfloat16), group_size=128, bits=4)
    return QLinear(w, s, b)


def tiny_engine(max_len=256):
    mx.random.seed(0)
    c = tiny_cfg()
    bf = lambda *s, std=0.05: (mx.random.normal(s) * std).astype(mx.bfloat16)
    layers = []
    for t in c.layer_types:
        if t == "linear_attention":
            mixer = GDNWeights(in_qkvz=qlin(c.conv_dim + c.gdn_val_dim, c.hidden), in_ba=bf(2 * c.gdn_v_heads, c.hidden),
                               conv_w=bf(c.conv_dim, 4, std=0.3), A=-mx.random.uniform(0.5, 2.0, (c.gdn_v_heads,)),
                               dt_bias=mx.random.normal((c.gdn_v_heads,)), norm_w=1 + bf(c.gdn_v_dim),
                               out=qlin(c.hidden, c.gdn_val_dim))
        else:
            hd = c.head_dim
            mixer = AttnWeights(qkv=qlin(c.n_heads * 2 * hd + 2 * c.n_kv_heads * hd, c.hidden),
                                o=qlin(c.hidden, c.n_heads * hd), q_norm_w=bf(hd), k_norm_w=bf(hd))
        layers.append(LayerWeights(ln1=bf(c.hidden), ln2=bf(c.hidden), mixer=mixer,
                                   gate_up=qlin(2 * c.ffn, c.hidden), down=qlin(c.hidden, c.ffn)))
    w = ModelWeights(embed=bf(c.vocab, c.hidden, std=1.0), layers=layers, final_norm=bf(c.hidden),
                     lm_head=qlin(c.vocab, c.hidden))
    return Engine(c, w, max_len)


def test_prefill_chunking_and_decode_agree():
    """One-shot all-row logits == chunked prefill == token-by-token decode."""
    e = tiny_engine()
    toks = mx.random.randint(1, 512, (40,))
    full = e.forward(toks, all_logits=True)
    mx.eval(full)
    e.reset()
    a = e.forward(toks[:17], all_logits=True)
    b = e.forward(toks[17:], all_logits=True)
    close(mx.concatenate([a, b]), full, 3e-2)
    e.reset()
    steps = []
    for t in range(40):
        steps.append(e.forward(toks[t:t + 1]))
        mx.eval(steps[-1], e.state.arrays())
    close(mx.concatenate(steps), full, 3e-2)


@pytest.mark.parametrize("K", [1, 3, 6])
def test_verify_commit_matches_greedy(K):
    """Drive verify/commit with drafts taken from raw greedy and corrupted on a fixed
    pattern: every acceptance length 0..K occurs, and the output must equal raw greedy
    token for token (deferred GDN commit, attention and conv rows overwritten)."""
    e = tiny_engine()
    prompt = mx.random.randint(1, 512, (9,))
    logits = e.forward(prompt)
    tok = mx.argmax(logits, -1)
    ref = [int(tok.item())]
    for _ in range(40):
        tok = mx.argmax(e.forward(tok), -1)
        mx.eval(tok, e.state.arrays())
        ref.append(int(tok.item()))

    e.reset()
    tok = mx.argmax(e.forward(prompt), -1)
    out, step = [], 0
    while len(out) < 30:
        i = len(out)
        drafts = ref[i + 1:i + 1 + K]
        bad = step % (K + 1)                       # corrupt draft `bad` (none when bad == K)
        if bad < len(drafts):
            drafts[bad] = (drafts[bad] + 1) % 512
        seq = mx.concatenate([tok, mx.array(drafts, mx.int32)]) if drafts else tok
        n, pred, _ = e.verify(seq)
        mx.eval(n, pred, e.state.arrays())
        n = int(n.item())
        assert n == min(bad, len(drafts))
        e.commit(n)
        out += seq[:n + 1].tolist()
        tok = pred[n:n + 1]
        step += 1
    assert out[:30] == ref[:30]


def test_sampling_greedy_and_topk1():
    from tokenrush.mlx.spec import Sampling
    lg = mx.random.normal((5, 512))
    am = mx.argmax(lg, -1)
    assert (Sampling()(lg) == am).all().item()
    assert (Sampling(temperature=0.7, top_k=1)(lg) == am).all().item()


class _Tok:
    eos_token_id = 0


def _restored_vs_cold(drop_pend=False):
    """Run a turn, restore its end snapshot, prefill more tokens; and prefill the same
    sequence cold. -> (hidden after restore, hidden cold)."""
    from tokenrush.mlx.session import Session
    e = tiny_engine(max_len=512)
    s = Session(e, _Tok(), e.cfg, chunk=16)
    first = list(range(5, 45))
    out1 = [t for step in s.generate(first, max_new=20, mode="raw") for t in step]
    snap = s.snaps[-1]
    assert snap.pos == len(first) + len(out1) and snap.n_pend is not None
    if drop_pend:
        snap.pend, snap.n_pend = None, None
    more = mx.array(first + out1 + list(range(100, 130)))
    s._restore(snap)
    warm = e.forward_hidden(more[snap.pos:])
    cold_e = tiny_engine(max_len=512)
    cold = cold_e.forward_hidden(more)[snap.pos:]
    mx.eval(warm, cold)
    return warm, cold


def test_session_snapshot_resume_matches_cold():
    """Resuming from an end-of-generation snapshot (which carries a verify step's
    pending GDN rows) and prefilling more gives the cold prefill's hidden states, to
    bf16 path noise. Token equality is not asserted: on a random tiny model the
    different but equivalent arithmetic paths flip near-tied argmaxes."""
    warm, cold = _restored_vs_cold()
    close(warm, cold, 3e-2)
    bad, cold = _restored_vs_cold(drop_pend=True)
    with pytest.raises(AssertionError):
        close(bad, cold, 3e-2)


class _OracleDraft:
    """Stands in for DFlashDraft: proposes the reference continuation, so whole blocks are
    accepted and a stop id lands in the middle of a step."""
    block_size, mask_token_id, target_layer_ids = 8, 0, [1]

    def __init__(self, engine, prompt_len, ref):
        self.e, self.T, self.ref, self.ctx = engine, prompt_len, ref, 0

    def reset(self):
        self.ctx = 0

    def arrays(self):
        return []

    def add_context(self, feats):
        self.ctx += feats.shape[0]

    def forward(self, feats, block):
        if feats is not None:
            self.add_context(feats)
        i = self.e.state.pos - self.T + 1                 # index in ref of the first draft
        return mx.array((self.ref + [1] * 16)[i:i + self.block_size - 1], mx.int32)


def test_session_stop_inside_a_step_commits_only_to_it():
    """The stop id lands inside an accepted run: the state must end right after it, so
    the end snapshot describes exactly the returned tokens and the next turn reuses it."""
    from tokenrush.mlx.session import Session
    e = tiny_engine(max_len=512)
    first = list(range(5, 45))
    ref = [t for step in Session(e, _Tok(), e.cfg, chunk=16).generate(first, max_new=24, mode="raw") for t in step]
    stop = next(t for i, t in enumerate(ref) if i >= 3 and t not in ref[:i])   # first occurrence at i >= 3
    s = Session(e, _Tok(), e.cfg, dflash=_OracleDraft(e, len(first), ref), chunk=16)
    s.stop_ids = {stop}
    out = [t for step in s.generate(first, max_new=40, mode="dflash") for t in step]
    assert s.stats.new_tokens / s.stats.steps > 1                               # multi-token steps
    assert out == ref[:ref.index(stop) + 1]
    end = s.snaps[-1]
    assert end.pos == len(first) + len(out) == len(s.ids)
    list(s.generate(first + out + [7, 8, 9], max_new=2, mode="dflash"))
    assert s.stats.reused_tokens == end.pos


@pytest.mark.parametrize("M", [1, 3, 5, 8])
@pytest.mark.parametrize("N,K", [(256, 512), (1024, 2048)])
def test_int4_kernels_match_mlx(M, N, K):
    """The three M-row int4 kernels against mx.quantized_matmul on the same codes."""
    mx.random.seed(M * 7 + N)
    q = mx.quantize((mx.random.normal((N, K)) * 0.05).astype(mx.bfloat16), group_size=128, bits=4)
    x = mx.random.normal((M, K)).astype(mx.bfloat16)
    ref = mx.quantized_matmul(x, *q, transpose=True, group_size=128, bits=4)
    close(kernels.qmv_rows(x, *q), ref, 1e-2)
    close(kernels.qmm8(x, *q), ref, 1e-2)
    from tokenrush.mlx.weights import _gpu_generation
    if _gpu_generation() >= 17:                     # matmul2d on the neural accelerators: M5 and later
        close(kernels.qmm_nax(x, *q), ref, 2e-2)    # weights rounded to bf16 before the multiply
        close(kernels.qmm_nax(x, *q, split=4), ref, 2e-2)


def test_generate_stops_at_the_context_limit():
    from tokenrush.mlx.spec import generate
    e = tiny_engine(max_len=64)
    out, st = generate(e, None, list(range(5, 45)), max_new=100, K=0)
    assert e.state.pos == e.max_len and len(out) == e.max_len - 40


def test_session_keeps_snapshots_of_a_shared_prefix():
    """One document, two questions: the second reuses the document even though the
    first question's own snapshots are not prefixes of it."""
    from tokenrush.mlx.session import Session
    e = tiny_engine(max_len=512)
    s = Session(e, _Tok(), e.cfg, chunk=16)
    doc = list(range(5, 85))
    list(s.generate(doc, max_new=0, mode="raw"))
    list(s.generate(doc + [300, 301, 302], max_new=4, mode="raw"))
    assert s.stats.reused_tokens == len(doc)
    list(s.generate(doc + [400, 401], max_new=6, mode="raw"))
    assert s.stats.reused_tokens == len(doc)


def _fused_inputs():
    """Inputs at the real model's scales for every compiled group, by name."""
    mx.random.seed(11)
    bf = lambda *sh, sc=1.0: (mx.random.normal(sh) * sc).astype(mx.bfloat16)
    return {
        "gdn_prep": (bf(4, 16, 128), bf(4, 16, 128), bf(4, 48), bf(4, 48),
                     -mx.random.uniform(0.5, 2, (48,)), mx.random.normal((48,)), 128 ** -0.5),
        "gated_norm": (bf(4 * 48, 128), 1 + bf(128, sc=0.1), bf(4 * 48, 128), 1e-6),
        "swiglu": (bf(4, 2 * 1024, sc=4),),
        "sigmoid_gate": (bf(4, 6144), bf(4, 6144)),
        "conv_silu": (bf(7, 1024, sc=3), bf(1024, 4, sc=0.3)),
        "add_norm": (bf(4, 5120, sc=20), bf(4, 5120, sc=5), 1 + mx.random.normal((5120,)) * 0.1, 1e-6),
    }


def test_fused_groups_are_bit_identical():
    """model.FUSE compiles elementwise groups; none may change a bit (gdn_prep did, which
    moved the real model's KL to mlx-lm, and is not compiled). Every compiled group
    needs inputs here."""
    import tokenrush.mlx.model as M
    inputs = _fused_inputs()
    assert M.FUSED and all(f.__name__ in inputs for f in M.FUSED)
    for f in M.FUSED:
        args = inputs[f.__name__]
        M.FUSE = False
        a = f(*args)
        M.FUSE = True
        b = f(*args)
        for x, y in zip(a if isinstance(a, tuple) else (a,), b if isinstance(b, tuple) else (b,)):
            assert mx.array_equal(x, y).item(), f.__name__
