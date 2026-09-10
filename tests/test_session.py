"""Session: prefix reuse must leave the same state as a cold prefill, and the
tokens it generates must be the ones the plain loops generate."""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_ops import CFG, DEV, random_weights  # noqa: E402
from test_mtp import _random_mtp  # noqa: E402
from test_spec import _random_dflash  # noqa: E402

from tokenrush.dflash import DFlashDraft  # noqa: E402
from tokenrush.model import Engine  # noqa: E402
from tokenrush.mtp import MTPHead  # noqa: E402
from tokenrush.session import Session  # noqa: E402


class _Tok:
    eos_token_id = 1


def _build(mode, max_len=256, K=3):
    w = random_weights(CFG, "triton")
    eng = Engine(CFG, w, max_len=max_len, fused=True, max_spec=7 if mode == "dflash" else 4, consistent=True)
    eng.capture()
    dflash = mtp = None
    if mode == "dflash":
        dflash = DFlashDraft(_random_dflash(CFG), w.embed, w.lm_head, CFG.hidden, max_len)
        eng.attach_dflash(dflash)
        eng.capture_spec_dflash()
    elif mode == "mtp":
        mtp = MTPHead(CFG, _random_mtp(CFG), w.embed, w.lm_head, max_len)
        eng.attach_mtp(mtp)
        for k in (3, 4):
            eng.capture_spec(k)
    cfg = type("C", (), {"eos_ids": (0,)})()
    return Session(eng, _Tok(), cfg, dflash=dflash, mtp=mtp, chunk=8, kmin=3, kmax=4)


def _state_of(sess):
    st = sess.engine.state
    return st.pos, st.rec[st.slot_h].clone(), st.conv.clone(), st.k[:, :, :st.pos].clone()


def _run(sess, ids, n, mode):
    out = []
    for toks in sess.generate(ids, max_new=n, mode=mode):
        out.extend(toks)
    return out


REL = lambda a, b: ((a.float() - b.float()).norm() / (b.float().norm() + 1e-6)).item()


def _noise_floor(sess, ids, mode):
    """The state difference that chunking alone produces: a cold prefill of ids in
    8-token chunks against one in a single chunk. Random weights have near-ties
    everywhere, so token equality after a resume is not a meaningful check; the
    state must land within this band (the "bounded, no jump" criterion of test_ops)."""
    import tokenrush.session as S
    sess.forget(); sess.chunk = 8
    g = sess.generate(ids, max_new=1, mode=mode); next(g); g.close()
    a = _state_of(sess)
    sess.forget(); sess.chunk = 4096
    short, S.SHORT_PER_SLOT = S.SHORT_PER_SLOT, 0        # force the one-chunk eager path
    try:
        g = sess.generate(ids, max_new=1, mode=mode); next(g); g.close()
    finally:
        S.SHORT_PER_SLOT = short
    b = _state_of(sess)
    sess.forget(); sess.chunk = 8
    return max(REL(a[1], b[1]), REL(a[3], b[3]), 1e-3)


def _cold_vs_warm(sess, mode, ids, n_new):
    """The state and first token after a warm (resumed) request against a cold one."""
    out_warm = _run(sess, ids, n_new, mode)
    warm = _state_of(sess), sess.stats.reused_tokens, out_warm
    sess.forget()
    out_cold = _run(sess, ids, n_new, mode)
    cold = _state_of(sess), sess.stats.reused_tokens, out_cold
    return warm, cold


def test_prefix_reuse_matches_cold_prefill():
    """Prompt A, then prompt A+B: the second request must resume from A's end-of-prompt
    snapshot (prefill only B) and leave the state a cold prefill of A+B leaves."""
    for mode in ("dflash", "mtp"):
        sess = _build(mode)
        torch.manual_seed(0)
        A = torch.randint(2, CFG.vocab, (40,)).tolist()
        B = torch.randint(2, CFG.vocab, (12,)).tolist()
        floor = _noise_floor(sess, A + B, mode)
        _run(sess, A, 6, mode)
        assert sess.stats.reused_tokens == 0
        # A+B, warm: must reuse the prompt-end snapshot of A (the generated tokens do not
        # follow in A+B, so the generation-end snapshot is not a prefix)
        (st_w, reused, out_w), (st_c, _, out_c) = _cold_vs_warm(sess, mode, A + B, 6)
        assert reused == len(A)
        assert st_w[0] == st_c[0]
        assert out_w[0] == out_c[0]
        assert REL(st_w[1], st_c[1]) < 3 * floor, (mode, REL(st_w[1], st_c[1]), floor)
        assert REL(st_w[3], st_c[3]) < 3 * floor, (mode, REL(st_w[3], st_c[3]), floor)


def test_generation_end_snapshot_continues_the_conversation():
    """Prompt A -> output O; then A+O+C (the client sends back what we generated plus a
    new turn): the generation-end snapshot must be reused, prefilling only C, and the
    state must match a cold prefill of A+O+C within the chunking noise."""
    for mode in ("dflash", "mtp"):
        sess = _build(mode)
        torch.manual_seed(1)
        A = torch.randint(2, CFG.vocab, (33,)).tolist()
        C = torch.randint(2, CFG.vocab, (9,)).tolist()
        O = _run(sess, A, 7, mode)
        assert len(O) >= 7
        assert sess.ids == A + O
        assert any(s.pos == len(A) + len(O) for s in sess.snaps), [s.pos for s in sess.snaps]
        (st_w, reused, out_w), (st_c, _, out_c) = _cold_vs_warm(sess, mode, A + O + C, 5)
        assert reused == len(A) + len(O)
        if mode == "mtp":
            assert sess.mtp.state.pos == st_c[0]       # the head's cursor sits at the end too
        floor = _noise_floor(sess, A + O + C, mode)
        assert st_w[0] == st_c[0]
        assert out_w[0] == out_c[0]
        assert REL(st_w[1], st_c[1]) < 3 * floor, (mode, REL(st_w[1], st_c[1]), floor)
        assert REL(st_w[3], st_c[3]) < 3 * floor, (mode, REL(st_w[3], st_c[3]), floor)


def test_stop_id_cuts_inside_a_step_and_snapshot_lands_on_it():
    """With a stop id that the random model emits somewhere inside a step's committed
    tokens, the sequence and the snapshot must end exactly at the stop token."""
    sess = _build("mtp")
    sess.stop_ids = set(range(2, CFG.vocab, 7))          # many stop ids: a hit inside a step is likely
    torch.manual_seed(2)
    A = torch.randint(2, CFG.vocab, (20,)).tolist()
    out = _run(sess, A, 40, "mtp")
    assert out[-1] in sess.stop_ids
    assert all(t not in sess.stop_ids for t in out[:-1])
    assert sess.ids == A + out
    assert sess.snaps[-1].pos == len(A) + len(out)
    assert sess.engine.state.pos == len(A) + len(out)


def test_early_close_keeps_state_consistent():
    sess = _build("dflash")
    torch.manual_seed(3)
    A = torch.randint(2, CFG.vocab, (24,)).tolist()
    gen = sess.generate(A, max_new=50, mode="dflash")
    got = next(gen) + next(gen)
    gen.close()
    assert sess.ids[len(A):] == got
    assert sess.snaps[-1].pos == len(sess.ids) == sess.engine.state.pos
    # and the next request can continue from it
    _run(sess, sess.ids + [5, 6, 7], 3, "dflash")
    assert sess.stats.reused_tokens == len(A) + len(got)


def test_forward_hidden_short_last_chunk_keeps_the_right_state():
    """A prompt whose last prefill chunk is <= 8 tokens goes through the fused GDN
    path, which leaves the committed state in slot T-1, not slot 0. The state after
    prefilling in one 20-token chunk and in 16 + 4 must agree."""
    w = random_weights(CFG, "triton")
    eng = Engine(CFG, w, max_len=64, fused=True, max_spec=7)
    toks = torch.randint(2, CFG.vocab, (20,), device=DEV)
    eng.reset(); eng.forward_hidden(toks)
    rec_one = eng.state.rec[eng.state.slot_h].clone()
    eng.reset(); eng.forward_hidden(toks[:16]); eng.forward_hidden(toks[16:])
    assert eng.state.slot_h == 3
    rec_two = eng.state.rec[eng.state.slot_h].clone()
    assert REL(rec_two, rec_one) < 0.1, REL(rec_two, rec_one)          # chunking noise only
    assert REL(eng.state.rec[0], rec_one) > 0.3                        # what slot 0 (the old code's answer) holds
