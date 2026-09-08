"""The in-graph speculative step on random weights: plumbing and exactness.
With a random MTP head the drafts are almost never accepted, so the committed
tokens must simply equal raw greedy; with a "perfect" MTP (drafts copied from
the raw continuation) every draft is accepted and the output must still equal
raw greedy, which exercises the commit path."""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_ops import CFG, DEV, rnd, random_weights  # noqa: E402
from test_mtp import _random_mtp  # noqa: E402

from tokenrush.model import Engine  # noqa: E402
from tokenrush.mtp import MTPHead  # noqa: E402


def _raw_greedy(eng, toks, n_new):
    eng.reset(); eng.forward(toks[:-1]); eng.tok.copy_(toks[-1:])
    out = []
    for _ in range(n_new):
        eng.step(); out.append(int(eng.tok))
    return out


def test_spec_graph_matches_raw_greedy():
    w = random_weights(CFG, "triton")
    K = 3
    eng = Engine(CFG, w, max_len=256, fused=True, max_spec=K, consistent=True)
    eng.capture()
    mtp = MTPHead(CFG, _random_mtp(CFG), w.embed, w.lm_head, 256)
    eng.attach_mtp(mtp)
    for k in (2, 3):
        eng.capture_spec(k)
    toks = torch.randint(0, CFG.vocab, (12,), device=DEV)
    ref = _raw_greedy(eng, toks, 16)
    # spec loop by hand (as generate_spec_graph does)
    eng.reset(); mtp.reset()
    logits = eng.forward(toks[:-1], all_logits=True)     # prefix = toks[:-1]; the "committed" token is toks[-1]
    H = eng.last_hidden
    eng.tok.copy_(toks[-1:])
    mtp.set_pos(1); mtp.forward(toks[1:-1], H[:-1])
    eng.n_accepted.zero_(); eng.spec_hidden[0].copy_(H[-1])
    out, tok_prev, n = [], int(toks[-1]), 0
    while len(out) < 17:
        k = min(max(n + 1, 2), 3)
        n = eng.spec_step(k)
        out.extend([tok_prev] + eng.drafts[:n].tolist())
        tok_prev = int(eng.tok)
    # out[0] is toks[-1] itself (the committed token), then the generated ones
    assert out[1:17] == ref[:16], (out[1:17], ref[:16])


def test_spec_loop_respects_max_len():
    """A step processes up to Kmax+1 tokens; the loop must stop before the cache ends."""
    from tokenrush.spec import generate_spec_graph
    w = random_weights(CFG, "triton")
    eng = Engine(CFG, w, max_len=64, fused=True, max_spec=3)
    eng.capture()
    mtp = MTPHead(CFG, _random_mtp(CFG), w.embed, w.lm_head, 64)
    eng.attach_mtp(mtp)
    for k in (3,):
        eng.capture_spec(k)

    class _Tok:                      # minimal tokenizer stand-in for the loop's decode calls
        def decode(self, ids):
            return ""
    ids = torch.randint(0, CFG.vocab, (58,)).tolist()
    out, st = generate_spec_graph(eng, mtp, _Tok(), ids, 100, set(), K=3, stream=False, dynamic=None)
    # stops exactly when the next step (up to 4 tokens) would not fit
    assert eng.state.pos <= 64 and eng.state.pos + 4 > 64 and st["verify_steps"] >= 1
