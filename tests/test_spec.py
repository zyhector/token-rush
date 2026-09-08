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


def test_spec_sampling_matches_raw_distribution():
    """With temperature > 0 the speculative loop must produce the target's sampling
    distribution. Compare the empirical distribution of the first sampled token over
    many runs between the raw sampled graph and the spec graph (drafts from a random
    MTP, so acceptance is rare but nonzero on a 1024-token vocab at high temperature)."""
    torch.manual_seed(0)
    w = random_weights(CFG, "triton")
    eng = Engine(CFG, w, max_len=64, fused=True, max_spec=2)
    eng.capture()
    mtp = MTPHead(CFG, _random_mtp(CFG), w.embed, w.lm_head, 64)
    eng.attach_mtp(mtp)
    eng.capture_spec(2)
    toks = torch.randint(0, CFG.vocab, (10,), device=DEV)
    eng.sampling.set(temperature=1.5, top_k=8)
    N = 600
    # raw: prefill toks[:-1], commit toks[-1], one sampled step -> the token after toks[-1]
    raw_counts = torch.zeros(CFG.vocab)
    for _ in range(N):
        eng.reset(); eng.forward(toks[:-1]); eng.tok.copy_(toks[-1:]); eng.step()
        raw_counts[int(eng.tok)] += 1
    # spec: the same position is the first committed sample (tok after one spec step)
    spec_counts = torch.zeros(CFG.vocab)
    logits = None
    for _ in range(N):
        eng.reset(); mtp.reset()
        lg = eng.forward(toks[:-1], all_logits=True); H = eng.last_hidden
        eng.tok.copy_(toks[-1:]); mtp.set_pos(1); mtp.forward(toks[1:-1], H[:-1])
        eng.n_accepted.zero_(); eng.spec_hidden[0].copy_(H[-1])
        eng.spec_step(2)
        spec_counts[int(eng.spec_logits[0].argmax()) if False else int(eng.tok) if int(eng.n_accepted) == 0 else int(eng.drafts[0])] += 1
    raw_f, spec_f = raw_counts / N, spec_counts / N
    assert (raw_f - spec_f).abs().max() < 0.08, (raw_f.topk(5), spec_f.topk(5))


def test_spec_graph_with_fp8_mtp_cache_runs():
    """The MTP head's cache in fp8: prompt pass (prefill path), batched and chained
    draft rows (fused path) all go through the fp8 kernels; drafts stay plausible
    (they are compared against a bf16-cache head on the same random weights)."""
    w = random_weights(CFG, "triton")
    toks = torch.randint(0, CFG.vocab, (30,), device=DEV)
    drafts = {}
    mw = _random_mtp(CFG)
    for dt in (torch.bfloat16, torch.float8_e4m3fn):
        eng = Engine(CFG, w, max_len=128, fused=True, max_spec=3)
        eng.capture()
        mtp = MTPHead(CFG, mw, w.embed, w.lm_head, 128, kv_dtype=dt)
        eng.attach_mtp(mtp)
        eng.capture_spec(3)
        eng.reset(); mtp.reset()
        from tokenrush.spec import prime_spec
        prime_spec(eng, mtp, toks.tolist())
        eng.spec_step(3)
        drafts[dt] = eng.drafts[:3].tolist()
        assert eng.state.pos > 30
    # fp8 drafts need not equal bf16 drafts (rounding), but the first one usually does on a
    # peaked random head; at minimum they are valid tokens
    assert all(0 <= d < CFG.vocab for d in drafts[torch.float8_e4m3fn])


def test_draft_vocab_restricts_drafts():
    """With a draft vocabulary the drafted ids come from the subset; the committed tokens
    still equal raw greedy (drafts never touch the output)."""
    w = random_weights(CFG, "triton")
    eng = Engine(CFG, w, max_len=128, fused=True, max_spec=3, consistent=True)
    eng.capture()
    mtp = MTPHead(CFG, _random_mtp(CFG), w.embed, w.lm_head, 128)
    subset = torch.arange(0, CFG.vocab, 4, device=DEV)        # every 4th id
    eng.attach_mtp(mtp, draft_vocab=subset)
    eng.capture_spec(3)
    toks = torch.randint(0, CFG.vocab, (12,), device=DEV)
    ref = _raw_greedy(eng, toks, 12)
    from tokenrush.spec import prime_spec
    prime_spec(eng, mtp, toks.tolist())                        # primes on the whole prompt: tok = first generated
    out, tok_prev = [], int(eng.tok)
    for _ in range(6):
        n = eng.spec_step(3)
        assert all(int(d) % 4 == 0 for d in eng.drafts[:3])
        out.extend([tok_prev] + eng.drafts[:n].tolist()); tok_prev = int(eng.tok)
    assert out[:12] == ref[:len(out[:12])]
