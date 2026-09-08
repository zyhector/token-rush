"""The in-graph sampler: exact greedy, correct top-k / top-p masking, and
empirical frequencies matching the truncated softmax."""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_ops import CFG, DEV, random_weights  # noqa: E402

from tokenrush.model import Engine  # noqa: E402
from tokenrush.sample import CANDIDATES, SamplingParams, sample  # noqa: E402

torch.manual_seed(0)


def test_greedy_is_argmax():
    logits = torch.randn(1, 1000, device=DEV).bfloat16()
    p = SamplingParams(DEV)
    p.set(temperature=0.0)
    for _ in range(5):
        assert int(sample(logits, p)) == int(logits.argmax())


def test_masks_and_frequencies():
    logits = torch.zeros(1, 1000, device=DEV)
    logits[0, :8] = torch.tensor([4.0, 3.5, 3.0, 2.0, 1.0, 0.5, 0.0, -1.0], device=DEV)
    logits = logits.bfloat16()
    p = SamplingParams(DEV)
    # top_k = 3: only tokens 0, 1, 2 ever appear, with softmax(4, 3.5, 3) frequencies
    p.set(temperature=1.0, top_k=3)
    n = 20000
    u = torch.rand(n, 1, device=DEV)
    draws = torch.stack([sample(logits, p, u[i]) for i in range(n)]).flatten()
    assert draws.max() <= 2
    freq = torch.bincount(draws, minlength=3).float() / n
    ref = torch.softmax(torch.tensor([4.0, 3.5, 3.0], device=DEV), 0)
    assert (freq - ref).abs().max() < 0.02, (freq, ref)
    # top_p = 0.7: the smallest prefix with mass >= 0.7 of the full top-64 softmax
    p.set(temperature=1.0, top_p=0.7)
    vals, idx = logits.float()[0].topk(CANDIDATES)
    full = torch.softmax(vals, 0)
    cum = full.cumsum(0)
    n_keep = int(((cum - full) < 0.7).sum())
    kept_ids = set(idx[:n_keep].tolist())                        # ranks -> token ids
    draws = torch.stack([sample(logits, p, u[i]) for i in range(2000)]).flatten()
    assert set(draws.tolist()) <= kept_ids
    # temperature -> 0 concentrates on the argmax
    p.set(temperature=0.05)
    draws = torch.stack([sample(logits, p, u[i]) for i in range(500)]).flatten()
    assert (draws == 0).float().mean() > 0.99


def test_engine_samples_in_graph():
    w = random_weights(CFG, "triton")
    eng = Engine(CFG, w, max_len=128, fused=True)
    eng.capture()
    toks = torch.randint(0, CFG.vocab, (10,), device=DEV)
    # greedy replay == argmax of the logits it wrote
    eng.reset(); eng.forward(toks); eng.tok.copy_(toks[-1:])
    eng.sampling.set(temperature=0.0)
    for _ in range(3):
        eng.step()
        assert int(eng.tok) == int(eng.logits.argmax())
    # sampled replay: valid tokens from the top-64 of the logits it wrote, and the
    # generator advances (two runs from the same state differ somewhere)
    runs = []
    for seed in (1, 2):
        eng.reset(); eng.forward(toks); eng.tok.copy_(toks[-1:])
        eng.sampling.set(temperature=1.5, top_p=0.95, top_k=40)
        torch.manual_seed(seed)
        out = []
        for _ in range(12):
            eng.step()
            top = eng.logits.float().topk(40).indices[0]
            assert int(eng.tok) in top.tolist()
            out.append(int(eng.tok))
        runs.append(out)
    assert runs[0] != runs[1]
