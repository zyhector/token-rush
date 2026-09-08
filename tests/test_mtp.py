"""Smoke test for the MTP head module: shapes, cache advance, chaining."""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_ops import CFG, DEV, rnd  # noqa: E402

from tokenrush.mtp import MTPHead, MTPWeights  # noqa: E402
from tokenrush.model import AttnWeights, LayerWeights  # noqa: E402
from tokenrush.quant import Linear  # noqa: E402


def test_mtp_head_runs_and_chains():
    cfg = CFG
    mixer = AttnWeights(qkv=Linear(rnd(cfg.n_heads * cfg.head_dim * 2 + 2 * cfg.n_kv_heads * cfg.head_dim, cfg.hidden)),
                        o=Linear(rnd(cfg.hidden, cfg.n_heads * cfg.head_dim)),
                        q_norm_w=rnd(cfg.head_dim, std=0.5), k_norm_w=rnd(cfg.head_dim, std=0.5))
    layer = LayerWeights(ln1=rnd(cfg.hidden, std=0.5), ln2=rnd(cfg.hidden, std=0.5), mixer=mixer,
                         gate_up=Linear(rnd(2 * cfg.ffn, cfg.hidden)), down=Linear(rnd(cfg.hidden, cfg.ffn)))
    w = MTPWeights(fc=Linear(rnd(cfg.hidden, 2 * cfg.hidden)), layer=layer, norm=rnd(cfg.hidden, std=0.5),
                   pre_norm_emb=rnd(cfg.hidden, std=0.5), pre_norm_hidden=rnd(cfg.hidden, std=0.5))
    embed = rnd(cfg.vocab, cfg.hidden, std=1.0)
    mtp = MTPHead(cfg, w, embed, Linear(rnd(cfg.vocab, cfg.hidden)), max_len=64)
    toks = torch.randint(0, cfg.vocab, (9,), device=DEV)
    H = rnd(9, cfg.hidden, std=1.0)
    mtp.set_pos(1)
    lg, hn = mtp.forward(toks[1:], H[:-1])                     # prompt pairs at positions 1..8
    assert lg.shape == (8, cfg.vocab) and hn.shape == (8, cfg.hidden) and mtp.state.pos == 9
    lg1, hn1 = mtp.forward(toks[:1], H[-1:])                   # depth 1 at position 9
    lg2, _ = mtp.forward(lg1.argmax(-1), hn1)                  # depth 2 at position 10, chained
    assert lg2.shape == (1, cfg.vocab) and mtp.state.pos == 11 and torch.isfinite(lg2).all()
    mtp.set_pos(9)                                             # rewind: rows beyond are overwritten
    lg1b, _ = mtp.forward(toks[:1], H[-1:])
    torch.testing.assert_close(lg1b, lg1, rtol=0, atol=0)
