#!/usr/bin/env python3
"""Teacher-forced acceptance of static draft trees vs chains, with the eager MTP head.

A tree is a list of nodes (parent, rank): the node holds the parent's rank-th most
likely next token (parent -1 = the root, i.e. the committed token). The accepted
length is the longest root path whose every node's token equals the target's argmax
after its parent. Chains are trees with rank 0 everywhere.

    python bench/tree_accept.py --model <packed> [--new 256]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_accept import PROMPTS  # noqa: E402
from tokenrush.model import Engine  # noqa: E402
from tokenrush.mtp import MTPHead, build_mtp  # noqa: E402
from tokenrush.weights import load_packed  # noqa: E402

TREES = {
    "chain3":   [(-1, 0), (0, 0), (1, 0)],
    "chain4":   [(-1, 0), (0, 0), (1, 0), (2, 0)],
    "t2-1-1+1": [(-1, 0), (-1, 1), (0, 0), (2, 0), (1, 0)],                      # 5 nodes: a1 chain 3, a2 chain 2
    "t2-2-1":   [(-1, 0), (-1, 1), (0, 0), (0, 1), (1, 0), (2, 0), (3, 0)],      # 7 nodes, depth 3
    "t2-1-1-1+1+1": [(-1, 0), (-1, 1), (0, 0), (2, 0), (3, 0), (1, 0), (5, 0)],  # 7 nodes: a1 chain 4, a2 chain 3
    "t3-1-1":   [(-1, 0), (-1, 1), (-1, 2), (0, 0), (3, 0), (1, 0), (2, 0)],     # 7 nodes: 3 roots, a1 chain 3
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--new", type=int, default=256)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    cfg, w, mtp_t = load_packed(a.model, with_mtp=True)
    eng = Engine(cfg, w, max_len=4096)
    mtp = MTPHead(cfg, build_mtp(cfg, mtp_t, "cuda"), w.embed, w.lm_head, 4096)
    print(f"{'tree':16s} nodes " + " ".join(f"{n:>7}" for n in PROMPTS) + "   accepted tokens per verify step (with bonus)")
    results = {name: [] for name in TREES}
    for pname, text in PROMPTS.items():
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False))
        T = len(ids)
        x = torch.tensor(ids, device="cuda")
        eng.reset()
        logits = eng.forward(x, all_logits=True)
        hidden = [eng.last_hidden]
        toks = list(ids)
        nxt = int(logits[-1].argmax())
        for _ in range(a.new):
            toks.append(nxt)
            logits = eng.decode(torch.tensor([nxt], device="cuda"))
            hidden.append(eng.last_hidden)
            nxt = int(logits[-1].argmax())
        toks.append(nxt)
        H = torch.cat(hidden)
        toks_t = torch.tensor(toks, device="cuda")
        L = len(toks) - 1
        mtp.reset(); mtp.set_pos(1); mtp.forward(toks_t[1:T], H[:T - 1], logits=False)
        for tname, tree in TREES.items():
            total, n = 0, 0
            depth = [0] * len(tree)
            for i, (par, _) in enumerate(tree):
                depth[i] = 1 if par < 0 else depth[par] + 1
            for p in range(T - 1, L - max(depth) - 1):
                # draft the tree: node tokens via the MTP, chained along each path with rewinds
                node_tok, node_hid = [None] * len(tree), [None] * len(tree)
                node_probs = {}
                for i, (par, rank) in enumerate(tree):
                    if par < 0:
                        key = -1
                        if key not in node_probs:
                            mtp.set_pos(p + 1)
                            lg, hn = mtp.forward(toks_t[p + 1:p + 2], H[p:p + 1])
                            node_probs[key] = (lg[-1].topk(4).indices.tolist(), hn[-1:])
                    else:
                        key = par
                        if key not in node_probs:
                            mtp.set_pos(p + 1 + depth[par])
                            lg, hn = mtp.forward(torch.tensor([node_tok[par]], device="cuda"), node_hid[par])
                            node_probs[key] = (lg[-1].topk(4).indices.tolist(), hn[-1:])
                    cands, hid = node_probs[key]
                    node_tok[i], node_hid[i] = cands[rank], hid
                # accept: longest matching root path against the true continuation
                best = 0
                for i in range(len(tree)):
                    d, ok, j = depth[i], True, i
                    while j >= 0 and ok:
                        ok = node_tok[j] == toks[p + 1 + depth[j]]
                        j = tree[j][0]
                    if ok:
                        best = max(best, d)
                total += 1 + best
                n += 1
            results[tname].append(total / n)
    for tname, tree in TREES.items():
        print(f"{tname:16s} {len(tree):5d} " + " ".join(f"{v:7.2f}" for v in results[tname]))


if __name__ == "__main__":
    main()
