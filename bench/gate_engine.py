#!/usr/bin/env python3
"""Engine-correctness gate: our engine against the HF reference dump.

    python bench/gate_engine.py --ref /workspace/ref/hf_ref.pt --hf /workspace/models/Qwen3.8-27B        # bf16, streamed
    python bench/gate_engine.py --ref /workspace/ref/hf_ref.pt --packed /workspace/models/Qwen3.8-27B-int4g128

Three comparisons, in the order a bug would show up:
  1. residual stream after the embedding and after every layer, for the prompt
     (relative error per layer; a bug is a layer where it jumps);
  2. prompt logits: top-1 agreement per position, KL;
  3. teacher-forced decode: HF's greedy tokens fed one at a time, our logits
     against HF's at every step: top-1 agreement, HF's token within our top-5,
     and the logprob gap between our top-1 and HF's token.
The gate in CLAUDE.md is (3) on the bf16 path: identical greedy tokens up to
the first divergence, and at the divergence HF's token in our top-5 with the
gap inside bf16 noise.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenrush.config import ModelConfig  # noqa: E402
from tokenrush.model import Engine, StreamingBF16Engine  # noqa: E402


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def kl(p_logits, q_logits):
    """KL(p || q) in nats, fp32."""
    p = F.log_softmax(p_logits.float(), -1)
    q = F.log_softmax(q_logits.float(), -1)
    return (p.exp() * (p - q)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hf", help="HF bf16 checkpoint: run the streamed bf16 engine")
    ap.add_argument("--packed", help="packed int4 checkpoint: run the quantized engine")
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    ref = torch.load(a.ref)
    ids = ref["input_ids"].cuda()
    T = ids.numel()
    gen = ref["gen_tokens"]
    steps = a.steps or gen.numel()
    print(f"reference: transformers {ref['transformers']}, prompt {T} tokens, {gen.numel()} greedy tokens, "
          f"{ref['hidden'].shape[0]} hidden states")

    if a.hf:
        cfg = ModelConfig.load(a.hf)
        eng = StreamingBF16Engine(cfg, a.hf, max_len=1024)
        label = "bf16 (streamed)"
    else:
        from tokenrush.weights import load_packed
        cfg, w, _ = load_packed(a.packed)
        eng = Engine(cfg, w, max_len=1024)
        label = "int4 g128"
    print(f"engine: {label}")

    # 1. residual stream
    eng.reset()
    eng.trace = []
    t0 = time.time()
    logits = eng.forward(ids, all_logits=True)
    torch.cuda.synchronize()
    print(f"prefill pass {time.time() - t0:.0f}s")
    hid = ref["hidden"]
    n = min(len(eng.trace), hid.shape[0])
    # HF's last hidden state is after the final norm; ours is the raw residual
    from tokenrush import ops
    tr = list(eng.trace)
    if n == cfg.n_layers + 1:
        tr[-1] = ops.rmsnorm(tr[-1], eng.w.final_norm, cfg.eps)
    errs = [rel(tr[i], hid[i].cuda()) for i in range(n)]
    eng.trace = None
    print("residual stream rel error vs HF, embedding then after each layer:")
    for i in range(0, n, 8):
        print("  " + " ".join(f"{i + j:2d}:{errs[i + j]:.1e}" for j in range(min(8, n - i))))
    worst = max(range(n), key=lambda i: errs[i])
    jumps = [i for i in range(1, n) if errs[i] > 3 * max(errs[i - 1], 1e-4)]
    print(f"  worst layer {worst} ({errs[worst]:.2e}); jumps (>3x previous): {jumps or 'none'}")

    # 2. prompt logits
    pl = ref["prompt_logits"].cuda()
    top1 = (logits.argmax(-1) == pl.argmax(-1)).float().mean().item()
    k = kl(pl, logits)
    print(f"prompt logits: top-1 agreement {top1 * 100:.1f}% over {T} positions, "
          f"KL(HF||ours) mean {k.mean():.2e} max {k.max():.2e}, last-position rel err {rel(logits[-1], pl[-1]):.2e}")

    # 3. teacher-forced decode
    print(f"teacher-forced decode, {steps} steps:")
    gl = ref["gen_logits"]
    agree, in_top5, gaps, kls = 0, 0, [], []
    first_div = None
    for i in range(steps):
        tok = gen[i].cuda().view(1)
        ours = eng.decode(tok)[-1]
        theirs = gl[i].cuda()
        hf_tok = int(theirs.argmax())
        our_top = ours.float().topk(5).indices.tolist()
        lp = F.log_softmax(ours.float(), -1)
        gap = (lp[our_top[0]] - lp[hf_tok]).item()
        kls.append(kl(theirs, ours).item())
        ok = our_top[0] == hf_tok
        agree += ok
        in_top5 += hf_tok in our_top
        gaps.append(gap)
        if not ok and first_div is None:
            first_div = i
            print(f"  first divergence at step {i}: HF token {hf_tok} rank {our_top.index(hf_tok) if hf_tok in our_top else '>5'} "
                  f"in ours, logprob gap {gap:.3f}")
    print(f"  top-1 agreement {agree}/{steps}, HF token in our top-5 {in_top5}/{steps}, "
          f"max logprob gap {max(gaps):.3f}, KL mean {sum(kls) / len(kls):.2e} max {max(kls):.2e}")
    print(f"  greedy tokens identical for the first {steps if first_div is None else first_div} of {steps} steps")


if __name__ == "__main__":
    main()
