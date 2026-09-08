"""Greedy generation with timing. Streams text to stdout."""
import sys
import time

import torch


def generate(engine, tok, prompt_ids, max_new: int, stop_ids, stream: bool = True, chunk: int = 4096,
             graphed: bool = True, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 64, seed=None):
    from .sample import sample
    dev = engine.device
    graphed = graphed and bool(engine.graphs)
    engine.reset()
    engine.sampling.set(temperature, top_p, top_k)
    if seed is not None:
        torch.manual_seed(seed)
    ids = torch.tensor(prompt_ids, device=dev, dtype=torch.long)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = engine.prefill(ids, chunk=chunk)
    nxt = sample(logits[-1:], engine.sampling)[0]
    if graphed:
        engine.tok.copy_(nxt.view(1))
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    out = []
    printed = 0
    t1 = time.perf_counter()
    for _ in range(max_new):
        tid = int(nxt)
        out.append(tid)
        if stream:
            text = tok.decode(out[printed:])
            if not text.endswith("�"):        # do not print half a multibyte char
                sys.stdout.write(text)
                sys.stdout.flush()
                printed = len(out)
        if tid in stop_ids:
            break
        if graphed:
            nxt = engine.step()[0]
        else:
            nxt = sample(engine.decode(nxt)[-1:], engine.sampling)[0]
    torch.cuda.synchronize()
    t_decode = time.perf_counter() - t1
    if stream:
        sys.stdout.write(tok.decode(out[printed:]) + "\n")
        sys.stdout.flush()
    n_dec = max(len(out) - 1, 1)   # decode steps taken (the first token came from prefill)
    stats = {"prompt_tokens": len(prompt_ids), "prefill_s": t_prefill,
             "prefill_tok_s": len(prompt_ids) / t_prefill,
             "new_tokens": len(out), "decode_s": t_decode, "decode_tok_s": n_dec / t_decode}
    return out, stats
