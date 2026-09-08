"""Speculative greedy generation: drafts from the MTP head, verification in the
engine's graphed verify step. This version drafts eagerly (Phase 3 step 2/3);
the in-graph draft chain is step 4.

Per step, with the engine holding a committed-but-unprocessed token t at
position pos and the MTP cache holding rows < pos:
  1. MTP pair (t, target hidden at pos-1) at row pos -> draft d1; chain to d2..dK
     from the MTP's own hidden (rows pos+1 .. pos+K-1, speculative).
  2. engine.verify([d1..dK]): processes t, d1..dK; n accepted; new token t' at
     position pos+n+1, target hiddens h[0..n] at positions pos..pos+n.
  3. MTP rows pos+1..pos+n are rewritten with the true pairs (d_i, h[i-1]),
     which the next iteration's step 1 does implicitly: its first call feeds the
     n accepted drafts plus t' together, so the chain starts from true state.
"""
import time

import torch


def generate_spec(engine, mtp, tok, prompt_ids, max_new, stop_ids, K=3, stream=True, chunk=4096):
    dev = engine.device
    assert K <= engine.max_spec and K in engine.spec_graphs
    engine.reset()
    mtp.reset()
    ids = torch.tensor(prompt_ids, device=dev, dtype=torch.long)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = engine.prefill(ids, chunk=chunk)          # eager; sets last_hidden [T, hidden] for the last chunk
    T = ids.numel()
    # target hidden at every prompt position, for the MTP's prompt pass
    engine.reset()
    logits = engine.forward(ids, all_logits=True)
    H = engine.last_hidden                              # [T, hidden]
    nxt = logits[-1].argmax()
    engine.tok.copy_(nxt.view(1))
    mtp.set_pos(1)
    mtp.forward(ids[1:], H[:-1])                        # rows 1..T-1
    pending = (nxt.view(1), H[-1:])                     # the pair for row T: (t, h_{T-1})
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    out, printed, steps, accepted = [], 0, 0, 0
    t1 = time.perf_counter()
    while len(out) < max_new:
        # 1. draft chain
        pos = engine.state.pos
        toks_in, h_in = pending
        if K == 0:
            drafts = torch.empty(0, dtype=torch.long, device=dev)
        else:
            mtp.set_pos(pos + 1 - toks_in.numel())      # rows for the pending true pairs end at row pos
            lg, hn = mtp.forward(toks_in, h_in)
            drafts = []
            d = lg[-1].argmax().view(1)
            h = hn[-1:]
            for _ in range(K):
                drafts.append(d)
                if len(drafts) == K:
                    break
                lg, hn = mtp.forward(d, h)
                d, h = lg[-1].argmax().view(1), hn[-1:]
            drafts = torch.cat(drafts)
        # 2. verify
        committed = engine.tok.clone()
        n = engine.verify(drafts)
        steps += 1
        accepted += n
        new_tokens = [int(committed)] + drafts[:n].tolist()
        out.extend(new_tokens)
        # 3. next pending pairs: the accepted drafts (true now) then the new token
        hid = engine.spec_hidden[:n + 1]                # target hiddens at pos..pos+n
        pending = (torch.cat([drafts[:n], engine.tok]), hid)
        if stream:
            text = tok.decode(out[printed:])
            if not text.endswith("�"):
                print(text, end="", flush=True)
                printed = len(out)
        if any(t in stop_ids for t in new_tokens):
            cut = next(i for i, t in enumerate(out) if t in stop_ids) + 1
            out = out[:cut]
            break
    torch.cuda.synchronize()
    t_dec = time.perf_counter() - t1
    if stream:
        print(tok.decode(out[printed:]))
    return out, {"prompt_tokens": T, "prefill_s": t_prefill, "new_tokens": len(out), "decode_s": t_dec,
                 "decode_tok_s": len(out) / t_dec, "verify_steps": steps,
                 "accepted_per_step": (len(out)) / max(steps, 1), "ms_per_step": t_dec / max(steps, 1) * 1e3}


def generate_spec_graph(engine, mtp, tok, prompt_ids, max_new, stop_ids, K=3, stream=True, chunk=4096,
                        dynamic=(3, 4)):
    """Same as generate_spec, with the draft chain inside the graph (Engine.capture_spec(K)).
    dynamic=(Kmin, Kmax): pick each step's depth as clamp(n_prev + 2, Kmin, Kmax) — deeper
    after a well-accepted chain, shallower after an early rejection; needs graphs for every
    K in the range. The previous step's n accepted drafts are always <= the new K."""
    dev = engine.device
    Ks = list(range(dynamic[0], dynamic[1] + 1)) if dynamic else [K]
    assert all(("spec", k) in engine.spec_graphs for k in Ks)
    kmax = max(Ks)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    nxt, T = prime_spec(engine, mtp, prompt_ids, chunk)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    out, printed, steps = [], 0, 0
    tok_prev = int(nxt)
    n = 0
    depth_hist = {}
    t1 = time.perf_counter()
    while len(out) < max_new:
        k = min(max(n + 2, dynamic[0]), dynamic[1]) if dynamic else K     # n=0 -> Kmin.., n>=2 -> deeper
        if engine.state.pos + kmax + 1 > engine.state.max_len:             # a step processes up to K+1 tokens
            break
        depth_hist[k] = depth_hist.get(k, 0) + 1
        n = engine.spec_step(k)
        steps += 1
        new_tokens = [tok_prev] + engine.drafts[:n].tolist()
        tok_prev = int(engine.tok)
        out.extend(new_tokens)
        if stream:
            text = tok.decode(out[printed:])
            if not text.endswith("\ufffd"):
                print(text, end="", flush=True)
                printed = len(out)
        if any(t in stop_ids for t in new_tokens):
            cut = next(i for i, t in enumerate(out) if t in stop_ids) + 1
            out = out[:cut]
            break
    torch.cuda.synchronize()
    t_dec = time.perf_counter() - t1
    if stream:
        print(tok.decode(out[printed:]))
    return out, {"prompt_tokens": T, "prefill_s": t_prefill, "new_tokens": len(out), "decode_s": t_dec,
                 "decode_tok_s": len(out) / t_dec, "verify_steps": steps,
                 "accepted_per_step": len(out) / max(steps, 1), "ms_per_step": t_dec / max(steps, 1) * 1e3,
                 "depths": depth_hist}


def prime_spec(engine, mtp, prompt_ids, chunk=4096):
    """Prefill the engine (chunked, keeping every position's hidden), run the MTP head over
    the prompt rows, and set up the spec-step inputs. Returns (first token, prompt length)."""
    dev = engine.device
    engine.reset()
    mtp.reset()
    ids = torch.tensor(prompt_ids, device=dev, dtype=torch.long)
    logits, H = engine.prefill_hidden(ids, chunk=chunk)
    T = ids.numel()
    nxt = logits[-1].argmax()
    engine.tok.copy_(nxt.view(1))
    mtp.set_pos(1)
    for s in range(1, T, chunk):                            # MTP rows 1..T-1: pairs (ids[p], H[p-1])
        e = min(T, s + chunk)
        mtp.forward(ids[s:e], H[s - 1:e - 1])
    engine.n_accepted.zero_()
    engine.spec_hidden[0].copy_(H[-1])                      # row 0 pairs with tok at MTP row T
    return nxt, T
