#!/usr/bin/env python3
"""Speculative and raw decode vs. context length on real long text (fp8 KV).

Multi-position protocol (the one the README figure is drawn from):

    python bench/spec_context.py --text /workspace/data/pg19.txt --draft both \\
        --contexts 0,8000,16000,32000,64000,96000,128000,160000,200000,240000 --positions 6 --new 512

For every context length N and every position p the prompt is the N tokens of
the text that END at p and the continuation is `--new` greedy tokens generated
at p — the same positions for every N and for both drafts, so the drafters are
compared on identical text. The target is prefilled once per (N, p); its state
is snapshotted and raw decode, DFlash2 and the MTP chain each restart from it.
Per context, tok/s is total generated tokens over total decode seconds across
the positions (not the mean of ratios); mean / min / max of accepted tokens per
step and of per-position tok/s are printed on the `summary` line.
`--positions 6` spreads six positions evenly from (max context + 64) to the
end of the text; an explicit list of token offsets is accepted too, and the
resolved offsets are printed so the log records them.

Single-position protocol (Phase 4 and the first sweep): no --positions; the
prompt is the first N tokens, 200 new tokens, one draft per run.
"""
import argparse
import faulthandler
import os
import re
import signal
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenrush.generate import generate  # noqa: E402
from tokenrush.model import Engine  # noqa: E402
from tokenrush.mtp import MTPHead, build_mtp  # noqa: E402
from tokenrush.sample import sample  # noqa: E402
from tokenrush.spec import generate_dflash, generate_spec_graph  # noqa: E402
from tokenrush.weights import DEFAULT_REPO, DFLASH_REPO, load_packed, resolve_model  # noqa: E402


class Snapshot:
    """The engine's decode state after a prefill: enough to restart generation from it.
    KV rows below `pos` are never rewritten by generation, so only the GDN ring, the live
    recurrent slot and the position are copied."""

    def __init__(self, st):
        self.conv = st.conv.clone()
        self.rec = st.rec[st.slot_h].clone()
        self.pos = st.pos

    def restore(self, st):
        st.conv.copy_(self.conv)
        st.rec[0].copy_(self.rec)
        st.slot.zero_()
        st.slot_h = 0
        st.pos = self.pos
        st.pos_t.fill_(self.pos)


def prime_both(eng, mtp, draft, ids, chunk=4096):
    """One chunked prefill of the target feeding both drafts (the union of spec.prime_spec
    and spec.prime_dflash). Returns (first token, T, last hidden row)."""
    eng.reset()
    if draft is not None:
        draft.reset()
    if mtp is not None:
        mtp.reset()
        mtp.set_pos(1)
    T = ids.numel()
    prev_last = None
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        H = eng.forward_hidden(ids[s:e])
        if draft is not None:
            draft.prime(eng.features_of_last_chunk(e - s))
        if mtp is not None:
            Hp = H[:-1] if prev_last is None else torch.cat([prev_last, H[:-1]])
            if Hp.shape[0]:
                mtp.forward(ids[s + (1 if prev_last is None else 0):e], Hp, logits=False)
        prev_last = H[-1:]
    nxt = sample(eng.w.lm_head(prev_last), eng.sampling)[0]
    return nxt, T, prev_last


def run_raw(eng, nxt, new):
    """`new` graphed decode steps from the primed state; returns the stats and the greedy
    tokens (the first is the prefill's), which the speculative outputs must reproduce."""
    eng.tok.copy_(nxt.view(1))
    toks = torch.empty(new + 1, device=eng.device, dtype=torch.long)
    toks[0] = nxt
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(new):
        toks[i + 1] = eng.step()[0]
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    return {"new_tokens": new, "decode_s": t, "decode_tok_s": new / t}, toks.tolist()


def exact(raw_toks, spec_out):
    """'exact' when the speculative output equals the raw greedy tokens, else where it diverged."""
    if raw_toks is None:
        return ""
    n = min(len(raw_toks), len(spec_out))
    for i in range(n):
        if raw_toks[i] != spec_out[i]:
            return f" DIVERGED@{i}"
    return " exact"


def agg(stats):
    """Per-context aggregate over positions: total tokens / total seconds, plus spreads."""
    toks = sum(s["new_tokens"] for s in stats)
    secs = sum(s["decode_s"] for s in stats)
    tps = [s["decode_tok_s"] for s in stats]
    out = {"tok_s": toks / secs, "tok_s_min": min(tps), "tok_s_max": max(tps)}
    if "accepted_per_step" in stats[0]:
        acc = [s["accepted_per_step"] for s in stats]
        steps = sum(s["verify_steps"] for s in stats)
        out.update({"acc": toks / steps, "acc_min": min(acc), "acc_max": max(acc), "ms": secs / steps * 1e3})
    return out


LINE = re.compile(r"^context\s+(\d+) pos\s+(\d+):(.*)$")
SEG = {"raw": re.compile(r"raw\s+([\d.]+) tok/s"),
       "dflash": re.compile(r"dflash\s+([\d.]+) tok/s, ([\d.]+) tokens/step, ([\d.]+) ms/step(?: n=(\d+)/(\d+))?"),
       "mtp": re.compile(r"mtp\s+([\d.]+) tok/s, ([\d.]+) tokens/step, ([\d.]+) ms/step(?: n=(\d+)/(\d+))?")}


def done_units(path, new):
    """(context, position) -> {"raw": stats, "dflash": stats, "mtp": stats} from an earlier log's
    per-position lines. Older lines carry no token count; `new` stands in (the error is the few
    tokens a last verify step overshoots by, well under 1%)."""
    out = {}
    for line in open(path):
        m = LINE.match(line.strip())
        if not m:
            continue
        c, p, rest = int(m.group(1)), int(m.group(2)), m.group(3)
        u = {}
        r = SEG["raw"].search(rest)
        if r:
            tps = float(r.group(1))
            u["raw"] = {"new_tokens": new, "decode_s": new / tps, "decode_tok_s": tps}
        for k in ("dflash", "mtp"):
            r = SEG[k].search(rest)
            if r:
                tps, acc, ms = float(r.group(1)), float(r.group(2)), float(r.group(3))
                n = int(r.group(4)) if r.group(4) else new
                steps = int(r.group(5)) if r.group(5) else n / acc
                u[k] = {"new_tokens": n, "decode_s": n / tps, "decode_tok_s": tps, "accepted_per_step": acc,
                        "verify_steps": steps, "ms_per_step": ms}
        out[(c, p)] = u
    return out


def fmt(a, spec):
    s = f"{a['tok_s']:6.1f} tok/s [{a['tok_s_min']:.0f}-{a['tok_s_max']:.0f}]"
    if spec:
        s += f", {a['acc']:.2f} tokens/step [{a['acc_min']:.2f}-{a['acc_max']:.2f}], {a['ms']:.1f} ms/step"
    return s


def _log_signal(signum, frame):
    """A long sweep that dies silently is worth one line of evidence: which signal it was."""
    print(f"spec_context: received signal {signum} ({signal.Signals(signum).name}) at {time.strftime('%H:%M:%S')}, exiting", flush=True)
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def main():
    faulthandler.enable()                                # a traceback on SIGSEGV / SIGABRT / SIGBUS
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _log_signal)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_REPO, help="packed checkpoint: a Hub repo id or a local directory")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--text", required=True)
    ap.add_argument("--contexts", default="0,22000,90000,200000")
    ap.add_argument("--new", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=262144)
    ap.add_argument("--mtp-kv", default="fp8", choices=("bf16", "fp8"), help="the MTP head's own cache dtype")
    ap.add_argument("--draft", default="mtp", choices=("mtp", "dflash", "both"))
    ap.add_argument("--dflash-path", default=DFLASH_REPO)
    ap.add_argument("--positions", default=None,
                    help="multi-position protocol: a count (evenly spread) or a comma list of token offsets")
    ap.add_argument("--no-raw", action="store_true", help="skip the raw decode measurement")
    ap.add_argument("--resume-log", default=None,
                    help="multi-position protocol: skip the (context, position) units already in this log and fold "
                         "their per-line stats into the summaries (a crashed sweep loses only the unit in flight)")
    a = ap.parse_args()
    a.model = resolve_model(a.model, download=not a.no_download)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    want_mtp, want_dflash = a.draft in ("mtp", "both"), a.draft in ("dflash", "both")
    cfg, w, mtp_t = load_packed(a.model, with_mtp=want_mtp)
    eng = Engine(cfg, w, max_len=a.max_len, max_spec=7 if want_dflash else 4, kv_dtype=torch.float8_e4m3fn)
    eng.capture()
    draft = mtp = None
    if want_dflash:
        from tokenrush.dflash import DFlashDraft, load_dflash
        # the draft's cache is a 4096-row ring (84 MB): bf16, no fp8 needed
        draft = DFlashDraft(load_dflash(resolve_model(a.dflash_path, download=not a.no_download), int4=True),
                            w.embed, w.lm_head, cfg.hidden, a.max_len)
        eng.attach_dflash(draft, draft_vocab=torch.arange(131072))
        eng.capture_spec_dflash()
    if want_mtp:
        mtp = MTPHead(cfg, build_mtp(cfg, mtp_t, "cuda", int4=True), w.embed, w.lm_head, a.max_len,
                      kv_dtype=torch.float8_e4m3fn if a.mtp_kv == "fp8" else torch.bfloat16)
        eng.attach_mtp(mtp, draft_vocab=torch.arange(131072))
        for k in (3, 4):
            eng.capture_spec(k)
    print(f"cuda {torch.cuda.memory_allocated() / 1e9:.1f} GB after capture")
    text = open(a.text).read()
    all_ids = tok.encode(text[:6_000_000])
    print(f"{len(all_ids)} tokens available")
    stop = set()   # ignore eos: measure a fixed number of tokens
    contexts = [int(c) for c in a.contexts.split(",")]
    eng.sampling.set(0.0, 1.0, 64)

    if a.positions is None:                                  # the single-position protocol
        assert a.draft != "both", "--draft both needs --positions"
        run_spec = ((lambda ids: generate_dflash(eng, draft, tok, ids, a.new, stop, stream=False)) if want_dflash else
                    (lambda ids: generate_spec_graph(eng, mtp, tok, ids, a.new, stop, stream=False, dynamic=(3, 4))))
        for ctx in contexts:
            ctx = max(ctx, 64)
            ids = all_ids[:ctx]
            t0 = time.time()
            raw, st_raw = (None, {"decode_tok_s": float("nan"), "prefill_s": float("nan")}) if a.no_raw else generate(eng, tok, ids, a.new, stop, stream=False)
            spec, st_spec = run_spec(ids)
            kv = eng.state.kv_bytes_per_token * ctx / 1e9
            print(f"context {ctx:>7}: raw {st_raw['decode_tok_s']:6.1f} tok/s | spec {st_spec['decode_tok_s']:6.1f} tok/s, "
                  f"{st_spec['accepted_per_step']:.2f} tokens/step, {st_spec['ms_per_step']:.1f} ms/step | "
                  f"KV read {kv:.1f} GB/step | prefill {st_raw['prefill_s']:.0f}s | {time.time() - t0:.0f}s  "
                  f"| {tok.decode(spec[:12])!r}")
        return

    # the multi-position protocol
    lo = max(contexts) + 64
    hi = len(all_ids) - a.new
    if a.positions.isdigit() and int(a.positions) < 1000:              # a count; larger numbers are offsets
        P = int(a.positions)
        positions = [lo + (hi - lo) * i // (P - 1) for i in range(P)] if P > 1 else [lo]
    else:
        positions = [int(p) for p in a.positions.split(",")]
    assert all(lo <= p <= hi for p in positions), f"positions must lie in [{lo}, {hi}]"
    print(f"positions (token offsets of the generation point): {','.join(map(str, positions))}")
    print(f"{a.new} new tokens per position; drafts: " + ", ".join(n for n, ok in (("DFlash2", want_dflash), ("MTP chain", want_mtp)) if ok))
    dev = eng.device
    done = done_units(a.resume_log, a.new) if a.resume_log else {}
    if done:
        print(f"resuming: {len(done)} (context, position) units already in {a.resume_log}")
    for ctx in contexts:
        ctx = max(ctx, 64)
        stats = {"raw": [], "dflash": [], "mtp": []}
        t_ctx = time.time()
        for p in positions:
            if (ctx, p) in done:
                for k, st in done[(ctx, p)].items():
                    stats[k].append(st)
                continue
            ids = torch.tensor(all_ids[p - ctx:p], device=dev, dtype=torch.long)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            nxt, T, last = prime_both(eng, mtp, draft, ids)
            torch.cuda.synchronize()
            t_prefill = time.perf_counter() - t0
            snap = Snapshot(eng.state)
            line = f"context {ctx:>7} pos {p:>8}:"
            raw_toks = None
            if not a.no_raw:
                st, raw_toks = run_raw(eng, nxt, a.new)
                stats["raw"].append(st)
                line += f" raw {st['decode_tok_s']:6.1f} tok/s |"
                snap.restore(eng.state)
            sample_text = None
            if want_dflash:
                eng.tok.copy_(nxt.view(1))
                eng.n_accepted.fill_(-1)
                out, st = generate_dflash(eng, draft, tok, None, a.new, stop, stream=False, primed=(nxt, T))
                stats["dflash"].append(st)
                line += f" dflash {st['decode_tok_s']:6.1f} tok/s, {st['accepted_per_step']:.2f} tokens/step, {st['ms_per_step']:.1f} ms/step n={st['new_tokens']}/{st['verify_steps']}{exact(raw_toks, out)} |"
                sample_text = out[:12]
                snap.restore(eng.state)
            if want_mtp:
                eng.tok.copy_(nxt.view(1))
                eng.n_accepted.zero_()
                eng.spec_hidden[0].copy_(last[0])
                out, st = generate_spec_graph(eng, mtp, tok, None, a.new, stop, stream=False, dynamic=(3, 4), primed=(nxt, T))
                stats["mtp"].append(st)
                line += f" mtp {st['decode_tok_s']:6.1f} tok/s, {st['accepted_per_step']:.2f} tokens/step, {st['ms_per_step']:.1f} ms/step n={st['new_tokens']}/{st['verify_steps']}{exact(raw_toks, out)} |"
                if sample_text is None:
                    sample_text = out[:12]
            line += f" prefill {t_prefill:.0f}s | {tok.decode(sample_text)!r}"
            print(line, flush=True)
        kv = eng.state.kv_bytes_per_token * ctx / 1e9
        parts = [f"{name} {fmt(agg(stats[key]), key != 'raw')}" for key, name in (("raw", "raw"), ("dflash", "dflash"), ("mtp", "mtp")) if stats[key]]
        print(f"summary context {ctx:>7}: " + " | ".join(parts) +
              f" | KV read {kv:.1f} GB/step | {len(positions)} positions x {a.new} tokens | {time.time() - t_ctx:.0f}s", flush=True)


if __name__ == "__main__":
    main()
