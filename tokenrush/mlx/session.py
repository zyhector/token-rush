"""tokenrush/session.py's Session on the MLX backend: one request at a time, the
context reused between requests through snapshots. Same interface, so
tokenrush/serve.py's HTTP layer (OpenAI and Anthropic APIs, tool calls, the
output parser) runs on it unchanged.

Snapshots are cheaper here than on CUDA. MLX arrays are values: a GDN layer's
recurrent state and a verify step's pending rows are replaced, never written in
place, so a snapshot keeps references to them. Only the conv rings are updated in
place and get copied (48 x 16 x 10240 bf16, 16 MB). KV rows past the snapshot are
left in place and overwritten when generation resumes there, as on CUDA.
"""
import time
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from ..session import MIN_REUSE, Stats
from ..spec import cjk_share
from .model import Engine
from .spec import Sampling


@dataclass
class Snapshot:
    pos: int
    mode: str
    rec: list
    conv: list
    pend: Optional[list]
    n_pend: Optional[mx.array]
    hidden: mx.array          # [1, hidden], post-final-norm hidden of position pos-1


class Session:
    def __init__(self, engine: Engine, tok, cfg, dflash=None, mtp=None, chunk: int = 2048, k_mtp: int = 3):
        self.engine, self.tok, self.cfg = engine, tok, cfg
        self.dflash, self.mtp = dflash, mtp
        self.chunk, self.k_mtp = chunk, k_mtp
        self.stop_ids = set(cfg.eos_ids) | {tok.eos_token_id}
        self.ids, self.snaps = [], []
        self.max_len = engine.max_len
        self.stats = Stats()

    def pick_mode(self, text: str, mode: str = "auto") -> str:
        if mode != "auto":
            return mode
        if self.dflash is not None and self.mtp is not None:
            return "mtp" if cjk_share(text) >= 0.2 else "dflash"
        return "dflash" if self.dflash is not None else ("mtp" if self.mtp is not None else "raw")

    def forget(self):
        self.ids, self.snaps = [], []

    # ------------------------------------------------------------ snapshots

    def _take(self, mode: str, hidden: mx.array) -> Snapshot:
        st = self.engine.state
        conv = [c * 1 for c in st.conv]
        mx.eval(conv)
        return Snapshot(st.pos, mode, list(st.rec), conv, st.pend, st.n_pend, hidden)

    def _restore(self, s: Snapshot):
        st = self.engine.state
        st.pos, st.rec, st.conv = s.pos, list(s.rec), [c * 1 for c in s.conv]
        st.pend, st.n_pend = s.pend, s.n_pend
        if s.mode == "dflash":
            self.dflash.ctx = s.pos
        elif s.mode == "mtp":
            self.mtp.state.pos = s.pos

    def _prefix_len(self, ids) -> int:
        n = min(len(self.ids), len(ids))
        i = 0
        while i < n and self.ids[i] == ids[i]:
            i += 1
        return i

    # ------------------------------------------------------------ generation

    def _prefill(self, ids: list, start: int, mode: str, prev: Optional[mx.array]) -> mx.array:
        """Engine (and the mode's draft) over ids[start:]; -> hidden of the last position [1, H]."""
        eng = self.engine
        a = mx.array(ids)
        for s in range(start, len(ids), self.chunk):
            e = min(len(ids), s + self.chunk)
            H = eng.forward_hidden(a[s:e])
            if mode == "mtp":
                # rows p in [s, e) pair (ids[p], hidden at p-1); row 0 of a sequence has none
                hs = H[:-1] if prev is None else mx.concatenate([prev, H[:-1]])
                ts = a[s + 1:e] if prev is None else a[s:e]
                if prev is None:
                    self.mtp.state.pos = s + 1
                if ts.shape[0]:
                    self.mtp.hidden(ts, hs)
            elif mode == "dflash":
                self.dflash.add_context(eng.features)
            prev = H[-1:]
            extra = self.mtp.state.arrays() if mode == "mtp" else (self.dflash.arrays() if mode == "dflash" else [])
            mx.eval(prev, eng.state.arrays(), extra)
        return prev

    def generate(self, ids, max_new: int, mode: str = "dflash", temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int = 64, seed=None):
        """Yields the token ids each step commits (the last list may end with a stop id).
        Close the generator to stop early; the end-of-generation snapshot is taken either way."""
        eng = self.engine
        assert mode in ("dflash", "mtp", "raw")
        assert mode != "dflash" or self.dflash is not None
        assert mode != "mtp" or self.mtp is not None
        ids = [int(t) for t in ids]
        T = len(ids)
        if T >= self.max_len:
            raise ValueError(f"prompt of {T} tokens does not fit the {self.max_len}-token context")
        stats = self.stats = Stats(prompt_tokens=T)
        eng.feature_layers = self.dflash.target_layer_ids if mode == "dflash" else None
        sampling = Sampling(temperature, top_p, top_k)
        pick = sampling if temperature > 0 else None
        if seed is not None:
            mx.random.seed(seed)

        t0 = time.perf_counter()
        L = self._prefix_len(ids)
        best = None
        for s in self.snaps:
            if s.mode == mode and MIN_REUSE <= s.pos <= L and (best is None or s.pos > best.pos):
                best = s
        # Snapshots at or before the resume point stay valid: this request writes engine and
        # draft rows from `start` on only. Kept, a long shared prefix (one document, several
        # questions) is reused by every later request, not just the next one.
        start_at = best.pos if best is not None else 0
        keep = [s for s in self.snaps if s.pos <= start_at and s is not best]
        self.ids, self.snaps = [], []                # rewritten below only once the prefill succeeds
        if best is not None:
            self._restore(best)
            start, prev = best.pos, best.hidden
        else:
            eng.reset()
            if mode == "dflash":
                self.dflash.reset()
            elif mode == "mtp":
                self.mtp.reset()
            start, prev = 0, None
        stats.reused_tokens, stats.prefill_tokens = start, T - start
        last = self._prefill(ids, start, mode, prev) if start < T else prev
        self.ids = list(ids)
        self.snaps = keep + ([best] if best is not None else [])
        if not self.snaps or self.snaps[-1].pos != T:
            self.snaps.append(self._take(mode, last))
        tok = (pick or (lambda l: mx.argmax(l, -1)))(eng.w.lm_head(last)).astype(mx.int32)
        mx.eval(tok)
        stats.prefill_s = time.perf_counter() - t0

        out, steps = [], 0
        pending = (tok, last)                       # MTP: the true pairs not yet in its cache
        new_feats = None                            # DFlash: context rows not yet in its cache
        B = self.dflash.block_size if mode == "dflash" else 0
        mask = mx.full((B - 1,), self.dflash.mask_token_id, mx.int32) if mode == "dflash" else None
        failed = False
        t1 = time.perf_counter()
        try:
            while len(out) < max_new:
                pos = eng.state.pos
                room = self.max_len - pos - 1
                if room < 1:
                    return
                if mode == "dflash" and room >= B:
                    drafts = self.dflash.forward(new_feats, mx.concatenate([tok, mask])).astype(mx.int32)
                elif mode == "mtp" and room >= 2:
                    k = min(self.k_mtp, room - 1)
                    toks_in, h_in = pending
                    self.mtp.state.pos = pos + 1 - toks_in.shape[0]
                    hn = self.mtp.hidden(toks_in, h_in)[-1:]
                    ds = []
                    for i in range(k):
                        ds.append(self.mtp.draft(hn).astype(mx.int32))
                        if i + 1 < k:
                            hn = self.mtp.hidden(ds[-1], hn)
                    drafts = mx.concatenate(ds)
                else:
                    # no room for a draft: put the rows it would have consumed into its
                    # cache now, so the draft cache stays in step with the engine
                    if mode == "dflash" and new_feats is not None:
                        self.dflash.add_context(new_feats)
                    elif mode == "mtp":
                        toks_in, h_in = pending
                        self.mtp.state.pos = pos + 1 - toks_in.shape[0]
                        self.mtp.hidden(toks_in, h_in)
                    drafts = None
                seq = tok if drafts is None else mx.concatenate([tok, drafts])
                n, pred, H = eng.verify(seq, pick)
                feats = eng.features
                extra = (self.dflash.arrays() if mode == "dflash" else
                         self.mtp.state.arrays() if mode == "mtp" else [])
                mx.eval(n, pred, H, eng.state.arrays(), extra, *([feats] if feats is not None else []))
                n = int(n.item())
                new = seq[:n + 1].tolist()
                # a stop id inside the accepted run: commit only up to it, so the state
                # (and the end snapshot) describe exactly the tokens returned. Deferred
                # GDN commit makes a shorter commit free; rows beyond are overwritten later.
                cut = next((i for i, t in enumerate(new) if t in self.stop_ids), None)
                j = n if cut is None else cut
                eng.commit(j)
                new = new[:j + 1]
                tok = pred[j:j + 1].astype(mx.int32)
                pending = (mx.concatenate([seq[1:j + 1], tok]), H[:j + 1])
                new_feats = feats[:j + 1] if feats is not None else None
                last = H[j:j + 1]
                steps += 1
                out.extend(new)
                stats.new_tokens, stats.steps = len(out), steps
                yield new
                if cut is not None:
                    return
        except Exception:
            failed = True
            raise
        finally:
            stats.decode_s = time.perf_counter() - t1
            if failed:                               # the state may be half a step ahead: trust nothing
                self.forget()
            else:
                self._finish(mode, ids, out, pending, new_feats, last)

    def _finish(self, mode, ids, out, pending, new_feats, last):
        """Leave the drafts' caches in step with the engine, then snapshot the end."""
        eng = self.engine
        if mode == "dflash" and new_feats is not None:
            self.dflash.add_context(new_feats)
            mx.eval(self.dflash.arrays())
        if mode == "mtp" and out:
            toks_in, h_in = pending
            self.mtp.state.pos = eng.state.pos + 1 - toks_in.shape[0]
            if toks_in.shape[0] > 1:
                self.mtp.hidden(toks_in[:-1], h_in[:-1])
                mx.eval(self.mtp.state.arrays())
        if out:
            assert eng.state.pos == len(ids) + len(out)
            self.ids = ids + out
            self.snaps.append(self._take(mode, last))
