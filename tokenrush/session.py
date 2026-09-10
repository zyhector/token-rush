"""A resident engine that serves one request at a time and reuses the context
between requests.

`Session.generate(ids, ...)` is a generator yielding the tokens each step
commits; it drives the same graphs `run.py` uses (the DFlash2 block draft or
the MTP chain inside the spec graph, or raw graphed decode). What it adds is
**prefix reuse**: a chat client resends the whole conversation on every turn,
and at 100k tokens a re-prefill costs a minute. The engine's state is explicit
and position-addressed (contiguous KV, a conv ring by position, a fixed-size
recurrent state, the drafts' own position-addressed caches), so a *snapshot* of
the state at position P is small: the recurrent state (151 MB fp32), the conv
ring (16 MB), the post-final-norm hidden at P-1 (10 KB) and the draft cursors.
Two are kept per request — at the end of the prompt and at the end of the
generation — and the next request prefills only the tokens past the longest
snapshot that is a prefix of it. Everything else (KV rows, the MTP head's rows,
the DFlash ring) is left in place and truncated by resuming at P.

Nothing here changes what is generated: a resumed prefill is the same chunked
prefill from P instead of 0, and the tests check the state it leaves against a
cold one.
"""
import time
from dataclasses import dataclass

import torch

from .sample import sample
from .spec import cjk_share

MIN_REUSE = 16       # do not bother restoring a snapshot shorter than this
SHORT_PER_SLOT = 128 # a delta up to this many tokens per fused row goes through the fused M-row path


@dataclass
class Snapshot:
    pos: int                 # the state describes tokens ids[:pos]; the next token goes at pos
    mode: str                # "dflash" | "mtp" | "raw": which draft cache is valid up to pos
    rec: torch.Tensor        # [n_gdn, HV, K, V] fp32, the committed recurrent state
    conv: torch.Tensor       # the conv ring
    hidden: torch.Tensor     # [1, hidden] bf16, post-final-norm hidden of position pos-1


@dataclass
class Stats:
    prompt_tokens: int = 0
    reused_tokens: int = 0
    prefill_tokens: int = 0
    prefill_s: float = 0.0
    new_tokens: int = 0
    decode_s: float = 0.0
    steps: int = 0

    def as_dict(self):
        d = dict(self.__dict__)
        d["decode_tok_s"] = self.new_tokens / self.decode_s if self.decode_s else 0.0
        d["accepted_per_step"] = self.new_tokens / self.steps if self.steps else 0.0
        return d


class Session:
    def __init__(self, engine, tok, cfg, dflash=None, mtp=None, chunk: int = 4096, kmin: int = 3, kmax: int = 4):
        """engine: captured (decode graph, plus the spec graphs of the drafts given).
        dflash: an attached DFlashDraft; mtp: an attached MTPHead; either may be None."""
        self.engine = engine
        self.tok = tok
        self.cfg = cfg
        self.dflash = dflash
        self.mtp = mtp
        self.chunk = chunk
        self.kmin, self.kmax = kmin, kmax
        self.stop_ids = set(cfg.eos_ids) | {tok.eos_token_id}
        self.ids = []            # the token sequence the state describes (prompt + committed output)
        self.snaps = []          # Snapshot, ascending pos
        self.max_len = engine.state.max_len
        self.stats = Stats()

    # ------------------------------------------------------------ draft choice

    def pick_mode(self, text: str, mode: str = "auto") -> str:
        if mode != "auto":
            return mode
        if self.dflash is not None and self.mtp is not None:
            return "mtp" if cjk_share(text) >= 0.2 else "dflash"
        return "dflash" if self.dflash is not None else ("mtp" if self.mtp is not None else "raw")

    # ------------------------------------------------------------ snapshots

    def _take(self, pos: int, mode: str, rec_slot: int, hidden: torch.Tensor) -> Snapshot:
        st = self.engine.state
        return Snapshot(pos, mode, st.rec[rec_slot].clone(), st.conv.clone(), hidden.detach().clone().view(1, -1))

    def _restore(self, s: Snapshot):
        st = self.engine.state
        st.pos = s.pos
        st.pos_t.fill_(s.pos)
        st.rec[0].copy_(s.rec)
        st.slot.zero_()
        st.slot_h = 0
        st.conv.copy_(s.conv)
        if s.mode == "dflash":
            self.dflash.ctx = s.pos
        elif s.mode == "mtp":
            self.mtp.set_pos(s.pos)

    def _prefix_len(self, ids) -> int:
        n = min(len(self.ids), len(ids))
        i = 0
        while i < n and self.ids[i] == ids[i]:
            i += 1
        return i

    def forget(self):
        self.ids, self.snaps = [], []

    # ------------------------------------------------------------ prefill

    def _prefill(self, ids: torch.Tensor, start: int, mode: str, prev_hidden):
        """Run the engine (and the mode's draft) over ids[start:], chunked; returns the
        post-final-norm hidden of the last position [1, hidden]. prev_hidden: the hidden
        at start-1 (None when start == 0), which the MTP head's row `start` pairs with."""
        eng = self.engine
        last = prev_hidden
        T = ids.numel()
        # The eager path (a chunk longer than the fused row count) dequantizes every
        # weight before its GEMMs: ~1.8 s per call on the 27B, whatever the length,
        # amortized over a 4096-token chunk but not over a 20-token tool result. The
        # fused M-row path costs ~1.4 ms/token at 8 rows, so a short remainder is
        # prefilled that way (crossover ~1200 tokens at 8 rows).
        rows = eng.state.n_slots if eng.fused else 1
        short = SHORT_PER_SLOT * rows
        s = start
        while s < T:
            e = min(T, s + (rows if T - s <= short else self.chunk))
            h = eng.forward_hidden(ids[s:e])                       # [e-s, hidden]; writes eng.feat_last / feat_buf
            if mode == "dflash":
                self.dflash.prime(eng.features_of_last_chunk(e - s))
            elif mode == "mtp":
                # row p pairs (ids[p], hidden at p-1); row 0 has no pair
                if last is None:
                    self.mtp.set_pos(1)
                    if e - s > 1:
                        self.mtp.forward(ids[s + 1:e], h[:-1], logits=False)
                else:
                    self.mtp.set_pos(s)
                    self.mtp.forward(ids[s:e], torch.cat([last, h[:-1]]), logits=False)
            last = h[-1:]
            s = e
        return last

    # ------------------------------------------------------------ generation

    @torch.no_grad()
    def generate(self, ids, max_new: int, mode: str = "dflash", temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int = 64, seed=None):
        """Yields the list of token ids committed by each step (1..K+1 tokens; the last
        list may end with a stop id). Close the generator early to stop; the state is
        consistent at every step boundary and the end-of-generation snapshot is taken
        in either case. self.stats describes the run afterwards."""
        eng = self.engine
        st = eng.state
        assert mode in ("dflash", "mtp", "raw")
        assert mode != "dflash" or self.dflash is not None
        assert mode != "mtp" or self.mtp is not None
        ids = [int(t) for t in ids]
        T = len(ids)
        if T >= self.max_len:
            raise ValueError(f"prompt of {T} tokens does not fit the {self.max_len}-token context")
        stats = self.stats = Stats(prompt_tokens=T)
        eng.sampling.set(temperature, top_p, top_k)
        if seed is not None:
            torch.manual_seed(seed)

        # 1. the longest snapshot that is a prefix of this prompt, in this mode
        t0 = time.perf_counter()
        L = self._prefix_len(ids)
        best = None
        for s in self.snaps:
            if s.mode == mode and MIN_REUSE <= s.pos <= L and (best is None or s.pos > best.pos):
                best = s
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
        stats.reused_tokens = start
        stats.prefill_tokens = T - start
        dev_ids = torch.tensor(ids, device=eng.device, dtype=torch.long)
        last = self._prefill(dev_ids, start, mode, prev) if start < T else prev
        # the sequence and snapshots now describe this prompt
        self.ids = list(ids)
        self.snaps = [self._take(T, mode, st.slot_h, last)]
        prompt_snap = self.snaps[0]
        # 2. the first token, and the spec-step inputs
        nxt = sample(eng.w.lm_head(last), eng.sampling)[0]
        eng.tok.copy_(nxt.view(1))
        if mode == "dflash":
            eng.n_accepted.fill_(-1)
        elif mode == "mtp":
            eng.n_accepted.zero_()
            eng.spec_hidden[0].copy_(last[0])
        torch.cuda.synchronize()
        stats.prefill_s = time.perf_counter() - t0

        # 3. decode
        out = []
        tok_prev = int(nxt)
        n = 0
        j = None            # index, within the last step's committed tokens, of the last token kept
        t1 = time.perf_counter()
        try:
            if mode == "raw":
                tok = nxt
                while len(out) < max_new and st.pos < self.max_len:
                    t = int(tok)
                    out.append(t)
                    stats.new_tokens = len(out)
                    yield [t]
                    if t in self.stop_ids:
                        break
                    tok = eng.step()
                    stats.steps += 1
            else:
                kmax = self.kmax if mode == "mtp" else eng.dflash_K
                while len(out) < max_new:
                    if st.pos + kmax + 1 > self.max_len:
                        break
                    if mode == "mtp":
                        k = min(max(n + 2, self.kmin), self.kmax)
                        n = eng.spec_step(k)
                    else:
                        n = eng.spec_step_dflash()
                    stats.steps += 1
                    new = [tok_prev] + eng.drafts[:n].tolist()
                    tok_prev = int(eng.tok)
                    j = n
                    for i, t in enumerate(new):
                        if t in self.stop_ids:
                            new, j = new[:i + 1], i
                            break
                    out.extend(new)
                    stats.new_tokens = len(out)
                    yield new
                    if j < n or (new and new[-1] in self.stop_ids):
                        break
        finally:
            torch.cuda.synchronize()
            stats.decode_s = time.perf_counter() - t1
            self.ids = ids + out
            if mode != "raw" and j is not None:
                self._finish_spec(mode, T, out, n, j, prompt_snap)

    def _finish_spec(self, mode: str, T: int, out, n: int, j: int, prompt_snap: Snapshot):
        """After the last spec step, which processed M = K+1 tokens at pos0 .. pos0+K
        (pos0 = the position of that step's first committed token) and committed
        n+1 of them: keep the state after token j (the last one kept), complete the
        draft's cache to it, and snapshot there."""
        eng, st = self.engine, self.engine.state
        pos0 = st.pos - n - 1
        P = pos0 + j + 1
        assert P == T + len(out), (P, T, len(out))
        hidden = eng.spec_hidden[j:j + 1]                              # position pos0+j = P-1
        if mode == "mtp":
            # rows <= pos0 hold true pairs; rows pos0+1 .. P-1 pair the kept drafts with
            # the hiddens before them
            if j > 0:
                self.mtp.set_pos(pos0 + 1)
                self.mtp.forward(eng.drafts[:j], eng.spec_hidden[:j], logits=False)
            self.mtp.set_pos(P)
        else:
            # the draft's context cache ends at pos0 (the graphed step tracks that on the
            # device; the host mirror d.ctx is stale after replays); feat_buf rows 0..K are
            # the features of positions pos0 .. pos0+K
            d = self.dflash
            d.ctx = pos0
            d.prime(eng.feat_buf[:j + 1])
            assert d.ctx == P
            # generating past the prompt overwrote ring rows that the prompt-end
            # snapshot's window needs once the output is longer than the slack
            if P - prompt_snap.pos > d.ring - d.w.window - d.w.block_size - 8:
                self.snaps = [s for s in self.snaps if s is not prompt_snap]
        st.pos = P
        st.pos_t.fill_(P)
        self.snaps.append(self._take(P, mode, j, hidden))
        st.rec[0].copy_(st.rec[j])
        st.slot.zero_()
        st.slot_h = 0
