"""The text path of Qwen3.8-27B as plain functions over weight containers.

Layers are pure functions of (weights, state, activations); there is no
nn.Module and no autograd. Activations are [T, hidden] with bs=1 implicit.
"""
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn.functional as F

from . import fused as fused_k
from . import ops
from .config import ModelConfig
from .quant import Linear, QLinear
from .sample import SamplingParams, sample
from .state import State

Lin = Union[Linear, QLinear]


@dataclass
class GDNWeights:
    in_qkvz: Lin           # hidden -> 2*qk_dim + val_dim (conv path) | val_dim (gate z), one GEMV
    in_ba: torch.Tensor    # [2*HV, hidden] bf16: b rows then a rows
    conv_w: torch.Tensor   # [conv_dim, K] bf16
    A: torch.Tensor        # [HV] fp32, = -exp(A_log)
    dt_bias: torch.Tensor  # [HV] fp32
    norm_w: torch.Tensor   # [gdn_v_dim] bf16
    out: Lin               # val_dim -> hidden


@dataclass
class AttnWeights:
    qkv: Lin               # hidden -> n_heads*head_dim*2 (query|gate per head) | kv_heads*head_dim (k) | same (v)
    o: Lin
    q_norm_w: torch.Tensor  # [head_dim]
    k_norm_w: torch.Tensor


@dataclass
class LayerWeights:
    ln1: torch.Tensor
    ln2: torch.Tensor
    mixer: Union[GDNWeights, AttnWeights]
    gate_up: Lin           # hidden -> 2*ffn: gate rows then up rows
    down: Lin


@dataclass
class ModelWeights:
    embed: torch.Tensor    # [vocab, hidden] bf16
    layers: list
    final_norm: torch.Tensor
    lm_head: Lin

    @property
    def nbytes(self):
        n = self.embed.numel() * 2 + self.lm_head.nbytes
        for l in self.layers:
            n += l.gate_up.nbytes + l.down.nbytes
            m = l.mixer
            if isinstance(m, GDNWeights):
                n += m.in_qkvz.nbytes + m.out.nbytes + m.in_ba.numel() * 2 + m.conv_w.numel() * 2
            else:
                n += m.qkv.nbytes + m.o.nbytes
        return n


# ----------------------------------------------------------------- layers


def gdn_forward(x: torch.Tensor, w: GDNWeights, cfg: ModelConfig, state: State, slot: int,
                fused: bool = False) -> torch.Tensor:
    T = x.shape[0]
    qkvz = w.in_qkvz(x)                                                              # [T, conv_dim + val_dim]
    if fused and T <= state.n_slots:
        y = fused_k.gdn_step_fused(qkvz, x, w.in_ba, state.conv[slot], w.conv_w, w.A, w.dt_bias,
                                   state.rec[:, slot], state.slot, w.norm_w, state.pos_t, cfg, cfg.eps)
        return w.out.partials(y)                                                     # [S, T, hidden] fp32
    ba = F.linear(x, w.in_ba)                                                        # [T, 2*HV]
    qkv, z = torch.split(qkvz, [cfg.conv_dim, cfg.gdn_val_dim], dim=-1)
    b, a = torch.split(ba, [cfg.gdn_v_heads, cfg.gdn_v_heads], dim=-1)
    if T == 1:
        qkv = ops.conv_step(qkv, state.conv[slot], w.conv_w, state.pos_t)
    else:
        qkv = ops.conv_prefill(qkv, state.conv[slot], w.conv_w, state.pos)
    q, k, v = torch.split(qkv, [cfg.gdn_qk_dim, cfg.gdn_qk_dim, cfg.gdn_val_dim], dim=-1)
    q = q.view(T, cfg.gdn_k_heads, cfg.gdn_k_dim)
    k = k.view(T, cfg.gdn_k_heads, cfg.gdn_k_dim)
    v = v.view(T, cfg.gdn_v_heads, cfg.gdn_v_dim)
    beta = torch.sigmoid(b)                                      # [T, HV] bf16
    g = w.A * F.softplus(a.float() + w.dt_bias)                  # [T, HV] fp32
    # eager paths read the committed slot and leave the result in slot 0
    rec = state.rec_live(slot)
    if T == 1:
        o = ops.gdn_step(q[0], k[0], v[0], g[0], beta[0], rec)[None]
    else:
        o = ops.gdn_prefill(q, k, v, g, beta, rec)
    if state.slot_h != 0:
        state.rec[0, slot].copy_(rec)
    o = ops.gated_rmsnorm(o.reshape(-1, cfg.gdn_v_dim), w.norm_w, z.reshape(-1, cfg.gdn_v_dim), cfg.eps)
    return w.out(o.view(T, cfg.gdn_val_dim))


def attn_forward(x: torch.Tensor, w: AttnWeights, cfg: ModelConfig, state: State, slot: int,
                 cos: torch.Tensor, sin: torch.Tensor, bucket: int = None, fused: bool = False) -> torch.Tensor:
    """bucket: if given (decode only), attend over the first `bucket` cache rows with a
    mask derived from the device position, so the whole call is shape-static.
    fused (decode only): prep kernel + flash-decoding over the live length; no bucket."""
    T = x.shape[0]
    pos = state.pos
    hd = cfg.head_dim
    ks = state.k_scale[slot] if state.fp8 else None
    vs = state.v_scale[slot] if state.fp8 else None
    if fused and T <= 8:
        qkv = w.qkv(x)
        q = fused_k.attn_prep(qkv, w.q_norm_w, w.k_norm_w, cos, sin, state.pos_t, state.k[slot], state.v[slot], cfg, ks, vs)
        o = fused_k.attn_decode_fused(q, qkv, state.k[slot], state.v[slot], state.pos_t, cfg, ks, vs)
        return w.o.partials(o)                                                       # [S, T, hidden] fp32
    idx = state.pos_t + torch.arange(T, device=x.device)                          # positions, on device
    kv_dim = cfg.n_kv_heads * hd
    qg, k, v = torch.split(w.qkv(x), [cfg.n_heads * 2 * hd, kv_dim, kv_dim], dim=-1)
    qg = qg.view(T, cfg.n_heads, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:]
    gate = gate.reshape(T, cfg.n_heads * hd)
    q = ops.rmsnorm(q, w.q_norm_w, cfg.eps).transpose(0, 1)                       # [Hq, T, D]
    k = ops.rmsnorm(k.view(T, cfg.n_kv_heads, hd), w.k_norm_w, cfg.eps).transpose(0, 1)
    v = v.view(T, cfg.n_kv_heads, hd).transpose(0, 1)
    c, s_ = cos.index_select(0, idx), sin.index_select(0, idx)
    q = ops.apply_rope(q, c, s_)
    k = ops.apply_rope(k, c, s_)
    if state.fp8:
        # fp8 cache: write quantized, attend with the Triton prefill kernel (SDPA cannot read it)
        # (short fused steps never reach here; this is the T > 8 prefill path)
        fused_k.kv_write_prefill(k, v, state, slot, pos)
        o = fused_k.attn_prefill_fused(q.transpose(0, 1).contiguous(), state.k[slot], state.v[slot], pos, cfg, ks, vs)
        o = o.transpose(0, 1)                                                         # [Hq, T, D]
    else:
        state.k[slot].index_copy_(1, idx, k)
        state.v[slot].index_copy_(1, idx, v)
        if bucket is not None:
            assert T == 1
            o = ops.attn_decode_bucket(q, state.k[slot, :, :bucket], state.v[slot, :, :bucket], state.pos_t)
        elif T == 1:
            o = ops.attn_decode(q, state.k[slot, :, :pos + 1], state.v[slot, :, :pos + 1])
        else:
            o = ops.attn_prefill(q, state.k[slot, :, :pos + T], state.v[slot, :, :pos + T], pos)
    o = o.transpose(0, 1).reshape(T, cfg.n_heads * hd) * torch.sigmoid(gate)
    return w.o(o)


def mlp_forward(x: torch.Tensor, w: LayerWeights, fused: bool = False) -> torch.Tensor:
    gu = w.gate_up(x)
    if fused:
        return w.down.partials(fused_k.silu_mul(gu))                                 # [S, T, hidden] fp32
    gate, up = torch.chunk(gu, 2, dim=-1)
    return w.down(F.silu(gate) * up)


# ----------------------------------------------------------------- engine


class Engine:
    """Holds weights, state and rotary tables; runs prefill eagerly and decode
    either eagerly or as one CUDA graph per context bucket."""

    def __init__(self, cfg: ModelConfig, weights: ModelWeights, max_len: int, device="cuda", fused: bool = True,
                 kv_dtype=torch.bfloat16, max_spec: int = 0, consistent: bool = False):
        """max_spec: the longest draft chain a verify step must hold (K); the recurrent
        state gets K+1 snapshot slots and the ring must exceed K+3 columns.
        consistent: run single-token steps on the same M-row GEMV kernel the verify
        step uses, so speculative and raw greedy output are bit-identical (raw decode
        is ~9% slower on that kernel; without it the two agree except at bf16 near-ties)."""
        self.cfg = cfg
        self.w = weights
        import tokenrush.quant as _q
        _q.ROWS_FOR_ONE = consistent
        self.fused = fused            # Phase 2 fused kernels on the short (T <= 8) path
        self.device = torch.device(device)
        from .state import RING
        assert max_spec + 3 < RING and max_spec + 1 <= 8
        self.max_spec = max_spec
        self.state = State(cfg, max_len, self.device, kv_dtype=kv_dtype, n_slots=max_spec + 1)
        if self.state.fp8:
            assert fused, "the fp8 cache is read only by the fused kernels"
        self.cos, self.sin = ops.rope_table(max_len, cfg.rotary_dim, cfg.rope_theta, self.device)
        # graph I/O: the token read at the start of a step and written at its end
        self.tok = torch.zeros(1, device=self.device, dtype=torch.long)
        self.sampling = SamplingParams(self.device)     # greedy unless set()
        self.logits = None            # [1, vocab], written by the last graphed step
        self.graphs = {}              # bucket -> CUDAGraph
        self.pool = None
        # verify-step I/O: the committed-but-unprocessed token followed by K drafts,
        # the logits of all K+1 positions, the number of accepted drafts
        self.spec_toks = torch.zeros(max_spec + 1, device=self.device, dtype=torch.long)
        self.spec_logits = None       # [K+1, vocab]
        self.spec_hidden = None       # [K+1, hidden], post-final-norm (the MTP head's input)
        self.n_accepted = torch.zeros(1, device=self.device, dtype=torch.long)
        self.spec_graphs = {}         # K -> CUDAGraph
        self.trace = None             # set to a list to collect the residual stream per layer (eager only)
        self.last_hidden = None       # post-final-norm hidden of the last forward

    def layer(self, li: int) -> LayerWeights:
        return self.w.layers[li]

    def _body(self, tokens: torch.Tensor, bucket=None, all_logits=False, no_head=False) -> torch.Tensor:
        """The forward pass without the position bookkeeping. no_head: return the
        post-final-norm hidden of every row instead of logits."""
        cfg, st = self.cfg, self.state
        x = self.w.embed[tokens]
        if self.trace is not None:
            self.trace.append(x.clone())
        fused = self.fused and tokens.shape[0] <= self.state.n_slots
        h = None                                     # pending residual contribution
        for li in range(cfg.n_layers):
            lw = self.layer(li)
            if fused:
                x, n = fused_k.add_rmsnorm(x, h, lw.ln1, cfg.eps)
            else:
                x = x if h is None else x + h
                n = ops.rmsnorm(x, lw.ln1, cfg.eps)
            if cfg.layer_types[li] == "linear_attention":
                h = gdn_forward(n, lw.mixer, cfg, st, st.gdn_slot[li], fused)
            else:
                h = attn_forward(n, lw.mixer, cfg, st, st.attn_slot[li], self.cos, self.sin, bucket, fused)
            if fused:
                x, n = fused_k.add_rmsnorm(x, h, lw.ln2, cfg.eps)
            else:
                x = x + h
                n = ops.rmsnorm(x, lw.ln2, cfg.eps)
            h = mlp_forward(n, lw, fused)
            if self.trace is not None:
                hs = h.sum(0).to(x.dtype) if h.dim() == 3 else h                     # split-K partials [S, T, N]
                self.trace.append(x + hs)
        if fused:                                    # short step: all T rows are wanted; h may be [S, T, N]
            _, n = fused_k.add_rmsnorm(x, h, self.w.final_norm, cfg.eps)
        else:
            if not all_logits:
                x, h = x[-1:], h[-1:]
            n = ops.rmsnorm(x + h, self.w.final_norm, cfg.eps)
        self.last_hidden = n                          # post-final-norm, [T or 1, hidden]: the MTP head's input
        if no_head:
            return n
        return self.w.lm_head(n)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, all_logits: bool = False) -> torch.Tensor:
        """Eager. tokens [T] on device at positions state.pos .. state.pos+T-1; advances
        the state by T. Returns logits [1, vocab] for the last token (or [T, vocab])."""
        T = tokens.shape[0]
        assert self.state.pos + T <= self.state.max_len, "context exceeds the preallocated cache"
        logits = self._body(tokens, all_logits=all_logits)
        if self.fused and T <= self.state.n_slots:
            # the fused GDN kernel wrote slots 0..T-1; the state after the last token is slot T-1
            self.state.slot.fill_(T - 1)
            self.state.slot_h = T - 1
            if not all_logits:
                logits = logits[-1:]
                self.last_hidden = self.last_hidden[-1:]
        else:
            self.state.slot.zero_()
            self.state.slot_h = 0
        self.state.advance(T)
        return logits

    # ------------------------------------------------------------ graphs

    def _graph_step(self, bucket: int):
        """One decode step, shape-static: read self.tok, write self.logits and the next
        token back into self.tok, advance the device position. Captured per bucket."""
        logits = self._body(self.tok, bucket=bucket)
        self.logits.copy_(logits)
        self.tok.copy_(sample(logits, self.sampling))
        self.state.pos_t += 1
        self.state.slot.zero_()

    def _verify_step(self, K: int):
        """One greedy verify step for K drafts, shape-static: process spec_toks[:K+1]
        (the committed token and K drafts) at pos.., compare each position's argmax
        with the next draft, commit the accepted prefix: n_accepted, tok (the first
        wrong or bonus token), pos += n+1, recurrent-state slot = n."""
        M = K + 1
        logits = self._body(self.spec_toks[:M])                    # [M, vocab]
        self.spec_logits[:M].copy_(logits)
        self.spec_hidden[:M].copy_(self.last_hidden)
        # greedy: the argmax per position. sampling (temperature > 0): one draw per
        # position from the target's own distribution; a draft is accepted only when
        # it equals the draw, and the draw itself is what gets committed, so the
        # output distribution is exactly the target's, drafts or not.
        pred = sample(logits, self.sampling)                       # [M]
        match = (pred[:K] == self.spec_toks[1:M]).long()
        n = match.cumprod(0).sum().view(1)                         # leading accepted drafts, 0..K (device)
        self.n_accepted.copy_(n)
        self.tok.copy_(pred.gather(0, n))                          # no host read: graph-safe
        self.state.pos_t += n + 1
        self.state.slot.copy_(n)

    @staticmethod
    def buckets_for(max_len: int, smallest: int = 1024):
        b, out = smallest, []
        while b < max_len:
            out.append(b)
            b *= 2
        return out + [max_len]

    # ------------------------------------------------- speculative step

    def attach_mtp(self, mtp):
        """The MTP head whose draft chain runs inside the spec graph."""
        self.mtp = mtp
        K = self.max_spec
        self.drafts = torch.zeros(max(K, 1), device=self.device, dtype=torch.long)
        self.draft_hidden = torch.empty(1, self.cfg.hidden, device=self.device, dtype=torch.bfloat16)

    def _spec_step(self, K: int):
        """Draft chain + verify in one shape-static step. Enters with: tok = the token
        to process at pos (device), n_accepted and drafts from the previous step, and
        spec_hidden[0..K] = the previous verify's target hiddens at pos-n-1 .. pos-1+K-n.

        A. MTP pass over K+1 rows at MTP positions pos-n .. pos-n+K: tokens
           [d_1..d_n, tok, pad...] paired with spec_hidden[0..K]; row n is the true
           (tok, hidden at pos-1) pair whose output drafts d'_1.
        B. K-1 chained single-row MTP calls at pos+1.. for d'_2..d'_K.
        C. verify(tok, d'_1..d'_K); commit n, tok, pos, state slot."""
        mtp, st = self.mtp, self.state
        n = self.n_accepted                                                   # [1]
        idx = torch.arange(K + 1, device=self.device)
        prev_drafts = torch.cat([self.drafts[:K], self.tok])                  # [K+1]: d_1..d_K, tok
        toks_a = torch.where(idx < n, self.drafts[:K + 1] if K + 1 <= self.drafts.numel() else prev_drafts,
                             torch.where(idx == n, self.tok.expand(K + 1), self.tok.expand(K + 1)))
        toks_a = torch.where(idx < n, prev_drafts, self.tok.expand(K + 1))    # rows > n carry tok (garbage rows)
        mtp.state.pos_t.copy_(st.pos_t - n)                                   # MTP row of d_1 .. is pos-n
        hn = mtp.hidden_rows(toks_a, self.spec_hidden[:K + 1])                # [K+1, hidden]
        h_sel = hn.index_select(0, n)                                         # row n: after (tok, h_{pos-1})
        d = self.w.lm_head(h_sel).argmax(-1)                                  # d'_1
        mtp.state.pos_t.copy_(st.pos_t + 1)                                   # chain rows pos+1 ..
        new_drafts = [d]
        for _ in range(K - 1):
            hn = mtp.hidden_rows(d, h_sel)
            h_sel = hn
            d = self.w.lm_head(h_sel).argmax(-1)
            new_drafts.append(d)
        self.drafts[:K].copy_(torch.cat(new_drafts))
        # C. verify
        M = K + 1
        self.spec_toks[0].copy_(self.tok[0])
        self.spec_toks[1:M].copy_(self.drafts[:K])
        self._verify_step(K)

    @torch.no_grad()
    def capture_spec(self, K: int, warmup: int = 2):
        """Record the draft-chain + verify graph for K drafts (needs attach_mtp)."""
        assert 1 <= K <= self.max_spec and hasattr(self, "mtp")
        if self.spec_logits is None:
            self.spec_logits = torch.empty(self.max_spec + 1, self.cfg.vocab, device=self.device, dtype=torch.bfloat16)
            self.spec_hidden = torch.empty(self.max_spec + 1, self.cfg.hidden, device=self.device, dtype=torch.bfloat16)
        self.pool = self.pool or torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self.state.reset(); self.mtp.reset(); self.n_accepted.zero_()
                self._spec_step(K)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.state.reset(); self.mtp.reset(); self.n_accepted.zero_()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            self._spec_step(K)
        self.spec_graphs[("spec", K)] = g
        self.state.reset(); self.mtp.reset(); self.n_accepted.zero_(); self.tok.zero_()
        torch.cuda.synchronize()

    def spec_step(self, K: int) -> int:
        """Replay the draft + verify graph; returns n (one host read). The tokens produced
        are [the tok before the call] + drafts[:n] (drafts as left by this call)."""
        self.spec_graphs[("spec", K)].replay()
        n = int(self.n_accepted)
        self.state.pos += n + 1
        self.state.slot_h = n
        return n

    @torch.no_grad()
    def capture_verify(self, Ks, warmup: int = 2):
        """Record one graph per draft length K (K = 0 is a plain greedy step)."""
        cfg, st = self.cfg, self.state
        assert all(0 <= K <= self.max_spec for K in Ks)
        if self.spec_logits is None:
            self.spec_logits = torch.empty(self.max_spec + 1, cfg.vocab, device=self.device, dtype=torch.bfloat16)
            self.spec_hidden = torch.empty(self.max_spec + 1, cfg.hidden, device=self.device, dtype=torch.bfloat16)
        self.pool = self.pool or torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for K in Ks:
                for _ in range(warmup):
                    st.reset()
                    self._verify_step(K)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        for K in Ks:
            st.reset()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._verify_step(K)
            self.spec_graphs[K] = g
        st.reset()
        self.tok.zero_()
        torch.cuda.synchronize()

    def verify(self, drafts: torch.Tensor) -> int:
        """Replay the verify step for these K drafts (device tensor [K]); the committed
        token is self.tok. Returns K after syncing the host position mirror from the
        device count; reads back n_accepted (one sync)."""
        K = drafts.numel()
        self.spec_toks[0].copy_(self.tok[0])
        if K:
            self.spec_toks[1:K + 1].copy_(drafts)
        self.spec_graphs[K].replay()
        n = int(self.n_accepted)
        self.state.pos += n + 1
        self.state.slot_h = n
        return n

    @torch.no_grad()
    def capture(self, buckets=None, warmup: int = 2):
        """Record one graph per context bucket. Runs from a clean state and leaves it
        clean; call once after loading. Warmup runs the step eagerly on a side stream
        (autotuning, workspaces) and is what mutates the state, so it is reset after."""
        cfg, st = self.cfg, self.state
        buckets = buckets or ([st.max_len] if self.fused else self.buckets_for(st.max_len))
        assert all(b <= st.max_len for b in buckets)
        self.logits = torch.empty(1, cfg.vocab, device=self.device, dtype=torch.bfloat16)
        self.pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for b in buckets:
                for _ in range(warmup):
                    self._graph_step(b)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        for b in buckets:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._graph_step(b)
            self.graphs[b] = g
        st.reset()
        self.tok.zero_()
        torch.cuda.synchronize()

    def bucket(self, pos: int) -> int:
        """The smallest captured bucket that holds positions 0..pos."""
        for b in sorted(self.graphs):
            if pos < b:
                return b
        raise ValueError(f"position {pos} exceeds the largest bucket")

    def step(self) -> torch.Tensor:
        """One graphed decode step for the token in self.tok, at position state.pos.
        Returns self.tok, now holding the argmax of this step's logits."""
        self.graphs[self.bucket(self.state.pos)].replay()
        self.state.pos += 1
        self.state.slot_h = 0
        return self.tok

    def prefill(self, tokens: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        logits = None
        for s in range(0, tokens.shape[0], chunk):
            logits = self.forward(tokens[s:s + chunk])
        return logits

    @torch.no_grad()
    def prefill_hidden(self, tokens: torch.Tensor, chunk: int = 4096):
        """Chunked prefill that keeps the post-final-norm hidden of every position (the
        MTP head's prompt input) and applies lm_head to the last one only.
        Returns (logits [1, vocab], hidden [T, hidden])."""
        hs = []
        for s in range(0, tokens.shape[0], chunk):
            t = tokens[s:s + chunk]
            assert self.state.pos + t.shape[0] <= self.state.max_len, "context exceeds the preallocated cache"
            hs.append(self._body(t, all_logits=True, no_head=True))
            self.state.slot.zero_(); self.state.slot_h = 0
            self.state.advance(t.shape[0])
        hidden = torch.cat(hs)
        self.last_hidden = hidden[-1:]
        return self.w.lm_head(hidden[-1:]), hidden

    def decode(self, token: torch.Tensor) -> torch.Tensor:
        return self.forward(token.view(1))

    def reset(self):
        self.state.reset()


class StreamingBF16Engine(Engine):
    """The same engine in bf16, for the engine-correctness gate: 55.6 GB does not
    fit the card, so each layer's weights are built from the HF checkpoint on
    demand and dropped after use. Embedding, final norm and lm_head stay
    resident (5 GB). Eager only. About 25 s per forward pass from disk."""

    def __init__(self, cfg: ModelConfig, src: str, max_len: int, device="cuda"):
        from .weights import HFTensors, build_layer
        self.tensors = HFTensors(src)
        self._build_layer = build_layer
        w = ModelWeights(embed=self.tensors["embed_tokens.weight"].to(device), layers=[None] * cfg.n_layers,
                         final_norm=self.tensors["norm.weight"].to(device),
                         lm_head=Linear(self.tensors["lm_head.weight"].to(device)))
        super().__init__(cfg, w, max_len, device)

    def layer(self, li: int) -> LayerWeights:
        return self._build_layer(self.cfg, self.tensors, li, self.device, backend=None)
