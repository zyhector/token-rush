"""Speculative decoding on the MLX backend: the shipped MTP head as the draft,
verification in Engine.verify, one host sync per step.

Same semantics as tokenrush/mtp.py and tokenrush/spec.py (vLLM's
Qwen3_5MultiTokenPredictor; the chain feeds its own normed hidden forward).
What differs is the execution: the K-draft chain and the verify step are one
lazy MLX graph, evaluated once; the only value that crosses to the host is n,
the number of accepted drafts, which the position bookkeeping needs.
"""
import time
from dataclasses import dataclass, replace
from typing import Optional

import mlx.core as mx

from ..config import ModelConfig
from .model import Engine, State, attn_forward, mlp_forward, rmsnorm
from .weights import AttnWeights, LayerWeights, Linear, QLinear, _leaves

CANDIDATES = 64


# ------------------------------------------------------------------ sampling


@dataclass
class Sampling:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = CANDIDATES

    def __call__(self, logits: mx.array) -> mx.array:
        """logits [M, V] -> tokens [M], one independent draw per row (argmax when greedy).
        Candidates are the top 64, as in tokenrush/sample.py."""
        if self.temperature <= 0:
            return mx.argmax(logits, -1)
        lf = logits.astype(mx.float32)
        idx = mx.argpartition(-lf, CANDIDATES - 1, axis=-1)[:, :CANDIDATES]
        vals = mx.take_along_axis(lf, idx, -1)
        order = mx.argsort(-vals, -1)
        vals, idx = mx.take_along_axis(vals, order, -1), mx.take_along_axis(idx, order, -1)
        p = mx.softmax(vals / self.temperature, -1)
        cum = mx.cumsum(p, -1)
        rank = mx.arange(CANDIDATES)[None, :]
        keep = ((cum - p) < self.top_p) & (rank < self.top_k)
        p = mx.where(keep, p, 0.0)
        cum = mx.cumsum(p, -1)
        u = mx.random.uniform(shape=(logits.shape[0], 1)) * cum[:, -1:]
        pick = mx.minimum((cum < u).sum(-1), CANDIDATES - 1)
        return mx.take_along_axis(idx, pick[:, None], -1)[:, 0]


# ------------------------------------------------------------------ the MTP head


@dataclass
class MTPWeights:
    fc: object
    layer: LayerWeights
    gains: list            # fp32 (1 + w): pre_norm_emb, pre_norm_hidden, ln1, ln2, q_norm, k_norm, norm


def build_mtp(mtp: dict, int4: bool = True) -> MTPWeights:
    """mtp: 'mtp.*' name -> bf16 array. int4 quantizes the projections on load
    (0.85 -> 0.21 GB read per draft), the same group-128 affine format as the body."""
    def lin(names):
        w = mx.concatenate([mtp[n] for n in names], axis=0)
        return QLinear(*mx.quantize(w, group_size=128, bits=4)) if int4 else Linear(w)

    p = "mtp.layers.0."
    m = p + "self_attn."
    mixer = AttnWeights(qkv=lin([m + "q_proj.weight", m + "k_proj.weight", m + "v_proj.weight"]),
                        o=lin([m + "o_proj.weight"]), q_norm_w=mtp[m + "q_norm.weight"], k_norm_w=mtp[m + "k_norm.weight"])
    layer = LayerWeights(ln1=mtp[p + "input_layernorm.weight"], ln2=mtp[p + "post_attention_layernorm.weight"],
                         mixer=mixer, gate_up=lin([p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"]),
                         down=lin([p + "mlp.down_proj.weight"]))
    g = lambda n: 1.0 + mtp[n].astype(mx.float32)
    gains = [g("mtp.pre_fc_norm_embedding.weight"), g("mtp.pre_fc_norm_hidden.weight"),
             g(p + "input_layernorm.weight"), g(p + "post_attention_layernorm.weight"),
             g(m + "q_norm.weight"), g(m + "k_norm.weight"), g("mtp.norm.weight")]
    return MTPWeights(fc=lin(["mtp.fc.weight"]), layer=layer, gains=gains)


class MTPHead:
    """The MTP block with its own one-layer KV cache. Rows are the target's positions:
    the pair (token at p, target hidden at p-1) sits at row p."""

    def __init__(self, engine: Engine, mtp: dict, int4: bool = True, draft_vocab: Optional[mx.array] = None):
        """draft_vocab: token ids the drafts may propose; the drafts' lm_head is those
        rows of the int4 lm_head, so each draft reads a fraction of the 0.68 GB head. A
        token outside it is never drafted (only rejected); the output is unaffected."""
        self.cfg = engine.cfg
        self.engine = engine
        self.w = build_mtp(mtp, int4)
        self.cfg1 = replace(self.cfg, layer_types=("full_attention",), n_layers=1)
        self.state = State.new(self.cfg1, engine.max_len)
        self.draft_map = draft_vocab
        self.head = engine.w.lm_head if draft_vocab is None else engine.w.lm_head.rows(draft_vocab)
        mx.eval(_leaves(self.w), _leaves(self.head), self.state.arrays())

    def reset(self):
        self.state = State.new(self.cfg1, self.engine.max_len)

    def hidden(self, toks: mx.array, target_hidden: mx.array) -> mx.array:
        """toks [T], target_hidden [T, hidden] at rows state.pos.. -> normed hidden [T, hidden]."""
        cfg, g = self.cfg, self.w.gains
        e = rmsnorm(self.engine.w.embed[toks], g[0], cfg.eps)
        h = rmsnorm(target_hidden, g[1], cfg.eps)
        x = self.w.fc(mx.concatenate([e, h], axis=-1))
        lw = self.w.layer
        x = x + attn_forward(rmsnorm(x, g[2], cfg.eps), lw.mixer, cfg, self.state, 0, g[4:6])
        x = x + mlp_forward(rmsnorm(x, g[3], cfg.eps), lw)
        self.state.pos += toks.shape[0]
        return rmsnorm(x, g[6], cfg.eps)

    def draft(self, h: mx.array) -> mx.array:
        d = mx.argmax(self.head(h), -1)
        return d if self.draft_map is None else self.draft_map[d]


# ------------------------------------------------------------------ generation


def prime(engine: Engine, mtp: Optional[MTPHead], ids: list, chunk: int = 2048):
    """Prefill the target (keeping every row's hidden for the MTP's prompt pass) and the
    MTP cache rows 1..T-1. -> the pending pair for row T: (first token [1], hidden [1, H])."""
    engine.reset()
    if mtp is not None:
        mtp.reset()
        mtp.state.pos = 1
    ids_a = mx.array(ids)
    prev_last = None
    for s in range(0, len(ids), chunk):
        H = engine.forward_hidden(ids_a[s:s + chunk])
        if mtp is not None:
            # pairs (ids[p], H[p-1]) for rows p in this chunk except row 0 of the prompt
            hs = H[:-1] if prev_last is None else mx.concatenate([prev_last, H[:-1]])
            ts = ids_a[s + 1:s + H.shape[0]] if prev_last is None else ids_a[s:s + H.shape[0]]
            if ts.shape[0]:
                mtp.hidden(ts, hs)
        prev_last = H[-1:]
        mx.eval(H, engine.state.arrays(), *(mtp.state.arrays() if mtp else []))
    first = mx.argmax(engine.w.lm_head(prev_last), -1)
    mx.eval(first)
    return first, prev_last


def generate(engine: Engine, mtp: Optional[MTPHead], ids: list, max_new: int, stop_ids=(), K: int = 3,
             sampling: Sampling = None, dynamic=None, on_tokens=None):
    """Speculative generation from prompt ids. K = 0 is plain decode through the same
    verify path. dynamic=(Kmin, Kmax): each step's depth is clamp(n_prev + 2, Kmin, Kmax),
    as in the CUDA engine. on_tokens(list) is called with each step's new tokens.
    -> (tokens, stats)."""
    sampling = sampling or Sampling()
    pick = None if sampling.temperature <= 0 else sampling
    t0 = time.perf_counter()
    tok, h_last = prime(engine, mtp if K or dynamic else None, ids)
    if sampling.temperature > 0:
        tok = sampling(engine.w.lm_head(h_last))
    t_prefill = time.perf_counter() - t0
    pending = (tok, h_last)
    out, steps, n_prev, hist = [], 0, K, {}
    t1 = time.perf_counter()
    while len(out) < max_new and engine.state.pos < engine.max_len:
        k = K if dynamic is None else max(dynamic[0], min(dynamic[1], n_prev + 2))
        k = max(0, min(k, engine.max_len - engine.state.pos - 1))
        pos = engine.state.pos
        toks_in, h_in = pending
        drafts = []
        if k:
            mtp.state.pos = pos + 1 - toks_in.shape[0]       # the pending true pairs end at row pos
            hn = mtp.hidden(toks_in, h_in)[-1:]
            for i in range(k):
                d = mtp.draft(hn)
                drafts.append(d)
                if i + 1 < k:
                    hn = mtp.hidden(d, hn)
        cur = toks_in[-1:]
        seq = mx.concatenate([cur] + drafts) if drafts else cur
        n, pred, H = engine.verify(seq, pick)
        mx.eval(n, pred, H, engine.state.arrays(), *(mtp.state.arrays() if mtp else []))
        n = int(n.item())
        engine.commit(n)
        new = seq[:n + 1].tolist()
        nxt = pred[n:n + 1]
        out.extend(new)
        steps += 1
        hist[k] = hist.get(k, 0) + 1
        n_prev = n
        pending = (mx.concatenate([seq[1:n + 1], nxt]), H[:n + 1])
        if on_tokens:
            on_tokens(new)
        if any(t in stop_ids for t in new):
            out = out[:next(i for i, t in enumerate(out) if t in stop_ids) + 1]
            break
    out = out[:max_new]
    dt = time.perf_counter() - t1
    return out, {"prefill_s": t_prefill, "decode_s": dt, "tok_s": len(out) / dt, "steps": steps,
                 "tokens_per_step": len(out) / max(steps, 1), "ms_per_step": 1e3 * dt / max(steps, 1),
                 "depths": hist}
