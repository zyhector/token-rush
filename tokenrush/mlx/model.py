"""The text path of Qwen3.8-27B on MLX, mirroring tokenrush/model.py.

Same shape as the CUDA engine: layers are functions of (weights, state,
activations), activations are [T, hidden] with bs=1 implicit, state is explicit
and preallocated. What changes is the execution model. MLX is lazy: a forward
builds a graph and `mx.eval` runs it, so there is no eager-vs-graph split. The
CUDA graph's job (one submission per step, nothing returned to the host) is
done by `mx.compile` over the whole step with the state passed in and out.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx

from ..config import ModelConfig
from . import kernels
from .weights import AttnWeights, GDNWeights, LayerWeights, ModelWeights

RING = 16         # conv ring rows; a verify step of K+1 rows needs K + 4 <= RING
FOLD_ROWS = 16    # attention steps of 2..FOLD_ROWS rows fold the GQA heads into rows


# ------------------------------------------------------------------ norms


def rmsnorm(x, gain, eps):
    """Qwen3.5 RMSNorm with the (1 + w) gain precomputed in fp32 (State.gains)."""
    return mx.fast.rms_norm(x.astype(mx.float32), gain, eps).astype(x.dtype)


def gated_rmsnorm(x, weight, gate, eps):
    """GDN output norm in HF's dtype order: normalize fp32, cast, bf16 gain, fp32 silu gate, cast."""
    h = mx.fast.rms_norm(x.astype(mx.float32), None, eps).astype(x.dtype)
    h = weight * h
    gf = gate.astype(mx.float32)
    return (h.astype(mx.float32) * (gf * mx.sigmoid(gf))).astype(x.dtype)


def l2norm(x, eps=1e-6):
    xf = x.astype(mx.float32)
    return xf * mx.rsqrt((xf * xf).sum(-1, keepdims=True) + eps * eps)


# ------------------------------------------------- fused elementwise groups
# Each decode step dispatches ~25 small kernels per layer; with weights streaming at
# the wall, that dispatch work is most of what separates raw decode from it. The
# position-free elementwise groups below are compiled once per shape, so mx.compile
# fuses each into a few kernels; each is checked bit-identical to its uncompiled form
# on real inputs. FUSE = False runs the same functions uncompiled.

FUSE = True
_compiled = {}
FUSED = []          # every compiled group, for tests/mac's bit-identity check


def _fused(fn):
    def call(*args):
        if not FUSE:
            return fn(*args)
        f = _compiled.get(fn)
        if f is None:
            f = _compiled[fn] = mx.compile(fn)
        return f(*args)
    call.__name__ = fn.__name__
    FUSED.append(call)
    return call


@_fused
def add_norm(x, h, gain, eps):
    """The residual add and the next RMSNorm: -> (x + h, rmsnorm(x + h))."""
    x = x + h
    return x, rmsnorm(x, gain, eps)


def gdn_prep(q, k, b, a, A, dt_bias, scale):
    """-> (q l2-normalized and scaled, k l2-normalized, g = A * softplus(a + dt_bias), beta).
    Not compiled: the compiled form differs in fp32 (3.8e-6 on real inputs) and the
    recurrence carries it through every later token, doubling the mean KL to mlx-lm
    on one of the three correctness prompts (docs/mac.md). The other groups are
    bit-identical compiled."""
    beta = mx.sigmoid(b).astype(mx.float32)                                 # HF: sigmoid in bf16
    af = a.astype(mx.float32) + dt_bias
    g = A * mx.logaddexp(af, mx.zeros_like(af))
    return l2norm(q) * scale, l2norm(k), g, beta


@_fused
def gated_norm(o, weight, z, eps):
    return gated_rmsnorm(o, weight, z, eps)


@_fused
def swiglu(gu):
    f = gu.shape[-1] // 2
    gate = gu[:, :f]
    return (gate * mx.sigmoid(gate)) * gu[:, f:]


@_fused
def sigmoid_gate(o, gate):
    gf = gate.astype(mx.float32)
    return (o.astype(mx.float32) * mx.sigmoid(gf)).astype(o.dtype)


@_fused
def conv_silu(window, w):
    """Depthwise causal conv over window [T+Kc-1, C] with w [C, Kc], fp32 products and one
    rounding to the weight dtype, then silu in the activation dtype."""
    Kc = w.shape[1]
    T = window.shape[0] - Kc + 1
    wf = w.astype(mx.float32)
    y = window[0:T].astype(mx.float32) * wf[:, 0]
    for j in range(1, Kc):
        y = y + window[j:j + T].astype(mx.float32) * wf[:, j]
    yf = y.astype(w.dtype).astype(mx.float32)
    return (yf * mx.sigmoid(yf)).astype(window.dtype)


# ------------------------------------------------------------------ state


@dataclass
class State:
    """Everything a step reads and writes besides weights. Arrays are replaced, not
    mutated: MLX donates the old buffer to the update when nothing else holds it."""
    pos: int
    conv: List[mx.array]            # per GDN layer [RING, conv_dim] bf16, row = position mod RING
    rec: List[mx.array]             # per GDN layer [HV, V, K] fp32
    k: List[mx.array]               # per attention layer [Hkv, max_len, D] bf16
    v: List[mx.array]
    pend: Optional[list] = None     # per GDN layer (k, v, g, beta) rows of the last verify step
    n_pend: Optional[mx.array] = None

    @classmethod
    def new(cls, cfg: ModelConfig, max_len: int):
        n_gdn, n_attn = len(cfg.gdn_layers), len(cfg.attn_layers)
        return cls(
            pos=0,
            conv=[mx.zeros((RING, cfg.conv_dim), mx.bfloat16) for _ in range(n_gdn)],
            rec=[mx.zeros((cfg.gdn_v_heads, cfg.gdn_v_dim, cfg.gdn_k_dim), mx.float32) for _ in range(n_gdn)],
            k=[mx.zeros((cfg.n_kv_heads, max_len, cfg.head_dim), mx.bfloat16) for _ in range(n_attn)],
            v=[mx.zeros((cfg.n_kv_heads, max_len, cfg.head_dim), mx.bfloat16) for _ in range(n_attn)],
        )

    def arrays(self):
        return self.conv + self.rec + self.k + self.v


# ------------------------------------------------------------------ layers


def conv(x, ring, w, pos: int):
    """Causal depthwise conv over the 4-tap window, then silu. x [T, C] at positions
    pos..pos+T-1; ring [RING, C] holds pre-conv inputs, row = position mod RING. Rows
    never written are zero, which is the causal padding at pos < 3 (reset zeroes it).
    -> (y [T, C], new ring)."""
    T = x.shape[0]
    Kc = w.shape[1]
    prev = ring[mx.array([(pos - (Kc - 1) + j) % RING for j in range(Kc - 1)])]
    window = mx.concatenate([prev, x], axis=0)                              # [T+3, C]
    keep = min(T, RING)
    ring[mx.array([(pos + t) % RING for t in range(T - keep, T)])] = x[T - keep:]
    return conv_silu(window, w), ring


def gdn_forward(x, w: GDNWeights, cfg: ModelConfig, st: State, gi: int, commit: bool = True):
    T = x.shape[0]
    qkvz = w.in_qkvz(x)                                                     # [T, conv_dim + val_dim]
    ba = x @ w.in_ba.T                                                      # [T, 2*HV]
    qkv, z = qkvz[:, :cfg.conv_dim], qkvz[:, cfg.conv_dim:]
    b, a = ba[:, :cfg.gdn_v_heads], ba[:, cfg.gdn_v_heads:]
    qkv, st.conv[gi] = conv(qkv, st.conv[gi], w.conv_w, st.pos)
    qd = cfg.gdn_qk_dim
    q = qkv[:, :qd].reshape(T, cfg.gdn_k_heads, cfg.gdn_k_dim)
    k = qkv[:, qd:2 * qd].reshape(T, cfg.gdn_k_heads, cfg.gdn_k_dim)
    v = qkv[:, 2 * qd:].reshape(T, cfg.gdn_v_heads, cfg.gdn_v_dim)
    qn, kn, g, beta = gdn_prep(q, k, b, a, w.A, w.dt_bias, cfg.gdn_k_dim ** -0.5)
    pend = None
    if st.pend is not None:
        pend = (*st.pend[gi], st.n_pend)
    o, rec = kernels.gdn(qn, kn, v, g, beta, st.rec[gi], pend=pend, commit=commit)
    st.rec[gi] = rec
    if not commit:
        st.new_pend.append((kn, v, g, beta))
    o = gated_norm(o.reshape(-1, cfg.gdn_v_dim), w.norm_w, z.reshape(-1, cfg.gdn_v_dim), cfg.eps)
    return w.out(o.reshape(T, cfg.gdn_val_dim))


def attn_forward(x, w: AttnWeights, cfg: ModelConfig, st: State, ai: int, gains):
    T = x.shape[0]
    hd, pos = cfg.head_dim, st.pos
    kv_dim = cfg.n_kv_heads * hd
    qkv = w.qkv(x)
    qg = qkv[:, :cfg.n_heads * 2 * hd].reshape(T, cfg.n_heads, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:].reshape(T, cfg.n_heads * hd)
    k = qkv[:, cfg.n_heads * 2 * hd:cfg.n_heads * 2 * hd + kv_dim].reshape(T, cfg.n_kv_heads, hd)
    v = qkv[:, cfg.n_heads * 2 * hd + kv_dim:].reshape(T, cfg.n_kv_heads, hd)
    q = rmsnorm(q, gains[0], cfg.eps).transpose(1, 0, 2)                    # [Hq, T, D]
    k = rmsnorm(k, gains[1], cfg.eps).transpose(1, 0, 2)
    v = v.transpose(1, 0, 2)
    q = mx.fast.rope(q, cfg.rotary_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos)
    k = mx.fast.rope(k, cfg.rotary_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos)
    st.k[ai][:, pos:pos + T] = k
    st.v[ai][:, pos:pos + T] = v
    K, V = st.k[ai][:, :pos + T], st.v[ai][:, :pos + T]
    if 1 < T <= FOLD_ROWS:
        # A verify step: MLX's multi-row kernel reads K/V once per query head, 6x for our
        # 24/4 GQA. Fold each KV head's 6 query heads into the row dimension instead, so
        # K/V are read once, with an explicit causal mask per row: row (g, i) is position
        # pos + i. 3.2x faster at 60k context (bench/mac/ctx_step.py), same output.
        G = cfg.n_heads // cfg.n_kv_heads
        qf = q.reshape(cfg.n_kv_heads, G * T, hd)
        rowpos = pos + mx.tile(mx.arange(T), (G,))
        mask = mx.arange(pos + T)[None, :] <= rowpos[:, None]
        o = mx.fast.scaled_dot_product_attention(qf[None], K[None], V[None], scale=hd ** -0.5, mask=mask)[0]
        o = o.reshape(cfg.n_heads, T, hd)
    else:
        o = mx.fast.scaled_dot_product_attention(q[None], K[None], V[None], scale=hd ** -0.5,
                                                 mask="causal" if T > 1 else None)[0]
    o = o.transpose(1, 0, 2).reshape(T, cfg.n_heads * hd)
    return w.o(sigmoid_gate(o, gate))


def mlp_forward(x, w: LayerWeights):
    return w.down(swiglu(w.gate_up(x)))


# ----------------------------------------------------------------- engine


@dataclass
class Engine:
    cfg: ModelConfig
    w: ModelWeights
    max_len: int
    state: State = None
    gains: list = field(default_factory=list)      # per layer (ln1, ln2, [q_norm, k_norm]) as fp32 (1 + w)
    final_gain: mx.array = None
    feature_layers: Optional[list] = None           # layers whose residual outputs a DFlash draft reads
    features: Optional[mx.array] = None             # [T, len(feature_layers) * hidden] of the last forward
    last_hidden: Optional[mx.array] = None          # post-final-norm hidden of the last forward

    def __post_init__(self):
        cfg = self.cfg
        self.state = State.new(cfg, self.max_len)
        g = lambda a: 1.0 + a.astype(mx.float32)
        for lw in self.w.layers:
            e = [g(lw.ln1), g(lw.ln2)]
            if isinstance(lw.mixer, AttnWeights):
                e += [g(lw.mixer.q_norm_w), g(lw.mixer.k_norm_w)]
            self.gains.append(e)
        self.final_gain = g(self.w.final_norm)
        self._gi = {li: i for i, li in enumerate(cfg.gdn_layers)}
        self._ai = {li: i for i, li in enumerate(cfg.attn_layers)}
        mx.eval(self.state.arrays(), self.gains, self.final_gain)

    def reset(self):
        """Back to position 0. The KV caches are kept as they are (rows >= pos are never read
        before they are written); the conv rings and recurrent states must start at zero."""
        st, cfg = self.state, self.cfg
        st.pos, st.pend, st.n_pend = 0, None, None
        st.conv = [mx.zeros_like(c) for c in st.conv]
        st.rec = [mx.zeros_like(r) for r in st.rec]
        mx.eval(st.conv, st.rec)

    def body(self, tokens: mx.array, commit: bool = True, all_rows: bool = False):
        """The forward over tokens [T] at state.pos.. -> post-final-norm hidden [T or 1, hidden].
        Builds the graph; the caller evaluates. Does not advance pos."""
        cfg, st = self.cfg, self.state
        st.new_pend = []
        x = self.w.embed[tokens]
        feats = []
        h = None                                     # the residual contribution not yet added to x
        for li in range(cfg.n_layers):
            lw, gn = self.w.layers[li], self.gains[li]
            if h is None:
                n = rmsnorm(x, gn[0], cfg.eps)
            else:
                x, n = add_norm(x, h, gn[0], cfg.eps)
            if cfg.layer_types[li] == "linear_attention":
                h = gdn_forward(n, lw.mixer, cfg, st, self._gi[li], commit)
            else:
                h = attn_forward(n, lw.mixer, cfg, st, self._ai[li], gn[2:])
            x, n = add_norm(x, h, gn[1], cfg.eps)
            h = mlp_forward(n, lw)
            if self.feature_layers is not None and li in self.feature_layers:
                feats.append(x + h)
        self.features = mx.concatenate(feats, axis=-1) if feats else None
        if not all_rows:
            x, h = x[-1:], h[-1:]
        return add_norm(x, h, self.final_gain, cfg.eps)[1]

    def forward(self, tokens: mx.array, all_logits: bool = False) -> mx.array:
        """tokens [T] -> logits [1 or T, vocab]; commits and advances the state by T."""
        T = tokens.shape[0]
        assert self.state.pos + T <= self.max_len, "context exceeds the preallocated cache"
        h = self.body(tokens, all_rows=all_logits)
        self.last_hidden = h
        logits = self.w.lm_head(h)
        self.state.pend = None
        self.state.pos += T
        return logits

    def forward_hidden(self, tokens: mx.array) -> mx.array:
        """Like forward, but returns the post-final-norm hidden of every row and no logits
        (the MTP head's prompt input; T x vocab logits would be gigabytes)."""
        T = tokens.shape[0]
        assert self.state.pos + T <= self.max_len, "context exceeds the preallocated cache"
        h = self.body(tokens, all_rows=True)
        self.last_hidden = h
        self.state.pend = None
        self.state.pos += T
        return h

    # ------------------------------------------------------- speculative step

    def verify(self, tokens: mx.array, pick=None):
        """The committed-but-unprocessed token followed by K drafts, tokens [K+1], at pos..
        Builds (does not evaluate) the step: GDN layers produce outputs without touching
        their state (the rows are kept for commit), attention and conv rows are written in
        place (rejected rows are overwritten before anything reads them).
        pick(logits [K+1, vocab]) -> tokens [K+1], argmax by default; a draft is accepted
        while it equals the pick at its position, and the pick after the last accepted
        draft is the next committed token, so the output is the target's own.
        -> (n accepted [scalar, device], picks [K+1], hidden [K+1, hidden])."""
        M = tokens.shape[0]
        assert self.state.pos + M <= self.max_len and M + 3 <= RING
        h = self.body(tokens, commit=False, all_rows=True)
        logits = self.w.lm_head(h)
        pred = mx.argmax(logits, -1) if pick is None else pick(logits)
        match = (pred[:-1] == tokens[1:]).astype(mx.int32)
        n = mx.cumprod(match).sum() if M > 1 else mx.array(0, mx.int32)
        self.last_hidden = h
        return n, pred, h

    def commit(self, n: int):
        """After a verify step is evaluated: n of its drafts were accepted, so its first
        n+1 rows are history. The GDN layers replay them at the start of the next call."""
        st = self.state
        st.pend = st.new_pend
        st.n_pend = mx.array([n + 1], mx.int32)
        st.pos += n + 1

    def prefill(self, tokens: mx.array, chunk: int = 2048) -> mx.array:
        logits = None
        for s in range(0, tokens.shape[0], chunk):
            logits = self.forward(tokens[s:s + chunk])
            mx.eval(logits, self.state.arrays())
        return logits
