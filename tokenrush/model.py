"""The text path of Qwen3.8-27B as plain functions over weight containers.

Layers are pure functions of (weights, state, activations); there is no
nn.Module and no autograd. Activations are [T, hidden] with bs=1 implicit.
"""
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn.functional as F

from . import ops
from .config import ModelConfig
from .quant import Linear, QLinear
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


def gdn_forward(x: torch.Tensor, w: GDNWeights, cfg: ModelConfig, state: State, slot: int) -> torch.Tensor:
    T = x.shape[0]
    qkv, z = torch.split(w.in_qkvz(x), [cfg.conv_dim, cfg.gdn_val_dim], dim=-1)   # [T, conv_dim], [T, val_dim]
    b, a = torch.split(F.linear(x, w.in_ba), [cfg.gdn_v_heads, cfg.gdn_v_heads], dim=-1)   # [T, HV] each
    if T == 1:
        qkv = ops.conv_step(qkv, state.conv[slot], w.conv_w)
    else:
        qkv = ops.conv_prefill(qkv, state.conv[slot], w.conv_w)
    q, k, v = torch.split(qkv, [cfg.gdn_qk_dim, cfg.gdn_qk_dim, cfg.gdn_val_dim], dim=-1)
    q = q.view(T, cfg.gdn_k_heads, cfg.gdn_k_dim)
    k = k.view(T, cfg.gdn_k_heads, cfg.gdn_k_dim)
    v = v.view(T, cfg.gdn_v_heads, cfg.gdn_v_dim)
    beta = torch.sigmoid(b)                                      # [T, HV] bf16
    g = w.A * F.softplus(a.float() + w.dt_bias)                  # [T, HV] fp32
    if T == 1:
        o = ops.gdn_step(q[0], k[0], v[0], g[0], beta[0], state.rec[slot])[None]
    else:
        o = ops.gdn_prefill(q, k, v, g, beta, state.rec[slot])
    o = ops.gated_rmsnorm(o.reshape(-1, cfg.gdn_v_dim), w.norm_w, z.reshape(-1, cfg.gdn_v_dim), cfg.eps)
    return w.out(o.view(T, cfg.gdn_val_dim))


def attn_forward(x: torch.Tensor, w: AttnWeights, cfg: ModelConfig, state: State, slot: int,
                 cos: torch.Tensor, sin: torch.Tensor, bucket: int = None) -> torch.Tensor:
    """bucket: if given (decode only), attend over the first `bucket` cache rows with a
    mask derived from the device position, so the whole call is shape-static."""
    T = x.shape[0]
    pos = state.pos
    hd = cfg.head_dim
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


def mlp_forward(x: torch.Tensor, w: LayerWeights) -> torch.Tensor:
    gate, up = torch.chunk(w.gate_up(x), 2, dim=-1)
    return w.down(F.silu(gate) * up)


# ----------------------------------------------------------------- engine


class Engine:
    """Holds weights, state and rotary tables; runs prefill eagerly and decode
    either eagerly or as one CUDA graph per context bucket."""

    def __init__(self, cfg: ModelConfig, weights: ModelWeights, max_len: int, device="cuda"):
        self.cfg = cfg
        self.w = weights
        self.device = torch.device(device)
        self.state = State(cfg, max_len, self.device)
        self.cos, self.sin = ops.rope_table(max_len, cfg.rotary_dim, cfg.rope_theta, self.device)
        # graph I/O: the token read at the start of a step and written at its end
        self.tok = torch.zeros(1, device=self.device, dtype=torch.long)
        self.logits = None            # [1, vocab], written by the last graphed step
        self.graphs = {}              # bucket -> CUDAGraph
        self.pool = None

    def _body(self, tokens: torch.Tensor, bucket=None, all_logits=False) -> torch.Tensor:
        """The forward pass without the position bookkeeping."""
        cfg, st = self.cfg, self.state
        x = self.w.embed[tokens]
        for li, lw in enumerate(self.w.layers):
            h = ops.rmsnorm(x, lw.ln1, cfg.eps)
            if cfg.layer_types[li] == "linear_attention":
                h = gdn_forward(h, lw.mixer, cfg, st, st.gdn_slot[li])
            else:
                h = attn_forward(h, lw.mixer, cfg, st, st.attn_slot[li], self.cos, self.sin, bucket)
            x = x + h
            h = ops.rmsnorm(x, lw.ln2, cfg.eps)
            x = x + mlp_forward(h, lw)
        if not all_logits:
            x = x[-1:]
        x = ops.rmsnorm(x, self.w.final_norm, cfg.eps)
        return self.w.lm_head(x)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, all_logits: bool = False) -> torch.Tensor:
        """Eager. tokens [T] on device at positions state.pos .. state.pos+T-1; advances
        the state by T. Returns logits [1, vocab] for the last token (or [T, vocab])."""
        T = tokens.shape[0]
        assert self.state.pos + T <= self.state.max_len, "context exceeds the preallocated cache"
        logits = self._body(tokens, all_logits=all_logits)
        self.state.advance(T)
        return logits

    # ------------------------------------------------------------ graphs

    def _graph_step(self, bucket: int):
        """One decode step, shape-static: read self.tok, write self.logits and the next
        token back into self.tok, advance the device position. Captured per bucket."""
        logits = self._body(self.tok, bucket=bucket)
        self.logits.copy_(logits)
        self.tok.copy_(logits.argmax(-1))
        self.state.pos_t += 1

    @staticmethod
    def buckets_for(max_len: int, smallest: int = 1024):
        b, out = smallest, []
        while b < max_len:
            out.append(b)
            b *= 2
        return out + [max_len]

    @torch.no_grad()
    def capture(self, buckets=None, warmup: int = 2):
        """Record one graph per context bucket. Runs from a clean state and leaves it
        clean; call once after loading. Warmup runs the step eagerly on a side stream
        (autotuning, workspaces) and is what mutates the state, so it is reset after."""
        cfg, st = self.cfg, self.state
        buckets = buckets or self.buckets_for(st.max_len)
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
        return self.tok

    def prefill(self, tokens: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        logits = None
        for s in range(0, tokens.shape[0], chunk):
            logits = self.forward(tokens[s:s + chunk])
        return logits

    def decode(self, token: torch.Tensor) -> torch.Tensor:
        return self.forward(token.view(1))

    def reset(self):
        self.state.reset()
