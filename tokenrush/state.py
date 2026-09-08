"""Explicit, preallocated decode state. Nothing here grows, moves, or is
looked up through a table: the KV cache is one contiguous tensor per layer
group, the GDN state is fixed-size, and the position is a counter. This is
what lets Phase 2 record a whole decode step into a single CUDA graph."""
import torch

from .config import ModelConfig


class State:
    def __init__(self, cfg: ModelConfig, max_len: int, device, kv_dtype=torch.bfloat16):
        n_attn = len(cfg.attn_layers)
        n_gdn = len(cfg.gdn_layers)
        self.max_len = max_len
        # attention KV: [layer, kv_head, position, head_dim]
        self.k = torch.zeros(n_attn, cfg.n_kv_heads, max_len, cfg.head_dim, device=device, dtype=kv_dtype)
        self.v = torch.zeros_like(self.k)
        # GDN: a ring of the last K pre-conv inputs and the fp32 recurrent state
        self.conv = torch.zeros(n_gdn, cfg.conv_dim, cfg.conv_k, device=device, dtype=torch.bfloat16)   # ring, col = pos % 4
        self.rec = torch.zeros(n_gdn, cfg.gdn_v_heads, cfg.gdn_k_dim, cfg.gdn_v_dim, device=device,
                               dtype=torch.float32)
        # position: a host mirror for slicing in prefill and choosing a graph bucket,
        # and the device tensor every in-graph index derives from
        self.pos = 0
        self.pos_t = torch.zeros(1, device=device, dtype=torch.long)
        # layer index -> slot in the per-type tensors
        self.attn_slot = {l: i for i, l in enumerate(cfg.attn_layers)}
        self.gdn_slot = {l: i for i, l in enumerate(cfg.gdn_layers)}

    def reset(self):
        self.conv.zero_()
        self.rec.zero_()
        self.pos = 0
        self.pos_t.zero_()

    def advance(self, T: int):
        self.pos += T
        self.pos_t += T

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.k, self.v, self.conv, self.rec))
