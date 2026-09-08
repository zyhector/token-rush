"""Model configuration for the text path, read from the checkpoint's config.json."""
import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    hidden: int
    n_layers: int
    layer_types: tuple            # 'linear_attention' | 'full_attention' per layer
    vocab: int
    ffn: int
    # attention layers
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rotary_dim: int               # partial rotary: only the first rotary_dim of head_dim rotate
    rope_theta: float
    # gated deltanet layers
    gdn_k_heads: int
    gdn_v_heads: int
    gdn_k_dim: int
    gdn_v_dim: int
    conv_k: int
    # misc
    eps: float
    max_pos: int
    eos_ids: tuple

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        c = json.load(open(os.path.join(path, "config.json")))
        t = c.get("text_config", c)
        rp = t.get("rope_parameters", {})
        eos = t.get("eos_token_id")
        eos = tuple(eos) if isinstance(eos, list) else (eos,)
        # Interleaved mRoPE collapses to plain RoPE for text: the three position
        # streams are identical, so the interleave copies a value onto itself.
        prf = rp.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0))
        return cls(
            hidden=t["hidden_size"], n_layers=t["num_hidden_layers"],
            layer_types=tuple(t["layer_types"]), vocab=t["vocab_size"], ffn=t["intermediate_size"],
            n_heads=t["num_attention_heads"], n_kv_heads=t["num_key_value_heads"],
            head_dim=t["head_dim"], rotary_dim=int(t["head_dim"] * prf), rope_theta=rp["rope_theta"],
            gdn_k_heads=t["linear_num_key_heads"], gdn_v_heads=t["linear_num_value_heads"],
            gdn_k_dim=t["linear_key_head_dim"], gdn_v_dim=t["linear_value_head_dim"],
            conv_k=t["linear_conv_kernel_dim"], eps=t["rms_norm_eps"],
            max_pos=t["max_position_embeddings"], eos_ids=eos,
        )

    @property
    def gdn_qk_dim(self):        # 16 * 128 = 2048
        return self.gdn_k_heads * self.gdn_k_dim

    @property
    def gdn_val_dim(self):       # 48 * 128 = 6144
        return self.gdn_v_heads * self.gdn_v_dim

    @property
    def conv_dim(self):          # q, k, v all pass through the conv: 10240
        return 2 * self.gdn_qk_dim + self.gdn_val_dim

    @property
    def attn_layers(self):
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @property
    def gdn_layers(self):
        return [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]
