"""
BLOOM Model Adapter

Supports BLOOM (1B through 176B) and related architectures.
Features:
- ALiBi (Attention with Linear Biases) for position encoding
- Parallel attention + MLP (residual goes through both)
- LayerNorm (not RMSNorm)
- Multi-query attention option
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from draco.models.adapter import ModelAdapter
from draco.models.config import ModelConfig
from draco.models._kv_cache_utils import concat_kv

logger = logging.getLogger("draco.models.bloom")


# ======================================================================
# BLOOM Components
# ======================================================================

class BloomLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, eps=self.eps)


class BloomALiBi(nn.Module):
    """ALiBi (Attention with Linear Biases) for BLOOM."""

    def __init__(self, num_heads: int, max_position_embeddings: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings

        # Precompute slopes for each head
        slopes = self._get_slopes(num_heads)
        slopes = torch.tensor(slopes).float()  # (num_heads,)
        self.register_buffer("slopes", slopes, persistent=False)

    @staticmethod
    def _get_slopes(num_heads: int) -> List[float]:
        """Get attention bias slopes for ALiBi."""
        import math
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
        powers = torch.arange(1, closest_power_of_2 + 1, dtype=torch.float32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != num_heads:
            extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
            extra_powers = torch.arange(1, 2 * (num_heads - closest_power_of_2) + 1, 2, dtype=torch.float32)
            extra_slopes = torch.pow(extra_base, extra_powers)
            slopes = torch.cat([slopes, extra_slopes])

        return slopes.tolist()

    def forward(self, seq_len: int) -> torch.Tensor:
        """Compute ALiBi bias matrix.

        Returns:
            bias: (1, num_heads, seq_len, seq_len)
        """
        position_ids = torch.arange(seq_len, device=self.slopes.device).float()
        relative_position = position_ids.unsqueeze(0) - position_ids.unsqueeze(1)  # (seq, seq)
        relative_position = relative_position.abs()  # (seq, seq)
        # slopes: (num_heads,) -> (1, num_heads, 1)
        # relative_position: (seq, seq) -> (1, 1, seq, seq)
        bias = -relative_position.unsqueeze(0).unsqueeze(0) * self.slopes.view(1, -1, 1, 1)
        return bias


class BloomAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi_bias: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        k, v, present_key_value = concat_kv(past_key_value, k, v, use_cache)
        kv_len = k.shape[2]
        past_len = kv_len - seq_len

        # Expand KV heads if needed
        if self.num_kv_groups > 1:
            k_rep = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_rep = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_rep, v_rep = k, v

        # Scaled dot-product attention with ALiBi
        attn_weights = torch.matmul(q, k_rep.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # alibi_bias is sized for the FULL kv_len; slice this chunk's rows.
        attn_weights = attn_weights + alibi_bias[:, :, past_len:kv_len, :kv_len]

        # Causal mask (cache-offset aware)
        query_pos = torch.arange(seq_len, device=hidden_states.device).unsqueeze(1) + past_len
        key_pos = torch.arange(kv_len, device=hidden_states.device).unsqueeze(0)
        causal_mask = torch.where(
            key_pos <= query_pos,
            torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype),
            torch.full((), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype),
        )
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v_rep)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.dense(attn_output), present_key_value


class BloomMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        intermediate = config.intermediate_size or 4 * config.hidden_size
        self.dense_h_to_4h = nn.Linear(config.hidden_size, intermediate, bias=True)
        self.dense_4h_to_h = nn.Linear(intermediate, config.hidden_size, bias=True)
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_4h_to_h(self.act_fn(self.dense_h_to_4h(x)))


class BloomDecoderLayer(nn.Module):
    """BLOOM decoder layer with parallel attention+MLP."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.ln_attention = BloomLayerNorm(config.hidden_size)
        self.self_attention = BloomAttention(config, layer_idx)
        self.ln_mlp = BloomLayerNorm(config.hidden_size)
        self.mlp = BloomMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi_bias: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Parallel residual: attn and MLP see the same input
        residual = hidden_states
        attn_input = self.ln_attention(hidden_states)
        mlp_input = self.ln_mlp(hidden_states)

        attn_output, present_key_value = self.self_attention(attn_input, alibi_bias, attention_mask, past_key_value, use_cache)
        mlp_output = self.mlp(mlp_input)

        hidden_states = residual + attn_output + mlp_output
        return hidden_states, present_key_value


class BloomModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.h = nn.ModuleList([BloomDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.ln_f = BloomLayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # ALiBi
        self.alibi = BloomALiBi(
            num_heads=config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
                use_cache: bool = False):
        hidden_states = self.word_embeddings(input_ids)

        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[2]
        kv_len = past_len + input_ids.shape[1]
        alibi_bias = self.alibi.forward(kv_len)

        if past_key_values is None:
            past_key_values = [None] * len(self.h)
        present_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for layer, past_kv in zip(self.h, past_key_values):
            hidden_states, present_kv = layer(hidden_states, alibi_bias, attention_mask, past_kv, use_cache)
            present_key_values.append(present_kv)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        if use_cache:
            return logits, present_key_values
        return logits


# ======================================================================
# BLOOM Adapter
# ======================================================================

class BloomAdapter(ModelAdapter):
    """Model adapter for BLOOM-family architectures.

    Supports: BloomForCausalLM
    """

    @property
    def architecture_name(self) -> str:
        return "BloomForCausalLM"

    @property
    def model_type(self) -> str:
        return "bloom"

    @property
    def supported_model_types(self) -> List[str]:
        return ["bloom"]

    def build_model(self) -> nn.Module:
        return BloomModel(self.config)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids

    def attention(self, layer_idx, hidden_states, position_ids, kv_cache=None, **kwargs):
        raise NotImplementedError("Use build_model() for full forward pass")

    def mlp(self, layer_idx, hidden_states):
        raise NotImplementedError("Use build_model() for full forward pass")

    def norm(self, hidden_states):
        raise NotImplementedError("Use build_model() for full forward pass")

    def lm_head(self, hidden_states):
        raise NotImplementedError("Use build_model() for full forward pass")

    def rotary_embedding(self, position_ids, head_dim):
        raise NotImplementedError("BLOOM uses ALiBi, not RoPE")

    def get_position_encoding_type(self) -> str:
        return "alibi"

    def get_attention_type(self) -> str:
        if self.config.num_key_value_heads < self.config.num_attention_heads:
            return "MQA"
        return "MHA"

    def prepare_inputs(self, input_ids, position_ids, kv_cache=None, **kwargs):
        return {"attention_mask": kwargs.get("attention_mask")}

    def load_weights(self, model, weight_path, **kwargs):
        from draco.weights.loader import get_checkpoint_loader
        loader = get_checkpoint_loader()
        weights = loader.load(weight_path, device=next(model.parameters()).device)
        weight_map = self.get_weight_map()
        remapped = {weight_map.get(k, k): v for k, v in weights.items()}
        model.load_state_dict(remapped, strict=False)
        return model

    def get_weight_map(self) -> Dict[str, str]:
        return {}

    def get_num_layers(self) -> int:
        return self.config.num_hidden_layers

    def get_hidden_size(self) -> int:
        return self.config.hidden_size

    def get_num_heads(self) -> int:
        return self.config.num_attention_heads

    def get_num_kv_heads(self) -> int:
        return self.config.num_key_value_heads

    def get_head_dim(self) -> int:
        return self.config.hidden_size // self.config.num_attention_heads

    def get_vocab_size(self) -> int:
        return self.config.vocab_size
