"""
Baichuan Model Adapter

Supports Baichuan and Baichuan2 architectures.
Features:
- ALiBi (Attention with Linear Biases) for position encoding
- SwiGLU MLP activation
- RMSNorm
- GQA (Baichuan2) or MHA (Baichuan-7B/13B)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from draco.models.adapter import ModelAdapter
from draco.models.config import ModelConfig
from draco.models._kv_cache_utils import concat_kv

logger = logging.getLogger("draco.models.baichuan")


class BaichuanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class BaichuanALiBi(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        slopes = self._get_slopes(num_heads)
        slopes = torch.tensor(slopes).float()
        self.register_buffer("slopes", slopes, persistent=False)

    @staticmethod
    def _get_slopes(num_heads: int) -> List[float]:
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
        powers = torch.arange(1, closest_power_of_2 + 1, dtype=torch.float32)
        slopes = torch.pow(base, powers)
        if closest_power_of_2 != num_heads:
            extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
            extra_powers = torch.arange(1, 2 * (num_heads - closest_power_of_2) + 1, 2, dtype=torch.float32)
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)])
        return slopes.tolist()

    def forward(self, seq_len: int) -> torch.Tensor:
        position_ids = torch.arange(seq_len, device=self.slopes.device).float()
        relative = (position_ids.unsqueeze(0) - position_ids.unsqueeze(1)).abs()
        return -relative.unsqueeze(0).unsqueeze(0) * self.slopes.view(1, -1, 1, 1)


class BaichuanAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi_bias: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k, v, present_key_value = concat_kv(past_key_value, k, v, use_cache)
        kv_len = k.shape[2]
        past_len = kv_len - seq_len

        if self.num_kv_groups > 1:
            k_rep = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_rep = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_rep, v_rep = k, v

        attn_weights = torch.matmul(q, k_rep.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # alibi_bias arrives sized for the FULL kv_len (computed once at
        # model level from kv_len, not just this chunk's seq_len); take
        # only the rows for this chunk's (absolute) query positions.
        attn_weights = attn_weights + alibi_bias[:, :, past_len:kv_len, :kv_len]

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
        return self.o_proj(attn_output), present_key_value


class BaichuanMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        intermediate = config.intermediate_size or 4 * config.hidden_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class BaichuanDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = BaichuanRMSNorm(config.hidden_size)
        self.self_attn = BaichuanAttention(config, layer_idx)
        self.post_attention_layernorm = BaichuanRMSNorm(config.hidden_size)
        self.mlp = BaichuanMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi_bias: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(hidden_states, alibi_bias, attention_mask, past_key_value, use_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value


class BaichuanModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            BaichuanDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = BaichuanRMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.alibi = BaichuanALiBi(config.num_attention_heads)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ):
        hidden_states = self.embed_tokens(input_ids)

        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[2]
        kv_len = past_len + input_ids.shape[1]
        alibi_bias = self.alibi.forward(kv_len)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        present_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present_kv = layer(hidden_states, alibi_bias, attention_mask, past_kv, use_cache)
            present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if use_cache:
            return logits, present_key_values
        return logits


class BaichuanAdapter(ModelAdapter):
    """Model adapter for Baichuan-family architectures."""

    @property
    def architecture_name(self) -> str:
        return "BaichuanForCausalLM"

    @property
    def model_type(self) -> str:
        return "baichuan"

    @property
    def supported_model_types(self) -> List[str]:
        return ["baichuan", "baichuan2"]

    def build_model(self) -> nn.Module:
        return BaichuanModel(self.config)

    def embed(self, input_ids):
        return input_ids

    def attention(self, layer_idx, hidden_states, position_ids, kv_cache=None, **kwargs):
        raise NotImplementedError("Use build_model()")

    def mlp(self, layer_idx, hidden_states):
        raise NotImplementedError("Use build_model()")

    def norm(self, hidden_states):
        raise NotImplementedError("Use build_model()")

    def lm_head(self, hidden_states):
        raise NotImplementedError("Use build_model()")

    def rotary_embedding(self, position_ids, head_dim):
        raise NotImplementedError("Baichuan uses ALiBi")

    def get_position_encoding_type(self) -> str:
        return "alibi"

    def get_attention_type(self) -> str:
        if self.config.num_key_value_heads < self.config.num_attention_heads:
            return "GQA"
        return "MHA"

    def has_sliding_window(self) -> bool:
        return False

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
