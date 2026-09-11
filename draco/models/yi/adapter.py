"""
Yi Model Adapter

Supports Yi, Yi-1.5, and Yi-Vision architectures.
Features:
- GQA (Grouped-Query Attention)
- RoPE (Rotary Position Embedding)
- SwiGLU MLP
- RMSNorm
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from draco.models.adapter import ModelAdapter
from draco.models.config import ModelConfig
from draco.models._kv_cache_utils import cache_aware_causal_mask, concat_kv, derive_position_ids

logger = logging.getLogger("draco.models.yi")


class YiRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class YiRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.unsqueeze(0).unsqueeze(0)
        position_ids = position_ids.unsqueeze(-1).float()
        freqs = inv_freq * position_ids
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


class YiAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos_q = cos.unsqueeze(1)
        sin_q = sin.unsqueeze(1)
        q = _apply_rotary_emb(q, cos_q, sin_q)
        k = _apply_rotary_emb(k, cos_q, sin_q)

        k, v, present_key_value = concat_kv(past_key_value, k, v, use_cache)
        kv_len = k.shape[2]

        if self.num_kv_groups > 1:
            k_rep = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_rep = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_rep, v_rep = k, v

        attn_weights = torch.matmul(q, k_rep.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal_mask = cache_aware_causal_mask(seq_len, kv_len, hidden_states.device, hidden_states.dtype)
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v_rep)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output), present_key_value


class YiMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        intermediate = config.intermediate_size or 4 * config.hidden_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class YiDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = YiAttention(config, layer_idx)
        self.mlp = YiMLP(config)
        self.input_layernorm = YiRMSNorm(config.hidden_size)
        self.post_attention_layernorm = YiRMSNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(hidden_states, cos, sin, attention_mask, past_key_value, use_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value


class YiModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            YiDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = YiRMSNorm(config.hidden_size)
        self.rotary_emb = YiRotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ):
        hidden_states = self.embed_tokens(input_ids)
        position_ids = derive_position_ids(input_ids, past_key_values, position_ids)
        cos, sin = self.rotary_emb(position_ids)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        present_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present_kv = layer(hidden_states, cos, sin, attention_mask, past_kv, use_cache)
            present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = torch.matmul(hidden_states, self.embed_tokens.weight.t())
        if use_cache:
            return logits, present_key_values
        return logits


class YiAdapter(ModelAdapter):
    """Model adapter for Yi-family architectures."""

    @property
    def architecture_name(self) -> str:
        return "YiForCausalLM"

    @property
    def model_type(self) -> str:
        return "yi"

    @property
    def supported_model_types(self) -> List[str]:
        return ["yi", "yi1.5"]

    def build_model(self) -> nn.Module:
        return YiModel(self.config)

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
        emb = YiRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
        )
        return emb(position_ids)

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
        return self.config.head_dim

    def get_vocab_size(self) -> int:
        return self.config.vocab_size
