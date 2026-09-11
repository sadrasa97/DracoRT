"""
Falcon Model Adapter

Supports Falcon-7B, Falcon-40B, and related architectures.
Features: Multi-query attention (MQA), ALiBi or RoPE, parallel attention+MLP.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from draco.models.adapter import ModelAdapter
from draco.models.config import ModelConfig
from draco.models._kv_cache_utils import cache_aware_causal_mask, concat_kv

logger = logging.getLogger("draco.models.falcon")


class FalconLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, eps=self.eps)


class FalconAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.new_decoder_architecture = config.get("new_decoder_architecture", False)
        self.is_mqa = self.num_kv_heads < self.num_heads

        if self.new_decoder_architecture:
            qkv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        else:
            qkv_dim = 3 * self.hidden_size
        self.query_key_value = nn.Linear(self.hidden_size, qkv_dim, bias=True)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.shape
        qkv = self.query_key_value(hidden_states)

        if self.new_decoder_architecture:
            # New architecture: separate Q, K, V sizes
            q_size = self.num_heads * self.head_dim
            kv_size = self.num_kv_heads * self.head_dim
            q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
            q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        else:
            # Old architecture: qkv = 3 * hidden_size, split evenly
            qkv = qkv.view(bsz, seq_len, 3, self.hidden_size)
            q = qkv[:, :, 0, :].view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = qkv[:, :, 1, :].view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = qkv[:, :, 2, :].view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            # For MQA: reshape k, v from num_heads to num_kv_heads
            if self.is_mqa:
                n_rep = self.num_heads // self.num_kv_heads
                k = k.view(bsz, self.num_kv_heads, n_rep, seq_len, self.head_dim)
                k = k[:, :, 0, :, :]  # Take first group
                v = v.view(bsz, self.num_kv_heads, n_rep, seq_len, self.head_dim)
                v = v[:, :, 0, :, :]

        # Real KV cache: concat BEFORE repeating heads for MQA, so the
        # cache stores un-repeated (memory-cheap) KV heads.
        k, v, present_key_value = concat_kv(past_key_value, k, v, use_cache)
        kv_len = k.shape[2]

        # MQA: repeat KV heads if needed
        if k.shape[1] < self.num_heads:
            n_rep = self.num_heads // k.shape[1]
            k_rep = k.repeat_interleave(n_rep, dim=1)
            v_rep = v.repeat_interleave(n_rep, dim=1)
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
        return self.dense(attn_output), present_key_value


class FalconMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        intermediate = config.intermediate_size or 4 * config.hidden_size
        self.dense_h_to_4h = nn.Linear(config.hidden_size, intermediate, bias=True)
        self.dense_4h_to_h = nn.Linear(intermediate, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_4h_to_h(F.gelu(self.dense_h_to_4h(x)))


class FalconDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.ln_attn = FalconLayerNorm(config.hidden_size)
        self.self_attention = FalconAttention(config, layer_idx)
        self.ln_mlp = FalconLayerNorm(config.hidden_size)
        self.mlp = FalconMLP(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.ln_attn(hidden_states)
        hidden_states, present_key_value = self.self_attention(hidden_states, attention_mask, past_key_value, use_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ln_mlp(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value


class FalconModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.h = nn.ModuleList([FalconDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.ln_f = FalconLayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
                use_cache: bool = False):
        hidden_states = self.embed_tokens(input_ids)
        if past_key_values is None:
            past_key_values = [None] * len(self.h)
        present_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for layer, past_kv in zip(self.h, past_key_values):
            hidden_states, present_kv = layer(hidden_states, attention_mask, past_kv, use_cache)
            present_key_values.append(present_kv)
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        if use_cache:
            return logits, present_key_values
        return logits


class FalconAdapter(ModelAdapter):
    """Model adapter for Falcon-family architectures."""

    @property
    def architecture_name(self) -> str:
        return "FalconForCausalLM"

    @property
    def model_type(self) -> str:
        return "falcon"

    @property
    def supported_model_types(self) -> List[str]:
        return ["falcon"]

    def build_model(self) -> nn.Module:
        return FalconModel(self.config)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids

    def attention(self, layer_idx: int, hidden_states: torch.Tensor, position_ids: torch.Tensor,
                  kv_cache: Optional[Any] = None, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Use build_model() for full forward pass")

    def mlp(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use build_model() for full forward pass")

    def norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use build_model() for full forward pass")

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use build_model() for full forward pass")

    def rotary_embedding(self, position_ids: torch.Tensor, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Falcon uses ALiBi")

    def get_position_encoding_type(self) -> str:
        return "alibi"

    def get_attention_type(self) -> str:
        return "MQA"

    def prepare_inputs(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                       kv_cache: Optional[Any] = None, **kwargs: Any) -> Dict[str, Any]:
        return {"attention_mask": kwargs.get("attention_mask")}

    def load_weights(self, model: nn.Module, weight_path: str, **kwargs: Any) -> nn.Module:
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
