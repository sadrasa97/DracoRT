"""
OPT Model Adapter

Supports OPT (Open Pre-trained Transformer) family: OPT-125M through OPT-175B.
Features:
- Learned position embeddings (not RoPE)
- LayerNorm (not RMSNorm)
- Parallel attention + MLP with scaled residual
- GELU activation in MLP
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

logger = logging.getLogger("draco.models.opt")


class OPTLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, eps=self.eps)


class OPTAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        k, v, present_key_value = concat_kv(past_key_value, k, v, use_cache)
        kv_len = k.shape[2]

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        causal_mask = cache_aware_causal_mask(seq_len, kv_len, hidden_states.device, hidden_states.dtype)
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(attn_output), present_key_value


class OPTMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        intermediate = config.intermediate_size or 4 * config.hidden_size
        self.fc1 = nn.Linear(config.hidden_size, intermediate, bias=True)
        self.fc2 = nn.Linear(intermediate, config.hidden_size, bias=True)
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))


class OPTDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = OPTAttention(config, layer_idx)
        self.self_attn_layer_norm = OPTLayerNorm(config.hidden_size)
        self.mlp = OPTMLP(config)
        self.final_layer_norm = OPTLayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # OPT uses sequential (not parallel) residual
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, present_key_value = self.self_attn(hidden_states, attention_mask, past_key_value, use_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value


class OPTModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.layers = nn.ModuleList([
            OPTDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.final_layer_norm = OPTLayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ):
        bsz, seq_len = input_ids.shape
        if position_ids is None:
            past_len = 0
            if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
                past_len = past_key_values[0][0].shape[2]
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1) + past_len

        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(position_ids)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        present_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present_kv = layer(hidden_states, attention_mask, past_kv, use_cache)
            present_key_values.append(present_kv)

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if use_cache:
            return logits, present_key_values
        return logits


class OPTAdapter(ModelAdapter):
    """Model adapter for OPT-family architectures."""

    @property
    def architecture_name(self) -> str:
        return "OPTForCausalLM"

    @property
    def model_type(self) -> str:
        return "opt"

    @property
    def supported_model_types(self) -> List[str]:
        return ["opt"]

    def build_model(self) -> nn.Module:
        return OPTModel(self.config)

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
        raise NotImplementedError("OPT uses learned position embeddings")

    def get_position_encoding_type(self) -> str:
        return "learned"

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
        return self.config.num_attention_heads  # OPT is MHA

    def get_head_dim(self) -> int:
        return self.config.hidden_size // self.config.num_attention_heads

    def get_vocab_size(self) -> int:
        return self.config.vocab_size
