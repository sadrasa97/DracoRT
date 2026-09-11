"""
GPT-2 Model Adapter

Supports GPT-2 and GPT-NeoX-family architectures.
Features: MHA, learned positional embeddings, standard MLP.
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

logger = logging.getLogger("draco.models.gpt2")


class GPT2LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class GPT2Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        self.c_attn = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.shape
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.hidden_size, dim=-1)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

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
        return self.c_proj(attn_output), present_key_value


class GPT2MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        intermediate = config.intermediate_size or 4 * config.hidden_size
        self.c_fc = nn.Linear(config.hidden_size, intermediate)
        self.c_proj = nn.Linear(intermediate, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class GPT2DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.ln_1 = GPT2LayerNorm(config.hidden_size)
        self.attn = GPT2Attention(config, layer_idx)
        self.ln_2 = GPT2LayerNorm(config.hidden_size)
        self.mlp = GPT2MLP(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states, present_key_value = self.attn(hidden_states, attention_mask, past_key_value, use_cache)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value


class GPT2FullModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.h = nn.ModuleList([GPT2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.ln_f = GPT2LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
                use_cache: bool = False):
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            past_len = 0
            if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
                past_len = past_key_values[0][0].shape[2]
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1) + past_len

        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
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


class GPT2Adapter(ModelAdapter):
    """Model adapter for GPT-2-family architectures."""

    @property
    def architecture_name(self) -> str:
        return "GPT2LMHeadModel"

    @property
    def model_type(self) -> str:
        return "gpt2"

    @property
    def supported_model_types(self) -> List[str]:
        return ["gpt2", "gpt_neox", "gptj"]

    def build_model(self) -> nn.Module:
        return GPT2FullModel(self.config)

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
        raise NotImplementedError("GPT-2 uses learned positional embeddings")

    def get_position_encoding_type(self) -> str:
        return "learned"

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
        return self.config.head_dim

    def get_vocab_size(self) -> int:
        return self.config.vocab_size
