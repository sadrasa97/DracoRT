"""
Qwen2 Model Adapter

Supports Qwen2 and Qwen2.5 architectures.
Features: GQA, RoPE, SwiGLU MLP.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from draco.models.adapter import ModelAdapter
from draco.models.config import ModelConfig

logger = logging.getLogger("draco.models.qwen2")


class Qwen2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 1000000.0):
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


class Qwen2Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # cos/sin are already sliced to this chunk's absolute positions by
        # the caller (Qwen2Model.forward), so no extra [:seq_len] slicing
        # here — that used to silently assume positions always start at 0,
        # which breaks the moment we feed only the newest token with a
        # nonzero position (the whole point of KV-cached decoding).
        cos_ = cos.unsqueeze(1)
        sin_ = sin.unsqueeze(1)
        q = _apply_rotary_emb(q, cos_, sin_)
        k = _apply_rotary_emb(k, cos_, sin_)

        # Concatenate with cached K/V from previous steps (real KV cache).
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present_key_value = (k, v) if use_cache else None

        kv_len = k.shape[2]
        if self.num_kv_groups > 1:
            k_rep = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_rep = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_rep, v_rep = k, v

        attn_weights = torch.matmul(q, k_rep.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask must cover the FULL key length (cached + new), and
        # only mask query position i against key positions > i in
        # absolute terms — i.e. offset by how many cached keys precede
        # this chunk's queries. With seq_len==1 (decode step) this mask
        # is an effective no-op (nothing to hide), which is correct.
        past_len = kv_len - seq_len
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


class Qwen2MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states, cos, sin, attention_mask, past_key_value, use_cache
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value


class Qwen2Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen2RMSNorm(config.hidden_size)
        self.rotary_emb = Qwen2RotaryEmbedding(
            dim=config.head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
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

        # When decoding with a cache, position_ids must continue from
        # where the cache left off (e.g. token 47, not "0" again) — that
        # off-by-cache-length bug is what made single-new-token decoding
        # produce garbage before any cache existed at all.
        if position_ids is None:
            past_len = 0
            if past_key_values is not None and past_key_values[0] is not None:
                past_len = past_key_values[0][0].shape[2]
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device) + past_len
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

        cos, sin = self.rotary_emb(position_ids)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        present_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present_kv = layer(
                hidden_states, cos, sin, attention_mask, past_kv, use_cache
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = torch.matmul(hidden_states, self.embed_tokens.weight.t())

        if use_cache:
            return logits, present_key_values
        return logits


class Qwen2Adapter(ModelAdapter):
    """Model adapter for Qwen2-family architectures."""

    @property
    def architecture_name(self) -> str:
        return "Qwen2ForCausalLM"

    @property
    def model_type(self) -> str:
        return "qwen2"

    @property
    def supported_model_types(self) -> List[str]:
        return ["qwen2", "qwen2_5", "qwen3"]

    def build_model(self) -> nn.Module:
        return Qwen2Model(self.config)

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
        emb = Qwen2RotaryEmbedding(dim=head_dim, max_position_embeddings=self.config.max_position_embeddings, base=self.config.rope_theta)
        return emb(position_ids)

    def prepare_inputs(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                       kv_cache: Optional[Any] = None, **kwargs: Any) -> Dict[str, Any]:
        return {"attention_mask": kwargs.get("attention_mask")}

    def load_weights(self, model: nn.Module, weight_path: str, **kwargs: Any) -> nn.Module:
        from draco.weights.loader import get_checkpoint_loader
        loader = get_checkpoint_loader()
        weights = loader.load(weight_path, device=next(model.parameters()).device)
        weight_map = self.get_weight_map()
        remapped = {weight_map.get(k, k): v for k, v in weights.items()}

        # HuggingFace checkpoints store transformer weights under a
        # "model." prefix (e.g. "model.embed_tokens.weight") but our
        # Draco Qwen2Model uses unprefixed names ("embed_tokens.weight").
        # With strict=False any unmatched keys are silently ignored, which
        # means the model would keep its random initialization and produce
        # garbage output.  Strip the prefix so the weights actually land.
        model_state = model.state_dict()
        remapped_stripped: dict = {}
        for k, v in remapped.items():
            stripped = k
            if k.startswith("model.") and k[6:] not in remapped:
                stripped = k[6:]
            if stripped in model_state:
                remapped_stripped[stripped] = v
            else:
                remapped_stripped[k] = v

        model.load_state_dict(remapped_stripped, strict=False)
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
