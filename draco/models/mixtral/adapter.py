"""
Mixtral Model Adapter

Supports Mixtral-8x7B and Mixtral-8x22B architectures.
Features: Sparse Mixture-of-Experts (8 experts, top-2 routing),
GQA attention, RoPE, sliding window attention.
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
from draco.models.moe.expert import ExpertGroup
from draco.models.moe.router import TopKRouter

logger = logging.getLogger("draco.models.mixtral")


# ======================================================================
# Mixtral Components
# ======================================================================


class MixtralRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class MixtralRotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.unsqueeze(0).unsqueeze(0)
        position_ids = position_ids.unsqueeze(-1).float()
        freqs = inv_freq * position_ids
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


class MixtralAttention(nn.Module):
    """Multi-Head / Grouped-Query Attention with optional sliding window."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.sliding_window = config.attention.sliding_window or config.attention.sliding_window_size

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

        cos_ = cos.unsqueeze(1)
        sin_ = sin.unsqueeze(1)
        q = _apply_rotary_emb(q, cos_, sin_)
        k = _apply_rotary_emb(k, cos_, sin_)

        k, v, present_key_value = concat_kv(past_key_value, k, v, use_cache)
        kv_len = k.shape[2]
        past_len = kv_len - seq_len

        if self.num_kv_groups > 1:
            k_rep = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_rep = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_rep, v_rep = k, v

        attn_weights = torch.matmul(q, k_rep.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Sliding window mask, absolute-position aware
        if self.sliding_window and self.sliding_window < kv_len:
            for i in range(seq_len):
                q_abs = i + past_len
                start = max(0, q_abs - self.sliding_window + 1)
                attn_weights[:, :, i, :start] = float("-inf")

        # Causal mask
        causal_mask = cache_aware_causal_mask(seq_len, kv_len, hidden_states.device, hidden_states.dtype)
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v_rep)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output), present_key_value


class MixtralSparseMoeLayer(nn.Module):
    """Sparse Mixture-of-Experts layer for Mixtral.

    Each token is routed to top-2 experts out of 8.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        num_experts = config.moe.num_experts or 8
        top_k = config.moe.num_experts_per_tok or 2
        expert_intermediate = config.moe.moe_intermediate_size or config.intermediate_size

        self.top_k = top_k
        self.num_experts = num_experts

        # Router decides which experts each token goes to
        self.router = TopKRouter(
            hidden_dim=config.hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            norm_topk_prob=config.moe.norm_topk_prob,
        )

        # Expert group: the actual FFN experts
        self.experts = ExpertGroup(
            num_experts=num_experts,
            hidden_dim=config.hidden_size,
            intermediate_dim=expert_intermediate,
            activation="silu",
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through MoE layer.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            output: (batch, seq_len, hidden_dim)
            aux_loss: load balancing loss scalar
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Reshape for routing: (batch * seq_len, hidden_dim)
        flat_hidden = hidden_states.view(-1, hidden_dim)

        # Route tokens to experts
        expert_indices, expert_weights, aux_info = self.router.forward(flat_hidden)

        # Process through experts
        expert_output = self.experts(flat_hidden, expert_indices, expert_weights)

        # Reshape back
        output = expert_output.view(batch_size, seq_len, hidden_dim)

        aux_loss = aux_info.get("load_balancing_loss", torch.tensor(0.0))
        return output, aux_loss


class MixtralDecoderLayer(nn.Module):
    """Single Mixtral decoder layer with attention + MoE FFN."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MixtralAttention(config, layer_idx)
        self.block_sparse_moe = MixtralSparseMoeLayer(config, layer_idx)
        self.input_layernorm = MixtralRMSNorm(config.hidden_size)
        self.post_attention_layernorm = MixtralRMSNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(hidden_states, cos, sin, attention_mask, past_key_value, use_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, aux_loss, present_key_value


class MixtralModel(nn.Module):
    """Full Mixtral model with sparse MoE layers."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = MixtralRMSNorm(config.hidden_size)
        self.rotary_emb = MixtralRotaryEmbedding(
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
        """Returns just ``logits`` for plain inference (matching every
        other adapter's contract, since ModelExecutor.forward() does not
        know how to unpack a training-only aux_loss), or ``(logits,
        present_key_values)`` when use_cache=True. total_aux_loss is
        still computed (for anyone training against this adapter
        directly) but is no longer part of the inference return value —
        previously this returned (logits, aux_loss) unconditionally,
        which silently broke ModelExecutor.forward()'s plain
        ``logits = self.model(input_ids)`` call (it received a tuple)."""
        hidden_states = self.embed_tokens(input_ids)

        position_ids = derive_position_ids(input_ids, past_key_values, position_ids)
        cos, sin = self.rotary_emb(position_ids)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        present_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, aux_loss, present_kv = layer(hidden_states, cos, sin, attention_mask, past_kv, use_cache)
            total_aux_loss = total_aux_loss + aux_loss
            present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = torch.matmul(hidden_states, self.embed_tokens.weight.t())

        if use_cache:
            return logits, present_key_values
        return logits


# ======================================================================
# Mixtral Adapter
# ======================================================================


class MixtralAdapter(ModelAdapter):
    """Model adapter for Mixtral-family architectures.

    Supports: MixtralForCausalLM (Mixtral-8x7B, Mixtral-8x22B)
    """

    @property
    def architecture_name(self) -> str:
        return "MixtralForCausalLM"

    @property
    def model_type(self) -> str:
        return "mixtral"

    @property
    def supported_model_types(self) -> List[str]:
        return ["mixtral"]

    def build_model(self) -> nn.Module:
        return MixtralModel(self.config)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids

    def attention(self, layer_idx: int, hidden_states: torch.Tensor, position_ids: torch.Tensor,
                  kv_cache: Optional[Any] = None, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Use build_model() for full forward pass")

    def mlp(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use build_model() for full forward pass (MoE)")

    def norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use build_model() for full forward pass")

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use build_model() for full forward pass")

    def rotary_embedding(self, position_ids: torch.Tensor, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = MixtralRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
        )
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

    # MoE-specific
    def is_moe(self) -> bool:
        return True

    def get_num_experts(self) -> int:
        return self.config.moe.num_experts or 8

    def get_expert_top_k(self) -> int:
        return self.config.moe.num_experts_per_tok or 2

    # Sliding window
    def has_sliding_window(self) -> bool:
        return self.config.attention.sliding_window is not None

    def get_sliding_window_size(self) -> Optional[int]:
        return self.config.attention.sliding_window
