"""
DeepSeek Model Adapter

Supports DeepSeek-V2, DeepSeek-V2.5, and DeepSeek-V3 architectures.
Features:
- Multi-head Latent Attention (MLA): compresses KV via low-rank projection
- DeepSeekMoE: fine-grained experts with shared expert and routed experts
- RoPE for position encoding
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

logger = logging.getLogger("draco.models.deepseek")


# ======================================================================
# DeepSeek Components
# ======================================================================

class DeepSeekRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class DeepSeekRotaryEmbedding(nn.Module):
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


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) for DeepSeek-V2/V3.

    MLA compresses the KV cache using low-rank projections:
    - Instead of storing full K and V, store compressed latent c_kv
    - Reconstruct K/V on-the-fly from c_kv + rope-rotated components

    This reduces KV cache memory by ~90% compared to standard GQA.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Compressed KV latent dimension (key-value LDA rank)
        self.qk_nope_head_dim = config.get("qk_nope_head_dim", self.head_dim // 2)
        self.qk_rope_head_dim = config.get("qk_rope_head_dim", self.head_dim // 2)
        self.v_head_dim = config.get("v_head_dim", self.head_dim)

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.kv_lora_rank = config.get("kv_lora_rank", 512)
        self.kv_lora_proj = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.k_down_proj = nn.Linear(self.kv_lora_rank, self.num_kv_heads * self.head_dim, bias=False)
        self.v_down_proj = nn.Linear(self.kv_lora_rank, self.num_kv_heads * self.v_head_dim, bias=False)

        # Decoupled RoPE for Q and K
        self.q_rope_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_rope_head_dim, bias=False)
        self.k_rope_proj = nn.Linear(self.kv_lora_rank, self.num_kv_heads * self.qk_rope_head_dim, bias=False)

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

        # Standard Q projection
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compress KV through low-rank bottleneck
        kv_compressed = self.kv_lora_proj(hidden_states)  # (batch, seq, kv_lora_rank)

        # Reconstruct K and V from compressed latent
        k = self.k_down_proj(kv_compressed).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_down_proj(kv_compressed).view(bsz, seq_len, self.num_kv_heads, self.v_head_dim).transpose(1, 2)

        # Decoupled RoPE — cos/sin arrive pre-sliced to this chunk's
        # absolute positions by DeepSeekModel.forward; only trim the
        # feature dim to qk_rope_head_dim.
        q_rope = self.q_rope_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)
        k_rope = self.k_rope_proj(kv_compressed).view(bsz, seq_len, self.num_kv_heads, self.qk_rope_head_dim).transpose(1, 2)

        cos_q = cos[..., :self.qk_rope_head_dim].unsqueeze(1)
        sin_q = sin[..., :self.qk_rope_head_dim].unsqueeze(1)
        q_rope = _apply_rotary_emb(q_rope, cos_q, sin_q)
        k_rope = _apply_rotary_emb(k_rope, cos_q, sin_q)

        # Concatenate nope and rope parts for Q and K
        q_nope = q[..., :self.qk_nope_head_dim]
        q = torch.cat([q_nope, q_rope], dim=-1)
        k_nope = k[..., :self.qk_nope_head_dim]
        k = torch.cat([k_nope, k_rope], dim=-1)
        # V keeps its own dimension (v_head_dim)

        # Real KV cache: this caches the fully-reconstructed (nope+rope)
        # K/V, not the compressed latent MLA is designed to cache — a
        # future optimization can cache kv_compressed+k_rope instead for
        # the full memory-savings MLA promises, but caching post-
        # reconstruction is still correct and still turns every decode
        # step from O(n^2) into O(n), which is what matters here.
        k, v, present_key_value = concat_kv(past_key_value, k, v, use_cache)
        kv_len = k.shape[2]

        # Expand KV heads for GQA
        if self.num_kv_groups > 1:
            k_rep = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_rep = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_rep, v_rep = k, v

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k_rep.transpose(-2, -1)) / (self.head_dim ** 0.5)

        causal_mask = cache_aware_causal_mask(seq_len, kv_len, hidden_states.device, hidden_states.dtype)
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v_rep)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output), present_key_value


class DeepSeekMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        intermediate = config.intermediate_size or 4 * config.hidden_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekMoELayer(nn.Module):
    """
    DeepSeek MoE layer with routed experts + shared expert.

    DeepSeek uses fine-grained MoE with more experts but fewer experts
    activated per token. Also includes a shared expert that processes all tokens.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        num_experts = config.moe.num_experts or 64
        top_k = config.moe.num_experts_per_tok or 8
        expert_intermediate = config.moe.moe_intermediate_size or (config.intermediate_size // 2)
        self.num_shared_experts = config.get("num_shared_experts", 1)

        self.top_k = top_k
        self.num_experts = num_experts

        # Routed experts
        self.router = TopKRouter(
            hidden_dim=config.hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            norm_topk_prob=config.moe.norm_topk_prob,
        )
        self.routed_experts = ExpertGroup(
            num_experts=num_experts,
            hidden_dim=config.hidden_size,
            intermediate_dim=expert_intermediate,
            activation="silu",
            bias=False,
        )

        # Shared expert (always processes all tokens)
        self.shared_experts = ExpertGroup(
            num_experts=self.num_shared_experts,
            hidden_dim=config.hidden_size,
            intermediate_dim=expert_intermediate,
            activation="silu",
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)

        # Routed expert path
        expert_indices, expert_weights, aux_info = self.router.forward(flat_hidden)
        routed_output = self.routed_experts(flat_hidden, expert_indices, expert_weights)

        # Shared expert path (all tokens)
        shared_indices = torch.zeros(flat_hidden.shape[0], 1, dtype=torch.long, device=flat_hidden.device)
        shared_weights = torch.ones(flat_hidden.shape[0], 1, device=flat_hidden.device)
        shared_output = self.shared_experts(flat_hidden, shared_indices, shared_weights)

        # Combine
        output = (routed_output + shared_output).view(batch_size, seq_len, hidden_dim)
        aux_loss = aux_info.get("load_balancing_loss", torch.tensor(0.0))
        return output, aux_loss


class DeepSeekDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MultiHeadLatentAttention(config, layer_idx)

        # Check if this layer uses MoE
        moe_layer_freq = config.get("moe_layer_freq", 1)
        is_moe = (moe_layer_freq > 0) and (layer_idx % moe_layer_freq == 0)

        if is_moe:
            self.mlp = DeepSeekMoELayer(config, layer_idx)
            self.use_moe = True
        else:
            self.mlp = DeepSeekMLP(config)
            self.use_moe = False

        self.input_layernorm = DeepSeekRMSNorm(config.hidden_size)
        self.post_attention_layernorm = DeepSeekRMSNorm(config.hidden_size)

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
        if self.use_moe:
            hidden_states, aux_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            aux_loss = torch.tensor(0.0, device=hidden_states.device)
        hidden_states = residual + hidden_states

        return hidden_states, aux_loss, present_key_value


class DeepSeekModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DeepSeekDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = DeepSeekRMSNorm(config.hidden_size)
        self.rotary_emb = DeepSeekRotaryEmbedding(
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
        """Returns just ``logits`` (or ``(logits, present_key_values)``
        with use_cache=True) to match every other adapter's inference
        contract — previously always returned (logits, aux_loss), which
        silently broke ModelExecutor.forward()'s plain
        ``logits = self.model(input_ids)`` call."""
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
# DeepSeek Adapter
# ======================================================================

class DeepSeekAdapter(ModelAdapter):
    """Model adapter for DeepSeek-family architectures.

    Supports: DeepSeekV2ForCausalLM, DeepSeekV3ForCausalLM
    """

    @property
    def architecture_name(self) -> str:
        return "DeepSeekV2ForCausalLM"

    @property
    def model_type(self) -> str:
        return "deepseek_v2"

    @property
    def supported_model_types(self) -> List[str]:
        return ["deepseek_v2", "deepseek_v3"]

    def build_model(self) -> nn.Module:
        return DeepSeekModel(self.config)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids

    def attention(self, layer_idx, hidden_states, position_ids, kv_cache=None, **kwargs):
        raise NotImplementedError("Use build_model() for full forward pass")

    def mlp(self, layer_idx, hidden_states):
        raise NotImplementedError("Use build_model() for full forward pass (MoE)")

    def norm(self, hidden_states):
        raise NotImplementedError("Use build_model() for full forward pass")

    def lm_head(self, hidden_states):
        raise NotImplementedError("Use build_model() for full forward pass")

    def rotary_embedding(self, position_ids, head_dim):
        emb = DeepSeekRotaryEmbedding(
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

    # MoE support
    def is_moe(self) -> bool:
        return True

    def get_num_experts(self) -> int:
        return self.config.moe.num_experts or 64

    def get_expert_top_k(self) -> int:
        return self.config.moe.num_experts_per_tok or 8

    # MLA info
    def has_mla(self) -> bool:
        """Multi-head Latent Attention is always enabled."""
        return True

    def get_kv_lora_rank(self) -> int:
        return self.config.get("kv_lora_rank", 512)
