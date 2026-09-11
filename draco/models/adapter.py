"""
Model Adapter Base Classes

Every model implementation exposes standardized operations:
forward(), embed(), attention(), mlp(), norm(), rotary_embedding(),
prepare_inputs(), load_weights(), allocate_kv_cache().

Where architectures differ, use specialized implementations.
Do not force every architecture into an incorrect common implementation.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from draco.models.config import ModelConfig


class ModelAdapter(abc.ABC):
    """
    Abstract base class for all model adapters.

    Every HuggingFace causal LM architecture implements this interface.
    The adapter translates between HuggingFace config/weights and Draco's
    internal model representation.

    Subclasses must implement all abstract methods. Non-abstract helper
    methods provide sensible defaults where applicable.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Architecture detection
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def architecture_name(self) -> str:
        """HuggingFace architecture string, e.g. 'LlamaForCausalLM'."""
        ...

    @property
    @abc.abstractmethod
    def model_type(self) -> str:
        """HuggingFace model_type, e.g. 'llama', 'qwen2'."""
        ...

    @property
    def supported_model_types(self) -> List[str]:
        """List of model_type strings this adapter handles (default: [model_type])."""
        return [self.model_type]

    # ------------------------------------------------------------------
    # Core model components (must be implemented per-architecture)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return the complete model (all layers)."""
        ...

    @abc.abstractmethod
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token embedding lookup."""
        ...

    @abc.abstractmethod
    def attention(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[Any] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention for a single transformer layer."""
        ...

    @abc.abstractmethod
    def mlp(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute MLP/FFN for a single transformer layer."""
        ...

    @abc.abstractmethod
    def norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply final layer norm (e.g. RMSNorm, LayerNorm)."""
        ...

    @abc.abstractmethod
    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits."""
        ...

    @abc.abstractmethod
    def rotary_embedding(
        self,
        position_ids: torch.Tensor,
        head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary position embeddings (cos, sin)."""
        ...

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def prepare_inputs(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare all inputs for a forward pass."""
        ...

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load_weights(
        self,
        model: nn.Module,
        weight_path: str,
        **kwargs: Any,
    ) -> nn.Module:
        """Load weights into the model from a checkpoint path."""
        ...

    @abc.abstractmethod
    def get_weight_map(self) -> Dict[str, str]:
        """
        Return a mapping from HuggingFace weight names to Draco weight names.

        Example:
            {"model.embed_tokens.weight": "embed_tokens.weight",
             "model.layers.0.self_attn.q_proj.weight": "layers.0.self_attn.q_proj.weight"}
        """
        ...

    @abc.abstractmethod
    def get_num_layers(self) -> int:
        """Return the number of transformer layers."""
        ...

    @abc.abstractmethod
    def get_hidden_size(self) -> int:
        """Return the hidden dimension."""
        ...

    @abc.abstractmethod
    def get_num_heads(self) -> int:
        """Return the number of attention heads."""
        ...

    @abc.abstractmethod
    def get_num_kv_heads(self) -> int:
        """Return the number of key/value heads (for GQA/MQA)."""
        ...

    @abc.abstractmethod
    def get_head_dim(self) -> int:
        """Return the per-head dimension."""
        ...

    @abc.abstractmethod
    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        ...

    # ------------------------------------------------------------------
    # MoE support (optional override)
    # ------------------------------------------------------------------

    def is_moe(self) -> bool:
        """Whether this architecture is a Mixture-of-Experts model."""
        return False

    def get_num_experts(self) -> int:
        """Number of experts (only for MoE)."""
        return 0

    def get_expert_top_k(self) -> int:
        """Top-K routing (only for MoE)."""
        return 0

    # ------------------------------------------------------------------
    # KV cache
    # ------------------------------------------------------------------

    def allocate_kv_cache(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Any:
        """
        Allocate KV cache tensors.

        Default implementation allocates standard key/value cache.
        Override for architecture-specific caching (e.g. GQA reshaping).
        """
        key_cache = torch.zeros(
            num_layers, num_blocks, num_kv_heads, block_size, head_dim,
            dtype=dtype, device=device,
        )
        value_cache = torch.zeros(
            num_layers, num_blocks, num_kv_heads, block_size, head_dim,
            dtype=dtype, device=device,
        )
        return {"key": key_cache, "value": value_cache}

    # ------------------------------------------------------------------
    # Attention type queries
    # ------------------------------------------------------------------

    def get_attention_type(self) -> str:
        """Return the attention type: 'MHA', 'MQA', or 'GQA'."""
        num_kv = self.get_num_kv_heads()
        num_heads = self.get_num_heads()
        if num_kv == num_heads:
            return "MHA"
        elif num_kv == 1:
            return "MQA"
        else:
            return "GQA"

    def has_sliding_window(self) -> bool:
        """Whether the architecture uses sliding window attention."""
        return False

    def get_sliding_window_size(self) -> Optional[int]:
        """Sliding window size if applicable."""
        return None

    # ------------------------------------------------------------------
    # Position encoding
    # ------------------------------------------------------------------

    def get_position_encoding_type(self) -> str:
        """Return position encoding type: 'rope', 'alibi', 'learned', etc."""
        return "rope"

    # ------------------------------------------------------------------
    # Quantization support
    # ------------------------------------------------------------------

    def supports_quantization(self, method: str) -> bool:
        """Check if this architecture supports a given quantization method."""
        return method.lower() in ("fp8", "int8", "int4")

    def get_quantizable_modules(self) -> List[str]:
        """Return list of module names that can be quantized."""
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"architecture={self.architecture_name!r}, "
            f"model_type={self.model_type!r}, "
            f"layers={self.get_num_layers()}, "
            f"hidden={self.get_hidden_size()}, "
            f"heads={self.get_num_heads()}, "
            f"kv_heads={self.get_num_kv_heads()}, "
            f"vocab={self.get_vocab_size()}"
            f")"
        )
