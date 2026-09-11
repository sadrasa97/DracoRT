"""
Model Configuration

Represents the configuration for a model, derived from HuggingFace config.json.
Provides a unified interface for model properties regardless of architecture.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from draco.exceptions import ConfigError


@dataclass
class AttentionConfig:
    """Attention-specific configuration."""

    attention_bias: bool = False
    sliding_window: Optional[int] = None
    sliding_window_size: Optional[int] = None
    attention_dropout: float = 0.0
    attention_type: str = "MHA"  # MHA, MQA, GQA
    use_flash_attention: bool = True


@dataclass
class MLPConfig:
    """MLP/FFN-specific configuration."""

    intermediate_size: int = 0
    mlp_bias: bool = False
    hidden_act: str = "silu"
    mlp_type: str = "standard"  # standard, gated, moe


@dataclass
class MoEConfig:
    """Mixture-of-Experts configuration."""

    num_experts: int = 0
    num_experts_per_tok: int = 0
    expert_capacity: Optional[int] = None
    use_expert_parallelism: bool = False
    router_type: str = "top_k"
    norm_topk_prob: bool = True
    moe_intermediate_size: Optional[int] = None


@dataclass
class PositionConfig:
    """Position encoding configuration."""

    position_encoding_type: str = "rope"  # rope, alibi, learned
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    max_position_embeddings: int = 4096
    rope_adjustments: Optional[Dict[str, Any]] = None


@dataclass
class QuantizationConfig:
    """Quantization-related configuration detected from metadata."""

    quantization_method: Optional[str] = None
    bits: Optional[int] = None
    group_size: Optional[int] = None
    zero_point: Optional[bool] = None
    symmetric: Optional[bool] = None
    quantization_dtype: Optional[str] = None
    quantization_scheme: Optional[str] = None

    @property
    def is_quantized(self) -> bool:
        return self.quantization_method is not None


class ModelConfig:
    """
    Unified model configuration.

    Wraps HuggingFace config.json and provides architecture-agnostic access
    to all model properties. Detects architecture from config and provides
    typed access to model parameters.
    """

    def __init__(self, config_data: Optional[Dict[str, Any]] = None, model_path: Optional[str] = None):
        if config_data is not None:
            self._data = config_data
        elif model_path is not None:
            self._data = self._load_config(model_path)
        else:
            self._data = {}

        # Parse sub-configs
        self.attention = self._parse_attention_config()
        self.mlp = self._parse_mlp_config()
        self.moe = self._parse_moe_config()
        self.position = self._parse_position_config()
        self.quantization = self._parse_quantization_config()

    @staticmethod
    def _load_config(model_path: str) -> Dict[str, Any]:
        """Load config.json from a model path."""
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            raise ConfigError(f"config.json not found at {config_path}")
        with open(config_path, "r") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Architecture detection
    # ------------------------------------------------------------------

    @property
    def architectures(self) -> List[str]:
        """List of architecture names from config."""
        return self._data.get("architectures", [])

    @property
    def architecture(self) -> Optional[str]:
        """First (primary) architecture name."""
        archs = self.architectures
        return archs[0] if archs else None

    @property
    def model_type(self) -> str:
        """HuggingFace model_type identifier."""
        return self._data.get("model_type", "unknown")

    @property
    def auto_map(self) -> Optional[Dict[str, str]]:
        """Auto-map from config for custom architectures."""
        return self._data.get("auto_map")

    # ------------------------------------------------------------------
    # Core model dimensions
    # ------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        return self._data.get("hidden_size", self._data.get("n_embd", 0))

    @property
    def num_hidden_layers(self) -> int:
        return self._data.get("num_hidden_layers", self._data.get("n_layer", 0))

    @property
    def num_attention_heads(self) -> int:
        return self._data.get("num_attention_heads", self._data.get("n_head", 0))

    @property
    def num_key_value_heads(self) -> int:
        return self._data.get("num_key_value_heads", self.num_attention_heads)

    @property
    def vocab_size(self) -> int:
        return self._data.get("vocab_size", 0)

    @property
    def max_position_embeddings(self) -> int:
        return self._data.get("max_position_embeddings", 4096)

    @property
    def head_dim(self) -> int:
        """Per-head dimension, computed if not in config."""
        if "head_dim" in self._data:
            return self._data["head_dim"]
        return self.hidden_size // self.num_attention_heads

    @property
    def intermediate_size(self) -> int:
        return self._data.get("intermediate_size", self._data.get("n_inner", 0))

    @property
    def rope_theta(self) -> float:
        return float(self._data.get("rope_theta", 10000.0))

    @property
    def rope_scaling(self) -> Optional[Dict[str, Any]]:
        return self._data.get("rope_scaling")

    @property
    def hidden_act(self) -> str:
        return self._data.get("hidden_act", "silu")

    @property
    def tie_word_embeddings(self) -> bool:
        return self._data.get("tie_word_embeddings", False)

    @property
    def dtype_str(self) -> str:
        return self._data.get("torch_dtype", "float16")

    # ------------------------------------------------------------------
    # Quantization detection
    # ------------------------------------------------------------------

    @property
    def quantization_config(self) -> Optional[Dict[str, Any]]:
        """Raw quantization config from config.json."""
        return self._data.get("quantization_config")

    @property
    def is_quantized(self) -> bool:
        """Whether this model appears to be quantized."""
        qc = self.quantization_config
        if qc is not None:
            return True
        # Check for GPTQ/AWQ specific keys
        if "quantization_config" in self._data:
            return True
        return self.quantization.is_quantized

    # ------------------------------------------------------------------
    # Sub-config parsing
    # ------------------------------------------------------------------

    def _parse_attention_config(self) -> AttentionConfig:
        sliding = self._data.get("sliding_window")
        return AttentionConfig(
            attention_bias=self._data.get("attention_bias", False),
            sliding_window=sliding,
            sliding_window_size=sliding,
            attention_dropout=self._data.get("attention_dropout", 0.0),
        )

    def _parse_mlp_config(self) -> MLPConfig:
        num_experts = self._data.get("num_experts", 0)
        mlp_type = "moe" if num_experts > 0 else "standard"
        return MLPConfig(
            intermediate_size=self.intermediate_size,
            mlp_bias=self._data.get("mlp_bias", False),
            hidden_act=self.hidden_act,
            mlp_type=mlp_type,
        )

    def _parse_moe_config(self) -> MoEConfig:
        num_experts = self._data.get("num_experts", 0)
        return MoEConfig(
            num_experts=num_experts,
            num_experts_per_tok=self._data.get("num_experts_per_tok",
                                                 self._data.get("num_expert_per_tok", 0)),
            expert_capacity=self._data.get("expert_capacity"),
            use_expert_parallelism=self._data.get("use_expert_parallelism", False),
            norm_topk_prob=self._data.get("norm_topk_prob", True),
            moe_intermediate_size=self._data.get("moe_intermediate_size"),
        )

    def _parse_position_config(self) -> PositionConfig:
        rope_type = "rope"
        # Check for ALiBi
        if self._data.get("alibi", False):
            rope_type = "alibi"
        # Check for learned position embeddings
        if self._data.get("position_embedding_type") == "relative":
            rope_type = "learned"

        return PositionConfig(
            position_encoding_type=rope_type,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            max_position_embeddings=self.max_position_embeddings,
            rope_adjustments=self._data.get("rope_adjustments"),
        )

    def _parse_quantization_config(self) -> QuantizationConfig:
        qc = self.quantization_config
        if qc is None:
            return QuantizationConfig()
        return QuantizationConfig(
            quantization_method=qc.get("quant_method", qc.get("quantization_method")),
            bits=qc.get("bits"),
            group_size=qc.get("group_size"),
            zero_point=qc.get("zero_point"),
            symmetric=qc.get("symmetric"),
            quantization_dtype=qc.get("dtype"),
            quantization_scheme=qc.get("scheme"),
        )

    # ------------------------------------------------------------------
    # Raw data access
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Access raw config data."""
        return self._data.get(key, default)

    @property
    def data(self) -> Dict[str, Any]:
        """Raw config dictionary."""
        return self._data

    # ------------------------------------------------------------------
    # Detection from model path
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_path: str) -> ModelConfig:
        """Create a ModelConfig from a local path or HuggingFace model ID."""
        path = Path(model_path)
        if path.exists():
            return cls(model_path=str(path))
        # For HuggingFace IDs, try downloading config.json
        try:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(repo_id=model_path, filename="config.json")
            with open(config_file, "r") as f:
                data = json.load(f)
            return cls(config_data=data)
        except Exception as e:
            raise ConfigError(f"Failed to load config for '{model_path}': {e}")

    def detect_architecture(self) -> str:
        """Detect the architecture string for registry lookup."""
        arch = self.architecture
        if arch:
            return arch
        # Fallback to model_type
        mt = self.model_type
        if mt:
            return mt
        raise ConfigError(
            "Cannot detect architecture: no 'architectures' or 'model_type' in config"
        )

    def __repr__(self) -> str:
        return (
            f"ModelConfig("
            f"arch={self.architecture!r}, "
            f"type={self.model_type!r}, "
            f"layers={self.num_hidden_layers}, "
            f"hidden={self.hidden_size}, "
            f"heads={self.num_attention_heads}, "
            f"kv_heads={self.num_key_value_heads}, "
            f"vocab={self.vocab_size}"
            f")"
        )
