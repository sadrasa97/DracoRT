"""
Model Executor

Orchestrates model forward passes, weight loading, and KV cache management.
Acts as the bridge between the LLM Engine and individual ModelAdapters.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401 (Tuple used in signatures)

import torch
import torch.nn as nn

from draco.models.adapter import ModelAdapter
from draco.models.config import ModelConfig

logger = logging.getLogger("draco.models.executor")


class ModelExecutor:
    """
    Executes forward passes through the model.

    Manages the lifecycle of a model adapter: weight loading, KV cache,
    input preparation, and forward pass execution.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        config: ModelConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.device = device
        self.dtype = dtype
        self.model: Optional[nn.Module] = None
        self.kv_cache: Optional[Any] = None

    def initialize(self) -> None:
        """Build the model and move to device."""
        logger.info("Building model: %s", self.adapter.architecture_name)
        self.model = self.adapter.build_model()
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        logger.info("Model built and moved to %s", self.device)

    def load_weights(self, model_path: str, **kwargs: Any) -> None:
        """Load model weights from a checkpoint."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        logger.info("Loading weights from %s", model_path)
        self.model = self.adapter.load_weights(self.model, model_path, **kwargs)
        logger.info("Weights loaded successfully")

    def allocate_kv_cache(
        self,
        num_blocks: int,
        block_size: int,
    ) -> Any:
        """Allocate KV cache for the model."""
        self.kv_cache = self.adapter.allocate_kv_cache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            dtype=self.dtype,
            device=self.device,
        )
        return self.kv_cache

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ):
        """
        Execute a forward pass through the model.

        With ``use_cache=False`` (default), returns the logits tensor of
        shape (batch_size, seq_len, vocab_size) — unchanged behavior for
        any adapter that hasn't been upgraded to accept a cache yet.

        With ``use_cache=True``, returns ``(logits, present_key_values)``
        and only architectures whose build_model() nn.Module accepts
        ``past_key_values``/``use_cache`` kwargs (all 14 currently
        registered architectures do) will actually reuse cached K/V.

        ``attention_mask`` in kwargs (an additive bias, e.g. for padded
        batched sequences — see LLM._generate_batch) is forwarded to the
        model on both branches; it used to be silently dropped on the
        use_cache=True branch, which broke padding masks for anything
        calling forward(use_cache=True) with a batch of different-length
        sequences.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        attention_mask = kwargs.get("attention_mask")

        with torch.no_grad():
            if use_cache:
                out = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits, present_key_values = out
                self.kv_cache = present_key_values
                return logits, present_key_values
            # Use the built nn.Module directly — adapters raise
            # NotImplementedError for per-layer methods and only
            # support the full model forward pass via build_model().
            logits = self.model(input_ids, attention_mask=attention_mask)
            return logits

    @torch.inference_mode()
    def generate_logits(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate logits with inference mode enabled."""
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[-1], device=self.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)
        return self.forward(input_ids, position_ids)

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the loaded model."""
        param_count = 0
        if self.model is not None:
            param_count = sum(p.numel() for p in self.model.parameters())

        memory_bytes = 0
        if self.model is not None:
            memory_bytes = sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            )

        return {
            "architecture": self.adapter.architecture_name,
            "model_type": self.adapter.model_type,
            "num_layers": self.adapter.get_num_layers(),
            "hidden_size": self.adapter.get_hidden_size(),
            "num_heads": self.adapter.get_num_heads(),
            "num_kv_heads": self.adapter.get_num_kv_heads(),
            "head_dim": self.adapter.get_head_dim(),
            "vocab_size": self.adapter.get_vocab_size(),
            "num_parameters": param_count,
            "memory_bytes": memory_bytes,
            "attention_type": self.adapter.get_attention_type(),
            "is_moe": self.adapter.is_moe(),
            "device": str(self.device),
            "dtype": str(self.dtype),
        }
