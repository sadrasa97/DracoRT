"""
Model Runner

High-level model runner that handles initialization, weight loading, and
provides the execution interface used by the LLM Engine.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from draco.exceptions import ConfigError, ModelNotFoundError
from draco.models.adapter import ModelAdapter
from draco.models.config import ModelConfig
from draco.models.executor import ModelExecutor
from draco.models.registry import MODEL_REGISTRY

logger = logging.getLogger("draco.models.runner")


class ModelRunner:
    """
    High-level model runner.

    Handles:
    - Model resolution (detecting architecture from config)
    - Weight loading (safetensors, pytorch, etc.)
    - Device placement
    - Forward pass orchestration
    """

    def __init__(
        self,
        model_path: str,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        quantization: Optional[str] = None,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        trust_remote_code: bool = False,
    ):
        self.model_path = model_path
        self.dtype = dtype or torch.float16
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantization = quantization
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code

        self.config: Optional[ModelConfig] = None
        self.adapter: Optional[ModelAdapter] = None
        self.executor: Optional[ModelExecutor] = None

    def initialize(self) -> None:
        """Full initialization: detect architecture, build model, load weights."""
        # 1. Load config
        logger.info("Loading model config from %s", self.model_path)
        self.config = ModelConfig.from_pretrained(self.model_path)

        # 2. Resolve adapter from registry
        arch = self.config.detect_architecture()
        model_type = self.config.model_type
        logger.info("Detected architecture: %s (type: %s)", arch, model_type)

        adapter_cls = MODEL_REGISTRY.resolve(
            architecture=arch,
            model_type=model_type,
        )
        logger.info("Using adapter: %s", adapter_cls.__name__)

        # 3. Create adapter instance
        self.adapter = adapter_cls(config=self.config)

        # 4. Create executor
        self.executor = ModelExecutor(
            adapter=self.adapter,
            config=self.config,
            device=self.device,
            dtype=self.dtype,
        )

        # 5. Build model
        self.executor.initialize()

        # 6. Load weights
        self.executor.load_weights(self.model_path)

        logger.info("Model initialized successfully")
        logger.info("Model info: %s", self.executor.get_model_info())

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute a forward pass. Returns logits, or (logits, present_key_values)
        when use_cache=True — see ModelExecutor.forward."""
        if self.executor is None:
            raise RuntimeError("ModelRunner not initialized. Call initialize() first.")
        return self.executor.forward(
            input_ids,
            position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.executor is None:
            return {"status": "not_initialized"}
        return self.executor.get_model_info()

    def supports_quantization(self, method: str) -> bool:
        """Check if the loaded model supports a quantization method."""
        if self.adapter is None:
            return False
        return self.adapter.supports_quantization(method)
