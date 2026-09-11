"""
Draco Model System

Universal model architecture support through a plugin/registry-based system.

Each Hugging Face causal LM architecture is added as a ModelAdapter that plugs into
the MODEL_REGISTRY. The runtime automatically detects architecture from config.json
and selects the correct implementation.
"""

from draco.models.registry import MODEL_REGISTRY, ModelRegistry
from draco.models.adapter import ModelAdapter
from draco.models.config import ModelConfig

__all__ = ["MODEL_REGISTRY", "ModelRegistry", "ModelAdapter", "ModelConfig"]
