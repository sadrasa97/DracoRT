"""
Model Registry

Plugin/registry-based model system. Every HuggingFace causal LM architecture
is registered here. The runtime detects architecture from config.json and
automatically selects the correct implementation.

Example:
    MODEL_REGISTRY.register(
        architecture="LlamaForCausalLM",
        implementation=LlamaModel,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union

from draco.exceptions import ModelNotFoundError

logger = logging.getLogger("draco.models.registry")


class ModelRegistry:
    """
    Central registry for model adapters.

    Maps HuggingFace architecture strings to Draco ModelAdapter implementations.
    Supports registration by architecture name, model_type, or both.
    """

    def __init__(self) -> None:
        # architecture string -> adapter class
        self._architectures: Dict[str, Type] = {}
        # model_type string -> adapter class
        self._model_types: Dict[str, Type] = {}
        # architecture -> registered metadata
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        architecture: str,
        implementation: Type,
        model_type: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        **metadata: Any,
    ) -> None:
        """
        Register a model adapter for a given architecture.

        Args:
            architecture: HuggingFace architecture string (e.g. "LlamaForCausalLM")
            implementation: ModelAdapter subclass
            model_type: Optional HuggingFace model_type (e.g. "llama")
            aliases: Additional architecture strings this adapter handles
            **metadata: Additional metadata (e.g. status, capabilities)
        """
        if architecture in self._architectures:
            existing = self._architectures[architecture]
            if existing is not implementation:
                logger.warning(
                    "Re-registering architecture '%s': %s -> %s",
                    architecture,
                    existing.__name__,
                    implementation.__name__,
                )

        self._architectures[architecture] = implementation
        self._metadata[architecture] = {
            "model_type": model_type,
            "implementation": implementation.__name__,
            **metadata,
        }

        if model_type:
            self._model_types[model_type] = implementation

        # Register aliases
        if aliases:
            for alias in aliases:
                if alias not in self._architectures:
                    self._architectures[alias] = implementation
                    self._metadata[alias] = {
                        "model_type": model_type,
                        "implementation": implementation.__name__,
                        "alias_for": architecture,
                        **metadata,
                    }

        logger.debug("Registered '%s' -> %s", architecture, implementation.__name__)

    def get(self, architecture: str) -> Type:
        """
        Get the adapter class for a given architecture string.

        Tries architecture name first, then model_type.
        Raises ModelNotFoundError if not found.
        """
        # Try exact architecture match
        if architecture in self._architectures:
            return self._architectures[architecture]

        # Try model_type match
        if architecture in self._model_types:
            return self._model_types[architecture]

        raise ModelNotFoundError(architecture)

    def get_by_model_type(self, model_type: str) -> Optional[Type]:
        """Get adapter by model_type string."""
        return self._model_types.get(model_type)

    def has(self, architecture: str) -> bool:
        """Check if an architecture is registered."""
        return architecture in self._architectures or architecture in self._model_types

    def list_architectures(self) -> List[str]:
        """List all registered architecture names."""
        return sorted(self._architectures.keys())

    def list_model_types(self) -> List[str]:
        """List all registered model_type strings."""
        return sorted(self._model_types.keys())

    def get_metadata(self, architecture: str) -> Dict[str, Any]:
        """Get metadata for a registered architecture."""
        if architecture in self._metadata:
            return self._metadata[architecture].copy()
        return {}

    def resolve(
        self,
        architecture: Optional[str] = None,
        model_type: Optional[str] = None,
        auto_config: Optional[Any] = None,
    ) -> Type:
        """
        Resolve the adapter class from available information.

        Tries in order:
        1. Explicit architecture string
        2. model_type string
        3. AutoConfig from HuggingFace
        4. Raises ModelNotFoundError
        """
        if architecture and architecture in self._architectures:
            return self._architectures[architecture]

        if model_type and model_type in self._model_types:
            return self._model_types[model_type]

        # Try HuggingFace AutoConfig
        if auto_config is not None:
            arch = getattr(auto_config, "architectures", None)
            if arch and arch[0] in self._architectures:
                return self._architectures[arch[0]]
            mt = getattr(auto_config, "model_type", None)
            if mt and mt in self._model_types:
                return self._model_types[mt]

        # Build helpful error message
        available = self.list_architectures()
        raise ModelNotFoundError(
            architecture or model_type or "unknown"
        )

    def register_decorator(
        self,
        architecture: str,
        model_type: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        **metadata: Any,
    ) -> Callable[[Type], Type]:
        """
        Decorator form of register().

        @MODEL_REGISTRY.register_decorator("LlamaForCausalLM")
        class LlamaModel(ModelAdapter):
            ...
        """

        def decorator(cls: Type) -> Type:
            self.register(
                architecture=architecture,
                implementation=cls,
                model_type=model_type,
                aliases=aliases,
                **metadata,
            )
            return cls

        return decorator

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __len__(self) -> int:
        return len(set(list(self._architectures.keys()) + list(self._model_types.keys())))

    def __repr__(self) -> str:
        archs = self.list_architectures()
        types = self.list_model_types()
        return (
            f"ModelRegistry("
            f"architectures={len(archs)}, "
            f"model_types={len(types)}"
            f")"
        )

    def summary(self) -> str:
        """Return a human-readable summary of the registry."""
        lines = ["Model Registry Summary", "=" * 40]
        for arch in self.list_architectures():
            meta = self._metadata.get(arch, {})
            impl = meta.get("implementation", "?")
            lines.append(f"  {arch} -> {impl}")
        return "\n".join(lines)


# Global registry singleton
MODEL_REGISTRY = ModelRegistry()
