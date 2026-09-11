"""
Quantization Registry

Central registry for quantization backends. Each backend is registered
by name and can be looked up for quantization operations.

Example:
    QUANT_REGISTRY.register("gptq", GPTQBackend())
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from draco.quantization.base import QuantizationBackend

logger = logging.getLogger("draco.quantization.registry")


class QuantizationRegistry:
    """
    Central registry for quantization backends.

    Maps quantization method names to backend implementations.
    """

    def __init__(self) -> None:
        self._backends: Dict[str, QuantizationBackend] = {}

    def register(self, name: str, backend: QuantizationBackend) -> None:
        """Register a quantization backend."""
        name = name.lower()
        if name in self._backends:
            logger.warning(
                "Re-registering quantization backend '%s': %s -> %s",
                name,
                type(self._backends[name]).__name__,
                type(backend).__name__,
            )
        self._backends[name] = backend
        logger.debug("Registered quantization backend '%s'", name)

    def get(self, name: str) -> QuantizationBackend:
        """Get a quantization backend by name."""
        name = name.lower()
        if name not in self._backends:
            raise ValueError(
                f"Unknown quantization method: {name!r}. "
                f"Available: {self.list_methods()}"
            )
        return self._backends[name]

    def has(self, name: str) -> bool:
        """Check if a quantization method is registered."""
        return name.lower() in self._backends

    def list_methods(self) -> List[str]:
        """List all registered quantization methods."""
        return sorted(self._backends.keys())

    def list_backends(self) -> Dict[str, QuantizationBackend]:
        """Return all registered backends."""
        return dict(self._backends)

    def detect_from_config(self, quantization_config: Dict[str, Any]) -> Optional[str]:
        """
        Auto-detect quantization method from config.json metadata.
        Returns the method name or None if not detected.
        """
        method = quantization_config.get(
            "quant_method", quantization_config.get("quantization_method")
        )
        if method and method.lower() in self._backends:
            return method.lower()
        return None

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __len__(self) -> int:
        return len(self._backends)

    def __repr__(self) -> str:
        methods = self.list_methods()
        return f"QuantizationRegistry(methods={methods})"


# Global registry singleton
QUANT_REGISTRY = QuantizationRegistry()
