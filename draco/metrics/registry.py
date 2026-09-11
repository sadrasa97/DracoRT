"""
Metrics Registry

Allows registration of custom metric collectors and exporters.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("draco.metrics.registry")


class MetricsRegistry:
    """
    Registry for custom metric collectors and exporters.

    Allows users to add custom metric collection and export.
    """

    def __init__(self) -> None:
        self._collectors: Dict[str, Callable] = {}
        self._exporters: Dict[str, Callable] = {}

    def register_collector(
        self, name: str, collector: Callable[..., Any]
    ) -> None:
        """Register a custom metric collector."""
        self._collectors[name] = collector
        logger.debug("Registered metric collector '%s'", name)

    def register_exporter(
        self, name: str, exporter: Callable[..., Any]
    ) -> None:
        """Register a custom metric exporter (e.g., Prometheus)."""
        self._exporters[name] = exporter
        logger.debug("Registered metric exporter '%s'", name)

    def get_collector(self, name: str) -> Optional[Callable]:
        return self._collectors.get(name)

    def get_exporter(self, name: str) -> Optional[Callable]:
        return self._exporters.get(name)

    def list_collectors(self) -> List[str]:
        return sorted(self._collectors.keys())

    def list_exporters(self) -> List[str]:
        return sorted(self._exporters.keys())


# Global registry singleton
METRICS_REGISTRY = MetricsRegistry()
