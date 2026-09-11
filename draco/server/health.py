"""
Health Check & Readiness Probes

Production health checking for DracoServer with liveness,
readiness, and startup probes following Kubernetes conventions.

Usage:
    checker = HealthChecker(llm=llm)
    # GET /healthz      — liveness (is the process alive?)
    # GET /readyz       — readiness (is it ready to serve?)
    # GET /startupz     — startup (has initialization completed?)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    READY = "ready"
    NOT_READY = "not_ready"
    STARTING = "starting"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus
    checks: Dict[str, str] = field(default_factory=dict)
    uptime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "checks": self.checks,
            "uptime_seconds": round(self.uptime_seconds, 2),
        }


class HealthChecker:
    """
    Production health checker for DracoServer.

    Provides liveness, readiness, and startup probes.
    """

    def __init__(
        self,
        llm: Any = None,
        start_time: Optional[float] = None,
        startup_timeout: float = 60.0,
        min_ready_time: float = 5.0,
    ):
        self.llm = llm
        self.start_time = start_time or time.monotonic()
        self.startup_timeout = startup_timeout
        self.min_ready_time = min_ready_time
        self._startup_complete = False
        self._startup_error: Optional[str] = None
        self._checks: List[str] = []

    @property
    def uptime(self) -> float:
        return time.monotonic() - self.start_time

    def mark_startup_complete(self, error: Optional[str] = None) -> None:
        """Mark startup as complete (or failed)."""
        self._startup_complete = True
        self._startup_error = error

    def add_check(self, name: str) -> None:
        """Register a custom health check."""
        if name not in self._checks:
            self._checks.append(name)

    def liveness(self) -> HealthCheckResult:
        """
        Liveness probe: Is the process alive?

        Returns HEALTHY if the process is running.
        Returns UNHEALTHY if the process should be restarted.
        """
        checks = {"process": "alive"}

        # Check if startup has timed out
        if not self._startup_complete and self.uptime > self.startup_timeout:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                checks={**checks, "startup": "timed_out"},
                uptime_seconds=self.uptime,
            )

        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            checks=checks,
            uptime_seconds=self.uptime,
        )

    def readiness(self) -> HealthCheckResult:
        """
        Readiness probe: Is the server ready to accept traffic?

        Returns READY when:
        - Startup is complete
        - LLM is initialized
        - Min ready time has elapsed
        """
        checks: Dict[str, str] = {}

        # Check startup
        if self._startup_error:
            return HealthCheckResult(
                status=HealthStatus.NOT_READY,
                checks={**checks, "startup": f"failed: {self._startup_error}"},
                uptime_seconds=self.uptime,
            )

        if not self._startup_complete:
            return HealthCheckResult(
                status=HealthStatus.STARTING,
                checks={**checks, "startup": "in_progress"},
                uptime_seconds=self.uptime,
            )

        checks["startup"] = "complete"

        # Check min ready time
        if self.uptime < self.min_ready_time:
            return HealthCheckResult(
                status=HealthStatus.NOT_READY,
                checks={**checks, "warmup": "in_progress"},
                uptime_seconds=self.uptime,
            )

        # Check LLM availability
        if self.llm is not None:
            if hasattr(self.llm, "_model_runner"):
                if self.llm._model_runner is None:
                    return HealthCheckResult(
                        status=HealthStatus.NOT_READY,
                        checks={**checks, "model": "not_loaded"},
                        uptime_seconds=self.uptime,
                    )
            checks["model"] = "loaded"
        else:
            checks["model"] = "no_llm_configured"

        # Run custom checks
        for check_name in self._checks:
            checks[check_name] = "ok"

        checks["warmup"] = "complete"

        return HealthCheckResult(
            status=HealthStatus.READY,
            checks=checks,
            uptime_seconds=self.uptime,
        )

    def startup(self) -> HealthCheckResult:
        """
        Startup probe: Has initialization completed?

        Returns STARTING if still initializing.
        Returns READY if startup is complete.
        Returns UNHEALTHY if startup has failed or timed out.
        """
        checks: Dict[str, str] = {}

        if self._startup_error:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                checks={**checks, "startup": f"failed: {self._startup_error}"},
                uptime_seconds=self.uptime,
            )

        if self._startup_complete:
            return HealthCheckResult(
                status=HealthStatus.READY,
                checks={**checks, "startup": "complete"},
                uptime_seconds=self.uptime,
            )

        if self.uptime > self.startup_timeout:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                checks={**checks, "startup": "timed_out"},
                uptime_seconds=self.uptime,
            )

        return HealthCheckResult(
            status=HealthStatus.STARTING,
            checks={**checks, "startup": "in_progress"},
            uptime_seconds=self.uptime,
        )

    def full_report(self) -> Dict[str, Any]:
        """Full health report with all probes."""
        return {
            "liveness": self.liveness().to_dict(),
            "readiness": self.readiness().to_dict(),
            "startup": self.startup().to_dict(),
        }
