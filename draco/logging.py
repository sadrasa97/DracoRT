"""Draco logging system with structured JSON support."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_logger: Optional[logging.Logger] = None

# Context variable for request ID tracking
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_logger(name: str = "draco") -> logging.Logger:
    """Get a Draco logger."""
    global _logger
    if _logger is None:
        _logger = logging.getLogger(name)
        if not _logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
    return _logger.getChild(name) if name != "draco" else _logger


def set_log_level(level: str | int) -> None:
    """Set the log level for Draco loggers."""
    logger = get_logger()
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)


# ======================================================================
# Structured Logging
# ======================================================================

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def __init__(self, include_stack_trace: bool = False):
        super().__init__()
        self.include_stack_trace = include_stack_trace

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        req_id = _request_id.get()
        if req_id:
            log_entry["request_id"] = req_id

        log_entry["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        for key in ("model", "tokens", "latency_ms", "tokens_per_second",
                     "batch_size", "status_code", "error"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }
            if self.include_stack_trace:
                log_entry["exception"]["traceback"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter with color support."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        if self.use_colors and sys.stderr.isatty():
            color = self.COLORS.get(level, "")
            level = f"{color}{level:8s}{self.RESET}"
        else:
            level = f"{level:8s}"

        req_id = _request_id.get()
        prefix = f"[{req_id[:8]}] " if req_id else ""

        msg = f"{prefix}{level} {record.name}: {record.getMessage()}"

        if record.exc_info and record.exc_info[1]:
            msg += f"\n  {type(record.exc_info[1]).__name__}: {record.exc_info[1]}"

        return msg


def setup_structured_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[str] = None,
    include_stack_trace: bool = False,
) -> None:
    """Configure structured logging for Draco."""
    root = logging.getLogger("draco")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(JSONFormatter(include_stack_trace=include_stack_trace))
    else:
        handler.setFormatter(HumanReadableFormatter(use_colors=sys.stderr.isatty()))
    root.addHandler(handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter(include_stack_trace=include_stack_trace))
        root.addHandler(file_handler)


def set_request_id(request_id: Optional[str]) -> None:
    """Set the current request ID for structured logging context."""
    _request_id.set(request_id)


def get_request_id() -> Optional[str]:
    """Get the current request ID."""
    return _request_id.get()


class RequestTimer:
    """Context manager for timing request processing."""

    def __init__(self, logger: Optional[logging.Logger] = None, operation: str = "request"):
        self.logger = logger or logging.getLogger("draco")
        self.operation = operation
        self.start_time: float = 0
        self.end_time: float = 0
        self.metadata: Dict[str, Any] = {}

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        elapsed_ms = (self.end_time - self.start_time) * 1000
        extra: Dict[str, Any] = {"latency_ms": round(elapsed_ms, 2), **self.metadata}

        if exc_type is not None:
            self.logger.error(
                f"{self.operation} failed after {elapsed_ms:.1f}ms: {exc_val}",
                extra=extra,
            )
        else:
            self.logger.info(
                f"{self.operation} completed in {elapsed_ms:.1f}ms",
                extra=extra,
            )

    @property
    def elapsed_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time else 0
