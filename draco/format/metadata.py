"""Model/architecture metadata helpers for .draco files."""

from __future__ import annotations

import json
from typing import Any, Dict

REQUIRED_KEYS = ("architecture", "model_type")


def validate_metadata(metadata: Dict[str, Any]) -> None:
    from draco.exceptions import DracoFormatError

    missing = [k for k in REQUIRED_KEYS if k not in metadata]
    if missing:
        raise DracoFormatError(
            f"Model metadata is missing required key(s): {missing}. "
            f"At minimum, 'architecture' and 'model_type' must be present "
            f"so the runtime can resolve a ModelAdapter via MODEL_REGISTRY."
        )


def encode(metadata: Dict[str, Any]) -> bytes:
    return json.dumps(metadata, sort_keys=True).encode("utf-8")


def decode(data: bytes) -> Dict[str, Any]:
    return json.loads(data.decode("utf-8"))
