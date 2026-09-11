"""
Weight Loading Abstraction

Unified weight loading layer supporting:
- .safetensors
- .pth / .pt / .bin (PyTorch checkpoints)
- Hugging Face repositories
- Local checkpoints
- Direct GPU loading (future)
- Sharded checkpoints (future)
- Memory-mapped loading (future)
- Streaming weight loading (future)
- Tensor-parallel loading (future)
- Quantized weight loading (future)

Avoids loading the entire model through Python-level abstraction when
this causes unnecessary CPU memory usage.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch

from draco.exceptions import CheckpointError, WeightLoadingError

logger = logging.getLogger("draco.weights")


class WeightLoader(ABC):
    """
    Abstract base class for weight loaders.

    All weight loaders provide a consistent interface for loading model
    weights from various checkpoint formats.
    """

    @abstractmethod
    def can_load(self, path: str) -> bool:
        """Check if this loader can handle the given path."""
        ...

    @abstractmethod
    def load_weights(
        self,
        path: str,
        target: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Load weights from path.

        Args:
            path: Path to checkpoint file or directory
            target: Optional dict to load weights into
            **kwargs: Additional loader-specific arguments

        Returns:
            Dictionary mapping weight names to tensors.
        """
        ...

    @abstractmethod
    def iter_weights(
        self,
        path: str,
        **kwargs: Any,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Iterate over weights without loading all at once.

        This is the memory-efficient path for large models.
        """
        ...

    def get_weight_info(self, path: str) -> Dict[str, Any]:
        """Get metadata about weights without loading them."""
        return {}

    def map_weights(
        self,
        weights: Dict[str, torch.Tensor],
        weight_map: Dict[str, str],
    ) -> Dict[str, torch.Tensor]:
        """
        Remap weight names using a mapping dictionary.

        Args:
            weights: Original weight dictionary
            weight_map: {original_name: target_name}

        Returns:
            Remapped weight dictionary.
        """
        result = {}
        for original_name, tensor in weights.items():
            if original_name in weight_map:
                result[weight_map[original_name]] = tensor
            else:
                # Try partial match
                for pattern, target_name in weight_map.items():
                    if original_name.endswith(pattern) or pattern in original_name:
                        result[target_name] = tensor
                        break
                else:
                    # Keep unmapped weights with original name
                    result[original_name] = tensor
        return result


class SafetensorsLoader(WeightLoader):
    """Load weights from .safetensors files."""

    def can_load(self, path: str) -> bool:
        p = Path(path)
        if p.is_file():
            return p.suffix == ".safetensors"
        if p.is_dir():
            safetensors_files = list(p.glob("*.safetensors"))
            return len(safetensors_files) > 0
        return False

    def load_weights(
        self,
        path: str,
        target: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Load weights from safetensors file(s)."""
        try:
            from safetensors import safe_open
            from safetensors.torch import load_file
        except ImportError:
            raise WeightLoadingError(
                "safetensors package is required. Install with: pip install safetensors"
            )

        p = Path(path)
        weights: Dict[str, torch.Tensor] = {}

        if p.is_file():
            logger.info("Loading safetensors: %s", p)
            tensors = load_file(str(p), device=str(device) if device else "cpu")
            weights.update(tensors)
        elif p.is_dir():
            safetensors_files = sorted(p.glob("*.safetensors"))
            for sf in safetensors_files:
                logger.info("Loading safetensors shard: %s", sf)
                tensors = load_file(str(sf), device=str(device) if device else "cpu")
                weights.update(tensors)
        else:
            raise CheckpointError(f"Path not found: {path}")

        logger.info("Loaded %d tensors from safetensors", len(weights))
        return weights

    def iter_weights(
        self,
        path: str,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Iterate over safetensors weights one tensor at a time."""
        try:
            from safetensors import safe_open
        except ImportError:
            raise WeightLoadingError(
                "safetensors package is required. Install with: pip install safetensors"
            )

        p = Path(path)
        if p.is_file():
            with safe_open(str(p), framework="pt", device=str(device) if device else "cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)
        elif p.is_dir():
            for sf in sorted(p.glob("*.safetensors")):
                with safe_open(str(sf), framework="pt", device=str(device) if device else "cpu") as f:
                    for key in f.keys():
                        yield key, f.get_tensor(key)

    def get_weight_info(self, path: str) -> Dict[str, Any]:
        """Get metadata about safetensors weights."""
        try:
            from safetensors import safe_open
        except ImportError:
            return {}

        p = Path(path)
        info = {"format": "safetensors", "files": [], "total_tensors": 0}

        files = [p] if p.is_file() else sorted(p.glob("*.safetensors"))
        for f_path in files:
            with safe_open(str(f_path), framework="pt") as f:
                keys = f.keys()
                info["files"].append({
                    "path": str(f_path),
                    "tensors": len(keys),
                    "keys": list(keys),
                })
                info["total_tensors"] += len(keys)

        return info


class PyTorchLoader(WeightLoader):
    """Load weights from PyTorch checkpoint files (.pth, .pt, .bin)."""

    EXTENSIONS = {".pth", ".pt", ".bin"}

    def can_load(self, path: str) -> bool:
        p = Path(path)
        if p.is_file():
            return p.suffix in self.EXTENSIONS
        if p.is_dir():
            for ext in self.EXTENSIONS:
                if list(p.glob(f"*{ext}")):
                    return True
        return False

    def load_weights(
        self,
        path: str,
        target: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Load weights from PyTorch checkpoint."""
        p = Path(path)
        weights: Dict[str, torch.Tensor] = {}
        map_location = str(device) if device else "cpu"

        if p.is_file():
            logger.info("Loading PyTorch checkpoint: %s", p)
            checkpoint = torch.load(str(p), map_location=map_location, weights_only=True)
            if isinstance(checkpoint, dict):
                # Could be state_dict or nested
                if "state_dict" in checkpoint:
                    weights.update(checkpoint["state_dict"])
                elif "model" in checkpoint:
                    weights.update(checkpoint["model"])
                else:
                    weights.update(checkpoint)
            else:
                raise WeightLoadingError(f"Unexpected checkpoint format: {type(checkpoint)}")
        elif p.is_dir():
            files = []
            for ext in self.EXTENSIONS:
                files.extend(sorted(p.glob(f"*{ext}")))
            for f_path in files:
                logger.info("Loading PyTorch shard: %s", f_path)
                checkpoint = torch.load(str(f_path), map_location=map_location, weights_only=True)
                if isinstance(checkpoint, dict):
                    if "state_dict" in checkpoint:
                        weights.update(checkpoint["state_dict"])
                    else:
                        weights.update(checkpoint)
        else:
            raise CheckpointError(f"Path not found: {path}")

        logger.info("Loaded %d tensors from PyTorch checkpoint", len(weights))
        return weights

    def iter_weights(
        self,
        path: str,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Iterate over PyTorch checkpoint weights."""
        p = Path(path)
        map_location = str(device) if device else "cpu"

        files = [p] if p.is_file() else sorted(
            [f for ext in self.EXTENSIONS for f in p.glob(f"*{ext}")]
        )

        for f_path in files:
            checkpoint = torch.load(str(f_path), map_location=map_location, weights_only=True)
            if isinstance(checkpoint, dict):
                sd = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
                for key, tensor in sd.items():
                    yield key, tensor


class HFLoader(WeightLoader):
    """
    Load weights from HuggingFace model repositories.

    Downloads weights from the Hub if not cached locally.
    Supports sharded and non-sharded checkpoints.
    """

    def can_load(self, path: str) -> bool:
        # If it looks like a HF repo ID (no / at start, has org/model format)
        if "/" in path and not Path(path).exists():
            return True
        # Or if it has HF-specific files
        p = Path(path)
        if p.is_dir():
            return (p / "config.json").exists() or (p / "model.safetensors").exists()
        return False

    def load_weights(
        self,
        path: str,
        target: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Load weights from HuggingFace repository."""
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError:
            raise WeightLoadingError(
                "huggingface-hub is required. Install with: pip install huggingface-hub"
            )

        p = Path(path)
        if p.is_dir():
            # Local directory — find checkpoint files
            weights = self._load_from_directory(p, device)
        else:
            # Remote repo — download first
            model_id = path
            cache_dir = kwargs.get("cache_dir")
            revision = kwargs.get("revision")

            local_dir = snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                revision=revision,
            )
            weights = self._load_from_directory(Path(local_dir), device)

        return weights

    def _load_from_directory(
        self,
        directory: Path,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Load weights from a local directory."""
        safetensors_files = list(directory.glob("*.safetensors"))
        if safetensors_files:
            loader = SafetensorsLoader()
            return loader.load_weights(str(directory), device=device)

        pytorch_files = []
        for ext in (".bin", ".pth", ".pt"):
            pytorch_files.extend(directory.glob(f"*{ext}"))
        if pytorch_files:
            loader = PyTorchLoader()
            return loader.load_weights(str(directory), device=device)

        raise CheckpointError(f"No checkpoint files found in {directory}")

    def iter_weights(
        self,
        path: str,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Iterate over HF weights."""
        p = Path(path)
        if p.is_dir():
            safetensors_files = list(p.glob("*.safetensors"))
            if safetensors_files:
                loader = SafetensorsLoader()
                yield from loader.iter_weights(str(p), device=device)
            else:
                loader = PyTorchLoader()
                yield from loader.iter_weights(str(p), device=device)
        else:
            raise CheckpointError(f"Not a directory: {path}")


class CheckpointLoader:
    """
    Unified checkpoint loader that auto-detects format and delegates
    to the appropriate WeightLoader.
    """

    def __init__(self) -> None:
        self._loaders: List[WeightLoader] = [
            SafetensorsLoader(),
            PyTorchLoader(),
            HFLoader(),
        ]

    def register_loader(self, loader: WeightLoader, priority: int = -1) -> None:
        """Register a custom weight loader."""
        if priority < 0:
            self._loaders.insert(0, loader)
        else:
            self._loaders.append(loader)

    def load(
        self,
        path: str,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Auto-detect format and load weights."""
        for loader in self._loaders:
            if loader.can_load(path):
                logger.info("Using %s for %s", type(loader).__name__, path)
                return loader.load_weights(path, device=device, **kwargs)

        raise CheckpointError(
            f"No suitable weight loader found for '{path}'. "
            f"Supported formats: .safetensors, .pth, .pt, .bin, HuggingFace repos"
        )

    def iter(
        self,
        path: str,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Iterate over weights without loading all at once."""
        for loader in self._loaders:
            if loader.can_load(path):
                yield from loader.iter_weights(path, device=device, **kwargs)
                return
        raise CheckpointError(f"No suitable weight loader found for '{path}'")

    def get_info(self, path: str) -> Dict[str, Any]:
        """Get checkpoint information."""
        for loader in self._loaders:
            if loader.can_load(path):
                return loader.get_weight_info(path)
        return {"path": path, "status": "unknown"}

    def detect_format(self, path: str) -> str:
        """Detect the checkpoint format."""
        for loader in self._loaders:
            if loader.can_load(path):
                return type(loader).__name__
        return "unknown"


# Global checkpoint loader
_CHECKPOINT_LOADER = CheckpointLoader()


def get_checkpoint_loader() -> CheckpointLoader:
    """Get the global checkpoint loader."""
    return _CHECKPOINT_LOADER
