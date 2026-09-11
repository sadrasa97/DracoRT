"""
ONNX model support for Draco's native backend (backend="draco").

Unlike the .safetensors/.draco path (draco/models/*/adapter.py — hand-written
PyTorch re-implementations of each architecture) and the .gguf path
(draco/format/gguf.py + draco/runtime/cpu — a pure-numpy execution graph),
an ONNX model is already a fully-specified, architecture-agnostic
computation graph exported by someone else (typically via HF Optimum:
`optimum-cli export onnx ...`). There is nothing to "adapt" per
architecture here — ONNXModelReader just needs to locate the right
`.onnx` file, start an onnxruntime.InferenceSession, and know which
input/output names to feed and read, which is what this module does.

Supports the standard HF Optimum decoder export conventions:
  - stateless (no KV cache): inputs `input_ids` (+ `attention_mask`),
    output `logits`. Every step re-feeds the full sequence — same
    correctness/perf tradeoff as `_generate_single_full_refeed`.
  - decoder-with-past (KV cache): inputs `input_ids`, `attention_mask`,
    `past_key_values.{i}.key` / `past_key_values.{i}.value` per layer;
    outputs `logits`, `present.{i}.key` / `present.{i}.value` per layer.
    Detected automatically from the session's own declared input names —
    no config flag needed.
  - decoder-with-past "merged" export (a single graph that handles both
    the first, cache-empty call and subsequent cached calls via an extra
    `use_cache_branch` boolean input) is also detected and fed correctly.
"""
from __future__ import annotations

import glob
import logging
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("draco.format.onnx")


def find_onnx_model_file(path: str) -> Optional[str]:
    """Locate the .onnx file to load for `path`, which may be a single
    .onnx file or a directory (an Optimum export directory typically
    contains `model.onnx` or `decoder_model_merged.onnx` alongside
    `config.json` and external-data `.onnx_data` weight files)."""
    if os.path.isfile(path) and path.endswith(".onnx"):
        return path
    if not os.path.isdir(path):
        return None
    # Prefer a merged decoder-with-past export (handles prefill+decode in
    # one graph) over a plain/no-cache one, if both are present.
    preferred = [
        "decoder_model_merged.onnx",
        "decoder_with_past_model.onnx",
        "decoder_model.onnx",
        "model.onnx",
    ]
    for name in preferred:
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            return candidate
    matches = glob.glob(os.path.join(path, "*.onnx"))
    return matches[0] if matches else None


def is_onnx_model(path: str) -> bool:
    return find_onnx_model_file(path) is not None


class ONNXModelReader:
    """Wraps an onnxruntime.InferenceSession with the same
    `forward(input_ids, attention_mask=None, past_key_values=None,
    use_cache=False) -> logits | (logits, present_key_values)` contract
    every draco/models/*/adapter.py Model class exposes, so
    LLM._generate_single_onnx can reuse the same prefill-then-decode
    shape as the native-adapter KV cache path.
    """

    def __init__(self, model_path: str, device: torch.device):
        onnx_file = find_onnx_model_file(model_path)
        if onnx_file is None:
            raise FileNotFoundError(f"No .onnx model file found at '{model_path}'")
        self.onnx_file = onnx_file

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "ONNX model support requires the 'onnxruntime' package. "
                "Install it with `pip install onnxruntime` (CPU) or "
                "`pip install onnxruntime-gpu` (CUDA)."
            ) from e

        providers = ["CPUExecutionProvider"]
        if device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.device = device
        self.session = ort.InferenceSession(onnx_file, providers=providers)

        input_names = {i.name for i in self.session.get_inputs()}
        output_names = {o.name for o in self.session.get_outputs()}
        self.input_names = input_names
        self.output_names = output_names

        self.uses_cache = any(n.startswith("past_key_values.") for n in input_names)
        self.uses_merged_branch = "use_cache_branch" in input_names
        self.has_attention_mask = "attention_mask" in input_names
        self.has_position_ids = "position_ids" in input_names

        # Figure out num_layers / num_kv_heads / head_dim from the
        # declared past_key_values.N.key input shapes, so the caller
        # doesn't need a separate config lookup that may not match
        # (ONNX export shapes are the ground truth for this graph).
        self.num_layers = 0
        self.num_kv_heads: Optional[int] = None
        self.head_dim: Optional[int] = None
        if self.uses_cache:
            layer_indices = set()
            for name in input_names:
                if name.startswith("past_key_values.") and name.endswith(".key"):
                    layer_indices.add(int(name.split(".")[1]))
            self.num_layers = (max(layer_indices) + 1) if layer_indices else 0
            key0 = next(
                i for i in self.session.get_inputs() if i.name == "past_key_values.0.key"
            )
            # Shape is typically [batch, num_kv_heads, past_seq_len, head_dim];
            # dims may be dynamic (strings) except num_kv_heads/head_dim.
            shape = key0.shape
            self.num_kv_heads = shape[1] if isinstance(shape[1], int) else None
            self.head_dim = shape[3] if isinstance(shape[3], int) else None

        logger.info(
            "ONNX model loaded: %s (uses_cache=%s, merged_branch=%s, num_layers=%d, providers=%s)",
            onnx_file, self.uses_cache, self.uses_merged_branch, self.num_layers,
            self.session.get_providers(),
        )

    def _empty_past(self, batch_size: int) -> dict:
        """Zero-length past_key_values inputs for the first (prefill) call
        of a decoder-with-past graph — needed even though there's no real
        past yet, since the graph declares these as required inputs."""
        feed = {}
        kv_heads = self.num_kv_heads or 1
        hd = self.head_dim or 1
        for i in range(self.num_layers):
            shape = (batch_size, kv_heads, 0, hd)
            empty = np.zeros(shape, dtype=np.float32)
            feed[f"past_key_values.{i}.key"] = empty
            feed[f"past_key_values.{i}.value"] = empty
        return feed

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ):
        bsz, seq_len = input_ids.shape
        feed = {"input_ids": input_ids.cpu().numpy().astype(np.int64)}

        if self.has_attention_mask:
            if attention_mask is None:
                past_len = (
                    past_key_values[0][0].shape[2] if past_key_values is not None else 0
                )
                mask = np.ones((bsz, past_len + seq_len), dtype=np.int64)
            else:
                # Accept either an additive float mask (our convention
                # elsewhere) or a plain 0/1 mask; ONNX decoder exports
                # expect the plain 0/1 form.
                mask_np = attention_mask.cpu().numpy()
                mask = (mask_np > 0).astype(np.int64) if mask_np.dtype != np.int64 else mask_np
            feed["attention_mask"] = mask

        if self.has_position_ids:
            if position_ids is None:
                past_len = (
                    past_key_values[0][0].shape[2] if past_key_values is not None else 0
                )
                position_ids = torch.arange(past_len, past_len + seq_len).unsqueeze(0).expand(bsz, -1)
            feed["position_ids"] = position_ids.cpu().numpy().astype(np.int64)

        if self.uses_cache:
            if past_key_values is None:
                feed.update(self._empty_past(bsz))
                if self.uses_merged_branch:
                    feed["use_cache_branch"] = np.array([False])
            else:
                for i, (k, v) in enumerate(past_key_values):
                    feed[f"past_key_values.{i}.key"] = k.cpu().numpy().astype(np.float32)
                    feed[f"past_key_values.{i}.value"] = v.cpu().numpy().astype(np.float32)
                if self.uses_merged_branch:
                    feed["use_cache_branch"] = np.array([True])

        outputs = self.session.run(None, feed)
        out_by_name = dict(zip([o.name for o in self.session.get_outputs()], outputs))
        logits = torch.from_numpy(out_by_name["logits"]).to(self.device)

        if not use_cache or not self.uses_cache:
            return logits

        present_key_values = []
        for i in range(self.num_layers):
            k = torch.from_numpy(out_by_name[f"present.{i}.key"]).to(self.device)
            v = torch.from_numpy(out_by_name[f"present.{i}.value"]).to(self.device)
            present_key_values.append((k, v))
        return logits, present_key_values
