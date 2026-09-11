"""
HuggingFace/safetensors -> .draco converter (spec section 8-10).

Scope of what "verify" actually checks in this implementation: per-tensor
quantization round-trip error (max/mean absolute error, mean relative
error) against the CPU quantization codecs' own dequantize(). Full
model-level logit-agreement verification (running the real model forward
pass and comparing token-by-token) needs the source framework's forward
pass wired in and is not implemented here — don't rely on this verify
step to catch architecture/adapter bugs, only weight-quantization error.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from draco.exceptions import ConversionError
from draco.format.writer import DracoWriter
from draco.quantization.cpu import int4, int8

logger = logging.getLogger("draco.convert")

# Weight tensors we consider quantization candidates: 2D matrices whose name
# suggests a linear/projection layer. Everything else (norms, biases,
# embeddings' 1D pieces) is stored at full precision — this mirrors the
# "weight-only, not everything" guidance in the spec.
_QUANTIZABLE_NAME_HINTS = ("proj", "fc", "linear", "gate", "up", "down", "w1", "w2", "w3")

_QUANT_MAP = {"int8": "int8_sym", "int4": "int4_groupwise", "none": None}


def _is_quantizable(name: str, shape) -> bool:
    if len(shape) != 2:
        return False
    lname = name.lower()
    return any(hint in lname for hint in _QUANTIZABLE_NAME_HINTS)


def convert_model(
    model: str,
    output: str,
    quantization: str = "none",
    verify: bool = True,
    verify_tolerance: float = 0.05,
) -> str:
    if quantization not in _QUANT_MAP:
        raise ConversionError(
            f"Unsupported --quantization '{quantization}'. Supported: {list(_QUANT_MAP)}."
        )
    quant_name = _QUANT_MAP[quantization]

    # GGUF -> .draco: read the llama.cpp file directly (dequantized to
    # float32 by GGUFReader) and write it in Draco's own format.
    from draco.format.gguf import is_gguf_file

    if is_gguf_file(model):
        return _convert_gguf(
            model,
            output,
            quant_name,
            verify=verify,
            verify_tolerance=verify_tolerance,
        )

    try:
        from draco.models.config import ModelConfig
    except ImportError as e:
        raise ConversionError(f"Could not import draco.models.config: {e}") from e

    try:
        config = ModelConfig.from_pretrained(model)
    except Exception as e:
        raise ConversionError(
            f"Could not resolve model config for '{model}': {e}. "
            f"Conversion requires a local path or HF repo with a config.json."
        ) from e

    from draco.weights.loader import CheckpointLoader

    loader = CheckpointLoader()
    try:
        weight_iter = list(loader.iter(model))
    except Exception as e:
        raise ConversionError(f"Could not load weights from '{model}': {e}") from e

    if not weight_iter:
        raise ConversionError(f"No weight tensors found at '{model}'.")

    writer = DracoWriter(output)
    writer.add_metadata(
        {
            "architecture": config.detect_architecture(),
            "model_type": config.model_type,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": config.max_position_embeddings,
            "rope_theta": getattr(config, "rope_theta", None),
        }
    )

    verify_stats = []
    quantized_count = 0
    total_count = 0

    for name, tensor in weight_iter:
        total_count += 1
        arr = tensor.detach().cpu().to(dtype=__import__("torch").float32).numpy()

        use_quant = quant_name is not None and _is_quantizable(name, arr.shape)
        if use_quant:
            codec = int8 if quant_name == "int8_sym" else int4
            if verify:
                stats = codec.roundtrip_error(arr)
                verify_stats.append((name, stats))
            writer.add_tensor(name, arr.astype(np.float32), quantization=quant_name)
            quantized_count += 1
        else:
            writer.add_tensor(name, arr.astype(np.float32))

    if verify and verify_stats:
        bad = [
            (n, s) for n, s in verify_stats if s["mean_rel_error"] > verify_tolerance
        ]
        if bad:
            names = ", ".join(n for n, _ in bad[:5])
            raise ConversionError(
                f"Numerical verification failed: {len(bad)} tensor(s) exceeded "
                f"{verify_tolerance:.1%} mean relative quantization error "
                f"(e.g. {names}). Try a less aggressive --quantization, or pass "
                f"--no-verify to skip this check."
            )
        worst = max(verify_stats, key=lambda t: t[1]["mean_rel_error"])
        logger.info(
            "Verification passed: %d tensors quantized, worst mean_rel_error=%.4f (%s)",
            quantized_count,
            worst[1]["mean_rel_error"],
            worst[0],
        )

    path = writer.finalize()
    logger.info(
        "Wrote %s: %d tensors total, %d quantized (%s)",
        path,
        total_count,
        quantized_count,
        quant_name or "none",
    )
    return path


def _convert_gguf(
    gguf_path: str,
    output: str,
    quant_name: Optional[str],
    verify: bool = True,
    verify_tolerance: float = 0.05,
) -> str:
    """Convert a llama.cpp .gguf file to .draco by dequantizing every
    tensor to float32 (GGUFReader) and optionally re-quantizing the
    linear/projection weights to a Draco CPU format."""
    from draco.format.gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    try:
        metadata = dict(reader.metadata)
        # make sure the executor's required keys survive (they already do:
        # reader.metadata is built for CPUExecutorConfig.from_metadata)
        writer = DracoWriter(output)
        writer.add_metadata(metadata)

        verify_stats = []
        quantized_count = 0
        total_count = 0
        for name in reader.tensor_names():
            total_count += 1
            arr = reader.get_tensor(name).astype(np.float32)
            use_quant = quant_name is not None and _is_quantizable(name, arr.shape)
            if use_quant:
                codec = int8 if quant_name == "int8_sym" else int4
                if verify:
                    stats = codec.roundtrip_error(arr)
                    verify_stats.append((name, stats))
                writer.add_tensor(name, arr, quantization=quant_name)
                quantized_count += 1
            else:
                writer.add_tensor(name, arr)

        if verify and verify_stats:
            bad = [
                (n, s) for n, s in verify_stats if s["mean_rel_error"] > verify_tolerance
            ]
            if bad:
                names = ", ".join(n for n, _ in bad[:5])
                raise ConversionError(
                    f"Numerical verification failed: {len(bad)} tensor(s) exceeded "
                    f"{verify_tolerance:.1%} mean relative quantization error "
                    f"(e.g. {names}). Try a less aggressive --quantization, or pass "
                    f"--no-verify to skip this check."
                )

        path = writer.finalize()
        logger.info(
            "Wrote %s (from GGUF %s): %d tensors total, %d quantized (%s)",
            path,
            gguf_path,
            total_count,
            quantized_count,
            quant_name or "none",
        )
        return path
    finally:
        reader.close()
