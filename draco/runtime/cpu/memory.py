"""
CPU memory planning (spec section 17).

Distinguishes *conversion* memory (source checkpoint + quantization
workspace + output file) from *inference* memory (mmap'd weights +
working buffers + KV cache), since the two are not the same and
conflating them is how planners under-provision KV cache or over-report
required RAM.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferenceMemoryEstimate:
    weights_bytes: int  # resident via mmap; not fully duplicated in RAM
    activation_workspace_bytes: int
    kv_cache_bytes: int
    runtime_overhead_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.activation_workspace_bytes
            + self.kv_cache_bytes
            + self.runtime_overhead_bytes
        )


@dataclass
class ConversionMemoryEstimate:
    source_checkpoint_bytes: int
    quantization_workspace_bytes: int
    output_file_bytes: int

    @property
    def peak_bytes(self) -> int:
        # Source + workspace must coexist; output is written incrementally
        # and is not required to be fully buffered in RAM by the writer.
        return self.source_checkpoint_bytes + self.quantization_workspace_bytes


_BYTES_PER_DTYPE = {"f32": 4, "f16": 2, "bf16": 2, "int8_sym": 1, "int4_groupwise": 0.5}


class CPUMemoryPlanner:
    def estimate_inference(
        self,
        num_params: int,
        quantization: str,
        num_layers: int,
        hidden_size: int,
        num_key_value_heads: int,
        head_dim: int,
        max_model_len: int,
        max_num_seqs: int,
        kv_cache_dtype: str = "f16",
        activation_workspace_multiplier: float = 2.0,
    ) -> InferenceMemoryEstimate:
        bytes_per_param = _BYTES_PER_DTYPE.get(quantization, 4)
        weights_bytes = int(num_params * bytes_per_param)

        # One row of activations per token in flight, generously sized as a
        # multiple of the model's largest per-layer intermediate (hidden
        # size is a reasonable, conservative proxy without access to the
        # real intermediate_size here).
        activation_workspace_bytes = int(
            hidden_size * 4 * activation_workspace_multiplier * max_num_seqs
        )

        kv_bytes_per_token = (
            2  # K and V
            * num_layers
            * num_key_value_heads
            * head_dim
            * _BYTES_PER_DTYPE.get(kv_cache_dtype, 2)
        )
        kv_cache_bytes = int(kv_bytes_per_token * max_model_len * max_num_seqs)

        runtime_overhead_bytes = 512 * 1024 * 1024  # tokenizer, scheduler, misc buffers

        return InferenceMemoryEstimate(
            weights_bytes=weights_bytes,
            activation_workspace_bytes=activation_workspace_bytes,
            kv_cache_bytes=kv_cache_bytes,
            runtime_overhead_bytes=runtime_overhead_bytes,
        )

    def estimate_conversion(
        self, num_params: int, source_dtype: str = "f32", target_quantization: str = "int8_sym"
    ) -> ConversionMemoryEstimate:
        source_bytes = int(num_params * _BYTES_PER_DTYPE.get(source_dtype, 4))
        # Quantization workspace: one full-precision copy of the tensor being
        # quantized at a time, not the whole model — conversion here is
        # tensor-at-a-time (see draco.convert.converter), so this is
        # deliberately not num_params-scaled.
        workspace_bytes = int(256 * 1024 * 1024)
        output_bytes = int(num_params * _BYTES_PER_DTYPE.get(target_quantization, 4))
        return ConversionMemoryEstimate(
            source_checkpoint_bytes=source_bytes,
            quantization_workspace_bytes=workspace_bytes,
            output_file_bytes=output_bytes,
        )
