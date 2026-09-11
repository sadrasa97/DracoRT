"""
Native CPU execution graph (spec sections 37-39), generalized across
architecture families rather than hardcoded to one.

Supported axes (read from .draco metadata, populated at conversion time
from the resolved MODEL_REGISTRY adapter/config — see spec section 26 —
so this module doesn't reinvent architecture detection, just consumes it):

  norm_type:              "rmsnorm" (Llama/Qwen2/Mistral/...) | "layernorm" (GPT-2/Falcon/BLOOM/...)
  position_encoding_type: "rope" | "alibi" (BLOOM) | "learned" (GPT-2)
  mlp_type:               "gated" (SwiGLU/GeGLU: gate+up+down) | "standard" (fc1+fc2, single activation)
  hidden_act:             "silu" | "gelu"
  qkv_fused:              True for GPT-2-style single c_attn projection, False for separate q/k/v
  attention_bias / mlp_bias / norm_bias: whether those layers carry a bias tensor

This does not claim to correctly run every one of the repo's 13 model
adapters out of the box — attention variants like Falcon's parallel
attention+MLP or MQA/multi-query sharing aren't wired in here — but it
covers the two largest architecture families (RMSNorm+RoPE+gated-MLP,
and LayerNorm+learned-position+standard-MLP with fused QKV), each with
an independent-reference-implementation correctness test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from draco.exceptions import CPUError
from draco.format.reader import DracoReader
from draco.runtime.cpu.attention import CPUAttentionBackend
from draco.runtime.cpu.dispatcher import KernelSelection, select_kernel
from draco.runtime.cpu.kv_cache import CPUKVCache, CPUKVCacheConfig, budget_to_max_blocks
from draco.runtime.cpu.ops import (
    apply_rope,
    build_alibi_bias,
    build_rope_cache,
    gelu,
    layer_norm,
    rms_norm,
    swiglu,
)


def _linear(kernel_selection: KernelSelection, reader: DracoReader, name: str, x: np.ndarray, bias_name: Optional[str] = None) -> np.ndarray:
    info = reader.tensor_info(name)
    if info.quantization is None:
        weight = reader.get_tensor(name, dequantize=True)
        out = kernel_selection.kernel.matmul_f32(x, weight)
    else:
        payload = reader.get_tensor(name, dequantize=False)
        kwargs = {}
        if info.quantization == "int8_sym":
            kwargs["scale"] = np.frombuffer(
                reader._raw_bytes(info.scale_offset, info.scale_nbytes), dtype=np.float32
            )
        elif info.quantization == "int8_asym":
            kwargs["scale"] = np.frombuffer(
                reader._raw_bytes(info.scale_offset, info.scale_nbytes), dtype=np.float32
            )
            kwargs["zero_point"] = np.frombuffer(
                reader._raw_bytes(info.zero_point_offset, info.zero_point_nbytes), dtype=np.float32
            )
        elif info.quantization == "int4_groupwise":
            rows, cols = info.shape
            n_groups = cols // info.group_size
            kwargs["scale"] = np.frombuffer(
                reader._raw_bytes(info.scale_offset, info.scale_nbytes), dtype=np.float32
            ).reshape(rows, n_groups)
            kwargs["zero_point"] = np.frombuffer(
                reader._raw_bytes(info.zero_point_offset, info.zero_point_nbytes), dtype=np.float32
            ).reshape(rows, n_groups)
        out = kernel_selection.kernel.matmul(x, info, payload.tobytes(), **kwargs)

    if bias_name is not None and bias_name in reader._index:
        out = out + reader.get_tensor(bias_name, dequantize=True)
    return out


@dataclass
class CPUExecutorConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    layer_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

    norm_type: str = "rmsnorm"  # "rmsnorm" | "layernorm"
    norm_bias: bool = False
    position_encoding_type: str = "rope"  # "rope" | "alibi" | "learned"
    mlp_type: str = "gated"  # "gated" | "standard"
    hidden_act: str = "silu"  # "silu" | "gelu"
    qkv_fused: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_metadata(cls, metadata: dict) -> "CPUExecutorConfig":
        return cls(
            hidden_size=metadata["hidden_size"],
            num_hidden_layers=metadata["num_hidden_layers"],
            num_attention_heads=metadata["num_attention_heads"],
            num_key_value_heads=metadata.get("num_key_value_heads", metadata["num_attention_heads"]),
            vocab_size=metadata["vocab_size"],
            rope_theta=metadata.get("rope_theta") or 10000.0,
            max_position_embeddings=metadata.get("max_position_embeddings", 4096),
            rms_norm_eps=metadata.get("rms_norm_eps", 1e-6),
            layer_norm_eps=metadata.get("layer_norm_eps", 1e-5),
            tie_word_embeddings=metadata.get("tie_word_embeddings", False),
            norm_type=metadata.get("norm_type", "rmsnorm"),
            norm_bias=metadata.get("norm_bias", False),
            position_encoding_type=metadata.get("position_encoding_type", "rope"),
            mlp_type=metadata.get("mlp_type", "gated"),
            hidden_act=metadata.get("hidden_act", "silu"),
            qkv_fused=metadata.get("qkv_fused", False),
            attention_bias=metadata.get("attention_bias", False),
            mlp_bias=metadata.get("mlp_bias", False),
        )


class CPUExecutionBackend:
    def __init__(
        self,
        reader: DracoReader,
        max_num_seqs: int = 1,
        kv_cache_dtype: str = "f32",
        kv_budget_bytes: Optional[int] = None,
        memory_utilization: Optional[float] = None,
    ) -> None:
        self.reader = reader
        self.config = CPUExecutorConfig.from_metadata(reader.metadata)
        self.kernel_selection = select_kernel()

        if kv_budget_bytes is None:
            from draco.runtime.cpu.mem_util import DEFAULT_CPU_MEMORY_UTILIZATION, auto_memory_budget

            util = memory_utilization if memory_utilization is not None else DEFAULT_CPU_MEMORY_UTILIZATION
            total_budget = auto_memory_budget(util)
            kv_budget_bytes = max(64 * 1024 * 1024, int(total_budget * 0.5))
        self.kv_budget_bytes = kv_budget_bytes

        kv_config = CPUKVCacheConfig(
            num_layers=self.config.num_hidden_layers,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            dtype=kv_cache_dtype,
        )
        kv_config.max_blocks = max(1, budget_to_max_blocks(kv_config, kv_budget_bytes))
        self.kv_cache = CPUKVCache(kv_config)
        self.attention = CPUAttentionBackend(self.kv_cache, self.config.num_attention_heads)

        if self.config.position_encoding_type == "rope":
            self._cos, self._sin = build_rope_cache(
                self.config.head_dim, self.config.max_position_embeddings, self.config.rope_theta
            )
        else:
            self._cos = self._sin = None
        self._next_seq_id = 0

    def new_sequence(self) -> int:
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        self.kv_cache.allocate_sequence(seq_id)
        return seq_id

    def end_sequence(self, seq_id: int) -> None:
        self.kv_cache.free_sequence(seq_id)

    def truncate_sequence(self, seq_id: int, new_length: int) -> None:
        self.kv_cache.truncate_sequence(seq_id, new_length)

    def _weight_name(self, *parts: str) -> str:
        candidates = [".".join(parts), ".".join(["model"] + list(parts)), ".".join(["transformer"] + list(parts))]
        for name in candidates:
            if name in self.reader._index:
                return name
        raise CPUError(
            f"Could not find weight tensor for {parts} — tried {candidates}."
        )

    def _has_weight(self, *parts: str) -> Optional[str]:
        for prefix in ("", "model.", "transformer."):
            name = prefix + ".".join(parts)
            if name in self.reader._index:
                return name
        return None

    def _norm(self, x: np.ndarray, weight_name: str) -> np.ndarray:
        cfg = self.config
        weight = self.reader.get_tensor(weight_name)
        if cfg.norm_type == "layernorm":
            bias_name = weight_name.replace(".weight", ".bias")
            bias = self.reader.get_tensor(bias_name) if cfg.norm_bias and bias_name in self.reader._index else None
            return layer_norm(x, weight, bias, cfg.layer_norm_eps)
        return rms_norm(x, weight, cfg.rms_norm_eps)

    def _mlp(self, x: np.ndarray, prefix: str) -> np.ndarray:
        cfg = self.config
        bias = cfg.mlp_bias
        if cfg.mlp_type == "gated":
            gate = _linear(
                self.kernel_selection, self.reader, self._weight_name(prefix, "mlp.gate_proj.weight"), x,
                bias_name=self._has_weight(prefix, "mlp.gate_proj.bias") if bias else None,
            )
            up = _linear(
                self.kernel_selection, self.reader, self._weight_name(prefix, "mlp.up_proj.weight"), x,
                bias_name=self._has_weight(prefix, "mlp.up_proj.bias") if bias else None,
            )
            hidden = swiglu(gate, up)
            return _linear(
                self.kernel_selection, self.reader, self._weight_name(prefix, "mlp.down_proj.weight"), hidden,
                bias_name=self._has_weight(prefix, "mlp.down_proj.bias") if bias else None,
            )
        else:
            fc1_name = self._has_weight(prefix, "mlp.fc1.weight") or self._weight_name(prefix, "mlp.c_fc.weight")
            fc2_name = self._has_weight(prefix, "mlp.fc2.weight") or self._weight_name(prefix, "mlp.c_proj.weight")
            hidden = _linear(
                self.kernel_selection, self.reader, fc1_name, x,
                bias_name=self._has_weight(fc1_name.rsplit(".weight", 1)[0] + ".bias") if bias else None,
            )
            if cfg.hidden_act == "gelu":
                act = gelu(hidden)
            else:
                from draco.runtime.cpu.ops import silu

                act = silu(hidden)
            return _linear(
                self.kernel_selection, self.reader, fc2_name, act,
                bias_name=self._has_weight(fc2_name.rsplit(".weight", 1)[0] + ".bias") if bias else None,
            )

    def _qkv(self, normed: np.ndarray, prefix: str, n: int):
        cfg = self.config
        bias = cfg.attention_bias
        if cfg.qkv_fused:
            fused_name = self._has_weight(prefix, "attn.c_attn.weight") or self._weight_name(prefix, "self_attn.qkv_proj.weight")
            fused_bias = self._has_weight(prefix, "attn.c_attn.bias") if bias else None
            qkv = _linear(self.kernel_selection, self.reader, fused_name, normed, bias_name=fused_bias)
            hd = cfg.head_dim
            q_dim = cfg.num_attention_heads * hd
            kv_dim = cfg.num_key_value_heads * hd
            q, k, v = qkv[:, :q_dim], qkv[:, q_dim : q_dim + kv_dim], qkv[:, q_dim + kv_dim :]
        else:
            q = _linear(
                self.kernel_selection, self.reader, self._weight_name(prefix, "self_attn.q_proj.weight"), normed,
                bias_name=self._has_weight(prefix, "self_attn.q_proj.bias") if bias else None,
            )
            k = _linear(
                self.kernel_selection, self.reader, self._weight_name(prefix, "self_attn.k_proj.weight"), normed,
                bias_name=self._has_weight(prefix, "self_attn.k_proj.bias") if bias else None,
            )
            v = _linear(
                self.kernel_selection, self.reader, self._weight_name(prefix, "self_attn.v_proj.weight"), normed,
                bias_name=self._has_weight(prefix, "self_attn.v_proj.bias") if bias else None,
            )
        return (
            q.reshape(n, cfg.num_attention_heads, cfg.head_dim),
            k.reshape(n, cfg.num_key_value_heads, cfg.head_dim),
            v.reshape(n, cfg.num_key_value_heads, cfg.head_dim),
        )

    def _o_proj(self, attn_out: np.ndarray, prefix: str) -> np.ndarray:
        cfg = self.config
        name = self._has_weight(prefix, "self_attn.o_proj.weight") or self._weight_name(prefix, "attn.c_proj.weight")
        bias_name = self._has_weight(name.rsplit(".weight", 1)[0] + ".bias") if cfg.attention_bias else None
        return _linear(self.kernel_selection, self.reader, name, attn_out, bias_name=bias_name)

    def _initial_hidden(self, token_ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
        embed_name = self._weight_name("embed_tokens.weight") if self._has_weight("embed_tokens.weight") else self._weight_name("wte.weight")
        embed = self.reader.get_tensor(embed_name, dequantize=True)
        hidden = embed[token_ids].astype(np.float32)
        if self.config.position_encoding_type == "learned":
            wpe_name = self._has_weight("wpe.weight")
            if wpe_name:
                pos_embed = self.reader.get_tensor(wpe_name, dequantize=True)
                hidden = hidden + pos_embed[positions].astype(np.float32)
        return hidden, embed

    def forward_step(
        self, seq_id: int, token_ids: np.ndarray, start_position: int, return_all_positions: bool = False
    ) -> np.ndarray:
        """token_ids: (seq_len,) int array.

        Returns logits for the LAST position (vocab_size,) by default, or
        for every position (seq_len, vocab_size) when return_all_positions
        is set (used by speculative-decoding verification).
        """
        cfg = self.config
        seq_len = len(token_ids)
        positions = np.arange(start_position, start_position + seq_len)
        hidden, embed = self._initial_hidden(token_ids, positions)

        alibi_bias = None
        if cfg.position_encoding_type == "alibi":
            alibi_bias = build_alibi_bias(cfg.num_attention_heads, positions, positions)

        for layer in range(cfg.num_hidden_layers):
            prefix = f"layers.{layer}" if self._has_weight(f"layers.{layer}.input_layernorm.weight") or self._has_weight(f"layers.{layer}.ln_1.weight") else f"h.{layer}"
            residual = hidden
            norm1_name = self._has_weight(prefix, "input_layernorm.weight") or self._weight_name(prefix, "ln_1.weight")
            normed = self._norm(hidden, norm1_name)

            q, k, v = self._qkv(normed, prefix, seq_len)

            if cfg.position_encoding_type == "rope":
                q = apply_rope(q, self._cos, self._sin, positions)
                k = apply_rope(k, self._cos, self._sin, positions)

            attn_out = self.attention.prefill(seq_id, layer, q, k, v, causal=True, extra_bias=alibi_bias)
            attn_out = attn_out.reshape(seq_len, cfg.num_attention_heads * cfg.head_dim)
            attn_out = self._o_proj(attn_out, prefix)
            hidden = residual + attn_out

            residual = hidden
            norm2_name = self._has_weight(prefix, "post_attention_layernorm.weight") or self._weight_name(prefix, "ln_2.weight")
            normed = self._norm(hidden, norm2_name)
            mlp_out = self._mlp(normed, prefix)
            hidden = residual + mlp_out

        final_norm_name = self._has_weight("norm.weight") or self._weight_name("ln_f.weight")
        hidden = self._norm(hidden, final_norm_name)

        if cfg.tie_word_embeddings:
            logits_all = hidden @ embed.T
        else:
            lm_head_name = self._has_weight("lm_head.weight") or "lm_head.weight"
            logits_all = _linear(self.kernel_selection, self.reader, lm_head_name, hidden)

        if return_all_positions:
            return logits_all
        return logits_all[-1]

    def forward_batch_decode(
        self, seq_ids: "list[int]", token_ids: "list[int]", positions: "list[int]"
    ) -> np.ndarray:
        """One decode step for N sequences at once (continuous batching).
        Linear/norm/MLP ops are vectorized across the batch dimension; attention
        loops per-sequence internally (different cached lengths per sequence).
        Returns logits of shape (N, vocab_size)."""
        cfg = self.config
        n = len(seq_ids)
        if not (len(token_ids) == len(positions) == n):
            raise CPUError("seq_ids, token_ids, and positions must be the same length.")

        positions_arr = np.array(positions)
        hidden, embed = self._initial_hidden(np.array(token_ids), positions_arr)

        for layer in range(cfg.num_hidden_layers):
            prefix = f"layers.{layer}" if self._has_weight(f"layers.{layer}.input_layernorm.weight") or self._has_weight(f"layers.{layer}.ln_1.weight") else f"h.{layer}"
            residual = hidden
            norm1_name = self._has_weight(prefix, "input_layernorm.weight") or self._weight_name(prefix, "ln_1.weight")
            normed = self._norm(hidden, norm1_name)

            q, k, v = self._qkv(normed, prefix, n)

            attn_out = np.empty((n, cfg.num_attention_heads, cfg.head_dim), dtype=np.float32)
            for i in range(n):
                qi, ki, vi = q[i : i + 1], k[i : i + 1], v[i : i + 1]
                if cfg.position_encoding_type == "rope":
                    qi = apply_rope(qi, self._cos, self._sin, positions_arr[i : i + 1])
                    ki = apply_rope(ki, self._cos, self._sin, positions_arr[i : i + 1])
                extra_bias = None
                if cfg.position_encoding_type == "alibi":
                    total_k = self.kv_cache._layer_len.get((seq_ids[i], layer), 0) + 1
                    key_pos = np.arange(total_k)
                    extra_bias = build_alibi_bias(cfg.num_attention_heads, key_pos, positions_arr[i : i + 1])
                attn_out[i] = self.attention.decode(seq_ids[i], layer, qi, ki, vi, extra_bias=extra_bias)[0]

            attn_out = attn_out.reshape(n, cfg.num_attention_heads * cfg.head_dim)
            attn_out = self._o_proj(attn_out, prefix)
            hidden = residual + attn_out

            residual = hidden
            norm2_name = self._has_weight(prefix, "post_attention_layernorm.weight") or self._weight_name(prefix, "ln_2.weight")
            normed = self._norm(hidden, norm2_name)
            mlp_out = self._mlp(normed, prefix)
            hidden = residual + mlp_out

        final_norm_name = self._has_weight("norm.weight") or self._weight_name("ln_f.weight")
        hidden = self._norm(hidden, final_norm_name)

        if cfg.tie_word_embeddings:
            return hidden @ embed.T
        lm_head_name = self._has_weight("lm_head.weight") or "lm_head.weight"
        return _linear(self.kernel_selection, self.reader, lm_head_name, hidden)

    def generate_greedy(self, prompt_ids: "list[int]", max_new_tokens: int) -> "list[int]":
        seq_id = self.new_sequence()
        try:
            all_ids = list(prompt_ids)
            logits = self.forward_step(seq_id, np.array(all_ids, dtype=np.int64), start_position=0)
            for _ in range(max_new_tokens):
                next_id = int(np.argmax(logits))
                all_ids.append(next_id)
                logits = self.forward_step(
                    seq_id, np.array([next_id], dtype=np.int64), start_position=len(all_ids) - 1
                )
            return all_ids
        finally:
            self.end_sequence(seq_id)
