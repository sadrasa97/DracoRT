"""
Draco — Universal GPU + CPU LLM Runtime

A plugin/registry-based model system supporting Hugging Face causal language
models with universal quantization, metrics, and benchmarking. LLM(...)
defaults to backend="transformers" (delegates to the installed transformers
library, supporting whatever architectures it does); pass backend="draco"
to use Draco's own native CPU/GPU adapters (MODEL_REGISTRY) instead.
"""

__version__ = "0.8.0"

# Register all model adapters and quantization backends on import
import draco.models._register_all  # noqa: F401
import draco.quantization._register_all  # noqa: F401

from draco.engine.llm import LLM
from draco.engine.sampling import SamplingParams
from draco.engine.stream import StreamingGenerator, StreamOutput, StreamDelta
from draco.engine.speculative import SpeculativeDecoder, SpeculativeConfig, SpeculativeStats
from draco.kv_cache import KVCacheBlockManager
from draco.scheduler import ContinuousBatchScheduler
from draco.prefix_cache import PrefixCache
from draco.models.registry import MODEL_REGISTRY
from draco.quantization.registry import QUANT_REGISTRY
from draco.metrics import MetricsCollector
from draco.server import DracoServer
from draco.server.rate_limiter import RateLimiter
from draco.engine.chunked_prefill import ChunkedPrefiller
from draco.engine.async_llm import AsyncLLM
from draco.parallel import TensorParallelConfig, split_tensor, merge_tensor
from draco.kernels.ops import fused_rms_norm, fused_rope, fused_swiGLU, fused_attention
from draco.attention.paged import PagedAttention

__all__ = [
    "LLM", "SamplingParams",
    "StreamingGenerator", "StreamOutput", "StreamDelta",
    "SpeculativeDecoder", "SpeculativeConfig", "SpeculativeStats",
    "KVCacheBlockManager",
    "ContinuousBatchScheduler",
    "PrefixCache",
    "DracoServer", "RateLimiter",
    "ChunkedPrefiller",
    "AsyncLLM",
    "TensorParallelConfig", "split_tensor", "merge_tensor",
    "fused_rms_norm", "fused_rope", "fused_swiGLU", "fused_attention",
    "PagedAttention",
    "MODEL_REGISTRY", "QUANT_REGISTRY", "MetricsCollector",
]
