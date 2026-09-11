"""
LLM Engine

Core inference engine that orchestrates model loading, scheduling,
KV cache management, and generation. This is the primary user-facing API.

Example::

    from draco import LLM, SamplingParams

    llm = LLM(model="meta-llama/Llama-3-8B", dtype="bfloat16")
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=64))
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import torch

from draco.config import DracoConfig, get_config
from draco.engine.sampling import SamplingParams
from draco.exceptions import ConfigError, DracoError, KVCacheError, ModelNotFoundError
from draco.utils.oom_guard import is_oom_error
from draco.models.config import ModelConfig
from draco.models.registry import MODEL_REGISTRY
from draco.models.runner import ModelRunner

logger = logging.getLogger("draco.engine")


class RequestOutput:
    """Output from a single generation request."""

    def __init__(
        self,
        request_id: int,
        prompt: str,
        outputs: List["CompletionOutput"],
        prompt_token_ids: Optional[List[int]] = None,
    ):
        self.request_id = request_id
        self.prompt = prompt
        self.outputs = outputs
        self.prompt_token_ids = prompt_token_ids

    def __repr__(self) -> str:
        texts = [o.text for o in self.outputs]
        return f"RequestOutput(request_id={self.request_id}, outputs={texts!r})"


class CompletionOutput:
    """Single completion output."""

    def __init__(
        self,
        text: str,
        token_ids: List[int],
        cumulative_logprob: float = 0.0,
        logprobs: Optional[Dict[int, float]] = None,
    ):
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = logprobs

    @property
    def tokens(self) -> List[int]:
        return self.token_ids

    def __repr__(self) -> str:
        return (
            f"CompletionOutput(text={self.text!r}, "
            f"tokens={len(self.token_ids)})"
        )


class LLM:
    """
    Universal GPU LLM Runtime.

    The primary user-facing interface for loading models and generating text.
    Automatically detects model architecture, loads weights, and manages
    inference.

    Example::

        llm = LLM(
            model="meta-llama/Llama-3-8B",
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
        )

        outputs = llm.generate(
            prompts=["Hello, how are you?"],
            sampling_params=SamplingParams(max_tokens=128),
        )
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        dtype: Optional[str] = "auto",
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        trust_remote_code: bool = False,
        seed: Optional[int] = None,
        gpu_device: Optional[str] = None,
        config: Optional[DracoConfig] = None,
        backend: Optional[str] = None,
        draft_model: Optional[str] = None,
        num_speculative_tokens: int = 5,
        speculative_temperature: float = 0.0,
        enable_paged_kv_cache: bool = False,
        kv_cache_block_size: int = 16,
        **kwargs: Any,
    ):
        """
        Args:
            backend: Which execution backend to use.
                - None / "transformers" (default): delegate model loading and
                  generation to the installed ``transformers`` library. This
                  is the safe default — it supports whatever architectures
                  your transformers version supports (including newer
                  text+vision "ForConditionalGeneration" models), and uses
                  transformers' own battle-tested KV-cached generate() loop.
                - "draco": use Draco's own native model adapters and
                  execution engine (CPU/GPU). Only architectures registered
                  in MODEL_REGISTRY are supported in this mode; unsupported
                  architectures raise ModelNotFoundError with a suggestion
                  to use backend="transformers" instead.
            draft_model: Optional path to a small draft model (backend="draco"
                only). When given, generation uses speculative decoding
                (draco/engine/speculative.py): the draft model proposes
                several tokens per step, the target model verifies them all
                in one forward pass, and matching tokens are accepted —
                turning several sequential target forward passes into one.
                Pick a draft model from the SAME tokenizer/vocab family as
                the target (e.g. Qwen2.5-0.5B-Instruct as a draft for a
                larger Qwen2.5 model) — mismatched vocabularies make every
                draft token a guaranteed rejection.
            num_speculative_tokens: How many tokens the draft model proposes
                per step (ignored without draft_model). 3-8 is typical;
                higher values raise potential speedup but also raise how
                much draft-model compute is wasted on a rejection.
            speculative_temperature: Sampling temperature used when the
                DRAFT model proposes tokens (ignored without draft_model).
                0.0 (greedy) is standard — it gives the draft model its best
                shot at guessing what the (equally-greedy-by-default) target
                model will pick, maximizing the acceptance rate.
            enable_paged_kv_cache: backend="draco" only. When True, the KV
                cache is stored in a fixed, size-bounded pool of blocks
                (draco.kv_cache.block.KVCacheBlockManager — the same
                block-allocator design PagedAttention/vLLM use) sized from
                real free device memory via `gpu_memory_utilization`,
                instead of letting each sequence's cache grow as an
                unbounded Python-list of tensors. Generation raises
                KVCacheError with an actionable message when the pool is
                exhausted, instead of either silently degrading or an
                opaque OOM crash. Off by default (False) since the
                unbounded per-sequence cache from step 1-3 is simpler and
                fine for single/few concurrent requests; turn this on for
                long-running or memory-constrained serving.
            kv_cache_block_size: Tokens per KV cache block (only used when
                enable_paged_kv_cache=True). 16 (vLLM's default) is a
                reasonable balance between allocation granularity and
                per-block bookkeeping overhead.
        """
        self.model_path = model
        self.tokenizer_path = tokenizer or model
        self.dtype_str = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code
        self.seed = seed
        self.config = config or get_config()

        self.backend = (backend or "transformers").lower()
        if self.backend not in ("transformers", "draco"):
            raise ConfigError(
                f"Unknown backend '{backend}'. Expected 'transformers' or 'draco'."
            )

        self.draft_model_path = draft_model
        self.num_speculative_tokens = num_speculative_tokens
        self.speculative_temperature = speculative_temperature
        self.enable_paged_kv_cache = enable_paged_kv_cache
        self.kv_cache_block_size = kv_cache_block_size

        # Resolve device
        if gpu_device:
            self.device = torch.device(gpu_device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve dtype
        self.dtype = self._resolve_dtype(dtype)

        # State
        self._model_runner: Optional[ModelRunner] = None
        self._model_config: Optional[ModelConfig] = None
        self._hf_model: Optional[Any] = None  # set when backend == "transformers"
        self._tokenizer = None
        self._request_counter = 0
        self._speculative_decoder: Optional[Any] = None  # set when draft_model given
        self._kv_block_manager: Optional[Any] = None  # set when enable_paged_kv_cache=True

        # GGUF (llama.cpp format) state, set by _initialize_gguf_backend
        self._gguf_reader = None
        self._cpu_executor = None

        # ONNX state, set by _initialize_onnx_backend
        self._onnx_reader = None

        # Metrics (lazy init)
        self._metrics: Optional[Any] = None

        # Initialize on construction
        self._initialize()

    def _resolve_dtype(self, dtype: Optional[str]) -> torch.dtype:
        """Resolve dtype string to torch.dtype."""
        if dtype is None or dtype == "auto":
            if torch.cuda.is_available():
                # Prefer bfloat16 if supported
                capability = torch.cuda.get_device_capability()
                if capability[0] >= 8:
                    return torch.bfloat16
                return torch.float16
            return torch.float32

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float8": torch.float8_e4m3fn,
            "fp8": torch.float8_e4m3fn,
            "int8": torch.int8,
            "int4": torch.quint4x2,
        }
        return dtype_map.get(dtype.lower(), torch.float16)

    def _initialize(self) -> None:
        """Initialize the engine: load config, detect architecture, build model."""
        if self.backend == "transformers":
            self._initialize_transformers_backend()
        else:
            self._initialize_draco_backend()

    def _initialize_transformers_backend(self) -> None:
        """Load the model and tokenizer via the installed `transformers`
        library. This is the default backend: it defers architecture
        support entirely to transformers, so it works for any model your
        transformers version supports (text-only or text+vision
        "ForConditionalGeneration" models alike), and reuses transformers'
        own KV-cached generate() loop rather than Draco's native engine.
        """
        logger.info("Initializing transformers backend...")
        logger.info("Model: %s", self.model_path)
        logger.info("Device: %s, dtype: %s", self.device, self.dtype)

        # Best-effort: still try to load Draco's ModelConfig for
        # introspection (architecture(), get_model_config()), but never let
        # a failure here (e.g. an architecture Draco doesn't recognize)
        # block the transformers backend from working.
        try:
            self._model_config = ModelConfig.from_pretrained(self.model_path)
        except Exception as e:
            logger.debug("Could not load Draco ModelConfig (informational only): %s", e)
            self._model_config = None

        self._load_tokenizer()

        from_pretrained_kwargs: Dict[str, Any] = dict(
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.dtype,
        )
        if self.quantization:
            from_pretrained_kwargs["quantization_config"] = self._build_hf_quantization_config()

        # Try progressively more specific/newer Auto classes so that
        # text+vision "ForConditionalGeneration" architectures (e.g. newer
        # Qwen-VL-style models) load correctly too, not just plain
        # text-only causal LMs.
        last_error: Optional[Exception] = None
        auto_class_names = [
            "AutoModelForCausalLM",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModel",
        ]
        model = None
        for class_name in auto_class_names:
            try:
                import transformers

                auto_cls = getattr(transformers, class_name, None)
                if auto_cls is None:
                    continue
                model = auto_cls.from_pretrained(self.model_path, **from_pretrained_kwargs)
                logger.info("Loaded model via transformers.%s", class_name)
                break
            except Exception as e:
                last_error = e
                logger.debug("transformers.%s failed to load model: %s", class_name, e)
                continue

        if model is None:
            raise ConfigError(
                f"Failed to load '{self.model_path}' with any transformers Auto* class "
                f"({', '.join(auto_class_names)}). Last error: {last_error}. "
                f"If this is a custom architecture, try trust_remote_code=True, or "
                f"check that your installed transformers version supports it."
            )

        model = model.to(device=self.device)
        model.eval()
        self._hf_model = model

        if self.max_model_len is None:
            self.max_model_len = getattr(
                getattr(model, "config", None), "max_position_embeddings", None
            ) or self.config.default_max_model_len

        logger.info("transformers backend initialized successfully")

    def _build_hf_quantization_config(self) -> Any:
        """Best-effort mapping of Draco's quantization method name to a
        transformers/bitsandbytes BitsAndBytesConfig, when applicable."""
        method = (self.quantization or "").lower()
        if method in ("bitsandbytes", "int8", "int4", "bnb"):
            try:
                from transformers import BitsAndBytesConfig

                if method == "int4":
                    return BitsAndBytesConfig(load_in_4bit=True)
                return BitsAndBytesConfig(load_in_8bit=True)
            except ImportError as e:
                raise ConfigError(
                    f"quantization='{self.quantization}' requires the bitsandbytes package: {e}"
                )
        # gptq/awq/fp8: transformers resolves these from the checkpoint's own
        # quantization_config in config.json automatically; nothing to build here.
        return None

    def _initialize_draco_backend(self) -> None:
        """Initialize Draco's own native model adapters and execution
        engine (CPU/GPU). Only architectures registered in MODEL_REGISTRY
        are supported here."""
        logger.info("Initializing Draco LLM engine...")
        logger.info("Model: %s", self.model_path)
        logger.info("Device: %s, dtype: %s", self.device, self.dtype)

        # GGUF (llama.cpp format) models bypass the config.json/safetensors
        # machinery entirely and load through the native CPU reader + executor.
        from draco.format.gguf import is_gguf_file

        if os.path.isfile(self.model_path) and is_gguf_file(self.model_path):
            self._initialize_gguf_backend()
            return

        # ONNX models (a single .onnx file, or a directory containing one
        # — e.g. an `optimum-cli export onnx` output) likewise bypass the
        # PyTorch adapter machinery: the computation graph is already
        # fully specified, so there's no per-architecture adapter to pick.
        from draco.format.onnx import is_onnx_model

        if is_onnx_model(self.model_path):
            self._initialize_onnx_backend()
            return

        # Load model config
        try:
            self._model_config = ModelConfig.from_pretrained(self.model_path)
        except Exception as e:
            raise ConfigError(f"Failed to load model config: {e}")

        # Apply max_model_len
        if self.max_model_len is None:
            self.max_model_len = min(
                self._model_config.max_position_embeddings,
                self.config.default_max_model_len,
            )

        # Validate quantization auto-detection
        if self.quantization is None and self._model_config.is_quantized:
            detected = self._model_config.quantization.quantization_method
            if detected:
                logger.info("Auto-detected quantization: %s", detected)
                self.quantization = detected

        # Resolve adapter
        arch = self._model_config.detect_architecture()
        model_type = self._model_config.model_type
        logger.info("Architecture: %s (model_type: %s)", arch, model_type)

        # Build runner
        self._model_runner = ModelRunner(
            model_path=self.model_path,
            dtype=self.dtype,
            device=self.device,
            quantization=self.quantization,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
        )
        self._model_runner.initialize()

        # Load tokenizer
        self._load_tokenizer()

        # Optional speculative decoding: load a small draft model and wrap
        # both models in the existing SpeculativeDecoder (engine/speculative.py)
        # — previously built but never referenced anywhere in this file.
        if self.draft_model_path:
            self._init_speculative_decoder()

        if self.enable_paged_kv_cache:
            self._init_paged_kv_cache()

        logger.info("Draco LLM engine initialized successfully")
        model_info = self._model_runner.get_model_info()
        logger.info("Model info: %s", model_info)

    def _init_paged_kv_cache(self) -> None:
        """Build a size-bounded KVCacheBlockManager sized from real free
        device memory, wired in place of the unbounded Python-list KV
        cache used by _generate_single/_generate_batch — see
        _generate_single_paged for the append/read bridge between the
        per-layer (k, v) tuples the adapters return and this manager's
        block-table storage."""
        from draco.kv_cache.block import KVCacheBlockManager

        cfg = self._model_config
        try:
            self._kv_block_manager = KVCacheBlockManager.from_gpu_memory_utilization(
                num_layers=cfg.num_hidden_layers,
                num_kv_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                block_size=self.kv_cache_block_size,
                dtype=self.dtype if isinstance(self.dtype, torch.dtype) else torch.float16,
                device=self.device,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
        except RuntimeError:
            # from_gpu_memory_utilization -> plan_kv_cache_blocks requires a
            # CUDA device to query free memory; on CPU there's no
            # equivalent "free VRAM" query, so size from host RAM instead
            # via a fixed, conservative block count. Paged mode still
            # gives bounded memory (the actual point of this feature) —
            # it just can't be auto-sized from device introspection here.
            self._kv_block_manager = KVCacheBlockManager(
                num_layers=cfg.num_hidden_layers,
                num_kv_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                block_size=self.kv_cache_block_size,
                num_blocks=2048,
                dtype=self.dtype if isinstance(self.dtype, torch.dtype) else torch.float32,
                device=self.device,
            )
        logger.info("Paged KV cache enabled: %r", self._kv_block_manager)

    def _init_speculative_decoder(self) -> None:
        """Load the draft model (backend="draco" only, same device/dtype as
        the target) and construct a SpeculativeDecoder wrapping both."""
        from draco.engine.speculative import SpeculativeConfig, SpeculativeDecoder

        logger.info("Loading draft model for speculative decoding: %s", self.draft_model_path)
        draft_runner = ModelRunner(
            model_path=self.draft_model_path,
            dtype=self.dtype,
            device=self.device,
            quantization=None,
            tensor_parallel_size=1,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
        )
        draft_runner.initialize()
        self._draft_model_runner = draft_runner

        self._speculative_decoder = SpeculativeDecoder(
            target_model=self._model_runner.executor.model,
            draft_model=draft_runner.executor.model,
            config=SpeculativeConfig(
                num_speculative_tokens=self.num_speculative_tokens,
                draft_temperature=self.speculative_temperature,
            ),
            tokenizer=self._tokenizer,
        )
        logger.info(
            "Speculative decoding enabled: draft=%s, num_speculative_tokens=%d",
            self.draft_model_path, self.num_speculative_tokens,
        )

    def _initialize_gguf_backend(self) -> None:
        """Load a llama.cpp .gguf model through the native CPU reader and
        execution graph (the same pure-numpy machinery the .draco CPU
        runtime uses), reusing this engine's sampling / stop-token logic.
        The CPU executor is KV-cached, so generation here is O(n), unlike
        the no-KV-cache torch adapters.
        """
        logger.info("Initializing Draco GGUF backend...")
        logger.info("Model: %s", self.model_path)

        from draco.format.gguf import GGUFReader
        from draco.runtime.cpu.executor import CPUExecutionBackend

        self._gguf_reader = GGUFReader(self.model_path)
        self._model_config = None
        self._model_runner = None
        self._cpu_executor = CPUExecutionBackend(
            self._gguf_reader,
            max_num_seqs=1,
            kv_budget_bytes=int(512 * 1024 * 1024),  # cap KV preallocation
        )
        self.max_model_len = int(
            self._gguf_reader.metadata.get("max_position_embeddings", 4096)
        )

        self._load_tokenizer()
        if self._tokenizer is None:
            self._tokenizer = self._build_gguf_tokenizer()
        if self._tokenizer is None:
            raise ConfigError(
                f"GGUF model '{self.model_path}' has no usable tokenizer. Pass "
                f"tokenizer=<path-to-a-transformers-tokenizer> (e.g. the HF "
                f"model directory the GGUF was converted from), or use a GGUF "
                f"file that embeds tokenizer.ggml.* metadata."
            )
        logger.info("GGUF backend initialized: %s", self.architecture)

    def _build_gguf_tokenizer(self):
        """Build the byte-level BPE tokenizer from the GGUF-embedded
        tokenizer metadata, so .gguf models work without transformers."""
        from draco.format.gguf_tokenizer import GGUFBPETokenizer

        meta = getattr(self._gguf_reader, "tokenizer", None) or {}
        tokens = meta.get("tokens")
        if not tokens:
            return None
        try:
            return GGUFBPETokenizer(
                tokens=tokens,
                merges=meta.get("merges") or [],
                scores=meta.get("scores"),
                token_type=meta.get("token_type"),
                bos_token_id=meta.get("bos_token_id"),
                eos_token_id=meta.get("eos_token_id"),
                unknown_token_id=meta.get("unknown_token_id"),
                chat_template=meta.get("chat_template"),
                model_type=meta.get("model") or "llama",
            )
        except Exception as e:
            logger.warning("Could not build GGUF tokenizer from metadata: %s", e)
            return None

    def _initialize_onnx_backend(self) -> None:
        """Load a .onnx model (single file, or a directory containing one
        — e.g. an Optimum `export onnx` output) via onnxruntime. See
        draco/format/onnx.py for the supported input/output conventions
        (stateless, decoder-with-past, and merged decoder-with-past).
        """
        logger.info("Initializing Draco ONNX backend...")
        logger.info("Model: %s", self.model_path)

        from draco.format.onnx import ONNXModelReader

        self._onnx_reader = ONNXModelReader(self.model_path, device=self.device)
        self._model_runner = None

        # Best-effort: load config.json for max_model_len / architecture
        # introspection, same as the transformers backend does — an ONNX
        # export directory usually still ships the original config.json.
        try:
            config_dir = (
                self.model_path
                if os.path.isdir(self.model_path)
                else os.path.dirname(self.model_path)
            )
            self._model_config = ModelConfig.from_pretrained(config_dir)
        except Exception as e:
            logger.debug("Could not load Draco ModelConfig for ONNX model (informational only): %s", e)
            self._model_config = None

        if self.max_model_len is None:
            self.max_model_len = (
                self._model_config.max_position_embeddings
                if self._model_config is not None
                else self.config.default_max_model_len
            )

        self._load_tokenizer()
        if self._tokenizer is None:
            raise ConfigError(
                f"ONNX model '{self.model_path}' has no usable tokenizer. Pass "
                f"tokenizer=<path-to-a-transformers-tokenizer> (e.g. the "
                f"directory the model was exported from)."
            )
        logger.info(
            "ONNX backend initialized: uses_cache=%s, num_layers=%d",
            self._onnx_reader.uses_cache, self._onnx_reader.num_layers,
        )


        """Load tokenizer for encoding/decoding."""
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=self.trust_remote_code,
            )
            logger.info("Tokenizer loaded: %s", type(self._tokenizer).__name__)
        except Exception as e:
            logger.warning("Failed to load tokenizer: %s", e)
            self._tokenizer = None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """
        Generate text for one or more prompts.

        Args:
            prompts: Single prompt string or list of prompt strings.
            sampling_params: Sampling parameters. Uses defaults if None.
            use_tqdm: Whether to show a progress bar.

        Returns:
            List of RequestOutput objects.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        if isinstance(prompts, str):
            prompts = [prompts]

        if self.backend == "transformers":
            return self._generate_transformers(prompts, sampling_params)

        # Encode prompts (applies the tokenizer's chat template for
        # instruction-tuned models, so generation starts from the
        # "assistant" turn and stops on the model's chat end token).
        prompt_token_ids_list = []
        for prompt in prompts:
            ids = self._encode_prompt_for_generation(prompt)
            prompt_token_ids_list.append(ids)

        # Run generation. Multiple prompts on backend="draco" (excluding
        # GGUF and speculative-decoding paths, which aren't batched yet —
        # see _generate_batch's docstring) go through one batched forward
        # pass per decode step instead of N fully-sequential per-prompt
        # generate() calls — this is what draco.scheduler.scheduler.
        # ContinuousBatchScheduler already does for the GGUF/native-CPU
        # path; this wires the same idea (one step, many sequences, one
        # forward pass) into the main HF-directory-loading path.
        if (
            len(prompts) > 1
            and self._cpu_executor is None
            and self._onnx_reader is None
            and self._speculative_decoder is None
        ):
            request_ids = [self._next_request_id() for _ in prompts]
            completions = self._generate_batch(prompt_token_ids_list, sampling_params)
            return [
                RequestOutput(
                    request_id=rid, prompt=prompt, outputs=[completion],
                    prompt_token_ids=token_ids,
                )
                for rid, prompt, token_ids, completion in zip(
                    request_ids, prompts, prompt_token_ids_list, completions
                )
            ]

        outputs = []
        for i, (prompt, token_ids) in enumerate(
            zip(prompts, prompt_token_ids_list)
        ):
            request_id = self._next_request_id()
            if self._cpu_executor is not None:
                completion = self._generate_single_gguf(
                    token_ids, sampling_params, request_id
                )
            elif self._onnx_reader is not None:
                completion = self._generate_single_onnx(
                    token_ids, sampling_params, request_id
                )
            else:
                completion = self._generate_single(
                    token_ids, sampling_params, request_id
                )
            output = RequestOutput(
                request_id=request_id,
                prompt=prompt,
                outputs=[completion],
                prompt_token_ids=token_ids,
            )
            outputs.append(output)

        return outputs

    def _generate_batch(
        self,
        prompt_token_ids_list: List[List[int]],
        params: SamplingParams,
    ) -> List["CompletionOutput"]:
        """Generate for several prompts at once: left-pad them to a common
        length, prefill in ONE forward pass, then decode one token per
        sequence per step in ONE forward pass — instead of the fully
        sequential N-separate-generate-calls loop `generate()` otherwise
        uses. Relies on the same per-adapter KV cache as `_generate_single`
        (every registered architecture supports batch_size > 1 already,
        since none of the cache/attention code above is batch_size==1
        specific); falls back to N sequential `_generate_single` calls for
        any architecture whose adapter doesn't return a cache.

        This is static batching (the batch composition is fixed for the
        whole call, unlike true continuous batching which admits/evicts
        requests mid-stream) — still a real, substantial improvement over
        one-forward-pass-per-token-per-prompt for the common "generate for
        a list of prompts" case.
        """
        if self._model_runner is None:
            raise RuntimeError("LLM not initialized")

        bsz = len(prompt_token_ids_list)
        pad_id = 0
        if self._tokenizer is not None:
            pad_id = (
                getattr(self._tokenizer, "pad_token_id", None)
                or getattr(self._tokenizer, "eos_token_id", None)
                or 0
            )

        max_len = max(len(ids) for ids in prompt_token_ids_list)
        # Left-pad: real tokens end up right-aligned, so the newest
        # (rightmost) position is always the one every sequence continues
        # generating from — no per-sequence bookkeeping needed for that.
        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=self.device)
        attn_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=self.device)
        for i, ids in enumerate(prompt_token_ids_list):
            input_ids[i, max_len - len(ids):] = torch.tensor(ids, dtype=torch.long, device=self.device)
            attn_mask[i, max_len - len(ids):] = 1

        # RoPE/learned-position architectures need each sequence's REAL
        # token positions, not a uniform arange (that would treat left
        # pad tokens as real position 0, shifting every real token's
        # position by the pad length). ALiBi/no-position architectures
        # ignore this (their Model.forward doesn't take position_ids at
        # all) since ALiBi's relative-distance bias is pad-length
        # invariant for real-token-to-real-token pairs anyway.
        position_ids = attn_mask.cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)

        def additive_mask(mask_1d: torch.Tensor) -> torch.Tensor:
            # (bsz, kv_len) of 0/1 -> (bsz, 1, 1, kv_len) additive mask
            return (1.0 - mask_1d[:, None, None, :].to(self.dtype)) * torch.finfo(self.dtype).min

        with torch.no_grad():
            out = self._model_runner.forward(
                input_ids, position_ids=position_ids,
                attention_mask=additive_mask(attn_mask), use_cache=True,
            )
        using_cache = isinstance(out, tuple) and out[1] is not None and out[1][0] is not None
        if not using_cache:
            # Adapter doesn't support caching -> fall back to N sequential
            # (still-correct) generate() calls instead of guessing at a
            # batched-without-cache implementation.
            return [
                self._generate_single(ids, params, self._next_request_id())
                for ids in prompt_token_ids_list
            ]

        logits, past_key_values = out
        next_logits = logits[:, -1, :]

        generated: List[List[int]] = [[] for _ in range(bsz)]
        finished = [False] * bsz
        cur_attn_mask = attn_mask
        # Each row's REAL (non-pad) token count — needed because rows are
        # padded by different amounts, so "position = cache length" (the
        # single-sequence fallback derive_position_ids uses) is wrong
        # here: a row with 4 pad + 3 real tokens must continue from RoPE
        # position 3, not from the padded cache length 7. Every row
        # advances by exactly one real position per decode step (finished
        # rows still get a dummy token fed through, but that doesn't
        # affect any other row, and we never read their generated output
        # past the stop point).
        real_lens = attn_mask.sum(dim=-1)

        for step in range(params.max_tokens):
            next_tokens = torch.empty(bsz, dtype=torch.long, device=self.device)
            for i in range(bsz):
                if finished[i]:
                    # Keep the batch shape intact; this row's token is
                    # never appended to its output, so its value here is
                    # irrelevant to the result (it only ever feeds back
                    # into that same row's own future attention).
                    next_tokens[i] = pad_id
                    continue
                token = self._sample_token(next_logits[i].clone(), params, generated[i])
                next_tokens[i] = token
                if self._should_stop(token, params, generated[i]):
                    finished[i] = True
                else:
                    generated[i].append(token)

            if all(finished):
                break
            if step == params.max_tokens - 1:
                break

            step_input = next_tokens.unsqueeze(1)
            cur_attn_mask = torch.cat(
                [cur_attn_mask, torch.ones((bsz, 1), dtype=torch.long, device=self.device)], dim=1
            )
            step_position_ids = (real_lens + step).unsqueeze(1)
            with torch.no_grad():
                logits, past_key_values = self._model_runner.forward(
                    step_input, position_ids=step_position_ids,
                    past_key_values=past_key_values, use_cache=True,
                    attention_mask=additive_mask(cur_attn_mask),
                )
            next_logits = logits[:, -1, :]

        return [
            CompletionOutput(
                text=self.decode(ids) if self._tokenizer else "",
                token_ids=ids,
            )
            for ids in generated
        ]

    def _generate_transformers(
        self, prompts: List[str], params: SamplingParams
    ) -> List[RequestOutput]:
        """Generation path for backend='transformers': delegates the whole
        autoregressive loop to the model's own .generate(), which handles
        KV caching, position ids, and attention masks correctly (and
        consistently across architectures) — Draco doesn't reimplement any
        of that here.
        """
        if self._hf_model is None:
            raise RuntimeError("LLM not initialized (transformers backend)")

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=params.max_tokens,
            min_new_tokens=params.min_tokens or None,
            do_sample=params.temperature > 1e-7 and not params.use_beam_search,
            num_beams=max(1, params.best_of) if params.use_beam_search else 1,
            repetition_penalty=params.repetition_penalty,
        )
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = params.effective_temperature
            gen_kwargs["top_p"] = params.top_p
            if params.top_k > 0:
                gen_kwargs["top_k"] = params.top_k
        if params.ignore_eos:
            gen_kwargs["eos_token_id"] = None
        elif params.stop_token_ids:
            eos_ids = list(params.stop_token_ids)
            base_eos = getattr(self._tokenizer, "eos_token_id", None)
            if base_eos is not None and base_eos not in eos_ids:
                eos_ids.append(base_eos)
            gen_kwargs["eos_token_id"] = eos_ids
        if self.seed is not None:
            torch.manual_seed(self.seed)

        outputs = []
        for prompt in prompts:
            request_id = self._next_request_id()
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                out_ids = self._hf_model.generate(**inputs, **gen_kwargs)

            generated_ids = out_ids[0, prompt_len:].tolist()
            text = self.decode(generated_ids)

            completion = CompletionOutput(text=text, token_ids=generated_ids)
            outputs.append(
                RequestOutput(
                    request_id=request_id,
                    prompt=prompt,
                    outputs=[completion],
                    prompt_token_ids=inputs["input_ids"][0].tolist(),
                )
            )
        return outputs

    def _generate_single(
        self,
        prompt_token_ids: List[int],
        params: SamplingParams,
        request_id: int,
    ) -> CompletionOutput:
        """Generate a single completion (autoregressive loop), Draco native backend.

        Uses a real per-layer KV cache when the loaded architecture's
        adapter supports it (currently: Qwen2/Qwen2.5/Qwen3 — see
        draco/models/qwen2/adapter.py): the prompt is run once to fill the
        cache, then every decode step feeds only the single newest token —
        O(n) instead of O(n^2), matching how vLLM/llama.cpp/HF's own
        generate() all work.

        For any architecture whose adapter has NOT been upgraded yet,
        ModelExecutor.forward(use_cache=True) has no cache to return, so
        we fall back to the previous (correct but O(n^2)) full-sequence
        refeed on every step — see _generate_single_full_refeed below.
        """
        if self._model_runner is None:
            raise RuntimeError("LLM not initialized")

        if self._speculative_decoder is not None:
            return self._generate_single_speculative(prompt_token_ids, params, request_id)

        if self._kv_block_manager is not None:
            return self._generate_single_paged(prompt_token_ids, params, request_id)

        generated_ids: List[int] = []
        all_token_ids = list(prompt_token_ids)

        # Prefill: run the whole prompt once, requesting a cache.
        model_input = torch.tensor([all_token_ids], dtype=torch.long, device=self.device)
        out = self._model_runner.forward(model_input, use_cache=True)
        if not (isinstance(out, tuple) and out[1] is not None and out[1][0] is not None):
            # Adapter doesn't support caching yet -> old, safe fallback.
            return self._generate_single_full_refeed(prompt_token_ids, params, request_id)

        logits, past_key_values = out
        next_logits = logits[0, -1, :]

        for step in range(params.max_tokens):
            next_token = self._sample_token(next_logits, params, generated_ids)
            if self._should_stop(next_token, params, generated_ids):
                break
            generated_ids.append(next_token)
            all_token_ids.append(next_token)

            if step == params.max_tokens - 1:
                break

            # Decode step: feed ONLY the new token; position_ids is left
            # None so Qwen2Model.forward derives it from the cache length
            # (past_len + 0), which is what keeps RoPE positions correct.
            step_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            logits, past_key_values = self._model_runner.forward(
                step_input, past_key_values=past_key_values, use_cache=True
            )
            next_logits = logits[0, -1, :]

        text = self.decode(generated_ids) if self._tokenizer else ""
        return CompletionOutput(text=text, token_ids=generated_ids)

    def _generate_single_speculative(
        self,
        prompt_token_ids: List[int],
        params: SamplingParams,
        request_id: int,
    ) -> CompletionOutput:
        """Generate using the wired-up SpeculativeDecoder (draft model
        proposes several tokens, target model verifies them all in one
        forward pass). Both models still use the O(n^2) full-refeed
        algorithm internally (draco/engine/speculative.py re-feeds the
        growing sequence every step) — it does not yet reuse the
        per-adapter KV cache added for the plain decode path. That's a
        further optimization on top of this wiring, not required to get
        speculative decoding working correctly."""
        stop_ids = list(params.stop_token_ids) if params.stop_token_ids else None
        if self._tokenizer is not None:
            eos_id = getattr(self._tokenizer, "eos_token_id", None)
            if eos_id is not None:
                stop_ids = (stop_ids or []) + [eos_id]

        generated_ids = self._speculative_decoder.generate(
            prompt_token_ids=prompt_token_ids,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            stop_token_ids=stop_ids,
        )
        text = self.decode(generated_ids) if self._tokenizer else ""
        return CompletionOutput(text=text, token_ids=generated_ids)

    def _generate_single_paged(
        self,
        prompt_token_ids: List[int],
        params: SamplingParams,
        request_id: int,
    ) -> CompletionOutput:
        """Same algorithm as _generate_single (prefill once, then one new
        token per decode step), but the KV cache lives in
        self._kv_block_manager's fixed block pool instead of an
        ever-growing Python list of tensors: after every forward call, the
        newly-produced per-layer (k, v) tuples are appended into the block
        manager (KVCacheBlockManager.append_kv), and the NEXT forward call
        reads its `past_key_values` back out of the block manager
        (get_kv per layer) rather than carrying the raw tensors forward
        itself. This bounds total KV cache memory to the pool size chosen
        at LLM init (enable_paged_kv_cache=True) instead of growing
        without limit — and raises KVCacheError with an actionable message
        when the pool is exhausted, rather than an opaque OOM crash or (worse)
        the block manager's own default of silently truncating on overflow.
        """
        if self._model_runner is None or self._kv_block_manager is None:
            raise RuntimeError("LLM not initialized")
        mgr = self._kv_block_manager
        seq_id = request_id

        def _append(present_key_values, new_chunk_len: int) -> None:
            # present_key_values: list of per-layer (k, v), each shaped
            # (1, num_kv_heads, total_len, head_dim) where total_len is the
            # PAST (read back from the manager) + this call's new tokens
            # concatenated together (the adapters' own cache concat, which
            # doesn't know anything already lives in the block manager).
            # Only the trailing `new_chunk_len` tokens are actually new —
            # appending the whole tensor every step would re-store
            # everything already in the pool again and again.
            if not mgr.ensure_space(seq_id, new_chunk_len):
                mgr.free_sequence(seq_id)
                raise KVCacheError(
                    f"Paged KV cache pool exhausted (request {seq_id}): "
                    f"{mgr.num_free_blocks} free / {mgr.num_blocks} total blocks "
                    f"of {mgr.block_size} tokens each. Lower gpu_memory_utilization's "
                    f"competing usage, reduce max_model_len, or reduce concurrent "
                    f"requests."
                )
            stacked_k = torch.stack(
                [k[0, :, -new_chunk_len:, :] for k, v in present_key_values], dim=0
            )
            stacked_v = torch.stack(
                [v[0, :, -new_chunk_len:, :] for k, v in present_key_values], dim=0
            )
            mgr.append_kv(seq_id, stacked_k, stacked_v)

        def _read_back(num_layers: int):
            # Read the FULL cached sequence back out of the block manager
            # for every layer, restoring the batch dim so it matches the
            # shape every adapter's attention module expects.
            return [
                tuple(t.unsqueeze(0) for t in mgr.get_kv(seq_id, layer_idx))
                for layer_idx in range(num_layers)
            ]

        try:
            generated_ids: List[int] = []
            all_token_ids = list(prompt_token_ids)

            model_input = torch.tensor([all_token_ids], dtype=torch.long, device=self.device)
            try:
                out = self._model_runner.forward(model_input, use_cache=True)
            except Exception as e:
                if is_oom_error(e):
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                raise
            if not (isinstance(out, tuple) and out[1] is not None and out[1][0] is not None):
                # Adapter doesn't support caching -> paging doesn't apply either.
                return self._generate_single_full_refeed(prompt_token_ids, params, request_id)

            logits, present_key_values = out
            _append(present_key_values, new_chunk_len=model_input.shape[1])
            next_logits = logits[0, -1, :]
            num_layers = len(present_key_values)

            for step in range(params.max_tokens):
                next_token = self._sample_token(next_logits, params, generated_ids)
                if self._should_stop(next_token, params, generated_ids):
                    break
                generated_ids.append(next_token)
                all_token_ids.append(next_token)

                if step == params.max_tokens - 1:
                    break

                past_key_values = _read_back(num_layers)
                step_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                try:
                    logits, present_key_values = self._model_runner.forward(
                        step_input, past_key_values=past_key_values, use_cache=True
                    )
                except Exception as e:
                    if is_oom_error(e):
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    raise
                _append(present_key_values, new_chunk_len=1)
                next_logits = logits[0, -1, :]

            text = self.decode(generated_ids) if self._tokenizer else ""
            return CompletionOutput(text=text, token_ids=generated_ids)
        finally:
            # Always release this request's blocks, success or error —
            # otherwise a raised KVCacheError (or any other exception)
            # would leak blocks and make the pool exhaustion permanent.
            mgr.free_sequence(seq_id)

    def _generate_single_full_refeed(
        self,
        prompt_token_ids: List[int],
        params: SamplingParams,
        request_id: int,
    ) -> CompletionOutput:
        """Fallback autoregressive loop for architectures whose adapter
        doesn't implement a KV cache yet: re-feeds the FULL sequence every
        step. O(n^2), but unambiguously correct for any adapter that only
        implements a plain ``forward(input_ids) -> logits``."""
        if self._model_runner is None:
            raise RuntimeError("LLM not initialized")

        generated_ids: List[int] = []
        all_token_ids = list(prompt_token_ids)

        for step in range(params.max_tokens):
            model_input = torch.tensor(
                [all_token_ids], dtype=torch.long, device=self.device
            )
            position_ids = torch.arange(len(all_token_ids), device=self.device).unsqueeze(0)

            logits = self._model_runner.forward(model_input, position_ids=position_ids)

            next_logits = logits[0, -1, :]
            next_token = self._sample_token(next_logits, params, generated_ids)

            if self._should_stop(next_token, params, generated_ids):
                break

            generated_ids.append(next_token)
            all_token_ids.append(next_token)

        text = self.decode(generated_ids) if self._tokenizer else ""
        return CompletionOutput(
            text=text,
            token_ids=generated_ids,
        )

    def _generate_single_gguf(
        self,
        prompt_token_ids: List[int],
        params: SamplingParams,
        request_id: int,
    ) -> CompletionOutput:
        """KV-cached autoregressive loop over the native CPU executor
        (used for .gguf models). The CPU executor keeps K/V per token, so
        each step feeds only the newest token — O(n), no full-sequence
        refeed.
        """
        import numpy as np
        import torch

        if self._cpu_executor is None:
            raise RuntimeError("GGUF backend not initialized")

        seq_id = self._cpu_executor.new_sequence()
        try:
            generated_ids: List[int] = []
            all_ids = list(prompt_token_ids)
            logits = self._cpu_executor.forward_step(
                seq_id, np.array(all_ids, dtype=np.int64), start_position=0
            )
            for _ in range(params.max_tokens):
                logits_t = torch.from_numpy(
                    np.ascontiguousarray(logits, dtype=np.float32)
                )
                next_token = self._sample_token(logits_t, params, generated_ids)
                if self._should_stop(next_token, params, generated_ids):
                    break
                generated_ids.append(next_token)
                all_ids.append(next_token)
                logits = self._cpu_executor.forward_step(
                    seq_id,
                    np.array([next_token], dtype=np.int64),
                    start_position=len(all_ids) - 1,
                )
            text = self.decode(generated_ids) if self._tokenizer else ""
            return CompletionOutput(text=text, token_ids=generated_ids)
        finally:
            self._cpu_executor.end_sequence(seq_id)

    def _generate_single_onnx(
        self,
        prompt_token_ids: List[int],
        params: SamplingParams,
        request_id: int,
    ) -> CompletionOutput:
        """Same prefill-then-decode shape as _generate_single, but backed
        by ONNXModelReader.forward instead of a draco/models/*/adapter.py
        nn.Module. When the loaded ONNX graph has no past_key_values I/O
        (a stateless export), falls back to re-feeding the full sequence
        every step — same tradeoff as _generate_single_full_refeed."""
        if self._onnx_reader is None:
            raise RuntimeError("ONNX backend not initialized")

        generated_ids: List[int] = []
        all_token_ids = list(prompt_token_ids)

        model_input = torch.tensor([all_token_ids], dtype=torch.long, device=self.device)
        out = self._onnx_reader.forward(model_input, use_cache=True)
        using_cache = isinstance(out, tuple)
        if using_cache:
            logits, past_key_values = out
        else:
            logits = out

        next_logits = logits[0, -1, :]
        for step in range(params.max_tokens):
            next_token = self._sample_token(next_logits, params, generated_ids)
            if self._should_stop(next_token, params, generated_ids):
                break
            generated_ids.append(next_token)
            all_token_ids.append(next_token)

            if step == params.max_tokens - 1:
                break

            if using_cache:
                step_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                logits, past_key_values = self._onnx_reader.forward(
                    step_input, past_key_values=past_key_values, use_cache=True
                )
            else:
                full_input = torch.tensor([all_token_ids], dtype=torch.long, device=self.device)
                logits = self._onnx_reader.forward(full_input, use_cache=False)
            next_logits = logits[0, -1, :]

        text = self.decode(generated_ids) if self._tokenizer else ""
        return CompletionOutput(text=text, token_ids=generated_ids)

    def _sample_token(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generated_ids: Optional[List[int]] = None,
    ) -> int:
        """Sample a single token from logits."""
        if params.temperature <= 0 or params.temperature < 1e-7:
            # Greedy
            return logits.argmax().item()

        # Apply temperature
        logits = logits / params.effective_temperature

        # Apply repetition / presence / frequency penalties to the logits of
        # previously generated tokens only (HF-style). Penalizing every logit
        # uniformly (as earlier versions did) is a no-op for greedy decoding
        # and does nothing to discourage loops for sampling.
        generated_ids = generated_ids or []
        if generated_ids:
            if params.repetition_penalty != 1.0:
                penalty = params.repetition_penalty
                for token_id in set(generated_ids):
                    if logits[token_id] > 0:
                        logits[token_id] = logits[token_id] / penalty
                    else:
                        logits[token_id] = logits[token_id] * penalty
            if params.presence_penalty != 0.0 or params.frequency_penalty != 0.0:
                from collections import Counter

                counts = Counter(generated_ids)
                for token_id, count in counts.items():
                    logits[token_id] -= (
                        params.presence_penalty + params.frequency_penalty * count
                    )

        # Apply top-k
        if params.top_k > 0:
            top_k_vals, _ = torch.topk(logits, params.top_k)
            min_val = top_k_vals[-1]
            logits[logits < min_val] = float("-inf")

        # Apply top-p (nucleus)
        if params.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > params.top_p
            # Keep at least one token
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float("-inf")

        # Sample
        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1).item()
        return token

    def _should_stop(
        self,
        token_id: int,
        params: SamplingParams,
        generated_ids: List[int],
    ) -> bool:
        """Check if generation should stop."""
        # EOS tokens: the tokenizer's eos_token_id plus any additional EOS
        # ids declared in the model's generation_config.json (e.g. Qwen2.5
        # lists both <|im_end|> and <|endoftext|>). Missing these is what
        # made generations run all the way to max_tokens and ramble.
        if not params.ignore_eos:
            for eos_id in self._eos_token_ids():
                if token_id == eos_id:
                    return True

        # Stop token IDs
        if params.stop_token_ids and token_id in params.stop_token_ids:
            return True

        return False

    # ------------------------------------------------------------------
    # Tokenizer interface
    # ------------------------------------------------------------------

    def _chat_wrapped_ids(self, prompt: str) -> Optional[List[int]]:
        """Wrap a raw user prompt as a single-turn chat message using the
        tokenizer's chat template, so instruction-tuned models generate from
        the "assistant" turn and emit their chat end token (<|im_end|> for
        Qwen, <|eot_id|> for Llama-3, ...) to stop naturally.

        Returns None when there is no chat template, or when the prompt is
        already chat-formatted (contains a special token string) — the
        latter keeps server /v1/chat/completions prompts from being wrapped
        twice.
        """
        tok = self._tokenizer
        if tok is None or not getattr(tok, "chat_template", None):
            return None
        special_tokens = getattr(tok, "all_special_tokens", None)
        if special_tokens and any(s in prompt for s in special_tokens):
            return None
        try:
            formatted = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            if formatted is None:
                return None
            return tok.encode(formatted)
        except Exception as e:
            logger.debug("chat template wrapping failed, using raw prompt: %s", e)
            return None

    def _encode_prompt_for_generation(self, prompt: str) -> List[int]:
        """Encode a prompt for native-backend generation. Applies the chat
        template when the model has one; falls back to plain encoding."""
        wrapped = self._chat_wrapped_ids(prompt)
        if wrapped is not None:
            return wrapped
        return self.encode(prompt)

    def _eos_token_ids(self) -> List[int]:
        """All end-of-sequence token ids for this model: the tokenizer's
        eos_token_id plus any additional ids declared in the model's
        generation_config.json."""
        ids: List[int] = []

        def add(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, (list, tuple)):
                for v in value:
                    if v is not None:
                        ids.append(int(v))
            else:
                ids.append(int(value))

        if self._tokenizer is not None:
            add(getattr(self._tokenizer, "eos_token_id", None))
            # some tokenizers carry additional end tokens (e.g. Qwen's
            # <|im_end|> alongside <|endoftext|>)
            add(getattr(self._tokenizer, "extra_eos_ids", None))
        try:
            import json

            gen_cfg = os.path.join(self.model_path, "generation_config.json")
            if os.path.isfile(gen_cfg):
                with open(gen_cfg, encoding="utf-8") as f:
                    add(json.load(f).get("eos_token_id"))
        except Exception as e:
            logger.debug("could not read generation_config.json for EOS ids: %s", e)
        return sorted(set(ids))

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs (raw, no chat template)."""
        if self._tokenizer is not None:
            return self._tokenizer.encode(text)
        # Fallback: character-level
        return list(text.encode("utf-8"))

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self._tokenizer is not None:
            return self._tokenizer.decode(token_ids, skip_special_tokens=True)
        return "".join(chr(t) if t < 128 else "?" for t in token_ids)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics(self) -> Any:
        """Return current runtime metrics."""
        if self._metrics is None:
            from draco.metrics import MetricsCollector

            self._metrics = MetricsCollector()
        return self._metrics.snapshot()

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_model_config(self) -> Optional[ModelConfig]:
        """Return the model configuration."""
        return self._model_config

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        if self.backend == "transformers":
            if self._hf_model is None:
                return {"status": "not_initialized"}
            hf_config = getattr(self._hf_model, "config", None)
            return {
                "backend": "transformers",
                "architecture": getattr(hf_config, "architectures", ["unknown"])[0]
                if getattr(hf_config, "architectures", None)
                else "unknown",
                "model_type": getattr(hf_config, "model_type", "unknown"),
                "num_params": sum(p.numel() for p in self._hf_model.parameters()),
            }
        if self._gguf_reader is not None:
            import numpy as np

            reader = self._gguf_reader
            return {
                "backend": "draco-gguf",
                "architecture": reader.architecture,
                "model_type": reader.metadata.get("model_type", "unknown"),
                "format": "GGUF",
                "num_params": sum(
                    int(np.prod(t.shape)) for t in reader.tensors()
                ),
            }
        if self._model_runner is None:
            return {"status": "not_initialized"}
        return self._model_runner.get_model_info()

    @property
    def architecture(self) -> str:
        """Return the model architecture name."""
        if self.backend == "transformers" and self._hf_model is not None:
            hf_config = getattr(self._hf_model, "config", None)
            archs = getattr(hf_config, "architectures", None)
            if archs:
                return archs[0]
        if self._gguf_reader is not None:
            return self._gguf_reader.architecture or "unknown"
        if self._model_config:
            return self._model_config.architecture or "unknown"
        return "unknown"

    @property
    def supported_quantization(self) -> List[str]:
        """Return list of supported quantization methods."""
        if self._model_runner and self._model_runner.adapter:
            methods = []
            for method in ("gptq", "awq", "fp8", "int8", "int4", "bitsandbytes"):
                if self._model_runner.adapter.supports_quantization(method):
                    methods.append(method)
            return methods
        return []

    def _next_request_id(self) -> int:
        self._request_counter += 1
        return self._request_counter

    def __repr__(self) -> str:
        return (
            f"LLM(model={self.model_path!r}, "
            f"backend={self.backend!r}, "
            f"architecture={self.architecture!r}, "
            f"device={self.device}, "
            f"dtype={self.dtype})"
        )

    def __del__(self) -> None:
        """Cleanup resources."""
        try:
            if hasattr(self, "_model_runner") and self._model_runner is not None:
                del self._model_runner
            if hasattr(self, "_hf_model") and self._hf_model is not None:
                del self._hf_model
            if hasattr(self, "_gguf_reader") and self._gguf_reader is not None:
                self._gguf_reader.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
