"""
# Install from the built wheel
pip install dist/draco_llm-0.8.0-py3-none-any.whl

# Or install in editable/development mode
pip install -e .

# With optional extras
pip install dist/draco_llm-0.1.0-py3-none-any.whl[quantization]  # GPTQ/AWQ/bitsandbytes
pip install dist/draco_llm-0.1.0-py3-none-any.whl[bench]         # Benchmarking tools
pip install dist/draco_llm-0.1.0-py3-none-any.whl[all]            # Everything

"""



from draco import LLM, SamplingParams
from draco.models.registry import MODEL_REGISTRY
from draco.quantization import QUANT_REGISTRY
from draco.metrics import MetricsCollector

print("Draco imported successfully!")
print(f"Registered architectures: {len(MODEL_REGISTRY.list_architectures())}")
print(f"Quantization backends: {QUANT_REGISTRY.list_methods()}")

# Option A: HuggingFace repo ID (downloads automatically)
model_path = r"D:\models\Qwen2.5-0.5B-Instruct"

"""
# Option B: Local path to a saved checkpoint
model_path = "/path/to/local/Llama-3-8B"

# Option C: Quantized model
model_path = "TheBloke/Llama-2-7B-Chat-GPTQ"

"""

llm = LLM(
    model=model_path,
    dtype="auto",                    # auto-detect: bf16/fp16/fp32
    max_model_len=4096,             # max context length
    gpu_memory_utilization=0.90,     # GPU memory fraction to use
    tensor_parallel_size=1,          # number of GPUs
    quantization=None,               # "gptq", "awq", "fp8", "bitsandbytes"
    trust_remote_code=False,
    seed=42,
    # backend: which execution backend to use.
    #   "transformers" (default) — delegates to the installed transformers
    #       library (AutoModelForCausalLM / AutoModelForImageTextToText /
    #       AutoModelForVision2Seq fallback chain + its own .generate()).
    #       Supports whatever architectures your transformers version does,
    #       text-only or text+vision, with correct KV-cached generation.
    #   "draco" — Draco's own native CPU/GPU adapters (MODEL_REGISTRY).
    #       Only registered architectures are supported in this mode. Every
    #       registered architecture (Qwen2, Llama, Mistral, Gemma, Yi, Phi,
    #       Mixtral, DeepSeek, Baichuan, BLOOM, Falcon, GPT-2, OPT) now has
    #       a real per-layer KV cache, so decoding is O(n) per token
    #       instead of re-feeding the whole sequence every step.
    backend="draco",
    # --- Optional: speculative decoding (backend="draco" only) ---
    # Point draft_model at a small model from the SAME tokenizer/vocab
    # family as `model` (e.g. Qwen2.5-0.5B-Instruct as a draft for a
    # larger Qwen2.5 model). It proposes several tokens per step; the
    # target model verifies them all in one forward pass, so a high
    # acceptance rate means several tokens per target forward pass
    # instead of one. Leave as None to disable (the default).
    # draft_model=r"D:\models\Qwen2.5-0.5B-Instruct",
    # num_speculative_tokens=5,
)

print(llm)
# LLM(model='Qwen/Qwen2.5-0.5B', architecture='LlamaForCausalLM', device=cuda, dtype=bfloat16)


# Creative writing — note: SamplingParams defaults are temperature=1.0 /
# top_p=1.0 / top_k=-1 (full multinomial), NOT greedy. Set explicit
# temperature/top_p/top_k/repetition_penalty for controlled output.
sampling_params = SamplingParams(
    temperature=0.7,      # 0 = greedy, >0 = random sampling
    top_p=0.95,           # nucleus sampling threshold
    top_k=50,             # top-k filtering
    max_tokens=256,       # max tokens to generate
    repetition_penalty=1.1,
)

# Single prompt
outputs = llm.generate(
    ["What is the capital of France?"],
    sampling_params,
)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Response: {output.outputs[0].text!r}")
    print(f"Tokens generated: {len(output.outputs[0].token_ids)}")


prompts = [
    "Explain quantum computing in one sentence.",
    "Write a haiku about programming.",
    "What are the three laws of thermodynamics?",
]

# Multiple prompts here run as ONE batched forward pass per decode step
# (LLM._generate_batch: left-padding + a shared KV cache across the whole
# batch), not one full sequential generate() call per prompt.
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Q: {output.prompt}")
    print(f"A: {output.outputs[0].text}")
    print()


from draco import StreamingGenerator

generator = StreamingGenerator(llm)
for stream_output in generator.stream(
    "Tell me a story about a robot",
    max_tokens=256,
    temperature=0.8,
):
    # Print tokens as they arrive
    if stream_output.deltas:
        print(stream_output.deltas[-1].text, end="", flush=True)
    if stream_output.finished:
        print("\n[Done]")


# --- Speculative decoding example ---
# Uncomment and point draft_model at a smaller model from the same
# tokenizer/vocab family to try it. Prints the acceptance rate (how many
# draft-proposed tokens the target model actually agreed with) and the
# estimated speedup over plain autoregressive decoding.
#
# llm_spec = LLM(
#     model=model_path,
#     backend="draco",
#     draft_model=r"D:\models\Qwen2.5-0.5B-Instruct",  # small, same family
#     num_speculative_tokens=5,
# )
# outputs = llm_spec.generate(["What is the capital of France?"], sampling_params)
# print(outputs[0].outputs[0].text)
# stats = llm_spec._speculative_decoder.get_stats()
# print(f"acceptance_rate={stats.acceptance_rate:.2f}  speedup={stats.speedup:.2f}x")


# --- Bounded-memory (paged) KV cache example ---
# By default (enable_paged_kv_cache=False) each request's KV cache grows
# as an unbounded Python list of tensors — fine for a handful of
# requests, but with no ceiling on total memory use. Turning this on
# routes the cache through draco.kv_cache.block.KVCacheBlockManager (the
# same fixed block-pool design PagedAttention/vLLM use), sized from real
# free device memory via gpu_memory_utilization. Once the pool is full,
# generation raises KVCacheError with an actionable message instead of
# an opaque OOM crash or unbounded growth. Currently applies to
# single-prompt generation (backend="draco"); multi-prompt batched calls
# (see llm.generate(prompts, ...) above) don't use the paged pool yet.
#
# llm_paged = LLM(
#     model=model_path,
#     backend="draco",
#     enable_paged_kv_cache=True,
#     kv_cache_block_size=16,
# )
# outputs = llm_paged.generate(["Explain quantum computing in one sentence."], sampling_params)
# print(outputs[0].outputs[0].text)


# --- ONNX model example ---
# Point `model` at a .onnx file, or a directory containing one (e.g. the
# output of `optimum-cli export onnx --model <hf-model> <output-dir>`).
# backend="draco" auto-detects it — same as it already does for .gguf —
# so no separate backend name is needed. Supports both a stateless
# export (re-feeds the full sequence every step, like _generate_single_
# full_refeed) and a decoder-with-past / merged-decoder-with-past export
# (real KV cache, O(n) decoding) — detected automatically from the
# ONNX graph's own declared input names. See draco/format/onnx.py.
#
# llm_onnx = LLM(
#     model=r"D:\models\Qwen2.5-0.5B-Instruct-onnx",  # dir with *.onnx + config.json
#     backend="draco",
# )
# outputs = llm_onnx.generate(["What is the capital of France?"], sampling_params)
# print(outputs[0].outputs[0].text)


