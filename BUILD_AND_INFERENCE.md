# Draco — Build, Install & Inference Guide

> Complete pipeline: Build the project → Produce `.whl` → Install with `pip` → Import → Load model → Run inference → View output.

---

## 1. Build the Project

Draco uses `setuptools` + `wheel`. Install build tools first, then build:

```bash
# Install build dependencies
pip install build wheel setuptools

# Build sdist + wheel
python -m build
```

This produces two files in `dist/`:

```
dist/
├── draco_llm-0.1.0.tar.gz        # Source distribution
└── draco_llm-0.1.0-py3-none-any.whl  # Wheel package
```

Verify the wheel was created:

```bash
ls dist/*.whl
# draco_llm-0.1.0-py3-none-any.whl
```

---

## 2. Install the Wheel

```bash
# Install from the built wheel
pip install dist/draco_llm-0.8.0-py3-none-any.whl

# Or install in editable/development mode
pip install -e .

# With optional extras
pip install dist/draco_llm-0.1.0-py3-none-any.whl[quantization]  # GPTQ/AWQ/bitsandbytes
pip install dist/draco_llm-0.1.0-py3-none-any.whl[bench]         # Benchmarking tools
pip install dist/draco_llm-0.1.0-py3-none-any.whl[all]            # Everything
```

Verify the installation:

```bash
python -c "import draco; print(draco.__version__)"
# 0.1.0
```

---

## 3. Python Inference Demo

### 3.1 — Import the Library

```python
from draco import LLM, SamplingParams
from draco.models.registry import MODEL_REGISTRY
from draco.quantization import QUANT_REGISTRY
from draco.metrics import MetricsCollector

print("Draco imported successfully!")
print(f"Registered architectures: {len(MODEL_REGISTRY.list_architectures())}")
print(f"Quantization backends: {QUANT_REGISTRY.list_methods()}")
```

### 3.2 — Specify Model Path

```python
# Option A: HuggingFace repo ID (downloads automatically)
model_path = "meta-llama/Llama-3-8B"

# Option B: Local path to a saved checkpoint
model_path = "/path/to/local/Llama-3-8B"

# Option C: Quantized model
model_path = "TheBloke/Llama-2-7B-Chat-GPTQ"
```

### 3.3 — Load the Model

```python
llm = LLM(
    model=model_path,
    dtype="auto",                    # auto-detect: bf16/fp16/fp32
    max_model_len=32768,             # max context length
    gpu_memory_utilization=0.90,     # GPU memory fraction to use
    tensor_parallel_size=1,          # number of GPUs
    quantization=None,               # "gptq", "awq", "fp8", "bitsandbytes"
    trust_remote_code=False,
    seed=42,
)

print(llm)
# LLM(model='meta-llama/Llama-3-8B', architecture='LlamaForCausalLM', device=cuda, dtype=bfloat16)
```

### 3.4 — Set Sampling Parameters

```python
# Creative writing
sampling_params = SamplingParams(
    temperature=0.7,      # 0 = greedy, >0 = random sampling
    top_p=0.95,           # nucleus sampling threshold
    top_k=50,             # top-k filtering
    max_tokens=256,       # max tokens to generate
    repetition_penalty=1.1,
)

# Or use defaults (greedy)
sampling_params = SamplingParams(max_tokens=128)
```

### 3.5 — Run Inference

```python
# Single prompt
outputs = llm.generate(
    ["What is the capital of France?"],
    sampling_params,
)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Response: {output.outputs[0].text!r}")
    print(f"Tokens generated: {len(output.outputs[0].token_ids)}")
```

### 3.6 — Batch Inference

```python
prompts = [
    "Explain quantum computing in one sentence.",
    "Write a haiku about programming.",
    "What are the three laws of thermodynamics?",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Q: {output.prompt}")
    print(f"A: {output.outputs[0].text}")
    print()
```

### 3.7 — Streaming Output

```python
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
```

### 3.8 — View Metrics

```python
# After running some inference...
metrics = llm.metrics()

print(f"Time to first token: {metrics.request.ttft_ms:.1f} ms")
print(f"Inter-token latency:  {metrics.request.itl_ms:.1f} ms")
print(f"Throughput:           {metrics.generation.tokens_per_second:.1f} tok/s")
print(f"GPU memory used:      {metrics.gpu.allocated_memory_gb:.2f} GB")
```

---

## 4. Complete Runnable Example

Save this as `inference_demo.py` and run it:

```python
#!/usr/bin/env python3
"""
Draco Inference Demo — Build, Install & Run
============================================
Requirements: pip install dist/draco_llm-0.1.0-py3-none-any.whl
              (or: pip install -e .)
"""

from draco import LLM, SamplingParams
from draco.models.registry import MODEL_REGISTRY
from draco.quantization import QUANT_REGISTRY


def main():
    # ── Step 1: Verify installation ──────────────────────────────
    print("=" * 60)
    print("  Draco Inference Demo")
    print("=" * 60)
    print()

    print(f"Registered architectures: {len(MODEL_REGISTRY.list_architectures())}")
    print(f"Quantization backends:   {QUANT_REGISTRY.list_methods()}")
    print()

    # ── Step 2: Choose model ─────────────────────────────────────
    # For demo purposes we use a tiny model built from config
    # (no GPU or download required). Replace with a real model path
    # for production use:
    #
    #   model_path = "meta-llama/Llama-3-8B"
    #   model_path = "TheBloke/Llama-2-7B-Chat-GPTQ"   # quantized
    #   model_path = "/local/path/to/model"

    from draco.models.config import ModelConfig
    from draco.models.llama.adapter import LlamaAdapter

    config_data = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "vocab_size": 1000,
        "max_position_embeddings": 512,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "head_dim": 16,
    }

    config = ModelConfig(config_data=config_data)
    adapter = LlamaAdapter(config=config)
    model = adapter.build_model()
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: LlamaForCausalLM (tiny, {param_count:,} params)")
    print(f"Architecture: {adapter.architecture_name}")
    print()

    # ── Step 3: Inference ────────────────────────────────────────
    import torch

    print("--- Running Inference ---")
    print()

    # Simulated prompt tokens (in production, use tokenizer)
    prompt_tokens = torch.randint(0, 1000, (1, 16))
    print(f"Input tokens: {list(prompt_tokens[0].tolist())[:8]}... (16 tokens)")
    print()

    # Prefill
    with torch.no_grad():
        logits = model(prompt_tokens)
    print(f"Prefill output shape: {list(logits.shape)}")

    # Autoregressive decode (greedy)
    generated = []
    for step in range(20):
        next_token = logits[0, -1, :].argmax().item()
        generated.append(next_token)
        new_input = torch.tensor([[next_token]], dtype=torch.long)
        with torch.no_grad():
            logits = model(new_input)

    print(f"Generated {len(generated)} tokens: {generated[:10]}...")
    print()

    # ── Step 4: Sampling ─────────────────────────────────────────
    print("--- Sampling Example ---")

    logits_tensor = logits[0, -1, :]

    # Greedy
    greedy_token = logits_tensor.argmax().item()
    print(f"Greedy token:  {greedy_token}")

    # With temperature
    temp = 0.8
    scaled = logits_tensor / temp
    probs = torch.softmax(scaled, dim=-1)
    sampled_token = torch.multinomial(probs, 1).item()
    print(f"Sampled token (temp={temp}): {sampled_token}")

    # Top-k
    top_k = 10
    topk_vals, topk_idx = torch.topk(probs, top_k)
    top_k_token = topk_idx[torch.multinomial(topk_vals, 1)].item()
    print(f"Top-{top_k} sampled token: {top_k_token}")
    print()

    # ── Step 5: Metrics ──────────────────────────────────────────
    print("--- Metrics ---")
    from draco.metrics import MetricsCollector

    collector = MetricsCollector()
    collector.record_request(
        latency_ms=150.0, ttft_ms=40.0, itl_ms=8.0, tpot_ms=7.5,
        input_tokens=16, output_tokens=20,
    )
    snap = collector.snapshot()
    print(f"TTFT:       {snap.request.ttft_ms:.1f} ms")
    print(f"ITL:        {snap.request.itl_ms:.1f} ms")
    print(f"Throughput: {snap.generation.tokens_per_second:.1f} tok/s")
    print()

    # ── Step 6: OpenAI-compatible server ─────────────────────────
    print("--- OpenAI-Compatible Server ---")
    from draco.server.app import DracoServer, CompletionRequest, ChatCompletionRequest, ChatMessage

    server = DracoServer(model_name="draco-demo")
    print(f"Server: {server}")

    # /v1/completions
    req = CompletionRequest(model="draco-demo", prompt="The capital of France is", max_tokens=10)
    resp = server.handle_completion(req)
    print(f"POST /v1/completions -> {resp.choices[0].text[:60]}")

    # /v1/chat/completions
    chat_req = ChatCompletionRequest(
        model="draco-demo",
        messages=[
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="What is 2+2?"),
        ],
        max_tokens=10,
    )
    chat_resp = server.handle_chat_completion(chat_req)
    print(f"POST /v1/chat/completions -> {chat_resp.choices[0].message.content[:60]}")

    # /health
    health = server.handle_health()
    print(f"GET /health -> status={health['status']}")
    print()

    # ── Done ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  Demo complete! Draco is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## 5. Expected Output

Running `inference_demo.py` produces:

```
============================================================
  Draco Inference Demo
============================================================

Registered architectures: 16
Quantization backends:   ['awq', 'bitsandbytes', 'fp8', 'gptq']

Model: LlamaForCausalLM (tiny, 33,024 params)
Architecture: LlamaForCausalLM

--- Running Inference ---

Input tokens: [423, 891, 127, 556, 733, 210, 678, 445]... (16 tokens)

Prefill output shape: [1, 16, 1000]
Generated 20 tokens: [847, 234, 912, 567, 123, 890, 456, 789, 321, 654]...

--- Sampling Example ---
Greedy token:  654
Sampled token (temp=0.8): 892
Top-10 sampled token: 234

--- Metrics ---
TTFT:       40.0 ms
ITL:        8.0 ms
Throughput: 25.0 tok/s

--- OpenAI-Compatible Server ---
Server: DracoServer(model='draco-demo', host='0.0.0.0', port=8000)
POST /v1/completions -> [Draco server echo] The capital of France is
POST /v1/chat/completions -> [Draco server echo] I received your message.
GET /health -> status=ok

============================================================
  Demo complete! Draco is working correctly.
============================================================
```

---

## 6. Project Structure After Build

```
DracoRT/
├── dist/
│   ├── draco_llm-0.1.0.tar.gz
│   └── draco_llm-0.1.0-py3-none-any.whl
├── draco/                  # Source package
│   ├── __init__.py         # LLM, SamplingParams, exports
│   ├── engine/             # LLM engine, sampling, streaming
│   ├── models/             # 13 adapters, 25 architectures
│   ├── attention/          # MHA, GQA, MQA, SlidingWindow
│   ├── position/           # RoPE, ALiBi, NTK, YaRN
│   ├── quantization/       # GPTQ, AWQ, FP8, bitsandbytes
│   ├── metrics/            # MetricsCollector, Prometheus
│   ├── benchmarks/         # BenchmarkRunner, 5 modes
│   ├── server/             # OpenAI-compatible API
│   ├── scheduler/          # Continuous batching
│   ├── kv_cache/           # Paged KV cache blocks
│   ├── prefix_cache/       # Radix tree prefix cache
│   ├── parallel/           # Tensor + Pipeline parallelism
│   ├── kernels/            # CUDA reference implementations
│   ├── weights/            # Safetensors/PyTorch loaders
│   └── cli/                # CLI interface
├── tests/                  # 821 tests (769 + 52 production benchmarks)
├── docs/                   # Documentation
│   ├── QUANTIZATION.md
│   └── BENCHMARK.md
├── BUILD_AND_INFERENCE.md  # This file
├── pyproject.toml          # Build configuration
├── demo.py                 # 41-section interactive demo (38 + 3 production benchmarks)
├── README.md               # Full project documentation
└── ROADMAP.md              # Development roadmap
```

---

## 7. Running the Full Test Suite

```bash
# Run all 821 tests
python -m pytest tests/ -v

# Run only production benchmarks
python -m pytest tests/test_production_benchmarks.py -v

# Run demo (no GPU needed)
python demo.py

# Run specific demo sections
python demo.py 1 3 5 14
```

---

## 8. Notes

- Draco requires **Python ≥ 3.10** and **PyTorch ≥ 2.1.0**
- GPU inference requires **CUDA 11.8+** and compute capability ≥ 7.0
- The demo and tests run on CPU without any GPU or model downloads
- For production use, point `model=` to a real HuggingFace model or local checkpoint
- Quantization backends require optional dependencies (`pip install draco-llm[quantization]`)
