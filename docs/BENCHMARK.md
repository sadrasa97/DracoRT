# Draco Benchmark Guide

Draco provides a built-in benchmarking framework for measuring inference performance across multiple dimensions.

## Quick Start

### CLI Benchmarking

```bash
# Single request benchmark
draco bench --model meta-llama/Llama-3-8B \
    --input-tokens 2048 --output-tokens 256

# Batch scaling
draco bench --model meta-llama/Llama-3-8B \
    --mode batch_scaling --batch-sizes 1,2,4,8

# Context scaling
draco bench --model meta-llama/Llama-3-8B \
    --mode context_scaling --context-lengths 512,2048,8192

# Generation scaling
draco bench --model meta-llama/Llama-3-8B \
    --mode generation_scaling --generation-lengths 64,256,1024

# Backend comparison
draco bench-compare --backend draco,vllm --model meta-llama/Llama-3-8B
```

### Python API

```python
from draco.benchmarks.framework import BenchmarkRunner, EnvironmentInfo

# Collect environment info
env = EnvironmentInfo.collect()
print(f"GPU: {env.gpu_model}")
print(f"CUDA: {env.cuda_version}")
print(f"PyTorch: {env.pytorch_version}")

# Create runner
runner = BenchmarkRunner(output_dir="benchmarks/results", save_results=True)

# Define benchmark function
def my_benchmark(batch_size=1, input_tokens=512, output_tokens=128, **kwargs):
    import time
    time.sleep(0.1)  # Replace with actual inference
    return {
        "ttft_ms": 42.0,
        "tpot_ms": 8.0,
        "itl_ms": 8.5,
        "tokens_generated": output_tokens,
        "gpu_memory_gb": 14.2,
    }

# Run
result = runner.run_benchmark(
    name="llama3_8b_test",
    benchmark_fn=my_benchmark,
    model="Llama-3-8B",
    dtype="bfloat16",
    batch_size=1,
    input_tokens=512,
    output_tokens=128,
    num_iterations=5,
    warmup_iterations=1,
)

print(result.format_table())
```

## Benchmark Modes

### 1. Single Request

Measures TTFT, latency, and tokens/sec for a single prompt.

```python
from draco.benchmarks.modes import SingleRequestBenchmark

bench = SingleRequestBenchmark(runner)
result = bench.run(llm, max_tokens=128)
```

### 2. Concurrent Requests

Tests behavior under parallel load.

```python
from draco.benchmarks.modes import ConcurrentRequestBenchmark

bench = ConcurrentRequestBenchmark(runner)
results = bench.run(llm, concurrency_levels=[1, 2, 4, 8, 16])
```

### 3. Batch Scaling

Tests throughput scaling with batch size.

```python
from draco.benchmarks.modes import BatchScalingBenchmark

bench = BatchScalingBenchmark(runner)
results = bench.run(llm, batch_sizes=[1, 2, 4, 8, 16])
```

### 4. Context Scaling

Tests performance at different input lengths.

```python
from draco.benchmarks.modes import ContextScalingBenchmark

bench = ContextScalingBenchmark(runner)
results = bench.run(llm, context_lengths=[512, 1024, 2048, 8192])
```

### 5. Generation Scaling

Tests performance at different output lengths.

```python
from draco.benchmarks.modes import GenerationScalingBenchmark

bench = GenerationScalingBenchmark(runner)
results = bench.run(llm, generation_lengths=[64, 256, 1024, 2048])
```

## Backend Comparison

Compare Draco against other backends:

```python
from draco.benchmarks.comparison import ComparisonBenchmark

comparator = ComparisonBenchmark(runner)
comparison = comparator.compare(
    backends=["draco", "vllm"],
    model="Llama-3-8B",
    input_tokens=512,
    output_tokens=128,
    num_iterations=5,
)
print(comparison)
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to first token (ms) |
| **TPOT** | Time per output token (ms) |
| **ITL** | Inter-token latency (ms) |
| **Throughput** | Tokens per second |
| **p50/p95/p99 Latency** | Latency percentiles |
| **GPU Memory** | Peak GPU memory usage (GB) |
| **Acceptance Rate** | Speculative decoding acceptance rate |
| **Memory Reduction** | Quantization memory savings |

## Production Benchmarks

Draco includes 52 production-level CPU benchmarks covering all subsystems:

```bash
# Run all production benchmarks
python -m pytest tests/test_production_benchmarks.py -v

# Categories:
# Level 1:  Basic throughput (decode, prefill 32/128, large model)
# Level 2:  Scheduler (batch formation, request throughput)
# Level 3:  KV cache (allocation, deallocation, utilization)
# Level 5:  Batch scaling (1→32 batch sizes)
# Level 7:  Quantization comparison (FP16 vs INT8, memory estimation)
# Level 8:  Speculative decoding (acceptance rates, speedup curves)
# Level 10: Prefix cache (hit rates, eviction, speed)
# Level 11: Streaming (latency, SSE serialization, overhead)
# Level 12: Metrics (collection overhead, Prometheus export)
# Level 14: Server API (completion, chat, streaming, parsing)
# Level 15: Production quality (registry, config, end-to-end)
```

## Result Persistence

Results are saved as JSON in the output directory:

```python
runner = BenchmarkRunner(output_dir="benchmarks/results", save_results=True)
# Results automatically saved to benchmarks/results/<timestamp>_<name>.json
```

## Environment Info

Collect system information for reproducibility:

```python
from draco.benchmarks.framework import EnvironmentInfo

env = EnvironmentInfo.collect()
print(f"Draco:  {env.draco_version}")
print(f"Python: {env.python_version}")
print(f"PyTorch: {env.pytorch_version}")
print(f"CUDA:   {env.cuda_version}")
print(f"GPU:    {env.gpu_model}")
print(f"Memory: {env.gpu_memory_gb} GB")
```

## Regression Detection

Compare against baselines:

```python
from draco.benchmarks.comparison import ComparisonBenchmark

comparator = ComparisonBenchmark(runner)
diff = comparator.compare(
    backends=["draco_v1", "draco_v2"],
    model="Llama-3-8B",
    input_tokens=512,
    output_tokens=128,
    num_iterations=10,
)
# Positive speedup = v2 is faster
```
