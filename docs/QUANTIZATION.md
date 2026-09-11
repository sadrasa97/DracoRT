# Draco Quantization Guide

Draco supports **4 quantization backends** via a pluggable registry system, enabling reduced memory footprint and faster inference on compatible hardware.

## Supported Backends

| Backend | Method | Bits | Min GPU | Library |
|---------|--------|------|---------|---------|
| **GPTQ** | Weight-only INT4/INT3 | 3, 4 | Compute ≥ 7.0 | `auto-gptq` |
| **AWQ** | Weight-only W4A16 | 4 | Compute ≥ 7.0 | `autoawq` |
| **FP8** | FP8 E4M3/E5M2 | 8 | Compute ≥ 8.9 (Hopper) | PyTorch native |
| **bitsandbytes** | NF4/FP4/INT8 | 4, 8 | Compute ≥ 7.0 | `bitsandbytes` |

## Memory Savings (7B Model)

| Method | Bits | Memory (GB) | Reduction vs FP16 |
|--------|------|-------------|---------------------|
| FP16 (baseline) | 16 | 14.00 | — |
| GPTQ | 4 | 3.50 | 75% |
| AWQ | 4 | 3.50 | 75% |
| FP8 | 8 | 7.00 | 50% |
| bitsandbytes INT8 | 8 | 7.00 | 50% |
| bitsandbytes NF4 | 4 | 3.50 | 75% |

## Quick Start

### Auto-detection (from checkpoint)

Draco auto-detects quantization from `config.json`:

```python
from draco import LLM

# Auto-detects GPTQ from config.json quantization_config
llm = LLM(model="TheBloke/Llama-2-7B-Chat-GPTQ")
```

### Explicit quantization

```python
from draco import LLM

llm = LLM(
    model="meta-llama/Llama-3-8B",
    quantization="gptq",  # or "awq", "fp8", "bitsandbytes"
)
```

### List supported methods

```python
from draco import QUANT_REGISTRY

methods = QUANT_REGISTRY.list_methods()
print(methods)  # ['awq', 'bitsandbytes', 'fp8', 'gptq']

for method in methods:
    backend = QUANT_REGISTRY.get(method)
    print(f"{method}: bits={backend.supported_bits}, schemes={backend.supported_schemes}")
```

## Memory Estimation

Estimate memory before loading:

```python
from draco.quantization import QUANT_REGISTRY

num_params = 7_000_000_000  # 7B model
fp16_mem = num_params * 2 / (1024**3)  # 14 GB

for method in QUANT_REGISTRY.list_methods():
    backend = QUANT_REGISTRY.get(method)
    for bits in backend.supported_bits:
        mem = backend.estimate_memory(num_params, bits=bits)
        mem_gb = mem / (1024**3)
        reduction = 1 - mem / (num_params * 2)
        print(f"{method:15s} {bits:2d}-bit: {mem_gb:.2f} GB ({reduction:.0%} reduction)")
```

## Quantization Config Auto-Detection

```python
from draco.quantization.config import QuantizationConfig

# From a model's config.json
config_data = {
    "quantization_config": {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "desc_act": True,
    }
}

qc = QuantizationConfig.auto_detect(config_data)
print(qc)  # QuantizationConfig(method='gptq', bits=4, group_size=128, ...)
```

## Supported Architectures

All 13 Draco adapters support quantization (CPU reference mode):

| Architecture | GPTQ | AWQ | FP8 | bitsandbytes |
|-------------|------|-----|-----|-------------|
| Llama / 2 / 3 / 3.x | ✓ | ✓ | ✓ | ✓ |
| Mistral | ✓ | ✓ | ✓ | ✓ |
| Mixtral (MoE) | ✓ | ✓ | ✓ | ✓ |
| Qwen2 / 2.5 / 3 | ✓ | ✓ | ✓ | ✓ |
| Gemma / 2 / 3 | ✓ | ✓ | ✓ | ✓ |
| Phi / 2 / 3 / 4 | ✓ | ✓ | ✓ | ✓ |
| GPT-2 / NeoX / J | ✓ | ✓ | ✓ | ✓ |
| Falcon | ✓ | ✓ | ✓ | ✓ |
| DeepSeek V2 / V3 | ✓ | ✓ | ✓ | ✓ |
| BLOOM | ✓ | ✓ | ✓ | ✓ |
| OPT | ✓ | ✓ | ✓ | ✓ |
| Baichuan / 2 | ✓ | ✓ | ✓ | ✓ |
| Yi / 1.5 | ✓ | ✓ | ✓ | ✓ |

## Requirements

```bash
# Basic quantization (CPU reference mode)
pip install draco-llm

# With quantization libraries
pip install draco-llm[quantization]

# Individual backends
pip install auto-gptq     # GPTQ
pip install autoawq       # AWQ
pip install bitsandbytes  # bitsandbytes (NF4/FP4/INT8)
# FP8 is native in PyTorch 2.1+ with CUDA 11.8+
```

## Notes

- FP8 requires NVIDIA Hopper GPU (compute capability ≥ 8.9) for real acceleration
- GPTQ/AWQ require CUDA GPU with compute capability ≥ 7.0 for dequantization kernels
- bitsandbytes requires CUDA GPU for NF4/FP4/INT8 dequantization
- Without GPU, Draco uses CPU reference implementations for validation and testing
- Quantized models may have slightly lower accuracy than FP16 — test on your use case
