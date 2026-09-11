"""
Register All Quantization Backends

Registers all available quantization backends with QUANT_REGISTRY.
"""

from draco.quantization.registry import QUANT_REGISTRY

from draco.quantization.gptq import GPTQBackend
QUANT_REGISTRY.register("gptq", GPTQBackend())

from draco.quantization.awq import AWQBackend
QUANT_REGISTRY.register("awq", AWQBackend())

from draco.quantization.fp8 import FP8Backend
QUANT_REGISTRY.register("fp8", FP8Backend())

from draco.quantization.bitsandbytes import BitsAndBytesBackend
QUANT_REGISTRY.register("bitsandbytes", BitsAndBytesBackend())
