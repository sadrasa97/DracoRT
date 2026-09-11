"""CPU GEMM kernels, by ISA tier.

Only the 'generic' portable tier is actually implemented right now (pure
numpy, correctness-tested against the quantization codecs' reference
dequant path). AVX2/AVX512/VNNI/AMX/NEON tiers require a compiled
extension built and tested against real hardware — see
draco.runtime.cpu.dispatcher for how unimplemented tiers are refused
rather than silently downgraded without telling the caller.
"""
