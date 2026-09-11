"""
Register All Model Adapters

Registers all available model adapters with the MODEL_REGISTRY.
This module is imported at startup to ensure all adapters are available.
"""

from draco.models.registry import MODEL_REGISTRY

# Llama family
from draco.models.llama.adapter import LlamaAdapter
MODEL_REGISTRY.register(
    architecture="LlamaForCausalLM",
    implementation=LlamaAdapter,
    model_type="llama",
    aliases=["Llama2ForCausalLM", "CodeLlamaForCausalLM"],
    status="active",
)

# Mistral family
from draco.models.mistral.adapter import MistralAdapter
MODEL_REGISTRY.register(
    architecture="MistralForCausalLM",
    implementation=MistralAdapter,
    model_type="mistral",
    status="active",
)

# Qwen2 family
from draco.models.qwen2.adapter import Qwen2Adapter
MODEL_REGISTRY.register(
    architecture="Qwen2ForCausalLM",
    implementation=Qwen2Adapter,
    model_type="qwen2",
    aliases=["Qwen3ForCausalLM"],
    status="active",
)

# Gemma family
from draco.models.gemma.adapter import GemmaAdapter
MODEL_REGISTRY.register(
    architecture="GemmaForCausalLM",
    implementation=GemmaAdapter,
    model_type="gemma",
    aliases=["Gemma2ForCausalLM", "Gemma3ForCausalLM"],
    status="active",
)

# Phi family
from draco.models.phi.adapter import PhiAdapter
MODEL_REGISTRY.register(
    architecture="PhiForCausalLM",
    implementation=PhiAdapter,
    model_type="phi",
    aliases=["Phi3ForCausalLM", "Phi4ForCausalLM"],
    status="active",
)

# GPT-2 family
from draco.models.gpt2.adapter import GPT2Adapter
MODEL_REGISTRY.register(
    architecture="GPT2LMHeadModel",
    implementation=GPT2Adapter,
    model_type="gpt2",
    aliases=["GPTNeoXForCausalLM", "GPTJForCausalLM"],
    status="active",
)

# Mixtral family (Sparse MoE)
from draco.models.mixtral.adapter import MixtralAdapter
MODEL_REGISTRY.register(
    architecture="MixtralForCausalLM",
    implementation=MixtralAdapter,
    model_type="mixtral",
    status="active",
)

# Falcon family
from draco.models.falcon.adapter import FalconAdapter
MODEL_REGISTRY.register(
    architecture="FalconForCausalLM",
    implementation=FalconAdapter,
    model_type="falcon",
    status="active",
)

# DeepSeek family
from draco.models.deepseek.adapter import DeepSeekAdapter
MODEL_REGISTRY.register(
    architecture="DeepSeekV2ForCausalLM",
    implementation=DeepSeekAdapter,
    model_type="deepseek_v2",
    aliases=["DeepSeekV3ForCausalLM"],
    status="active",
)

# BLOOM family
from draco.models.bloom.adapter import BloomAdapter
MODEL_REGISTRY.register(
    architecture="BloomForCausalLM",
    implementation=BloomAdapter,
    model_type="bloom",
    status="active",
)

# OPT family
from draco.models.opt.adapter import OPTAdapter
MODEL_REGISTRY.register(
    architecture="OPTForCausalLM",
    implementation=OPTAdapter,
    model_type="opt",
    status="active",
)

# Baichuan family
from draco.models.baichuan.adapter import BaichuanAdapter
MODEL_REGISTRY.register(
    architecture="BaichuanForCausalLM",
    implementation=BaichuanAdapter,
    model_type="baichuan",
    aliases=["Baichuan2ForCausalLM"],
    status="active",
)

# Yi family
from draco.models.yi.adapter import YiAdapter
MODEL_REGISTRY.register(
    architecture="YiForCausalLM",
    implementation=YiAdapter,
    model_type="yi",
    aliases=["Yi1_5ForCausalLM"],
    status="active",
)
