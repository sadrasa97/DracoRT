"""Draco — 10-line inference demo. Run: python demo_simple.py"""

from draco import LLM, SamplingParams

# 1. Load model (auto-detects architecture from config.json)
llm = LLM(model="Qwen/Qwen2.5-0.5B", dtype="auto")

# 2. Run inference
outputs = llm.generate(
    ["What is the capital of France? Tell me in one sentence."],
    SamplingParams(max_tokens=128, temperature=0.7),
)

# 3. Print result
print(outputs[0].outputs[0].text)
