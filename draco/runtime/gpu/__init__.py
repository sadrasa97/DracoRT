"""GPU-side memory planning and OOM-avoidance helpers.

These are additive utilities that sit alongside the existing GPU engine
(draco.engine.llm.LLM) rather than modifying it — this environment has no
GPU to validate changes to that engine against, so it is left untouched.
"""
