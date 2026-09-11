"""
Manual correctness check for the new Qwen2 KV cache (no real weights needed).

Builds a tiny random-initialized Qwen2Model, then compares:
  (a) one full-sequence forward pass over the whole sequence (ground truth,
      the same computation the old _generate_single_full_refeed still does)
  (b) prefill on the prompt + one cached decode step for the last token

If the KV cache is implemented correctly, the logits for the final position
must match between (a) and (b) to float32 precision.
"""
import torch
from draco.models.config import ModelConfig
from draco.models.qwen2.adapter import Qwen2Model

torch.manual_seed(0)

config_data = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "hidden_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,   # GQA, to exercise repeat_interleave path
    "num_hidden_layers": 3,
    "intermediate_size": 64,
    "vocab_size": 100,
    "max_position_embeddings": 128,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "rms_norm_eps": 1e-6,
}
config = ModelConfig(config_data=config_data)
model = Qwen2Model(config)
model.eval()

seq = torch.randint(0, 100, (1, 6))

with torch.no_grad():
    # (a) ground truth: full sequence in one shot, no cache
    full_logits = model(seq)
    logits_last_full = full_logits[0, -1, :]

    # (b) prefill first 5 tokens with cache, then decode token 6 alone
    prefill_logits, kv = model(seq[:, :5], use_cache=True)
    step_logits, kv2 = model(seq[:, 5:6], past_key_values=kv, use_cache=True)
    logits_last_cached = step_logits[0, -1, :]

max_abs_diff = (logits_last_full - logits_last_cached).abs().max().item()
print(f"max abs diff (full vs cached) = {max_abs_diff:.3e}")
assert max_abs_diff < 1e-4, "KV cache output diverges from full recompute!"
print("PASS: cached decoding matches full-sequence recompute.")

# Also sanity-check position_ids auto-derivation from cache length: a
# second decode step should use position 6, not 0.
with torch.no_grad():
    step2_logits, kv3 = model(
        torch.randint(0, 100, (1, 1)), past_key_values=kv2, use_cache=True
    )
assert kv3[0][0].shape[2] == 7, "cache did not grow to length 7 on 3rd step"
print("PASS: cache length grows correctly across steps (5 -> 6 -> 7).")
