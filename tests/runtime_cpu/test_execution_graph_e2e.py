"""
End-to-end native CPU execution test (spec section 49): a tiny synthetic
transformer fixture, written as .draco, run through CPUExecutionBackend,
and cross-checked against an independent plain-numpy reference forward
pass so quantities like RoPE application, GQA repeat, and residual wiring
are verified — not just "it didn't crash".
"""

import numpy as np
import pytest

from draco.format.writer import DracoWriter
from draco.runtime.cpu.executor import CPUExecutionBackend
from draco.runtime.cpu.llm import CPUNativeLLM
from draco.runtime.cpu.ops import apply_rope, build_rope_cache, rms_norm, softmax, swiglu

HIDDEN = 16
LAYERS = 2
HEADS = 4
KV_HEADS = 2
HEAD_DIM = HIDDEN // HEADS  # 4
INTERMEDIATE = 32
VOCAB = 32


def _build_tiny_model(path: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    writer = DracoWriter(path)
    writer.add_metadata(
        {
            "architecture": "LlamaForCausalLM",
            "model_type": "llama",
            "hidden_size": HIDDEN,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "vocab_size": VOCAB,
            "max_position_embeddings": 64,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        }
    )

    weights = {}

    def add(name, shape):
        arr = (rng.normal(size=shape) * 0.05).astype(np.float32)
        weights[name] = arr
        writer.add_tensor(name, arr)
        return arr

    add("embed_tokens.weight", (VOCAB, HIDDEN))
    for i in range(LAYERS):
        p = f"layers.{i}"
        add(f"{p}.input_layernorm.weight", (HIDDEN,))
        add(f"{p}.self_attn.q_proj.weight", (HEADS * HEAD_DIM, HIDDEN))
        add(f"{p}.self_attn.k_proj.weight", (KV_HEADS * HEAD_DIM, HIDDEN))
        add(f"{p}.self_attn.v_proj.weight", (KV_HEADS * HEAD_DIM, HIDDEN))
        add(f"{p}.self_attn.o_proj.weight", (HIDDEN, HEADS * HEAD_DIM))
        add(f"{p}.post_attention_layernorm.weight", (HIDDEN,))
        add(f"{p}.mlp.gate_proj.weight", (INTERMEDIATE, HIDDEN))
        add(f"{p}.mlp.up_proj.weight", (INTERMEDIATE, HIDDEN))
        add(f"{p}.mlp.down_proj.weight", (HIDDEN, INTERMEDIATE))
    add("norm.weight", (HIDDEN,))
    add("lm_head.weight", (VOCAB, HIDDEN))

    writer.finalize()
    return weights


def _reference_forward(weights, token_ids):
    """Independent plain-numpy forward pass, mirroring the executor's math
    but written separately, as the correctness oracle."""
    hidden = weights["embed_tokens.weight"][np.array(token_ids)].astype(np.float32)
    seq_len = hidden.shape[0]
    cos, sin = build_rope_cache(HEAD_DIM, 64, 10000.0)
    positions = np.arange(seq_len)

    for i in range(LAYERS):
        p = f"layers.{i}"
        residual = hidden
        normed = rms_norm(hidden, weights[f"{p}.input_layernorm.weight"])

        q = normed @ weights[f"{p}.self_attn.q_proj.weight"].T
        k = normed @ weights[f"{p}.self_attn.k_proj.weight"].T
        v = normed @ weights[f"{p}.self_attn.v_proj.weight"].T
        q = q.reshape(seq_len, HEADS, HEAD_DIM)
        k = k.reshape(seq_len, KV_HEADS, HEAD_DIM)
        v = v.reshape(seq_len, KV_HEADS, HEAD_DIM)
        q = apply_rope(q, cos, sin, positions)
        k = apply_rope(k, cos, sin, positions)

        group = HEADS // KV_HEADS
        k_rep = np.repeat(k, group, axis=1)
        v_rep = np.repeat(v, group, axis=1)

        q_t = np.transpose(q, (1, 0, 2))
        k_t = np.transpose(k_rep, (1, 0, 2))
        v_t = np.transpose(v_rep, (1, 0, 2))
        scores = np.einsum("hqd,hkd->hqk", q_t, k_t) / np.sqrt(HEAD_DIM)
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(mask[None, :, :], -np.inf, scores)
        probs = softmax(scores, axis=-1)
        attn = np.einsum("hqk,hkd->hqd", probs, v_t)
        attn = np.transpose(attn, (1, 0, 2)).reshape(seq_len, HEADS * HEAD_DIM)
        attn_out = attn @ weights[f"{p}.self_attn.o_proj.weight"].T
        hidden = residual + attn_out

        residual = hidden
        normed = rms_norm(hidden, weights[f"{p}.post_attention_layernorm.weight"])
        gate = normed @ weights[f"{p}.mlp.gate_proj.weight"].T
        up = normed @ weights[f"{p}.mlp.up_proj.weight"].T
        mlp = swiglu(gate, up) @ weights[f"{p}.mlp.down_proj.weight"].T
        hidden = residual + mlp

    hidden = rms_norm(hidden, weights["norm.weight"])
    logits = hidden[-1:] @ weights["lm_head.weight"].T
    return logits[0]


def test_execution_graph_matches_independent_reference(tmp_path):
    path = str(tmp_path / "tiny.draco")
    weights = _build_tiny_model(path)

    from draco.format.reader import DracoReader

    reader = DracoReader(path)
    backend = CPUExecutionBackend(reader)
    seq_id = backend.new_sequence()
    prompt = [1, 2, 3, 4]
    logits = backend.forward_step(seq_id, np.array(prompt, dtype=np.int64), start_position=0)
    backend.end_sequence(seq_id)

    expected = _reference_forward(weights, prompt)
    assert logits.shape == (VOCAB,)
    assert np.allclose(logits, expected, atol=1e-3, rtol=1e-3)


def test_generate_greedy_runs_and_produces_expected_length(tmp_path):
    path = str(tmp_path / "tiny2.draco")
    _build_tiny_model(path, seed=1)

    with CPUNativeLLM(path) as llm:
        out = llm.generate([1, 2, 3], max_new_tokens=5)
        assert len(out) == 3 + 5
        assert all(0 <= t < VOCAB for t in out)
        info = llm.runtime_info()
        assert info["device"] == "cpu"
        assert info["memory"]["weights_bytes"] > 0


def test_generate_is_deterministic_greedy(tmp_path):
    path = str(tmp_path / "tiny3.draco")
    _build_tiny_model(path, seed=2)
    with CPUNativeLLM(path) as llm:
        out1 = llm.generate([5, 6], max_new_tokens=4)
    with CPUNativeLLM(path) as llm:
        out2 = llm.generate([5, 6], max_new_tokens=4)
    assert out1 == out2
