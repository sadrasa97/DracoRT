"""
Proves the executor genuinely supports a second architecture family, not
just Llama-style: LayerNorm (not RMSNorm), learned position embeddings
(not RoPE), a fused c_attn QKV projection (not separate q/k/v), and a
standard (non-gated) GELU MLP (not SwiGLU) — the GPT-2 family shape.

Cross-checked against an independently-written plain-numpy reference,
same as the Llama-style test.
"""

import numpy as np

from draco.format.reader import DracoReader
from draco.format.writer import DracoWriter
from draco.runtime.cpu.executor import CPUExecutionBackend
from draco.runtime.cpu.ops import gelu, layer_norm, softmax

HIDDEN = 16
LAYERS = 2
HEADS = 4
HEAD_DIM = HIDDEN // HEADS
INTERMEDIATE = 32
VOCAB = 32
MAX_POS = 32


def _build_gpt2_style_model(path: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    writer = DracoWriter(path)
    writer.add_metadata(
        {
            "architecture": "GPT2LMHeadModel",
            "model_type": "gpt2",
            "hidden_size": HIDDEN,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS,
            "num_key_value_heads": HEADS,  # MHA, no GQA
            "vocab_size": VOCAB,
            "max_position_embeddings": MAX_POS,
            "tie_word_embeddings": True,
            "norm_type": "layernorm",
            "norm_bias": True,
            "position_encoding_type": "learned",
            "mlp_type": "standard",
            "hidden_act": "gelu",
            "qkv_fused": True,
            "attention_bias": True,
            "mlp_bias": True,
        }
    )

    weights = {}

    def add(name, shape, zero=False):
        arr = np.zeros(shape, dtype=np.float32) if zero else (rng.normal(size=shape) * 0.05).astype(np.float32)
        weights[name] = arr
        writer.add_tensor(name, arr)

    add("wte.weight", (VOCAB, HIDDEN))
    add("wpe.weight", (MAX_POS, HIDDEN))
    for i in range(LAYERS):
        p = f"h.{i}"
        add(f"{p}.ln_1.weight", (HIDDEN,))
        add(f"{p}.ln_1.bias", (HIDDEN,), zero=True)
        add(f"{p}.attn.c_attn.weight", (3 * HIDDEN, HIDDEN))
        add(f"{p}.attn.c_attn.bias", (3 * HIDDEN,), zero=True)
        add(f"{p}.attn.c_proj.weight", (HIDDEN, HIDDEN))
        add(f"{p}.attn.c_proj.bias", (HIDDEN,), zero=True)
        add(f"{p}.ln_2.weight", (HIDDEN,))
        add(f"{p}.ln_2.bias", (HIDDEN,), zero=True)
        add(f"{p}.mlp.c_fc.weight", (INTERMEDIATE, HIDDEN))
        add(f"{p}.mlp.c_fc.bias", (INTERMEDIATE,), zero=True)
        add(f"{p}.mlp.c_proj.weight", (HIDDEN, INTERMEDIATE))
        add(f"{p}.mlp.c_proj.bias", (HIDDEN,), zero=True)
    add("ln_f.weight", (HIDDEN,))
    add("ln_f.bias", (HIDDEN,), zero=True)

    writer.finalize()
    return weights


def _reference_forward(weights, token_ids):
    hidden = weights["wte.weight"][np.array(token_ids)].astype(np.float32)
    seq_len = hidden.shape[0]
    positions = np.arange(seq_len)
    hidden = hidden + weights["wpe.weight"][positions]

    for i in range(LAYERS):
        p = f"h.{i}"
        residual = hidden
        normed = layer_norm(hidden, weights[f"{p}.ln_1.weight"], weights[f"{p}.ln_1.bias"])

        qkv = normed @ weights[f"{p}.attn.c_attn.weight"].T + weights[f"{p}.attn.c_attn.bias"]
        q, k, v = qkv[:, :HIDDEN], qkv[:, HIDDEN : 2 * HIDDEN], qkv[:, 2 * HIDDEN :]
        q = q.reshape(seq_len, HEADS, HEAD_DIM)
        k = k.reshape(seq_len, HEADS, HEAD_DIM)
        v = v.reshape(seq_len, HEADS, HEAD_DIM)

        q_t = np.transpose(q, (1, 0, 2))
        k_t = np.transpose(k, (1, 0, 2))
        v_t = np.transpose(v, (1, 0, 2))
        scores = np.einsum("hqd,hkd->hqk", q_t, k_t) / np.sqrt(HEAD_DIM)
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(mask[None, :, :], -np.inf, scores)
        probs = softmax(scores, axis=-1)
        attn = np.einsum("hqk,hkd->hqd", probs, v_t)
        attn = np.transpose(attn, (1, 0, 2)).reshape(seq_len, HIDDEN)
        attn_out = attn @ weights[f"{p}.attn.c_proj.weight"].T + weights[f"{p}.attn.c_proj.bias"]
        hidden = residual + attn_out

        residual = hidden
        normed = layer_norm(hidden, weights[f"{p}.ln_2.weight"], weights[f"{p}.ln_2.bias"])
        h1 = normed @ weights[f"{p}.mlp.c_fc.weight"].T + weights[f"{p}.mlp.c_fc.bias"]
        act = gelu(h1)
        mlp_out = act @ weights[f"{p}.mlp.c_proj.weight"].T + weights[f"{p}.mlp.c_proj.bias"]
        hidden = residual + mlp_out

    hidden = layer_norm(hidden, weights["ln_f.weight"], weights["ln_f.bias"])
    logits = hidden[-1:] @ weights["wte.weight"].T  # tied embeddings
    return logits[0]


def test_gpt2_style_architecture_matches_reference(tmp_path):
    path = str(tmp_path / "gpt2_style.draco")
    weights = _build_gpt2_style_model(path)

    reader = DracoReader(path)
    backend = CPUExecutionBackend(reader)
    seq_id = backend.new_sequence()
    prompt = [1, 2, 3, 4, 5]
    logits = backend.forward_step(seq_id, np.array(prompt, dtype=np.int64), start_position=0)
    backend.end_sequence(seq_id)
    reader.close()

    expected = _reference_forward(weights, prompt)
    assert logits.shape == (VOCAB,)
    assert np.allclose(logits, expected, atol=1e-3, rtol=1e-3)


def test_gpt2_style_generation_runs(tmp_path):
    path = str(tmp_path / "gpt2_style2.draco")
    _build_gpt2_style_model(path, seed=5)
    reader = DracoReader(path)
    backend = CPUExecutionBackend(reader)
    out = backend.generate_greedy([1, 2, 3], max_new_tokens=5)
    reader.close()
    assert len(out) == 3 + 5
    assert all(0 <= t < VOCAB for t in out)
