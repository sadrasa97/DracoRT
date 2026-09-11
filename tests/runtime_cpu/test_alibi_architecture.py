"""
Proves a third architecture family: ALiBi position encoding (BLOOM-style)
instead of RoPE or learned embeddings, with LayerNorm and a standard GELU
MLP. Cross-checked against an independent plain-numpy reference including
the ALiBi bias construction itself.
"""

import numpy as np

from draco.format.reader import DracoReader
from draco.format.writer import DracoWriter
from draco.runtime.cpu.executor import CPUExecutionBackend
from draco.runtime.cpu.ops import build_alibi_bias, gelu, layer_norm, softmax

HIDDEN = 16
LAYERS = 2
HEADS = 4
HEAD_DIM = HIDDEN // HEADS
INTERMEDIATE = 32
VOCAB = 32


def _build_alibi_model(path: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    writer = DracoWriter(path)
    writer.add_metadata(
        {
            "architecture": "BloomForCausalLM",
            "model_type": "bloom",
            "hidden_size": HIDDEN,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS,
            "num_key_value_heads": HEADS,
            "vocab_size": VOCAB,
            "max_position_embeddings": 64,
            "tie_word_embeddings": False,
            "norm_type": "layernorm",
            "norm_bias": True,
            "position_encoding_type": "alibi",
            "mlp_type": "standard",
            "hidden_act": "gelu",
            "qkv_fused": False,
            "attention_bias": False,
            "mlp_bias": False,
        }
    )

    weights = {}

    def add(name, shape, zero=False):
        arr = np.zeros(shape, dtype=np.float32) if zero else (rng.normal(size=shape) * 0.05).astype(np.float32)
        weights[name] = arr
        writer.add_tensor(name, arr)

    add("embed_tokens.weight", (VOCAB, HIDDEN))
    for i in range(LAYERS):
        p = f"layers.{i}"
        add(f"{p}.input_layernorm.weight", (HIDDEN,))
        add(f"{p}.input_layernorm.bias", (HIDDEN,), zero=True)
        add(f"{p}.self_attn.q_proj.weight", (HIDDEN, HIDDEN))
        add(f"{p}.self_attn.k_proj.weight", (HIDDEN, HIDDEN))
        add(f"{p}.self_attn.v_proj.weight", (HIDDEN, HIDDEN))
        add(f"{p}.self_attn.o_proj.weight", (HIDDEN, HIDDEN))
        add(f"{p}.post_attention_layernorm.weight", (HIDDEN,))
        add(f"{p}.post_attention_layernorm.bias", (HIDDEN,), zero=True)
        add(f"{p}.mlp.fc1.weight", (INTERMEDIATE, HIDDEN))
        add(f"{p}.mlp.fc2.weight", (HIDDEN, INTERMEDIATE))
    add("norm.weight", (HIDDEN,))
    add("norm.bias", (HIDDEN,), zero=True)
    add("lm_head.weight", (VOCAB, HIDDEN))

    writer.finalize()
    return weights


def _reference_forward(weights, token_ids):
    hidden = weights["embed_tokens.weight"][np.array(token_ids)].astype(np.float32)
    seq_len = hidden.shape[0]
    positions = np.arange(seq_len)
    alibi_bias = build_alibi_bias(HEADS, positions, positions)

    for i in range(LAYERS):
        p = f"layers.{i}"
        residual = hidden
        normed = layer_norm(hidden, weights[f"{p}.input_layernorm.weight"], weights[f"{p}.input_layernorm.bias"])

        q = normed @ weights[f"{p}.self_attn.q_proj.weight"].T
        k = normed @ weights[f"{p}.self_attn.k_proj.weight"].T
        v = normed @ weights[f"{p}.self_attn.v_proj.weight"].T
        q = q.reshape(seq_len, HEADS, HEAD_DIM)
        k = k.reshape(seq_len, HEADS, HEAD_DIM)
        v = v.reshape(seq_len, HEADS, HEAD_DIM)

        q_t = np.transpose(q, (1, 0, 2))
        k_t = np.transpose(k, (1, 0, 2))
        v_t = np.transpose(v, (1, 0, 2))
        scores = np.einsum("hqd,hkd->hqk", q_t, k_t) / np.sqrt(HEAD_DIM) + alibi_bias
        probs = softmax(scores, axis=-1)
        attn = np.einsum("hqk,hkd->hqd", probs, v_t)
        attn = np.transpose(attn, (1, 0, 2)).reshape(seq_len, HIDDEN)
        attn_out = attn @ weights[f"{p}.self_attn.o_proj.weight"].T
        hidden = residual + attn_out

        residual = hidden
        normed = layer_norm(
            hidden, weights[f"{p}.post_attention_layernorm.weight"], weights[f"{p}.post_attention_layernorm.bias"]
        )
        h1 = normed @ weights[f"{p}.mlp.fc1.weight"].T
        mlp_out = gelu(h1) @ weights[f"{p}.mlp.fc2.weight"].T
        hidden = residual + mlp_out

    hidden = layer_norm(hidden, weights["norm.weight"], weights["norm.bias"])
    logits = hidden[-1:] @ weights["lm_head.weight"].T
    return logits[0]


def test_alibi_architecture_matches_reference(tmp_path):
    path = str(tmp_path / "alibi.draco")
    weights = _build_alibi_model(path)

    reader = DracoReader(path)
    backend = CPUExecutionBackend(reader)
    seq_id = backend.new_sequence()
    prompt = [1, 2, 3, 4, 5, 6]
    logits = backend.forward_step(seq_id, np.array(prompt, dtype=np.int64), start_position=0)
    backend.end_sequence(seq_id)
    reader.close()

    expected = _reference_forward(weights, prompt)
    assert logits.shape == (VOCAB,)
    assert np.allclose(logits, expected, atol=1e-3, rtol=1e-3)


def test_alibi_batched_decode_matches_single_sequence(tmp_path):
    """ALiBi bias depends on absolute position — must stay correct when
    driven through the batched continuous-batching decode path too."""
    path = str(tmp_path / "alibi2.draco")
    _build_alibi_model(path, seed=2)

    reader = DracoReader(path)
    backend = CPUExecutionBackend(reader, max_num_seqs=1)
    single = backend.generate_greedy([1, 2, 3], max_new_tokens=4)
    reader.close()

    from draco.runtime.cpu.batch_engine import CPUBatchEngine

    reader2 = DracoReader(path)
    backend2 = CPUExecutionBackend(reader2, max_num_seqs=1)
    engine = CPUBatchEngine(backend2, max_num_seqs=1)
    engine.add_request(0, [1, 2, 3], max_tokens=4)
    results = engine.run_to_completion()
    reader2.close()

    assert results[0] == single
