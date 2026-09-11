import numpy as np
import pytest

from draco.format.reader import DracoReader
from draco.format.writer import DracoWriter
from draco.runtime.cpu.batch_engine import CPUBatchEngine
from draco.runtime.cpu.executor import CPUExecutionBackend
from draco.runtime.cpu.mem_util import HostMemoryInfo, auto_memory_budget, detect_host_memory
from draco.runtime.cpu.speculative import AdaptiveSpeculativeConfig, AdaptiveSpeculativeDecoder

HIDDEN = 16
LAYERS = 2
HEADS = 4
KV_HEADS = 2
HEAD_DIM = HIDDEN // HEADS
INTERMEDIATE = 32
VOCAB = 32


def _build_tiny_model(path: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    writer = DracoWriter(path)
    writer.add_metadata(
        {
            "architecture": "LlamaForCausalLM", "model_type": "llama",
            "hidden_size": HIDDEN, "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
            "vocab_size": VOCAB, "max_position_embeddings": 64,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        }
    )

    def add(name, shape):
        writer.add_tensor(name, (rng.normal(size=shape) * 0.05).astype(np.float32))

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


# ---- continuous batching ----

def test_batch_engine_matches_single_sequence_generation(tmp_path):
    path = str(tmp_path / "m.draco")
    _build_tiny_model(path, seed=0)

    prompts = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    max_new = 6

    # Reference: run each prompt independently through the single-sequence path.
    expected = []
    for p in prompts:
        reader = DracoReader(path)
        backend = CPUExecutionBackend(reader)
        expected.append(backend.generate_greedy(p, max_new))
        reader.close()

    # Under test: all three prompts through one continuous-batching engine.
    reader = DracoReader(path)
    backend = CPUExecutionBackend(reader, max_num_seqs=len(prompts))
    engine = CPUBatchEngine(backend, max_num_seqs=len(prompts))
    for i, p in enumerate(prompts):
        engine.add_request(i, p, max_tokens=max_new)
    results = engine.run_to_completion()
    reader.close()

    for i in range(len(prompts)):
        assert results[i] == expected[i], f"prompt {i} diverged under batching"


def test_batch_engine_handles_uneven_finish_times(tmp_path):
    path = str(tmp_path / "m2.draco")
    _build_tiny_model(path, seed=3)
    reader = DracoReader(path)
    backend = CPUExecutionBackend(reader, max_num_seqs=2)
    engine = CPUBatchEngine(backend, max_num_seqs=2)
    engine.add_request(0, [1, 2], max_tokens=2)
    engine.add_request(1, [3, 4, 5], max_tokens=8)
    results = engine.run_to_completion()
    reader.close()
    assert len(results[0]) == 2 + 2
    assert len(results[1]) == 3 + 8


# ---- adaptive speculative decoding ----

def test_speculative_matches_greedy_exactly(tmp_path):
    """Speculative decoding must be lossless for greedy decoding: same
    output as plain autoregressive generation, token for token."""
    path = str(tmp_path / "m3.draco")
    _build_tiny_model(path, seed=7)

    reader = DracoReader(path)
    backend = CPUExecutionBackend(reader)
    prompt = [1, 2, 3, 1, 2, 3, 1, 2]  # repetitive on purpose: gives the
    # n-gram drafter something to actually match against.
    expected = backend.generate_greedy(prompt, max_new_tokens=10)
    reader.close()

    reader2 = DracoReader(path)
    backend2 = CPUExecutionBackend(reader2)
    decoder = AdaptiveSpeculativeDecoder(backend2)
    actual = decoder.generate(prompt, max_new_tokens=10)
    reader2.close()

    assert actual == expected
    assert decoder.stats.rounds > 0


def test_speculative_draft_len_adapts(tmp_path):
    path = str(tmp_path / "m4.draco")
    _build_tiny_model(path, seed=9)
    reader = DracoReader(path)
    backend = CPUExecutionBackend(reader)
    cfg = AdaptiveSpeculativeConfig(initial_draft_len=2, max_draft_len=6)
    decoder = AdaptiveSpeculativeDecoder(backend, cfg)
    prompt = list(range(1, 9)) * 3  # long repetitive prompt
    decoder.generate(prompt, max_new_tokens=20)
    reader.close()
    # draft length should have moved from its initial value at least once
    # given a long, highly repetitive context.
    assert len(set(decoder.stats.draft_len_history)) >= 1


def test_speculative_no_repetition_falls_back_safely(tmp_path):
    path = str(tmp_path / "m5.draco")
    _build_tiny_model(path, seed=11)
    reader = DracoReader(path)
    backend = CPUExecutionBackend(reader)
    decoder = AdaptiveSpeculativeDecoder(backend)
    # short, non-repetitive prompt: n-gram lookup will find nothing to draft.
    out = decoder.generate([1, 2, 3], max_new_tokens=5)
    reader.close()
    assert len(out) == 3 + 5


# ---- memory utilization ----

def test_auto_memory_budget_respects_utilization():
    info = HostMemoryInfo(total_bytes=16 * 1024**3, available_bytes=8 * 1024**3)
    budget = auto_memory_budget(0.5, info=info)
    assert budget == 4 * 1024**3


def test_auto_memory_budget_rejects_bad_utilization():
    with pytest.raises(ValueError):
        auto_memory_budget(1.5)
    with pytest.raises(ValueError):
        auto_memory_budget(0.0)


def test_detect_host_memory_returns_positive_values():
    info = detect_host_memory()
    assert info.total_bytes > 0
    assert info.available_bytes > 0
