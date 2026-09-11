"""
Draco CLI — Main Entry Point

Usage:
    draco bench --model MODEL --input-tokens 2048 --output-tokens 256 --batch-size 8
    draco bench compare --backend draco,vllm --model MODEL
    draco info --model MODEL
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def cmd_bench(args: argparse.Namespace) -> None:
    """Run a benchmark."""
    try:
        from draco import LLM, SamplingParams
        from draco.benchmarks.framework import BenchmarkRunner, EnvironmentInfo
        from draco.benchmarks.modes import (
            SingleRequestBenchmark,
            BatchScalingBenchmark,
            ContextScalingBenchmark,
            GenerationScalingBenchmark,
        )
    except ImportError as e:
        print(f"Error importing Draco: {e}", file=sys.stderr)
        sys.exit(1)

    env = EnvironmentInfo.collect()
    runner = BenchmarkRunner(output_dir=args.output_dir, save_results=True)

    print(f"Draco Benchmark")
    print(f"{'─' * 44}")
    print(f"  Model:      {args.model}")
    print(f"  dtype:      {args.dtype}")
    print(f"  Batch:      {args.batch_size}")
    print(f"  Input:      {args.input_tokens} tokens")
    print(f"  Output:     {args.output_tokens} tokens")
    if env.gpu_model:
        print(f"  GPU:        {env.gpu_model}")
        print(f"  Memory:     {env.gpu_memory_gb:.1f} GB")
    print()

    # Initialize LLM
    print("Loading model...")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Run specified benchmark mode
    mode = args.mode

    if mode == "single":
        bench = SingleRequestBenchmark(runner)
        result = bench.run(
            llm,
            max_tokens=args.output_tokens,
            input_tokens=args.input_tokens,
        )
        print(result.format_table())

    elif mode == "batch_scaling":
        bench = BatchScalingBenchmark(runner)
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
        results = bench.run(llm, batch_sizes=batch_sizes,
                          input_tokens=args.input_tokens, output_tokens=args.output_tokens)
        print(f"\n{'Batch Scaling Results':─^44}")
        print(f"  {'Batch':>6s}  {'TTFT (ms)':>10s}  {'TPOT (ms)':>10s}  {'tok/s':>10s}  {'Mem (GB)':>10s}")
        for r in results:
            print(f"  {r.batch_size:>6d}  {r.ttft_ms:>10.1f}  {r.tpot_ms:>10.1f}  {r.tokens_per_second:>10.1f}  {r.gpu_memory_gb:>10.2f}")

    elif mode == "context_scaling":
        bench = ContextScalingBenchmark(runner)
        ctx_lengths = [int(x) for x in args.context_lengths.split(",")]
        results = bench.run(llm, context_lengths=ctx_lengths, output_tokens=args.output_tokens)
        print(f"\n{'Context Scaling Results':─^44}")
        print(f"  {'Context':>8s}  {'TTFT (ms)':>10s}  {'TPOT (ms)':>10s}  {'tok/s':>10s}")
        for r in results:
            print(f"  {r.input_tokens:>8d}  {r.ttft_ms:>10.1f}  {r.tpot_ms:>10.1f}  {r.tokens_per_second:>10.1f}")

    elif mode == "generation_scaling":
        bench = GenerationScalingBenchmark(runner)
        gen_lengths = [int(x) for x in args.generation_lengths.split(",")]
        results = bench.run(llm, generation_lengths=gen_lengths, input_tokens=args.input_tokens)
        print(f"\n{'Generation Scaling Results':─^44}")
        print(f"  {'Gen Len':>8s}  {'TTFT (ms)':>10s}  {'TPOT (ms)':>10s}  {'tok/s':>10s}")
        for r in results:
            print(f"  {r.output_tokens:>8d}  {r.ttft_ms:>10.1f}  {r.tpot_ms:>10.1f}  {r.tokens_per_second:>10.1f}")

    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)


def cmd_bench_compare(args: argparse.Namespace) -> None:
    """Compare Draco against other backends."""
    try:
        from draco.benchmarks.framework import BenchmarkRunner
        from draco.benchmarks.comparison import ComparisonBenchmark
    except ImportError as e:
        print(f"Error importing: {e}", file=sys.stderr)
        sys.exit(1)

    runner = BenchmarkRunner(output_dir=args.output_dir, save_results=True)
    backends = args.backend.split(",")

    print(f"{'Backend Comparison':─^44}")
    print(f"  Backends:   {', '.join(backends)}")
    print(f"  Model:      {args.model}")
    print(f"  Input:      {args.input_tokens} tokens")
    print(f"  Output:     {args.output_tokens} tokens")
    print()

    comparator = ComparisonBenchmark(runner)
    comparison = comparator.compare(
        backends=backends,
        model=args.model,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        dtype=args.dtype,
        num_iterations=args.num_iterations,
    )
    print(comparison)


def cmd_info(args: argparse.Namespace) -> None:
    """Show model information."""
    try:
        from draco import LLM
        from draco.models.config import ModelConfig
        from draco.models.registry import MODEL_REGISTRY
    except ImportError as e:
        print(f"Error importing: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"{'Draco Model Info':─^44}")
    print()

    # Config info
    try:
        config = ModelConfig.from_pretrained(args.model)
        print(f"  Architecture:   {config.architecture}")
        print(f"  Model Type:     {config.model_type}")
        print(f"  Hidden Size:    {config.hidden_size}")
        print(f"  Num Layers:     {config.num_hidden_layers}")
        print(f"  Num Heads:      {config.num_attention_heads}")
        print(f"  KV Heads:       {config.num_key_value_heads}")
        print(f"  Head Dim:       {config.head_dim}")
        print(f"  Vocab Size:     {config.vocab_size}")
        print(f"  Max Positions:  {config.max_position_embeddings}")
        print(f"  Intermediate:   {config.intermediate_size}")
        print(f"  Tie Embeddings: {config.tie_word_embeddings}")
        print(f"  Act Function:   {config.hidden_act}")
        print(f"  RoPE theta:     {config.rope_theta}")
        if config.attention.sliding_window:
            print(f"  Sliding Window: {config.attention.sliding_window}")
        if config.is_quantized:
            print(f"  Quantization:   {config.quantization}")
        if config.moe.num_experts > 0:
            print(f"  MoE:            {config.moe.num_experts} experts, top-{config.moe.num_experts_per_tok}")

        # Registry status
        arch = config.detect_architecture()
        if MODEL_REGISTRY.has(arch):
            print(f"\n  Registry:       ✅ {arch} registered")
        else:
            print(f"\n  Registry:       ❌ {arch} NOT registered")
            print(f"  Available:      {MODEL_REGISTRY.list_architectures()}")
    except Exception as e:
        print(f"  Error loading config: {e}", file=sys.stderr)

    # Registered architectures
    print(f"\n{'Registered Architectures':─^44}")
    for arch in MODEL_REGISTRY.list_architectures():
        meta = MODEL_REGISTRY.get_metadata(arch)
        impl = meta.get("implementation", "?")
        print(f"  {arch:40s} → {impl}")


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="draco",
        description="Draco — Universal GPU LLM Runtime",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # bench command
    bench_parser = subparsers.add_parser("bench", help="Run benchmarks")
    bench_parser.add_argument("--model", required=True, help="Model name or path")
    bench_parser.add_argument("--mode", default="single",
                             choices=["single", "batch_scaling", "context_scaling", "generation_scaling"],
                             help="Benchmark mode")
    bench_parser.add_argument("--input-tokens", type=int, default=512, help="Input token count")
    bench_parser.add_argument("--output-tokens", type=int, default=128, help="Output token count")
    bench_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    bench_parser.add_argument("--dtype", default="auto", help="Data type (auto, float16, bfloat16)")
    bench_parser.add_argument("--quantization", default=None, help="Quantization method")
    bench_parser.add_argument("--max-model-len", type=int, default=None, help="Max model length")
    bench_parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory utilization")
    bench_parser.add_argument("--output-dir", default="benchmarks/results", help="Results output directory")
    bench_parser.add_argument("--batch-sizes", default="1,2,4,8,16", help="Batch sizes for batch_scaling mode")
    bench_parser.add_argument("--context-lengths", default="512,1024,2048,4096,8192", help="Context lengths")
    bench_parser.add_argument("--generation-lengths", default="16,32,64,128,256,512", help="Generation lengths")
    bench_parser.set_defaults(func=cmd_bench)

    # bench compare command
    compare_parser = subparsers.add_parser("bench-compare", help="Compare backends")
    compare_parser.add_argument("--backend", required=True, help="Comma-separated backend names")
    compare_parser.add_argument("--model", required=True, help="Model name or path")
    compare_parser.add_argument("--input-tokens", type=int, default=512, help="Input token count")
    compare_parser.add_argument("--output-tokens", type=int, default=128, help="Output token count")
    compare_parser.add_argument("--dtype", default="auto", help="Data type")
    compare_parser.add_argument("--num-iterations", type=int, default=3, help="Number of iterations")
    compare_parser.add_argument("--output-dir", default="benchmarks/results", help="Results directory")
    compare_parser.set_defaults(func=cmd_bench_compare)

    # info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("--model", required=True, help="Model name or path")
    info_parser.set_defaults(func=cmd_info)

    # serve command
    from draco.cli.serve import register_serve_parser, register_validate_parser
    register_serve_parser(subparsers)
    register_validate_parser(subparsers)

    # native CPU runtime commands: inspect, system-info, convert
    from draco.cli.native import register_native_parsers
    register_native_parsers(subparsers)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
