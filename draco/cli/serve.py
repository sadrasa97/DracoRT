"""
CLI Serve Command

Starts the Draco OpenAI-compatible server with configurable options.

Usage:
    draco serve --model meta-llama/Llama-3-8B --host 0.0.0.0 --port 8000
    draco serve --model model-path --dtype bfloat16 --max-model-len 32768
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from typing import List, Optional

logger = logging.getLogger("draco.cli.serve")


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the Draco inference server."""
    try:
        from draco import LLM
        from draco.server.app import DracoServer
        from draco.logging import setup_structured_logging
    except ImportError as e:
        print(f"Error importing Draco: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_structured_logging(level=log_level, json_output=args.json_logging)

    logger.info("Starting Draco server...")
    logger.info("Model: %s", args.model)
    logger.info("Host: %s:%d", args.host, args.port)

    # Initialize LLM
    print(f"Loading model: {args.model}")
    try:
        llm = LLM(
            model=args.model,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            quantization=args.quantization,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"Model loaded successfully: {llm.architecture}")
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    # Create server
    server = DracoServer(
        llm=llm,
        host=args.host,
        port=args.port,
        model_name=args.model_name or args.model,
    )

    print(f"Server ready: {server}")
    print(f"Endpoints:")
    print(f"  POST /v1/completions")
    print(f"  POST /v1/chat/completions")
    print(f"  GET  /v1/models")
    print(f"  GET  /health")
    print(f"  GET  /metrics")
    print()
    print(f"Listening on {args.host}:{args.port}")

    # Start simple HTTP server
    try:
        from http.server import HTTPServer
        from draco.server.app import DracoHTTPHandler

        httpd = HTTPServer((args.host, args.port), DracoHTTPHandler)
        httpd.draco_server = server

        # Graceful shutdown
        def shutdown_handler(signum, frame):
            print("\nShutting down gracefully...")
            httpd.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate a model configuration and adapter, or a .draco file."""
    if str(args.model).endswith(".draco"):
        from draco.format.validator import validate_file

        report = validate_file(args.model)
        status = "OK" if report.ok else "FAILED"
        print(f"Draco format validation: {status}")
        print(f"  Tensors checked: {report.num_tensors}")
        if report.checksum_verified:
            print("  Checksum:        verified")
        for w in report.warnings:
            print(f"  Warning: {w}")
        for e in report.errors:
            print(f"  Error:   {e}")
        if not report.ok:
            sys.exit(1)
        return
    try:
        from draco.models.config import ModelConfig
        from draco.models.registry import MODEL_REGISTRY
    except ImportError as e:
        print(f"Error importing Draco: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Validating model: {args.model}")
    print()

    # Load config
    try:
        config = ModelConfig.from_pretrained(args.model)
        print(f"Config loaded successfully:")
        print(f"  Architecture: {config.architecture}")
        print(f"  Model Type:   {config.model_type}")
        print(f"  Hidden Size:  {config.hidden_size}")
        print(f"  Num Layers:   {config.num_hidden_layers}")
        print(f"  Num Heads:    {config.num_attention_heads}")
        print(f"  KV Heads:     {config.num_key_value_heads}")
        print(f"  Head Dim:     {config.head_dim}")
        print(f"  Vocab Size:   {config.vocab_size}")
        print(f"  Max Positions: {config.max_position_embeddings}")
        if config.moe.num_experts > 0:
            print(f"  MoE:          {config.moe.num_experts} experts, top-{config.moe.num_experts_per_tok}")
        if config.is_quantized:
            print(f"  Quantization: {config.quantization.quantization_method}")
    except Exception as e:
        print(f"  Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    # Check registry
    arch = config.detect_architecture()
    print()
    if MODEL_REGISTRY.has(arch):
        meta = MODEL_REGISTRY.get_metadata(arch)
        impl = meta.get("implementation", "?")
        print(f"Registry: OK")
        print(f"  Architecture: {arch}")
        print(f"  Adapter:      {impl}")
        print(f"  Status:       {meta.get('status', 'unknown')}")
    else:
        print(f"Registry: NOT FOUND")
        print(f"  Architecture: {arch}")
        print(f"  Available:    {MODEL_REGISTRY.list_architectures()}")
        sys.exit(1)

    # Build adapter to verify
    try:
        adapter_cls = MODEL_REGISTRY.get(arch)
        adapter = adapter_cls(config)
        model = adapter.build_model()
        param_count = sum(p.numel() for p in model.parameters())
        print()
        print(f"Build: OK")
        print(f"  Parameters: {param_count:,}")
        print(f"  Memory (fp16): {param_count * 2 / (1024**3):.2f} GB")
        print()
        print("Validation passed!")
    except Exception as e:
        print(f"  Build failed: {e}", file=sys.stderr)
        sys.exit(1)


def register_serve_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the serve command parser."""
    serve_parser = subparsers.add_parser("serve", help="Start the inference server")
    serve_parser.add_argument("--model", required=True, help="Model name or path")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    serve_parser.add_argument("--model-name", default=None, help="Model name for API responses")
    serve_parser.add_argument("--dtype", default="auto", help="Data type")
    serve_parser.add_argument("--quantization", default=None, help="Quantization method")
    serve_parser.add_argument("--max-model-len", type=int, default=None, help="Max model length")
    serve_parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory utilization")
    serve_parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    serve_parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    serve_parser.add_argument("--json-logging", action="store_true", help="JSON log output")
    serve_parser.set_defaults(func=cmd_serve)


def register_validate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the validate command parser."""
    validate_parser = subparsers.add_parser("validate", help="Validate a model")
    validate_parser.add_argument("--model", required=True, help="Model name or path")
    validate_parser.set_defaults(func=cmd_validate)
