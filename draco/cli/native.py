"""
CLI commands for the native .draco CPU runtime: inspect, system-info,
and convert (HuggingFace/safetensors -> .draco).
"""

from __future__ import annotations

import argparse
import sys


def cmd_inspect(args: argparse.Namespace) -> None:
    """draco inspect model.draco"""
    from draco.format.reader import DracoReader
    from draco.runtime.cpu.capabilities import detect
    from draco.runtime.cpu.dispatcher import registered_tiers

    from draco.format.gguf import GGUFReader, is_gguf_file

    try:
        if is_gguf_file(args.model):
            reader = GGUFReader(args.model)
        else:
            reader = DracoReader(args.model)
    except Exception as e:
        print(f"Error opening '{args.model}': {e}", file=sys.stderr)
        sys.exit(1)

    with reader:
        total_bytes = sum(t.nbytes for t in reader.tensors())
        quant_counts = {}
        for t in reader.tensors():
            key = t.quantization or "none"
            quant_counts[key] = quant_counts.get(key, 0) + 1

        print("Draco Model")
        print("=" * 12)
        print()
        if hasattr(reader, "header"):
            print(f"Format:\n    Draco v{reader.header.version}")
        else:
            print(f"Format:\n    GGUF (llama.cpp) v{reader.header.version}")
        print()
        print(f"Architecture:\n    {reader.architecture}")
        print()
        print(f"Tensors:\n    {len(reader.tensor_names())}")
        for q, n in sorted(quant_counts.items()):
            print(f"      {q}: {n}")
        print()
        print(f"Weight memory (stored, on-disk):\n    {total_bytes / (1024**3):.2f} GB")
        print()
        print(f"Tokenizer:\n    {'embedded' if reader.tokenizer is not None else 'not embedded'}")
        print()

        caps = detect()
        tiers = registered_tiers()
        print("CPU:")
        print(f"    {caps.vendor} — {caps.model_name}")
        print(f"    physical cores: {caps.physical_cores}, logical cores: {caps.logical_cores}")
        for tier in ["generic", "avx2", "avx512", "avx512_vnni", "amx", "neon"]:
            has_hw = tier == caps.best_kernel_tier or (
                tier == "generic"
            )
            implemented = tier in tiers
            status = "available" if implemented else "not implemented in this build"
            if tier == "generic" or tier == caps.best_kernel_tier:
                print(f"    {tier} kernel: {status}")
        print()
        print(f"Recommended runtime:\n    CPU, {min(caps.physical_cores, 16)} threads")


def cmd_system_info(args: argparse.Namespace) -> None:
    """draco system-info"""
    from draco.runtime.cpu.capabilities import detect
    from draco.runtime.cpu.dispatcher import registered_tiers, select_kernel

    caps = detect()
    selection = select_kernel(caps)

    print("Draco CPU System Info")
    print("=" * 22)
    print(f"  Architecture:     {caps.architecture}")
    print(f"  Vendor:           {caps.vendor}")
    print(f"  Model:            {caps.model_name}")
    print(f"  Physical cores:   {caps.physical_cores}")
    print(f"  Logical cores:    {caps.logical_cores}")
    print(f"  AVX2:             {caps.avx2}")
    print(f"  AVX512F:          {caps.avx512f}")
    print(f"  AVX512 VNNI:      {caps.avx512vnni}")
    print(f"  AMX INT8:         {caps.amx_int8}")
    print(f"  AMX BF16:         {caps.amx_bf16}")
    print(f"  NEON:             {caps.neon}")
    print()
    print(f"  Best detected ISA tier: {selection.requested_tier}")
    print(f"  Implemented kernel tiers in this build: {registered_tiers()}")
    print(f"  Kernel that would be used: {selection.actual_tier}"
          f"{' (downgraded — no kernel for the detected tier)' if selection.downgraded else ''}")

    from draco.runtime.cpu.threading import plan_threads
    from draco.runtime.device_manager import resolve_device

    threads = plan_threads(caps)
    print(f"  Thread plan:      {threads.num_threads} threads (source: {threads.source})")

    device = resolve_device("auto")
    print(f"  device='auto' would resolve to: {device.device} ({device.reason})")


def cmd_convert(args: argparse.Namespace) -> None:
    """draco convert --model <hf-model-or-path|model.gguf> --output out.draco --quantization ..."""
    from draco.convert.converter import convert_model

    try:
        path = convert_model(
            model=args.model,
            output=args.output,
            quantization=args.quantization,
            verify=not args.no_verify,
        )
    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {path}")


def register_native_parsers(subparsers: argparse._SubParsersAction) -> None:
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a .draco model file")
    inspect_parser.add_argument("model", help="Path to a .draco file")
    inspect_parser.set_defaults(func=cmd_inspect)

    sysinfo_parser = subparsers.add_parser("system-info", help="Show CPU capability/kernel info")
    sysinfo_parser.set_defaults(func=cmd_system_info)

    convert_parser = subparsers.add_parser("convert", help="Convert a HF/safetensors model to .draco")
    convert_parser.add_argument("--model", required=True, help="HF model id or local path")
    convert_parser.add_argument("--output", required=True, help="Output .draco path")
    convert_parser.add_argument(
        "--quantization",
        default="none",
        choices=["none", "int8", "int4"],
        help="CPU-native quantization to apply (only implemented formats are offered)",
    )
    convert_parser.add_argument(
        "--no-verify", action="store_true", help="Skip numerical verification after conversion"
    )
    convert_parser.set_defaults(func=cmd_convert)
