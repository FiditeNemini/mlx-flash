#!/usr/bin/env python3
"""
quick_start.py — Minimal Flash Mode demo.

Usage:
    python examples/quick_start.py --model /path/to/model
    python examples/quick_start.py --model /path/to/model --flash --ram 8
    python examples/quick_start.py --model /path/to/model --benchmark
"""

import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to mlx model")
    parser.add_argument("--flash", action="store_true", help="Enable Flash Mode")
    parser.add_argument("--ram", type=float, default=10.0, help="RAM budget (GB)")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--prompt", default="Hello! Tell me something interesting.")
    args = parser.parse_args()

    if args.flash:
        from mlx_flash import FlashConfig
        from mlx_flash.integration.lmstudio import apply_flash_patch
        cfg = FlashConfig(enabled=True, ram_budget_gb=args.ram, debug=True)
        apply_flash_patch(cfg)
        print(f"[demo] Flash Mode ON  (RAM budget: {args.ram} GB)")
    else:
        print("[demo] Flash Mode OFF (normal loading)")

    import mlx_lm

    print(f"[demo] Loading {args.model} ...")
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(args.model)
    load_s = time.perf_counter() - t0
    print(f"[demo] Loaded in {load_s:.2f}s")

    print(f"\nPrompt: {args.prompt}\nResponse: ", end="", flush=True)
    t1 = time.perf_counter()
    tokens = 0
    for token in mlx_lm.stream_generate(model, tokenizer,
                                         prompt=args.prompt,
                                         max_tokens=args.max_tokens):
        print(token, end="", flush=True)
        tokens += 1
    elapsed = time.perf_counter() - t1
    print()

    if args.benchmark:
        print("\n--- Benchmark ---")
        print(f"Load:   {load_s:.2f}s")
        print(f"Gen:    {tokens / elapsed:.1f} tok/s  ({tokens} tokens, {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
