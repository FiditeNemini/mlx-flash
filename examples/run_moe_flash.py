#!/usr/bin/env python3
"""
run_moe_flash.py — Flash Mode demo for MoE models (Mixtral, DeepSeek, etc.)

The MoE-specific path is exercised automatically when the model contains
num_experts in its config.json.  No special flags needed beyond --flash.

Usage:
    python examples/run_moe_flash.py \
        --model /path/to/Mixtral-8x7B-Instruct-v0.1-4bit \
        --flash --ram 12 --top-k 2
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--flash", action="store_true")
    parser.add_argument("--ram", type=float, default=12.0)
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override MoE top-K (default: model config)")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--prompt",
                        default="Explain the Mixture of Experts architecture in one paragraph.")
    args = parser.parse_args()

    if args.flash:
        from mlx_flash import FlashConfig
        from mlx_flash.integration.lmstudio import apply_flash_patch
        cfg = FlashConfig(
            enabled=True,
            ram_budget_gb=args.ram,
            moe_top_k_override=args.top_k,
            debug=True,
        )
        apply_flash_patch(cfg)

    import time

    import mlx_lm
    print(f"Loading MoE model: {args.model}")
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(args.model)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s\n")

    print(f"Prompt: {args.prompt}\n\nResponse: ", end="", flush=True)
    t1 = time.perf_counter()
    n = 0
    for tok in mlx_lm.stream_generate(model, tokenizer,
                                       prompt=args.prompt,
                                       max_tokens=args.max_tokens):
        print(tok, end="", flush=True)
        n += 1
    elapsed = time.perf_counter() - t1
    print(f"\n\n{n / elapsed:.1f} tok/s")


if __name__ == "__main__":
    main()
