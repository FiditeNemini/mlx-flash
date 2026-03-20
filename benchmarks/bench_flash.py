#!/usr/bin/env python3
"""
bench_flash.py — Compare Flash vs Normal loading and inference speed.

Usage:
    python benchmarks/bench_flash.py --model /path/to/model [--tokens 20]
    python benchmarks/bench_flash.py --model /path/to/model --mode flash
    python benchmarks/bench_flash.py --model /path/to/model --mode normal
    python benchmarks/bench_flash.py --model /path/to/model --mode both

Output:
    Markdown table suitable for pasting into README.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path


def bench_load(model_path: str, flash: bool, ram_budget: float = 10.0) -> dict:
    """Time the model load and return stats dict."""
    import tracemalloc

    if flash:
        from mlx_engine_flash import FlashConfig
        from mlx_engine_flash.integration.lmstudio import (
            apply_flash_patch,
            remove_flash_patch,
        )
        cfg = FlashConfig(enabled=True, ram_budget_gb=ram_budget, debug=True)
        apply_flash_patch(cfg)

    tracemalloc.start()
    t0 = time.perf_counter()

    try:
        import mlx_lm
        model, tokenizer = mlx_lm.load(model_path)
        load_s = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {"load_s": load_s, "peak_ram_mb": peak / 1e6, "error": None,
                "model": model, "tokenizer": tokenizer}
    except Exception as e:
        tracemalloc.stop()
        if flash:
            from mlx_engine_flash.integration.lmstudio import remove_flash_patch
            remove_flash_patch()
        return {"load_s": None, "peak_ram_mb": None, "error": str(e)}


def bench_generate(model, tokenizer, prompt: str, n_tokens: int) -> dict:
    """Run inference and measure throughput."""
    import mlx_lm
    t0 = time.perf_counter()
    tokens = []
    for tok in mlx_lm.stream_generate(model, tokenizer, prompt=prompt,
                                       max_tokens=n_tokens):
        tokens.append(tok)
    elapsed = time.perf_counter() - t0
    n_gen = len(tokens)
    return {
        "tokens": n_gen,
        "elapsed_s": elapsed,
        "tok_per_s": n_gen / elapsed if elapsed > 0 else 0,
    }


def print_table(results: list[dict]) -> None:
    print("\n| Mode   | Load (s) | Peak RAM | tok/s |")
    print("|--------|----------|----------|-------|")
    for r in results:
        mode  = r.get("mode", "?")
        load  = f"{r['load_s']:.1f}" if r.get("load_s") else "OOM"
        ram   = f"{r['peak_ram_mb']:.0f} MB" if r.get("peak_ram_mb") else "—"
        tps   = f"{r['tok_per_s']:.1f}" if r.get("tok_per_s") else "—"
        print(f"| {mode:<6} | {load:<8} | {ram:<8} | {tps:<5} |")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Flash vs Normal benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["flash", "normal", "both"], default="both")
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--ram-budget", type=float, default=10.0)
    parser.add_argument("--prompt", default="Explain quantum computing in one sentence.")
    args = parser.parse_args()

    print(f"Model: {Path(args.model).name}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Tokens: {args.tokens}")

    results = []
    modes = ["flash", "normal"] if args.mode == "both" else [args.mode]

    for mode in modes:
        print(f"\n--- {mode.upper()} ---")
        flash = mode == "flash"
        r = bench_load(args.model, flash=flash, ram_budget=args.ram_budget)
        r["mode"] = mode

        if r["error"]:
            print(f"  Load FAILED: {r['error']}")
            results.append(r)
            continue

        print(f"  Load: {r['load_s']:.2f}s  Peak RAM: {r['peak_ram_mb']:.0f} MB")
        gen = bench_generate(r["model"], r["tokenizer"], args.prompt, args.tokens)
        r.update(gen)
        print(f"  Gen:  {gen['tok_per_s']:.1f} tok/s  ({gen['tokens']} tokens)")

        if flash:
            from mlx_engine_flash.integration.lmstudio import remove_flash_patch
            remove_flash_patch()

        del r["model"], r["tokenizer"]
        gc.collect()
        results.append(r)

    print_table(results)


if __name__ == "__main__":
    cli()
