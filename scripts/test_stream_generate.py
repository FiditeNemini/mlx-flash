
import argparse
import os
import time

import mlx.core as mx
import psutil
from mlx_lm import stream_generate

from mlx_flash import FlashConfig
from mlx_flash.integration.lmstudio import apply_flash_patch


def get_rss_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-size", type=int, default=2000)
    args = parser.parse_args()

    print(f"Initial RAM: {get_rss_gb():.2f} GB")

    # Configure MLX memory management
    os.environ["MLX_MEMORY_MAPPING"] = "1" # Allow mmap
    
    # Enable debug mode for telemetry
    config = FlashConfig(enabled=True, ram_budget_gb=8.0, debug=True, prefetch_layers=0)
    apply_flash_patch(config)
    
    from mlx_lm import load
    print(f"Loading {args.model} via patched mlx_lm.load...")
    model, tokenizer = load(args.model)
    
    prompt = "This is a test to simulate long context. " * (args.prompt_size // 8)
    print(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")

    print("\n--- Starting stream_generate (Production Mode) ---")
    start_time = time.time()
    try:
        # Use stream_generate directly with the new safety patches in lmstudio.py
        for tokens_count, _ in enumerate(stream_generate(model, tokenizer, prompt, max_tokens=10), 1):
            if tokens_count == 1:
                print(f"Prompt prefill complete in {time.time() - start_time:.2f}s")
            
            rss, active, cache = 0.0, 0.0, 0.0
            try:
                rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
                active = mx.metal.get_active_memory() / 1e9
                cache = mx.metal.get_cache_memory() / 1e9
            except Exception:
                pass
            
            print(f"Token {tokens_count} | RSS: {rss:.2f}GB | Active: {active:.2f}GB | Cache: {cache:.2f}GB")
        
        print(f"\nGeneration complete in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"\n\nCaught exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
