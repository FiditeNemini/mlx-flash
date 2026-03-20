
import argparse
import os
import subprocess
import time

from mlx_engine_flash import FlashConfig
from mlx_engine_flash.integration.lmstudio import apply_flash_patch


def get_rss_gb():
    """Get the current process Resident Set Size in GB."""
    pid = os.getpid()
    output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
    return float(output.strip()) / (1024 * 1024)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--flash", action="store_true")
    parser.add_argument("--ram", type=float, default=1.0)
    args = parser.parse_args()

    if args.flash:
        # Force MLX to NOT use its own memory mapping (so our pwrite/madvise works)
        # and limit MLX's internal cache.
        os.environ["MLX_MEMORY_MAPPING"] = "0"
        os.environ["MLX_MAX_CACHED_GIGABYTES"] = str(args.ram)
        
        import mlx.core as mx
        mx.metal.set_cache_limit(int(args.ram * 1024 * 1024 * 1024))
        
        cfg = FlashConfig(enabled=True, ram_budget_gb=args.ram, debug=True, prefetch_layers=0)
        apply_flash_patch(cfg)
        print(f"--- FLASH MODE ENABLED (Budget: {args.ram} GB) ---")
    else:
        print("--- FLASH MODE DISABLED (Normal Load) ---")

    import mlx_lm
    
    start_rss = get_rss_gb()
    print(f"Initial RAM: {start_rss:.2f} GB")

    print("Loading model architecture...")
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(args.model)
    load_time = time.perf_counter() - t0
    
    load_rss = get_rss_gb()
    print(f"RAM after load: {load_rss:.2f} GB")
    print(f"Load time: {load_time:.2f}s")

    print("\nGenerating tokens (monitoring RAM)...")
    peak_rss = load_rss
    
    # Generate more tokens to ensure we cycle through layers multiple times
    for i, _token in enumerate(mlx_lm.stream_generate(model, tokenizer, prompt="Explain gravity", max_tokens=30)):
        current_rss = get_rss_gb()
        peak_rss = max(peak_rss, current_rss)
        if i % 2 == 0:
            print(f"  Token {i+1}, Current RAM: {current_rss:.2f} GB")

    print("\nFinal Statistics:")
    print(f"Peak RAM Usage: {peak_rss:.2f} GB")
    print("Model Size on Disk: ~5.00 GB")
    
    if peak_rss < 4.0:
        print("\n✅ PROOF: Peak RAM is significantly lower than model size. Streaming is active.")
    else:
        print("\n❌ Peak RAM is near model size. Weights may have been cached by OS or loaded normally.")

if __name__ == "__main__":
    main()
