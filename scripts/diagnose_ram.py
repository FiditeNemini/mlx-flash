#!/usr/bin/env python3
import argparse
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_config

from mlx_flash.diagnostics import RAMProfiler
from mlx_flash.manager import FlashConfig, FlashManager


def estimate_layer_size(model_path: str) -> float:
    path = Path(model_path)
    # Recursively find all weight files
    total_bytes = sum(f.stat().st_size for f in path.glob("**/*.safetensors"))
    if total_bytes == 0:
        total_bytes = sum(f.stat().st_size for f in path.glob("**/*.npz"))
    
    config = load_config(path)
    n_layers = config.get("num_hidden_layers") or config.get("n_layers") or config.get("num_layers")
    if not n_layers:
        # Try to infer from architecture
        model_type = config.get("model_type", "").lower()
        if "llama" in model_type or "mistral" in model_type:
            n_layers = 32
        elif "gemma" in model_type:
            n_layers = 28
        elif "phi3" in model_type:
            n_layers = 32
        else:
            n_layers = 32 # Fallback
        
    return (total_bytes / n_layers) / 1e9

def main():
    parser = argparse.ArgumentParser(description="MLX Flash RAM Diagnostics")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--flash", action="store_true", default=True, help="Use Flash mode (default: True)")
    args = parser.parse_args()

    model_path = args.model
    if not Path(model_path).exists():
        print(f"[!] Model path {model_path} does not exist.")
        return

    layer_size_gb = estimate_layer_size(model_path)
    print(f"[*] Estimated single layer size: {layer_size_gb:.2f} GB")

    profiler = RAMProfiler()
    
    # 1. Initialize Flash Manager
    config = FlashConfig(enabled=args.flash)
    manager = FlashManager(config)
    
    # 2. Load model (skeleton + permanent weights)
    print("[*] Loading model...")
    profiler.snapshot("Before Load")
    model, tokenizer = manager.load(model_path)
    profiler.snapshot("After Load")
    
    # 3. Patch layers for profiling
    backbone = getattr(model, "model", getattr(model, "backbone", model))
    layers = backbone.layers
    
    def make_profiled_call(original_call, idx):
        def _profiled_call(*args, **kwargs):
            with profiler.layer_context(idx):
                return original_call(*args, **kwargs)
        return _profiled_call

    for i in range(len(layers)):
        layers[i].__call__ = make_profiled_call(layers[i].__call__, i)

    # 4. Run single forward pass
    print("[*] Running single forward pass...")
    prompt = "Hello"
    tokens = mx.array(tokenizer.encode(prompt))[None]
    
    profiler.snapshot("Forward Start")
    output = model(tokens)
    mx.eval(output)
    mx.synchronize()
    profiler.snapshot("Forward End")

    # 5. Report
    print("\n" + "="*90)
    print(" RAM DIAGNOSTICS REPORT")
    print("="*90)
    print(profiler.report())
    
    # 6. Peak Analysis
    peak_rss_mb = max(s["rss_mb"] for s in profiler.snapshots)
    peak_rss_gb = peak_rss_mb / 1024.0
    
    activation_overhead_gb = 0.5 
    expected_peak_gb = layer_size_gb + activation_overhead_gb
    
    print(f"\nTotal Peak RSS: {peak_rss_gb:.2f} GB")
    print(f"Expected Peak: ~{expected_peak_gb:.2f} GB (layer={layer_size_gb:.2f}GB + overhead={activation_overhead_gb:.2f}GB)")
    
    if peak_rss_gb > 2 * layer_size_gb:
        print("\n" + "!"*80)
        print(" WARNING: Peak RAM is significantly higher than expected (> 2x Layer Size)!")
        print(" Possible causes:")
        print(" - \"WARNING: Layers are not being zeroed after eval\"")
        print(" - \"WARNING: mx.metal.clear_cache() may not have been called\"")
        print(" - \"WARNING: mmap slices may have been .copy()'d before release\"")
        print(" - \"WARNING: mlx_lm.load() was called before FlashManager could intercept\"")
        print("!"*80)

    manager.shutdown()

if __name__ == "__main__":
    main()
