import argparse
import time
import math
import mlx.core as mx
import mlx_lm
import numpy as np
from pathlib import Path

from mlx_flash import FlashConfig
from mlx_flash.engine.engine import FlashEngine

def calculate_perplexity(model, tokenizer, text, seq_len=512, is_synthetic=False):
    if is_synthetic:
        # Use random tokens for synthetic model as its tokenizer is dummy
        vocab_size = 1024 # Standard for our massive synthetic test model
        # Try to get actual vocab size if possible
        if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
            vocab_size = model.config.vocab_size
        
        # Set seed for reproducibility between runs
        np.random.seed(42)
        tokens = np.random.randint(0, vocab_size, (seq_len,)).tolist()
    else:
        tokens = tokenizer.encode(text)
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
    
    if len(tokens) <= 1:
        return float('inf')

    input_ids = mx.array(tokens)[None]
    
    # Standard cross-entropy loss calculation
    logits = model(input_ids)
    
    # Shift so that tokens predict the next token
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    
    # Flatten
    import mlx.nn as nn
    loss = nn.losses.cross_entropy(shift_logits, shift_labels)
    avg_loss = mx.mean(loss).item()
    
    return math.exp(avg_loss)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Perplexity of FlashEngine vs Standard MLX")
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-1B-Instruct-4bit-mlx", help="HuggingFace model ID or local path")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for PPL")
    parser.add_argument("--kv-bits", type=int, default=8, help="KV cache bits")
    args = parser.parse_args()

    print(f"[*] Loading model: {args.model}")
    
    if args.model == "synthetic":
        model_dir = Path("/tmp/mlx_ppl_synthetic")
        from benchmarks.run_synthetic_proof import create_massive_synthetic
        if not (model_dir / "model.safetensors").exists():
            create_massive_synthetic(model_dir, n_layers=4, hidden_dim=1024)
        model_path = str(model_dir)
    else:
        model_path = args.model

    model, tokenizer = mlx_lm.load(model_path)
    
    # Test text (WikiText-like snippet)
    test_text = (
        "The technological singularity—also, simply, the singularity—is a hypothetical point in time at which "
        "technological growth becomes uncontrollable and irreversible, resulting in unfathomable changes to human civilization. "
        "According to the most popular version of the singularity hypothesis, called intelligence explosion, an upgradable "
        "intelligent agent will eventually enter a 'runaway reaction' of self-improvement cycles, with each new and more "
        "intelligent generation appearing more and more rapidly, causing an 'explosion' in intelligence and resulting in "
        "a powerful superintelligence that qualitatively far surpasses all human intelligence."
    )

    is_synth = (args.model == "synthetic")

    # 1. Measure Standard MLX PPL
    print("\n[1] Measuring Standard MLX Perplexity...")
    t0 = time.perf_counter()
    std_ppl = calculate_perplexity(model, tokenizer, test_text, seq_len=args.seq_len, is_synthetic=is_synth)
    t1 = time.perf_counter()
    print(f"    PPL: {std_ppl:.4f} (Time: {t1-t0:.2f}s)")

    # 2. Measure FlashEngine PPL (Tiling Only)
    print(f"\n[2] Measuring FlashEngine Perplexity (Tiling Only)...")
    config = FlashConfig(
        enabled=True,
        tiled_execution=True,
        tile_size=512,
        pipelined_execution=True,
        kv_cache_quantized=False,
        debug=False
    )
    
    # We need to re-load lazily for FlashEngine to work properly with weights
    model_lazy, _ = mlx_lm.load(model_path, lazy=True)
    engine = FlashEngine(model_lazy, tokenizer, config)
    
    t0 = time.perf_counter()
    flash_ppl = calculate_perplexity(engine, tokenizer, test_text, seq_len=args.seq_len, is_synthetic=is_synth)
    t1 = time.perf_counter()
    print(f"    PPL: {flash_ppl:.4f} (Time: {t1-t0:.2f}s)")

    diff = abs(flash_ppl - std_ppl)
    print(f"\n[*] Perplexity Delta: {diff:.6f}")
    
    # Accept slightly higher delta for synthetic since weights are random and loss can be high
    threshold = 0.5 if is_synth else 0.1
    
    if diff < threshold:
        print(f"✅ SUCCESS: Perplexity is within acceptable bounds (< {threshold}).")
    else:
        print("❌ FAILURE: Perplexity drift detected!")

if __name__ == "__main__":
    main()
