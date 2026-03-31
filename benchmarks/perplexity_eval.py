import argparse
import time
import math
import mlx.core as mx
import mlx_lm
import numpy as np
from pathlib import Path

from mlx_flash import FlashConfig
from mlx_flash.engine.engine import FlashEngine

def calculate_loss(model, tokenizer, data, seq_len=512, is_synthetic=False, label=""):
    if is_synthetic:
        tokens = data
    else:
        tokens = tokenizer.encode(data)
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
    
    if len(tokens) <= 1:
        return float('inf')

    input_ids = mx.array(tokens)[None]
    
    # Forward Pass
    if isinstance(model, FlashEngine):
        logits = model(input_ids)
    else:
        # Standard MLX Path
        inner = getattr(model, "model", getattr(model, "backbone", model))
        embed = getattr(inner, "embed_tokens", getattr(inner, "wte", getattr(inner, "embeddings", None)))
        x = embed(input_ids)
        
        # Causal Mask
        L = x.shape[1]
        mask = mx.triu(mx.full((L, L), -mx.inf, dtype=x.dtype), k=1)
        
        layers = inner.layers if hasattr(inner, "layers") else []
        for layer in layers:
            x = layer(x, mask=mask)
            
        norm = getattr(inner, "norm", getattr(inner, "ln_f", None))
        if norm:
            x = norm(x)
            
        head = getattr(model, "lm_head", None)
        if head:
            logits = head(x)
        else:
            logits = x

    # Shift so that tokens predict the next token
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    
    # Flatten
    import mlx.nn as nn
    loss = nn.losses.cross_entropy(shift_logits, shift_labels)
    avg_loss = mx.mean(loss).item()
    
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Evaluate Accuracy of FlashEngine vs Standard MLX")
    parser.add_argument("--model", type=str, default="synthetic", help="HuggingFace model ID or local path")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for eval")
    args = parser.parse_args()

    if args.model == "synthetic":
        model_dir = Path("test_models/mlx_ppl_synthetic")
        from benchmarks.run_synthetic_proof import create_massive_synthetic
        if not (model_dir / "model.safetensors").exists():
            create_massive_synthetic(model_dir, n_layers=4, hidden_dim=1024)
        model_path = str(model_dir)
    else:
        model_path = args.model

    print(f"[*] Loading model: {model_path}")
    model, tokenizer = mlx_lm.load(model_path)
    
    is_synth = (args.model == "synthetic")
    if is_synth:
        np.random.seed(42)
        vocab_size = getattr(model, "vocab_size", 1024)
        common_tokens = np.random.randint(0, vocab_size, (args.seq_len,)).tolist()
        eval_data = common_tokens
    else:
        eval_data = "The technological singularity—also, simply, the singularity—is a hypothetical point in time."

    # 1. Measure Standard MLX Loss
    print("\n[1] Measuring Standard MLX Loss...")
    t0 = time.perf_counter()
    std_loss = calculate_loss(model, tokenizer, eval_data, seq_len=args.seq_len, is_synthetic=is_synth, label="STD")
    t1 = time.perf_counter()
    print(f"    Loss: {std_loss:.10f} (Time: {t1-t0:.2f}s)")

    # 2. Measure FlashEngine Loss
    print(f"\n[2] Measuring FlashEngine Loss (Full Stack)...")
    config = FlashConfig(
        enabled=True,
        tiled_execution=True,
        tile_size=1024,
        pipelined_execution=True, 
        kv_cache_quantized=True,
        kv_cache_bits=8,
        debug=False
    )
    
    model_lazy, _ = mlx_lm.load(model_path, lazy=True)
    engine = FlashEngine(model_lazy, tokenizer, config)
    
    t0 = time.perf_counter()
    flash_loss = calculate_loss(engine, tokenizer, eval_data, seq_len=args.seq_len, is_synthetic=is_synth, label="FLASH")
    t1 = time.perf_counter()
    print(f"    Loss: {flash_loss:.10f} (Time: {t1-t0:.2f}s)")

    diff = abs(flash_loss - std_loss)
    print(f"\n[*] Absolute Loss Delta: {diff:.12f}")
    
    # Accept slightly higher delta for quantized KV (8-bit)
    threshold = 1e-4 if config.kv_cache_quantized else 1e-6
    if diff < threshold:
        print(f"✅ SUCCESS: Numerical parity achieved (Delta < {threshold}).")
    else:
        print("❌ FAILURE: Numerical drift detected!")

if __name__ == "__main__":
    main()
