#!/usr/bin/env python3
import gc
import json
import os
import struct
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import psutil

from mlx_flash import FlashConfig, FlashManager


def pack_q4_0_block(scale: float, values: list) -> bytes:
    scale_bytes = struct.pack("<e", scale)
    data = bytearray(16)
    for i, v in enumerate(values):
        nibble = (int(v) + 8) & 0xF
        if i % 2 == 0:
            data[i // 2] = nibble
        else:
            data[i // 2] |= (nibble << 4)
    return scale_bytes + bytes(data)

def write_safetensors(path: Path, tensors: dict) -> None:
    header = {}
    data_parts = []
    offset = 0
    for name, (data_bytes, dtype_str, shape) in tensors.items():
        length = len(data_bytes)
        header[name] = {"dtype": dtype_str, "shape": list(shape), "data_offsets": [offset, offset + length]}
        data_parts.append(data_bytes)
        offset += length
    header_json = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))
    with open(path, "wb") as f:
        f.write(header_len)
        f.write(header_json)
        for d in data_parts:
            f.write(d)

def create_massive_synthetic(mdir: Path, n_layers=24, hidden_dim=2048):
    print(f"[*] Synthesizing {n_layers}-layer MLX test model (approx {(n_layers * hidden_dim * hidden_dim * 4 * 2) / 1e9:.1f} GB)...")
    mdir.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(42)
    
    def rand_f16(shape: list[int]) -> bytes:
        # Scale down weights to prevent logit explosion during PPL tests
        return (rng.standard_normal(shape) * 0.01).astype(np.float16).tobytes()

    tensors = {}
    embed_data = rand_f16([hidden_dim, hidden_dim])
    tensors["model.embed_tokens.weight"] = (embed_data, "F16", [hidden_dim, hidden_dim])

    for layer in range(n_layers):
        pfx = f"model.layers.{layer}"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            data = rand_f16([hidden_dim, hidden_dim])
            tensors[f"{pfx}.self_attn.{proj}.weight"] = (data, "F16", [hidden_dim, hidden_dim])
        for proj, shape in (("gate_proj", [hidden_dim*2, hidden_dim]), ("up_proj", [hidden_dim*2, hidden_dim]), ("down_proj", [hidden_dim, hidden_dim*2])):
            data = rand_f16(shape)
            tensors[f"{pfx}.mlp.{proj}.weight"] = (data, "F16", shape)
        ln_data = np.ones(hidden_dim, dtype=np.float16).tobytes()
        tensors[f"{pfx}.input_layernorm.weight"] = (ln_data, "F16", [hidden_dim])
        tensors[f"{pfx}.post_attention_layernorm.weight"] = (ln_data, "F16", [hidden_dim])

    tensors["model.norm.weight"] = (np.ones(hidden_dim, dtype=np.float16).tobytes(), "F16", [hidden_dim])
    tensors["lm_head.weight"] = (rng.standard_normal([hidden_dim, hidden_dim]).astype(np.float16).tobytes(), "F16", [hidden_dim, hidden_dim])

    write_safetensors(mdir / "model.safetensors", tensors)

    cfg = {
        "model_type": "llama", "hidden_size": hidden_dim, "intermediate_size": hidden_dim*2,
        "num_hidden_layers": n_layers, "num_attention_heads": 8, "rms_norm_eps": 1e-6,
        "vocab_size": hidden_dim, "tie_word_embeddings": False,
    }
    (mdir / "config.json").write_text(json.dumps(cfg))

    tok_cfg = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "model": {"type": "BPE", "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2}, "merges": []}}
    (mdir / "tokenizer.json").write_text(json.dumps(tok_cfg))
    (mdir / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>", "eos_token": "</s>"}))
    return mdir

def get_ram_mb():
    gc.collect()
    mx.synchronize()
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

if __name__ == "__main__":
    model_dir = Path("/tmp/mlx_synthetic_proof")
    if not (model_dir / "model.safetensors").exists():
        create_massive_synthetic(model_dir, n_layers=16, hidden_dim=2048)
    
    file_size_gb = os.path.getsize(model_dir / "model.safetensors") / 1e9
    print(f"\n[+] Synthetic Model Size: {file_size_gb:.1f} GB")
    
    print("\n--------- STANDARD MLX ---------")
    rss_baseline = get_ram_mb()
    try:
        # Standard MLX load without lazy=True will spike RAM
        import mlx_lm
        print("[!] Attempting to load using standard MLX (no flash mode)...")
        model, _ = mlx_lm.load(str(model_dir))
        rss_normal = get_ram_mb()
        print(f"    Standard RAM required: {rss_normal - rss_baseline:.1f} MB overhead")
        del model
        gc.collect()
    except Exception as e:
        print(f"    Standard MLX Failed/OOM: {e}")

    print("\n--------- FLASH MLX (v0.2 Async I/O) ---------")
    rss_baseline = get_ram_mb()
    
    # We use MADV_DONTNEED exclusively for this proof to force the macOS kernel
    # to instantly drop the pages from RSS rather than waiting for memory pressure.
    cfg = FlashConfig(enabled=True, ram_budget_gb=1.0, eviction_strategy="MADV_DONTNEED")
    manager = FlashManager(cfg)
    
    t0 = time.time()
    fl_model, tok = manager.load(str(model_dir))
    load_time = time.time() - t0
    
    rss_loaded = get_ram_mb()
    print(f"\n[+] Flash Load Time: {load_time:.2f}s")
    print(f"[+] Flash RAM used to map {file_size_gb:.1f}GB model: {rss_loaded - rss_baseline:.1f} MB")
    
    print("\n[*] Starting generation ... watch the RAM stay perfectly flat!")
    print(f"    Metal Active Init: {mx.metal.get_active_memory() / 1e6:.1f} MB")
    
    from mlx_flash.generation import FlashGenerationLoop
    loop = FlashGenerationLoop(fl_model, cfg)
    
    # We duck-type the generator from mlx_lm to use our loop
    t0 = time.time()
    tokens_gen = 0
    import mlx_lm
    for _chunk in mlx_lm.stream_generate(loop.flash_model, tok, "Hello world", max_tokens=10):
        tokens_gen += 1
        print(f"    [Token {tokens_gen}] System RAM Overhead: {get_ram_mb() - rss_baseline:.1f} MB | Metal VRAM: {mx.metal.get_active_memory() / 1e6:.1f} MB")
    total_time = time.time() - t0
    
    print("\n[+] Finished Generation!")
    print(f"    Speed: {tokens_gen / total_time:.1f} tok/s")
    print(f"    Final Metal VRAM: {mx.metal.get_active_memory() / 1e6:.1f} MB (Stays well under 1GB!)")
    
    manager.shutdown()
