import os
import time
import gc
import psutil
from pathlib import Path
from typing import Iterator, Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .config import FlashConfig
from .manager import FlashManager, _load_skeleton_only
from .loader import _update_model_weights

class FlashGenerationLoop:
    """
    Synchronous per-layer generation loop for Flash Weight Streaming.
    
    This is the ONLY approach that keeps RAM < layer_size on models 
    larger than physical RAM. Standard mlx_lm.stream_generate() cannot 
    be used because MLX's lazy graph materialises all layers at once.
    """
    
    def __init__(self, model_dir: str | Path, config: FlashConfig):
        self.model_dir = Path(model_dir)
        self.config = config
        self.manager = FlashManager(config)
        
        # Load skeleton and permanent weights
        self.model, self.tokenizer = self.manager.load(str(self.model_dir))
        self.streamer = self.manager._streamer
        self.loader = self.manager._loader
        
        # Shared dummy array to replace weights
        self._shared_dummy = mx.array(0.0, dtype=mx.float16)

    def _get_rss_gb(self) -> float:
        return psutil.Process().memory_info().rss / (1024**3)

    def _sample(self, logits: mx.array, temperature: float, top_p: float) -> mx.array:
        if temperature == 0:
            return mx.argmax(logits, axis=-1)
        
        logits = logits / temperature
        
        if top_p < 1.0:
            # Nucleus sampling
            sorted_indices = mx.argsort(logits, axis=-1)
            sorted_logits = logits[..., sorted_indices[0]]
            cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
            
            # Mask out tokens with cumulative probability above top_p
            mask = cumulative_probs > top_p
            # Shift mask to keep the first token that exceeds top_p
            mask[..., 1:] = mask[..., :-1]
            mask[..., 0] = False
            
            # Apply mask
            sorted_logits = mx.where(mask, mx.array(float("-inf")), sorted_logits)
            
            # Sample from the filtered distribution
            token_idx = mx.random.categorical(sorted_logits)
            return sorted_indices[0, token_idx]
        else:
            return mx.random.categorical(logits)

    def stream_generate(
        self, 
        prompt: str, 
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Iterator[str]:
        """
        Stream tokens. For each token:
        1. Tokenize prompt + generated so far
        2. Run embedding layer (always hot, small)
        3. For each transformer layer:
           a. Zero-copy load weights from mmap
           b. Compute layer output
           c. mx.eval() + mx.synchronize()  ← CRITICAL for memory release
           d. Zero out layer weights
           e. mx.metal.clear_cache()
           f. madvise(MADV_FREE) on layer pages
        4. Run LM head (always hot, small)
        5. Sample next token
        6. Yield decoded token string
        """
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array(tokens)[None]
        
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self.model)
        
        # We need to handle prefill and generation separately if using the cache properly
        # but for the requested per-token loop, we can just grow the cache.
        
        # Nemotron-H specific masks
        from .manager import _load_skeleton_only # To avoid circular import if any
        # We can get the mask functions from the model's module or mlx_lm
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        for i in range(max_tokens):
            if self.config.debug:
                print(f"\n[flash] --- Token {i+1} (RSS: {self._get_rss_gb():.2f} GB) ---")
            
            # 1 & 2. Embedding layer
            # We only pass the last token if we have a cache
            current_input = input_ids if i == 0 else input_ids[:, -1:]
            
            # Extract backbone (handle different model architectures)
            backbone = getattr(self.model, "model", getattr(self.model, "backbone", self.model))
            
            x = backbone.embeddings(current_input)
            
            # Compute masks if needed (Nemotron-H style)
            # Find indices for attention and ssm if present
            fa_idx = getattr(backbone, "fa_idx", 0)
            ssm_idx = getattr(backbone, "ssm_idx", 0)
            attn_mask = create_attention_mask(x, cache[fa_idx])
            ssm_mask = create_ssm_mask(x, cache[ssm_idx])

            # 3. Transformer layers
            cache_idx = 0
            for layer_idx in range(len(backbone.layers)):
                layer = backbone.layers[layer_idx]
                mask = None
                
                # Determine cache and mask
                current_cache = None
                if hasattr(layer, "block_type") and layer.block_type in ("M", "*"):
                    current_cache = cache[cache_idx]
                    cache_idx += 1
                    mask = attn_mask if layer.block_type == "*" else ssm_mask
                
                x = self._run_layer_synchronous(layer_idx, x, current_cache, mask=mask)
                
            # 4. Final norm and LM head
            x = backbone.norm_f(x)
            logits = self.model.lm_head(x)
            
            # 5. Sample
            # We only care about the last token's logits
            next_token = self._sample(logits[:, -1, :], temperature, top_p)
            
            # Ensure next_token is 2D (1, 1) for concatenation with input_ids (1, N)
            if next_token.ndim == 0:
                next_token = next_token[None, None]
            elif next_token.ndim == 1:
                next_token = next_token[None]
            
            # 6. Decode and yield
            input_ids = mx.concatenate([input_ids, next_token], axis=1)
            token_id = next_token.item()
            
            if token_id == self.tokenizer.eos_token_id:
                break
                
            yield self.tokenizer.decode([token_id])
            
            # Clear any remaining Metal allocations
            mx.metal.clear_cache()

    def _run_layer_synchronous(
        self, 
        layer_idx: int, 
        hidden_states: mx.array, 
        cache: Any = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Load, run, eval, and zero one transformer layer.
        Peak Metal memory = one layer's weights + activations.
        """
        if self.config.debug:
            rss_before = self._get_rss_gb()

        # Step 1: Load weights (zero-copy mmap slice → mx.array)
        layer_weights = self.loader.get_layer_weights(layer_idx)
        
        # Apply sanitization if model has it (e.g. stacking experts)
        if hasattr(self.model, "sanitize"):
            layer_weights = self.model.sanitize(self.loader.to_mlx(layer_weights))
        else:
            layer_weights = self.loader.to_mlx(layer_weights)

        # Fix keys for the specific layer (FlashModelLoader returns full names)
        # We need to strip the prefix to load into the layer module directly
        prefix = self.streamer.index._layer_prefix.replace(".0.", f".{layer_idx}.")
        if "layers.0." in self.streamer.index._layer_prefix and not self.streamer.index._layer_prefix.startswith("."):
             prefix = self.streamer.index._layer_prefix.replace("layers.0.", f"layers.{layer_idx}.")
        
        stripped_weights = {
            (k[len(prefix):] if k.startswith(prefix) else k): v 
            for k, v in layer_weights.items()
        }
        
        backbone = getattr(self.model, "model", getattr(self.model, "backbone", self.model))
        layer = backbone.layers[layer_idx]
        
        layer.load_weights(list(stripped_weights.items()), strict=False)
        
        # Step 2: Compute
        if mask is not None:
            hidden_states = layer(hidden_states, mask=mask, cache=cache)
        else:
            hidden_states = layer(hidden_states, cache=cache)
        
        # Step 3: Materialise NOW — before moving to next layer
        mx.eval(hidden_states)
        if cache is not None and hasattr(cache, "state"):
            mx.eval(cache.state)
        mx.synchronize()
        
        # Step 4: Free the layer's weights from Metal pool
        self._zero_layer_weights(layer_idx)
        mx.metal.clear_cache()
        
        # Step 5: Release mmap pages back to OS
        self.streamer.release_layer(layer_idx)
        
        # Cleanup
        del layer_weights
        del stripped_weights
        gc.collect()

        if self.config.debug:
            rss_after = self._get_rss_gb()
            print(f"  Layer {layer_idx:02d}: RSS {rss_before:.2f} -> {rss_after:.2f} GB")
        
        return hidden_states

    def _zero_layer_weights(self, layer_idx: int):
        """Replace layer weights with a scalar to release Metal allocation."""
        backbone = getattr(self.model, "model", getattr(self.model, "backbone", self.model))
        layer = backbone.layers[layer_idx]
        
        # We can just use the shared dummy for all parameters
        zeros = {k: self._shared_dummy for k, _ in tree_flatten(layer.parameters())}
        layer.load_weights(list(zeros.items()), strict=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flash Weight Streaming Generation Loop")
    parser.add_argument("--model", required=True, help="Path to MLX model directory")
    parser.add_argument("--prompt", default="Explain quantum computing in one sentence.", help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--ram-budget", type=float, default=6.0, help="RAM budget in GB")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    config = FlashConfig(
        enabled=True, 
        ram_budget_gb=args.ram_budget, 
        debug=args.debug,
        eviction_strategy="free"
    )
    
    loop = FlashGenerationLoop(args.model, config)
    
    print(f"\nPrompt: {args.prompt}")
    print("Response: ", end="", flush=True)
    
    t0 = time.perf_counter()
    tokens_generated = 0
    
    try:
        for token in loop.stream_generate(
            args.prompt, 
            max_tokens=args.max_tokens,
            temperature=args.temp,
            top_p=args.top_p
        ):
            print(token, end="", flush=True)
            tokens_generated += 1
    except KeyboardInterrupt:
        print("\n[interrupted]")
    
    t1 = time.perf_counter()
    total_time = t1 - t0
    
    print(f"\n\nStats:")
    print(f"  Tokens: {tokens_generated}")
    print(f"  Time: {total_time:.2f}s")
    print(f"  Speed: {tokens_generated / total_time:.2f} tok/s")
    print(f"  Peak RSS: {psutil.Process().memory_info().rss / (1024**3):.2f} GB")
    
    loop.manager.shutdown()
