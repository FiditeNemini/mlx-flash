from __future__ import annotations

import functools
import gc
import os
import psutil
import time
from pathlib import Path
from typing import Any, Generator

try:
    import mlx.core as mx
    import mlx.nn as nn
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

try:
    import mlx_lm
    from mlx_lm.utils import generate_step
    from mlx_lm.models import cache as cache_utils
    _HAS_MLX_LM = True
except ImportError:
    _HAS_MLX_LM = False

from .config import FlashConfig
from .loader import FlashModelLoader, _update_model_weights
from .prefetch import WeightPrefetcher
from .streamer import WeightStreamer

class FlashManager:
    """
    Synchronous Inference Engine for Flash Weight Streaming.
    Manually orchestrates prefill and generation to bypass MLX graph limits.
    """

    def __init__(self, config: FlashConfig) -> None:
        self.config = config
        self._loader: FlashModelLoader | None = None
        self._streamer: WeightStreamer | None = None
        self._prefetcher: WeightPrefetcher | None = None
        self._metrics: dict[str, int] = {"cache_hits": 0, "cache_misses": 0}
        self._shared_dummy = mx.array(0.0, dtype=mx.float16)

    def load(self, model_path: str, load_fn: Any | None = None, **mlx_lm_kwargs: Any) -> tuple[Any, Any]:
        import mlx_lm
        if load_fn is None: load_fn = mlx_lm.load

        self.config.validate()
        model_dir = Path(model_path)
        
        # 1. LOCK DOWN ENVIRONMENT
        os.environ["MLX_MEMORY_MAPPING"] = "0"
        
        # 2. SKELETON LOAD (Small Dummies to bypass Metal checks)
        original_mx_load = mx.load
        def _dummy_load(*args, **kwargs):
            w = original_mx_load(*args, **kwargs)
            return { k: mx.array([0.0], v.dtype) for k, v in w.items() }
        
        original_load_weights = nn.Module.load_weights
        nn.Module.load_weights = lambda self, weights, strict=False: self
        
        _log("Building tiny-weight skeleton...")
        mx.load = _dummy_load
        try:
            mlx_lm_kwargs["lazy"] = True
            model, tokenizer = load_fn(str(model_dir), **mlx_lm_kwargs)
        finally:
            mx.load = original_mx_load
            nn.Module.load_weights = original_load_weights

        # 3. SETUP FLASH ASSETS
        self._loader = FlashModelLoader(model_dir, self.config).__enter__()
        self._streamer = self._loader._streamer
        self._prefetcher = WeightPrefetcher(self._streamer, self.config, self._loader.n_layers, loader=self._loader)
        if self.config.prefetch_layers > 0: self._prefetcher.start()

        # 4. POPULATE PERMANENT WEIGHTS
        _log("Loading permanent weights...")
        idx = self._loader._streamer.index
        perm_names = [n for n in idx.tensor_names() if "layers." not in n]
        perm_weights = self._loader.to_mlx(self._streamer.stream_tensors(perm_names))
        _update_model_weights(model, perm_weights)
        
        # 5. PATCH FOR SYNCHRONOUS FLOW
        self._patch_layers_for_sync(model)
        
        # 6. SET LIMITS
        mx.metal.set_cache_limit(1024 * 1024 * 1024)
        mx.metal.set_wired_limit(1024 * 1024 * 1024)
        
        return model, tokenizer

    def _patch_layers_for_sync(self, model: Any) -> None:
        backbone = getattr(model, "model", getattr(model, "backbone", model))
        layers = backbone.layers
        manager = self

        def _make_sync_call(original_call, layer_idx):
            @functools.wraps(original_call)
            def _sync_call(*args, **kwargs):
                # i. Load
                layer_weights = (manager._prefetcher.get_buffered_weights(layer_idx) if manager._prefetcher else None)
                if layer_weights is None:
                    layer_weights = manager._loader.get_layer_weights(layer_idx)
                
                prefix = manager._loader._streamer.index._layer_prefix.replace(".0.", f".{layer_idx}.")
                if "layers.0." in manager._loader._streamer.index._layer_prefix and not manager._loader._streamer.index._layer_prefix.startswith("."):
                    prefix = manager._loader._streamer.index._layer_prefix.replace("layers.0.", f"layers.{layer_idx}.")
                
                stripped = { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in layer_weights.items() }
                if any(not isinstance(v, mx.array) for v in stripped.values()):
                    stripped = manager._loader.to_mlx(stripped)
                
                _update_model_weights(layers[layer_idx], stripped)
                if manager._prefetcher: manager._prefetcher.notify(layer_idx)

                # ii. Execute
                output = original_call(*args, **kwargs)
                
                # iii. REALIZE
                mx.eval(output)
                mx.synchronize()
                
                # iv. EVICT
                evict_map = { k: manager._shared_dummy for k in stripped }
                _update_model_weights(layers[layer_idx], evict_map)
                mx.clear_cache()
                
                return output
            return _sync_call

        for i, layer in enumerate(layers):
            layer.__call__ = _make_sync_call(layer.__call__, i)

    def _chunked_prefill(self, model: Any, tokens: mx.array, chunk_size: int = 256) -> Any:
        from mlx_lm.models.cache import make_prompt_cache
        num_tokens = tokens.shape[1]
        cache = make_prompt_cache(model)
        
        from mlx_lm.generate import maybe_quantize_kv_cache
        maybe_quantize_kv_cache(cache, kv_bits=4, kv_group_size=64, quantized_kv_start=0)

        _log(f"Starting chunked prefill: {num_tokens} tokens")
        pos = 0
        while pos < num_tokens:
            end = min(pos + chunk_size, num_tokens)
            chunk = tokens[:, pos:end]
            
            # LIFT LIMITS
            safe_max = int(psutil.virtual_memory().total * 0.95)
            mx.metal.set_cache_limit(safe_max)
            mx.metal.set_wired_limit(safe_max)
            
            try:
                model(chunk, cache=cache)
                mx.eval([c.state for c in cache])
                mx.synchronize()
            finally:
                limit = 1024 * 1024 * 1024
                mx.metal.set_cache_limit(limit)
                mx.metal.set_wired_limit(limit)
                mx.clear_cache()
            
            pos = end
        return cache

    def generate(self, model: Any, tokenizer: Any, prompt: str, **kwargs: Any) -> Generator[Any, None, None]:
        if isinstance(prompt, str):
            tokens = mx.array(tokenizer.encode(prompt))[None]
        else:
            tokens = prompt
            
        prompt_cache = self._chunked_prefill(model, tokens, chunk_size=32)

        from mlx_lm.utils import generate_step
        for response in generate_step(mx.array([], dtype=mx.uint32)[None], model, tokenizer=tokenizer, prompt_cache=prompt_cache, **kwargs):
            yield response
            mx.synchronize()
            mx.clear_cache()

    def shutdown(self) -> None:
        if self._prefetcher: self._prefetcher.stop()
        if self._loader: self._loader.close()

def _log(msg: str) -> None:
    import sys
    print(f"[flash] {msg}", file=sys.stderr)
