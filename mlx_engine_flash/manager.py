from __future__ import annotations

import functools
import gc
import importlib
import os
import psutil
import time
from pathlib import Path
from typing import Any, Generator

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

import mlx_lm
from mlx_lm.utils import load_config, hf_repo_to_path, load_tokenizer
from mlx_lm.generate import generate_step
from mlx_lm.models import cache as cache_utils

from .config import FlashConfig
from .loader import FlashModelLoader, _update_model_weights
from .prefetch import WeightPrefetcher
from .streamer import WeightStreamer

def _load_skeleton_only(model_path: str):
    """Load model architecture with NO weights. Returns (model, tokenizer)."""
    if not Path(model_path).exists():
        model_path = Path(hf_repo_to_path(model_path))
    else:
        model_path = Path(model_path)
    
    config = load_config(model_path)
    
    # Use the same logic as mlx_lm.utils.load_model but skip weight loading
    from mlx_lm.utils import _get_classes
    
    if (model_file := config.get("model_file")) is not None:
        spec = importlib.util.spec_from_file_location(
            "custom_model",
            model_path / model_file,
        )
        arch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arch)
        model_class, model_args_class = arch.Model, arch.ModelArgs
    else:
        model_class, model_args_class = _get_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)
    model.eval()
    
    tokenizer = load_tokenizer(model_path)
    return model, tokenizer

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

    def load(self, model_path: str, **kwargs: Any) -> tuple[Any, Any]:
        self.config.validate()
        model_dir = Path(model_path)
        
        # 1. LOCK DOWN ENVIRONMENT
        os.environ["MLX_MEMORY_MAPPING"] = "0"
        
        # 2. SKELETON LOAD (No weights loaded from disk)
        _log("Building architecture-only skeleton...")
        model, tokenizer = _load_skeleton_only(str(model_dir))
        
        # 3. QUANTIZE SKELETON (If needed)
        config = load_config(model_dir)
        if (quantization := config.get("quantization")) is not None:
            _log(f"Quantizing skeleton: {quantization['bits']}-bit")
            nn.quantize(
                model,
                group_size=quantization["group_size"],
                bits=quantization["bits"],
                mode=quantization.get("mode", "affine"),
            )

        # 4. SETUP FLASH ASSETS
        self._check_ram(model_dir)
        self._loader = FlashModelLoader(model_dir, self.config).__enter__()
        self._streamer = self._loader._streamer
        self._prefetcher = WeightPrefetcher(self._streamer, self.config, self._loader.n_layers, loader=self._loader)
        if self.config.prefetch_layers > 0: self._prefetcher.start()

        # 5. POPULATE PERMANENT WEIGHTS
        _log("Loading permanent weights...")
        idx = self._loader._streamer.index
        perm_names = [n for n in idx.tensor_names() if "layers." not in n]
        weights = self._streamer.stream_tensors(perm_names)
        
        # Handle sanitization (e.g. stacking experts, custom dequant)
        if hasattr(model, "sanitize"):
            _log("Sanitizing weights...")
            # We need to use mx.arrays for sanitization if the model expects them
            mlx_weights = self._loader.to_mlx(weights)
            mlx_weights = model.sanitize(mlx_weights)
            _update_model_weights(model, mlx_weights)
        else:
            perm_weights = self._loader.to_mlx(weights)
            _update_model_weights(model, perm_weights)
        
        # 6. PATCH FOR SYNCHRONOUS FLOW
        self._patch_layers_for_sync(model)
        
        # 7. SET LIMITS
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
                # 1. Replace weights in model first (drops refcounts to the actual arrays)
                evict_map = { k: manager._shared_dummy for k in stripped }
                _update_model_weights(layers[layer_idx], evict_map)
                
                # 2. Delete our references to the arrays
                del stripped
                if layer_weights is not None:
                    del layer_weights
                
                # 3. Force GC to release memoryview references (crucial for zero-copy mmap)
                gc.collect()
                
                # 4. Release pages in page cache
                manager._streamer.release_layer(layer_idx)
                
                # 5. Return GPU allocation
                mx.metal.clear_cache()
                
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
                mx.metal.clear_cache()
            
            pos = end
        return cache

    def generate(self, model: Any, tokenizer: Any, prompt: str, **kwargs: Any) -> Generator[Any, None, None]:
        if isinstance(prompt, str):
            tokens = mx.array(tokenizer.encode(prompt))[None]
        else:
            tokens = prompt
            
        prompt_cache = self._chunked_prefill(model, tokens, chunk_size=32)

        from mlx_lm.generate import generate_step
        for response in generate_step(mx.array([], dtype=mx.uint32)[None], model, tokenizer=tokenizer, prompt_cache=prompt_cache, **kwargs):
            yield response
            mx.synchronize()
            mx.metal.clear_cache()

    def shutdown(self) -> None:
        if self._prefetcher: self._prefetcher.stop()
        if self._loader: self._loader.close()

    def _check_ram(self, model_dir: Path) -> None:
        import warnings
        model_gb = sum(f.stat().st_size for f in model_dir.glob("*.safetensors")) / 1e9
        avail_gb = psutil.virtual_memory().available / 1e9
        if model_gb > avail_gb and not self.config.enabled:
            warnings.warn(
                f"Model is {model_gb:.1f} GB but only {avail_gb:.1f} GB "
                f"RAM is available. Enable Flash Mode to run this model.",
                ResourceWarning
            )

__all__ = ["FlashManager"]

def _log(msg: str) -> None:
    import sys
    print(f"[flash] {msg}", file=sys.stderr)
