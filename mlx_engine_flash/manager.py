
from __future__ import annotations

import functools
import json
import warnings
import types
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

try:
    import mlx_lm
    _HAS_MLX_LM = True
except ImportError:
    _HAS_MLX_LM = False

from .config import FlashConfig
from .loader import FlashModelLoader
from .prefetch import WeightPrefetcher
from .streamer import WeightStreamer


class FlashManager:
    """
    Orchestrates the Flash Weight Streaming process.
    """

    def __init__(self, config: FlashConfig) -> None:
        self.config = config
        self._loader: Optional[FlashModelLoader] = None
        self._streamer: Optional[WeightStreamer] = None
        self._prefetcher: Optional[WeightPrefetcher] = None

    def load(
        self,
        model_path: str,
        load_fn: Optional[Any] = None,
        **mlx_lm_kwargs: Any,
    ) -> Tuple[Any, Any]:
        if not _HAS_MLX_LM:
            raise ImportError("mlx_lm not installed")

        import mlx_lm
        if load_fn is None:
            load_fn = mlx_lm.load

        self.config.validate()
        model_dir = Path(model_path)
        self._check_ram(model_dir)

        # ── Step 1: load skeleton ─────────────────────────────────────────
        tokenizer_kwargs = mlx_lm_kwargs.pop("tokenizer_config", {})
        mlx_lm_kwargs["lazy"] = True
        
        # Load skeleton on CPU to avoid initial Metal memory checks
        mx.set_default_device(mx.cpu)
        try:
            model, tokenizer = load_fn(
                model_path,
                tokenizer_config=tokenizer_kwargs,
                **mlx_lm_kwargs,
            )
        finally:
            mx.set_default_device(mx.gpu)

        # ── Step 2: setup streaming ───────────────────────────────────────
        self._loader = FlashModelLoader(model_dir, self.config).__enter__()
        
        # ── Step 3: Patch layers for synchronous forward pass ──────────────
        self._patch_layers_for_sync_inference(model)

        # ── Step 4: start prefetcher ──────────────────────────────────────
        self._streamer = self._loader._streamer
        self._prefetcher = WeightPrefetcher(
            self._streamer, self.config, self._loader.n_layers
        )
        if self.config.prefetch_layers > 0:
            self._prefetcher.start()

        if self.config.debug:
            _log(f"FlashManager: loaded {model_dir.name} ({self._loader.n_layers} layers)")

        return model, tokenizer

    def _patch_layers_for_sync_inference(self, model: Any) -> None:
        """
        Intercepts each layer's forward pass to load/evict weights
        and force synchronous evaluation.
        """
        layers = []
        if hasattr(model, "backbone") and hasattr(model.backbone, "layers"):
            layers = model.backbone.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        
        if not layers:
            return

        manager = self
        
        # We patch the CLASS level __call__ for the block type
        # this is the only way to reliably intercept MLX modules
        block_class = layers[0].__class__
        original_block_call = block_class.__call__

        def _sync_block_call(instance: Any, *args: Any, **kwargs: Any) -> Any:
            # Check if this instance belongs to a model managed by us
            mgr = getattr(instance, "_flash_manager", None)
            idx = getattr(instance, "_flash_layer_idx", None)
            
            if mgr is None or idx is None:
                return original_block_call(instance, *args, **kwargs)
            
            if mgr.config.debug:
                _log(f"Layer {idx}: streaming weights...")

            # 1. Load weights
            layer_weights = mgr._loader.get_layer_weights(idx)
            _update_model_weights(instance, mgr._loader.to_mlx(layer_weights))
            
            if mgr._prefetcher is not None:
                mgr._prefetcher.notify(idx)
            
            # 2. Compute
            output = original_block_call(instance, *args, **kwargs)
            
            # 3. Eval & Sync (CRITICAL for large models on low RAM)
            mx.eval(output)
            mx.synchronize()
            
            # 4. Evict immediately
            dummy_weights = {k: mx.array(0.0) for k in layer_weights.keys()}
            _update_model_weights(instance, dummy_weights)
            
            # 5. Clear Metal cache
            mx.clear_cache()
            
            return output

        block_class.__call__ = _sync_block_call

        # Tag instances so the class-level patch knows what to do
        for i, layer in enumerate(layers):
            layer._flash_layer_idx = i
            layer._flash_manager = manager

    def shutdown(self) -> None:
        if self._prefetcher is not None:
            self._prefetcher.stop()
        if self._loader is not None:
            self._loader.close()

    def _check_ram(self, model_dir: Path) -> None:
        total_bytes = sum(f.stat().st_size for f in model_dir.glob("*.safetensors"))
        if self.config.debug:
            _log(f"Model size: {total_bytes/1e9:.1f} GB (Budget: {self.config.ram_budget_gb} GB)")


def _update_model_weights(model: Any, weights: Dict[str, Any]) -> None:
    try:
        model.load_weights(list(weights.items()), strict=False)
    except AttributeError:
        # Manual fallback
        for name, arr in weights.items():
            parts = name.split(".")
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part, None)
                if obj is None: break
            if obj is not None: setattr(obj, parts[-1], arr)

def _log(msg: str) -> None:
    import sys
    print(f"[flash] {msg}", file=sys.stderr)
