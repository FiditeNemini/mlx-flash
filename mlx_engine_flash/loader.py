"""
loader.py — Flash-aware model weight loader for mlx-lm models.

Responsibilities
----------------
1.  Validate the model directory and detect quant level.
2.  Build a SafetensorsIndex without loading any weights.
3.  Provide layer-at-a-time weight dicts compatible with mlx-lm's
    model.update_modules() / model.load_weights() interface.
4.  Emit prefetch hints for the *next* layer while the caller processes
    the *current* layer.

This loader does NOT subclass or monkey-patch any mlx-lm class — it is a
pure data-provider.  Integration with mlx-engine is handled by
FlashManager (manager.py) which wraps the standard mlx_lm.load() call.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

from .config import FlashConfig
from .streamer import SafetensorsIndex, WeightStreamer


class FlashModelLoader:
    """
    Iterates over transformer layers and streams their weights on demand.

    Usage::

        loader = FlashModelLoader(model_dir, config)
        for layer_idx, weights in loader.iter_layers():
            model.layers[layer_idx].update(weights)   # mlx-lm API
    """

    def __init__(self, model_dir: Path, config: FlashConfig) -> None:
        self.model_dir = Path(model_dir)
        self.config = config
        self._streamer: WeightStreamer | None = None
        self._model_config: dict = self._load_model_config()
        self._n_layers: int = self._detect_n_layers()
        self._is_moe: bool = self._detect_moe()

    # ── Initialisation ────────────────────────────────────────────────────

    def _load_model_config(self) -> dict:
        cfg_path = self.model_dir / "config.json"
        if cfg_path.exists():
            return json.loads(cfg_path.read_text())
        return {}

    def _detect_n_layers(self) -> int:
        for key in ("num_hidden_layers", "n_layer", "num_layers"):
            if key in self._model_config:
                return int(self._model_config[key])
        # Fallback: count from index
        idx = SafetensorsIndex(self.model_dir)
        n = idx.n_layers
        del idx
        return max(n, 1)

    def _detect_moe(self) -> bool:
        return any(k in self._model_config for k in
                   ("num_experts", "n_routed_experts", "num_local_experts"))

    def _validate_quant(self) -> None:
        """Warn or raise if quant bits are below the configured minimum."""
        idx = SafetensorsIndex(self.model_dir)
        idx.open_mmaps()
        bits = idx.min_quant_bits
        idx.close_mmaps()
        if bits < self.config.min_quant_bits:
            msg = (f"Model uses {bits}-bit quantisation which is below the "
                   f"configured minimum of {self.config.min_quant_bits} bits. "
                   f"Quality degradation is possible.")
            if self.config.strict_quant:
                raise ValueError(msg)
            warnings.warn(msg, UserWarning, stacklevel=3)

    # ── Public API ────────────────────────────────────────────────────────

    def open(self) -> FlashModelLoader:
        self._validate_quant()
        self._streamer = WeightStreamer(self.model_dir, self.config)
        # Prefetch the first N layers immediately
        for i in range(min(self.config.prefetch_layers, self._n_layers)):
            self._streamer.prefetch_layer(i)
        return self

    def close(self) -> None:
        if self._streamer is not None:
            self._streamer.close()
            self._streamer = None

    def __enter__(self) -> FlashModelLoader:
        return self.open()

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def is_moe(self) -> bool:
        return self._is_moe

    def get_non_layer_weights(self) -> dict[str, np.ndarray]:
        """
        Return weights that are NOT part of any transformer layer
        (embedding tables, final norm, lm_head, etc.).  These are always
        loaded eagerly — they're small relative to the full model.
        """
        assert self._streamer is not None, "call open() first"
        non_layer = [
            n for n in self._streamer.index.tensor_names()
            if "layers." not in n
        ]
        return self._streamer.stream_tensors(non_layer)

    def get_layer_weights(
        self,
        layer_idx: int,
        prefetch_ahead: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Stream all weights for transformer layer *layer_idx*.
        If *prefetch_ahead*, also issues hints for layers up to
        *layer_idx + prefetch_layers*.
        """
        assert self._streamer is not None, "call open() first"

        names = self._streamer.index.layer_tensor_names(layer_idx)

        prefetch_names: list[str] = []
        if prefetch_ahead:
            for ahead in range(1, self.config.prefetch_layers + 1):
                nxt = layer_idx + ahead
                if nxt < self._n_layers:
                    prefetch_names.extend(
                        self._streamer.index.layer_tensor_names(nxt)
                    )

        weights = self._streamer.stream_tensors(names, prefetch_names or None)

        # Release pages for layers we've moved far past
        evict_layer = layer_idx - self.config.prefetch_layers - 1
        if evict_layer >= 0:
            self._streamer.release_layer(evict_layer)

        return weights

    def iter_layers(self) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """
        Iterate over all layers, streaming weights one layer at a time.
        Prefetches ahead and releases behind automatically.
        """
        assert self._streamer is not None, "call open() first"
        for i in range(self._n_layers):
            yield i, self.get_layer_weights(i, prefetch_ahead=True)

    def to_mlx(self, weights: dict[str, np.ndarray]) -> dict[str, mx.array]:
        """Convert a dict of NumPy arrays to MLX arrays (no copy on unified mem)."""
        if not _HAS_MLX:
            raise ImportError("mlx is not installed")
        
        res = {}
        for k, v in weights.items():
            arr = mx.array(v)
            # If we fell back to uint16 for BF16 in numpy, bitcast it back to bfloat16 in MLX
            if v.dtype == np.uint16:
                arr = arr.view(mx.bfloat16)
            res[k] = arr
        return res
