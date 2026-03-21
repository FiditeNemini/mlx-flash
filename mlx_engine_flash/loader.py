
from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

from .config import FlashConfig
from .streamer import SafetensorsIndex, WeightStreamer


def _update_model_weights(model: Any, weights: dict[str, mx.array]) -> None:
    """
    Surgically updates a model's weights using MLX's load_weights.
    """
    # model.load_weights(list(weights.items())) is the correct way to 
    # inject a flat dict of arrays into an MLX module.
    model.load_weights(list(weights.items()), strict=False)


class FlashModelLoader:
    """
    Iterates over transformer layers and streams their weights on demand.
    """

    def __init__(self, model_dir: Path, config: FlashConfig) -> None:
        self.model_dir = Path(model_dir)
        self.config = config
        self._streamer: WeightStreamer | None = None
        self._model_config: dict = self._load_model_config()
        self._n_layers: int = self._detect_n_layers()
        self._is_moe: bool = self._detect_moe()

    def _load_model_config(self) -> dict:
        cfg_path = self.model_dir / "config.json"
        if cfg_path.exists():
            return json.loads(cfg_path.read_text())
        return {}

    def _detect_n_layers(self) -> int:
        for key in ("num_hidden_layers", "n_layer", "num_layers"):
            if key in self._model_config:
                return int(self._model_config[key])
        idx = SafetensorsIndex(self.model_dir)
        n = idx.n_layers
        del idx
        return max(n, 1)

    def _detect_moe(self) -> bool:
        is_moe = any(k in self._model_config for k in
                    ("num_experts", "n_routed_experts", "num_local_experts"))
        if is_moe:
            raise NotImplementedError(
                "MoE streaming not yet implemented in FlashModelLoader; contributions welcome. "
                "Track issue #42."
            )
        return False

    def _validate_quant(self) -> None:
        idx = SafetensorsIndex(self.model_dir)
        idx.open_mmaps()
        bits = idx.min_quant_bits
        idx.close_mmaps()
        if bits < self.config.min_quant_bits:
            msg = (f"Model uses {bits}-bit quantisation which is below the "
                   f"configured minimum of {self.config.min_quant_bits} bits.")
            if self.config.strict_quant:
                raise ValueError(msg)
            warnings.warn(msg, UserWarning, stacklevel=3)

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def is_moe(self) -> bool:
        return self._is_moe

    def open(self) -> FlashModelLoader:
        self._validate_quant()
        self._streamer = WeightStreamer(self.model_dir, self.config)
        return self

    def close(self) -> None:
        if self._streamer is not None:
            self._streamer.close()

    def __enter__(self) -> FlashModelLoader:
        return self.open()

    def __exit__(self, *_) -> None:
        self.close()

    def get_layer_weights(self, layer_idx: int, prefetch_ahead: bool = True) -> dict[str, np.ndarray]:
        assert self._streamer is not None
        names = self._streamer.index.layer_tensor_names(layer_idx)
        
        prefetch_names = []
        if prefetch_ahead:
            for ahead in range(1, self.config.prefetch_layers + 1):
                nxt = layer_idx + ahead
                if nxt < self._n_layers:
                    prefetch_names.extend(
                        self._streamer.index.layer_tensor_names(nxt)
                    )

        return self._streamer.stream_tensors(names, prefetch_names or None)

    def get_non_layer_weights(self) -> dict[str, np.ndarray]:
        """Return weights that are NOT part of any transformer layer (embeddings, norm, head)."""
        assert self._streamer is not None
        idx = self._streamer.index
        layer_pfx = idx._layer_prefix.split(".0.")[0]
        # Heuristic: anything without "layers." in it
        names = [n for n in idx.tensor_names() if "layers." not in n]
        return self._streamer.stream_tensors(names)

    def iter_layers(self) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        assert self._streamer is not None
        for i in range(self._n_layers):
            yield i, self.get_layer_weights(i, prefetch_ahead=True)

    def to_mlx(self, weights: dict[str, np.ndarray]) -> dict[str, mx.array]:
        if not _HAS_MLX:
            raise ImportError("mlx is not installed")
        
        res = {}
        for k, v in weights.items():
            arr = mx.array(v)
            if v.dtype == np.uint16:
                arr = arr.view(mx.bfloat16)
            res[k] = arr
        return res


__all__ = ["FlashModelLoader"]
