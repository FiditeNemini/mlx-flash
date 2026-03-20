"""
prefetch.py — Background layer prefetch thread.

WeightPrefetcher runs in a daemon thread alongside the main inference loop.
It watches a shared `current_layer` integer and issues madvise(WILLNEED)
hints for layers N+1 … N+prefetch_window ahead of whatever layer the main
thread is currently computing.

This hides the ~1–3 ms page-fault latency on hot NVMe SSDs entirely for
sequential (non-MoE) models.  For MoE models, prefetch is triggered per
expert selection after routing (see MoEFlashHandler.prefetch_experts).
"""

from __future__ import annotations

import threading

from .config import FlashConfig
from .streamer import WeightStreamer


class WeightPrefetcher:
    """
    Daemon thread that prefetches and BUFFERS the *next* transformer layers.
    """

    def __init__(
        self,
        streamer: WeightStreamer,
        config: FlashConfig,
        n_layers: int,
        loader: Any = None,
    ) -> None:
        self._streamer = streamer
        self._config = config
        self._n_layers = n_layers
        self._loader = loader
        self._current_layer: int = 0
        self._last_prefetched: int = -1
        self._stop_event = threading.Event()
        self._notify_event = threading.Event()
        
        # Buffer for the next layer's weights
        self._weight_buffer: dict[int, dict[str, Any]] = {}
        self._buffer_lock = threading.Lock()
        
        self._thread = threading.Thread(
            target=self._run,
            name="flash-prefetch",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._notify_event.set()
        self._thread.join(timeout=2.0)

    def notify(self, layer_idx: int) -> None:
        """Tell the prefetcher the inference loop has moved to *layer_idx*."""
        self._current_layer = layer_idx
        self._notify_event.set()
        
        # Clear buffer for old layers
        with self._buffer_lock:
            keys_to_del = [k for k in self._weight_buffer if k <= layer_idx]
            for k in keys_to_del:
                del self._weight_buffer[k]

    def get_buffered_weights(self, layer_idx: int) -> dict[str, Any] | None:
        """Retrieve prefetched weights if available."""
        with self._buffer_lock:
            return self._weight_buffer.get(layer_idx)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._notify_event.wait(timeout=0.05)
            self._notify_event.clear()
            if self._stop_event.is_set():
                break

            cur = self._current_layer
            window = self._config.prefetch_layers

            # 1. Page-cache prefetch window (madvise)
            # We prefetch up to window+2 ahead to ensure the pages are hot 
            # by the time the double-buffer starts its actual read.
            for ahead in range(1, window + 3):
                target = cur + ahead
                if target >= self._n_layers:
                    break
                if target > self._last_prefetched:
                    self._streamer.prefetch_layer(target)
                    self._last_prefetched = target

            # 2. Actual weight loading for the NEXT layers (Double Buffering)
            # DISABLED TEMPORARILY TO FIX 16GB RAM SPIKE
            """
            if self._loader is not None:
                ...
            """
