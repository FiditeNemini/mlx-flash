"""
moe.py — MoE expert routing and selective streaming.

For Mixture-of-Experts models (Mixtral, DeepSeek-V2/V3/R1, Qwen2-MoE, etc.)
we need only the top-K active experts per layer per token batch.  This can
reduce the data read from disk by 75–99% vs loading all experts.

Architecture
------------
MoEConfig
    Extracted from model config.json.

MoERouter
    Runs the router matmul (always hot in RAM — it is tiny) and returns
    the top-K expert indices for a token batch using pure NumPy / MLX.
    This is intentionally decoupled from the weight streamer so the I/O
    and the routing logic can be developed and tested independently.

MoEFlashHandler
    Orchestrates: route → select experts → issue parallel pread → return
    expert weight dicts.  Plugs into FlashManager.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FlashConfig
from .streamer import WeightStreamer

try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False


@dataclass
class MoEConfig:
    """MoE topology extracted from model config.json."""
    n_experts: int                    # total number of experts
    top_k: int                        # experts per token (router top-K)
    n_layers: int                     # transformer layers that have MoE FFN
    expert_tensor_pattern: str = "mlp.experts.{eidx}"  # name template

    @classmethod
    def from_model_config(cls, cfg: dict) -> MoEConfig | None:
        """
        Parse MoE parameters from a HuggingFace config.json dict.
        Returns None if the model is not MoE.
        """
        n_experts = (
            cfg.get("num_experts")
            or cfg.get("n_routed_experts")
            or cfg.get("num_local_experts")
        )
        if n_experts is None:
            return None

        top_k = (
            cfg.get("num_experts_per_tok")
            or cfg.get("top_k_experts")
            or cfg.get("num_selected_experts")
            or 2
        )
        n_layers = (
            cfg.get("num_hidden_layers")
            or cfg.get("n_layer")
            or 0
        )
        return cls(n_experts=int(n_experts), top_k=int(top_k),
                   n_layers=int(n_layers))


class MoERouter:
    """
    Computes top-K expert assignments for a batch of token hidden states.

    The router weights are small (hidden_dim × n_experts) and are kept
    permanently in memory — they are needed every single forward pass.
    Only the selected expert FFN weights are streamed from disk.
    """

    def __init__(
        self,
        router_weights: np.ndarray,    # shape: [hidden_dim, n_experts]
        moe_cfg: MoEConfig,
        top_k_override: int | None = None,
    ) -> None:
        self._weights = router_weights
        self.moe_cfg = moe_cfg
        self.top_k = top_k_override or moe_cfg.top_k
        self._weights_mx: mx.array | None = None

    def _to_mlx(self) -> mx.array:
        if self._weights_mx is None:
            self._weights_mx = mx.array(self._weights)
        return self._weights_mx

    def route(
        self,
        hidden_states: np.ndarray,   # [batch * seq, hidden_dim]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run router and return (expert_indices, expert_weights).

        expert_indices: int array [batch*seq, top_k] — sorted by score desc
        expert_weights: float array [batch*seq, top_k] — softmax probabilities
        """
        # Router logits: [batch*seq, n_experts]
        logits = hidden_states @ self._weights   # float matmul

        # Top-K indices per token
        top_k = min(self.top_k, self.moe_cfg.n_experts)
        # np.argpartition gives top_k fastest; then sort within partition
        part = np.argpartition(logits, -top_k, axis=-1)[:, -top_k:]
        top_logits = np.take_along_axis(logits, part, axis=-1)
        order = np.argsort(-top_logits, axis=-1)
        expert_indices = np.take_along_axis(part, order, axis=-1)
        top_logits_sorted = np.take_along_axis(top_logits, order, axis=-1)

        # Softmax over selected logits for combining weights
        logits_max = top_logits_sorted.max(axis=-1, keepdims=True)
        exp_l = np.exp(top_logits_sorted - logits_max)
        expert_weights = exp_l / exp_l.sum(axis=-1, keepdims=True)

        return expert_indices.astype(np.int32), expert_weights.astype(np.float32)

    def unique_experts(self, expert_indices: np.ndarray) -> list[int]:
        """Return sorted list of unique expert indices needed for this batch."""
        return sorted(set(expert_indices.flatten().tolist()))


class MoEFlashHandler:
    """
    Orchestrates MoE routing + selective expert streaming with LRU cache.
    """

    def __init__(
        self,
        streamer: WeightStreamer,
        moe_cfg: MoEConfig,
        config: FlashConfig,
        router_weights_by_layer: dict[int, np.ndarray],
    ) -> None:
        self._streamer = streamer
        self._moe_cfg = moe_cfg
        self._config = config
        self._routers: dict[int, MoERouter] = {
            layer_idx: MoERouter(
                rw, moe_cfg,
                top_k_override=config.moe_top_k_override,
            )
            for layer_idx, rw in router_weights_by_layer.items()
        }
        self._current_layer: int = 0
        
        # Expert LRU Cache: (layer_idx, expert_idx) -> {tensor_name: mx.array}
        self._cache: dict[tuple[int, int], dict[str, Any]] = {}
        self._lru: list[tuple[int, int]] = []

    def load_experts(
        self, expert_idxs: list[int]
    ) -> dict[int, dict[str, Any]]:
        """
        Stream weights for each expert index in *expert_idxs* with LRU caching.
        Returns {expert_idx: {tensor_name: mx.array}}.
        """
        result: dict[int, dict[str, Any]] = {}
        needed_idxs: list[int] = []
        
        # 1. Check cache
        for eidx in expert_idxs:
            key = (self._current_layer, eidx)
            if key in self._cache:
                result[eidx] = self._cache[key]
                # Update LRU
                self._lru.remove(key)
                self._lru.append(key)
                if self._config.debug:
                    import sys
                    print(f"[flash] MoE Cache HIT: Layer {self._current_layer} Expert {eidx}", file=sys.stderr)
            else:
                needed_idxs.append(eidx)

        if not needed_idxs:
            return result

        # 2. Load missing experts from disk
        all_names: list[str] = []
        for eidx in needed_idxs:
            names = self._streamer.index.expert_tensor_names(
                self._current_layer, eidx
            )
            all_names.extend(names)

        all_weights = self._streamer.stream_tensors(all_names)

        for eidx in needed_idxs:
            prefix = (f"model.layers.{self._current_layer}"
                      f".mlp.experts.{eidx}.")
            expert_weights = {
                k[len(prefix):]: v
                for k, v in all_weights.items()
                if k.startswith(prefix)
            }
            
            # Convert to MLX immediately
            mlx_expert = {k: mx.array(v) for k, v in expert_weights.items()}
            # Bitcast BF16 if needed (match loader.py logic)
            for k, v in expert_weights.items():
                if v.dtype == np.uint16:
                    mlx_expert[k] = mlx_expert[k].view(mx.bfloat16)
            
            # Add to cache (and result)
            key = (self._current_layer, eidx)
            self._cache[key] = mlx_expert
            self._lru.append(key)
            result[eidx] = mlx_expert

        # 3. Enforce cache size limit
        while len(self._lru) > self._config.expert_cache_size:
            oldest_key = self._lru.pop(0)
            del self._cache[oldest_key]
            if self._config.debug:
                import sys
                print(f"[flash] MoE Cache EVICT: Layer {oldest_key[0]} Expert {oldest_key[1]}", file=sys.stderr)

        return result
