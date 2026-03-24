import mlx.core as mx
from collections import OrderedDict
from typing import List, Dict, Any, Tuple


class ExpertCache:
    """
    Manages a hot-cache of MoE experts in GPU memory (Metal).
    Uses an LRU (Least Recently Used) eviction policy.
    """
    def __init__(self, max_experts: int = 8):
        self.max_experts = max_experts
        # OrderedDict allows us to easily track LRU. 
        # Key: (layer_idx, expert_idx), Value: Expert Weights Dictionary
        self.cache: OrderedDict[Tuple[int, int], Any] = OrderedDict()
        
    def get(self, layer_idx: int, expert_idx: int) -> Any:
        """Retrieve an expert from the cache. Updates LRU if found."""
        key = (layer_idx, expert_idx)
        if key in self.cache:
            # Move to end to mark as recently used
            self.cache.move_to_end(key)
            try:
                from benchmarks.profiler.profiler import StreamingProfiler
                StreamingProfiler().record_moe_cache(hit=True)
            except ImportError:
                pass
            return self.cache[key]
            
        try:
            from benchmarks.profiler.profiler import StreamingProfiler
            StreamingProfiler().record_moe_cache(hit=False)
        except ImportError:
            pass
        return None

    def put(self, layer_idx: int, expert_idx: int, expert_weights: Any):
        """Add an expert to the cache, evicting the oldest if at capacity."""
        key = (layer_idx, expert_idx)
        
        # If already present, just update and mark as recent
        if key in self.cache:
            self.cache[key] = expert_weights
            self.cache.move_to_end(key)
            return
            
        # Evict oldest if full
        if len(self.cache) >= self.max_experts:
            # popitem(last=False) pops the oldest entry (front of the dict)
            oldest_key, oldest_weights = self.cache.popitem(last=False)
            
            # Explicitly delete and clear metal cache to free VRAM
            del oldest_weights
            mx.metal.clear_cache()
            
        # Insert new expert
        self.cache[key] = expert_weights


class MoEPrefetcher:
    """
    Manages speculative execution and fetching for Mixture of Experts.
    """
    def __init__(self, io_prefetcher, cache: ExpertCache):
        self.io = io_prefetcher
        self.cache = cache
        
        # History of routing decisions for predictive prefetching
        # Maps layer_idx -> list of recently active expert indices
        self.routing_history: Dict[int, List[int]] = {}

    def update_history(self, layer_idx: int, top_k_indices: List[int]):
        """Records the experts chosen for the current token."""
        if layer_idx not in self.routing_history:
            self.routing_history[layer_idx] = []
            
        # Keep a rolling window of the last e.g. 5 tokens
        self.routing_history[layer_idx].extend(top_k_indices)
        if len(self.routing_history[layer_idx]) > 20: 
            self.routing_history[layer_idx] = self.routing_history[layer_idx][-20:]

    def predict_next_experts(self, layer_idx: int, num_predictions: int = 2) -> List[int]:
        """
        Heuristic: Predicts which experts will be needed for the *next* token 
        based on the most frequent experts in recent history.
        Language models often stay "on topic" for long stretches.
        """
        history = self.routing_history.get(layer_idx, [])
        if not history:
            return []
            
        # Count frequencies
        counts = {}
        for exp in history:
            counts[exp] = counts.get(exp, 0) + 1
            
        # Sort by frequency, descending
        sorted_experts = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)
        return sorted_experts[:num_predictions]

    def enqueue_expert(self, layer_idx: int, expert_idx: int, file_path: str, offset: int, size: int):
        """Requests the IO thread to load an expert if it isn't already cached."""
        if self.cache.get(layer_idx, expert_idx) is None:
            self.io.enqueue(file_path, offset, size, layer_idx=layer_idx)
