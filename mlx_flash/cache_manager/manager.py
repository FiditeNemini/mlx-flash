import enum
from typing import Dict, Optional, List
import time
import math


class CacheTier(enum.Enum):
    HOT = 1   # Metal active memory (Pinned/Eval'd mx.array)
    WARM = 2  # OS Page Cache (Loaded via os.pread, backed by RAM)
    COLD = 3  # SSD (On disk)


class WeightBlock:
    """Represents a trackable unit of weights (e.g. an entire layer, or an expert)."""
    def __init__(self, block_id: str, layer_idx: int, size_bytes: int, is_attention: bool = False, is_router: bool = False):
        self.block_id = block_id
        self.layer_idx = layer_idx
        self.size_bytes = size_bytes
        self.tier = CacheTier.COLD
        
        self.is_attention = is_attention
        self.is_router = is_router
        
        # Access statistics
        self.frequency: float = 1.0 if not is_router else float('inf')
        self.last_access_step: int = 0
        
        # Exponential moving average factor for frequency
        self.decay_factor: float = 0.9

    def record_access(self, current_step: int):
        self.frequency = (self.frequency * self.decay_factor) + 1.0
        self.last_access_step = current_step

    def get_score(self, current_layer: int, total_layers: int) -> float:
        """Calculate the retention score. Higher score = keep in hot/warm cache."""
        if self.is_router:
            return float('inf')
            
        # Expected distance until next use
        distance = self.layer_idx - current_layer
        if distance <= 0:
            distance += total_layers # Wrap around to next token pass
            
        # Avoid division by zero if currently executing this exact layer
        if distance == 0:
            distance = 1
            
        # Attention multiplier: Attention blocks are heavily intertwined with KV Cache updates
        attn_boost = 2.0 if self.is_attention else 1.0
        
        # Base latency cost (simulated relative cost based on size)
        miss_cost = self.size_bytes * 1.5 
        
        return (self.frequency * miss_cost * attn_boost) / distance


class MultiTierCacheManager:
    """
    Manages the lifecycle of WeightBlocks across COLD (SSD), WARM (RAM), and HOT (GPU) tiers.
    Uses a predictive cost-aware policy blending Belady's Optimal Algorithm and LFU.
    """
    def __init__(self, hot_budget_bytes: int, warm_budget_bytes: int, total_layers: int):
        self.blocks: Dict[str, WeightBlock] = {}
        
        self.hot_budget = hot_budget_bytes
        self.warm_budget = warm_budget_bytes
        
        self.current_hot_bytes = 0
        self.current_warm_bytes = 0
        
        self.total_layers = total_layers
        self.current_layer_executing = 0
        self.global_step = 0

    def register_block(self, block: WeightBlock):
        """Register a new block with the cache manager."""
        self.blocks[block.block_id] = block

    def step_layer(self, next_layer_idx: int):
        """Advance the execution state. Call this before executing a layer."""
        self.current_layer_executing = next_layer_idx
        self.global_step += 1

    def access_block(self, block_id: str) -> WeightBlock:
        """Mark a block as accessed and ensure it is promoted to HOT."""
        block = self.blocks.get(block_id)
        if not block:
            raise KeyError(f"Block {block_id} not registered.")
            
        block.record_access(self.global_step)
        
        if block.tier != CacheTier.HOT:
            self._promote_to_hot(block)
            
        return block

    def _promote_to_hot(self, block: WeightBlock):
        """Promote a block to HOT, evicting others if necessary."""
        # 1. Ensure space in HOT
        while self.current_hot_bytes + block.size_bytes > self.hot_budget:
            evicted = self._evict_from(CacheTier.HOT)
            if not evicted:
                break # Cannot evict anything else (budget too small for a single block)
            
        # 2. Update accounting based on previous tier
        if block.tier == CacheTier.WARM:
            self.current_warm_bytes -= block.size_bytes
            
        block.tier = CacheTier.HOT
        self.current_hot_bytes += block.size_bytes

    def _evict_from(self, tier: CacheTier) -> bool:
        """Finds the lowest scoring block in the tier and demotes it. Returns True if something was evicted."""
        lowest_score = float('inf')
        victim: Optional[WeightBlock] = None
        
        # Linear scan (acceptable for small N layers, e.g. < 100 blocks). 
        # For thousands of blocks (e.g. fine-grained MoE), a bucketed priority queue should be used.
        for block in self.blocks.values():
            if block.tier == tier:
                score = block.get_score(self.current_layer_executing, self.total_layers)
                if score < lowest_score:
                    lowest_score = score
                    victim = block
                    
        if victim:
            if tier == CacheTier.HOT:
                self._demote_hot_to_warm(victim)
            elif tier == CacheTier.WARM:
                self._demote_warm_to_cold(victim)
            return True
            
        return False

    def _demote_hot_to_warm(self, block: WeightBlock):
        """Demote from GPU to RAM."""
        self.current_hot_bytes -= block.size_bytes
        block.tier = CacheTier.WARM
        self.current_warm_bytes += block.size_bytes
        
        # Cascade eviction if WARM is now full
        while self.current_warm_bytes > self.warm_budget:
            evicted = self._evict_from(CacheTier.WARM)
            if not evicted:
                break

    def _demote_warm_to_cold(self, block: WeightBlock):
        """Demote from RAM to SSD."""
        self.current_warm_bytes -= block.size_bytes
        block.tier = CacheTier.COLD

    def get_tier_stats(self) -> Dict[str, Any]:
        """Return current memory usage statistics."""
        return {
            "hot_bytes": self.current_hot_bytes,
            "hot_utilization": self.current_hot_bytes / max(1, self.hot_budget),
            "warm_bytes": self.current_warm_bytes,
            "warm_utilization": self.current_warm_bytes / max(1, self.warm_budget)
        }
