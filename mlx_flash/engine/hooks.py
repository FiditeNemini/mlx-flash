from typing import Any, Dict, List, Set
import mlx.core as mx
import mlx.nn as nn

class ExecutionContext:
    """Carries state through the generation loop without polluting method signatures."""
    def __init__(self, engine, x: mx.array, mask: mx.array = None, cache=None):
        self.engine = engine
        self.x = x
        self.mask = mask
        self.cache = cache
        self.layer_idx = 0
        self.metadata: Dict[str, Any] = {} # For hooks to share data
        
        # Extracted signatures from the engine for convenience
        self.has_mask = False
        self.has_cache = False
        self.cache_entry = None

class InferenceHook:
    """Base class for all lifecycle side-effects, representing a Node in the Execution Graph."""
    
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def dependencies(self) -> List[str]:
        """List of hook names that MUST execute before this hook (if they exist in the graph)."""
        return []
    
    # 1. Structural Phase (Initialization)
    def on_model_load(self, model: nn.Module) -> nn.Module:
        """Replace monkey-patching. e.g., Tiling modifies the model here."""
        return model

    # 2. Generation Phase
    def on_generation_start(self, ctx: ExecutionContext): pass
    def on_generation_end(self, ctx: ExecutionContext): pass

    # 3. Layer Loop Phase
    def on_layer_start(self, ctx: ExecutionContext, layer: nn.Module):
        """Ideal for triggering IO prefetch N layers ahead."""
        pass
        
    def on_layer_end(self, ctx: ExecutionContext, layer: nn.Module):
        """Ideal for triggering MADV_DONTNEED cache eviction or profiling."""
        pass
        
    # 4. Sub-component Phase (for Pipelined execution)
    def on_router_decision(self, ctx: ExecutionContext, top_k_indices: list):
        """Ideal for speculative MoE prefetching."""
        pass

class ExecutionGraph:
    """
    A Deterministic Execution Graph (DAG) for Hooks.
    Replaces implicit registration-order dispatching with an explicit topological sort.
    """
    def __init__(self):
        self.nodes: Dict[str, InferenceHook] = {}
        self._execution_order: List[InferenceHook] = []
        self._is_compiled = False
        
    def add_node(self, hook: InferenceHook):
        self.nodes[hook.name] = hook
        self._is_compiled = False
        
    def compile(self):
        """Builds a deterministic execution order via topological sort."""
        visited: Set[str] = set()
        temp_mark: Set[str] = set()
        order: List[InferenceHook] = []
        
        def visit(node_name: str):
            if node_name in temp_mark:
                raise ValueError(f"Cyclic dependency detected involving {node_name}")
            if node_name not in visited:
                temp_mark.add(node_name)
                
                node = self.nodes.get(node_name)
                if node:
                    # Process dependencies first
                    for dep in sorted(node.dependencies): # Sort to ensure determinism if multiple
                        visit(dep)
                        
                    temp_mark.remove(node_name)
                    visited.add(node_name)
                    order.append(node)
                else:
                    # Soft-fail for optional dependencies not present in graph
                    temp_mark.remove(node_name)
                    visited.add(node_name)
                    
        # Sort node keys for deterministic iteration order
        for node_name in sorted(self.nodes.keys()):
            if node_name not in visited:
                visit(node_name)
                
        self._execution_order = order
        self._is_compiled = True
        
    def dispatch(self, event_name: str, *args, **kwargs):
        """Dispatches an event through the DAG in topological order."""
        if not self._is_compiled:
            self.compile()
            
        for node in self._execution_order:
            method = getattr(node, event_name, None)
            # Only call if the subclass actually overrode it, or if it exists
            # In our case it always exists on the base class, but we can call it.
            if method and method.__func__ != getattr(InferenceHook, event_name):
                method(*args, **kwargs)
                
    def dispatch_reduce(self, event_name: str, initial_value: Any, *args, **kwargs) -> Any:
        """Dispatches an event through the DAG, threading the return value."""
        if not self._is_compiled:
            self.compile()
            
        value = initial_value
        for node in self._execution_order:
            method = getattr(node, event_name, None)
            if method and method.__func__ != getattr(InferenceHook, event_name):
                value = method(value, *args, **kwargs)
        return value

class PipeliningHook(InferenceHook):
    """
    Detects model type and configures the appropriate Pipelined strategy.
    """
    def __init__(self, config):
        self.config = config
        self._executor = None
        self._moe_prefetcher = None
        
    @property
    def dependencies(self) -> List[str]:
        # Pipelining needs to happen after Tiling structural modifications
        return ["TilingHook"]
        
    def on_generation_start(self, ctx: ExecutionContext):
        if not self.config.pipelined_execution:
            return
            
        # Initialize Executor and Prefetchers lazily
        if self._executor is None:
            from .strategies import PipelinedDenseStrategy, PipelinedMoEStrategy
            from ..pipeline.executor import PipelinedExecutor
            
            # Find the mmap_cache from the model or engine
            mmap_cache = getattr(ctx.engine, 'mmap_cache', None)
            if mmap_cache is None and hasattr(ctx.engine.model, 'manager'):
                 mmap_cache = getattr(ctx.engine.model.manager.model, 'mmap_cache', None)
            
            self._executor = PipelinedExecutor(mmap_cache)
            
            # Setup MoE Prefetcher if needed
            from ..moe.manager import MoEPrefetcher, ExpertCache
            # We need a cache for the prefetcher
            moe_cache = ExpertCache(max_experts=8) # Placeholder
            self._moe_prefetcher = MoEPrefetcher(mmap_cache.prefetch_worker if mmap_cache else None, moe_cache)
            
            self._dense_strategy = PipelinedDenseStrategy(self._executor)
            self._moe_strategy = PipelinedMoEStrategy(self._executor, self._moe_prefetcher)

        self._executor.disable_prefetch = getattr(ctx.engine, '_is_warmup', False)

        # Assign strategies per layer
        for i, layer in enumerate(ctx.engine.layers):
            # Detect MoE layer
            is_moe = False
            mlp = getattr(layer, "mlp", getattr(layer, "mixer", None))
            if mlp is not None and (hasattr(mlp, "gate") or hasattr(mlp, "router")):
                is_moe = True
            
            strategy = self._moe_strategy if is_moe else self._dense_strategy
            ctx.metadata[f'strategy_{i}'] = strategy

    def on_layer_start(self, ctx: ExecutionContext, layer: nn.Module):
        if self._executor:
            # Enqueue next layer's weights if we are pipelining
            # This is a simple lookahead. Real system might do N layers.
            next_idx = ctx.layer_idx
            self._executor._enqueue_tensor(next_idx, "all")

class TilingHook(InferenceHook):
    """
    Applies tiled execution to the model layers to bound peak memory usage.
    """
    def __init__(self, config):
        self.config = config
        
    def on_model_load(self, model: nn.Module) -> nn.Module:
        if not self.config.tiled_execution:
            return model
            
        from ..tiled import apply_tiling
        apply_tiling(model, tile_size=self.config.tile_size)
        return model

class DiagnosticsHook(InferenceHook):
    """
    Automates profiling and bottleneck analysis.
    """
    def __init__(self, config):
        self.config = config

    @property
    def dependencies(self) -> List[str]:
        # Diagnostics should run absolutely last, after all actual engine/hook execution
        return ["PipeliningHook", "TilingHook"]

    def on_generation_start(self, ctx: ExecutionContext):
        pass

    def on_generation_end(self, ctx: ExecutionContext):
        if getattr(self.config, 'debug', False):
            from benchmarks.profiler.profiler import StreamingProfiler
            prof = StreamingProfiler()
            print(f"[flash] End of generation. Intervals: IO={len(prof.io_intervals)}, Comp={len(prof.compute_intervals)}")
            print(prof.analyze_bottlenecks())
            prof.print_waterfall()

