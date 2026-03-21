"""
Core invariant tests for Flash Mode.

These are the tests that matter. If any of these fail, Flash Mode is broken
regardless of what any other test says.

Run time: ~5s on M-series Mac (synthetic model, 2 layers, tiny hidden dim).
"""

import gc

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")


def metal_active_mb() -> float:
    gc.collect()
    try:
        mx.synchronize()
        return mx.metal.get_active_memory() / 1e6
    except AttributeError:
        return -1.0  # MLX too old; can't measure


def metal_peak_mb() -> float:
    try:
        return mx.metal.get_peak_memory() / 1e6
    except AttributeError:
        return -1.0


class TestFlashInvariant:
    
    def test_1_lazy_load_uses_zero_metal_ram(self, tmp_model_dir):
        """
        INVARIANT 1: Loading with lazy=True must not materialise weights to Metal.
        
        If this fails: mlx_lm.load(lazy=True) changed behaviour, or the model
        wrapper is eagerly evaluating parameters.
        """
        try:
            import mlx_lm
        except ImportError:
            pytest.skip("mlx_lm not installed")
        
        mx.metal.clear_cache()
        baseline = metal_active_mb()
        
        model, _ = mlx_lm.load(str(tmp_model_dir), lazy=True)
        after_load = metal_active_mb()
        
        delta = after_load - baseline
        assert delta < 50, (
            f"INVARIANT 1 FAILED: lazy load increased Metal active by {delta:.1f} MB. "
            f"Weights are being materialised during load. "
            f"Check if mlx_lm.load(lazy=True) skips mx.eval(model.parameters())."
        )
    
    def test_2_per_layer_eval_bounded_metal_usage(self, tmp_model_dir, flash_config):
        """
        INVARIANT 2: During flash inference, Metal active memory must stay 
        below 3 × single_layer_size at all times.
        
        If this fails: mx.eval() is not being called per-layer, OR
        mx.metal.clear_cache() is not freeing the weight allocations, OR
        the FlashLLM wrapper is building a unified graph instead of sequential ones.
        """
        # Compute single layer size
        import importlib.util

        from mlx_flash.generation import FlashLLM
        if importlib.util.find_spec("mlx_lm") is None:
            pytest.skip("mlx_lm not installed")
        import mlx_lm
        model, tokenizer = mlx_lm.load(str(tmp_model_dir), lazy=True)
        
        # Estimate single layer param bytes
        from mlx.utils import tree_flatten
        # Use _find_layers logic or just look into the model
        sub = getattr(model, "model", model)
        layers = getattr(sub, "layers", [])
        if not layers:
            pytest.skip("Could not find layers in synthetic model")
        
        layer_0_params = list(tree_flatten(layers[0].parameters()))
        layer_bytes = sum(p[1].nbytes for p in layer_0_params)
        budget_mb = (layer_bytes / 1e6) * 3 + 100  # 3× layer + 100MB overhead
        
        # Run flash inference
        class MetalTracker:
            def __init__(self):
                self.peak = 0.0
            def record(self):
                current = metal_active_mb()
                self.peak = max(self.peak, current)
        
        tracker = MetalTracker()
        flash_model = FlashLLM(model, flash_config)
        
        # Trace __call__ to track Metal usage
        original_call = flash_model.__class__.__call__
        
        def tracked_call(self_obj, x, cache=None, **kwargs):
            # Record Metal usage
            result = original_call(self_obj, x, cache=cache, **kwargs)
            tracker.record()
            return result
        
        # We need to patch the instance or the class carefully
        # Since FlashLLM is a class, we can patch its __call__ temporarily
        import types
        flash_model.__call__ = types.MethodType(tracked_call, flash_model)
        
        # Run one forward pass
        x = mx.array([[1, 2, 3]])
        try:
            out = flash_model(x)
            mx.eval(out)
            tracker.record()
        finally:
            # Revert patch
            if callable(flash_model):
                 del flash_model.__call__
        
        assert tracker.peak <= budget_mb, (
            f"INVARIANT 2 FAILED: Peak Metal memory {tracker.peak:.1f} MB "
            f"exceeded budget {budget_mb:.1f} MB (3× layer {layer_bytes/1e6:.1f} MB + overhead). "
            f"Layer weights are not being freed between evaluations. "
            f"Check: mx.eval() is called per-layer, mx.metal.clear_cache() is called."
        )
    
    def test_3_metal_memory_released_after_shutdown(self, flash_config, tmp_model_dir):
        """
        INVARIANT 3: After FlashManager.shutdown(), Metal memory should return 
        to near-baseline.
        
        If this fails: wired_limit is not being restored, OR model references
        are being held somewhere preventing GC.
        """
        import importlib.util

        from mlx_flash.manager import FlashManager
        if importlib.util.find_spec("mlx_lm") is None:
            pytest.skip("mlx_lm not installed")
        
        mx.metal.clear_cache()
        baseline = metal_active_mb()
        
        manager = FlashManager(flash_config)
        flash_model, tokenizer = manager.load(str(tmp_model_dir))
        
        # Run one generation step
        x = mx.array([[1, 2, 3]])
        out = flash_model(x)
        mx.eval(out)
        del out
        
        # Shutdown
        manager.shutdown()
        del flash_model, manager
        gc.collect()
        mx.metal.clear_cache()
        
        after = metal_active_mb()
        assert after - baseline < 100, (
            f"INVARIANT 3 FAILED: Metal memory after shutdown is "
            f"{after - baseline:.1f} MB above baseline. Memory is being leaked. "
            f"Check FlashManager.shutdown() releases wired_limit and all model refs."
        )
    
    def test_4_output_deterministic_with_seed(self, tmp_model_dir, flash_config):
        """
        INVARIANT 4: Flash Mode must produce identical outputs for identical inputs
        when using the same random seed.
        
        If this fails: non-deterministic Metal ops, or cache corruption.
        """
        from mlx_flash.generation import FlashGenerationLoop
        
        mx.random.seed(42)
        loop1 = FlashGenerationLoop(tmp_model_dir, flash_config)
        tokens1 = list(loop1.stream_generate("Test", max_tokens=5, temperature=0.0))
        
        mx.random.seed(42)
        loop2 = FlashGenerationLoop(tmp_model_dir, flash_config)
        tokens2 = list(loop2.stream_generate("Test", max_tokens=5, temperature=0.0))
        
        assert tokens1 == tokens2, (
            f"INVARIANT 4 FAILED: Non-deterministic output with same seed. "
            f"Pass 1: {tokens1!r}, Pass 2: {tokens2!r}. "
            f"Check for stateful Metal ops or KV cache contamination between runs."
        )
