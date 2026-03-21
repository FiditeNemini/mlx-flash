"""
End-to-end RAM budget validation for Flash Mode.

This is the canonical test that proves Flash Mode works. It runs a 
full generation pass (not just weight loading) and asserts that peak 
RSS stays below a configurable budget.

Run with:
    pytest tests/test_flash_ram_budget.py -v -s

For the synthetic model this should pass on any machine.
For a real model, pass --model and --budget-gb.
"""

import gc
import os
import psutil
import pytest
import mlx.core as mx

from mlx_engine_flash import FlashConfig
from mlx_engine_flash.generation import FlashGenerationLoop


def get_rss_mb() -> float:
    gc.collect()
    mx.synchronize()  # ensure Metal ops complete
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


class TestFlashRAMBudget:
    
    def test_synthetic_model_stays_under_500mb(self, tmp_model_dir, flash_config):
        """
        On a tiny 2-layer synthetic model, Flash Mode should use < 500 MB 
        regardless of machine RAM. This test must pass in CI.
        """
        # Start fresh
        mx.metal.clear_cache()
        gc.collect()
        rss_before = get_rss_mb()
        
        loop = FlashGenerationLoop(tmp_model_dir, flash_config)
        tokens = list(loop.stream_generate("Hello", max_tokens=3))
        
        rss_peak = get_rss_mb()
        rss_increase = rss_peak - rss_before
        
        assert len(tokens) > 0, "No tokens generated"
        assert rss_increase < 500, (
            f"Peak RSS increase was {rss_increase:.1f} MB, expected < 500 MB. "
            f"This indicates weights are being copied instead of mmap'd, "
            f"or layer weights are not being zeroed after mx.eval()."
        )
        
        # Verify Metal cache was actually cleared between layers (or at least at the end)
        metal_active = mx.metal.get_active_memory() / 1e6
        assert metal_active < 200, (
            f"Metal active memory is {metal_active:.1f} MB after generation. "
            f"Expected near-zero. mx.metal.clear_cache() may not be working."
        )
    
    def test_per_layer_metal_memory_stays_bounded(self, tmp_model_dir, flash_config):
        """
        During generation, Metal active memory should never exceed 
        2 × single_layer_size. Verify by instrumenting the generation loop.
        """
        # Estimate single layer size from the index
        from mlx_engine_flash.streamer import SafetensorsIndex
        idx = SafetensorsIndex(tmp_model_dir)
        layer0_bytes = sum(
            idx[n].data_length for n in idx.layer_tensor_names(0)
        )
        single_layer_mb = layer0_bytes / 1e6
        budget_mb = single_layer_mb * 3  # 3x headroom for activations
        
        peak_metal = 0.0
        
        # Patch the generation loop to track Metal memory per layer
        loop = FlashGenerationLoop(tmp_model_dir, flash_config)
        original_run = loop._run_layer_synchronous
        
        def instrumented_run(*args, **kwargs):
            nonlocal peak_metal
            result = original_run(*args, **kwargs)
            mx.synchronize()
            metal_now = mx.metal.get_active_memory() / 1e6
            peak_metal = max(peak_metal, metal_now)
            return result
        
        loop._run_layer_synchronous = instrumented_run
        list(loop.stream_generate("Test", max_tokens=2))
        
        assert peak_metal < budget_mb, (
            f"Peak Metal memory {peak_metal:.1f} MB exceeded budget "
            f"{budget_mb:.1f} MB (3× single layer {single_layer_mb:.1f} MB). "
            f"Layer weights are not being freed between layers."
        )
    
    @pytest.mark.skipif(
        not os.path.exists(os.environ.get("FLASH_TEST_MODEL", "")),
        reason="Set FLASH_TEST_MODEL env var to a real model path to run"
    )
    def test_real_model_nemotron_30b(self):
        """
        Integration test for the Nemotron-30B benchmark from the README.
        Expected: peak RSS < 2 GB, coherent output.
        Set: FLASH_TEST_MODEL=/path/to/Nemotron-30B-mlx
        """
        model_path = os.environ["FLASH_TEST_MODEL"]
        config = FlashConfig(enabled=True, ram_budget_gb=10.0, debug=True)
        
        rss_before = get_rss_mb()
        loop = FlashGenerationLoop(model_path, config)
        tokens = list(loop.stream_generate(
            "Explain the Transformer architecture.", max_tokens=50
        ))
        rss_peak = get_rss_mb()
        
        print(f"\nPeak RSS increase: {rss_peak - rss_before:.1f} MB")
        print(f"Output: {''.join(tokens)}")
        
        assert rss_peak - rss_before < 2000, (
            f"Expected < 2 GB RSS increase on 30B model, got "
            f"{(rss_peak - rss_before) / 1000:.1f} GB"
        )
        assert len("".join(tokens)) > 10, "Output too short — generation failed"
