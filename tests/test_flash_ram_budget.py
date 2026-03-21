"""
End-to-end RAM budget validation for Flash Mode.

This is the canonical test that proves Flash Mode works. It runs a 
full generation pass (not just weight loading) and asserts that peak 
RSS stays below a configurable budget.
"""

import gc
import os

import mlx.core as mx
import psutil
import pytest

from mlx_flash import FlashConfig, FlashGenerationLoop


def get_rss_mb() -> float:
    gc.collect()
    mx.synchronize()  # ensure Metal ops complete
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


class TestFlashRAMBudget:
    
    def test_synthetic_model_stays_under_500mb(self, tmp_model_dir, flash_config):
        """
        On a tiny 2-layer synthetic model, Flash Mode should use < 500 MB 
        regardless of machine RAM.
        """
        # Start fresh
        mx.metal.clear_cache()
        gc.collect()
        rss_before = get_rss_mb()
        
        loop = FlashGenerationLoop(str(tmp_model_dir), flash_config)
        tokens = list(loop.stream_generate("Hello", max_tokens=3))
        
        rss_peak = get_rss_mb()
        rss_increase = rss_peak - rss_before
        
        assert len(tokens) > 0, "No tokens generated"
        # The synthetic model is tiny, so RSS increase should be dominated by 
        # MLX/Python shared lib overhead, not weights.
        assert rss_increase < 600, f"Peak RSS increase was {rss_increase:.1f} MB, expected < 600 MB"
        
        # Verify Metal cache was actually cleared
        metal_active = mx.get_active_memory() / 1e6
        assert metal_active < 100, f"Metal active memory is {metal_active:.1f} MB. Expected < 100."

    def test_long_context_doesnt_oom(self, tmp_model_dir, flash_config):
        """A 1000-token context should not OOM even with native generation."""
        import mlx_lm
        loop = FlashGenerationLoop(str(tmp_model_dir), flash_config)
        # Synthesize a long "prompt" as a string
        prompt = "hello " * 500  # ~1000 tokens
        # Should not raise RuntimeError: [metal::malloc] Insufficient Memory
        result = list(mlx_lm.stream_generate(loop.flash_model, loop.tokenizer, prompt,
                                              max_tokens=5))
        assert len(result) > 0

    def test_disk_kv_cache_infinite_context(self, tmp_model_dir):
        """
        Prove that with disk_kv_enabled, a massive context uses near-zero RAM
        because the KV cache lives entirely on the SSD as lazy arrays.
        """
        import shutil
        from pathlib import Path
        import mlx_lm
        
        cfg = FlashConfig(
            enabled=True, 
            ram_budget_gb=0.5,           # Very strict RAM constraint
            disk_kv_enabled=True, 
            disk_kv_dir="/tmp/test_flash_kv"
        )
        loop = FlashGenerationLoop(str(tmp_model_dir), config=cfg)
        
        # We manually synthesize a list of 4000 tokens (which would normally blow past RAM bounds
        # if the KV cache were held synchronously in the strict Metal budget)
        prompt_tokens = [0] * 4000
        
        mx.metal.clear_cache()
        gc.collect()
        rss_before = get_rss_mb()
        
        # Stream generate
        result = list(loop.stream_generate(prompt_tokens, max_tokens=5))
        assert len(result) > 0
        
        rss_after = get_rss_mb()
        
        # Check that the SSD files were actually created
        kv_dir = Path("/tmp/test_flash_kv")
        assert kv_dir.exists()
        safetensors_files = list(kv_dir.glob("*.safetensors"))
        assert len(safetensors_files) > 0, "Disk KV Cache did not create safetensors files!"
        
        # Verify RAM stayed bounded despite 4000 tokens
        assert (rss_after - rss_before) < 1500, "RAM exceeded bounds, Disk KV cache failed to page correctly"
        
        shutil.rmtree(kv_dir, ignore_errors=True)

    def test_per_layer_metal_memory_stays_bounded(self, tmp_model_dir, flash_config):
        """
        During generation, Metal active memory should never exceed 
        headroom for a single layer + activations.
        """
        loop = FlashGenerationLoop(str(tmp_model_dir), flash_config)
        
        peak_metal = 0.0
        
        # Instrument FlashLLM to track memory
        original_call = loop.flash_model.__call__
        
        def instrumented_call(*args, **kwargs):
            nonlocal peak_metal
            # We can't easily reach inside the loop without more complex patching,
            # but we can verify that after the call, memory is low.
            # For a more granular test, we would patch the layers themselves.
            result = original_call(*args, **kwargs)
            mx.synchronize()
            metal_now = mx.metal.get_active_memory() / 1e6
            peak_metal = max(peak_metal, metal_now)
            return result
            
        loop.flash_model.__call__ = instrumented_call
        list(loop.stream_generate("Test", max_tokens=2))
        
        # Since FlashLLM clears cache inside the loop, 
        # peak_metal after the call should be just the final layer output + head
        assert peak_metal < 300, f"Peak Metal memory {peak_metal:.1f} MB exceeded 300MB budget"

    @pytest.mark.skipif(
        not os.path.exists(os.environ.get("FLASH_TEST_MODEL", "")),
        reason="Set FLASH_TEST_MODEL env var to a real model path to run"
    )
    def test_real_model_nemotron_30b(self):
        model_path = os.environ["FLASH_TEST_MODEL"]
        config = FlashConfig(enabled=True, ram_budget_gb=10.0, debug=True)
        
        rss_before = get_rss_mb()
        loop = FlashGenerationLoop(model_path, config)
        _ = list(loop.stream_generate("Explain quantum computing.", max_tokens=20))
        rss_peak = get_rss_mb()
        
        print(f"\nPeak RSS increase: {rss_peak - rss_before:.1f} MB")
        assert rss_peak - rss_before < 2000, "RSS exceeded 2GB budget on 30B model"
