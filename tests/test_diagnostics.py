import pytest
from mlx_engine_flash.diagnostics import RAMProfiler

def test_ram_profiler_snapshot_valid():
    profiler = RAMProfiler()
    snap = profiler.snapshot("test")
    
    assert snap["label"] == "test"
    assert snap["rss_mb"] >= 0
    assert snap["metal_active_mb"] >= -1  # -1 if metal is not available
    assert snap["metal_peak_mb"] >= -1
    assert snap["page_cache_mb"] >= -1
    assert "timestamp" in snap
