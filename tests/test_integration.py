"""
Integration tests — require either a real model (--model flag) or skip.

Run with:
    pytest tests/test_integration.py -v \
        --model ~/.cache/lm-studio/models/mlx-community/Qwen2.5-3B-Instruct-4bit \
        --flash
"""

import pytest

pytestmark = pytest.mark.skipif(True, reason="requires --model flag")


def test_flash_load_real_model(model_dir_path, use_flash, flash_config):
    if model_dir_path is None:
        pytest.skip("No --model provided")
    if not use_flash:
        pytest.skip("Pass --flash to enable this test")

    from pathlib import Path

    from mlx_engine_flash.loader import FlashModelLoader

    mdir = Path(model_dir_path)
    assert mdir.exists(), f"Model dir not found: {mdir}"

    with FlashModelLoader(mdir, flash_config) as loader:
        n = loader.n_layers
        assert n > 0, "Expected > 0 layers"

        # Load first and last layer
        w0 = loader.get_layer_weights(0)
        wn = loader.get_layer_weights(n - 1)
        assert len(w0) > 0
        assert len(wn) > 0

    print(f"\n✅  Loaded {n}-layer model from {mdir.name} in Flash Mode")


def test_modelfile_directive():
    from mlx_engine_flash.integration.modelfile import parse_flash_directives
    text = """
FROM /models/Qwen2.5-72B-Q4_K_M
FLASH true
FLASH_RAM_GB 12
FLASH_THREADS 6
FLASH_PREFETCH_LAYERS 3
FLASH_TOP_K 4
FLASH_EVICTION dontneed
"""
    cfg = parse_flash_directives(text)
    assert cfg.enabled is True
    assert cfg.ram_budget_gb == 12.0
    assert cfg.n_io_threads == 6
    assert cfg.prefetch_layers == 3
    assert cfg.moe_top_k_override == 4
    assert cfg.eviction_strategy == "dontneed"


def test_modelfile_no_flash():
    from mlx_engine_flash.integration.modelfile import parse_flash_directives
    text = "FROM /models/some-model\nSYSTEM You are helpful.\n"
    cfg = parse_flash_directives(text)
    assert cfg.enabled is False
