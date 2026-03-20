"""Tests for FlashModelLoader."""

import numpy as np
import pytest

from mlx_engine_flash.loader import FlashModelLoader


def test_loader_detects_layers(tmp_model_dir, flash_config):
    loader = FlashModelLoader(tmp_model_dir, flash_config)
    assert loader.n_layers == 2


def test_loader_detects_dense(tmp_model_dir, flash_config):
    loader = FlashModelLoader(tmp_model_dir, flash_config)
    assert loader.is_moe is False


def test_loader_iter_layers(tmp_model_dir, flash_config):
    with FlashModelLoader(tmp_model_dir, flash_config) as loader:
        layers_seen = []
        for layer_idx, weights in loader.iter_layers():
            layers_seen.append(layer_idx)
            assert isinstance(weights, dict)
            assert all(isinstance(v, np.ndarray) for v in weights.values())
        assert layers_seen == [0, 1]


def test_loader_non_layer_weights(tmp_model_dir, flash_config):
    with FlashModelLoader(tmp_model_dir, flash_config) as loader:
        non_layer = loader.get_non_layer_weights()
        assert "model.embed_tokens.weight" in non_layer
        assert "lm_head.weight" in non_layer


def test_loader_layer_weights_contain_projections(tmp_model_dir, flash_config):
    with FlashModelLoader(tmp_model_dir, flash_config) as loader:
        w = loader.get_layer_weights(0)
        keys = set(w.keys())
        assert any("self_attn" in k for k in keys)
        assert any("mlp" in k for k in keys)


def test_loader_quant_validation_warn(tmp_model_dir, flash_config):
    """min_quant_bits=16 should warn since model has Q4_0."""
    import warnings
    flash_config.min_quant_bits = 16
    flash_config.strict_quant = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loader = FlashModelLoader(tmp_model_dir, flash_config)
        loader.open()
        loader.close()
    assert any("4-bit" in str(warning.message) or "quantis" in str(warning.message).lower()
               for warning in w), "Expected quant warning"


def test_loader_quant_validation_strict(tmp_model_dir, flash_config):
    flash_config.min_quant_bits = 16
    flash_config.strict_quant = True
    with pytest.raises(ValueError, match="quantis"):
        loader = FlashModelLoader(tmp_model_dir, flash_config)
        loader.open()
