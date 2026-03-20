"""Tests for MoE routing and expert streaming."""

import numpy as np
import pytest

from mlx_engine_flash.moe import MoEConfig, MoERouter


@pytest.fixture
def small_moe_cfg():
    return MoEConfig(n_experts=8, top_k=2, n_layers=4)


@pytest.fixture
def random_router(small_moe_cfg):
    rng = np.random.default_rng(0)
    weights = rng.standard_normal((64, 8)).astype(np.float32)
    return MoERouter(weights, small_moe_cfg)


def test_moe_config_from_dict():
    cfg = MoEConfig.from_model_config({
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 32,
    })
    assert cfg is not None
    assert cfg.n_experts == 8
    assert cfg.top_k == 2


def test_moe_config_from_dict_none():
    cfg = MoEConfig.from_model_config({"num_hidden_layers": 32})
    assert cfg is None


def test_router_output_shape(random_router):
    hidden = np.random.randn(4, 64).astype(np.float32)  # 4 tokens
    indices, weights = random_router.route(hidden)
    assert indices.shape == (4, 2)
    assert weights.shape == (4, 2)


def test_router_weights_sum_to_one(random_router):
    hidden = np.random.randn(8, 64).astype(np.float32)
    indices, weights = random_router.route(hidden)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-5)


def test_router_indices_in_range(random_router, small_moe_cfg):
    hidden = np.random.randn(16, 64).astype(np.float32)
    indices, _ = random_router.route(hidden)
    assert indices.min() >= 0
    assert indices.max() < small_moe_cfg.n_experts


def test_router_top_k_distinct(random_router):
    """Top-K experts per token should be distinct."""
    hidden = np.random.randn(32, 64).astype(np.float32)
    indices, _ = random_router.route(hidden)
    for row in indices:
        assert len(set(row.tolist())) == len(row), "Duplicate experts in top-K"


def test_unique_experts(random_router):
    hidden = np.random.randn(4, 64).astype(np.float32)
    indices, _ = random_router.route(hidden)
    unique = random_router.unique_experts(indices)
    assert isinstance(unique, list)
    assert len(unique) <= 8
    assert sorted(unique) == unique  # should be sorted


def test_top_k_override():
    cfg = MoEConfig(n_experts=8, top_k=2, n_layers=4)
    weights = np.random.randn(64, 8).astype(np.float32)
    router = MoERouter(weights, cfg, top_k_override=4)
    hidden = np.random.randn(4, 64).astype(np.float32)
    indices, wts = router.route(hidden)
    assert indices.shape == (4, 4)  # K overridden to 4
