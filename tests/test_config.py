"""Tests for FlashConfig validation and construction."""

import pytest
from mlx_engine_flash.config import FlashConfig


def test_defaults():
    cfg = FlashConfig()
    assert cfg.enabled is False
    assert cfg.ram_budget_gb == 10.0
    assert cfg.n_io_threads == 4


def test_enabled():
    cfg = FlashConfig(enabled=True)
    assert cfg.enabled is True


def test_validation_ram():
    with pytest.raises(ValueError, match="ram_budget_gb"):
        FlashConfig(ram_budget_gb=0.5)


def test_validation_threads():
    with pytest.raises(ValueError, match="n_io_threads"):
        FlashConfig(n_io_threads=0)
    with pytest.raises(ValueError, match="n_io_threads"):
        FlashConfig(n_io_threads=64)


def test_from_dict():
    cfg = FlashConfig.from_dict({
        "enabled": True,
        "ram_budget_gb": 8.0,
        "n_io_threads": 6,
        "prefetch_layers": 3,
    })
    assert cfg.enabled is True
    assert cfg.ram_budget_gb == 8.0
    assert cfg.n_io_threads == 6
    assert cfg.prefetch_layers == 3


def test_from_dict_ignores_unknown():
    cfg = FlashConfig.from_dict({"enabled": True, "unknown_key": 999})
    assert cfg.enabled is True


def test_eviction_strategy_valid():
    for strat in ("dontneed", "free", "none"):
        cfg = FlashConfig(eviction_strategy=strat)
        assert cfg.eviction_strategy == strat


def test_repr():
    cfg = FlashConfig(enabled=True)
    r = repr(cfg)
    assert "FlashConfig" in r
    assert "enabled=True" in r
