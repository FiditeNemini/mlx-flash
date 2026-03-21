"""Tests for FlashConfig validation and construction."""

import pytest

from mlx_flash.config import FlashConfig


def test_defaults():
    cfg = FlashConfig()
    assert not cfg.enabled
    assert cfg.ram_budget_gb == 2.0

def test_enabled():
    cfg = FlashConfig(enabled=True)
    assert cfg.enabled is True

def test_validation_ram():
    with pytest.raises(ValueError, match="ram_budget_gb"):
        FlashConfig(ram_budget_gb=0.1)

def test_from_dict():
    cfg = FlashConfig.from_dict({
        "enabled": True,
        "ram_budget_gb": 8.0,
        "expert_cache_size": 16,
    })
    assert cfg.enabled is True
    assert cfg.ram_budget_gb == 8.0
    assert cfg.expert_cache_size == 16


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
