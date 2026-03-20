"""Tests for macOS page cache management (madvise wrappers)."""

import mmap
import os
import sys

import pytest

from mlx_engine_flash.page_cache import (
    MADV_WILLNEED,
    madvise_range,
    prefetch,
    release,
    set_sequential,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Page cache hints are macOS-specific here"
)

@pytest.fixture
def tmp_mmap(tmp_path):
    """Create a temporary file-backed mmap for testing."""
    p = tmp_path / "test_weights.bin"
    data = b"\x00" * (4 * 1024 * 1024)  # 4 MB
    p.write_bytes(data)
    fd = os.open(str(p), os.O_RDONLY)
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    yield mm
    mm.close()
    os.close(fd)


def test_prefetch_returns_bool(tmp_mmap):
    result = prefetch(tmp_mmap, 0, 1024 * 1024)
    assert isinstance(result, bool)


def test_release_returns_bool(tmp_mmap):
    result = release(tmp_mmap, 0, 1024 * 1024, strategy="free")
    assert isinstance(result, bool)


def test_set_sequential_returns_bool(tmp_mmap):
    result = set_sequential(tmp_mmap, 0, 1024 * 1024)
    assert isinstance(result, bool)


def test_madvise_aligned_offset(tmp_mmap):
    # Non-page-aligned offset should be rounded down without error
    result = madvise_range(tmp_mmap, 513, 8192, MADV_WILLNEED)
    assert isinstance(result, bool)


def test_release_strategies(tmp_mmap):
    for strat in ("free", "dontneed", "none"):
        result = release(tmp_mmap, 0, 4096, strategy=strat)
        assert isinstance(result, bool)
