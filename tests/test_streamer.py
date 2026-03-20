"""Tests for SafetensorsIndex and WeightStreamer."""

import numpy as np

from mlx_engine_flash.streamer import SafetensorsIndex, WeightStreamer


def test_index_loads(tmp_model_dir, flash_config):
    idx = SafetensorsIndex(tmp_model_dir)
    assert len(idx.tensor_names()) > 0


def test_index_contains_tensors(tmp_model_dir):
    idx = SafetensorsIndex(tmp_model_dir)
    assert "model.embed_tokens.weight" in idx
    assert "lm_head.weight" in idx


def test_index_layer_names(tmp_model_dir):
    idx = SafetensorsIndex(tmp_model_dir)
    layer0 = idx.layer_tensor_names(0)
    layer1 = idx.layer_tensor_names(1)
    assert len(layer0) > 0
    assert len(layer1) > 0
    # Layer names should be disjoint
    assert set(layer0).isdisjoint(set(layer1))


def test_index_n_layers(tmp_model_dir):
    idx = SafetensorsIndex(tmp_model_dir)
    assert idx.n_layers == 2


def test_streamer_stream_one(tmp_model_dir, flash_config):
    with WeightStreamer(tmp_model_dir, flash_config) as streamer:
        arr = streamer.stream_tensor("model.embed_tokens.weight")
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (256, 256)


def test_streamer_stream_layer(tmp_model_dir, flash_config):
    with WeightStreamer(tmp_model_dir, flash_config) as streamer:
        names = streamer.index.layer_tensor_names(0)
        result = streamer.stream_tensors(names)
        assert len(result) == len(names)
        for _name, arr in result.items():
            assert isinstance(arr, np.ndarray)


def test_streamer_parallel_consistent(tmp_model_dir, flash_config):
    """Parallel reads should produce the same result as sequential."""
    with WeightStreamer(tmp_model_dir, flash_config) as streamer:
        names = streamer.index.layer_tensor_names(0)
        # Parallel
        parallel = streamer.stream_tensors(names)
        # Sequential reference
        for name in names:
            single = streamer.stream_tensor(name)
            np.testing.assert_array_equal(parallel[name], single,
                                           err_msg=f"Mismatch for {name}")


def test_streamer_prefetch_release(tmp_model_dir, flash_config):
    """Prefetch and release should not raise even if madvise is unavailable."""
    with WeightStreamer(tmp_model_dir, flash_config) as streamer:
        streamer.prefetch_layer(0)
        streamer.release_layer(0)   # should be a no-op or succeed silently


def test_q4_0_decode_shape(tmp_model_dir, flash_config):
    with WeightStreamer(tmp_model_dir, flash_config) as streamer:
        # Q4_0 weights return raw uint8
        arr = streamer.stream_tensor("model.layers.0.self_attn.q_proj.weight")
        # Raw Q4_0 block data: each block = 18 bytes
        assert arr.dtype == np.uint8
