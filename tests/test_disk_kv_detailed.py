import shutil
import unittest
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx_flash.disk_kv_cache import DiskKVCache

class TestDiskKVDetailed(unittest.TestCase):
    def setUp(self):
        self.cache_dir = Path("/tmp/test_disk_kv_detailed")
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True)
        
        # Test shapes
        self.batch_size = 1
        self.n_heads = 4
        self.head_dim = 64
        self.dtype = mx.float16
        
    def tearDown(self):
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)

    def _make_dummy_kv(self, seq_len):
        k = mx.random.uniform(shape=(self.batch_size, self.n_heads, seq_len, self.head_dim), dtype=self.dtype)
        v = mx.random.uniform(shape=(self.batch_size, self.n_heads, seq_len, self.head_dim), dtype=self.dtype)
        return k, v

    def test_basic_append_and_fetch(self):
        with DiskKVCache(layer_idx=0, cache_dir=str(self.cache_dir)) as cache:
            # First append
            k1, v1 = self._make_dummy_kv(10)
            rk1, rv1 = cache.update_and_fetch(k1, v1)
            
            self.assertEqual(cache.size(), 10)
            self.assertEqual(rk1.shape, (self.batch_size, self.n_heads, 10, self.head_dim))
            
            # Second append
            k2, v2 = self._make_dummy_kv(5)
            rk2, rv2 = cache.update_and_fetch(k2, v2)
            
            self.assertEqual(cache.size(), 15)
            self.assertEqual(rk2.shape, (1, 4, 15, 64))
            
            # Verify data matches (MLX might evaluate lazily, so we check values)
            full_k = mx.concatenate([k1, k2], axis=2)
            np.testing.assert_allclose(np.array(rk2), np.array(full_k), atol=1e-3)

    def test_trim_functionality(self):
        with DiskKVCache(layer_idx=0, cache_dir=str(self.cache_dir)) as cache:
            k, v = self._make_dummy_kv(20)
            cache.update_and_fetch(k, v)
            
            # Trim 5 tokens from the end
            trimmed = cache.trim(5)
            self.assertEqual(trimmed, 5)
            self.assertEqual(cache.size(), 15)
            self.assertEqual(cache.keys.shape[2], 15)
            
            # Verify the file header was updated (we can check by reloading)
            # DiskKVCache updates headers and reloads internally on trim.
            np.testing.assert_allclose(np.array(cache.keys), np.array(k[:, :, :15, :]), atol=1e-3)

    def test_max_tokens_eviction(self):
        # Set max_tokens to 20. When we hit 21, it should halve to 10 tail tokens + 1 new = 11.
        with DiskKVCache(layer_idx=0, cache_dir=str(self.cache_dir), max_tokens=20) as cache:
            k1, v1 = self._make_dummy_kv(15)
            cache.update_and_fetch(k1, v1)
            self.assertEqual(cache.size(), 15)
            
            # Append 6 more tokens, total 21 > 20.
            # Logic: keep = 20 // 2 = 10. tail_k = k1[-10:]. Append 6. Final = 16.
            k2, v2 = self._make_dummy_kv(6)
            rk, rv = cache.update_and_fetch(k2, v2)
            
            self.assertEqual(cache.size(), 16)
            # Verify it contains the end of k1 and all of k2
            expected_k = mx.concatenate([k1[:, :, -10:, :], k2], axis=2)
            np.testing.assert_allclose(np.array(rk), np.array(expected_k), atol=1e-3)

    def test_resource_management_closing(self):
        cache = DiskKVCache(layer_idx=0, cache_dir=str(self.cache_dir))
        k, v = self._make_dummy_kv(5)
        cache.update_and_fetch(k, v)
        
        self.assertIsNotNone(cache.fd_k)
        cache.close()
        self.assertIsNone(cache.fd_k)
        
        # Should be able to re-open or finish without crash
        cache.close() # Idempotent

    def test_nbytes_reporting(self):
        with DiskKVCache(layer_idx=0, cache_dir=str(self.cache_dir)) as cache:
            k, v = self._make_dummy_kv(10)
            cache.update_and_fetch(k, v)
            
            # B=1, H=4, S=10, D=64, Dtype=F16 (2 bytes)
            # Total bytes = 1 * 4 * 10 * 64 * 2 (K) + 1 * 4 * 10 * 64 * 2 (V)
            expected = 1 * 4 * 10 * 64 * 2 * 2 
            self.assertEqual(cache.nbytes, expected)

if __name__ == "__main__":
    unittest.main()
