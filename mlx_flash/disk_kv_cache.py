import os
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.models.cache import KVCache

class DiskKVCache(KVCache):
    """
    Infinite Disk-Backed KV Cache that seamlessly bypasses Metal RAM limits.
    
    Architecture:
    We maintain two padded `.safetensors` files per layer (one for keys, one for values).
    As new tokens arrive, we transpose their tensors to:
        [SeqLen, Batch, Heads, HeadDim] 
    This allows us to physically append raw bytes linearly to the end of the file
    and guarantee they represent a contiguous sequence block.
    
    We then update the `.safetensors` JSON header in-place.
    To evaluate, we `mx.load(..., lazy=True)` the file, providing perfect 
    zero-copy infinite context via the macOS unified page cache.
    """
    def __init__(self, layer_idx: int, cache_dir: str = "/tmp/mlx_flash_kv"):
        super().__init__()
        self.layer_idx = layer_idx
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.k_path = self.cache_dir / f"L{self.layer_idx}_k.safetensors"
        self.v_path = self.cache_dir / f"L{self.layer_idx}_v.safetensors"
        
        # Clean up any old run
        for p in (self.k_path, self.v_path):
            if p.exists():
                p.unlink()
                
        self.offset = 0
        self.header_pad_size = 8192  # 8 KB padded header is plenty
        
        self.fd_k = None
        self.fd_v = None
        
        self.k_dtype_str = None
        self.k_shape = None
        self.v_shape = None
        self.bytes_per_elem = 2

    def __del__(self):
        if self.fd_k: self.fd_k.close()
        if self.fd_v: self.fd_v.close()

    def _init_files(self, k_shape, v_shape, dtype):
        self.fd_k = open(self.k_path, "wb+")
        self.fd_v = open(self.v_path, "wb+")
        
        self.k_dtype_str = "F16" if dtype == mx.float16 else "F32"
        if dtype == mx.bfloat16:
            self.k_dtype_str = "BF16"
            
        self.bytes_per_elem = 2 if "16" in self.k_dtype_str else 4
        self.k_shape = k_shape
        self.v_shape = v_shape
        
        self._write_header(self.fd_k, "keys", 0, k_shape)
        self._write_header(self.fd_v, "values", 0, v_shape)

    def _write_header(self, fd, name, seq_len: int, base_shape: tuple):
        """Write the padded safetensors header to the start of the file."""
        # disk shape is [SeqLen, Batch, Heads, HeadDim]
        disk_shape = [seq_len, base_shape[0], base_shape[1], base_shape[3]]
        n_bytes = seq_len * base_shape[0] * base_shape[1] * base_shape[3] * self.bytes_per_elem
        
        header = {
            name: {
                "dtype": self.k_dtype_str,
                "shape": disk_shape,
                "data_offsets": [0, n_bytes]
            },
            "__metadata__": {"format": "pt"}
        }
        
        header_json = json.dumps(header).encode("utf-8")
        assert len(header_json) <= self.header_pad_size, f"{name} Header exceeded pad size!"
        padded_json = header_json.ljust(self.header_pad_size, b" ")
        
        header_len_bytes = struct.pack("<Q", self.header_pad_size)
        
        fd.seek(0)
        fd.write(header_len_bytes)
        fd.write(padded_json)
        fd.flush()

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        # inputs are [Batch, Heads, NewSeqLen, HeadDim]
        new_seq = keys.shape[2]
        if self.fd_k is None:
            self._init_files(keys.shape, values.shape, keys.dtype)
            
        # 1. Transpose down to [NewSeqLen, Batch, Heads, HeadDim]
        k_t = keys.transpose(2, 0, 1, 3)
        v_t = values.transpose(2, 0, 1, 3)
        
        # 2. Extract raw bytes (we force eval via np.array to get the buffer)
        # Note: on MLX Metal, np.array() pulls the tensor to CPU RAM. 
        # Since this is just 1 new token per step, CPU overhead is negligible (a few KB).
        k_bytes = np.array(k_t).tobytes()
        v_bytes = np.array(v_t).tobytes()
        
        # 3. Append physical bytes
        self.fd_k.seek(0, 2)
        self.fd_k.write(k_bytes)
        self.fd_k.flush()
        
        self.fd_v.seek(0, 2)
        self.fd_v.write(v_bytes)
        self.fd_v.flush()
        
        self.offset += new_seq
        
        # 4. Rewrite the JSON headers linearly
        self._write_header(self.fd_k, "keys", self.offset, self.k_shape)
        self._write_header(self.fd_v, "values", self.offset, self.v_shape)
        
        # 5. Native MLX Lazy Load the entire growing cache
        # shape is [TotalSeq, Batch, Heads, HeadDim]
        lazy_k_t = mx.load(str(self.k_path))["keys"]
        lazy_v_t = mx.load(str(self.v_path))["values"]
        
        # 6. Transpose back into expected MLX format [Batch, Heads, TotalSeq, HeadDim]
        # This is a purely metadata zero-copy operation in MLX
        self.keys = lazy_k_t.transpose(1, 2, 0, 3)
        self.values = lazy_v_t.transpose(1, 2, 0, 3)
        
        return self.keys, self.values

    @property
    def state(self):
        return self.keys, self.values
