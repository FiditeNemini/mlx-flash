import contextlib
import json
import struct
from pathlib import Path
from typing import IO, cast, Optional, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache


class QuantizedDiskKVCache(KVCache):
    """
    Quantized Infinite Disk-Backed KV Cache.
    
    Compresses Key and Value tensors to 4-bit block-wise quantization before writing
    to disk, reducing I/O bandwidth requirements by ~75% during streaming prefill.
    
    Tensors are dequantized automatically on the fly when requested by the Attention
    kernel.
    """

    def __init__(self, layer_idx: int, cache_dir: str = "/tmp/mlx_flash_kv",
                 max_tokens: Optional[int] = None, bits: int = 4, group_size: int = 64):
        super().__init__()
        self.layer_idx = layer_idx
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.bits = bits
        self.group_size = group_size
        self._max_tokens = max_tokens
        
        # We need 4 files: K Data, K Scales, V Data, V Scales
        self.k_data_path = self.cache_dir / f"L{self.layer_idx}_k_data.safetensors"
        self.k_scales_path = self.cache_dir / f"L{self.layer_idx}_k_scales.safetensors"
        self.v_data_path = self.cache_dir / f"L{self.layer_idx}_v_data.safetensors"
        self.v_scales_path = self.cache_dir / f"L{self.layer_idx}_v_scales.safetensors"

        # Clean up any old run
        for p in (self.k_data_path, self.k_scales_path, self.v_data_path, self.v_scales_path):
            if p.exists():
                p.unlink()

        self.offset = 0
        self.header_pad_size = 8192  

        self.fd_k_data: Optional[IO[bytes]] = None
        self.fd_k_scales: Optional[IO[bytes]] = None
        self.fd_v_data: Optional[IO[bytes]] = None
        self.fd_v_scales: Optional[IO[bytes]] = None

        self.base_k_shape: Optional[Tuple[int, ...]] = None
        self.base_v_shape: Optional[Tuple[int, ...]] = None
        
        # Original float dtype (e.g., float16)
        self.original_dtype = mx.float16

        self._closed = False
        self._exit_stack = contextlib.ExitStack()

        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None

    def close(self):
        if self._closed:
            return
        self._closed = True
        for fd in (self.fd_k_data, self.fd_k_scales, self.fd_v_data, self.fd_v_scales):
            if fd is not None:
                with contextlib.suppress(Exception):
                    fd.close()
        self._exit_stack.close()
        self.fd_k_data = None
        self.fd_k_scales = None
        self.fd_v_data = None
        self.fd_v_scales = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    def _init_files(self, k_shape, v_shape, dtype):
        self.original_dtype = dtype
        self.base_k_shape = k_shape
        self.base_v_shape = v_shape
        
        self.fd_k_data = self._exit_stack.enter_context(open(self.k_data_path, "wb+"))
        self.fd_k_scales = self._exit_stack.enter_context(open(self.k_scales_path, "wb+"))
        self.fd_v_data = self._exit_stack.enter_context(open(self.v_data_path, "wb+"))
        self.fd_v_scales = self._exit_stack.enter_context(open(self.v_scales_path, "wb+"))

        self._write_headers(0)

    def _get_quantized_shapes(self, seq_len: int, base_shape: tuple):
        # Base shape is [Batch, Heads, SeqLen, HeadDim]
        # Transposed for disk: [SeqLen, Batch, Heads, HeadDim]
        batch, heads, _, head_dim = base_shape
        
        # 4-bit data packing: 2 elements per byte (uint8)
        packed_head_dim = head_dim // 2 if self.bits == 4 else head_dim 
        data_shape = [seq_len, batch, heads, packed_head_dim]
        
        # Scales shape: 1 scale (float32 for disk safety) per group
        scales_head_dim = head_dim // self.group_size
        scales_shape = [seq_len, batch, heads, scales_head_dim]
        
        return data_shape, scales_shape

    def _write_header(self, fd: IO[bytes], name: str, shape: list, dtype_str: str, bytes_per_elem: int):
        import math
        n_bytes = math.prod(shape) * bytes_per_elem

        header = {
            name: {
                "dtype": dtype_str,
                "shape": shape,
                "data_offsets": [0, n_bytes]
            },
            "__metadata__": {"format": "pt"}
        }

        header_json = json.dumps(header).encode("utf-8")
        padded_json = header_json.ljust(self.header_pad_size, b" ")
        header_len_bytes = struct.pack("<Q", self.header_pad_size)

        fd.seek(0)
        fd.write(header_len_bytes)
        fd.write(padded_json)

    def _write_headers(self, seq_len: int):
        assert self.base_k_shape is not None and self.base_v_shape is not None
        assert self.fd_k_data is not None and self.fd_k_scales is not None
        assert self.fd_v_data is not None and self.fd_v_scales is not None
        
        k_data_shape, k_scales_shape = self._get_quantized_shapes(seq_len, self.base_k_shape)
        v_data_shape, v_scales_shape = self._get_quantized_shapes(seq_len, self.base_v_shape)
        
        # Data is uint8 (1 byte per elem)
        self._write_header(self.fd_k_data, "data", k_data_shape, "U8", 1)
        self._write_header(self.fd_v_data, "data", v_data_shape, "U8", 1)
        
        # Scales are float32 on disk (4 bytes per elem)
        self._write_header(self.fd_k_scales, "scales", k_scales_shape, "F32", 4)
        self._write_header(self.fd_v_scales, "scales", v_scales_shape, "F32", 4)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        # inputs are [Batch, Heads, NewSeqLen, HeadDim]
        new_seq = keys.shape[2]
        if self.fd_k_data is None:
            self._init_files(keys.shape, values.shape, keys.dtype)

        # 1. Transpose to [NewSeqLen, Batch, Heads, HeadDim]
        k_t = keys.transpose(2, 0, 1, 3)
        v_t = values.transpose(2, 0, 1, 3)

        # 2. Quantize just-in-time
        # Quantize operates on the last dimension (HeadDim)
        k_q, k_s, _ = mx.quantize(k_t, group_size=self.group_size, bits=self.bits)
        v_q, v_s, _ = mx.quantize(v_t, group_size=self.group_size, bits=self.bits)
        
        # Convert scales to float32 for safe disk write
        k_s_f32 = k_s.astype(mx.float32)
        v_s_f32 = v_s.astype(mx.float32)
        
        mx.eval(k_q, k_s_f32, v_q, v_s_f32)

        # 3. Append physical bytes
        assert self.fd_k_data is not None and self.fd_v_data is not None
        assert self.fd_k_scales is not None and self.fd_v_scales is not None
        
        self.fd_k_data.seek(0, 2)
        self.fd_k_data.write(np.asarray(k_q).tobytes())
        self.fd_k_scales.seek(0, 2)
        self.fd_k_scales.write(np.asarray(k_s_f32).tobytes())
        
        self.fd_v_data.seek(0, 2)
        self.fd_v_data.write(np.asarray(v_q).tobytes())
        self.fd_v_scales.seek(0, 2)
        self.fd_v_scales.write(np.asarray(v_s_f32).tobytes())

        self.offset += new_seq

        # 4. Rewrite JSON headers
        self._write_headers(self.offset)
        
        self.fd_k_data.flush()
        self.fd_k_scales.flush()
        self.fd_v_data.flush()
        self.fd_v_scales.flush()

        # 5. Native MLX Lazy Load
        loaded_k_data = mx.load(str(self.k_data_path))
        loaded_k_scales = mx.load(str(self.k_scales_path))
        loaded_v_data = mx.load(str(self.v_data_path))
        loaded_v_scales = mx.load(str(self.v_scales_path))
        
        lazy_k_data = cast(mx.array, loaded_k_data["data"])
        lazy_k_scales = cast(mx.array, loaded_k_scales["scales"])
        lazy_v_data = cast(mx.array, loaded_v_data["data"])
        lazy_v_scales = cast(mx.array, loaded_v_scales["scales"])

        # 6. Dequantize on the fly
        # Cast scales back to appropriate fp16 format for MLX dequantizer
        k_scales_hw = lazy_k_scales.astype(self.original_dtype)
        v_scales_hw = lazy_v_scales.astype(self.original_dtype)
        
        k_full = mx.dequantize(lazy_k_data, k_scales_hw, None, group_size=self.group_size, bits=self.bits)
        v_full = mx.dequantize(lazy_v_data, v_scales_hw, None, group_size=self.group_size, bits=self.bits)

        # 7. Transpose back to MLX attention format [Batch, Heads, SeqLen, HeadDim]
        self.keys = k_full.transpose(1, 2, 0, 3)
        self.values = v_full.transpose(1, 2, 0, 3)

        return self.keys, self.values

    # ── KVCache Contract ────────────────────────────────────────────────────

    def size(self):
        return self.offset

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        if self.keys is not None:
            self.offset = self.keys.shape[2]

    def is_trimmable(self):
        return False # Quantized cache eviction requires re-packing, omit for now.

    def trim(self, n):
        raise NotImplementedError("Trimming not yet supported on Quantized Disk Cache")

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None or self.base_k_shape is None:
            return 0
            
        # Disk usage calculation
        batch, heads, _, head_dim = self.base_k_shape
        
        data_bytes_per_tok = (batch * heads * head_dim) * (self.bits / 8)
        scales_bytes_per_tok = (batch * heads * (head_dim / self.group_size)) * 4 # FP32
        
        per_token = data_bytes_per_tok + scales_bytes_per_tok
        return int(self.offset * per_token * 2) # * 2 for K and V
