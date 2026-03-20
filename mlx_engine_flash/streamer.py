"""
streamer.py — Parallel pread() weight streamer + safetensors lazy reader.

Architecture
------------
ParallelPreader
    Low-level: opens a file descriptor and issues multiple concurrent
    pread() calls using a thread pool.  pread() is:
      * Thread-safe (no shared seek position)
      * GIL-releasing in CPython (I/O syscalls release the GIL)
      * Zero-copy into bytes objects returned to caller

SafetensorsIndex
    Parses the safetensors header(s) for a model directory.  Builds a
    complete map of tensor name → (file_path, byte_offset, byte_length,
    dtype, shape).  Handles both single-file and sharded models.

WeightStreamer
    High-level: wraps ParallelPreader + SafetensorsIndex.  Provides
    `stream_tensors(names)` which issues parallel reads and returns a
    dict of NumPy arrays, ready to be wrapped in mx.array().
    Also manages madvise prefetch/release via PageCacheRegion.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import FlashConfig
from .page_cache import prefetch, release, set_sequential

# ── dtype mapping: safetensors dtype string → numpy dtype ─────────────────
def _get_st_dtype_map() -> Dict[str, np.dtype]:
    m = {
        "F64":  np.dtype("float64"),
        "F32":  np.dtype("float32"),
        "F16":  np.dtype("float16"),
        "I64":  np.dtype("int64"),
        "I32":  np.dtype("int32"),
        "I16":  np.dtype("int16"),
        "I8":   np.dtype("int8"),
        "U32":  np.dtype("uint32"),
        "U8":   np.dtype("uint8"),
        "BOOL": np.dtype("bool"),
    }
    # Try to add BF16 if available in numpy
    try:
        m["BF16"] = np.dtype("bfloat16")
    except TypeError:
        # Fallback to uint16; must be bitcast to BF16 when converted to MLX
        m["BF16"] = np.dtype("uint16")
    return m

_ST_DTYPE_MAP = _get_st_dtype_map()

# Quantised formats stored as raw uint8 blobs in safetensors
_QUANTISED_DTYPES = {"Q4_0", "Q4_1", "Q4_K", "Q6_K", "Q8_0", "Q2_K", "Q3_K"}


@dataclass(frozen=True)
class TensorEntry:
    """Metadata for one tensor in the safetensors index."""
    name: str
    file_path: Path
    data_offset: int     # absolute byte offset from start of file
    data_length: int     # byte length
    dtype: str           # safetensors dtype string
    shape: Tuple[int, ...]
    n_bits: int          # 4 / 8 / 16 / 32 — used for quant validation


def _parse_safetensors_header(path: Path) -> Tuple[dict, int]:
    """
    Read the safetensors header without loading the full file.
    Returns (header_dict, data_start_offset).
    """
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"File too small to be safetensors: {path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_bytes = f.read(header_len)
    header = json.loads(header_bytes.decode("utf-8"))
    data_start = 8 + header_len
    return header, data_start


def _quant_bits(dtype_str: str) -> int:
    """Best-effort extraction of effective bit-width from dtype string."""
    lower = dtype_str.lower()
    if "q2" in lower: return 2
    if "q3" in lower: return 3
    if "q4" in lower: return 4
    if "q5" in lower: return 5
    if "q6" in lower: return 6
    if "q8" in lower: return 8
    if lower in ("f16", "bf16"): return 16
    if lower == "f32": return 32
    return 8   # fallback


class SafetensorsIndex:
    """
    Index of all tensors across one or more safetensors shards.

    Handles:
    * Single-file models  (*.safetensors)
    * Sharded models      (model.safetensors.index.json + shards)
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._entries: Dict[str, TensorEntry] = {}
        self._mmaps: Dict[Path, mmap.mmap] = {}
        self._fds: Dict[Path, int] = {}
        self._lock = threading.Lock()
        self._load_index()

    def _load_index(self) -> None:
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            self._load_sharded(index_path)
        else:
            # Find all .safetensors files
            shards = sorted(self.model_dir.glob("*.safetensors"))
            if not shards:
                raise FileNotFoundError(
                    f"No .safetensors files found in {self.model_dir}"
                )
            for shard in shards:
                self._index_shard(shard)

    def _load_sharded(self, index_path: Path) -> None:
        idx = json.loads(index_path.read_text())
        weight_map: Dict[str, str] = idx.get("weight_map", {})
        shards_needed = set(weight_map.values())
        # Build per-shard header → entries
        shard_headers: Dict[str, Tuple[dict, int]] = {}
        for shard_name in shards_needed:
            shard_path = self.model_dir / shard_name
            header, data_start = _parse_safetensors_header(shard_path)
            shard_headers[shard_name] = (header, data_start)

        for tensor_name, shard_name in weight_map.items():
            header, data_start = shard_headers[shard_name]
            if tensor_name not in header or tensor_name == "__metadata__":
                continue
            meta = header[tensor_name]
            start, end = meta["data_offsets"]
            dtype_str = meta["dtype"]
            shape = tuple(meta["shape"])
            self._entries[tensor_name] = TensorEntry(
                name=tensor_name,
                file_path=self.model_dir / shard_name,
                data_offset=data_start + start,
                data_length=end - start,
                dtype=dtype_str,
                shape=shape,
                n_bits=_quant_bits(dtype_str),
            )

    def _index_shard(self, shard_path: Path) -> None:
        header, data_start = _parse_safetensors_header(shard_path)
        for tensor_name, meta in header.items():
            if tensor_name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            dtype_str = meta["dtype"]
            shape = tuple(meta["shape"])
            self._entries[tensor_name] = TensorEntry(
                name=tensor_name,
                file_path=shard_path,
                data_offset=data_start + start,
                data_length=end - start,
                dtype=dtype_str,
                shape=shape,
                n_bits=_quant_bits(dtype_str),
            )

    def open_mmaps(self) -> None:
        """Open memory-mapped views of all shard files (lazy)."""
        files_needed = {e.file_path for e in self._entries.values()}
        for path in files_needed:
            if path not in self._fds:
                fd = os.open(str(path), os.O_RDONLY)
                self._fds[path] = fd
                mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                self._mmaps[path] = mm
                # Hint: sequential access for the whole file during model loading
                set_sequential(mm, 0, mm.size())

    def close_mmaps(self) -> None:
        for mm in self._mmaps.values():
            try:
                mm.close()
            except BufferError:
                # Occurs if zero-copy arrays are still alive (e.g. in tests)
                pass
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._mmaps.clear()
        self._fds.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __getitem__(self, name: str) -> TensorEntry:
        return self._entries[name]

    def get(self, name: str) -> Optional[TensorEntry]:
        return self._entries.get(name)

    def tensor_names(self) -> List[str]:
        return list(self._entries.keys())

    def layer_tensor_names(self, layer_idx: int) -> List[str]:
        """Return all tensor names belonging to transformer layer *layer_idx*."""
        prefix = f"model.layers.{layer_idx}."
        return [n for n in self._entries if n.startswith(prefix)]

    def expert_tensor_names(self, layer_idx: int, expert_idx: int) -> List[str]:
        """Return tensor names for a specific MoE expert."""
        prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."
        return [n for n in self._entries if n.startswith(prefix)]

    def get_mmap(self, path: Path) -> mmap.mmap:
        with self._lock:
            if path not in self._mmaps:
                self.open_mmaps()
        return self._mmaps[path]

    @property
    def n_layers(self) -> int:
        """Heuristic: count unique layer indices."""
        idxs = set()
        for name in self._entries:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        idxs.add(int(parts[i + 1]))
                    except ValueError:
                        pass
        return max(idxs) + 1 if idxs else 0

    @property
    def min_quant_bits(self) -> int:
        bits = [e.n_bits for e in self._entries.values()]
        return min(bits) if bits else 16


# ─────────────────────────────────────────────────────────────────────────────

class MmapReader:
    """
    Reader that creates memory-mapped slices of files.
    
    This provides 'Zero-Copy' access to weights:
    * The OS maps the file into the process's address space.
    * No data is copied from the OS into a process-local buffer.
    * When madvise(MADV_FREE) is called, the OS immediately knows it
      can reclaim the physical pages from this process.
    """

    def __init__(self, index: "SafetensorsIndex") -> None:
        self.index = index

    def read_one(self, path: Path, offset: int, size: int) -> memoryview:
        mm = self.index.get_mmap(path)
        # Returns a zero-copy memoryview of the mmap'd region
        return memoryview(mm)[offset: offset + size]

    def read_many(
        self,
        path: Path,
        requests: List[Tuple[int, int]],  # (offset, size)
    ) -> List[memoryview]:
        # mmap-based 'reads' are just slice operations; no threads needed
        # as the actual I/O happens on-demand via page faults.
        return [self.read_one(path, off, sz) for off, sz in requests]

    def close(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────

class WeightStreamer:
    """
    High-level weight streamer: reads tensors by name from safetensors shards.

    Uses MmapReader to provide zero-copy memory regions backed by the
    macOS Unified Page Cache.
    """

    def __init__(self, model_dir: Path, config: FlashConfig) -> None:
        self.config = config
        self.index = SafetensorsIndex(model_dir)
        self.index.open_mmaps()
        self._reader = MmapReader(self.index)

    def stream_tensor(self, name: str) -> np.ndarray:
        """Read one tensor synchronously; returns a NumPy array."""
        entry = self.index[name]
        raw = self._reader.read_one(entry.file_path, entry.data_offset, entry.data_length)
        return self._decode(entry, raw)

    def stream_tensors(
        self,
        names: List[str],
        prefetch_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Read *names* in parallel (via page faults).
        Optionally issues a prefetch hint for *prefetch_names* (next layer).
        """
        # Issue prefetch hint for next batch before reading current
        if prefetch_names:
            for n in prefetch_names:
                entry = self.index.get(n)
                if entry is not None:
                    mm = self.index.get_mmap(entry.file_path)
                    prefetch(mm, entry.data_offset, entry.data_length)

        # Build per-file batches
        by_file: Dict[Path, List[Tuple[str, int, int]]] = {}
        for name in names:
            entry = self.index.get(name)
            if entry is None:
                continue
            by_file.setdefault(entry.file_path, []).append(
                (name, entry.data_offset, entry.data_length)
            )

        results: Dict[str, np.ndarray] = {}
        for fpath, batch in by_file.items():
            tensor_names = [b[0] for b in batch]
            requests = [(b[1], b[2]) for b in batch]
            raw_list = self._reader.read_many(fpath, requests)
            for tname, raw in zip(tensor_names, raw_list):
                entry = self.index[tname]
                results[tname] = self._decode(entry, raw)

        return results

    def prefetch_layer(self, layer_idx: int) -> None:
        """Issue async page-cache prefetch for all weights in layer *layer_idx*."""
        for name in self.index.layer_tensor_names(layer_idx):
            entry = self.index.get(name)
            if entry is not None:
                mm = self.index.get_mmap(entry.file_path)
                prefetch(mm, entry.data_offset, entry.data_length)

    def release_layer(self, layer_idx: int) -> None:
        """Release page-cache pages for layer *layer_idx*."""
        for name in self.index.layer_tensor_names(layer_idx):
            entry = self.index.get(name)
            if entry is not None:
                mm = self.index.get_mmap(entry.file_path)
                release(mm, entry.data_offset, entry.data_length,
                        self.config.eviction_strategy)

    def prefetch_experts(self, layer_idx: int, expert_idxs: List[int]) -> None:
        """Prefetch only the specified MoE experts for a layer."""
        for eidx in expert_idxs:
            for name in self.index.expert_tensor_names(layer_idx, eidx):
                entry = self.index.get(name)
                if entry is not None:
                    mm = self.index.get_mmap(entry.file_path)
                    prefetch(mm, entry.data_offset, entry.data_length)

    @staticmethod
    def _decode(entry: TensorEntry, raw: bytes) -> np.ndarray:
        """Convert raw bytes → NumPy array according to tensor dtype/shape."""
        if entry.dtype in _QUANTISED_DTYPES:
            # Return raw uint8 for quantised blobs; dequant happens on-GPU
            return np.frombuffer(raw, dtype=np.uint8)
        np_dtype = _ST_DTYPE_MAP.get(entry.dtype)
        if np_dtype is None:
            raise ValueError(f"Unknown dtype {entry.dtype!r} for tensor {entry.name!r}")
        arr = np.frombuffer(raw, dtype=np_dtype)
        if entry.shape:
            arr = arr.reshape(entry.shape)
        return arr

    def close(self) -> None:
        self._reader.close()
        self.index.close_mmaps()

    def __enter__(self) -> "WeightStreamer":
        return self

    def __exit__(self, *_) -> None:
        self.close()
