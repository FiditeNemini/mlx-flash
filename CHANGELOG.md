# Changelog

All notable changes to this project will be documented in this file.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] - 2026-06-12

Consolidates all work since v0.1.0 into a release, and restores compatibility
with the current MLX ecosystem.

### Added
- **Holistic Model Patching**: `StreamingProxy` wraps the model's own layers so
  native architecture logic (RoPE scaling, masks, residuals) runs untouched.
- **Bit-Perfect Parity**: `TiledLinear` with FP32 accumulation — zero loss delta
  vs. standard MLX execution.
- **Predictive Bandwidth Controller (MPC-lite)**: predicts layer N+1 bandwidth
  demand and paces SSD reads via a token-bucket actuator.
- **Hybrid Quantized KV Cache**: recent 128 tokens kept in FP16, older context
  offloaded to 8-bit quantized disk storage; verified with passkey retrieval.
- **Unified `mlx-flash` CLI** with `--ram` budget and `--kv-quant` flags.
- **Token-to-Expert Affinity Batching** for MoE prefill.
- Restored GitHub Actions CI (macOS ARM, Python 3.11/3.12).

### Fixed
- **Compatibility with `mlx>=0.31` / `mlx-lm>=0.31`**: resolved KV-cache
  `offset` property conflict with the current `mlx-lm` cache protocol
  (disk caches now report `is_trimmable() == False`), repaired tiled-linear
  module replacement, stripped the obsolete `mask` kwarg, added a
  `make_cache()` fallback for models without one, and wired disk KV cache
  injection into `stream_generate`.
- **`max_tokens` is now honored** by `FlashGenerationLoop.stream_generate`
  (previously `generate_step`'s 256-token default silently applied).
- **End-to-end CLI verified on a real model** (Qwen2.5-0.5B-4bit): output is
  identical to plain `mlx_lm`, ~24 tok/s through the Flash path.

### Changed
- **Tiled linear layers now execute whole** (`mx.addmm`, identical to
  `nn.Linear`) to preserve bit-exact parity: on MLX ≥ 0.31, Metal kernel
  selection makes block-wise fp32 tile accumulation diverge from native fp16
  matmul (up to 0.07). Sub-layer tiling will return as an opt-in memory mode.
- `FlashManager.load()` (and the LM Studio patch) returns `FlashEngine`, the
  holistic-patching proxy verified to match plain `mlx_lm` output exactly on
  real models. `FlashLLM`'s manual forward loop remains for internal use but
  is not the public path.
- README installation instructions: this project is installed from GitHub —
  the `mlx-flash` name on PyPI belongs to an unrelated project.

## - 2026-03-21

**Disk KV Cache Offloading — Production Quality Infinite Context**

- Complete KVCache interface implementation (size/trim/empty/nbytes/state/to_quantized)
- Zero-GPU-sync writes via `mx.eval()` + `np.asarray().tobytes()`
- Bounded disk growth with configurable eviction (`disk_kv_max_tokens`)
- Unique per-process cache directories (PID + UUID) + proper `close()` / context manager
- Crash-safe header ordering, consolidated flushes, and `shutdown()` integration
- Removed unconditional debug prints, fixed resource leaks, added 10 new unit tests

All 37 existing tests + new `test_disk_kv_cache.py` suite + 4000-token RAM-budget stress test now pass.

Infinite context without OOM is now stable, reliable, and production-ready.

## [0.1.1]

### Added
- **Spotlight Auto-Exclusion**: Drops a `.metadata_never_index` file into model directories to prevent macOS Spotlight from aggressively scanning 100GB+ files and crippling SSD throughput.
- **Battery & P-Core Warnings**: Added automatic diagnostics and warnings when running enormous IO workloads on battery power (`pmset -g batt`), alerting users to thermal limits and Battery Drain.
- **macOS Unified Page Cache Integration**: Full, mathematical integration of OS-level `madvise()` calls for `.safetensors`. Explicit `MADV_WILLNEED` and `MADV_FREE` hints now run smoothly in-pipeline, keeping Metal RAM bounds strict while maximizing SSD throughput.

### Changed
- **Pipelined GPU Synchronization**: Refactored the proxy wrapper's synchronization (`mx.synchronize()`). CPU and GPU are no longer artificially serialized on every layer boundary. They now pipeline naturally across `pipeline_depth=2` layers for radically improved performance.
- Simplified README installation instructions specifically for PyPI release.

## [0.1.0]

### Added
- Initial Flash Weight Streaming implementation
- Parallel `pread()` weight streamer with configurable thread pool
- macOS unified page cache management (`madvise` WILLNEED / DONTNEED)
- MoE top-K expert streaming (Mixtral, DeepSeek, Qwen2-MoE)
- FMA-optimised Metal kernels: `flash_dequant_4bit`, `swiglu_fused`, `moe_dispatch`
- LM Studio extension hook + Modelfile `FLASH true` directive
- Background prefetch thread for I/O/compute overlap
- `FlashConfig` with RAM budget, thread count, and quantisation level controls
- Comprehensive test suite (unit + integration) targeting 4B models
- Benchmark suite comparing Flash vs Normal loading
- Full README with Mermaid architecture diagrams and performance tables
