# Changelog

All notable changes to this project will be documented in this file.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
