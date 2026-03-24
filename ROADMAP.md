# mlx-flash Roadmap ⚡

This document outlines the planned milestones for `mlx-flash` as it moves from beta to a stable, production-grade utility.

## v0.1.x: Stability & Polished Beta
*Focus: Fixing the "paper cuts" and ensuring 100% correctness.*

- [ ] **PyPI Release**: Official package distribution for easier installation.
- [ ] **CI Benchmarking**: Automatic performance regression tracking in GitHub Actions.
- [ ] **Sampler Parity**: Ensure 100% numerical parity with standard `mlx-lm` for all sampling parameters.
- [ ] **Improved Diagnostics**: More granular RAM profiling in `flash-monitor`.

## v0.2.x: The Engine Rewrite & Optimizations
*Focus: Replacing monkey patches with robust structural pipelines.*

- [x] **Hook-Based Architecture**: Moved to a clean Event-Driven FlashEngine.
- [x] **Block-wise Tiled Execution**: Partitioning massive linear layers.
- [x] **Sub-Component Pipelining**: Overlapping IO with specific layer ops.
- [x] **Mixed-Precision Quantization**: Dynamically adapting block bits.
- [x] **Quantized Disk KV Cache**: 4x reduction in out-of-core memory.

## v0.3.x: Schedulers & Resource Management
*Focus: Advanced global optimization models.*

- [x] **Adaptive Predictive Prefetching**: Basic lookahead algorithms.
- [x] **Unified Bandwidth Controller**: Fabric-aware IO throttling.
- [x] **Multi-Tier Cache Manager**: Cold, Warm, Hot unified modeling.
- [x] **Learned Online Cost Model**: RLS solvers predicting IO latency.

## v0.4.x: High-Performance MoE
*Focus: Making massive MoE models (Mixtral, DeepSeek) run natively.*

- [x] **Speculative Expert Prefetching**: CPU router lookaheads.
- [x] **Token-to-Expert Affinity Batching**: O(1) Pre-sorting for prefill.
- [ ] **Speculative decoding awareness**: Fast/Slow model coordination.
- [ ] **macOS 15 / Sequoia IOMemoryDescriptor**: Zero-copy native DMA.

---

## v1.0.0: Native Integration & Autonomous Tuning
*Focus: Moving from a standalone engine to a standard feature.*

- [ ] **Adaptive "Auto-Budget"**: Dynamically adjust `ram_budget_gb` based on real-time OS memory pressure (`psutil`).
- [ ] **Upstream PR to `mlx-lm`**: Propose native `FlashLLM` support to eliminate the need for intercept layers.
- [ ] **Official LM Studio Integration**: Seamless "Flash" checkbox in the LM Studio UI.
- [ ] **Documentation**: Comprehensive API reference and integration guides for other frameworks.
