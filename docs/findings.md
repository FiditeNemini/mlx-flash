# Experimental Findings & The "Lazy Graph" Problem

This document outlines the engineering journey, the challenges encountered with MLX's lazy evaluation, and the exact steps taken to achieve true zero-copy weight streaming for models larger than physical RAM.

## 1. The Core Engine: `mmap` vs `os.pread`

Initially, the project used `os.pread()` combined with a thread pool to read weights from disk, followed by `.copy()` to convert them into NumPy/MLX arrays.
* **The Problem:** Calling `.copy()` pulls the data from the macOS Page Cache into the Python process's private memory (Resident Set Size or RSS). The OS cannot safely reclaim this memory until the Python garbage collector frees it, leading to inflated RAM usage and eventually Out-Of-Memory (OOM) crashes.
* **The Solution:** We migrated the core engine to use `mmap` and `memoryview`. This allows MLX to map the safetensors directly from the SSD into the GPU's unified address space (Zero-Copy). When we are done with a layer, calling `madvise(MADV_FREE)` allows macOS to instantly reclaim the physical pages without waiting on Python.

## 2. The "Lazy Graph" Challenge

Apple's MLX framework is fundamentally **lazy**. When you call `model(inputs)`, it doesn't immediately compute the matrix multiplications. Instead, it builds a massive "computation graph" representing the operations for all layers (e.g., all 52 layers of a 30B model).
* **The Crash:** MLX only allocates Metal (GPU) memory when `mx.eval()` is finally called at the end of the generation step. If the model is 18 GB and you have 16 GB of RAM, MLX attempts to allocate all 18 GB at once to evaluate the graph, resulting in: `[METAL] Command buffer execution failed: Insufficient Memory`.
* **The Limitation of Monkey-Patching:** Simply intercepting `mlx_lm.load()` to load weights dynamically isn't enough for models that exceed physical RAM, because the standard `mlx_lm.stream_generate` loop still builds a unified graph that requires evaluating the entire model state simultaneously.

## 3. The Solution: Synchronous Layer Execution

To prove that Flash Mode can run massive models, we wrote `scripts/prove_flash_manual.py`. This script manually orchestrates the forward pass:
1. Load Layer $N$ weights.
2. Compute Layer $N$.
3. **Crucially:** Call `mx.eval()` and `mx.synchronize()` immediately after Layer $N$ finishes.
4. Replace Layer $N$'s weights with dummy scalars and call `mx.metal.clear_cache()`.
5. Proceed to Layer $N+1$.

By breaking the unified graph into 52 sequential, synchronous evaluations, the Metal memory pool only ever holds the weights and activations for **one layer at a time**.

**Result:** An 18 GB model runs flawlessly on a 16 GB Mac with a peak active footprint of `< 1.0 GB`.

## 4. Production Integration & The Future

### What this means for LM Studio / `mlx-engine`
This project is intended as a streaming backend for C++ inference engines. If the host engine (like `mlx-engine`) evaluates the model layer-by-layer (or handles KV cache incrementally without building a massive unified graph), this zero-copy `mmap` backend is mathematically sound and ready for production.

### What this means for `mlx_lm` (Python)
Currently, to use this purely in Python for models larger than your RAM, you must rewrite the generation loop (as demonstrated in our manual proof script). Monkey-patching `model.__call__` is brittle because `mlx_lm`'s KV cache management expects the standard graph execution.

### Suggestions for the Future
1. **Upstream MLX Support:** The ideal solution would be for `mlx` to natively support a "streaming module" primitive that automatically yields memory between layer evaluations, rather than requiring manual graph breaking.
2. **KV Cache Offloading:** For long-context generations on massive models, the KV cache itself will eventually exceed RAM. Future iterations of Flash Mode should apply this exact `mmap` strategy to the KV cache, streaming it to and from disk.
3. **Dynamic RAM Budgeting:** Currently, the RAM budget is a hardcoded limit. A future version could dynamically read `psutil.virtual_memory().available` and adjust the prefetch window and eviction strategy on the fly.
