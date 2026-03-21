# Flash Mode — Architecture Deep Dive

## Overview
Flash Weight Streaming enables large language model inference on Apple Silicon
Macs with less RAM than the total model size by exploiting macOS's unified
memory architecture.

## Unified Memory Architecture
Apple Silicon chips have a single pool of DRAM shared by CPU, GPU, and Neural
Engine.  This is fundamentally different from discrete GPU systems where VRAM
is separate from system RAM.

┌─────────────────────────────────────────────────────┐
│                Apple Silicon SoC                     │
│                                                     │
│   ┌──────────┐    ┌──────────┐    ┌─────────────┐  │
│   │   CPU     │    │   GPU    │    │   Neural    │  │
│   │ (P+E cores│    │ (GPU     │    │   Engine    │  │
│   │  + cache) │    │  cores)  │    │             │  │
│   └────┬─────┘    └────┬─────┘    └──────┬──────┘  │
│        │               │                 │          │
│   ┌────┴───────────────┴─────────────────┴──────┐   │
│   │         Unified Memory (LPDDR5/5X)           │   │
│   │    ← same physical DRAM for all agents →     │   │
│   └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
         ↕  memory-mapped I/O (APFS + NVMe controller)
┌─────────────────────────────────────────────────────┐
│                  NVMe SSD / external TB               │
└─────────────────────────────────────────────────────┘

Key insight: when model weights are mmap()'d from disk, macOS maps them into the
virtual address space of all processes (CPU and Metal GPU share the same
physical page table entries in unified memory). A Metal shader can read from an
mmap'd region without any copy — it reads directly from the page cache.

## Safetensors Layout
┌────────────────────────────────────────────────────────┐
│ model.safetensors                                      │
├──────────┬──────────────────────────────────────────── │
│ 8 bytes  │ header_length (uint64 LE)                  │
├──────────┴──────────────────────────────────────────── │
│ header_length bytes │ JSON: {name: {dtype,shape,offsets}} │
├──────────────────────────────────────────────────────── │
│ data region (all tensors packed, no alignment gaps)    │
│  ┌────────────────────────────────────────────────┐   │
│  │ tensor 0 raw bytes (dtype as-is, no padding)   │   │
│  │ tensor 1 raw bytes                             │   │
│  │ ...                                            │   │
│  └────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘

SafetensorsIndex reads ONLY the header (typically a few KB) at open time. The
data region is never read until WeightStreamer.stream_tensors() is called.

## Parallel pread()
POSIX pread(fd, buf, n, offset) reads n bytes from absolute offset into buf,
atomically, without moving the file's seek position. This means:
1. Multiple threads can call pread() on the same fd simultaneously.
2. No locking needed around seek+read pairs.
3. CPython releases the GIL during the syscall — true parallelism.

With 4 threads issuing simultaneous pread() calls against an NVMe device, we can
achieve close to the device's sequential bandwidth even for discontiguous reads
(typical for sparse expert patterns in MoE).

## madvise() Strategy
Layer N-2   Layer N-1   Layer N     Layer N+1   Layer N+2
(evict)     (evict)     (COMPUTE)   (prefetch)  (prefetch)

* MADV_WILLNEED on layers N+1..N+prefetch_window → OS starts I/O immediately
* MADV_FREE on layers N-2 → OS may reclaim pages under memory pressure

Net result: resident set stays at ~3-4 layers × layer_size
For a 70B model (Q4_K_M, 40 GB total, ~2 layers/GB), this keeps ~8 GB resident
while the remaining 32 GB sits on SSD.

## Q4_0 Dequantisation
Q4_0 block (18 bytes):
  ┌──────────────────────────────────────────────────────────────┐
  │  scale (f16, 2B)  │  16 packed bytes (32 nibbles = 32 vals) │
  └──────────────────────────────────────────────────────────────┘

Dequant:
  For nibble n (0..15 → signed: n - 8 = -8..+7):
    f16_value = scale × (nibble - 8)

FMA on Metal: fma(scale, nibble_f32, 0.0f) → cast to f16
Our custom Metal kernel performs this 2 nibbles per thread in a tight loop,
reading from the page cache directly, writing to a temporary GPU buffer that
feeds immediately into the matmul.

## MoE Expert Streaming Timeline
Token arrives → Router (always in RAM) → Top-K indices

I/O thread:    pread(expert[0]) ──→ GPU
               pread(expert[1]) ──→ GPU    ← parallel
               pread(expert[2]) ──→ GPU

GPU timeline:  [expert 0 dequant] [FFN 0]
                     [expert 1 dequant] [FFN 1]
                          [expert 2 dequant] [FFN 2]
                                              [combine + softmax-weight]

The deferred 3-command-buffer pipeline (load → dequant → compute) allows I/O
and GPU to run concurrently for consecutive experts.

## File Structure
mlx_flash/
├── __init__.py          — public API
├── config.py            — FlashConfig dataclass
├── page_cache.py        — madvise() wrappers (macOS)
├── streamer.py          — SafetensorsIndex + WeightStreamer + ParallelPreader
├── loader.py            — FlashModelLoader (layer-at-a-time)
├── manager.py           — FlashManager (top-level, wraps mlx-lm)
├── moe.py               — MoEConfig + MoERouter + MoEFlashHandler
├── prefetch.py          — WeightPrefetcher daemon thread
└── kernels/
    ├── __init__.py      — kernel dispatch (Metal or fallback)
    ├── flash_dequant.metal
    ├── swiglu_fused.metal
    ├── moe_dispatch.metal
    └── compile_kernels.py

integration/
├── lmstudio.py          — apply_flash_patch() (monkey-patches mlx_lm.load)
└── modelfile.py         — parse_flash_directives()

## Memory Budget Management: Weights vs. KV Cache vs. Activations

To maintain a strict RAM budget (e.g., < 1GB for Llama-70B), mlx-flash manages three distinct memory consumers:

### 1. Weights (Flash Streaming)
Weights are loaded via `mx.load(lazy=True)`, meaning they are memory-mapped and consume zero RSS until used. `FlashLLM` evaluates layers one-by-one and calls `mx.clear_cache()` to release weight buffers from Metal immediately after each layer.

### 2. KV Cache (Rotating & Disk Offloading)
For long conversations, the KV cache grows linearly. We address this via:
* **Rotating KV Cache**: Uses `mlx_lm.models.cache.RotatingKVCache` to cap the number of tokens held in memory.
* **Disk Offloading**: If `kv_cache_dir` is set, old KV entries are evicted to `.safetensors` files on SSD and re-loaded (lazy mmap) only when needed.

### 3. Activations (Chunked Prefill)
Large prompts (prefill) create large intermediate activation matrices (e.g., attention masks).
* **Chunked Prefill**: Process long prompts in chunks (default 512 tokens). This caps the peak activation memory to the size of a single chunk, preventing OOM on 32k+ token prompts.

## Limitations & Known Issues
* Q2_K models: 2-bit dequant produces visibly degraded output; a warning is
  shown. Bit-exact guarantee applies only to ≥4-bit.
* First token latency: the first forward pass after a cold cache is slower
  (pages must be fetched from SSD). Subsequent passes benefit from the OS
  page cache.
* APFS encryption: reduces throughput by ~5–10%; prefer unencrypted volumes
  for external drives used as model storage.
* Spotlight indexing: Spotlight can cause I/O contention during inference.
  Add model directories to System Settings → Privacy → Full Disk Access exclusions.
* concurrent inference: running two Flash Mode models simultaneously doubles
  I/O load and may degrade throughput. Run one model at a time.

## Roadmap
- [ ] Quantised KV-cache to reduce attention memory footprint
- [ ] Expert pre-sorting for even better I/O coalescing on large MoE models
- [ ] macOS 15 / sequoia IOMemoryDescriptorCreateFromFileDescriptor for even
      lower-latency DMA from SSD
- [ ] Speculative decoding awareness (batch multiple candidate token sets)
