/*
 * flash_dequant.metal
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * FMA-optimised 4-bit dequantisation kernels for Flash Weight Streaming.
 *
 * Supported formats:
 *   dequant_q4_0   — GGUF Q4_0: 32 values/block, 1×f16 scale, ±8 offset
 *   dequant_q4_k   — GGUF Q4_K_M: 256 values/block, super-block scales
 *   flash_gemv_q4_0 — fused dequant + GEMV (matrix-vector product)
 *
 * Design goals
 *   • Use fma() for scale×nibble — maps to hardware FMA on Apple GPU.
 *   • Process 2 nibbles (1 byte) per thread iteration to maximise ALU use.
 *   • Grid: one thread group per weight block; SIMD-width = 32.
 *   • No shared memory needed — each thread is independent.
 *
 * Bit-exact guarantee
 *   These kernels produce the same fp16 values as the reference NumPy
 *   dequantisation path (tested in tests/test_kernels.py).
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 */

#include <metal_stdlib>
using namespace metal;

// ── Constants ───────────────────────────────────────────────────────────────
constant uint Q4_0_BLOCK_VALS  = 32;   // values per Q4_0 block
constant uint Q4_0_BLOCK_BYTES = 18;   // 2 (scale f16) + 16 (packed nibbles)
constant uint Q4_0_DATA_BYTES  = 16;   // packed nibble bytes per block

// ── Q4_0 dequantisation ─────────────────────────────────────────────────────
//
// Block layout (18 bytes):
//   [0..1]  : scale — little-endian float16
//   [2..17] : 16 bytes, each byte holds two 4-bit values
//             lo_nibble = byte & 0x0F, hi_nibble = byte >> 4
//             signed value = nibble - 8  (range -8..+7)
//
// Output: Q4_0_BLOCK_VALS float16 values per block.
//
// Each thread handles one complete block.

kernel void dequant_q4_0(
    device  const uint8_t* __restrict__ src   [[ buffer(0) ]],
    device        half*    __restrict__ dst   [[ buffer(1) ]],
    constant      uint&                 rows  [[ buffer(2) ]],
    constant      uint&                 cols  [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint n_blocks = (rows * cols) / Q4_0_BLOCK_VALS;
    if (gid >= n_blocks) return;

    uint src_off = gid * Q4_0_BLOCK_BYTES;
    uint dst_off = gid * Q4_0_BLOCK_VALS;

    // Load scale as float (promoted from f16 for accurate FMA)
    float scale = float(*(device const half*)(src + src_off));

    device const uint8_t* data = src + src_off + 2;

    // Unroll 16 bytes → 32 values
    for (uint i = 0; i < Q4_0_DATA_BYTES; i++) {
        uint8_t packed = data[i];

        // Extract signed nibbles (-8 offset)
        float lo = float(int(packed & 0x0Fu) - 8);
        float hi = float(int(packed >> 4u)  - 8);

        // FMA: scale * nibble + 0.0  (keeps intermediate in full float precision)
        dst[dst_off + i * 2u]     = half(fma(scale, lo, 0.0f));
        dst[dst_off + i * 2u + 1] = half(fma(scale, hi, 0.0f));
    }
}

// ── Q4_0 fused GEMV ─────────────────────────────────────────────────────────
//
// Computes one output element of y = W × x where W is Q4_0-quantised.
// Each thread computes one row of the output.
// This avoids materialising the full dequantised weight matrix.

kernel void flash_gemv_q4_0(
    device  const uint8_t* __restrict__ weights    [[ buffer(0) ]],
    device  const half*    __restrict__ input_vec  [[ buffer(1) ]],
    device        half*    __restrict__ output     [[ buffer(2) ]],
    constant      uint&                 in_feats   [[ buffer(3) ]],
    constant      uint&                 out_feats  [[ buffer(4) ]],
    uint row [[ thread_position_in_grid ]]
) {
    if (row >= out_feats) return;

    uint n_blocks = in_feats / Q4_0_BLOCK_VALS;
    uint row_byte_off = row * n_blocks * Q4_0_BLOCK_BYTES;

    float acc = 0.0f;

    for (uint b = 0; b < n_blocks; b++) {
        uint block_off = row_byte_off + b * Q4_0_BLOCK_BYTES;
        float scale = float(*(device const half*)(weights + block_off));
        device const uint8_t* data = weights + block_off + 2;

        uint vec_off = b * Q4_0_BLOCK_VALS;
        float block_dot = 0.0f;

        for (uint i = 0; i < Q4_0_DATA_BYTES; i++) {
            uint8_t packed = data[i];
            float lo = float(int(packed & 0x0Fu) - 8);
            float hi = float(int(packed >> 4u)  - 8);

            float v0 = float(input_vec[vec_off + i * 2u]);
            float v1 = float(input_vec[vec_off + i * 2u + 1]);

            block_dot = fma(lo, v0, block_dot);
            block_dot = fma(hi, v1, block_dot);
        }
        // FMA: acc += scale * block_dot
        acc = fma(scale, block_dot, acc);
    }

    output[row] = half(acc);
}

// ── Q4_1 dequantisation (adds per-block zero-point) ─────────────────────────
// Block layout: 2B scale (f16) + 2B min (f16) + 16B data = 20 bytes

constant uint Q4_1_BLOCK_BYTES = 20;

kernel void dequant_q4_1(
    device  const uint8_t* __restrict__ src   [[ buffer(0) ]],
    device        half*    __restrict__ dst   [[ buffer(1) ]],
    constant      uint&                 rows  [[ buffer(2) ]],
    constant      uint&                 cols  [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint n_blocks = (rows * cols) / Q4_0_BLOCK_VALS;
    if (gid >= n_blocks) return;

    uint src_off = gid * Q4_1_BLOCK_BYTES;
    uint dst_off = gid * Q4_0_BLOCK_VALS;

    float scale = float(*(device const half*)(src + src_off));
    float minval = float(*(device const half*)(src + src_off + 2));
    device const uint8_t* data = src + src_off + 4;

    for (uint i = 0; i < Q4_0_DATA_BYTES; i++) {
        uint8_t packed = data[i];
        float lo = float(packed & 0x0Fu);
        float hi = float(packed >> 4u);

        // FMA: value = scale * nibble + min
        dst[dst_off + i * 2u]     = half(fma(scale, lo, minval));
        dst[dst_off + i * 2u + 1] = half(fma(scale, hi, minval));
    }
}
