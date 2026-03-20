/*
 * swiglu_fused.metal
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Fused SwiGLU activation kernel.
 *
 * SwiGLU(gate, up) = SiLU(gate) × up
 * SiLU(x)         = x × σ(x) = x / (1 + exp(-x))
 *
 * Fusing the SiLU + element-wise multiply into one kernel avoids a
 * temporary intermediate buffer and one additional Metal dispatch.
 *
 * Grid: one thread per element of the intermediate (gate) tensor.
 * This kernel processes float16 inputs/outputs.
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 */

#include <metal_stdlib>
using namespace metal;

// ── f16 version ─────────────────────────────────────────────────────────────
kernel void swiglu_fused_f16(
    device  const half* __restrict__ gate [[ buffer(0) ]],
    device  const half* __restrict__ up   [[ buffer(1) ]],
    device        half* __restrict__ out  [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    float g = float(gate[gid]);
    float u = float(up[gid]);

    // SiLU: g * sigmoid(g)
    // Using precise_divide/exp for numerical stability
    float silu_g = g * (1.0f / (1.0f + exp(-g)));

    // SwiGLU output
    out[gid] = half(silu_g * u);
}

// ── f32 version ─────────────────────────────────────────────────────────────
kernel void swiglu_fused_f32(
    device  const float* __restrict__ gate [[ buffer(0) ]],
    device  const float* __restrict__ up   [[ buffer(1) ]],
    device        float* __restrict__ out  [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    float g = gate[gid];
    float u = up[gid];
    float silu_g = g / (1.0f + exp(-g));
    out[gid] = silu_g * u;
}

// ── SIMD-vectorised f16 version (4 elements per thread) ────────────────────
kernel void swiglu_fused_f16x4(
    device  const half4* __restrict__ gate [[ buffer(0) ]],
    device  const half4* __restrict__ up   [[ buffer(1) ]],
    device        half4* __restrict__ out  [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    float4 g = float4(gate[gid]);
    float4 u = float4(up[gid]);

    // Vectorised SiLU
    float4 silu_g = g / (1.0f + exp(-g));
    out[gid] = half4(silu_g * u);
}
