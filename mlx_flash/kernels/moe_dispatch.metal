/*
 * moe_dispatch.metal
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * MoE routing and expert dispatch kernels.
 *
 * moe_topk_routing
 *   For each token, selects top-K expert indices and computes softmax
 *   routing weights.  Input: router logits [n_tokens × n_experts].
 *   Output: expert_indices [n_tokens × top_k],
 *           expert_weights [n_tokens × top_k].
 *
 * moe_combine
 *   Weighted sum of K expert outputs for each token.
 *   Input:  expert_outputs [K × seq_len × hidden_dim] (packed)
 *           routing_weights [n_tokens × top_k]
 *   Output: combined [n_tokens × hidden_dim]
 *
 * Design: one thread-group per token.  K≤8 assumed; K>8 needs loop unroll.
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 */

#include <metal_stdlib>
using namespace metal;

constant uint MAX_TOP_K = 8u;

// ── Top-K routing ────────────────────────────────────────────────────────────
kernel void moe_topk_routing(
    device  const float* __restrict__ router_logits   [[ buffer(0) ]],
    device        int*   __restrict__ expert_indices  [[ buffer(1) ]],
    device        float* __restrict__ expert_weights  [[ buffer(2) ]],
    constant uint&                    n_tokens         [[ buffer(3) ]],
    constant uint&                    n_experts        [[ buffer(4) ]],
    constant uint&                    top_k            [[ buffer(5) ]],
    uint token_idx [[ thread_position_in_grid ]]
) {
    if (token_idx >= n_tokens) return;

    uint k = min(top_k, MAX_TOP_K);
    uint logit_base = token_idx * n_experts;
    uint out_base   = token_idx * k;

    // ── 1. Linear top-K scan (fast for small K and n_experts ≤ 256)
    float top_vals[MAX_TOP_K];
    int   top_idxs[MAX_TOP_K];

    for (uint i = 0; i < k; i++) {
        top_vals[i] = -INFINITY;
        top_idxs[i] = -1;
    }

    for (uint e = 0; e < n_experts; e++) {
        float val = router_logits[logit_base + e];
        // Find minimum in current top-k
        uint min_pos = 0;
        for (uint i = 1; i < k; i++) {
            if (top_vals[i] < top_vals[min_pos]) min_pos = i;
        }
        if (val > top_vals[min_pos]) {
            top_vals[min_pos] = val;
            top_idxs[min_pos] = int(e);
        }
    }

    // ── 2. Sort top-K by descending logit (insertion sort, K ≤ 8)
    for (uint i = 1; i < k; i++) {
        float v = top_vals[i];
        int   ix = top_idxs[i];
        uint j = i;
        while (j > 0 && top_vals[j - 1] < v) {
            top_vals[j] = top_vals[j - 1];
            top_idxs[j] = top_idxs[j - 1];
            j--;
        }
        top_vals[j] = v;
        top_idxs[j] = ix;
    }

    // ── 3. Softmax over top-K logits
    float max_val = top_vals[0];
    float sum_exp = 0.0f;
    float exp_vals[MAX_TOP_K];
    for (uint i = 0; i < k; i++) {
        exp_vals[i] = exp(top_vals[i] - max_val);
        sum_exp += exp_vals[i];
    }

    // ── 4. Write outputs
    for (uint i = 0; i < k; i++) {
        expert_indices[out_base + i] = top_idxs[i];
        expert_weights[out_base + i] = exp_vals[i] / sum_exp;
    }
}

// ── Expert output combination ────────────────────────────────────────────────
kernel void moe_combine(
    device  const half*  __restrict__ expert_outputs  [[ buffer(0) ]],
    device  const float* __restrict__ routing_weights [[ buffer(1) ]],
    device        half*  __restrict__ combined        [[ buffer(2) ]],
    constant uint&                    n_tokens        [[ buffer(3) ]],
    constant uint&                    hidden_dim      [[ buffer(4) ]],
    constant uint&                    top_k           [[ buffer(5) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint token_idx = gid.x;
    uint dim_idx   = gid.y;

    if (token_idx >= n_tokens || dim_idx >= hidden_dim) return;

    float acc = 0.0f;
    for (uint k = 0; k < top_k; k++) {
        uint expert_out_idx = k * n_tokens * hidden_dim
                              + token_idx * hidden_dim
                              + dim_idx;
        float weight = routing_weights[token_idx * top_k + k];
        acc = fma(float(expert_outputs[expert_out_idx]), weight, acc);
    }

    combined[token_idx * hidden_dim + dim_idx] = half(acc);
}
