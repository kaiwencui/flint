#include "kernels.hpp"

namespace flint::cuda {

namespace {

// Branchless, warp-independent, register-only FP4 E2M1 decode (ZERO cmem access, ZERO warp sync):
// 0xC8643210u encodes 2x magnitude for mag=0..7: {0, 1, 2, 3, 4, 6, 8, 12} * 0.5 = {0, 0.5, 1, 1.5, 2, 3, 4, 6}
__device__ __forceinline__ f32 fp4_decode(u32 nib) {
    u32 mag = nib & 7u;
    f32 v = 0.5f * static_cast<f32>((0xC8643210u >> (mag << 2)) & 0xFu);
    return (nib & 8u) ? -v : v;
}

__device__ __forceinline__ f32 softcap_f32(f32 x, f32 cap) {
    return (cap > 0.0f) ? (cap * tanhf(x / cap)) : x;
}

__device__ __forceinline__ f32 log_sigmoid_f32(f32 x) {
    return fminf(x, 0.0f) - log1pf(expf(-fabsf(x)));
}

// ============================================================================
// 1. Fused Inkling-2 MoE Top-K Router (1 block = 128 threads = 128 experts per token)
// ============================================================================
// Computes logits = x @ gate_w^T in FP32, selects top-8 by (sigmoid(logit) + bias)
// with lower-index tie break, and writes renormalized log_sigmoid * route_scale weights
__global__ void inkling_moe_router_kernel(const __nv_bfloat16* __restrict__ x,
                                          const __nv_bfloat16* __restrict__ gate_w,
                                          const f32* __restrict__ gate_bias,
                                          i32* __restrict__ out_ids,
                                          f32* __restrict__ out_weights,
                                          u32 M, u32 H, u32 E, u32 topk, f32 route_scale) {
    const u32 m = blockIdx.x;
    const u32 e = threadIdx.x;
    if (m >= M) return;

    const __nv_bfloat16* xm = x + static_cast<size_t>(m) * H;
    f32 logit = -1e30f;
    f32 score = -1e30f;
    if (e < E) {
        const auto* x4 = reinterpret_cast<const uint4*>(xm);
        const auto* w4 = reinterpret_cast<const uint4*>(gate_w + static_cast<size_t>(e) * H);
        u32 H8 = H >> 3;
        f32 acc = 0.0f;
        for (u32 i = 0; i < H8; ++i) {
            uint4 xv = x4[i];
            uint4 wv = __ldg(w4 + i);
            const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
            const auto* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 fx = __bfloat1622float2(xb[k]);
                float2 fw = __bfloat1622float2(wb[k]);
                acc = fmaf(fx.x, fw.x, acc);
                acc = fmaf(fx.y, fw.y, acc);
            }
        }
        logit = acc;
        f32 sig = 1.0f / (1.0f + expf(-logit));
        score = sig + (gate_bias ? __ldg(gate_bias + e) : 0.0f);
    }

    __shared__ f32 sm_logit[128];
    __shared__ f32 sm_score[128];
    __shared__ i32 sel_ids[8];
    __shared__ f32 sel_lp[8];
    sm_logit[e] = logit;
    sm_score[e] = score;
    __syncthreads();

    // Thread 0 performs exact top-8 selection (128 elements -> < 1 us, deterministic tie-break, 0 stack frame)
    if (e == 0) {
        f32 max_lp = -1e30f;
        for (u32 k = 0; k < topk; ++k) {
            i32 best_i = 0;
            f32 best_s = -1e30f;
            for (u32 i = 0; i < E; ++i) {
                if (sm_score[i] > best_s) {
                    best_s = sm_score[i];
                    best_i = static_cast<i32>(i);
                }
            }
            sm_score[best_i] = -1e30f;
            sel_ids[k] = best_i;
            f32 lp = log_sigmoid_f32(sm_logit[best_i]);
            sel_lp[k] = lp;
            max_lp = fmaxf(max_lp, lp);
        }
        f32 sum_e = 0.0f;
        for (u32 k = 0; k < topk; ++k) {
            sum_e += expf(sel_lp[k] - max_lp);
        }
        f32 lse = max_lp + logf(sum_e);
        for (u32 k = 0; k < topk; ++k) {
            out_ids[m * topk + k] = sel_ids[k];
            out_weights[m * topk + k] = expf(sel_lp[k] - lse) * route_scale;
        }
    }
}

// ============================================================================
// 2. Fused NVFP4 E2M1 Gate+Up+SiTU Across Active Experts (M >= 1, zero spills)
// ============================================================================
// w13_weight: [E, 2*inter, K/2] U8 (rows interleaved: row 2*j = gate[j], row 2*j+1 = up[j])
// w13_scale: [E, 2*inter, K/16] FP8 E4M3 (1 byte scale per 16 FP4 weights)
// w13_amax: [E] F32; outer_denom: [1] F32 -> global_scale = amax[e] / outer_denom[0]
// Grid: ((inter + 3)/4, topk, M), Block: 128 threads (4 warps -> 4 inter outputs per block)
// Each warp loads 16 FP4 weights (1 uint2 = 8 bytes) per lane -> 512 weights/iter
__global__ void inkling_moe_nvfp4_gate_up_situ_kernel(
    const __nv_bfloat16* __restrict__ x,
    const u8* __restrict__ w13_weight,
    const u8* __restrict__ w13_scale,
    const f32* __restrict__ w13_amax,
    const f32* __restrict__ w13_outer_denom,
    const i32* __restrict__ expert_ids,
    const f32* __restrict__ expert_weights,
    __nv_bfloat16* __restrict__ act_out, // [M, topk, inter]
    u32 inter, u32 K, u32 topk,
    f32 gate_cap, f32 up_cap) {
    const u32 wid  = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 j    = blockIdx.x * 4 + wid;
    const u32 slot = blockIdx.y;
    const u32 m    = blockIdx.z;
    if (j >= inter) return;

    const i32 exp_id = expert_ids[m * topk + slot];
    const f32 gamma  = expert_weights[m * topk + slot];
    const f32 g_scale = __ldg(w13_amax + exp_id) / __ldg(w13_outer_denom);

    const u32 K2  = K >> 1;  // bytes per row
    const u32 K16 = K >> 4;  // blocks of 16 per row (2304/16 = 144)
    const size_t row_g = static_cast<size_t>(exp_id) * (2 * inter) + (2 * j + 0);
    const size_t row_u = row_g + 1;

    const auto* wg_u2 = reinterpret_cast<const uint2*>(w13_weight + row_g * K2);
    const auto* wu_u2 = reinterpret_cast<const uint2*>(w13_weight + row_u * K2);
    const u8* sg = w13_scale + row_g * K16;
    const u8* su = w13_scale + row_u * K16;
    const auto* x4 = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K);

    f32 acc_g = 0.0f;
    f32 acc_u = 0.0f;

    // Each iteration handles 16 weights = 1 block-16 FP8 scale = 2 uint4s of BF16 x
    for (u32 blk = lane; blk < K16; blk += 32) {
        uint2 raw_g = __ldg(wg_u2 + blk);
        uint2 raw_u = __ldg(wu_u2 + blk);
        __nv_fp8_e4m3 fg_s, fu_s;
        *reinterpret_cast<u8*>(&fg_s) = __ldg(sg + blk);
        *reinterpret_cast<u8*>(&fu_s) = __ldg(su + blk);
        f32 sc_g = static_cast<f32>(fg_s);
        f32 sc_u = static_cast<f32>(fu_s);

        uint4 xv0 = x4[blk * 2 + 0];
        uint4 xv1 = x4[blk * 2 + 1];
        const auto* xb0 = reinterpret_cast<const __nv_bfloat162*>(&xv0);
        const auto* xb1 = reinterpret_cast<const __nv_bfloat162*>(&xv1);

        f32 dg = 0.0f, du = 0.0f;
        // Low 8 weights from .x against xv0
        u32 pg0 = raw_g.x, pu0 = raw_u.x;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 fx = __bfloat1622float2(xb0[k]);
            dg = fmaf(fp4_decode(pg0),      fx.x, dg);
            dg = fmaf(fp4_decode(pg0 >> 4), fx.y, dg);
            du = fmaf(fp4_decode(pu0),      fx.x, du);
            du = fmaf(fp4_decode(pu0 >> 4), fx.y, du);
            pg0 >>= 8;
            pu0 >>= 8;
        }
        // High 8 weights from .y against xv1
        u32 pg1 = raw_g.y, pu1 = raw_u.y;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 fx = __bfloat1622float2(xb1[k]);
            dg = fmaf(fp4_decode(pg1),      fx.x, dg);
            dg = fmaf(fp4_decode(pg1 >> 4), fx.y, dg);
            du = fmaf(fp4_decode(pu1),      fx.x, du);
            du = fmaf(fp4_decode(pu1 >> 4), fx.y, du);
            pg1 >>= 8;
            pu1 >>= 8;
        }
        acc_g = fmaf(dg, sc_g, acc_g);
        acc_u = fmaf(du, sc_u, acc_u);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc_g += __shfl_down_sync(0xFFFFFFFFu, acc_g, off);
        acc_u += __shfl_down_sync(0xFFFFFFFFu, acc_u, off);
    }

    if (lane == 0) {
        f32 g_raw = acc_g * g_scale;
        f32 u_raw = acc_u * g_scale;
        // Exact Inkling-2 SiTU: sigmoid on UN-CAPPED gate_raw, then softcap gate & up, then * gamma
        f32 sig = 1.0f / (1.0f + expf(-g_raw));
        f32 g_cap = softcap_f32(g_raw, gate_cap);
        f32 u_cap = softcap_f32(u_raw, up_cap);
        f32 y = ((g_cap * sig) * u_cap) * gamma;
        act_out[(static_cast<size_t>(m) * topk + slot) * inter + j] = __float2bfloat16_rn(y);
    }
}

// ============================================================================
// 3. Fused NVFP4 E2M1 Down-Projection + Top-8 Combine (1 kernel launch)
// ============================================================================
// w2_weight: [E, H, inter/2] U8; w2_scale: [E, H, inter/16] FP8 E4M3
// Grid: ((H + 3)/4, M), Block: 128 threads (4 warps -> 4 hidden outputs per block)
// Each warp loops over all 8 active experts and accumulates directly into FP32 sum -> BF16 out
__global__ void inkling_moe_nvfp4_down_combine_kernel(
    const __nv_bfloat16* __restrict__ act_in, // [M, topk, inter]
    const u8* __restrict__ w2_weight,
    const u8* __restrict__ w2_scale,
    const f32* __restrict__ w2_amax,
    const f32* __restrict__ w2_outer_denom,
    const i32* __restrict__ expert_ids,
    __nv_bfloat16* __restrict__ out,          // [M, H]
    u32 H, u32 inter, u32 topk) {
    const u32 wid  = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 h    = blockIdx.x * 4 + wid;
    const u32 m    = blockIdx.y;
    if (h >= H) return;

    const u32 I2  = inter >> 1;  // 576 bytes
    const u32 I16 = inter >> 4;  // 72 blocks of 16 -> topk * I16 = 8 * 72 = 576 = 18.00 exact warp iters (0% idle lanes)
    const u32 total_blks = topk * I16;
    const f32 inv_denom = 1.0f / __ldg(w2_outer_denom);

    f32 total_acc = 0.0f;
    for (u32 flat = lane; flat < total_blks; flat += 32) {
        u32 slot = flat / I16;
        u32 blk  = flat - slot * I16;
        const i32 exp_id = expert_ids[m * topk + slot];
        const f32 g_scale = __ldg(w2_amax + exp_id) * inv_denom;
        const size_t row = static_cast<size_t>(exp_id) * H + h;
        const auto* w_u2 = reinterpret_cast<const uint2*>(w2_weight + row * I2);
        const u8* s_row  = w2_scale + row * I16;
        const auto* a4   = reinterpret_cast<const uint4*>(act_in + (static_cast<size_t>(m) * topk + slot) * inter);

        uint2 raw_w = __ldg(w_u2 + blk);
        __nv_fp8_e4m3 fs;
        *reinterpret_cast<u8*>(&fs) = __ldg(s_row + blk);
        f32 sc = static_cast<f32>(fs) * g_scale;

        uint4 av0 = a4[blk * 2 + 0];
        uint4 av1 = a4[blk * 2 + 1];
        const auto* ab0 = reinterpret_cast<const __nv_bfloat162*>(&av0);
        const auto* ab1 = reinterpret_cast<const __nv_bfloat162*>(&av1);

        f32 d = 0.0f;
        u32 pw0 = raw_w.x;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 fa = __bfloat1622float2(ab0[k]);
            d = fmaf(fp4_decode(pw0),      fa.x, d);
            d = fmaf(fp4_decode(pw0 >> 4), fa.y, d);
            pw0 >>= 8;
        }
        u32 pw1 = raw_w.y;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 fa = __bfloat1622float2(ab1[k]);
            d = fmaf(fp4_decode(pw1),      fa.x, d);
            d = fmaf(fp4_decode(pw1 >> 4), fa.y, d);
            pw1 >>= 8;
        }
        total_acc = fmaf(d, sc, total_acc);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        total_acc += __shfl_down_sync(0xFFFFFFFFu, total_acc, off);
    }
    if (lane == 0) {
        out[static_cast<size_t>(m) * H + h] = __float2bfloat16_rn(total_acc);
    }
}

// ============================================================================
// 4. Native MXFP8 (FP8 E4M3 + UE8M0 block-32 scale) MoE Gate+Up+SiTU & Down
// Used by Inkling-2-Mini coda layers 20..23
// ============================================================================
__device__ __forceinline__ f32 ue8m0_to_f32(u8 e) {
    // IEEE-754 float32 power of 2: exponent byte e -> 2^(e - 127)
    return (e == 0) ? 0.0f : __uint_as_float(static_cast<u32>(e) << 23);
}

__global__ void inkling_moe_mxfp8_gate_up_situ_kernel(
    const __nv_bfloat16* __restrict__ x,
    const u8* __restrict__ w13_weight,
    const u8* __restrict__ w13_scale_e8m0,
    const i32* __restrict__ expert_ids,
    const f32* __restrict__ expert_weights,
    __nv_bfloat16* __restrict__ act_out,
    u32 inter, u32 K, u32 topk,
    f32 gate_cap, f32 up_cap) {
    const u32 wid  = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 j    = blockIdx.x * 4 + wid;
    const u32 slot = blockIdx.y;
    const u32 m    = blockIdx.z;
    if (j >= inter) return;

    const i32 exp_id = expert_ids[m * topk + slot];
    const f32 gamma  = expert_weights[m * topk + slot];
    const u32 K32    = K >> 5; // 2304 / 32 = 72 blocks of 32
    const size_t row_g = static_cast<size_t>(exp_id) * (2 * inter) + (2 * j + 0);
    const size_t row_u = row_g + 1;

    const auto* wg_u4 = reinterpret_cast<const uint4*>(w13_weight + row_g * K);
    const auto* wu_u4 = reinterpret_cast<const uint4*>(w13_weight + row_u * K);
    const u8* sg = w13_scale_e8m0 + row_g * K32;
    const u8* su = w13_scale_e8m0 + row_u * K32;
    const auto* x4 = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K);

    f32 acc_g = 0.0f, acc_u = 0.0f;
    // Each iteration = 1 MXFP8 block of 32 weights = 2 uint4s of FP8 = 4 uint4s of BF16 x
    for (u32 blk = lane; blk < K32; blk += 32) {
        f32 sc_g = ue8m0_to_f32(__ldg(sg + blk));
        f32 sc_u = ue8m0_to_f32(__ldg(su + blk));
        f32 dg = 0.0f, du = 0.0f;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint4 rg = __ldg(wg_u4 + blk * 2 + h);
            uint4 ru = __ldg(wu_u4 + blk * 2 + h);
            const auto* fg = reinterpret_cast<const __nv_fp8x2_e4m3*>(&rg);
            const auto* fu = reinterpret_cast<const __nv_fp8x2_e4m3*>(&ru);
            uint4 xv0 = x4[blk * 4 + h * 2 + 0];
            uint4 xv1 = x4[blk * 4 + h * 2 + 1];
            const auto* xb0 = reinterpret_cast<const __nv_bfloat162*>(&xv0);
            const auto* xb1 = reinterpret_cast<const __nv_bfloat162*>(&xv1);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 fx = __bfloat1622float2(xb0[k]);
                float2 gg = static_cast<float2>(fg[k]);
                float2 uu = static_cast<float2>(fu[k]);
                dg = fmaf(gg.x, fx.x, fmaf(gg.y, fx.y, dg));
                du = fmaf(uu.x, fx.x, fmaf(uu.y, fx.y, du));
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 fx = __bfloat1622float2(xb1[k]);
                float2 gg = static_cast<float2>(fg[4 + k]);
                float2 uu = static_cast<float2>(fu[4 + k]);
                dg = fmaf(gg.x, fx.x, fmaf(gg.y, fx.y, dg));
                du = fmaf(uu.x, fx.x, fmaf(uu.y, fx.y, du));
            }
        }
        acc_g = fmaf(dg, sc_g, acc_g);
        acc_u = fmaf(du, sc_u, acc_u);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc_g += __shfl_down_sync(0xFFFFFFFFu, acc_g, off);
        acc_u += __shfl_down_sync(0xFFFFFFFFu, acc_u, off);
    }
    if (lane == 0) {
        f32 sig = 1.0f / (1.0f + expf(-acc_g));
        f32 y = ((softcap_f32(acc_g, gate_cap) * sig) * softcap_f32(acc_u, up_cap)) * gamma;
        act_out[(static_cast<size_t>(m) * topk + slot) * inter + j] = __float2bfloat16_rn(y);
    }
}

__global__ void inkling_moe_mxfp8_down_combine_kernel(
    const __nv_bfloat16* __restrict__ act_in,
    const u8* __restrict__ w2_weight,
    const u8* __restrict__ w2_scale_e8m0,
    const i32* __restrict__ expert_ids,
    __nv_bfloat16* __restrict__ out,
    u32 H, u32 inter, u32 topk) {
    const u32 wid  = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 h    = blockIdx.x * 4 + wid;
    const u32 m    = blockIdx.y;
    if (h >= H) return;

    const u32 I32 = inter >> 5; // 1152 / 32 = 36 blocks -> 8 * 36 = 288 = 9.00 exact warp iters (0% idle lanes)
    const u32 total_blks = topk * I32;
    f32 total_acc = 0.0f;
    for (u32 flat = lane; flat < total_blks; flat += 32) {
        u32 slot = flat / I32;
        u32 blk  = flat - slot * I32;
        const i32 exp_id = expert_ids[m * topk + slot];
        const size_t row = static_cast<size_t>(exp_id) * H + h;
        const auto* w_u4 = reinterpret_cast<const uint4*>(w2_weight + row * inter);
        const u8* s_row  = w2_scale_e8m0 + row * I32;
        const auto* a4   = reinterpret_cast<const uint4*>(act_in + (static_cast<size_t>(m) * topk + slot) * inter);

        f32 sc = ue8m0_to_f32(__ldg(s_row + blk));
        f32 d = 0.0f;
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            uint4 rw = __ldg(w_u4 + blk * 2 + hh);
            const auto* fw = reinterpret_cast<const __nv_fp8x2_e4m3*>(&rw);
            uint4 av0 = a4[blk * 4 + hh * 2 + 0];
            uint4 av1 = a4[blk * 4 + hh * 2 + 1];
            const auto* ab0 = reinterpret_cast<const __nv_bfloat162*>(&av0);
            const auto* ab1 = reinterpret_cast<const __nv_bfloat162*>(&av1);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 fa = __bfloat1622float2(ab0[k]);
                float2 ff = static_cast<float2>(fw[k]);
                d = fmaf(ff.x, fa.x, fmaf(ff.y, fa.y, d));
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 fa = __bfloat1622float2(ab1[k]);
                float2 ff = static_cast<float2>(fw[4 + k]);
                d = fmaf(ff.x, fa.x, fmaf(ff.y, fa.y, d));
            }
        }
        total_acc = fmaf(d, sc, total_acc);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) total_acc += __shfl_down_sync(0xFFFFFFFFu, total_acc, off);
    if (lane == 0) out[static_cast<size_t>(m) * H + h] = __float2bfloat16_rn(total_acc);
}

// ============================================================================
// 5. Fast Parallel 32-SM Router Stage-2 + Expert-Grouped Batched MoE Prefill
// ============================================================================
__global__ void inkling_moe_topk_from_logits_kernel(
    const __nv_bfloat16* __restrict__ router_logits, // [M, E]
    const f32* __restrict__ gate_bias,
    i32* __restrict__ out_ids,
    f32* __restrict__ out_weights,
    u32 M, u32 E, u32 topk, f32 route_scale) {
    const u32 m = blockIdx.x;
    const u32 e = threadIdx.x;
    if (m >= M) return;
    __shared__ f32 sm_logit[128];
    __shared__ f32 sm_score[128];
    __shared__ i32 sel_ids[8];
    __shared__ f32 sel_lp[8];
    if (e < E) {
        f32 lg = __bfloat162float(router_logits[static_cast<size_t>(m) * E + e]);
        sm_logit[e] = lg;
        sm_score[e] = (1.0f / (1.0f + expf(-lg))) + (gate_bias ? __ldg(gate_bias + e) : 0.0f);
    } else {
        sm_logit[e] = -1e30f;
        sm_score[e] = -1e30f;
    }
    __syncthreads();
    if (e == 0) {
        f32 max_lp = -1e30f;
        for (u32 k = 0; k < topk; ++k) {
            i32 best_i = 0; f32 best_s = -1e30f;
            for (u32 i = 0; i < E; ++i) {
                if (sm_score[i] > best_s) { best_s = sm_score[i]; best_i = static_cast<i32>(i); }
            }
            sm_score[best_i] = -1e30f;
            sel_ids[k] = best_i;
            f32 lp = log_sigmoid_f32(sm_logit[best_i]);
            sel_lp[k] = lp;
            max_lp = fmaxf(max_lp, lp);
        }
        f32 sum_e = 0.0f;
        for (u32 k = 0; k < topk; ++k) sum_e += expf(sel_lp[k] - max_lp);
        f32 lse = max_lp + logf(sum_e);
        for (u32 k = 0; k < topk; ++k) {
            out_ids[m * topk + k] = sel_ids[k];
            out_weights[m * topk + k] = expf(sel_lp[k] - lse) * route_scale;
        }
    }
}

// Dequantize 1 NVFP4 expert matrix [rows, cols/2] U8 + [rows, cols/16] FP8 -> [rows, cols] BF16
__global__ void dequant_nvfp4_expert_kernel(
    const u8* __restrict__ w_e, const u8* __restrict__ s_e,
    const f32* __restrict__ amax_e, const f32* __restrict__ denom,
    __nv_bfloat16* __restrict__ out_bf16, u32 rows, u32 cols) {
    u32 col2 = blockIdx.x * blockDim.x + threadIdx.x; // 1 thread = 2 weights (1 byte)
    u32 row  = blockIdx.y;
    u32 C2   = cols >> 1;
    if (row >= rows || col2 >= C2) return;
    u8 byte = w_e[static_cast<size_t>(row) * C2 + col2];
    __nv_fp8_e4m3 fs;
    *reinterpret_cast<u8*>(&fs) = s_e[static_cast<size_t>(row) * (cols >> 4) + (col2 >> 3)];
    f32 sc = static_cast<f32>(fs) * (__ldg(amax_e) / __ldg(denom));
    size_t dst = static_cast<size_t>(row) * cols + static_cast<size_t>(col2) * 2;
    out_bf16[dst + 0] = __float2bfloat16_rn(fp4_decode(byte) * sc);
    out_bf16[dst + 1] = __float2bfloat16_rn(fp4_decode(byte >> 4) * sc);
}

// Dequantize 1 NVFP4 expert directly to per-row FP8 E4M3 (1 warp per row, 4 warps/block -> 0 spills)
__global__ void dequant_nvfp4_expert_to_fp8_kernel(
    const u8* __restrict__ w_e, const u8* __restrict__ s_e,
    const f32* __restrict__ amax_e, const f32* __restrict__ denom,
    u8* __restrict__ out_fp8, f32* __restrict__ out_row_scales,
    u32 rows, u32 cols) {
    const u32 wid  = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 row  = blockIdx.x * 4 + wid;
    if (row >= rows) return;

    const u32 C2  = cols >> 1;
    const u32 C16 = cols >> 4;
    const f32 g_scale = __ldg(amax_e) / __ldg(denom);
    const u8* w_row = w_e + static_cast<size_t>(row) * C2;
    const u8* s_row = s_e + static_cast<size_t>(row) * C16;
    u8* dst = out_fp8 + static_cast<size_t>(row) * cols;

    // Pass 1: exact per-row amax (from block-16 scales * 6.0 upper bound? No, exact amax over nibbles for top fidelity)
    f32 am = 0.0f;
    for (u32 b16 = lane; b16 < C16; b16 += 32) {
        __nv_fp8_e4m3 fs;
        *reinterpret_cast<u8*>(&fs) = __ldg(s_row + b16);
        f32 sc = static_cast<f32>(fs) * g_scale;
        uint2 raw = __ldg(reinterpret_cast<const uint2*>(w_row) + b16);
        u32 p0 = raw.x, p1 = raw.y;
        f32 bmax = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            bmax = fmaxf(bmax, fabsf(fp4_decode(p0)));
            bmax = fmaxf(bmax, fabsf(fp4_decode(p1)));
            p0 >>= 4; p1 >>= 4;
        }
        am = fmaxf(am, bmax * sc);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, off));
    f32 r_scale = fmaxf(am / 448.0f, 1e-8f);
    f32 inv_r   = 1.0f / r_scale;
    if (lane == 0) out_row_scales[row] = r_scale;

    // Pass 2: write packed FP8 E4M3 (16 bytes per block-16)
    for (u32 b16 = lane; b16 < C16; b16 += 32) {
        __nv_fp8_e4m3 fs;
        *reinterpret_cast<u8*>(&fs) = __ldg(s_row + b16);
        f32 sc = static_cast<f32>(fs) * g_scale * inv_r;
        uint2 raw = __ldg(reinterpret_cast<const uint2*>(w_row) + b16);
        u32 p0 = raw.x, p1 = raw.y;
        u8* d16 = dst + b16 * 16;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            __nv_fp8_e4m3 f0(fp4_decode(p0) * sc);
            __nv_fp8_e4m3 f1(fp4_decode(p1) * sc);
            d16[k]     = *reinterpret_cast<const u8*>(&f0);
            d16[8 + k] = *reinterpret_cast<const u8*>(&f1);
            p0 >>= 4; p1 >>= 4;
        }
    }
}

// Dequantize 1 MXFP8 expert matrix [rows, cols] FP8 + [rows, cols/32] UE8M0 -> [rows, cols] BF16
__global__ void dequant_mxfp8_expert_kernel(
    const u8* __restrict__ w_e, const u8* __restrict__ s_e,
    __nv_bfloat16* __restrict__ out_bf16, u32 rows, u32 cols) {
    u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    u32 row = blockIdx.y;
    if (row >= rows || col >= cols) return;
    __nv_fp8_e4m3 fw;
    *reinterpret_cast<u8*>(&fw) = w_e[static_cast<size_t>(row) * cols + col];
    f32 sc = ue8m0_to_f32(s_e[static_cast<size_t>(row) * (cols >> 5) + (col >> 5)]);
    out_bf16[static_cast<size_t>(row) * cols + col] = __float2bfloat16_rn(static_cast<f32>(fw) * sc);
}

__global__ void inkling_moe_gather_kernel(
    const __nv_bfloat16* __restrict__ x, const i32* __restrict__ tok_idx,
    __nv_bfloat16* __restrict__ out, u32 cnt, u32 H) {
    u32 d = blockIdx.x * blockDim.x + threadIdx.x;
    u32 r = blockIdx.y;
    if (r >= cnt || d >= H) return;
    out[static_cast<size_t>(r) * H + d] = x[static_cast<size_t>(tok_idx[r]) * H + d];
}

// Interleaved (gate, up) [cnt, 2*inter] -> SiTU * gamma -> [cnt, inter]
__global__ void inkling_moe_situ_batched_kernel(
    const __nv_bfloat16* __restrict__ w13_out, const f32* __restrict__ gammas,
    __nv_bfloat16* __restrict__ act_out, u32 cnt, u32 inter, f32 gate_cap, f32 up_cap) {
    u32 j = blockIdx.x * blockDim.x + threadIdx.x;
    u32 r = blockIdx.y;
    if (r >= cnt || j >= inter) return;
    size_t base = static_cast<size_t>(r) * (2 * inter) + 2 * j;
    f32 g = __bfloat162float(w13_out[base + 0]);
    f32 u = __bfloat162float(w13_out[base + 1]);
    f32 sig = 1.0f / (1.0f + expf(-g));
    f32 y = ((softcap_f32(g, gate_cap) * sig) * softcap_f32(u, up_cap)) * gammas[r];
    act_out[static_cast<size_t>(r) * inter + j] = __float2bfloat16_rn(y);
}

__global__ void inkling_moe_scatter_add_kernel(
    const __nv_bfloat16* __restrict__ y, const i32* __restrict__ tok_idx,
    f32* __restrict__ moe_out_f32, u32 cnt, u32 H) {
    u32 d = blockIdx.x * blockDim.x + threadIdx.x;
    u32 r = blockIdx.y;
    if (r >= cnt || d >= H) return;
    atomicAdd(&moe_out_f32[static_cast<size_t>(tok_idx[r]) * H + d],
              __bfloat162float(y[static_cast<size_t>(r) * H + d]));
}

// Fused M=1..8 NVFP4 down_combine + mlp_sconv_hidden + residual add (-48 launches & -48 round-trips on BOTH base & MTP)
template <int MAX_M>
__global__ void inkling_moe_nvfp4_down_sconv_kernel(
    const __nv_bfloat16* __restrict__ act_in,
    const u8* __restrict__ w2_weight,
    const u8* __restrict__ w2_scale,
    const f32* __restrict__ w2_amax,
    const f32* __restrict__ w2_outer_denom,
    const i32* __restrict__ expert_ids,
    const __nv_bfloat16* __restrict__ sconv_w,
    __nv_bfloat16* __restrict__ hist_ring,
    __nv_bfloat16* __restrict__ out_dst,
    const __nv_bfloat16* __restrict__ add_a,
    const __nv_bfloat16* __restrict__ add_b,
    const i32* __restrict__ d_pos, u32 start_pos,
    u32 M, u32 H, u32 inter, u32 topk) {
    const u32 wid  = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 h    = blockIdx.x * 4 + wid;
    if (h >= H) return;

    const u32 I2  = inter >> 1;
    const u32 I16 = inter >> 4;
    const u32 total_blks = topk * I16;
    const f32 inv_denom = 1.0f / __ldg(w2_outer_denom);

    f32 cur_vals[MAX_M];
    #pragma unroll 1
    for (int m = 0; m < MAX_M; ++m) {
        if (m < static_cast<int>(M)) {
            f32 total_acc = 0.0f;
            for (u32 flat = lane; flat < total_blks; flat += 32) {
                u32 slot = flat / I16;
                u32 blk  = flat - slot * I16;
                const i32 exp_id = expert_ids[static_cast<u32>(m) * topk + slot];
                const f32 g_scale = __ldg(w2_amax + exp_id) * inv_denom;
                const size_t row = static_cast<size_t>(exp_id) * H + h;
                const auto* w_u2 = reinterpret_cast<const uint2*>(w2_weight + row * I2);
                const u8* s_row  = w2_scale + row * I16;
                const auto* a4   = reinterpret_cast<const uint4*>(
                    act_in + (static_cast<size_t>(m) * topk + slot) * inter);

                uint2 raw_w = __ldg(w_u2 + blk);
                __nv_fp8_e4m3 fs;
                *reinterpret_cast<u8*>(&fs) = __ldg(s_row + blk);
                f32 sc = static_cast<f32>(fs) * g_scale;

                uint4 av0 = a4[blk * 2 + 0];
                uint4 av1 = a4[blk * 2 + 1];
                const auto* ab0 = reinterpret_cast<const __nv_bfloat162*>(&av0);
                const auto* ab1 = reinterpret_cast<const __nv_bfloat162*>(&av1);

                f32 d = 0.0f;
                u32 pw0 = raw_w.x;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float2 fa = __bfloat1622float2(ab0[k]);
                    d = fmaf(fp4_decode(pw0),      fa.x, d);
                    d = fmaf(fp4_decode(pw0 >> 4), fa.y, d);
                    pw0 >>= 8;
                }
                u32 pw1 = raw_w.y;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float2 fa = __bfloat1622float2(ab1[k]);
                    d = fmaf(fp4_decode(pw1),      fa.x, d);
                    d = fmaf(fp4_decode(pw1 >> 4), fa.y, d);
                    pw1 >>= 8;
                }
                total_acc = fmaf(d, sc, total_acc);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) total_acc += __shfl_down_sync(0xFFFFFFFFu, total_acc, off);
            cur_vals[m] = total_acc;
        }
    }
    if (lane == 0) {
        u32 p0 = d_pos ? static_cast<u32>(*d_pos) : start_pos;
        const __nv_bfloat16* sw = sconv_w + static_cast<size_t>(h) * 4;
        f32 w3 = __bfloat162float(sw[0]), w2 = __bfloat162float(sw[1]);
        f32 w1 = __bfloat162float(sw[2]), w0 = __bfloat162float(sw[3]) + 1.0f;
        f32 h_m3 = (p0 >= 3) ? __bfloat162float(hist_ring[static_cast<size_t>((p0 - 3) & 15u) * H + h]) : 0.0f;
        f32 h_m2 = (p0 >= 2) ? __bfloat162float(hist_ring[static_cast<size_t>((p0 - 2) & 15u) * H + h]) : 0.0f;
        f32 h_m1 = (p0 >= 1) ? __bfloat162float(hist_ring[static_cast<size_t>((p0 - 1) & 15u) * H + h]) : 0.0f;
        const f32 inv_s5 = rsqrtf(5.0f);
        for (u32 m = 0; m < M; ++m) {
            __nv_bfloat16 cur_bf = __float2bfloat16_rn(cur_vals[m]);
            f32 c0 = __bfloat162float(cur_bf);
            hist_ring[static_cast<size_t>((p0 + m) & 15u) * H + h] = cur_bf;
            f32 y = (w3 * h_m3 + w2 * h_m2 + w1 * h_m1 + w0 * c0) * inv_s5;
            f32 yr = __bfloat162float(__float2bfloat16_rn(y));
            size_t off = static_cast<size_t>(m) * H + h;
            if (add_a) yr += __bfloat162float(add_a[off]);
            if (add_b) yr += __bfloat162float(add_b[off]);
            out_dst[off] = __float2bfloat16_rn(yr);
            h_m3 = h_m2; h_m2 = h_m1; h_m1 = c0;
        }
    }
}

// 1-block (~2us) active-expert compactor for M=2..8 (M*topk <= 64):
// Builds compact list of ONLY the U unique active experts + 64-bit bitmask of routed (m, k) slots
__global__ void inkling_moe_compact_experts_kernel(
    const i32* __restrict__ expert_ids, // [M * topk]
    u32* __restrict__ d_uniq_e,         // [64]
    u64* __restrict__ d_uniq_mask,      // [64]
    u32* __restrict__ d_uniq_count,     // [1]
    u32 total_slots) {
    __shared__ u64 sm_mask[128];
    const u32 tid = threadIdx.x; // 128 threads
    sm_mask[tid] = 0ULL;
    __syncthreads();
    if (tid < total_slots) {
        u32 e = static_cast<u32>(expert_ids[tid]);
        if (e < 128u) atomicOr(reinterpret_cast<unsigned long long*>(&sm_mask[e]), 1ULL << tid);
    }
    __syncthreads();
    if (tid == 0) {
        u32 u = 0;
        for (u32 e = 0; e < 128u; ++e) {
            u64 m = sm_mask[e];
            if (m != 0ULL) {
                d_uniq_e[u]    = e;
                d_uniq_mask[u] = m;
                u++;
            }
        }
        *d_uniq_count = u;
    }
}

// Zero-AtomicAdd, Zero-Spill Compact Unique-Expert NVFP4 Gate+Up+SiTU for M=2..8
// Grid: ((inter+3)/4, M*topk). Inactive y >= *d_uniq_count returns in 2 cycles
// Reads w13_weight[exp_id] ONCE from HBM and multiplies against all routed tokens (popc(mask) <= 8)
// decoding FP4 nibbles on-the-fly -> 0 stack spills AND ~40-55% less HBM traffic on M=8 MTP verify
__global__ void inkling_moe_compact_nvfp4_w13_kernel(
    const __nv_bfloat16* __restrict__ x,
    const u8* __restrict__ w13_weight,
    const u8* __restrict__ w13_scale,
    const f32* __restrict__ w13_amax,
    const f32* __restrict__ w13_outer_denom,
    const f32* __restrict__ expert_weights,
    __nv_bfloat16* __restrict__ act_out,
    const u32* __restrict__ d_uniq_e,
    const u64* __restrict__ d_uniq_mask,
    const u32* __restrict__ d_uniq_count,
    u32 inter, u32 H, u32 topk,
    f32 gate_cap, f32 up_cap) {
    const u32 u = blockIdx.y;
    if (u >= *d_uniq_count) return;
    const u32 wid  = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 j    = blockIdx.x * 4 + wid;
    if (j >= inter) return;

    const u32 exp_id = __ldg(d_uniq_e + u);
    const u64 mask0  = __ldg(d_uniq_mask + u);
    const u32 cnt    = static_cast<u32>(__popcll(mask0));

    const u32 H2  = H >> 1;
    const u32 H16 = H >> 4;
    const f32 g_scale = __ldg(w13_amax + exp_id) / __ldg(w13_outer_denom);

    const size_t row_g = static_cast<size_t>(exp_id) * (2 * inter) + (2 * j + 0);
    const size_t row_u = row_g + 1;
    const auto* wg_u2  = reinterpret_cast<const uint2*>(w13_weight + row_g * H2);
    const auto* wu_u2  = reinterpret_cast<const uint2*>(w13_weight + row_u * H2);
    const u8*   sg_row = w13_scale + row_g * H16;
    const u8*   su_row = w13_scale + row_u * H16;

    // Fast path for cnt == 1 (common singleton expert): 0 array overhead, pure scalar registers
    if (cnt == 1) {
        u32 slot_idx = static_cast<u32>(__ffsll(static_cast<long long>(mask0)) - 1);
        u32 m = slot_idx / topk;
        const auto* x4 = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * H);
        f32 acc_g = 0.0f, acc_u = 0.0f;
        for (u32 blk = lane; blk < H16; blk += 32) {
            uint2 rwg = __ldg(wg_u2 + blk), rwu = __ldg(wu_u2 + blk);
            __nv_fp8_e4m3 fsg, fsu;
            *reinterpret_cast<u8*>(&fsg) = __ldg(sg_row + blk);
            *reinterpret_cast<u8*>(&fsu) = __ldg(su_row + blk);
            f32 scg = static_cast<f32>(fsg), scu = static_cast<f32>(fsu);
            uint4 xv0 = x4[blk * 2 + 0], xv1 = x4[blk * 2 + 1];
            const auto* xb0 = reinterpret_cast<const __nv_bfloat162*>(&xv0);
            const auto* xb1 = reinterpret_cast<const __nv_bfloat162*>(&xv1);
            f32 dg = 0.0f, du = 0.0f;
            u32 pg0 = rwg.x, pu0 = rwu.x, pg1 = rwg.y, pu1 = rwu.y;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 fx0 = __bfloat1622float2(xb0[k]), fx1 = __bfloat1622float2(xb1[k]);
                dg = fmaf(fp4_decode(pg0), fx0.x, fmaf(fp4_decode(pg0 >> 4), fx0.y, dg));
                du = fmaf(fp4_decode(pu0), fx0.x, fmaf(fp4_decode(pu0 >> 4), fx0.y, du));
                dg = fmaf(fp4_decode(pg1), fx1.x, fmaf(fp4_decode(pg1 >> 4), fx1.y, dg));
                du = fmaf(fp4_decode(pu1), fx1.x, fmaf(fp4_decode(pu1 >> 4), fx1.y, du));
                pg0 >>= 8; pu0 >>= 8; pg1 >>= 8; pu1 >>= 8;
            }
            acc_g = fmaf(dg, scg, acc_g);
            acc_u = fmaf(du, scu, acc_u);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc_g += __shfl_down_sync(0xFFFFFFFFu, acc_g, off);
            acc_u += __shfl_down_sync(0xFFFFFFFFu, acc_u, off);
        }
        if (lane == 0) {
            f32 g_raw = acc_g * g_scale;
            f32 u_raw = acc_u * g_scale;
            f32 sig   = 1.0f / (1.0f + expf(-g_raw));
            f32 g_cap = softcap_f32(g_raw, gate_cap);
            f32 u_cap = softcap_f32(u_raw, up_cap);
            f32 y     = ((g_cap * sig) * u_cap) * __ldg(expert_weights + slot_idx);
            act_out[static_cast<size_t>(slot_idx) * inter + j] = __float2bfloat16_rn(y);
        }
        return;
    }

    // Shared expert path (cnt = 2..8): store slot indices in shared memory per warp (32 B/warp -> 0 stack frame)
    __shared__ u8 sm_slots[4][8];
    if (lane == 0) {
        u64 tmp = mask0;
        u32 c = 0;
        while (tmp && c < 8) {
            u32 s = static_cast<u32>(__ffsll(static_cast<long long>(tmp)) - 1);
            sm_slots[wid][c++] = static_cast<u8>(s);
            tmp &= (tmp - 1ULL);
        }
    }
    __syncwarp();

    f32 g0=0,g1=0,g2=0,g3=0,g4=0,g5=0,g6=0,g7=0;
    f32 u0=0,u1=0,u2=0,u3=0,u4=0,u5=0,u6=0,u7=0;
    for (u32 blk = lane; blk < H16; blk += 32) {
        uint2 rwg = __ldg(wg_u2 + blk), rwu = __ldg(wu_u2 + blk);
        __nv_fp8_e4m3 fsg, fsu;
        *reinterpret_cast<u8*>(&fsg) = __ldg(sg_row + blk);
        *reinterpret_cast<u8*>(&fsu) = __ldg(su_row + blk);
        f32 scg = static_cast<f32>(fsg), scu = static_cast<f32>(fsu);

        #pragma unroll 1
        for (u32 c = 0; c < cnt; ++c) {
            u32 m = static_cast<u32>(sm_slots[wid][c]) / topk;
            const auto* x4 = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * H);
            uint4 xv0 = x4[blk * 2 + 0], xv1 = x4[blk * 2 + 1];
            const auto* xb0 = reinterpret_cast<const __nv_bfloat162*>(&xv0);
            const auto* xb1 = reinterpret_cast<const __nv_bfloat162*>(&xv1);
            f32 dg = 0.0f, du = 0.0f;
            u32 pg0 = rwg.x, pu0 = rwu.x, pg1 = rwg.y, pu1 = rwu.y;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 fx0 = __bfloat1622float2(xb0[k]), fx1 = __bfloat1622float2(xb1[k]);
                dg = fmaf(fp4_decode(pg0), fx0.x, fmaf(fp4_decode(pg0 >> 4), fx0.y, dg));
                du = fmaf(fp4_decode(pu0), fx0.x, fmaf(fp4_decode(pu0 >> 4), fx0.y, du));
                dg = fmaf(fp4_decode(pg1), fx1.x, fmaf(fp4_decode(pg1 >> 4), fx1.y, dg));
                du = fmaf(fp4_decode(pu1), fx1.x, fmaf(fp4_decode(pu1 >> 4), fx1.y, du));
                pg0 >>= 8; pu0 >>= 8; pg1 >>= 8; pu1 >>= 8;
            }
            if      (c == 0) { g0 = fmaf(dg, scg, g0); u0 = fmaf(du, scu, u0); }
            else if (c == 1) { g1 = fmaf(dg, scg, g1); u1 = fmaf(du, scu, u1); }
            else if (c == 2) { g2 = fmaf(dg, scg, g2); u2 = fmaf(du, scu, u2); }
            else if (c == 3) { g3 = fmaf(dg, scg, g3); u3 = fmaf(du, scu, u3); }
            else if (c == 4) { g4 = fmaf(dg, scg, g4); u4 = fmaf(du, scu, u4); }
            else if (c == 5) { g5 = fmaf(dg, scg, g5); u5 = fmaf(du, scu, u5); }
            else if (c == 6) { g6 = fmaf(dg, scg, g6); u6 = fmaf(du, scu, u6); }
            else             { g7 = fmaf(dg, scg, g7); u7 = fmaf(du, scu, u7); }
        }
    }
    #pragma unroll 1
    for (u32 c = 0; c < cnt; ++c) {
        f32 g  = (c==0)?g0:(c==1)?g1:(c==2)?g2:(c==3)?g3:(c==4)?g4:(c==5)?g5:(c==6)?g6:g7;
        f32 uv = (c==0)?u0:(c==1)?u1:(c==2)?u2:(c==3)?u3:(c==4)?u4:(c==5)?u5:(c==6)?u6:u7;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            g  += __shfl_down_sync(0xFFFFFFFFu, g, off);
            uv += __shfl_down_sync(0xFFFFFFFFu, uv, off);
        }
        if (lane == 0) {
            u32 slot_idx = static_cast<u32>(sm_slots[wid][c]);
            f32 g_raw = g * g_scale;
            f32 u_raw = uv * g_scale;
            f32 sig   = 1.0f / (1.0f + expf(-g_raw));
            f32 g_cap = softcap_f32(g_raw, gate_cap);
            f32 u_cap = softcap_f32(u_raw, up_cap);
            f32 y     = ((g_cap * sig) * u_cap) * __ldg(expert_weights + slot_idx);
            act_out[static_cast<size_t>(slot_idx) * inter + j] = __float2bfloat16_rn(y);
        }
    }
}

// ============================================================================
// 7. Blackwell sm_120a Hardware PTX mxf4nvf4.m16n8k64 FP4 Tensor-Core
// Zero-Host-Sync 1-Launch Grouped MoE (Prefill M>=16 AND MTP M=2..8)
// ============================================================================
__device__ __forceinline__ u8 f32_to_fp4_e2m1(f32 x) {
    u8 sign = (x < 0.0f) ? 8u : 0u;
    f32 ax = fabsf(x);
    u8 mag = (ax < 0.25f) ? 0u : (ax < 0.75f) ? 1u : (ax < 1.25f) ? 2u :
             (ax < 1.75f) ? 3u : (ax < 2.50f) ? 4u : (ax < 3.50f) ? 5u :
             (ax < 5.00f) ? 6u : 7u;
    return static_cast<u8>(sign | mag);
}

// 1-block (~5us) GPU Grouped-MoE Tile Builder (ZERO host sync, 100% CUDA-Graph capturable)
__global__ void inkling_moe_build_tiles_kernel(
    const i32* __restrict__ expert_ids,     // [M * topk]
    const f32* __restrict__ expert_weights, // [M * topk]
    i32* __restrict__ d_pack_idx,           // [M * topk] token m
    f32* __restrict__ d_pack_gam,           // [M * topk] gamma
    u32* __restrict__ d_tile_meta,          // [max_tiles]: (exp_id << 20) | pack_start
    u32* __restrict__ d_tile_cnt,           // [max_tiles]: 1..16
    u32* __restrict__ d_num_tiles,          // [1]
    u32 total_slots, u32 E, u32 topk) {
    __shared__ u32 sm_cnt[128];
    __shared__ u32 sm_wpos[128];
    const u32 tid = threadIdx.x; // 256 threads
    if (tid < 128u) sm_cnt[tid] = 0u;
    __syncthreads();

    for (u32 idx = tid; idx < total_slots; idx += 256u) {
        u32 e = static_cast<u32>(expert_ids[idx]);
        if (e < E) atomicAdd(&sm_cnt[e], 1u);
    }
    __syncthreads();

    if (tid == 0) {
        u32 off = 0u;
        u32 n_tiles = 0u;
        for (u32 e = 0; e < E; ++e) {
            u32 c = sm_cnt[e];
            sm_wpos[e] = off;
            u32 rem = c;
            u32 cur = off;
            while (rem > 0u) {
                u32 tc = (rem > 16u) ? 16u : rem;
                d_tile_meta[n_tiles] = (e << 20) | cur;
                d_tile_cnt[n_tiles]  = tc;
                n_tiles++;
                cur += tc;
                rem -= tc;
            }
            off += c;
        }
        *d_num_tiles = n_tiles;
    }
    __syncthreads();

    for (u32 idx = tid; idx < total_slots; idx += 256u) {
        u32 e = static_cast<u32>(expert_ids[idx]);
        if (e < E) {
            u32 p = atomicAdd(&sm_wpos[e], 1u);
            d_pack_idx[p] = static_cast<i32>(idx / topk);
            d_pack_gam[p] = expert_weights[idx];
        }
    }
}

// Quantize [rows, cols] BF16 -> NVFP4 E2M1 [rows, cols/2] + UE4M3 block-16 [rows, cols/16] + F32 gscale [rows]
// Grid: ((rows + 3)/4), Block: 128 threads (4 warps -> 1 warp/row, 0 spills)
__global__ void inkling_quant_bf16_to_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ src,
    u8* __restrict__ out_fp4,
    u8* __restrict__ out_sc,
    f32* __restrict__ out_gs,
    u32 rows, u32 cols) {
    const u32 wid  = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 row  = blockIdx.x * 4 + wid;
    if (row >= rows) return;

    const u32 C16 = cols >> 4;
    const auto* x4 = reinterpret_cast<const uint4*>(src + static_cast<size_t>(row) * cols);
    f32 am = 0.0f;
    for (u32 b16 = lane; b16 < C16; b16 += 32) {
        uint4 v0 = x4[b16 * 2 + 0], v1 = x4[b16 * 2 + 1];
        const auto* b0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
        const auto* b1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 f0 = __bfloat1622float2(b0[k]), f1 = __bfloat1622float2(b1[k]);
            am = fmaxf(am, fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y))));
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, off));
    f32 gs = fmaxf(am / (448.0f * 6.0f), 1e-12f);
    if (lane == 0) out_gs[row] = gs;

    auto* dst_u2 = reinterpret_cast<uint2*>(out_fp4 + static_cast<size_t>(row) * (cols >> 1));
    u8*   dst_sc = out_sc + static_cast<size_t>(row) * C16;
    for (u32 b16 = lane; b16 < C16; b16 += 32) {
        uint4 v0 = x4[b16 * 2 + 0], v1 = x4[b16 * 2 + 1];
        const auto* b0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
        const auto* b1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
        f32 vals[16];
        f32 bmax = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 f0 = __bfloat1622float2(b0[k]), f1 = __bfloat1622float2(b1[k]);
            vals[2 * k + 0]     = f0.x; vals[2 * k + 1]     = f0.y;
            vals[8 + 2 * k + 0] = f1.x; vals[8 + 2 * k + 1] = f1.y;
            bmax = fmaxf(bmax, fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y))));
        }
        __nv_fp8_e4m3 fs(bmax / (6.0f * gs));
        u8 fs_u8 = (*reinterpret_cast<const u8*>(&fs)) & 0x7Fu;
        dst_sc[b16] = fs_u8;
        f32 eff = static_cast<f32>(fs) * gs;
        f32 inv = (eff > 0.0f) ? (1.0f / eff) : 0.0f;
        u32 p0 = 0u, p1 = 0u;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            p0 |= static_cast<u32>(f32_to_fp4_e2m1(vals[k] * inv)) << (k * 4);
            p1 |= static_cast<u32>(f32_to_fp4_e2m1(vals[8 + k] * inv)) << (k * 4);
        }
        dst_u2[b16] = make_uint2(p0, p1);
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
__device__ __forceinline__ void mma_mxf4_m16n8k64(
    f32& d0, f32& d1, f32& d2, f32& d3,
    u32 a0, u32 a1, u32 a2, u32 a3,
    u32 b0, u32 b1, u32 sa, u32 sb) {
    u16 zero_h = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {%12, %12}, %11, {%12, %12};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(sa), "r"(sb), "h"(zero_h)
    );
}
#endif

// 1-Launch Blackwell sm_120a Hardware FP4 Tensor-Core Grouped w13 + Register-Fused SiTU
// Grid: (inter / 16, max_tiles), Block: 128 threads (4 warps x 8 interleaved w13 rows = 16 (gate,up) pairs)
__global__ void inkling_moe_grouped_w13_fp4tc_kernel(
    const u8* __restrict__ d_x_fp4,
    const u8* __restrict__ d_x_fp4_sc,
    const f32* __restrict__ d_x_fp4_gs,
    const u8* __restrict__ w13_weight,
    const u8* __restrict__ w13_scale,
    const f32* __restrict__ w13_amax,
    const f32* __restrict__ w13_denom,
    const i32* __restrict__ d_pack_idx,
    const f32* __restrict__ d_pack_gam,
    const u32* __restrict__ d_tile_meta,
    const u32* __restrict__ d_tile_cnt,
    const u32* __restrict__ d_num_tiles,
    __nv_bfloat16* __restrict__ d_act_bf16, // [total_slots, inter]
    u32 H, u32 inter, f32 gate_cap, f32 up_cap) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const u32 t = blockIdx.y;
    if (t >= *d_num_tiles) return;

    const u32 tid_blk = threadIdx.x;
    const u32 wid  = tid_blk >> 5;
    const u32 lane = tid_blk & 31;
    const u32 gid  = lane >> 2; // 0..7
    const u32 tid  = lane & 3;  // 0..3

    const u32 meta    = __ldg(d_tile_meta + t);
    const u32 exp_id  = meta >> 20;
    const u32 p_start = meta & 0xFFFFFu;
    const u32 cnt     = __ldg(d_tile_cnt + t);

    __shared__ i32 sm_tok[16];
    __shared__ f32 sm_gam[16];
    __shared__ f32 sm_xgs[16];
    if (tid_blk < 16u) {
        u32 valid = (tid_blk < cnt) ? tid_blk : 0u;
        i32 tok = __ldg(d_pack_idx + p_start + valid);
        sm_tok[tid_blk] = tok;
        sm_gam[tid_blk] = (tid_blk < cnt) ? __ldg(d_pack_gam + p_start + tid_blk) : 0.0f;
        sm_xgs[tid_blk] = (tid_blk < cnt) ? __ldg(d_x_fp4_gs + tok) : 0.0f;
    }
    __syncthreads();

    const u32 H2  = H >> 1;
    const u32 H16 = H >> 4;
    const i32 tok_lo = sm_tok[gid];
    const i32 tok_hi = sm_tok[gid + 8];
    const bool v_lo  = (gid < cnt);
    const bool v_hi  = ((gid + 8u) < cnt);

    const u8* x_lo_p = d_x_fp4 + static_cast<size_t>(tok_lo) * H2;
    const u8* x_hi_p = d_x_fp4 + static_cast<size_t>(tok_hi) * H2;
    const u8* s_lo_p = d_x_fp4_sc + static_cast<size_t>(tok_lo) * H16;
    const u8* s_hi_p = d_x_fp4_sc + static_cast<size_t>(tok_hi) * H16;

    const u32 n_row = blockIdx.x * 32u + wid * 8u + gid;
    const size_t w_row_idx = static_cast<size_t>(exp_id) * (2u * inter) + n_row;
    const u8* w_p  = w13_weight + w_row_idx * H2;
    const u8* ws_p = w13_scale  + w_row_idx * H16;

    f32 d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (u32 k0 = 0; k0 < H; k0 += 64u) {
        const u32 k2  = k0 >> 1;
        const u32 k16 = k0 >> 4;
        const auto* xlo32 = reinterpret_cast<const u32*>(x_lo_p + k2);
        const auto* xhi32 = reinterpret_cast<const u32*>(x_hi_p + k2);
        const auto* w32   = reinterpret_cast<const u32*>(w_p + k2);

        u32 a0 = v_lo ? __ldg(xlo32 + tid)     : 0u;
        u32 a1 = v_hi ? __ldg(xhi32 + tid)     : 0u;
        u32 a2 = v_lo ? __ldg(xlo32 + tid + 4) : 0u;
        u32 a3 = v_hi ? __ldg(xhi32 + tid + 4) : 0u;

        u32 b0 = __ldg(w32 + tid);
        u32 b1 = __ldg(w32 + tid + 4);

        u32 sa = 0u;
        if (tid == 0 && v_lo) sa = *reinterpret_cast<const u32*>(s_lo_p + k16);
        else if (tid == 1 && v_hi) sa = *reinterpret_cast<const u32*>(s_hi_p + k16);
        u32 sb = __ldg(reinterpret_cast<const u32*>(ws_p + k16));

        mma_mxf4_m16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sa, sb);
    }

    // Interleaved w13 rows: d0 = gate[j], d1 = up[j] for row gid; d2 = gate[j], d3 = up[j] for row gid+8
    const f32 w_gs = __ldg(w13_amax + exp_id) / __ldg(w13_denom);
    const u32 j = blockIdx.x * 16u + wid * 4u + tid;
    if (v_lo) {
        f32 sc = w_gs * sm_xgs[gid];
        f32 g_raw = d0 * sc, u_raw = d1 * sc;
        f32 y = ((softcap_f32(g_raw, gate_cap) / (1.0f + expf(-g_raw))) * softcap_f32(u_raw, up_cap)) * sm_gam[gid];
        d_act_bf16[static_cast<size_t>(p_start + gid) * inter + j] = __float2bfloat16_rn(y);
    }
    if (v_hi) {
        f32 sc = w_gs * sm_xgs[gid + 8u];
        f32 g_raw = d2 * sc, u_raw = d3 * sc;
        f32 y = ((softcap_f32(g_raw, gate_cap) / (1.0f + expf(-g_raw))) * softcap_f32(u_raw, up_cap)) * sm_gam[gid + 8u];
        d_act_bf16[static_cast<size_t>(p_start + gid + 8u) * inter + j] = __float2bfloat16_rn(y);
    }
#endif
}

// 1-Launch Blackwell sm_120a Hardware FP4 Tensor-Core Grouped w2 + Scatter-Add
// Grid: (H / 32, max_tiles), Block: 128 threads (4 warps x 8 output hidden dims = 32 hidden dims)
__global__ void inkling_moe_grouped_w2_fp4tc_kernel(
    const u8* __restrict__ d_act_fp4,
    const u8* __restrict__ d_act_fp4_sc,
    const f32* __restrict__ d_act_fp4_gs,
    const u8* __restrict__ w2_weight,
    const u8* __restrict__ w2_scale,
    const f32* __restrict__ w2_amax,
    const f32* __restrict__ w2_denom,
    const i32* __restrict__ d_pack_idx,
    const u32* __restrict__ d_tile_meta,
    const u32* __restrict__ d_tile_cnt,
    const u32* __restrict__ d_num_tiles,
    f32* __restrict__ d_moe_out_f32, // [M, H] zeroed prior
    u32 H, u32 inter) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const u32 t = blockIdx.y;
    if (t >= *d_num_tiles) return;

    const u32 tid_blk = threadIdx.x;
    const u32 wid  = tid_blk >> 5;
    const u32 lane = tid_blk & 31;
    const u32 gid  = lane >> 2;
    const u32 tid  = lane & 3;

    const u32 meta    = __ldg(d_tile_meta + t);
    const u32 exp_id  = meta >> 20;
    const u32 p_start = meta & 0xFFFFFu;
    const u32 cnt     = __ldg(d_tile_cnt + t);

    __shared__ i32 sm_tok[16];
    __shared__ f32 sm_ags[16];
    if (tid_blk < 16u) {
        u32 valid = (tid_blk < cnt) ? tid_blk : 0u;
        sm_tok[tid_blk] = __ldg(d_pack_idx + p_start + valid);
        sm_ags[tid_blk] = (tid_blk < cnt) ? __ldg(d_act_fp4_gs + p_start + tid_blk) : 0.0f;
    }
    __syncthreads();

    const u32 I2  = inter >> 1;
    const u32 I16 = inter >> 4;
    const bool v_lo = (gid < cnt);
    const bool v_hi = ((gid + 8u) < cnt);
    const u32 p_lo  = p_start + (v_lo ? gid : 0u);
    const u32 p_hi  = p_start + (v_hi ? (gid + 8u) : 0u);

    const u8* a_lo_p = d_act_fp4 + static_cast<size_t>(p_lo) * I2;
    const u8* a_hi_p = d_act_fp4 + static_cast<size_t>(p_hi) * I2;
    const u8* s_lo_p = d_act_fp4_sc + static_cast<size_t>(p_lo) * I16;
    const u8* s_hi_p = d_act_fp4_sc + static_cast<size_t>(p_hi) * I16;

    const u32 h_row = blockIdx.x * 32u + wid * 8u + gid;
    const size_t w_row_idx = static_cast<size_t>(exp_id) * H + h_row;
    const u8* w_p  = w2_weight + w_row_idx * I2;
    const u8* ws_p = w2_scale  + w_row_idx * I16;

    f32 d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (u32 k0 = 0; k0 < inter; k0 += 64u) {
        const u32 k2  = k0 >> 1;
        const u32 k16 = k0 >> 4;
        const auto* alo32 = reinterpret_cast<const u32*>(a_lo_p + k2);
        const auto* ahi32 = reinterpret_cast<const u32*>(a_hi_p + k2);
        const auto* w32   = reinterpret_cast<const u32*>(w_p + k2);

        u32 a0 = v_lo ? __ldg(alo32 + tid)     : 0u;
        u32 a1 = v_hi ? __ldg(ahi32 + tid)     : 0u;
        u32 a2 = v_lo ? __ldg(alo32 + tid + 4) : 0u;
        u32 a3 = v_hi ? __ldg(ahi32 + tid + 4) : 0u;

        u32 b0 = __ldg(w32 + tid);
        u32 b1 = __ldg(w32 + tid + 4);

        u32 sa = 0u;
        if (tid == 0 && v_lo) sa = *reinterpret_cast<const u32*>(s_lo_p + k16);
        else if (tid == 1 && v_hi) sa = *reinterpret_cast<const u32*>(s_hi_p + k16);
        u32 sb = __ldg(reinterpret_cast<const u32*>(ws_p + k16));

        mma_mxf4_m16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sa, sb);
    }

    const f32 w_gs = __ldg(w2_amax + exp_id) / __ldg(w2_denom);
    const u32 h0 = blockIdx.x * 32u + wid * 8u + tid * 2u;
    if (v_lo) {
        f32 sc = w_gs * sm_ags[gid];
        size_t base = static_cast<size_t>(sm_tok[gid]) * H + h0;
        atomicAdd(&d_moe_out_f32[base + 0], d0 * sc);
        atomicAdd(&d_moe_out_f32[base + 1], d1 * sc);
    }
    if (v_hi) {
        f32 sc = w_gs * sm_ags[gid + 8u];
        size_t base = static_cast<size_t>(sm_tok[gid + 8u]) * H + h0;
        atomicAdd(&d_moe_out_f32[base + 0], d2 * sc);
        atomicAdd(&d_moe_out_f32[base + 1], d3 * sc);
    }
#endif
}

__global__ void f32_to_bf16_kernel(const f32* __restrict__ src, __nv_bfloat16* __restrict__ dst, u32 total) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) dst[i] = __float2bfloat16_rn(src[i]);
}

} // namespace

void launch_inkling_moe_router(const __nv_bfloat16* x,
                               const __nv_bfloat16* gate_w,
                               const f32* gate_bias,
                               i32* out_ids,
                               f32* out_weights,
                               u32 M, u32 H, u32 E, u32 topk, f32 route_scale,
                               cudaStream_t stream) {
    inkling_moe_router_kernel<<<M, 128, 0, stream>>>(
        x, gate_w, gate_bias, out_ids, out_weights, M, H, E, topk, route_scale);
}

void launch_inkling_moe_nvfp4_gate_up_situ(const __nv_bfloat16* x,
                                           const u8* w13_weight,
                                           const u8* w13_scale,
                                           const f32* w13_amax,
                                           const f32* w13_outer_denom,
                                           const i32* expert_ids,
                                           const f32* expert_weights,
                                           __nv_bfloat16* act_out,
                                           u32 M, u32 inter, u32 K, u32 topk,
                                           f32 gate_cap, f32 up_cap,
                                           cudaStream_t stream) {
    dim3 grid((inter + 3) / 4, topk, M);
    inkling_moe_nvfp4_gate_up_situ_kernel<<<grid, 128, 0, stream>>>(
        x, w13_weight, w13_scale, w13_amax, w13_outer_denom,
        expert_ids, expert_weights, act_out, inter, K, topk, gate_cap, up_cap);
}

void launch_inkling_moe_nvfp4_down_combine(const __nv_bfloat16* act_in,
                                           const u8* w2_weight,
                                           const u8* w2_scale,
                                           const f32* w2_amax,
                                           const f32* w2_outer_denom,
                                           const i32* expert_ids,
                                           __nv_bfloat16* out,
                                           u32 M, u32 H, u32 inter, u32 topk,
                                           cudaStream_t stream) {
    dim3 grid((H + 3) / 4, M);
    inkling_moe_nvfp4_down_combine_kernel<<<grid, 128, 0, stream>>>(
        act_in, w2_weight, w2_scale, w2_amax, w2_outer_denom,
        expert_ids, out, H, inter, topk);
}

void launch_inkling_moe_mxfp8_gate_up_situ(const __nv_bfloat16* x,
                                           const u8* w13_weight,
                                           const u8* w13_scale_e8m0,
                                           const i32* expert_ids,
                                           const f32* expert_weights,
                                           __nv_bfloat16* act_out,
                                           u32 M, u32 inter, u32 K, u32 topk,
                                           f32 gate_cap, f32 up_cap,
                                           cudaStream_t stream) {
    dim3 grid((inter + 3) / 4, topk, M);
    inkling_moe_mxfp8_gate_up_situ_kernel<<<grid, 128, 0, stream>>>(
        x, w13_weight, w13_scale_e8m0, expert_ids, expert_weights,
        act_out, inter, K, topk, gate_cap, up_cap);
}

void launch_inkling_moe_mxfp8_down_combine(const __nv_bfloat16* act_in,
                                           const u8* w2_weight,
                                           const u8* w2_scale_e8m0,
                                           const i32* expert_ids,
                                           __nv_bfloat16* out,
                                           u32 M, u32 H, u32 inter, u32 topk,
                                           cudaStream_t stream) {
    dim3 grid((H + 3) / 4, M);
    inkling_moe_mxfp8_down_combine_kernel<<<grid, 128, 0, stream>>>(
        act_in, w2_weight, w2_scale_e8m0, expert_ids, out, H, inter, topk);
}

void launch_inkling_moe_topk_from_logits(const __nv_bfloat16* router_logits,
                                         const f32* gate_bias,
                                         i32* out_ids,
                                         f32* out_weights,
                                         u32 M, u32 E, u32 topk, f32 route_scale,
                                         cudaStream_t stream) {
    inkling_moe_topk_from_logits_kernel<<<M, 128, 0, stream>>>(
        router_logits, gate_bias, out_ids, out_weights, M, E, topk, route_scale);
}

void launch_dequant_nvfp4_expert(const u8* w_e, const u8* s_e, const f32* amax_e, const f32* denom,
                                 __nv_bfloat16* out_bf16, u32 rows, u32 cols, cudaStream_t stream) {
    dim3 grid(((cols >> 1) + 255) / 256, rows);
    dequant_nvfp4_expert_kernel<<<grid, 256, 0, stream>>>(w_e, s_e, amax_e, denom, out_bf16, rows, cols);
}

void launch_dequant_nvfp4_expert_to_fp8(const u8* w_e, const u8* s_e, const f32* amax_e, const f32* denom,
                                        u8* out_fp8, f32* out_row_scales, u32 rows, u32 cols, cudaStream_t stream) {
    dequant_nvfp4_expert_to_fp8_kernel<<<(rows + 3) / 4, 128, 0, stream>>>(
        w_e, s_e, amax_e, denom, out_fp8, out_row_scales, rows, cols);
}

void launch_inkling_moe_grouped_nvfp4_fp4tc(
    const __nv_bfloat16* x_in,
    const u8* w13_weight, const u8* w13_scale, const f32* w13_amax, const f32* w13_denom,
    const u8* w2_weight,  const u8* w2_scale,  const f32* w2_amax,  const f32* w2_denom,
    const i32* expert_ids, const f32* expert_weights,
    i32* d_pack_idx, f32* d_pack_gam,
    u32* d_tile_meta, u32* d_tile_cnt, u32* d_num_tiles,
    u8* d_x_fp4, u8* d_x_fp4_sc, f32* d_x_fp4_gs,
    __nv_bfloat16* d_act_bf16, u8* d_act_fp4, u8* d_act_fp4_sc, f32* d_act_fp4_gs,
    f32* d_moe_out_f32, __nv_bfloat16* d_moe_out,
    u32 M, u32 H, u32 inter, u32 E, u32 topk,
    f32 gate_cap, f32 up_cap, cudaStream_t stream) {
    const u32 total_slots = M * topk;
    const u32 max_tiles = (total_slots < E) ? total_slots : (E + total_slots / 16u);

    // 1. Build flat 16-row tile worklist on GPU (1 block, ~5us, ZERO host sync)
    inkling_moe_build_tiles_kernel<<<1, 256, 0, stream>>>(
        expert_ids, expert_weights, d_pack_idx, d_pack_gam, d_tile_meta, d_tile_cnt, d_num_tiles,
        total_slots, E, topk);

    // 2. Quantize input activations [M, H] BF16 -> NVFP4 E2M1 + UE4M3 block-16
    inkling_quant_bf16_to_nvfp4_kernel<<<(M + 3) / 4, 128, 0, stream>>>(
        x_in, d_x_fp4, d_x_fp4_sc, d_x_fp4_gs, M, H);

    // 3. 1-Launch Hardware Blackwell FP4 Tensor-Core Grouped w13 + Register-Fused SiTU
    dim3 g_w13(inter / 16u, max_tiles);
    inkling_moe_grouped_w13_fp4tc_kernel<<<g_w13, 128, 0, stream>>>(
        d_x_fp4, d_x_fp4_sc, d_x_fp4_gs,
        w13_weight, w13_scale, w13_amax, w13_denom,
        d_pack_idx, d_pack_gam, d_tile_meta, d_tile_cnt, d_num_tiles,
        d_act_bf16, H, inter, gate_cap, up_cap);

    // 4. Quantize SiTU activations [total_slots, inter] BF16 -> NVFP4 E2M1 + UE4M3 block-16
    inkling_quant_bf16_to_nvfp4_kernel<<<(total_slots + 3) / 4, 128, 0, stream>>>(
        d_act_bf16, d_act_fp4, d_act_fp4_sc, d_act_fp4_gs, total_slots, inter);

    // 5. 1-Launch Hardware Blackwell FP4 Tensor-Core Grouped w2 + Scatter-Add -> BF16
    cudaMemsetAsync(d_moe_out_f32, 0, static_cast<size_t>(M) * H * sizeof(f32), stream);
    dim3 g_w2(H / 32u, max_tiles);
    inkling_moe_grouped_w2_fp4tc_kernel<<<g_w2, 128, 0, stream>>>(
        d_act_fp4, d_act_fp4_sc, d_act_fp4_gs,
        w2_weight, w2_scale, w2_amax, w2_denom,
        d_pack_idx, d_tile_meta, d_tile_cnt, d_num_tiles,
        d_moe_out_f32, H, inter);
    f32_to_bf16_kernel<<<(M * H + 255) / 256, 256, 0, stream>>>(d_moe_out_f32, d_moe_out, M * H);
}

void launch_dequant_mxfp8_expert(const u8* w_e, const u8* s_e,
                                 __nv_bfloat16* out_bf16, u32 rows, u32 cols, cudaStream_t stream) {
    dim3 grid((cols + 255) / 256, rows);
    dequant_mxfp8_expert_kernel<<<grid, 256, 0, stream>>>(w_e, s_e, out_bf16, rows, cols);
}

void launch_inkling_moe_gather(const __nv_bfloat16* x, const i32* tok_idx, __nv_bfloat16* out,
                               u32 cnt, u32 H, cudaStream_t stream) {
    dim3 grid((H + 255) / 256, cnt);
    inkling_moe_gather_kernel<<<grid, 256, 0, stream>>>(x, tok_idx, out, cnt, H);
}

void launch_inkling_moe_situ_batched(const __nv_bfloat16* w13_out, const f32* gammas,
                                     __nv_bfloat16* act_out, u32 cnt, u32 inter,
                                     f32 gate_cap, f32 up_cap, cudaStream_t stream) {
    dim3 grid((inter + 255) / 256, cnt);
    inkling_moe_situ_batched_kernel<<<grid, 256, 0, stream>>>(w13_out, gammas, act_out, cnt, inter, gate_cap, up_cap);
}

void launch_inkling_moe_scatter_add(const __nv_bfloat16* y, const i32* tok_idx,
                                    f32* moe_out_f32, u32 cnt, u32 H, cudaStream_t stream) {
    dim3 grid((H + 255) / 256, cnt);
    inkling_moe_scatter_add_kernel<<<grid, 256, 0, stream>>>(y, tok_idx, moe_out_f32, cnt, H);
}

void launch_f32_to_bf16(const f32* src, __nv_bfloat16* dst, u32 total, cudaStream_t stream) {
    f32_to_bf16_kernel<<<(total + 255) / 256, 256, 0, stream>>>(src, dst, total);
}

void launch_inkling_moe_nvfp4_down_sconv(const __nv_bfloat16* act_in,
                                         const u8* w2_weight,
                                         const u8* w2_scale,
                                         const f32* w2_amax,
                                         const f32* w2_outer_denom,
                                         const i32* expert_ids,
                                         const __nv_bfloat16* sconv_w,
                                         __nv_bfloat16* hist_ring,
                                         __nv_bfloat16* out_dst,
                                         const __nv_bfloat16* add_a,
                                         const __nv_bfloat16* add_b,
                                         const i32* d_pos, u32 start_pos,
                                         u32 M, u32 H, u32 inter, u32 topk,
                                         cudaStream_t stream) {
    u32 blocks = (H + 3) / 4;
    if (M == 1) {
        inkling_moe_nvfp4_down_sconv_kernel<1><<<blocks, 128, 0, stream>>>(
            act_in, w2_weight, w2_scale, w2_amax, w2_outer_denom, expert_ids,
            sconv_w, hist_ring, out_dst, add_a, add_b, d_pos, start_pos, M, H, inter, topk);
    } else if (M <= 4) {
        inkling_moe_nvfp4_down_sconv_kernel<4><<<blocks, 128, 0, stream>>>(
            act_in, w2_weight, w2_scale, w2_amax, w2_outer_denom, expert_ids,
            sconv_w, hist_ring, out_dst, add_a, add_b, d_pos, start_pos, M, H, inter, topk);
    } else {
        inkling_moe_nvfp4_down_sconv_kernel<8><<<blocks, 128, 0, stream>>>(
            act_in, w2_weight, w2_scale, w2_amax, w2_outer_denom, expert_ids,
            sconv_w, hist_ring, out_dst, add_a, add_b, d_pos, start_pos, M, H, inter, topk);
    }
}

void launch_inkling_moe_compact_nvfp4_w13(
    const __nv_bfloat16* x, const u8* w13_weight, const u8* w13_scale,
    const f32* w13_amax, const f32* w13_outer_denom,
    const i32* expert_ids, const f32* expert_weights, __nv_bfloat16* act_out,
    u32* d_uniq_e, u64* d_uniq_mask, u32* d_uniq_count,
    u32 M, u32 inter, u32 H, u32 topk, f32 gate_cap, f32 up_cap, cudaStream_t stream) {
    u32 total_slots = M * topk;
    inkling_moe_compact_experts_kernel<<<1, 128, 0, stream>>>(
        expert_ids, d_uniq_e, d_uniq_mask, d_uniq_count, total_slots);
    dim3 grid((inter + 3) / 4, total_slots);
    inkling_moe_compact_nvfp4_w13_kernel<<<grid, 128, 0, stream>>>(
        x, w13_weight, w13_scale, w13_amax, w13_outer_denom, expert_weights,
        act_out, d_uniq_e, d_uniq_mask, d_uniq_count, inter, H, topk, gate_cap, up_cap);
}

} // namespace flint::cuda
