#include "kernels.hpp"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace flint::cuda {

namespace {

// Exact BF16x2 PTX lop3.b32 vectorized INT4 -> BF16x2 dot product (identical to gemv_w4a16.cu)
__device__ __forceinline__ void mk_dot_u4x8_bf16x8(u32 packed, uint4 xv, f32& dot_q, f32& sum_x) {
    const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
    const __nv_bfloat162 one_bf2 = __float2bfloat162_rn(1.0f);
    const __nv_bfloat162 neg128  = __float2bfloat162_rn(-128.0f);
    constexpr u32 kMagic = 0x43004300u;
    constexpr u32 kMask  = 0x000F000Fu;

    __nv_bfloat162 dq2 = __float2bfloat162_rn(0.0f);
    __nv_bfloat162 sx2 = __float2bfloat162_rn(0.0f);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        u32 sh = (j == 0) ? packed : (packed >> (j * 4));
        u32 h2;
        asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(h2) : "r"(sh), "n"(kMask), "n"(kMagic));
        __nv_bfloat162 q2 = __hadd2(*reinterpret_cast<const __nv_bfloat162*>(&h2), neg128);
        dq2 = __hfma2(q2, xb[j], dq2);
        sx2 = __hfma2(one_bf2, xb[j], sx2);
    }
    float2 fd = __bfloat1622float2(dq2);
    float2 fs = __bfloat1622float2(sx2);
    dot_q += (fd.x + fd.y);
    sum_x += (fs.x + fs.y);
}

__device__ __forceinline__ f32 mk_dot_fp8x16_bf16x16(uint4 w16, uint4 x0, uint4 x1) {
    const auto* w_fp8 = reinterpret_cast<const __nv_fp8x2_e4m3*>(&w16);
    const auto* xb0   = reinterpret_cast<const __nv_bfloat162*>(&x0);
    const auto* xb1   = reinterpret_cast<const __nv_bfloat162*>(&x1);
    f32 s = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 fw = static_cast<float2>(w_fp8[j]);
        float2 fx = __bfloat1622float2(xb0[j]);
        s = fmaf(fw.x, fx.x, fmaf(fw.y, fx.y, s));
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 fw = static_cast<float2>(w_fp8[4 + j]);
        float2 fx = __bfloat1622float2(xb1[j]);
        s = fmaf(fw.x, fx.x, fmaf(fw.y, fx.y, s));
    }
    return s;
}

__device__ __forceinline__ void mk_rmsnorm_block0(const __nv_bfloat16* __restrict__ in,
                                                  const __nv_bfloat16* __restrict__ w,
                                                  __nv_bfloat16* __restrict__ out,
                                                  u32 H, f32 eps, f32* sm8) {
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31, wid = tid >> 5;
    const auto* in4 = reinterpret_cast<const uint4*>(in);
    const u32 H8 = H >> 3;
    f32 ss = 0.0f;
    for (u32 i = tid; i < H8; i += 256) {
        uint4 v = in4[i];
        const auto* b = reinterpret_cast<const __nv_bfloat162*>(&v);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 f = __bfloat1622float2(b[k]);
            ss = fmaf(f.x, f.x, fmaf(f.y, f.y, ss));
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xFFFFFFFFu, ss, off);
    if (lane == 0) sm8[wid] = ss;
    __syncthreads();
    if (tid == 0) {
        f32 t = 0.0f;
        for (int i = 0; i < 8; ++i) t += sm8[i];
        sm8[0] = rsqrtf(t / static_cast<f32>(H) + eps);
    }
    __syncthreads();
    f32 rstd = sm8[0];
    const auto* w4 = reinterpret_cast<const uint4*>(w);
    auto* o4 = reinterpret_cast<uint4*>(out);
    for (u32 i = tid; i < H8; i += 256) {
        uint4 v = in4[i];
        uint4 wv = __ldg(w4 + i);
        const auto* b = reinterpret_cast<const __nv_bfloat162*>(&v);
        const auto* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
        uint4 ov;
        auto* ob = reinterpret_cast<__nv_bfloat162*>(&ov);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 f = __bfloat1622float2(b[k]);
            float2 fw = __bfloat1622float2(wb[k]);
            ob[k] = __floats2bfloat162_rn(f.x * rstd * fw.x, f.y * rstd * fw.y);
        }
        o4[i] = ov;
    }
}

// Persistent W4A16 GEMV across all active warps (low register pressure -> high GDDR6X MLP)
__device__ __forceinline__ void mk_gemv_w4a16(const __nv_bfloat16* __restrict__ x,
                                              const u32* __restrict__ qweight,
                                              const __nv_bfloat162* __restrict__ scales,
                                              const __nv_bfloat16* __restrict__ bias,
                                              __nv_bfloat16* __restrict__ out,
                                              u32 N, u32 K, u32 gwarp, u32 total_warps, u32 lane,
                                              bool add_to_out) {
    const u32 K8  = K >> 3;
    const u32 K16 = K >> 4;
    const u32 groups_per_row = K >> 7;
    const auto* x4 = reinterpret_cast<const uint4*>(x);

    for (u32 row = gwarp; row < N; row += total_warps) {
        const auto* q2_row = reinterpret_cast<const uint2*>(qweight + static_cast<size_t>(row) * K8);
        const auto* s_row  = scales + static_cast<size_t>(row) * groups_per_row;
        f32 acc = 0.0f;
        for (u32 idx = lane; idx < K16; idx += 32) {
            uint2 qw0 = __ldg(q2_row + idx);
            float2 sm0 = __bfloat1622float2(__ldg(s_row + (idx >> 3)));
            f32 dq0 = 0.0f, sx0 = 0.0f;
            mk_dot_u4x8_bf16x8(qw0.x, x4[idx * 2 + 0], dq0, sx0);
            mk_dot_u4x8_bf16x8(qw0.y, x4[idx * 2 + 1], dq0, sx0);
            acc = fmaf(dq0, sm0.x, fmaf(sx0, sm0.y, acc));
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0) {
            if (bias) acc += __bfloat162float(bias[row]);
            if (add_to_out) acc += __bfloat162float(out[row]);
            out[row] = __float2bfloat16_rn(acc);
        }
    }
}

__device__ __forceinline__ void mk_gate_up_silu_w4a16(const __nv_bfloat16* __restrict__ x,
                                                      const u32* __restrict__ qweight_gu,
                                                      const __nv_bfloat162* __restrict__ scales_gu,
                                                      __nv_bfloat16* __restrict__ out,
                                                      u32 inter, u32 K, u32 gwarp, u32 total_warps, u32 lane) {
    const u32 K8  = K >> 3;
    const u32 K16 = K >> 4;
    const u32 groups_per_row = K >> 7;
    const auto* x4 = reinterpret_cast<const uint4*>(x);

    for (u32 row = gwarp; row < inter; row += total_warps) {
        const auto* qg2 = reinterpret_cast<const uint2*>(qweight_gu + static_cast<size_t>(row) * K8);
        const auto* qu2 = reinterpret_cast<const uint2*>(qweight_gu + static_cast<size_t>(inter + row) * K8);
        const auto* sg  = scales_gu + static_cast<size_t>(row) * groups_per_row;
        const auto* su  = scales_gu + static_cast<size_t>(inter + row) * groups_per_row;
        f32 acc_g = 0.0f, acc_u = 0.0f;
        for (u32 idx = lane; idx < K16; idx += 32) {
            uint2 gq = __ldg(qg2 + idx);
            uint2 uq = __ldg(qu2 + idx);
            float2 smg = __bfloat1622float2(__ldg(sg + (idx >> 3)));
            float2 smu = __bfloat1622float2(__ldg(su + (idx >> 3)));
            uint4 xv0 = x4[idx * 2 + 0];
            uint4 xv1 = x4[idx * 2 + 1];
            f32 dqg = 0.0f, sxg = 0.0f, dqu = 0.0f, sxu = 0.0f;
            mk_dot_u4x8_bf16x8(gq.x, xv0, dqg, sxg);
            mk_dot_u4x8_bf16x8(gq.y, xv1, dqg, sxg);
            mk_dot_u4x8_bf16x8(uq.x, xv0, dqu, sxu);
            mk_dot_u4x8_bf16x8(uq.y, xv1, dqu, sxu);
            acc_g = fmaf(dqg, smg.x, fmaf(sxg, smg.y, acc_g));
            acc_u = fmaf(dqu, smu.x, fmaf(sxu, smu.y, acc_u));
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc_g += __shfl_down_sync(0xFFFFFFFFu, acc_g, off);
            acc_u += __shfl_down_sync(0xFFFFFFFFu, acc_u, off);
        }
        if (lane == 0) {
            f32 silu = acc_g / (1.0f + __expf(-acc_g));
            out[row] = __float2bfloat16_rn(silu * acc_u);
        }
    }
}

// ============================================================================
// LOW-REGISTER (<=64 regs/thread, 256 blocks x 256 threads = 2 blocks/SM = 16 warps/SM)
// SINGLE-LAUNCH PERSISTENT COOPERATIVE MEGAKERNEL (1 Launch = ALL 28 Layers)
//
// 1-Warp-Per-Q-Head GQA Attention (warps 0..6 = the 7 Q heads sharing 1 KV head)
// - Drops attention registers from 48 -> 6 regs/thread (eliminating o_reg[8][4] bloat)
// - Fits 256 blocks (2 blocks/SM x 256 threads = 2,048 active warps) with <=64 regs & ZERO spills
// - Only 2.3 KB shared memory per block
// ============================================================================
template <bool INT8_KV>
__global__ void __launch_bounds__(256, 2) megakernel_w4a16_decode_kernel(
    const MegakernelLayerDesc* __restrict__ layers,
    const __nv_bfloat16* __restrict__ d_embed,
    const __nv_bfloat16* __restrict__ d_final_norm,
    const u8* __restrict__ d_lm_head_fp8,
    const f32* __restrict__ d_lm_head_scales,
    const f32* __restrict__ d_inv_freq,
    const i32* __restrict__ d_block_table,
    __nv_bfloat16* __restrict__ d_x,
    __nv_bfloat16* __restrict__ d_norm,
    __nv_bfloat16* __restrict__ d_qkv,
    __nv_bfloat16* __restrict__ d_q,
    __nv_bfloat16* __restrict__ d_attn_out,
    __nv_bfloat16* __restrict__ d_mlp_act,
    f32* __restrict__ d_split_o,
    f32* __restrict__ d_split_ml,
    f32* __restrict__ d_argmax_val,
    i32* __restrict__ d_argmax_idx,
    i32* __restrict__ d_pos,
    i32* __restrict__ d_token_ids,
    i32* __restrict__ d_out_tokens,
    u32 num_layers, u32 H, u32 inter, u32 vocab_size,
    u32 n_q_heads, u32 n_kv_heads, u32 block_size, f32 rms_eps,
    u32 num_blocks_grid) {
    cg::grid_group grid = cg::this_grid();
    const u32 bid = blockIdx.x;
    const u32 tid = threadIdx.x;      // 0..255
    const u32 wid = tid >> 5;         // 0..7
    const u32 lane = tid & 31;        // 0..31
    const u32 gwarp = bid * 8 + wid;
    const u32 total_warps = num_blocks_grid * 8;
    const u32 num_splits  = num_blocks_grid / n_kv_heads; // e.g. 256/4 = 64 or 512/4 = 128

    const u32 pos = static_cast<u32>(*d_pos);
    const u32 kv_len = pos + 1;
    const u32 q_dim  = n_q_heads * 128;
    const u32 kv_dim = n_kv_heads * 128;
    const u32 qkv_dim = q_dim + 2 * kv_dim;
    const u32 gqa = n_q_heads / n_kv_heads; // 7 for Qwen2.5-7B

    // Tiny shared memory footprint (ONLY 1.1 KB/block -> allows up to 4 blocks/SM = 32 warps/SM)
    __shared__ f32 sm_k[128];
    __shared__ f32 sm_v[128];
    __shared__ f32 sm_a[8];
    __shared__ f32 sm_b[8];

    // Step 0: Embed lookup + Layer 0 input RMSNorm on block 0
    if (bid == 0) {
        i32 tok = d_token_ids[0];
        const auto* e4 = reinterpret_cast<const uint4*>(d_embed + static_cast<size_t>(tok) * H);
        auto* x4 = reinterpret_cast<uint4*>(d_x);
        for (u32 i = tid; i < (H >> 3); i += 256) x4[i] = e4[i];
        __syncthreads();
        mk_rmsnorm_block0(d_x, layers[0].input_norm, d_norm, H, rms_eps, sm_a);
    }
    grid.sync();

    for (u32 l = 0; l < num_layers; ++l) {
        const MegakernelLayerDesc dl = layers[l];

        // 1. Fused QKV W4A16 GEMV
        mk_gemv_w4a16(d_norm, dl.w_qkv, dl.s_qkv, dl.qkv_bias, d_qkv, qkv_dim, H, gwarp, total_warps, lane, false);
        grid.sync();

        // 2. Fused RoPE + Paged KV Write (blocks 0..n_q_heads+n_kv_heads-1)
        if (bid < n_q_heads + n_kv_heads) {
            if (tid < 64) {
                f32 ang = static_cast<f32>(pos) * d_inv_freq[tid];
                f32 s, c;
                __sincosf(ang, &s, &c);
                if (bid < n_q_heads) {
                    __nv_bfloat16* qh = d_qkv + bid * 128;
                    __nv_bfloat16* qo = d_q + bid * 128;
                    f32 x0 = __bfloat162float(qh[tid]);
                    f32 x1 = __bfloat162float(qh[tid + 64]);
                    qo[tid]      = __float2bfloat16_rn(x0 * c - x1 * s);
                    qo[tid + 64] = __float2bfloat16_rn(x1 * c + x0 * s);
                } else {
                    u32 kvh = bid - n_q_heads;
                    __nv_bfloat16* kh = d_qkv + q_dim + kvh * 128;
                    __nv_bfloat16* vh = d_qkv + q_dim + kv_dim + kvh * 128;
                    f32 x0 = __bfloat162float(kh[tid]);
                    f32 x1 = __bfloat162float(kh[tid + 64]);
                    f32 rk0 = x0 * c - x1 * s;
                    f32 rk1 = x1 * c + x0 * s;
                    if constexpr (INT8_KV) {
                        f32 v0 = __bfloat162float(vh[tid]), v1 = __bfloat162float(vh[tid + 64]);
                        f32 ka = fmaxf(fabsf(rk0), fabsf(rk1)), va = fmaxf(fabsf(v0), fabsf(v1));
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1) {
                            ka = fmaxf(ka, __shfl_xor_sync(0xFFFFFFFFu, ka, off));
                            va = fmaxf(va, __shfl_xor_sync(0xFFFFFFFFu, va, off));
                        }
                        if (lane == 0) { sm_a[wid] = ka; sm_b[wid] = va; }
                        sm_k[tid] = rk0; sm_k[tid + 64] = rk1;
                        sm_v[tid] = v0;  sm_v[tid + 64] = v1;
                    } else {
                        u32 blk = pos / block_size, slot = pos - blk * block_size;
                        i32 phys = d_block_table[blk];
                        size_t kv_off = ((static_cast<size_t>(phys) * n_kv_heads + kvh) * block_size + slot) * 128;
                        auto* kc = static_cast<__nv_bfloat16*>(dl.k_cache) + kv_off;
                        auto* vc = static_cast<__nv_bfloat16*>(dl.v_cache) + kv_off;
                        kc[tid] = __float2bfloat16_rn(rk0); kc[tid + 64] = __float2bfloat16_rn(rk1);
                        vc[tid] = vh[tid]; vc[tid + 64] = vh[tid + 64];
                    }
                }
            }
            if constexpr (INT8_KV) {
                if (bid >= n_q_heads) {
                    __syncthreads();
                    if (tid < 64) {
                        u32 kvh = bid - n_q_heads;
                        u32 blk = pos / block_size, slot = pos - blk * block_size;
                        i32 phys = d_block_table[blk];
                        size_t scale_off = (static_cast<size_t>(phys) * n_kv_heads + kvh) * block_size + slot;
                        size_t kv_off = scale_off * 128;
                        f32 k_sc = fmaxf(fmaxf(sm_a[0], sm_a[1]) / 127.0f, 1e-6f);
                        f32 v_sc = fmaxf(fmaxf(sm_b[0], sm_b[1]) / 127.0f, 1e-6f);
                        if (tid == 0) { dl.k_scale[scale_off] = k_sc; dl.v_scale[scale_off] = v_sc; }
                        auto q8 = [](f32 x) -> i8 { return static_cast<i8>(max(-127, min(127, __float2int_rn(x)))); };
                        auto* kc = static_cast<i8*>(dl.k_cache) + kv_off;
                        auto* vc = static_cast<i8*>(dl.v_cache) + kv_off;
                        kc[tid]      = q8(sm_k[tid]      / k_sc);
                        kc[tid + 64] = q8(sm_k[tid + 64] / k_sc);
                        vc[tid]      = q8(sm_v[tid]      / v_sc);
                        vc[tid + 64] = q8(sm_v[tid + 64] / v_sc);
                    }
                }
            }
        }
        grid.sync();

        // 3. Low-Register 1-Warp-Per-Q-Head GQA Split-K FlashDecoding
        // Each warp wid in 0..gqa-1 holds ONLY 1 Q-head (6 regs instead of 48 regs),
        // sharing K & V once per token through sm_k[128] / sm_v[128] (7x KV reuse).
        {
            const u32 kvh = bid / num_splits;
            const u32 sp  = bid - kvh * num_splits;
            const u32 qh_base = kvh * gqa;
            const u32 chunk   = (kv_len + num_splits - 1) / num_splits;
            const u32 t_start = sp * chunk;
            const u32 t_end   = min(kv_len, t_start + chunk);

            if (t_start >= t_end) {
                if (wid < gqa && lane == 0) {
                    size_t ml = ((qh_base + wid) * num_splits + sp) * 2;
                    d_split_ml[ml + 0] = -1e30f;
                    d_split_ml[ml + 1] = 0.0f;
                }
            } else {
                const f32 sc = rsqrtf(128.0f);
                f32 q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
                if (wid < gqa) {
                    const __nv_bfloat16* qp = d_q + (qh_base + wid) * 128 + lane * 4;
                    q0 = __bfloat162float(qp[0]) * sc;
                    q1 = __bfloat162float(qp[1]) * sc;
                    q2 = __bfloat162float(qp[2]) * sc;
                    q3 = __bfloat162float(qp[3]) * sc;
                }
                f32 m_i = -1e30f, l_i = 0.0f;
                f32 o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

                for (u32 t = t_start; t < t_end; ++t) {
                    // Load K & V once across threads 0..127 into sm_k[128] & sm_v[128]
                    if (tid < 128) {
                        u32 blk = t / block_size, slot = t - blk * block_size;
                        i32 phys = d_block_table[blk];
                        size_t scale_off = (static_cast<size_t>(phys) * n_kv_heads + kvh) * block_size + slot;
                        size_t kv_off = scale_off * 128 + tid;
                        if constexpr (INT8_KV) {
                            f32 k_sc = __ldg(dl.k_scale + scale_off);
                            f32 v_sc = __ldg(dl.v_scale + scale_off);
                            sm_k[tid] = static_cast<f32>(__ldg(static_cast<const i8*>(dl.k_cache) + kv_off)) * k_sc;
                            sm_v[tid] = static_cast<f32>(__ldg(static_cast<const i8*>(dl.v_cache) + kv_off)) * v_sc;
                        } else {
                            sm_k[tid] = __bfloat162float(__ldg(static_cast<const __nv_bfloat16*>(dl.k_cache) + kv_off));
                            sm_v[tid] = __bfloat162float(__ldg(static_cast<const __nv_bfloat16*>(dl.v_cache) + kv_off));
                        }
                    }
                    __syncthreads();
                    if (wid < gqa) {
                        f32 dot = fmaf(q0, sm_k[lane * 4 + 0],
                                  fmaf(q1, sm_k[lane * 4 + 1],
                                  fmaf(q2, sm_k[lane * 4 + 2], q3 * sm_k[lane * 4 + 3])));
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
                        f32 nm = fmaxf(m_i, dot);
                        f32 ep = __expf(m_i - nm), ec = __expf(dot - nm);
                        l_i = l_i * ep + ec;
                        o0 = fmaf(ec, sm_v[lane * 4 + 0], o0 * ep);
                        o1 = fmaf(ec, sm_v[lane * 4 + 1], o1 * ep);
                        o2 = fmaf(ec, sm_v[lane * 4 + 2], o2 * ep);
                        o3 = fmaf(ec, sm_v[lane * 4 + 3], o3 * ep);
                        m_i = nm;
                    }
                    __syncthreads();
                }

                if (wid < gqa) {
                    u32 qh = qh_base + wid;
                    size_t ml_b = (qh * num_splits + sp) * 2;
                    size_t o_b  = (qh * num_splits + sp) * 128 + lane * 4;
                    d_split_o[o_b + 0] = o0; d_split_o[o_b + 1] = o1;
                    d_split_o[o_b + 2] = o2; d_split_o[o_b + 3] = o3;
                    if (lane == 0) { d_split_ml[ml_b + 0] = m_i; d_split_ml[ml_b + 1] = l_i; }
                }
            }
        }
        grid.sync();

        // 4. Split-K Reduce (blocks 0..n_q_heads-1, first 128 threads)
        if (bid < n_q_heads && tid < 128) {
            size_t ml_h = bid * num_splits * 2;
            size_t o_h  = bid * num_splits * 128;
            f32 gm = -1e30f;
            for (u32 sp = 0; sp < num_splits; ++sp) {
                if (d_split_ml[ml_h + sp * 2 + 1] > 0.0f) gm = fmaxf(gm, d_split_ml[ml_h + sp * 2 + 0]);
            }
            f32 gl = 0.0f, acc = 0.0f;
            for (u32 sp = 0; sp < num_splits; ++sp) {
                f32 sl = d_split_ml[ml_h + sp * 2 + 1];
                if (sl > 0.0f) {
                    f32 w = __expf(d_split_ml[ml_h + sp * 2 + 0] - gm);
                    gl += sl * w;
                    acc = fmaf(d_split_o[o_h + sp * 128 + tid], w, acc);
                }
            }
            d_attn_out[bid * 128 + tid] = __float2bfloat16_rn((gl > 0.0f) ? (acc / gl) : 0.0f);
        }
        grid.sync();

        // 5. O-Proj W4A16 GEMV + Fused += d_x Residual Add
        mk_gemv_w4a16(d_attn_out, dl.w_o, dl.s_o, nullptr, d_x, H, q_dim, gwarp, total_warps, lane, true);
        grid.sync();

        // 6. Post-Attn RMSNorm (block 0)
        if (bid == 0) mk_rmsnorm_block0(d_x, dl.post_attn_norm, d_norm, H, rms_eps, sm_a);
        grid.sync();

        // 7. Fused Gate+Up+SiLU W4A16 GEMV
        mk_gate_up_silu_w4a16(d_norm, dl.w_gu, dl.s_gu, d_mlp_act, inter, H, gwarp, total_warps, lane);
        grid.sync();

        // 8. Down-Proj W4A16 GEMV + Fused += d_x Residual Add
        mk_gemv_w4a16(d_mlp_act, dl.w_down, dl.s_down, nullptr, d_x, H, inter, gwarp, total_warps, lane, true);
        grid.sync();

        // 9. Next layer input RMSNorm (or final norm after last layer) on block 0
        if (bid == 0) {
            const __nv_bfloat16* nw = (l + 1 < num_layers) ? layers[l + 1].input_norm : d_final_norm;
            mk_rmsnorm_block0(d_x, nw, d_norm, H, rms_eps, sm_a);
        }
        grid.sync();
    }

    // Step 10: Fused FP8 LM Head + In-Register Argmax (ZERO global logit traffic)
    {
        const u32 K16 = H >> 4;
        const auto* x8 = reinterpret_cast<const uint4*>(d_norm);
        f32 best_v = -1e30f;
        i32 best_i = 0;
        for (u32 row = gwarp; row < vocab_size; row += total_warps) {
            const auto* w16 = reinterpret_cast<const uint4*>(d_lm_head_fp8 + static_cast<size_t>(row) * H);
            f32 acc = 0.0f;
            for (u32 idx = lane; idx < K16; idx += 32) {
                acc += mk_dot_fp8x16_bf16x16(__ldg(w16 + idx), x8[idx * 2], x8[idx * 2 + 1]);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
            if (lane == 0) {
                f32 v = acc * __ldg(d_lm_head_scales + row);
                if (v > best_v) { best_v = v; best_i = static_cast<i32>(row); }
            }
        }
        if (lane == 0) { sm_a[wid] = best_v; reinterpret_cast<i32*>(sm_b)[wid] = best_i; }
        __syncthreads();
        if (tid == 0) {
            f32 bv = sm_a[0]; i32 bi = reinterpret_cast<i32*>(sm_b)[0];
            for (int w = 1; w < 8; ++w) {
                if (sm_a[w] > bv) { bv = sm_a[w]; bi = reinterpret_cast<i32*>(sm_b)[w]; }
            }
            d_argmax_val[bid] = bv;
            d_argmax_idx[bid] = bi;
        }
    }
    grid.sync();

    // Step 11: Final Argmax across all grid blocks + Advance (d_pos, d_token_ids)
    if (bid == 0) {
        f32 v = -1e30f; i32 idx = 0;
        for (u32 b = tid; b < num_blocks_grid; b += 256) {
            f32 cv = d_argmax_val[b];
            if (cv > v) { v = cv; idx = d_argmax_idx[b]; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            f32 ov = __shfl_down_sync(0xFFFFFFFFu, v, off);
            i32 oi = __shfl_down_sync(0xFFFFFFFFu, idx, off);
            if (ov > v) { v = ov; idx = oi; }
        }
        if (lane == 0) { sm_a[wid] = v; reinterpret_cast<i32*>(sm_b)[wid] = idx; }
        __syncthreads();
        if (tid == 0) {
            f32 bv = sm_a[0]; i32 bi = reinterpret_cast<i32*>(sm_b)[0];
            for (int w = 1; w < 8; ++w) {
                if (sm_a[w] > bv) { bv = sm_a[w]; bi = reinterpret_cast<i32*>(sm_b)[w]; }
            }
            d_out_tokens[0] = bi;
            d_token_ids[0]  = bi;
            d_pos[0]        = static_cast<i32>(pos + 1);
        }
    }
}

} // namespace

bool launch_megakernel_w4a16_decode(const MegakernelLayerDesc* d_layers,
                                    const __nv_bfloat16* d_embed,
                                    const __nv_bfloat16* d_final_norm,
                                    const u8* d_lm_head_fp8,
                                    const f32* d_lm_head_scales,
                                    const f32* d_inv_freq,
                                    const i32* d_block_table,
                                    __nv_bfloat16* d_x,
                                    __nv_bfloat16* d_norm,
                                    __nv_bfloat16* d_qkv,
                                    __nv_bfloat16* d_q,
                                    __nv_bfloat16* d_attn_out,
                                    __nv_bfloat16* d_mlp_act,
                                    f32* d_split_o,
                                    f32* d_split_ml,
                                    f32* d_argmax_val,
                                    i32* d_argmax_idx,
                                    i32* d_pos,
                                    i32* d_token_ids,
                                    i32* d_out_tokens,
                                    u32 num_layers, u32 H, u32 inter, u32 vocab_size,
                                    u32 n_q_heads, u32 n_kv_heads, u32 block_size, f32 rms_eps,
                                    bool int8_kv, cudaStream_t stream) {
    // Query exact max active blocks/SM so we saturate 100% of RTX 4090's SMs with 4 blocks/SM = 32 warps/SM
    int max_bpsm = 1;
    int num_sms = 128;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    const void* fn = int8_kv
        ? reinterpret_cast<const void*>(megakernel_w4a16_decode_kernel<true>)
        : reinterpret_cast<const void*>(megakernel_w4a16_decode_kernel<false>);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_bpsm, fn, 256, 0);
    // Cap splits at 128 (d_split_o size) -> max blocks = min(num_sms * max_bpsm, 128 * n_kv_heads)
    u32 num_blocks_grid = std::min<u32>(static_cast<u32>(num_sms * std::max(1, max_bpsm)), 128u * n_kv_heads);
    // Ensure multiple of n_kv_heads
    num_blocks_grid = (num_blocks_grid / n_kv_heads) * n_kv_heads;
    if (num_blocks_grid < n_kv_heads) num_blocks_grid = n_kv_heads;

    dim3 grid(num_blocks_grid);
    dim3 block(256);
    void* args[] = {
        &d_layers, &d_embed, &d_final_norm, &d_lm_head_fp8, &d_lm_head_scales,
        &d_inv_freq, &d_block_table, &d_x, &d_norm, &d_qkv, &d_q, &d_attn_out, &d_mlp_act,
        &d_split_o, &d_split_ml, &d_argmax_val, &d_argmax_idx,
        &d_pos, &d_token_ids, &d_out_tokens,
        &num_layers, &H, &inter, &vocab_size, &n_q_heads, &n_kv_heads, &block_size, &rms_eps,
        &num_blocks_grid
    };
    return cudaLaunchCooperativeKernel(fn, grid, block, args, 0, stream) == cudaSuccess;
}

} // namespace flint::cuda
