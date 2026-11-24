#include "inkling2_common.cuh"

namespace flint::cuda {

namespace {

// 9a. 1-Warp (32-thread, 0 SMEM) Global Dynamic Short-Conv + L2-Clip/RMSNorm + WHT128 + 4-bit/8-bit KV Write
// Grid: (n_kv_heads=8, 2 [0=K, 1=V]), Block: 32 threads (4 dims per lane = 128 dims, 100% coalesced)
template <bool K_IS_4BIT>
__global__ void inkling_kv_sconv_write_wht_kernel(
    const __nv_bfloat16* __restrict__ k_raw,
    const __nv_bfloat16* __restrict__ v_raw,
    const __nv_bfloat16* __restrict__ wk,     // [4, kv_dim] dynamic static_w
    const __nv_bfloat16* __restrict__ wv,
    const __nv_bfloat16* __restrict__ dk_dyn, // [M, n_kv_heads, 4]
    const __nv_bfloat16* __restrict__ dv_dyn,
    __nv_bfloat16* __restrict__ k_hist,       // [16, kv_dim]
    __nv_bfloat16* __restrict__ v_hist,
    u8* __restrict__ k_cache,
    u8* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ k_scales_bf16,
    __nv_bfloat16* __restrict__ v_scales_bf16,
    const i32* __restrict__ d_block_table,
    const i32* __restrict__ d_pos,
    u32 start_pos, u32 M, u32 n_kv_heads, u32 block_size,
    f32 k_clip, f32 rms_eps) {
    const u32 kvh  = blockIdx.x;
    const bool is_v = (blockIdx.y == 1);
    const int lane = static_cast<int>(threadIdx.x); // 0..31
    const u32 kv_dim = n_kv_heads * 128u;
    const u32 c0 = kvh * 128u + static_cast<u32>(lane) * 4u;

    const __nv_bfloat16* x_src = is_v ? v_raw  : k_raw;
    const __nv_bfloat16* w_src = is_v ? wv     : wk;
    const __nv_bfloat16* dyn   = is_v ? dv_dyn : dk_dyn;
    __nv_bfloat16* hist        = is_v ? v_hist : k_hist;
    u8* cache_dst              = is_v ? v_cache : k_cache;
    __nv_bfloat16* scales_dst  = is_v ? v_scales_bf16 : k_scales_bf16;

    const u32 p0 = d_pos ? static_cast<u32>(*d_pos) : start_pos;
    const f32 conv_scale = rsqrtf(9.0f);

    f32 sw0[4], sw1[4], sw2[4], sw3[4];
    f32 hm1[4], hm2[4], hm3[4];
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        u32 c = c0 + e;
        sw0[e] = __bfloat162float(w_src[0 * kv_dim + c]);
        sw1[e] = __bfloat162float(w_src[1 * kv_dim + c]);
        sw2[e] = __bfloat162float(w_src[2 * kv_dim + c]);
        sw3[e] = __bfloat162float(w_src[3 * kv_dim + c]);
        hm3[e] = (p0 >= 3u) ? __bfloat162float(hist[((p0 - 3u) & 15u) * kv_dim + c]) : 0.0f;
        hm2[e] = (p0 >= 2u) ? __bfloat162float(hist[((p0 - 2u) & 15u) * kv_dim + c]) : 0.0f;
        hm1[e] = (p0 >= 1u) ? __bfloat162float(hist[((p0 - 1u) & 15u) * kv_dim + c]) : 0.0f;
    }

    for (u32 m = 0; m < M; ++m) {
        const u32 pos = p0 + m;
        const size_t doff = (static_cast<size_t>(m) * n_kv_heads + kvh) * 4ULL;
        const f32 d0 = __bfloat162float(dyn[doff + 0]);
        const f32 d1 = __bfloat162float(dyn[doff + 1]);
        const f32 d2 = __bfloat162float(dyn[doff + 2]);
        const f32 d3 = __bfloat162float(dyn[doff + 3]);

        f32 y[4];
        f32 ss = 0.0f;
        const size_t x_base = static_cast<size_t>(m) * kv_dim + c0;
        const size_t h_base = static_cast<size_t>(pos & 15u) * kv_dim + c0;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            __nv_bfloat16 cur_bf = x_src[x_base + e];
            f32 cur = __bfloat162float(cur_bf);
            f32 val = ((sw0[e] + d0 + 1.0f) * cur + (sw1[e] + d1) * hm1[e] +
                       (sw2[e] + d2) * hm2[e] + (sw3[e] + d3) * hm3[e]) * conv_scale;
            val = __bfloat162float(__float2bfloat16_rn(val));
            y[e] = val;
            ss = fmaf(val, val, ss);
            hist[h_base + e] = cur_bf;
            hm3[e] = hm2[e]; hm2[e] = hm1[e]; hm1[e] = cur;
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
        f32 norm_mul = is_v ? rsqrtf(ss / 128.0f + rms_eps) : fminf(k_clip / (sqrtf(ss) + 1e-6f), 1.0f);
        y[0] *= norm_mul; y[1] *= norm_mul; y[2] *= norm_mul; y[3] *= norm_mul;

        const u32 blk  = pos / block_size;
        const u32 slot = pos - blk * block_size;
        const i32 phys = d_block_table[blk];
        const size_t scale_off = (static_cast<size_t>(phys) * n_kv_heads + kvh) * block_size + slot;

        if (is_v || K_IS_4BIT) {
            ink_wht128_warp(y[0], y[1], y[2], y[3], lane);
            f32 am = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, off));
            // Exact amax on WHT-K4V4 (cos=0.9878, top1=exact); Lloyd-Max 2.65*rms on WHT-K8V4 V (cos=0.9926)
            f32 rms_y = sqrtf(ss / 128.0f) * norm_mul;
            f32 clip_am = (!K_IS_4BIT && is_v) ? fminf(am, rms_y * 2.65f) : am;
            f32 sc = fmaxf(clip_am / 7.0f, 1e-6f);
            __nv_bfloat16 sc_bf = __float2bfloat16_rn(sc);
            if (lane == 0) scales_dst[scale_off] = sc_bf;
            f32 inv_sc = 1.0f / __bfloat162float(sc_bf);
            int q0 = max(-7, min(7, __float2int_rn(y[0] * inv_sc)));
            int q1 = max(-7, min(7, __float2int_rn(y[1] * inv_sc)));
            int q2 = max(-7, min(7, __float2int_rn(y[2] * inv_sc)));
            int q3 = max(-7, min(7, __float2int_rn(y[3] * inv_sc)));
            u8 b0 = static_cast<u8>(((q0 + 8) & 0xF) | (((q1 + 8) & 0xF) << 4));
            u8 b1 = static_cast<u8>(((q2 + 8) & 0xF) | (((q3 + 8) & 0xF) << 4));
            u16 packed = static_cast<u16>(b0) | (static_cast<u16>(b1) << 8);
            reinterpret_cast<u16*>(cache_dst + scale_off * 64ULL)[lane] = packed;
        } else {
            f32 am = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, off));
            f32 sc = fmaxf(am / 127.0f, 1e-6f);
            __nv_bfloat16 sc_bf = __float2bfloat16_rn(sc);
            if (lane == 0) scales_dst[scale_off] = sc_bf;
            f32 inv_sc = 1.0f / __bfloat162float(sc_bf);
            int q0 = max(-127, min(127, __float2int_rn(y[0] * inv_sc)));
            int q1 = max(-127, min(127, __float2int_rn(y[1] * inv_sc)));
            int q2 = max(-127, min(127, __float2int_rn(y[2] * inv_sc)));
            int q3 = max(-127, min(127, __float2int_rn(y[3] * inv_sc)));
            int packed = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
            reinterpret_cast<int*>(cache_dst + scale_off * 128ULL)[lane] = packed;
        }
    }
}

// Fast PTX lop3.b32 unpack of 2 offset-binary 4-bit nibbles (lo in bits 0..3, hi in bits 4..7, bias=8)
// directly into BF16x2 scaled by sc: out = (nibble - 8) * sc = (128 + nibble) * sc + (-136 * sc)
__device__ __forceinline__ __nv_bfloat162 unpack_2nib_bf162(u32 byte_val, __nv_bfloat162 sc2, __nv_bfloat162 bias2) {
    u32 packed = (byte_val & 0xFu) | ((byte_val & 0xF0u) << 12);
    u32 bf_bits;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(bf_bits) : "r"(packed), "r"(0x000F000Fu), "r"(0x43004300u));
    return __hfma2(*reinterpret_cast<const __nv_bfloat162*>(&bf_bits), sc2, bias2);
}

// 9b. Block-Contiguous 256-Split Global Decode Kernel for WHT-K4V4 & WHT-K8V4 (64-bit offsets, 0 spills)
// Grid: (n_kv_heads=8, num_splits, M), Block: 128 threads (4 warps; GQA=3 Q heads share each KV block)
template <bool K_IS_4BIT>
__global__ void inkling_global_attn_splitk_wht_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ r,
    const __nv_bfloat16* __restrict__ rel_proj_T,
    const u8* __restrict__ k_cache,
    const u8* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ k_scales_bf16,
    const __nv_bfloat16* __restrict__ v_scales_bf16,
    const i32* __restrict__ d_block_table,
    f32* __restrict__ d_split_o,
    f32* __restrict__ d_split_ml,
    const i32* __restrict__ d_pos,
    u32 start_pos, u32 n_q_heads, u32 n_kv_heads, u32 block_size,
    u32 num_splits) {
    constexpr int GQA = 3;
    const u32 kvh  = blockIdx.x;
    const u32 sp   = blockIdx.y;
    const u32 m    = blockIdx.z;
    const u32 tid  = threadIdx.x; // 0..127
    const u32 wid  = tid >> 5;
    const int lane = static_cast<int>(tid & 31u);
    const u32 qh_base = kvh * GQA;

    const u32 pos = (d_pos ? static_cast<u32>(*d_pos) : start_pos) + m;
    const u32 kv_end = pos + 1u;
    const u32 total_blks = (kv_end + block_size - 1u) / block_size;
    const u32 blks_per_sp = (total_blks + num_splits - 1u) / num_splits;
    const u32 b_begin = sp * blks_per_sp;
    const u32 b_end   = min(total_blks, b_begin + blks_per_sp);

    if (b_begin >= b_end) {
        if (tid < GQA) {
            u32 qh = qh_base + tid;
            size_t ml_base = ((static_cast<size_t>(m) * n_q_heads + qh) * num_splits + sp) * 2ULL;
            d_split_ml[ml_base + 0] = -1e30f;
            d_split_ml[ml_base + 1] = 0.0f;
        }
        return;
    }

    const f32 qk_scale = rsqrtf(128.0f);
    __shared__ f32 s_Q[GQA][128];
    __shared__ int s_Q_i8[GQA][32];
    __shared__ f32 s_Q_sc[GQA];
    __shared__ f32 s_R[GQA][16];

    if (wid < GQA) {
        const u32 qh = qh_base + wid;
        const __nv_bfloat16* qp = q + (static_cast<size_t>(m) * n_q_heads + qh) * 128ULL;
        f32 q0 = __bfloat162float(qp[lane * 4 + 0]) * qk_scale;
        f32 q1 = __bfloat162float(qp[lane * 4 + 1]) * qk_scale;
        f32 q2 = __bfloat162float(qp[lane * 4 + 2]) * qk_scale;
        f32 q3 = __bfloat162float(qp[lane * 4 + 3]) * qk_scale;
        if constexpr (K_IS_4BIT) {
            ink_wht128_warp(q0, q1, q2, q3, lane);
            s_Q[wid][lane * 4 + 0] = q0;
            s_Q[wid][lane * 4 + 1] = q1;
            s_Q[wid][lane * 4 + 2] = q2;
            s_Q[wid][lane * 4 + 3] = q3;
        } else {
            f32 am = fmaxf(fmaxf(fabsf(q0), fabsf(q1)), fmaxf(fabsf(q2), fabsf(q3)));
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, off));
            f32 qsc = fmaxf(am / 127.0f, 1e-8f);
            f32 inv_q = 1.0f / qsc;
            int i0 = max(-127, min(127, __float2int_rn(q0 * inv_q)));
            int i1 = max(-127, min(127, __float2int_rn(q1 * inv_q)));
            int i2 = max(-127, min(127, __float2int_rn(q2 * inv_q)));
            int i3 = max(-127, min(127, __float2int_rn(q3 * inv_q)));
            s_Q_i8[wid][lane] = (i0 & 0xFF) | ((i1 & 0xFF) << 8) | ((i2 & 0xFF) << 16) | ((i3 & 0xFF) << 24);
            if (lane == 0) s_Q_sc[wid] = qsc;
        }
        if (lane < 16) {
            s_R[wid][lane] = __bfloat162float(r[(static_cast<size_t>(m) * n_q_heads + qh) * 16ULL + lane]);
        }
    }
    __syncthreads();

    // Load Q into registers for zero SMEM reads in the inner loop
    f32 q_reg[GQA][4];
    int q_i8_reg[GQA];
    f32 q_sc_reg[GQA];
    #pragma unroll
    for (int g = 0; g < GQA; ++g) {
        if constexpr (K_IS_4BIT) {
            q_reg[g][0] = s_Q[g][lane * 4 + 0];
            q_reg[g][1] = s_Q[g][lane * 4 + 1];
            q_reg[g][2] = s_Q[g][lane * 4 + 2];
            q_reg[g][3] = s_Q[g][lane * 4 + 3];
        } else {
            q_i8_reg[g] = s_Q_i8[g][lane];
            q_sc_reg[g] = s_Q_sc[g];
        }
    }

    f32 m_i[GQA] = {-1e30f, -1e30f, -1e30f};
    f32 l_i[GQA] = {0.0f, 0.0f, 0.0f};
    f32 o_reg[GQA][4] = {};

    __shared__ u16 sm_k_blk[32][32]; // 2 KB: 2 blocks (32 tokens) x 64 B loaded via 128-bit LDG.E.128 uint4
    __shared__ u16 sm_v_blk[32][32]; // 2 KB
    __shared__ int sm_k8_blk[16][32];// 2 KB (used only when !K_IS_4BIT)
    __shared__ __nv_bfloat16 sm_ksc[32];
    __shared__ __nv_bfloat16 sm_vsc[32];

    if constexpr (K_IS_4BIT) {
        // 2-block-at-once 128-bit LDG.E.128 uint4 bulk streaming (warps 0-1 -> blk0, warps 2-3 -> blk1)
        const u32 half_id = tid >> 6;       // 0 or 1
        const u32 tid64   = tid & 63u;      // 0..63 (64 threads x 16 B = 1,024 B = 1 full 16-token block)
        for (u32 blk = b_begin; blk < b_end; blk += 2u) {
            const bool has_b1 = (blk + 1u < b_end);
            const i32 phys0 = __ldg(d_block_table + blk);
            const i32 phys1 = has_b1 ? __ldg(d_block_table + blk + 1u) : phys0;
            const size_t base0 = (static_cast<size_t>(phys0) * n_kv_heads + kvh) * block_size;
            const size_t base1 = (static_cast<size_t>(phys1) * n_kv_heads + kvh) * block_size;
            const size_t my_base = (half_id == 0u) ? base0 : base1;

            if (tid64 < 16u) {
                sm_ksc[half_id * 16u + tid64] = __ldg(k_scales_bf16 + my_base + tid64);
                sm_vsc[half_id * 16u + tid64] = __ldg(v_scales_bf16 + my_base + tid64);
            }
            reinterpret_cast<uint4*>(sm_v_blk)[tid] = __ldg(reinterpret_cast<const uint4*>(v_cache + my_base * 64ULL) + tid64);
            reinterpret_cast<uint4*>(sm_k_blk)[tid] = __ldg(reinterpret_cast<const uint4*>(k_cache + my_base * 64ULL) + tid64);
            __syncthreads();

            const u32 n_slots = has_b1 ? 32u : 16u;
            const u32 t_base  = blk * block_size;
            const bool pair_has_rel = (pos - t_base < 1024u + 32u);

            #pragma unroll 4
            for (u32 slot = wid; slot < n_slots; slot += 4u) {
                const u32 t = t_base + slot;
                if (t >= kv_end) break;
                const f32 k_sc = __bfloat162float(sm_ksc[slot]);
                const f32 v_sc = __bfloat162float(sm_vsc[slot]);

                const u16 vp = sm_v_blk[slot][lane];
                const u16 kp = sm_k_blk[slot][lane];
                const f32 v0 = static_cast<f32>(static_cast<int>(vp & 0xFu) - 8);
                const f32 v1 = static_cast<f32>(static_cast<int>((vp >> 4) & 0xFu) - 8);
                const f32 v2 = static_cast<f32>(static_cast<int>((vp >> 8) & 0xFu) - 8);
                const f32 v3 = static_cast<f32>(static_cast<int>((vp >> 12) & 0xFu) - 8);
                const f32 k0 = static_cast<f32>(static_cast<int>(kp & 0xFu) - 8);
                const f32 k1 = static_cast<f32>(static_cast<int>((kp >> 4) & 0xFu) - 8);
                const f32 k2 = static_cast<f32>(static_cast<int>((kp >> 8) & 0xFu) - 8);
                const f32 k3 = static_cast<f32>(static_cast<int>((kp >> 12) & 0xFu) - 8);

                f32 rp = 0.0f;
                const u32 dist = pos - t;
                const bool has_rel = pair_has_rel && (dist < 1024u);
                if (has_rel && lane < 16) {
                    rp = __bfloat162float(__ldg(rel_proj_T + static_cast<size_t>(dist) * 16ULL + lane));
                }

                #pragma unroll
                for (int g = 0; g < GQA; ++g) {
                    f32 d_part = fmaf(q_reg[g][0], k0, fmaf(q_reg[g][1], k1, fmaf(q_reg[g][2], k2, q_reg[g][3] * k3)));
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) d_part += __shfl_xor_sync(0xFFFFFFFFu, d_part, off);
                    f32 dot = d_part * k_sc;
                    if (has_rel) {
                        f32 r_part = (lane < 16) ? (s_R[g][lane] * rp) : 0.0f;
                        #pragma unroll
                        for (int off = 8; off > 0; off >>= 1) r_part += __shfl_xor_sync(0xFFFFFFFFu, r_part, off);
                        dot += __shfl_sync(0xFFFFFFFFu, r_part, 0);
                    }
                    f32 new_m = fmaxf(m_i[g], dot);
                    f32 ep = __expf(m_i[g] - new_m);
                    f32 ec = __expf(dot - new_m);
                    l_i[g] = l_i[g] * ep + ec;
                    f32 w_v = ec * v_sc;
                    o_reg[g][0] = fmaf(w_v, v0, o_reg[g][0] * ep);
                    o_reg[g][1] = fmaf(w_v, v1, o_reg[g][1] * ep);
                    o_reg[g][2] = fmaf(w_v, v2, o_reg[g][2] * ep);
                    o_reg[g][3] = fmaf(w_v, v3, o_reg[g][3] * ep);
                    m_i[g] = new_m;
                }
            }
            __syncthreads();
        }
    } else {
        for (u32 blk = b_begin; blk < b_end; ++blk) {
            const i32 phys = __ldg(d_block_table + blk);
            const size_t blk_scale_base = (static_cast<size_t>(phys) * n_kv_heads + kvh) * block_size;
            const u32 t_base = blk * block_size;
            const bool block_has_rel = (pos - t_base < 1024u + block_size);

            if (tid < 16u) {
                sm_ksc[tid] = __ldg(k_scales_bf16 + blk_scale_base + tid);
                sm_vsc[tid] = __ldg(v_scales_bf16 + blk_scale_base + tid);
            }
            reinterpret_cast<uint2*>(sm_v_blk)[tid]  = __ldg(reinterpret_cast<const uint2*>(v_cache + blk_scale_base * 64ULL) + tid);
            reinterpret_cast<uint4*>(sm_k8_blk)[tid] = __ldg(reinterpret_cast<const uint4*>(k_cache + blk_scale_base * 128ULL) + tid);
            __syncthreads();

            #pragma unroll
            for (u32 slot = wid; slot < 16u; slot += 4u) {
                const u32 t = t_base + slot;
                if (t >= kv_end) break;
                const f32 k_sc = __bfloat162float(sm_ksc[slot]);
                const f32 v_sc = __bfloat162float(sm_vsc[slot]);

                const u16 vp = sm_v_blk[slot][lane];
                const f32 v0 = static_cast<f32>(static_cast<int>(vp & 0xFu) - 8);
                const f32 v1 = static_cast<f32>(static_cast<int>((vp >> 4) & 0xFu) - 8);
                const f32 v2 = static_cast<f32>(static_cast<int>((vp >> 8) & 0xFu) - 8);
                const f32 v3 = static_cast<f32>(static_cast<int>((vp >> 12) & 0xFu) - 8);
                const int kw32 = sm_k8_blk[slot][lane];

                f32 rp = 0.0f;
                const u32 dist = pos - t;
                const bool has_rel = block_has_rel && (dist < 1024u);
                if (has_rel && lane < 16) {
                    rp = __bfloat162float(__ldg(rel_proj_T + static_cast<size_t>(dist) * 16ULL + lane));
                }

                #pragma unroll
                for (int g = 0; g < GQA; ++g) {
                    int idot = __dp4a(q_i8_reg[g], kw32, 0);
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) idot += __shfl_xor_sync(0xFFFFFFFFu, idot, off);
                    f32 dot = static_cast<f32>(idot) * (q_sc_reg[g] * k_sc);
                    if (has_rel) {
                        f32 r_part = (lane < 16) ? (s_R[g][lane] * rp) : 0.0f;
                        #pragma unroll
                        for (int off = 8; off > 0; off >>= 1) r_part += __shfl_xor_sync(0xFFFFFFFFu, r_part, off);
                        dot += __shfl_sync(0xFFFFFFFFu, r_part, 0);
                    }
                    f32 new_m = fmaxf(m_i[g], dot);
                    f32 ep = __expf(m_i[g] - new_m);
                    f32 ec = __expf(dot - new_m);
                    l_i[g] = l_i[g] * ep + ec;
                    f32 w_v = ec * v_sc;
                    o_reg[g][0] = fmaf(w_v, v0, o_reg[g][0] * ep);
                    o_reg[g][1] = fmaf(w_v, v1, o_reg[g][1] * ep);
                    o_reg[g][2] = fmaf(w_v, v2, o_reg[g][2] * ep);
                    o_reg[g][3] = fmaf(w_v, v3, o_reg[g][3] * ep);
                    m_i[g] = new_m;
                }
            }
            __syncthreads();
        }
    }

    __shared__ f32 sm_m[4], sm_l[4], sm_o[4][128];
    for (int g = 0; g < GQA; ++g) {
        if (lane == 0) { sm_m[wid] = m_i[g]; sm_l[wid] = l_i[g]; }
        #pragma unroll
        for (int e = 0; e < 4; ++e) sm_o[wid][lane * 4 + e] = o_reg[g][e];
        __syncthreads();
        if (wid == 0) {
            u32 qh = qh_base + g;
            size_t ml_base = ((static_cast<size_t>(m) * n_q_heads + qh) * num_splits + sp) * 2ULL;
            size_t o_base  = ((static_cast<size_t>(m) * n_q_heads + qh) * num_splits + sp) * 128ULL;
            f32 gm = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
            f32 gl = 0.0f, ws[4];
            #pragma unroll
            for (int w = 0; w < 4; ++w) { ws[w] = (sm_l[w] > 0.0f) ? __expf(sm_m[w] - gm) : 0.0f; gl += sm_l[w] * ws[w]; }
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                u32 d = static_cast<u32>(lane * 4 + e);
                d_split_o[o_base + d] = sm_o[0][d] * ws[0] + sm_o[1][d] * ws[1] + sm_o[2][d] * ws[2] + sm_o[3][d] * ws[3];
            }
            if (lane == 0) { d_split_ml[ml_base + 0] = gm; d_split_ml[ml_base + 1] = gl; }
        }
        __syncthreads();
    }
}

// 9c. Split-K Reduce + Single Inverse WHT128 + Gate & Sink (1 warp = 32 threads x 4 dims = 128, 0 SMEM)
__global__ void inkling_rel_attn_reduce_gate_sink_wht_kernel(
    const f32* __restrict__ d_split_o,
    const f32* __restrict__ d_split_ml,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ sink,
    __nv_bfloat16* __restrict__ attn_out,
    u32 n_q_heads, u32 num_splits) {
    const u32 qh   = blockIdx.x;
    const u32 m    = blockIdx.y;
    const int lane = static_cast<int>(threadIdx.x); // 0..31

    const size_t ml_head = (static_cast<size_t>(m) * n_q_heads + qh) * num_splits * 2ULL;
    const size_t o_head  = (static_cast<size_t>(m) * n_q_heads + qh) * num_splits * 128ULL;

    f32 gm = -1e30f;
    for (u32 sp = 0; sp < num_splits; ++sp) {
        if (d_split_ml[ml_head + sp * 2 + 1] > 0.0f) gm = fmaxf(gm, d_split_ml[ml_head + sp * 2 + 0]);
    }
    f32 gl = 0.0f;
    f32 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const u32 d0 = static_cast<u32>(lane * 4);
    for (u32 sp = 0; sp < num_splits; ++sp) {
        f32 sl = d_split_ml[ml_head + sp * 2 + 1];
        if (sl > 0.0f) {
            f32 w = __expf(d_split_ml[ml_head + sp * 2 + 0] - gm);
            gl += sl * w;
            const f32* op = d_split_o + o_head + static_cast<size_t>(sp) * 128ULL + d0;
            a0 = fmaf(op[0], w, a0);
            a1 = fmaf(op[1], w, a1);
            a2 = fmaf(op[2], w, a2);
            a3 = fmaf(op[3], w, a3);
        }
    }
    f32 inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    a0 *= inv_l; a1 *= inv_l; a2 *= inv_l; a3 *= inv_l;

    // Exact inverse 128-pt Walsh-Hadamard transform (H^2 = I) ONCE per attention head
    ink_wht128_warp(a0, a1, a2, a3, lane);

    const f32 lse = (gl > 0.0f) ? (gm + logf(gl)) : -1e30f;
    const f32 sink_mul = sigmoid_f32(lse - __bfloat162float(sink[static_cast<size_t>(m) * n_q_heads + qh]));
    const size_t o_off = (static_cast<size_t>(m) * n_q_heads + qh) * 128ULL + d0;
    attn_out[o_off + 0] = __float2bfloat16_rn(a0 * sigmoid_f32(__bfloat162float(gate[o_off + 0])) * sink_mul);
    attn_out[o_off + 1] = __float2bfloat16_rn(a1 * sigmoid_f32(__bfloat162float(gate[o_off + 1])) * sink_mul);
    attn_out[o_off + 2] = __float2bfloat16_rn(a2 * sigmoid_f32(__bfloat162float(gate[o_off + 2])) * sink_mul);
    attn_out[o_off + 3] = __float2bfloat16_rn(a3 * sigmoid_f32(__bfloat162float(gate[o_off + 3])) * sink_mul);
}

// Raw PTX m16n8k16 BF16->FP32 Tensor-Core MMA & ldmatrix helpers (sm_80+)
__device__ __forceinline__ void mma_m16n8k16_bf16(
    f32& d0, f32& d1, f32& d2, f32& d3,
    u32 a0, u32 a1, u32 a2, u32 a3,
    u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrix_x4(u32& r0, u32& r1, u32& r2, u32& r3, const void* smem_ptr) {
    u32 addr = static_cast<u32>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(u32& r0, u32& r1, const void* smem_ptr) {
    u32 addr = static_cast<u32>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(u32& r0, u32& r1, const void* smem_ptr) {
    u32 addr = static_cast<u32>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(addr));
}

// 9d. Canonical 128x64 (8-Warp, 256-Thread) Raw PTX FlashAttention-2 Global Prefill
// Grid: (n_q_heads=24, (M+127)/128), Block: 256 threads (8 warps x 16 Q rows = 128 Q rows x 64 KV tokens).
// Full dual-issue SM saturation (2 warps per warp scheduler), 2x less KV unpack than 64x64,
// zero sm_S / sm_P round-trips, 60.4 KB dynamic SMEM, 0 stack frame, 0 spills.
template <bool K_IS_4BIT>
__global__ void inkling_global_attn_prefill_wmma_wht_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ r,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ sink,
    const __nv_bfloat16* __restrict__ rel_proj_T,
    const u8* __restrict__ k_cache,
    const u8* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ k_scales_bf16,
    const __nv_bfloat16* __restrict__ v_scales_bf16,
    const i32* __restrict__ d_block_table,
    __nv_bfloat16* __restrict__ attn_out,
    u32 start_pos, u32 M, u32 n_q_heads, u32 n_kv_heads, u32 block_size) {
    constexpr int STRIDE = 136; // 128 + 8 BF16 pad (272 B = 16 B aligned, zero 32-bank conflicts)
    const u32 qh     = blockIdx.x;
    const u32 q_tile = blockIdx.y * 128u;
    const u32 tid    = threadIdx.x; // 0..255 (8 warps)
    const u32 wid    = tid >> 5;    // 0..7
    const int lane   = static_cast<int>(tid & 31u);
    const int gid    = lane >> 2;   // 0..7
    const int tid4   = lane & 3;    // 0..3
    const u32 kvh    = qh / (n_q_heads / n_kv_heads);
    const u32 qr_base = wid * 16u;
    const f32 qk_scale = rsqrtf(128.0f);
    constexpr f32 kLog2E = 1.4426950408889634f;

    // Dynamic SMEM layout (60,416 B total, 16-byte aligned):
    //   sm_Q[128][136] BF16 (34,816 B) + sm_KV[64][136] BF16 (17,408 B) + sm_R[128][16] f32 (8,192 B)
    extern __shared__ u8 smem_raw[];
    __nv_bfloat16 (*sm_Q)[STRIDE]  = reinterpret_cast<__nv_bfloat16 (*)[STRIDE]>(smem_raw);
    __nv_bfloat16 (*sm_KV)[STRIDE] = reinterpret_cast<__nv_bfloat16 (*)[STRIDE]>(smem_raw + 128ULL * STRIDE * sizeof(__nv_bfloat16));
    f32           (*sm_R)[16]      = reinterpret_cast<f32 (*)[16]>(smem_raw + (128ULL + 64ULL) * STRIDE * sizeof(__nv_bfloat16));

    // Load 128 Q rows (16 per warp) and apply 128-pt WHT once when K_IS_4BIT
    for (u32 r_i = 0; r_i < 16u; ++r_i) {
        u32 qr = qr_base + r_i;
        u32 m  = q_tile + qr;
        f32 q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
        if (m < M) {
            const __nv_bfloat16* qp = q + (static_cast<size_t>(m) * n_q_heads + qh) * 128ULL + lane * 4;
            q0 = __bfloat162float(qp[0]) * qk_scale;
            q1 = __bfloat162float(qp[1]) * qk_scale;
            q2 = __bfloat162float(qp[2]) * qk_scale;
            q3 = __bfloat162float(qp[3]) * qk_scale;
            if constexpr (K_IS_4BIT) ink_wht128_warp(q0, q1, q2, q3, lane);
        }
        __nv_bfloat162 q01 = __floats2bfloat162_rn(q0, q1);
        __nv_bfloat162 q23 = __floats2bfloat162_rn(q2, q3);
        *reinterpret_cast<uint2*>(&sm_Q[qr][lane * 4]) =
            make_uint2(*reinterpret_cast<const u32*>(&q01), *reinterpret_cast<const u32*>(&q23));
    }
    for (u32 idx = tid; idx < 128u * 16u; idx += 256u) {
        u32 qr = idx >> 4, j = idx & 15u;
        u32 m  = q_tile + qr;
        sm_R[qr][j] = (m < M) ? __bfloat162float(r[(static_cast<size_t>(m) * n_q_heads + qh) * 16ULL + j]) : 0.0f;
    }
    __syncthreads();

    // Per-warp output accumulators: 16 tiles of 16x8 = 16x128 FP32 (64 registers)
    f32 o_reg[16][4];
    #pragma unroll
    for (int d = 0; d < 16; ++d) {
        o_reg[d][0] = 0.0f; o_reg[d][1] = 0.0f; o_reg[d][2] = 0.0f; o_reg[d][3] = 0.0f;
    }
    f32 m_r0 = -1e30f, m_r1 = -1e30f;
    f32 l_r0 = 0.0f,   l_r1 = 0.0f;

    const u32 block_pos_min = start_pos + q_tile;
    const u32 block_pos_max = start_pos + min(M, q_tile + 128u) - 1u;
    const u32 kv_end = block_pos_max + 1u;

    const u32 row0_m = q_tile + qr_base + static_cast<u32>(gid);
    const u32 row1_m = row0_m + 8u;
    const u32 row0_pos = start_pos + row0_m;
    const u32 row1_pos = start_pos + row1_m;

    const u32 q_ld_row = qr_base + static_cast<u32>(lane & 15);
    const u32 q_ld_col = static_cast<u32>((lane >> 4) * 8);
    const u32 k_ld_row = static_cast<u32>(lane & 7);
    const u32 k_ld_col = static_cast<u32>(((lane >> 3) & 1) * 8);
    const u32 v_ld_row = static_cast<u32>(lane & 15);

    for (u32 kv0 = 0; kv0 < kv_end; kv0 += 64u) {
        // Keep 4 block bases in scalar registers (0 bytes stack frame)
        const size_t b0 = (static_cast<size_t>(__ldg(d_block_table + (kv0 / block_size))) * n_kv_heads + kvh) * block_size;
        const size_t b1 = (kv0 + 16u < kv_end) ? (static_cast<size_t>(__ldg(d_block_table + ((kv0 + 16u) / block_size))) * n_kv_heads + kvh) * block_size : 0ULL;
        const size_t b2 = (kv0 + 32u < kv_end) ? (static_cast<size_t>(__ldg(d_block_table + ((kv0 + 32u) / block_size))) * n_kv_heads + kvh) * block_size : 0ULL;
        const size_t b3 = (kv0 + 48u < kv_end) ? (static_cast<size_t>(__ldg(d_block_table + ((kv0 + 48u) / block_size))) * n_kv_heads + kvh) * block_size : 0ULL;

        const bool tile_has_rel = (block_pos_min < kv0 + 64u + 1024u);
        const bool tile_causal_edge = (kv0 + 63u >= block_pos_min) || (q_tile + 128u > M);

        // Stage 1: 8 warps (256 threads) unpack 64 K tokens (only 8 tokens/warp)
        #pragma unroll 4
        for (u32 kr = wid; kr < 64u; kr += 8u) {
            u32 t = kv0 + kr;
            auto* k_dst = reinterpret_cast<uint2*>(&sm_KV[kr][lane * 4]);
            if (t < kv_end) {
                u32 bi = kr >> 4;
                size_t bb = (bi == 0u) ? b0 : ((bi == 1u) ? b1 : ((bi == 2u) ? b2 : b3));
                size_t sc_off = bb + (kr & 15u);
                f32 ksc = __bfloat162float(__ldg(k_scales_bf16 + sc_off));
                __nv_bfloat162 k01, k23;
                if constexpr (K_IS_4BIT) {
                    u16 kp = __ldg(reinterpret_cast<const u16*>(k_cache + sc_off * 64ULL) + lane);
                    __nv_bfloat162 ksc2 = __float2bfloat162_rn(ksc);
                    __nv_bfloat162 kbi2 = __float2bfloat162_rn(-136.0f * ksc);
                    k01 = unpack_2nib_bf162(kp, ksc2, kbi2);
                    k23 = unpack_2nib_bf162(kp >> 8, ksc2, kbi2);
                } else {
                    char4 kw = __ldg(reinterpret_cast<const char4*>(k_cache + sc_off * 128ULL) + lane);
                    k01 = __floats2bfloat162_rn(static_cast<f32>(kw.x) * ksc, static_cast<f32>(kw.y) * ksc);
                    k23 = __floats2bfloat162_rn(static_cast<f32>(kw.z) * ksc, static_cast<f32>(kw.w) * ksc);
                }
                *k_dst = make_uint2(*reinterpret_cast<const u32*>(&k01), *reinterpret_cast<const u32*>(&k23));
            } else {
                *k_dst = make_uint2(0u, 0u);
            }
        }
        __syncthreads();

        // Stage 2: Raw PTX m16n8k16 Q * K^T -> 16x64 FP32 scores held purely in s_reg[8][4]
        f32 s_reg[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s_reg[nt][0] = 0.0f; s_reg[nt][1] = 0.0f; s_reg[nt][2] = 0.0f; s_reg[nt][3] = 0.0f;
        }
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            u32 qa0, qa1, qa2, qa3;
            ldmatrix_x4(qa0, qa1, qa2, qa3, &sm_Q[q_ld_row][ks * 16 + q_ld_col]);
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                u32 kb0, kb1;
                ldmatrix_x2(kb0, kb1, &sm_KV[nt * 8 + k_ld_row][ks * 16 + k_ld_col]);
                mma_m16n8k16_bf16(s_reg[nt][0], s_reg[nt][1], s_reg[nt][2], s_reg[nt][3],
                                  qa0, qa1, qa2, qa3, kb0, kb1);
            }
        }

        // Stage 3: 8 warps unpack 64 V tokens (only 8 tokens/warp) right after Q*K^T finishes
        __syncthreads();
        #pragma unroll 4
        for (u32 kr = wid; kr < 64u; kr += 8u) {
            u32 t = kv0 + kr;
            auto* v_dst = reinterpret_cast<uint2*>(&sm_KV[kr][lane * 4]);
            if (t < kv_end) {
                u32 bi = kr >> 4;
                size_t bb = (bi == 0u) ? b0 : ((bi == 1u) ? b1 : ((bi == 2u) ? b2 : b3));
                size_t sc_off = bb + (kr & 15u);
                f32 vsc = __bfloat162float(__ldg(v_scales_bf16 + sc_off));
                u16 vp = __ldg(reinterpret_cast<const u16*>(v_cache + sc_off * 64ULL) + lane);
                __nv_bfloat162 vsc2 = __float2bfloat162_rn(vsc);
                __nv_bfloat162 vbi2 = __float2bfloat162_rn(-136.0f * vsc);
                __nv_bfloat162 v01 = unpack_2nib_bf162(vp, vsc2, vbi2);
                __nv_bfloat162 v23 = unpack_2nib_bf162(vp >> 8, vsc2, vbi2);
                *v_dst = make_uint2(*reinterpret_cast<const u32*>(&v01), *reinterpret_cast<const u32*>(&v23));
            } else {
                *v_dst = make_uint2(0u, 0u);
            }
        }

        // Stage 4: In-register causal mask, relative bias, and online softmax (no sm_S / sm_P)
        if (tile_has_rel || tile_causal_edge) {
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                u32 tc0 = kv0 + static_cast<u32>(nt * 8 + tid4 * 2);
                u32 tc1 = tc0 + 1u;
                // Row 0 (gid)
                if (row0_m >= M || tc0 > row0_pos) {
                    s_reg[nt][0] = -1e30f;
                } else if (tile_has_rel && (row0_pos - tc0 < 1024u)) {
                    const __nv_bfloat16* rp = rel_proj_T + static_cast<size_t>(row0_pos - tc0) * 16ULL;
                    f32 rb = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) rb = fmaf(sm_R[qr_base + gid][j], __bfloat162float(__ldg(rp + j)), rb);
                    s_reg[nt][0] += rb;
                }
                if (row0_m >= M || tc1 > row0_pos) {
                    s_reg[nt][1] = -1e30f;
                } else if (tile_has_rel && (row0_pos - tc1 < 1024u)) {
                    const __nv_bfloat16* rp = rel_proj_T + static_cast<size_t>(row0_pos - tc1) * 16ULL;
                    f32 rb = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) rb = fmaf(sm_R[qr_base + gid][j], __bfloat162float(__ldg(rp + j)), rb);
                    s_reg[nt][1] += rb;
                }
                // Row 1 (gid + 8)
                if (row1_m >= M || tc0 > row1_pos) {
                    s_reg[nt][2] = -1e30f;
                } else if (tile_has_rel && (row1_pos - tc0 < 1024u)) {
                    const __nv_bfloat16* rp = rel_proj_T + static_cast<size_t>(row1_pos - tc0) * 16ULL;
                    f32 rb = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) rb = fmaf(sm_R[qr_base + gid + 8][j], __bfloat162float(__ldg(rp + j)), rb);
                    s_reg[nt][2] += rb;
                }
                if (row1_m >= M || tc1 > row1_pos) {
                    s_reg[nt][3] = -1e30f;
                } else if (tile_has_rel && (row1_pos - tc1 < 1024u)) {
                    const __nv_bfloat16* rp = rel_proj_T + static_cast<size_t>(row1_pos - tc1) * 16ULL;
                    f32 rb = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) rb = fmaf(sm_R[qr_base + gid + 8][j], __bfloat162float(__ldg(rp + j)), rb);
                    s_reg[nt][3] += rb;
                }
            }
        }

        // Row-wise max across the 4 lanes (tid4 = 0..3) owning each row
        f32 new_m0 = m_r0;
        f32 new_m1 = m_r1;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            new_m0 = fmaxf(new_m0, fmaxf(s_reg[nt][0], s_reg[nt][1]));
            new_m1 = fmaxf(new_m1, fmaxf(s_reg[nt][2], s_reg[nt][3]));
        }
        new_m0 = fmaxf(new_m0, __shfl_xor_sync(0xFFFFFFFFu, new_m0, 1));
        new_m0 = fmaxf(new_m0, __shfl_xor_sync(0xFFFFFFFFu, new_m0, 2));
        new_m1 = fmaxf(new_m1, __shfl_xor_sync(0xFFFFFFFFu, new_m1, 1));
        new_m1 = fmaxf(new_m1, __shfl_xor_sync(0xFFFFFFFFu, new_m1, 2));

        f32 ep0 = (l_r0 > 0.0f) ? exp2f((m_r0 - new_m0) * kLog2E) : 0.0f;
        f32 ep1 = (l_r1 > 0.0f) ? exp2f((m_r1 - new_m1) * kLog2E) : 0.0f;
        m_r0 = new_m0;
        m_r1 = new_m1;

        // Rescale existing o_reg in-place once per 64 tokens
        #pragma unroll
        for (int d = 0; d < 16; ++d) {
            o_reg[d][0] *= ep0; o_reg[d][1] *= ep0;
            o_reg[d][2] *= ep1; o_reg[d][3] *= ep1;
        }

        // Exponentiate scores and pack directly into m16n8k16 BF16x2 A-fragments (no shuffles)
        f32 p_sum0 = 0.0f, p_sum1 = 0.0f;
        u32 p_reg[4][4];
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            int n0 = ks * 2, n1 = ks * 2 + 1;
            f32 p00 = (s_reg[n0][0] > -1e29f) ? exp2f((s_reg[n0][0] - m_r0) * kLog2E) : 0.0f;
            f32 p01 = (s_reg[n0][1] > -1e29f) ? exp2f((s_reg[n0][1] - m_r0) * kLog2E) : 0.0f;
            f32 p10 = (s_reg[n0][2] > -1e29f) ? exp2f((s_reg[n0][2] - m_r1) * kLog2E) : 0.0f;
            f32 p11 = (s_reg[n0][3] > -1e29f) ? exp2f((s_reg[n0][3] - m_r1) * kLog2E) : 0.0f;

            f32 p02 = (s_reg[n1][0] > -1e29f) ? exp2f((s_reg[n1][0] - m_r0) * kLog2E) : 0.0f;
            f32 p03 = (s_reg[n1][1] > -1e29f) ? exp2f((s_reg[n1][1] - m_r0) * kLog2E) : 0.0f;
            f32 p12 = (s_reg[n1][2] > -1e29f) ? exp2f((s_reg[n1][2] - m_r1) * kLog2E) : 0.0f;
            f32 p13 = (s_reg[n1][3] > -1e29f) ? exp2f((s_reg[n1][3] - m_r1) * kLog2E) : 0.0f;

            p_sum0 += (p00 + p01) + (p02 + p03);
            p_sum1 += (p10 + p11) + (p12 + p13);

            __nv_bfloat162 a0 = __floats2bfloat162_rn(p00, p01);
            __nv_bfloat162 a1 = __floats2bfloat162_rn(p10, p11);
            __nv_bfloat162 a2 = __floats2bfloat162_rn(p02, p03);
            __nv_bfloat162 a3 = __floats2bfloat162_rn(p12, p13);
            p_reg[ks][0] = *reinterpret_cast<const u32*>(&a0);
            p_reg[ks][1] = *reinterpret_cast<const u32*>(&a1);
            p_reg[ks][2] = *reinterpret_cast<const u32*>(&a2);
            p_reg[ks][3] = *reinterpret_cast<const u32*>(&a3);
        }
        p_sum0 += __shfl_xor_sync(0xFFFFFFFFu, p_sum0, 1);
        p_sum0 += __shfl_xor_sync(0xFFFFFFFFu, p_sum0, 2);
        p_sum1 += __shfl_xor_sync(0xFFFFFFFFu, p_sum1, 1);
        p_sum1 += __shfl_xor_sync(0xFFFFFFFFu, p_sum1, 2);
        l_r0 = l_r0 * ep0 + p_sum0;
        l_r1 = l_r1 * ep1 + p_sum1;

        __syncthreads(); // Wait for Stage 3 V unpack into sm_KV to complete

        // Stage 5: Raw PTX m16n8k16 P * V -> accumulate directly into o_reg[16][4]
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            #pragma unroll
            for (int dt = 0; dt < 16; ++dt) {
                u32 vb0, vb1;
                ldmatrix_x2_trans(vb0, vb1, &sm_KV[ks * 16 + v_ld_row][dt * 8]);
                mma_m16n8k16_bf16(o_reg[dt][0], o_reg[dt][1], o_reg[dt][2], o_reg[dt][3],
                                  p_reg[ks][0], p_reg[ks][1], p_reg[ks][2], p_reg[ks][3],
                                  vb0, vb1);
            }
        }
        __syncthreads();
    }

    // Epilogue: reuse sm_KV (17.4 KB) as f32 sm_O[16][128] (8 KB) + f32 sm_ml[16][2] one warp at a time
    f32 (*sm_O)[128] = reinterpret_cast<f32 (*)[128]>(sm_KV);
    __shared__ f32 sm_m_ep[16];
    __shared__ f32 sm_l_ep[16];

    for (u32 w_turn = 0; w_turn < 8u; ++w_turn) {
        if (wid == w_turn) {
            if (tid4 == 0) {
                sm_m_ep[gid]     = m_r0;
                sm_l_ep[gid]     = l_r0;
                sm_m_ep[gid + 8] = m_r1;
                sm_l_ep[gid + 8] = l_r1;
            }
            #pragma unroll
            for (int dt = 0; dt < 16; ++dt) {
                sm_O[gid][dt * 8 + tid4 * 2 + 0]     = o_reg[dt][0];
                sm_O[gid][dt * 8 + tid4 * 2 + 1]     = o_reg[dt][1];
                sm_O[gid + 8][dt * 8 + tid4 * 2 + 0] = o_reg[dt][2];
                sm_O[gid + 8][dt * 8 + tid4 * 2 + 1] = o_reg[dt][3];
            }
            __syncwarp();
            for (u32 r_i = 0; r_i < 16u; ++r_i) {
                u32 m = q_tile + qr_base + r_i;
                if (m < M) {
                    f32 l_val = sm_l_ep[r_i];
                    f32 inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
                    f32 a0 = sm_O[r_i][lane * 4 + 0] * inv_l;
                    f32 a1 = sm_O[r_i][lane * 4 + 1] * inv_l;
                    f32 a2 = sm_O[r_i][lane * 4 + 2] * inv_l;
                    f32 a3 = sm_O[r_i][lane * 4 + 3] * inv_l;
                    ink_wht128_warp(a0, a1, a2, a3, lane);
                    f32 lse = (l_val > 0.0f) ? (sm_m_ep[r_i] + logf(l_val)) : -1e30f;
                    f32 sink_mul = sigmoid_f32(lse - __bfloat162float(sink[static_cast<size_t>(m) * n_q_heads + qh]));
                    size_t o_off = (static_cast<size_t>(m) * n_q_heads + qh) * 128ULL + static_cast<u32>(lane * 4);
                    attn_out[o_off + 0] = __float2bfloat16_rn(a0 * sigmoid_f32(__bfloat162float(gate[o_off + 0])) * sink_mul);
                    attn_out[o_off + 1] = __float2bfloat16_rn(a1 * sigmoid_f32(__bfloat162float(gate[o_off + 1])) * sink_mul);
                    attn_out[o_off + 2] = __float2bfloat16_rn(a2 * sigmoid_f32(__bfloat162float(gate[o_off + 2])) * sink_mul);
                    attn_out[o_off + 3] = __float2bfloat16_rn(a3 * sigmoid_f32(__bfloat162float(gate[o_off + 3])) * sink_mul);
                }
            }
        }
        __syncthreads();
    }
}

} // anonymous namespace (WHT global 1M kernels)


void launch_inkling_kv_sconv_write_wht(
    const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
    const __nv_bfloat16* wk, const __nv_bfloat16* wv,
    const __nv_bfloat16* dk_dyn, const __nv_bfloat16* dv_dyn,
    __nv_bfloat16* k_hist, __nv_bfloat16* v_hist,
    u8* k_cache, u8* v_cache,
    __nv_bfloat16* k_scales_bf16, __nv_bfloat16* v_scales_bf16,
    const i32* d_block_table, const i32* d_pos,
    u32 start_pos, u32 M, u32 n_kv_heads, u32 block_size,
    bool k_is_4bit, f32 k_clip, f32 rms_eps, cudaStream_t stream) {
    dim3 grid(n_kv_heads, 2);
    if (k_is_4bit) {
        inkling_kv_sconv_write_wht_kernel<true><<<grid, 32, 0, stream>>>(
            k_raw, v_raw, wk, wv, dk_dyn, dv_dyn, k_hist, v_hist,
            k_cache, v_cache, k_scales_bf16, v_scales_bf16,
            d_block_table, d_pos, start_pos, M, n_kv_heads, block_size, k_clip, rms_eps);
    } else {
        inkling_kv_sconv_write_wht_kernel<false><<<grid, 32, 0, stream>>>(
            k_raw, v_raw, wk, wv, dk_dyn, dv_dyn, k_hist, v_hist,
            k_cache, v_cache, k_scales_bf16, v_scales_bf16,
            d_block_table, d_pos, start_pos, M, n_kv_heads, block_size, k_clip, rms_eps);
    }
}

void launch_inkling_global_attn_splitk_wht(
    const __nv_bfloat16* q, const __nv_bfloat16* r,
    const __nv_bfloat16* gate, const __nv_bfloat16* sink,
    const __nv_bfloat16* rel_proj_T,
    const u8* k_cache, const u8* v_cache,
    const __nv_bfloat16* k_scales_bf16, const __nv_bfloat16* v_scales_bf16,
    const i32* d_block_table, __nv_bfloat16* attn_out,
    f32* d_split_o, f32* d_split_ml,
    const i32* d_pos, u32 start_pos, u32 M,
    u32 n_q_heads, u32 n_kv_heads, u32 block_size,
    u32 num_splits, bool k_is_4bit, cudaStream_t stream) {
    dim3 g1(n_kv_heads, num_splits, M);
    dim3 g2(n_q_heads, M);
    if (k_is_4bit) {
        inkling_global_attn_splitk_wht_kernel<true><<<g1, 128, 0, stream>>>(
            q, r, rel_proj_T, k_cache, v_cache, k_scales_bf16, v_scales_bf16,
            d_block_table, d_split_o, d_split_ml, d_pos, start_pos,
            n_q_heads, n_kv_heads, block_size, num_splits);
    } else {
        inkling_global_attn_splitk_wht_kernel<false><<<g1, 128, 0, stream>>>(
            q, r, rel_proj_T, k_cache, v_cache, k_scales_bf16, v_scales_bf16,
            d_block_table, d_split_o, d_split_ml, d_pos, start_pos,
            n_q_heads, n_kv_heads, block_size, num_splits);
    }
    inkling_rel_attn_reduce_gate_sink_wht_kernel<<<g2, 32, 0, stream>>>(
        d_split_o, d_split_ml, gate, sink, attn_out, n_q_heads, num_splits);
}

void launch_inkling_global_attn_prefill_wmma_wht(
    const __nv_bfloat16* q, const __nv_bfloat16* r,
    const __nv_bfloat16* gate, const __nv_bfloat16* sink,
    const __nv_bfloat16* rel_proj_T,
    const u8* k_cache, const u8* v_cache,
    const __nv_bfloat16* k_scales_bf16, const __nv_bfloat16* v_scales_bf16,
    const i32* d_block_table, __nv_bfloat16* attn_out,
    u32 start_pos, u32 M, u32 n_q_heads, u32 n_kv_heads, u32 block_size,
    bool k_is_4bit, cudaStream_t stream) {
    constexpr size_t kFa2SmemBytes = (128ULL + 64ULL) * 136ULL * sizeof(__nv_bfloat16) + 128ULL * 16ULL * sizeof(f32);
    static bool smem_configured = false;
    if (!smem_configured) {
        cudaFuncSetAttribute(inkling_global_attn_prefill_wmma_wht_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kFa2SmemBytes));
        cudaFuncSetAttribute(inkling_global_attn_prefill_wmma_wht_kernel<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kFa2SmemBytes));
        smem_configured = true;
    }
    dim3 grid(n_q_heads, (M + 127u) / 128u);
    if (k_is_4bit) {
        inkling_global_attn_prefill_wmma_wht_kernel<true><<<grid, 256, kFa2SmemBytes, stream>>>(
            q, r, gate, sink, rel_proj_T, k_cache, v_cache, k_scales_bf16, v_scales_bf16,
            d_block_table, attn_out, start_pos, M, n_q_heads, n_kv_heads, block_size);
    } else {
        inkling_global_attn_prefill_wmma_wht_kernel<false><<<grid, 256, kFa2SmemBytes, stream>>>(
            q, r, gate, sink, rel_proj_T, k_cache, v_cache, k_scales_bf16, v_scales_bf16,
            d_block_table, attn_out, start_pos, M, n_q_heads, n_kv_heads, block_size);
    }
}

} // namespace flint::cuda
