#include "inkling2_common.cuh"
#include <mma.h>

namespace flint::cuda {

namespace {

// ============================================================================
// 5. Inkling-2 Relative-Position-Bias GQA Attention + Output Gate & Sink
// ============================================================================
// Score(q_pos, k_pos) = dot(q, k) * rsqrt(128) + rel_bias(q_pos - k_pos)
// where rel_bias(dist) = dot(r_clipped[h], rel_proj_T[dist]) for 0 <= dist < 1024, else 0.
// Output epilogue: out[h, d] = (o[h, d] / l) * sigmoid(gate[h, d]) * sigmoid(lse - sink[h])
// Grid: (n_q_heads, M), Block: 256 threads (8 warps + PTX __dp4a -> 1-launch local SWA)
template <bool INT8_KV>
__global__ void inkling_rel_attn_kernel(
    const __nv_bfloat16* __restrict__ q,          // [M, Hq, 128]
    const __nv_bfloat16* __restrict__ r,          // [M, Hq, 16]
    const __nv_bfloat16* __restrict__ gate,       // [M, Hq, 128]
    const __nv_bfloat16* __restrict__ sink,       // [M, Hq]
    const __nv_bfloat16* __restrict__ rel_proj_T, // [1024, 16] transposed at load time
    const void* __restrict__ k_cache_raw,
    const void* __restrict__ v_cache_raw,
    const f32* __restrict__ k_scales,
    const f32* __restrict__ v_scales,
    const i32* __restrict__ d_block_table,
    __nv_bfloat16* __restrict__ attn_out,         // [M, Hq, 128]
    const i32* __restrict__ d_pos,
    u32 start_pos, u32 n_q_heads, u32 n_kv_heads, u32 block_size,
    bool is_local_ring, u32 swa_window) {
    const u32 qh   = blockIdx.x;
    const u32 m    = blockIdx.y;
    const u32 tid  = threadIdx.x; // 0..255 (8 warps)
    const u32 wid  = tid >> 5;
    const u32 lane = tid & 31;

    const u32 gqa = n_q_heads / n_kv_heads;
    const u32 kvh = qh / gqa;
    const u32 pos = (d_pos ? static_cast<u32>(*d_pos) : start_pos) + m;
    const u32 kv_start = (is_local_ring && pos >= swa_window) ? (pos - swa_window + 1) : 0;
    const u32 kv_end   = pos + 1;
    const f32 qk_scale = rsqrtf(128.0f);

    const __nv_bfloat16* q_ptr = q + (static_cast<size_t>(m) * n_q_heads + qh) * 128;
    f32 q_reg[4];
    #pragma unroll
    for (int e = 0; e < 4; ++e) q_reg[e] = __bfloat162float(q_ptr[lane * 4 + e]) * qk_scale;

    int q_i8_pack = 0;
    f32 q_i8_sc   = 1.0f;
    if constexpr (INT8_KV) {
        f32 am = fmaxf(fmaxf(fabsf(q_reg[0]), fabsf(q_reg[1])), fmaxf(fabsf(q_reg[2]), fabsf(q_reg[3])));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, off));
        q_i8_sc = fmaxf(am / 127.0f, 1e-8f);
        f32 inv_q = 1.0f / q_i8_sc;
        int q0 = max(-127, min(127, __float2int_rn(q_reg[0] * inv_q)));
        int q1 = max(-127, min(127, __float2int_rn(q_reg[1] * inv_q)));
        int q2 = max(-127, min(127, __float2int_rn(q_reg[2] * inv_q)));
        int q3 = max(-127, min(127, __float2int_rn(q_reg[3] * inv_q)));
        q_i8_pack = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
    }

    __shared__ f32 sm_r[16];
    if (tid < 16) {
        sm_r[tid] = __bfloat162float(r[(static_cast<size_t>(m) * n_q_heads + qh) * 16 + tid]);
    }
    __syncthreads();

    f32 m_i = -1e30f, l_i = 0.0f;
    f32 o_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (u32 t = kv_start + wid; t < kv_end; t += 8) {
        size_t scale_off;
        if (is_local_ring) {
            u32 rslot = t & (swa_window - 1u);
            scale_off = static_cast<size_t>(kvh) * swa_window + rslot;
        } else {
            u32 blk = t / block_size;
            u32 slot = t - blk * block_size;
            i32 phys = d_block_table[blk];
            scale_off = (static_cast<size_t>(phys) * n_kv_heads + kvh) * block_size + slot;
        }
        size_t kv_off = scale_off * 128;

        f32 v_reg[4];
        f32 dot = 0.0f;
        f32 v_sc = 1.0f;
        if constexpr (INT8_KV) {
            f32 k_sc = __ldg(k_scales + scale_off);
            v_sc = __ldg(v_scales + scale_off);
            int kw32 = __ldg(reinterpret_cast<const int*>(static_cast<const i8*>(k_cache_raw) + kv_off) + lane);
            char4 vw = __ldg(reinterpret_cast<const char4*>(static_cast<const i8*>(v_cache_raw) + kv_off) + lane);
            v_reg[0] = vw.x; v_reg[1] = vw.y; v_reg[2] = vw.z; v_reg[3] = vw.w;
            int idot = __dp4a(q_i8_pack, kw32, 0);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) idot += __shfl_xor_sync(0xFFFFFFFFu, idot, off);
            dot = static_cast<f32>(idot) * (q_i8_sc * k_sc);
        } else {
            uint2 ku = __ldg(reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(k_cache_raw) + kv_off) + lane);
            uint2 vu = __ldg(reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(v_cache_raw) + kv_off) + lane);
            const auto* k2 = reinterpret_cast<const __nv_bfloat162*>(&ku);
            const auto* v2 = reinterpret_cast<const __nv_bfloat162*>(&vu);
            float2 fk0 = __bfloat1622float2(k2[0]), fk1 = __bfloat1622float2(k2[1]);
            float2 fv0 = __bfloat1622float2(v2[0]), fv1 = __bfloat1622float2(v2[1]);
            v_reg[0] = fv0.x; v_reg[1] = fv0.y; v_reg[2] = fv1.x; v_reg[3] = fv1.y;
            dot = fmaf(q_reg[0], fk0.x, fmaf(q_reg[1], fk0.y, fmaf(q_reg[2], fk1.x, q_reg[3] * fk1.y)));
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
        }

        u32 dist = pos - t;
        f32 rel_b = 0.0f;
        if (dist < 1024u) {
            f32 r_part = (lane < 16) ? (sm_r[lane] * __bfloat162float(__ldg(rel_proj_T + static_cast<size_t>(dist) * 16 + lane))) : 0.0f;
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) r_part += __shfl_xor_sync(0xFFFFFFFFu, r_part, off);
            rel_b = __shfl_sync(0xFFFFFFFFu, r_part, 0);
        }
        dot += rel_b;

        f32 new_m = fmaxf(m_i, dot);
        f32 ep = __expf(m_i - new_m);
        f32 ec = __expf(dot - new_m);
        l_i = l_i * ep + ec;
        f32 w_v = INT8_KV ? (ec * v_sc) : ec;
        #pragma unroll
        for (int e = 0; e < 4; ++e) o_reg[e] = fmaf(w_v, v_reg[e], o_reg[e] * ep);
        m_i = new_m;
    }

    __shared__ f32 sm_m[8];
    __shared__ f32 sm_l[8];
    __shared__ f32 sm_o[8][128];
    if (lane == 0) { sm_m[wid] = m_i; sm_l[wid] = l_i; }
    #pragma unroll
    for (int e = 0; e < 4; ++e) sm_o[wid][lane * 4 + e] = o_reg[e];
    __syncthreads();

    if (tid < 128) {
        f32 mg = -1e30f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) mg = fmaxf(mg, sm_m[w]);
        f32 lg = 0.0f, acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            f32 ww = (sm_l[w] > 0.0f) ? __expf(sm_m[w] - mg) : 0.0f;
            lg  += ww * sm_l[w];
            acc += ww * sm_o[w][tid];
        }
        f32 o_val = (lg > 0.0f) ? (acc / lg) : 0.0f;

        f32 lse = (lg > 0.0f) ? (mg + logf(lg)) : -1e30f;
        size_t o_off = (static_cast<size_t>(m) * n_q_heads + qh) * 128 + tid;
        f32 g_val = __bfloat162float(gate[o_off]);
        f32 s_val = __bfloat162float(sink[static_cast<size_t>(m) * n_q_heads + qh]);
        o_val = o_val * sigmoid_f32(g_val) * sigmoid_f32(lse - s_val);
        attn_out[o_off] = __float2bfloat16_rn(o_val);
    }
}

// ============================================================================
// 7. GQA-Grouped Split-K Relative-Bias Attention (8 KV heads x splits -> 128-256 blocks)
// 3 Q-heads share 1 KV load (3x less KV bandwidth) + L1-cached rel_proj_T
// ============================================================================
template <bool INT8_KV>
__global__ void inkling_rel_attn_gqa_splitk_kernel(
    const __nv_bfloat16* __restrict__ q,          // [M, 24, 128]
    const __nv_bfloat16* __restrict__ r,          // [M, 24, 16]
    const __nv_bfloat16* __restrict__ rel_proj_T, // [1024, 16]
    const void* __restrict__ k_cache_raw,
    const void* __restrict__ v_cache_raw,
    const f32* __restrict__ k_scales,
    const f32* __restrict__ v_scales,
    const i32* __restrict__ d_block_table,
    f32* __restrict__ d_split_o,                  // [M, 24, num_splits, 128]
    f32* __restrict__ d_split_ml,                 // [M, 24, num_splits, 2]
    const i32* __restrict__ d_pos,
    u32 start_pos, u32 n_q_heads, u32 n_kv_heads, u32 block_size,
    u32 num_splits, bool is_local_ring, u32 swa_window) {
    constexpr int GQA = 3; // Inkling-2-Mini: 24 Q heads / 8 KV heads = 3
    const u32 kvh = blockIdx.x;
    const u32 sp  = blockIdx.y;
    const u32 m   = blockIdx.z;
    const u32 tid = threadIdx.x;
    const u32 wid = tid >> 5;
    const u32 lane = tid & 31;

    const u32 qh_base = kvh * GQA;
    const u32 pos = (d_pos ? static_cast<u32>(*d_pos) : start_pos) + m;
    const u32 kv_start = (is_local_ring && pos >= swa_window) ? (pos - swa_window + 1) : 0;
    const u32 kv_end   = pos + 1;
    const u32 span     = kv_end - kv_start;
    const u32 chunk    = (span + num_splits - 1) / num_splits;
    const u32 t_start  = kv_start + sp * chunk;
    const u32 t_end    = min(kv_end, t_start + chunk);

    if (t_start >= t_end) {
        if (wid == 0 && lane == 0) {
            #pragma unroll
            for (int g = 0; g < GQA; ++g) {
                size_t ml = ((static_cast<size_t>(m) * n_q_heads + qh_base + g) * num_splits + sp) * 2;
                d_split_ml[ml + 0] = -1e30f;
                d_split_ml[ml + 1] = 0.0f;
            }
        }
        return;
    }

    const f32 qk_scale = rsqrtf(128.0f);
    __shared__ f32 s_Q[GQA][128];
    __shared__ int s_Q_i8[GQA][32];
    __shared__ f32 s_Q_sc[GQA];
    __shared__ f32 s_R[GQA][16];
    for (u32 idx = tid; idx < GQA * 128; idx += 128) {
        u32 g = idx >> 7, d = idx & 127;
        s_Q[g][d] = __bfloat162float(q[(static_cast<size_t>(m) * n_q_heads + qh_base + g) * 128 + d]) * qk_scale;
    }
    if (tid < GQA * 16) {
        u32 g = tid >> 4, j = tid & 15;
        s_R[g][j] = __bfloat162float(r[(static_cast<size_t>(m) * n_q_heads + qh_base + g) * 16 + j]);
    }
    __syncthreads();
    if constexpr (INT8_KV) {
        // Warps 0..2 quantize Q heads 0..2 into packed int32_t (4x int8) for PTX __dp4a
        if (wid < GQA) {
            f32 q0 = s_Q[wid][lane * 4 + 0], q1 = s_Q[wid][lane * 4 + 1];
            f32 q2 = s_Q[wid][lane * 4 + 2], q3 = s_Q[wid][lane * 4 + 3];
            f32 am = fmaxf(fmaxf(fabsf(q0), fabsf(q1)), fmaxf(fabsf(q2), fabsf(q3)));
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, off));
            f32 qsc = fmaxf(am / 127.0f, 1e-8f);
            f32 iq = 1.0f / qsc;
            if (lane == 0) s_Q_sc[wid] = qsc;
            int i0 = max(-127, min(127, __float2int_rn(q0 * iq)));
            int i1 = max(-127, min(127, __float2int_rn(q1 * iq)));
            int i2 = max(-127, min(127, __float2int_rn(q2 * iq)));
            int i3 = max(-127, min(127, __float2int_rn(q3 * iq)));
            s_Q_i8[wid][lane] = (i0 & 0xFF) | ((i1 & 0xFF) << 8) | ((i2 & 0xFF) << 16) | ((i3 & 0xFF) << 24);
        }
        __syncthreads();
    }

    f32 m_i[GQA] = {-1e30f, -1e30f, -1e30f};
    f32 l_i[GQA] = {0.0f, 0.0f, 0.0f};
    f32 o_reg[GQA][4] = {};

    for (u32 t = t_start + wid; t < t_end; t += 4) {
        size_t scale_off;
        if (is_local_ring) {
            scale_off = static_cast<size_t>(kvh) * swa_window + (t & (swa_window - 1u));
        } else {
            u32 blk = t / block_size, slot = t - blk * block_size;
            i32 phys = d_block_table[blk];
            scale_off = (static_cast<size_t>(phys) * n_kv_heads + kvh) * block_size + slot;
        }
        size_t kv_off = scale_off * 128;

        f32 k_reg[4], v_reg[4];
        int kw32 = 0;
        f32 k_sc = 1.0f, v_sc = 1.0f;
        if constexpr (INT8_KV) {
            k_sc = __ldg(k_scales + scale_off);
            v_sc = __ldg(v_scales + scale_off);
            kw32 = __ldg(reinterpret_cast<const int*>(static_cast<const i8*>(k_cache_raw) + kv_off) + lane);
            char4 vw = __ldg(reinterpret_cast<const char4*>(static_cast<const i8*>(v_cache_raw) + kv_off) + lane);
            v_reg[0] = vw.x; v_reg[1] = vw.y; v_reg[2] = vw.z; v_reg[3] = vw.w;
        } else {
            uint2 ku = __ldg(reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(k_cache_raw) + kv_off) + lane);
            uint2 vu = __ldg(reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(v_cache_raw) + kv_off) + lane);
            const auto* k2 = reinterpret_cast<const __nv_bfloat162*>(&ku);
            const auto* v2 = reinterpret_cast<const __nv_bfloat162*>(&vu);
            float2 fk0 = __bfloat1622float2(k2[0]), fk1 = __bfloat1622float2(k2[1]);
            float2 fv0 = __bfloat1622float2(v2[0]), fv1 = __bfloat1622float2(v2[1]);
            k_reg[0] = fk0.x; k_reg[1] = fk0.y; k_reg[2] = fk1.x; k_reg[3] = fk1.y;
            v_reg[0] = fv0.x; v_reg[1] = fv0.y; v_reg[2] = fv1.x; v_reg[3] = fv1.y;
        }

        u32 dist = pos - t;
        f32 rp = (dist < 1024u && lane < 16) ? __bfloat162float(__ldg(rel_proj_T + static_cast<size_t>(dist) * 16 + lane)) : 0.0f;

        #pragma unroll
        for (int g = 0; g < GQA; ++g) {
            f32 dot;
            if constexpr (INT8_KV) {
                int idot = __dp4a(s_Q_i8[g][lane], kw32, 0);
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) idot += __shfl_xor_sync(0xFFFFFFFFu, idot, off);
                dot = static_cast<f32>(idot) * (s_Q_sc[g] * k_sc);
            } else {
                dot = fmaf(s_Q[g][lane * 4 + 0], k_reg[0],
                      fmaf(s_Q[g][lane * 4 + 1], k_reg[1],
                      fmaf(s_Q[g][lane * 4 + 2], k_reg[2], s_Q[g][lane * 4 + 3] * k_reg[3])));
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
            }
            if (dist < 1024u) {
                f32 r_part = (lane < 16) ? (s_R[g][lane] * rp) : 0.0f;
                #pragma unroll
                for (int off = 8; off > 0; off >>= 1) r_part += __shfl_xor_sync(0xFFFFFFFFu, r_part, off);
                dot += __shfl_sync(0xFFFFFFFFu, r_part, 0);
            }
            f32 new_m = fmaxf(m_i[g], dot);
            f32 ep = __expf(m_i[g] - new_m);
            f32 ec = __expf(dot - new_m);
            l_i[g] = l_i[g] * ep + ec;
            f32 w_v = INT8_KV ? (ec * v_sc) : ec;
            #pragma unroll
            for (int e = 0; e < 4; ++e) o_reg[g][e] = fmaf(w_v, v_reg[e], o_reg[g][e] * ep);
            m_i[g] = new_m;
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
            size_t ml_base = ((static_cast<size_t>(m) * n_q_heads + qh) * num_splits + sp) * 2;
            size_t o_base  = ((static_cast<size_t>(m) * n_q_heads + qh) * num_splits + sp) * 128;
            f32 gm = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
            f32 gl = 0.0f, ws[4];
            #pragma unroll
            for (int w = 0; w < 4; ++w) { ws[w] = (sm_l[w] > 0.0f) ? __expf(sm_m[w] - gm) : 0.0f; gl += sm_l[w] * ws[w]; }
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                u32 d = lane * 4 + e;
                d_split_o[o_base + d] = sm_o[0][d] * ws[0] + sm_o[1][d] * ws[1] + sm_o[2][d] * ws[2] + sm_o[3][d] * ws[3];
            }
            if (lane == 0) { d_split_ml[ml_base + 0] = gm; d_split_ml[ml_base + 1] = gl; }
        }
        __syncthreads();
    }
}

__global__ void inkling_rel_attn_reduce_gate_sink_kernel(
    const f32* __restrict__ d_split_o,
    const f32* __restrict__ d_split_ml,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ sink,
    __nv_bfloat16* __restrict__ attn_out,
    u32 n_q_heads, u32 num_splits) {
    const u32 qh = blockIdx.x;
    const u32 m  = blockIdx.y;
    const u32 d  = threadIdx.x; // 0..127
    size_t ml_head = (static_cast<size_t>(m) * n_q_heads + qh) * num_splits * 2;
    size_t o_head  = (static_cast<size_t>(m) * n_q_heads + qh) * num_splits * 128;

    f32 gm = -1e30f;
    for (u32 sp = 0; sp < num_splits; ++sp) {
        if (d_split_ml[ml_head + sp * 2 + 1] > 0.0f) gm = fmaxf(gm, d_split_ml[ml_head + sp * 2 + 0]);
    }
    f32 gl = 0.0f, acc = 0.0f;
    for (u32 sp = 0; sp < num_splits; ++sp) {
        f32 sl = d_split_ml[ml_head + sp * 2 + 1];
        if (sl > 0.0f) {
            f32 w = __expf(d_split_ml[ml_head + sp * 2 + 0] - gm);
            gl += sl * w;
            acc = fmaf(d_split_o[o_head + static_cast<size_t>(sp) * 128 + d], w, acc);
        }
    }
    f32 o_val = (gl > 0.0f) ? (acc / gl) : 0.0f;
    f32 lse   = (gl > 0.0f) ? (gm + logf(gl)) : -1e30f;
    size_t o_off = (static_cast<size_t>(m) * n_q_heads + qh) * 128 + d;
    o_val = o_val * sigmoid_f32(__bfloat162float(gate[o_off])) * sigmoid_f32(lse - __bfloat162float(sink[static_cast<size_t>(m) * n_q_heads + qh]));
    attn_out[o_off] = __float2bfloat16_rn(o_val);
}

// 8g. 16x16x16 WMMA Tensor-Core Relative-Bias Prefill Attention (3-4x faster pp1024/pp4096/pp16k)
template <bool INT8_KV>
__global__ void inkling_rel_attn_prefill_wmma_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ r,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ sink,
    const __nv_bfloat16* __restrict__ rel_proj_T,
    const void* __restrict__ k_cache_raw,
    const void* __restrict__ v_cache_raw,
    const f32* __restrict__ k_scales,
    const f32* __restrict__ v_scales,
    const i32* __restrict__ d_block_table,
    __nv_bfloat16* __restrict__ attn_out,
    u32 start_pos, u32 M, u32 n_q_heads, u32 n_kv_heads, u32 block_size,
    bool is_local_ring, u32 swa_window) {
    using namespace nvcuda;
    const u32 q_tile = blockIdx.x * 64;
    const u32 qh     = blockIdx.y;
    const u32 tid    = threadIdx.x; // 0..127
    const u32 wid    = tid >> 5;    // 0..3
    const u32 lane   = tid & 31;
    const u32 gqa    = n_q_heads / n_kv_heads;
    const u32 kvh    = qh / gqa;
    const f32 qk_scale = rsqrtf(128.0f);

    __shared__ __nv_bfloat16 sm_Q[64][128];   // 16 KB
    __shared__ f32           sm_R[64][16];    // 4 KB
    __shared__ __nv_bfloat16 sm_K[16][128];   // 4 KB
    __shared__ __nv_bfloat16 sm_V[16][128];   // 4 KB
    __shared__ f32           sm_S[4][16][16]; // 4 KB
    __shared__ __nv_bfloat16 sm_P[4][16][16]; // 2 KB
    __shared__ f32           sm_m[4][16];
    __shared__ f32           sm_l[4][16];
    __shared__ f32           sm_ep[4][16];

    for (u32 idx = tid; idx < 64 * 128; idx += 128) {
        u32 qr = idx >> 7, d = idx & 127;
        u32 m = q_tile + qr;
        f32 v = (m < M) ? (__bfloat162float(q[(static_cast<size_t>(m) * n_q_heads + qh) * 128 + d]) * qk_scale) : 0.0f;
        sm_Q[qr][d] = __float2bfloat16_rn(v);
    }
    for (u32 idx = tid; idx < 64 * 16; idx += 128) {
        u32 qr = idx >> 4, j = idx & 15;
        u32 m = q_tile + qr;
        sm_R[qr][j] = (m < M) ? __bfloat162float(r[(static_cast<size_t>(m) * n_q_heads + qh) * 16 + j]) : 0.0f;
    }
    if (lane < 16) { sm_m[wid][lane] = -1e30f; sm_l[wid][lane] = 0.0f; }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, f32> o_frag[8];
    #pragma unroll
    for (int d = 0; d < 8; ++d) wmma::fill_fragment(o_frag[d], 0.0f);

    const u32 block_pos_min = start_pos + q_tile;
    const u32 block_pos_max = start_pos + min(M, q_tile + 64) - 1;
    const u32 kv_begin = (is_local_ring && block_pos_min >= swa_window) ? (block_pos_min - swa_window + 1) : 0;
    const u32 kv_end   = block_pos_max + 1;

    for (u32 kv0 = kv_begin; kv0 < kv_end; kv0 += 16) {
        for (u32 idx = tid; idx < 16 * 128; idx += 128) {
            u32 kr = idx >> 7, d = idx & 127;
            u32 t = kv0 + kr;
            if (t < kv_end) {
                size_t scale_off;
                if (is_local_ring) {
                    scale_off = static_cast<size_t>(kvh) * swa_window + (t & (swa_window - 1u));
                } else {
                    u32 blk = t / block_size, slot = t - blk * block_size;
                    scale_off = (static_cast<size_t>(d_block_table[blk]) * n_kv_heads + kvh) * block_size + slot;
                }
                size_t kv_off = scale_off * 128 + d;
                if constexpr (INT8_KV) {
                    f32 ksc = __ldg(k_scales + scale_off);
                    f32 vsc = __ldg(v_scales + scale_off);
                    sm_K[kr][d] = __float2bfloat16_rn(static_cast<f32>(__ldg(static_cast<const i8*>(k_cache_raw) + kv_off)) * ksc);
                    sm_V[kr][d] = __float2bfloat16_rn(static_cast<f32>(__ldg(static_cast<const i8*>(v_cache_raw) + kv_off)) * vsc);
                } else {
                    sm_K[kr][d] = __ldg(static_cast<const __nv_bfloat16*>(k_cache_raw) + kv_off);
                    sm_V[kr][d] = __ldg(static_cast<const __nv_bfloat16*>(v_cache_raw) + kv_off);
                }
            } else {
                sm_K[kr][d] = __float2bfloat16_rn(0.0f);
                sm_V[kr][d] = __float2bfloat16_rn(0.0f);
            }
        }
        __syncthreads();

        wmma::fragment<wmma::accumulator, 16, 16, 16, f32> s_frag;
        wmma::fill_fragment(s_frag, 0.0f);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kb;
            wmma::load_matrix_sync(qa, &sm_Q[wid * 16][k * 16], 128);
            wmma::load_matrix_sync(kb, &sm_K[0][k * 16], 128);
            wmma::mma_sync(s_frag, qa, kb, s_frag);
        }
        wmma::store_matrix_sync(&sm_S[wid][0][0], s_frag, 16, wmma::mem_row_major);
        __syncwarp();

        if (lane < 16) {
            u32 qr = wid * 16 + lane;
            u32 m  = q_tile + qr;
            u32 q_pos = start_pos + m;
            f32 row_max = sm_m[wid][lane];
            f32 scores[16];
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                u32 t = kv0 + c;
                if (m >= M || t > q_pos || (is_local_ring && q_pos - t >= swa_window)) {
                    scores[c] = -1e30f;
                } else {
                    f32 s = sm_S[wid][lane][c];
                    u32 dist = q_pos - t;
                    if (dist < 1024u) {
                        const __nv_bfloat16* rp = rel_proj_T + static_cast<size_t>(dist) * 16;
                        f32 rb = 0.0f;
                        #pragma unroll
                        for (int j = 0; j < 16; ++j) rb = fmaf(sm_R[qr][j], __bfloat162float(__ldg(rp + j)), rb);
                        s += rb;
                    }
                    scores[c] = s;
                    if (s > row_max) row_max = s;
                }
            }
            f32 old_m = sm_m[wid][lane];
            f32 ep = (sm_l[wid][lane] > 0.0f && row_max > -1e20f) ? __expf(old_m - row_max) : 0.0f;
            f32 new_l = sm_l[wid][lane] * ep;
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                f32 p = (scores[c] > -1e20f) ? __expf(scores[c] - row_max) : 0.0f;
                new_l += p;
                sm_P[wid][lane][c] = __float2bfloat16_rn(p);
            }
            sm_m[wid][lane]  = row_max;
            sm_l[wid][lane]  = new_l;
            sm_ep[wid][lane] = ep;
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pa;
        wmma::load_matrix_sync(pa, &sm_P[wid][0][0], 16);
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            wmma::store_matrix_sync(&sm_S[wid][0][0], o_frag[d], 16, wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int idx = lane; idx < 256; idx += 32) {
                u32 rr = idx >> 4, cc = idx & 15;
                sm_S[wid][rr][cc] *= sm_ep[wid][rr];
            }
            __syncwarp();
            wmma::load_matrix_sync(o_frag[d], &sm_S[wid][0][0], 16, wmma::mem_row_major);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vb;
            wmma::load_matrix_sync(vb, &sm_V[0][d * 16], 128);
            wmma::mma_sync(o_frag[d], pa, vb, o_frag[d]);
        }
        __syncthreads();
    }

    const u32 w_q0 = wid * 16;
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        wmma::store_matrix_sync(&sm_S[wid][0][0], o_frag[d], 16, wmma::mem_row_major);
        __syncwarp();
        for (u32 idx = lane; idx < 256; idx += 32) {
            u32 rr = idx >> 4, cc = idx & 15;
            u32 m = q_tile + w_q0 + rr;
            if (m < M) {
                f32 lg = sm_l[wid][rr], mg = sm_m[wid][rr];
                f32 o_val = (lg > 0.0f) ? (sm_S[wid][rr][cc] / lg) : 0.0f;
                f32 lse = (lg > 0.0f) ? (mg + logf(lg)) : -1e30f;
                size_t o_off = (static_cast<size_t>(m) * n_q_heads + qh) * 128 + d * 16 + cc;
                f32 g_val = __bfloat162float(gate[o_off]);
                f32 s_val = __bfloat162float(sink[static_cast<size_t>(m) * n_q_heads + qh]);
                o_val = o_val * sigmoid_f32(g_val) * sigmoid_f32(lse - s_val);
                attn_out[o_off] = __float2bfloat16_rn(o_val);
            }
        }
        __syncwarp();
    }
}

} // namespace

void launch_inkling_rel_attention(const __nv_bfloat16* q, const __nv_bfloat16* r,
                                  const __nv_bfloat16* gate, const __nv_bfloat16* sink,
                                  const __nv_bfloat16* rel_proj_T,
                                  const void* k_cache_raw, const void* v_cache_raw,
                                  const f32* k_scales, const f32* v_scales,
                                  const i32* d_block_table,
                                  __nv_bfloat16* attn_out,
                                  const i32* d_pos, u32 start_pos, u32 M,
                                  u32 n_q_heads, u32 n_kv_heads, u32 block_size,
                                  bool is_local_ring, u32 swa_window, bool int8_kv,
                                  cudaStream_t stream) {
    dim3 grid(n_q_heads, M);
    if (int8_kv) {
        inkling_rel_attn_kernel<true><<<grid, 256, 0, stream>>>(
            q, r, gate, sink, rel_proj_T, k_cache_raw, v_cache_raw, k_scales, v_scales,
            d_block_table, attn_out, d_pos, start_pos, n_q_heads, n_kv_heads, block_size,
            is_local_ring, swa_window);
    } else {
        inkling_rel_attn_kernel<false><<<grid, 256, 0, stream>>>(
            q, r, gate, sink, rel_proj_T, k_cache_raw, v_cache_raw, k_scales, v_scales,
            d_block_table, attn_out, d_pos, start_pos, n_q_heads, n_kv_heads, block_size,
            is_local_ring, swa_window);
    }
}

void launch_inkling_rel_attention_splitk(const __nv_bfloat16* q, const __nv_bfloat16* r,
                                         const __nv_bfloat16* gate, const __nv_bfloat16* sink,
                                         const __nv_bfloat16* rel_proj_T,
                                         const void* k_cache_raw, const void* v_cache_raw,
                                         const f32* k_scales, const f32* v_scales,
                                         const i32* d_block_table,
                                         __nv_bfloat16* attn_out,
                                         f32* d_split_o, f32* d_split_ml,
                                         const i32* d_pos, u32 start_pos, u32 M,
                                         u32 n_q_heads, u32 n_kv_heads, u32 block_size,
                                         u32 num_splits, bool is_local_ring, u32 swa_window, bool int8_kv,
                                         cudaStream_t stream) {
    dim3 g1(n_kv_heads, num_splits, M);
    dim3 g2(n_q_heads, M);
    if (int8_kv) {
        inkling_rel_attn_gqa_splitk_kernel<true><<<g1, 128, 0, stream>>>(
            q, r, rel_proj_T, k_cache_raw, v_cache_raw, k_scales, v_scales, d_block_table,
            d_split_o, d_split_ml, d_pos, start_pos, n_q_heads, n_kv_heads, block_size,
            num_splits, is_local_ring, swa_window);
    } else {
        inkling_rel_attn_gqa_splitk_kernel<false><<<g1, 128, 0, stream>>>(
            q, r, rel_proj_T, k_cache_raw, v_cache_raw, k_scales, v_scales, d_block_table,
            d_split_o, d_split_ml, d_pos, start_pos, n_q_heads, n_kv_heads, block_size,
            num_splits, is_local_ring, swa_window);
    }
    inkling_rel_attn_reduce_gate_sink_kernel<<<g2, 128, 0, stream>>>(
        d_split_o, d_split_ml, gate, sink, attn_out, n_q_heads, num_splits);
}

void launch_inkling_rel_attention_prefill_wmma(
    const __nv_bfloat16* q, const __nv_bfloat16* r,
    const __nv_bfloat16* gate, const __nv_bfloat16* sink,
    const __nv_bfloat16* rel_proj_T,
    const void* k_cache_raw, const void* v_cache_raw,
    const f32* k_scales, const f32* v_scales,
    const i32* d_block_table, __nv_bfloat16* attn_out,
    u32 start_pos, u32 M, u32 n_q_heads, u32 n_kv_heads, u32 block_size,
    bool is_local_ring, u32 swa_window, bool int8_kv, cudaStream_t stream) {
    dim3 grid((M + 63) / 64, n_q_heads);
    if (int8_kv) {
        inkling_rel_attn_prefill_wmma_kernel<true><<<grid, 128, 0, stream>>>(
            q, r, gate, sink, rel_proj_T, k_cache_raw, v_cache_raw, k_scales, v_scales,
            d_block_table, attn_out, start_pos, M, n_q_heads, n_kv_heads, block_size,
            is_local_ring, swa_window);
    } else {
        inkling_rel_attn_prefill_wmma_kernel<false><<<grid, 128, 0, stream>>>(
            q, r, gate, sink, rel_proj_T, k_cache_raw, v_cache_raw, k_scales, v_scales,
            d_block_table, attn_out, start_pos, M, n_q_heads, n_kv_heads, block_size,
            is_local_ring, swa_window);
    }
}


} // namespace flint::cuda
