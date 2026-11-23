#include "kernels.hpp"
#include <mma.h>

namespace flint::cuda {

namespace {

// ============================================================================
// 1. GQA-Grouped Vectorized Split-K FlashDecoding (BF16 or FP8 E4M3 KV Cache)
// ============================================================================
// Grid: (n_kv_heads, num_splits, M)
// Block: 128 threads (4 warps)
// - BF16 KV: coalesced 64-bit load (4 BF16/thread = 256 B/warp), reused across all 7 GQA heads (7x less traffic)
// - FP8 KV: coalesced 32-bit load (4 FP8/thread = 128 B/warp), reused across all 7 GQA heads (14x less traffic)
template <int HEAD_DIM, int MAX_GQA, bool FP8_KV>
__global__ void __launch_bounds__(128, 3) flash_decode_gqa_splitk_kernel(
    const __nv_bfloat16* __restrict__ q,
    const void* __restrict__ k_cache_raw,
    const void* __restrict__ v_cache_raw,
    const f32* __restrict__ k_scales,
    const f32* __restrict__ v_scales,
    const i32* __restrict__ d_block_table,
    f32* __restrict__ d_split_o,
    f32* __restrict__ d_split_ml,
    const i32* __restrict__ d_pos,
    u32 start_pos,
    u32 n_q_heads, u32 n_kv_heads,
    u32 block_size, u32 num_splits) {
    constexpr int ELEMS = HEAD_DIM / 32; // 2 for 64, 4 for 128
    const u32 kvh  = blockIdx.x;
    const u32 sp   = blockIdx.y;
    const u32 m    = blockIdx.z;
    const u32 tid  = threadIdx.x;
    const u32 wid  = tid >> 5;
    const u32 lane = tid & 31;

    const u32 gqa = n_q_heads / n_kv_heads;
    const u32 qh_base = kvh * gqa;
    const u32 seq_len = (d_pos ? static_cast<u32>(*d_pos) : start_pos) + m + 1;

    const u32 tokens_per_split = (seq_len + num_splits - 1) / num_splits;
    const u32 t_start = sp * tokens_per_split;
    const u32 t_end = min(t_start + tokens_per_split, seq_len);

    if (t_start >= t_end) {
        if (tid < gqa) {
            u32 qh = qh_base + tid;
            size_t ml_base = ((static_cast<size_t>(m) * n_q_heads + qh) * num_splits + sp) * 2;
            d_split_ml[ml_base + 0] = -1e30f;
            d_split_ml[ml_base + 1] = 0.0f;
        }
        return;
    }

    const f32 scale = rsqrtf(static_cast<f32>(HEAD_DIM));

    __shared__ f32 s_Q[MAX_GQA][HEAD_DIM];
    __shared__ i32 s_Qi8[MAX_GQA][32]; // Packed 4x int8_t per lane for hardware PTX __dp4a
    __shared__ f32 s_Qsc[MAX_GQA];
    for (u32 idx = tid; idx < gqa * HEAD_DIM; idx += blockDim.x) {
        u32 g = idx / HEAD_DIM;
        u32 d = idx - g * HEAD_DIM;
        const __nv_bfloat16* qp = q + (static_cast<size_t>(m) * n_q_heads + qh_base + g) * HEAD_DIM;
        s_Q[g][d] = __bfloat162float(qp[d]) * scale;
    }
    __syncthreads();

    // Hardware PTX __dp4a quantization of Q once per block (when FP8_KV/INT8 && HEAD_DIM==128)
    if constexpr (FP8_KV && HEAD_DIM == 128) {
        for (u32 g = wid; g < gqa; g += 4) {
            f32 q0 = s_Q[g][lane * 4 + 0], q1 = s_Q[g][lane * 4 + 1];
            f32 q2 = s_Q[g][lane * 4 + 2], q3 = s_Q[g][lane * 4 + 3];
            f32 am = fmaxf(fmaxf(fabsf(q0), fabsf(q1)), fmaxf(fabsf(q2), fabsf(q3)));
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, off));
            f32 qsc = fmaxf(am / 127.0f, 1e-8f);
            f32 qinv = 1.0f / qsc;
            if (lane == 0) s_Qsc[g] = qsc;
            auto qi = [](f32 x) -> u32 {
                int v = max(-127, min(127, __float2int_rn(x)));
                return static_cast<u32>(static_cast<i8>(v)) & 0xFFu;
            };
            s_Qi8[g][lane] = static_cast<i32>(qi(q0 * qinv) | (qi(q1 * qinv) << 8) | (qi(q2 * qinv) << 16) | (qi(q3 * qinv) << 24));
        }
        __syncthreads();
    }

    f32 m_i[MAX_GQA];
    f32 l_i[MAX_GQA];
    f32 o_reg[MAX_GQA][ELEMS];
    #pragma unroll
    for (int g = 0; g < MAX_GQA; ++g) {
        m_i[g] = -1e30f;
        l_i[g] = 0.0f;
        #pragma unroll
        for (int e = 0; e < ELEMS; ++e) o_reg[g][e] = 0.0f;
    }

    for (u32 t = t_start + wid; t < t_end; t += 4) {
        u32 blk = t / block_size;
        u32 slot = t - blk * block_size;
        i32 phys = d_block_table[blk];
        size_t scale_off = (static_cast<size_t>(phys) * n_kv_heads + kvh) * block_size + slot;
        size_t kv_off = scale_off * HEAD_DIM;

        f32 k_reg[ELEMS];
        f32 v_reg[ELEMS];
        i32 k_i8_pack = 0;
        f32 k_sc = 1.0f, v_sc = 1.0f;

        if constexpr (FP8_KV) {
            k_sc = __ldg(k_scales + scale_off);
            v_sc = __ldg(v_scales + scale_off);
            const auto* k_i8 = static_cast<const i8*>(k_cache_raw) + kv_off;
            const auto* v_i8 = static_cast<const i8*>(v_cache_raw) + kv_off;
            if constexpr (ELEMS == 4) {
                // Single 32-bit coalesced load = 4 INT8 elements -> directly feeds hardware PTX __dp4a
                k_i8_pack = __ldg(reinterpret_cast<const i32*>(k_i8) + lane);
                char4 vw = __ldg(reinterpret_cast<const char4*>(v_i8) + lane);
                v_reg[0] = static_cast<f32>(vw.x); v_reg[1] = static_cast<f32>(vw.y);
                v_reg[2] = static_cast<f32>(vw.z); v_reg[3] = static_cast<f32>(vw.w);
            } else {
                char2 kw = __ldg(reinterpret_cast<const char2*>(k_i8) + lane);
                char2 vw = __ldg(reinterpret_cast<const char2*>(v_i8) + lane);
                k_reg[0] = static_cast<f32>(kw.x); k_reg[1] = static_cast<f32>(kw.y);
                v_reg[0] = static_cast<f32>(vw.x); v_reg[1] = static_cast<f32>(vw.y);
            }
        } else {
            const auto* k_bf = static_cast<const __nv_bfloat16*>(k_cache_raw) + kv_off;
            const auto* v_bf = static_cast<const __nv_bfloat16*>(v_cache_raw) + kv_off;
            if constexpr (ELEMS == 4) {
                uint2 ku = __ldg(reinterpret_cast<const uint2*>(k_bf) + lane);
                uint2 vu = __ldg(reinterpret_cast<const uint2*>(v_bf) + lane);
                const auto* kb = reinterpret_cast<const __nv_bfloat162*>(&ku);
                const auto* vb = reinterpret_cast<const __nv_bfloat162*>(&vu);
                float2 fk0 = __bfloat1622float2(kb[0]);
                float2 fk1 = __bfloat1622float2(kb[1]);
                float2 fv0 = __bfloat1622float2(vb[0]);
                float2 fv1 = __bfloat1622float2(vb[1]);
                k_reg[0] = fk0.x; k_reg[1] = fk0.y; k_reg[2] = fk1.x; k_reg[3] = fk1.y;
                v_reg[0] = fv0.x; v_reg[1] = fv0.y; v_reg[2] = fv1.x; v_reg[3] = fv1.y;
            } else {
                __nv_bfloat162 kb = __ldg(reinterpret_cast<const __nv_bfloat162*>(k_bf) + lane);
                __nv_bfloat162 vb = __ldg(reinterpret_cast<const __nv_bfloat162*>(v_bf) + lane);
                float2 fk = __bfloat1622float2(kb);
                float2 fv = __bfloat1622float2(vb);
                k_reg[0] = fk.x; k_reg[1] = fk.y;
                v_reg[0] = fv.x; v_reg[1] = fv.y;
            }
        }

        #pragma unroll
        for (int g = 0; g < MAX_GQA; ++g) {
            if (g < static_cast<int>(gqa)) {
                f32 dot = 0.0f;
                if constexpr (FP8_KV && HEAD_DIM == 128) {
                    // 1-instruction hardware PTX dp4a.s32.s32 4-way INT8 dot product
                    i32 idot = __dp4a(s_Qi8[g][lane], k_i8_pack, 0);
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) idot += __shfl_xor_sync(0xFFFFFFFFu, idot, off);
                    dot = static_cast<f32>(idot) * (s_Qsc[g] * k_sc);
                } else {
                    #pragma unroll
                    for (int e = 0; e < ELEMS; ++e) {
                        dot = fmaf(s_Q[g][lane * ELEMS + e], k_reg[e], dot);
                    }
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
                    }
                    if constexpr (FP8_KV) dot *= k_sc;
                }
                f32 new_m = fmaxf(m_i[g], dot);
                f32 exp_prev = __expf(m_i[g] - new_m);
                f32 exp_cur  = __expf(dot - new_m);
                l_i[g] = l_i[g] * exp_prev + exp_cur;
                f32 w_v = FP8_KV ? (exp_cur * v_sc) : exp_cur;
                #pragma unroll
                for (int e = 0; e < ELEMS; ++e) {
                    o_reg[g][e] = fmaf(w_v, v_reg[e], o_reg[g][e] * exp_prev);
                }
                m_i[g] = new_m;
            }
        }
    }

    __shared__ f32 s_m[4];
    __shared__ f32 s_l[4];
    __shared__ f32 s_o[4][HEAD_DIM];

    for (u32 g = 0; g < gqa; ++g) {
        if (lane == 0) {
            s_m[wid] = m_i[g];
            s_l[wid] = l_i[g];
        }
        #pragma unroll
        for (int e = 0; e < ELEMS; ++e) {
            s_o[wid][lane * ELEMS + e] = o_reg[g][e];
        }
        __syncthreads();

        if (wid == 0) {
            u32 qh = qh_base + g;
            size_t ml_base = ((static_cast<size_t>(m) * n_q_heads + qh) * num_splits + sp) * 2;
            size_t o_base  = ((static_cast<size_t>(m) * n_q_heads + qh) * num_splits + sp) * HEAD_DIM;

            f32 gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
            f32 gl = 0.0f;
            f32 w_scale[4];
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                w_scale[w] = (s_l[w] > 0.0f) ? expf(s_m[w] - gm) : 0.0f;
                gl += s_l[w] * w_scale[w];
            }
            #pragma unroll
            for (int e = 0; e < ELEMS; ++e) {
                u32 d = lane * ELEMS + e;
                f32 sum = 0.0f;
                #pragma unroll
                for (int w = 0; w < 4; ++w) sum += s_o[w][d] * w_scale[w];
                d_split_o[o_base + d] = sum;
            }
            if (lane == 0) {
                d_split_ml[ml_base + 0] = gm;
                d_split_ml[ml_base + 1] = gl;
            }
        }
        __syncthreads();
    }
}

// Single-Kernel Fused Short-Seq FlashDecode (seq <= 512, 1 block per (q_head, m)):
// Eliminates BOTH d_split_o/d_split_ml global traffic AND the 2nd flash_decode_reduce_kernel launch
// (28 fewer kernel launches & global barriers per token on W1).
template <int HEAD_DIM, bool FP8_KV>
__global__ void flash_decode_short_fused_kernel(
    const __nv_bfloat16* __restrict__ q,
    const void* __restrict__ k_cache_raw,
    const void* __restrict__ v_cache_raw,
    const f32* __restrict__ k_scales,
    const f32* __restrict__ v_scales,
    const i32* __restrict__ d_block_table,
    __nv_bfloat16* __restrict__ attn_out,
    const i32* __restrict__ d_pos,
    u32 start_pos,
    u32 n_q_heads, u32 n_kv_heads,
    u32 block_size) {
    constexpr int ELEMS = HEAD_DIM / 32;
    const u32 qh = blockIdx.x;
    const u32 m  = blockIdx.y;
    const u32 tid = threadIdx.x;
    const u32 wid = tid >> 5;
    const u32 lane = tid & 31;

    const u32 gqa = n_q_heads / n_kv_heads;
    const u32 kvh = qh / gqa;
    const u32 cur_pos = (d_pos ? static_cast<u32>(*d_pos) : start_pos) + m;
    const u32 kv_len = cur_pos + 1;
    const f32 scale = rsqrtf(static_cast<f32>(HEAD_DIM));

    const __nv_bfloat16* q_ptr = q + (static_cast<size_t>(m) * n_q_heads + qh) * HEAD_DIM;
    f32 q_reg[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; ++e) {
        q_reg[e] = __bfloat162float(q_ptr[lane * ELEMS + e]) * scale;
    }

    f32 m_i = -1e30f, l_i = 0.0f;
    f32 o_reg[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; ++e) o_reg[e] = 0.0f;

    for (u32 t = wid; t < kv_len; t += 4) {
        u32 blk = t / block_size;
        u32 slot = t - blk * block_size;
        i32 phys = d_block_table[blk];
        size_t scale_off = (static_cast<size_t>(phys) * n_kv_heads + kvh) * block_size + slot;
        size_t kv_off = scale_off * HEAD_DIM;

        f32 k_reg[ELEMS], v_reg[ELEMS];
        f32 k_sc = 1.0f, v_sc = 1.0f;
        if constexpr (FP8_KV) {
            k_sc = __ldg(k_scales + scale_off);
            v_sc = __ldg(v_scales + scale_off);
            const auto* k_i8 = static_cast<const i8*>(k_cache_raw) + kv_off;
            const auto* v_i8 = static_cast<const i8*>(v_cache_raw) + kv_off;
            if constexpr (ELEMS == 4) {
                char4 kw = __ldg(reinterpret_cast<const char4*>(k_i8) + lane);
                char4 vw = __ldg(reinterpret_cast<const char4*>(v_i8) + lane);
                k_reg[0] = kw.x; k_reg[1] = kw.y; k_reg[2] = kw.z; k_reg[3] = kw.w;
                v_reg[0] = vw.x; v_reg[1] = vw.y; v_reg[2] = vw.z; v_reg[3] = vw.w;
            } else {
                char2 kw = __ldg(reinterpret_cast<const char2*>(k_i8) + lane);
                char2 vw = __ldg(reinterpret_cast<const char2*>(v_i8) + lane);
                k_reg[0] = kw.x; k_reg[1] = kw.y; v_reg[0] = vw.x; v_reg[1] = vw.y;
            }
        } else {
            const auto* k_bf = static_cast<const __nv_bfloat16*>(k_cache_raw) + kv_off;
            const auto* v_bf = static_cast<const __nv_bfloat16*>(v_cache_raw) + kv_off;
            if constexpr (ELEMS == 4) {
                uint2 ku = __ldg(reinterpret_cast<const uint2*>(k_bf) + lane);
                uint2 vu = __ldg(reinterpret_cast<const uint2*>(v_bf) + lane);
                const auto* k2 = reinterpret_cast<const __nv_bfloat162*>(&ku);
                const auto* v2 = reinterpret_cast<const __nv_bfloat162*>(&vu);
                float2 fk0 = __bfloat1622float2(k2[0]), fk1 = __bfloat1622float2(k2[1]);
                float2 fv0 = __bfloat1622float2(v2[0]), fv1 = __bfloat1622float2(v2[1]);
                k_reg[0] = fk0.x; k_reg[1] = fk0.y; k_reg[2] = fk1.x; k_reg[3] = fk1.y;
                v_reg[0] = fv0.x; v_reg[1] = fv0.y; v_reg[2] = fv1.x; v_reg[3] = fv1.y;
            } else {
                float2 fk = __bfloat1622float2(__ldg(reinterpret_cast<const __nv_bfloat162*>(k_bf) + lane));
                float2 fv = __bfloat1622float2(__ldg(reinterpret_cast<const __nv_bfloat162*>(v_bf) + lane));
                k_reg[0] = fk.x; k_reg[1] = fk.y; v_reg[0] = fv.x; v_reg[1] = fv.y;
            }
        }

        f32 dot = 0.0f;
        #pragma unroll
        for (int e = 0; e < ELEMS; ++e) dot = fmaf(q_reg[e], k_reg[e], dot);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
        if constexpr (FP8_KV) dot *= k_sc;

        f32 new_m = fmaxf(m_i, dot);
        f32 ep = __expf(m_i - new_m);
        f32 ec = __expf(dot - new_m);
        l_i = l_i * ep + ec;
        f32 w_v = FP8_KV ? (ec * v_sc) : ec;
        #pragma unroll
        for (int e = 0; e < ELEMS; ++e) o_reg[e] = fmaf(w_v, v_reg[e], o_reg[e] * ep);
        m_i = new_m;
    }

    __shared__ f32 sm_m[4];
    __shared__ f32 sm_l[4];
    __shared__ f32 sm_o[4][HEAD_DIM];
    if (lane == 0) { sm_m[wid] = m_i; sm_l[wid] = l_i; }
    #pragma unroll
    for (int e = 0; e < ELEMS; ++e) sm_o[wid][lane * ELEMS + e] = o_reg[e];
    __syncthreads();

    if (tid < HEAD_DIM) {
        f32 mg = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
        f32 w0 = (sm_l[0] > 0.0f) ? __expf(sm_m[0] - mg) : 0.0f;
        f32 w1 = (sm_l[1] > 0.0f) ? __expf(sm_m[1] - mg) : 0.0f;
        f32 w2 = (sm_l[2] > 0.0f) ? __expf(sm_m[2] - mg) : 0.0f;
        f32 w3 = (sm_l[3] > 0.0f) ? __expf(sm_m[3] - mg) : 0.0f;
        f32 lg = w0 * sm_l[0] + w1 * sm_l[1] + w2 * sm_l[2] + w3 * sm_l[3];
        f32 acc = w0 * sm_o[0][tid] + w1 * sm_o[1][tid] + w2 * sm_o[2][tid] + w3 * sm_o[3][tid];
        attn_out[(static_cast<size_t>(m) * n_q_heads + qh) * HEAD_DIM + tid] =
            __float2bfloat16_rn((lg > 0.0f) ? (acc / lg) : 0.0f);
    }
}

template <int HEAD_DIM>
__global__ void flash_decode_reduce_kernel(const f32* __restrict__ d_split_o,
                                           const f32* __restrict__ d_split_ml,
                                           __nv_bfloat16* __restrict__ attn_out,
                                           u32 n_q_heads, u32 num_splits) {
    const u32 qh = blockIdx.x;
    const u32 m  = blockIdx.y;
    const u32 d  = threadIdx.x;
    if (d >= HEAD_DIM) return;

    const size_t head_idx = static_cast<size_t>(m) * n_q_heads + qh;
    const f32* ml = d_split_ml + head_idx * num_splits * 2;
    const f32* o  = d_split_o  + head_idx * num_splits * HEAD_DIM;

    f32 gm = -1e30f;
    for (u32 s = 0; s < num_splits; ++s) {
        if (ml[s * 2 + 1] > 0.0f) gm = fmaxf(gm, ml[s * 2 + 0]);
    }
    f32 gl = 0.0f;
    f32 acc = 0.0f;
    for (u32 s = 0; s < num_splits; ++s) {
        f32 sl = ml[s * 2 + 1];
        if (sl > 0.0f) {
            f32 w = expf(ml[s * 2 + 0] - gm);
            gl += sl * w;
            acc += o[s * HEAD_DIM + d] * w;
        }
    }
    f32 val = (gl > 0.0f) ? (acc / gl) : 0.0f;
    attn_out[head_idx * HEAD_DIM + d] = __float2bfloat16_rn(val);
}

// ============================================================================
// 2. Union-Smem High-Occupancy WMMA Tensor-Core FlashAttention-2 Prefill
// ============================================================================
// Unions `s_Q` (only needed before KV loop to fill register `frag_Q`) with `s_O`,
// slashing static smem from 40.2 KiB -> 30.8 KiB (2-3 blocks/SM on SM_89/SM_90).
// Supports BOTH BF16 KV cache and FP8 E4M3 KV cache natively
template <int HEAD_DIM, bool FP8_KV>
__global__ void flash_prefill_wmma_kernel(const __nv_bfloat16* __restrict__ q,
                                          const void* __restrict__ k_cache_raw,
                                          const void* __restrict__ v_cache_raw,
                                          const f32* __restrict__ k_scales,
                                          const f32* __restrict__ v_scales,
                                          const i32* __restrict__ d_block_table,
                                          __nv_bfloat16* __restrict__ attn_out,
                                          u32 start_pos, u32 M,
                                          u32 n_q_heads, u32 n_kv_heads,
                                          u32 block_size) {
    using namespace nvcuda;
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int STRIDE = HEAD_DIM + 8;

    const u32 qh    = blockIdx.x;
    const u32 q_blk = blockIdx.y;
    const u32 q0    = q_blk * BM;
    if (q0 >= M) return;

    const u32 tid  = threadIdx.x;
    const u32 wid  = tid >> 5;
    const u32 lane = tid & 31;
    const u32 kvh  = qh / (n_q_heads / n_kv_heads);
    const f32 scale = rsqrtf(static_cast<f32>(HEAD_DIM));

    // Union s_Q (used once at start) with s_O (used in KV loop): saves 8.7 KiB smem
    __shared__ union {
        __nv_bfloat16 s_Q[BM][STRIDE];
        f32 s_O[2][16][HEAD_DIM];
    } qo_buf;
    __shared__ __nv_bfloat16 s_KV[BN][STRIDE];
    __shared__ f32 s_S[2][16][BN];
    __shared__ __nv_bfloat16 s_P[2][16][BN];
    __shared__ f32 s_m[2][16];
    __shared__ f32 s_l[2][16];

    constexpr u32 VEC_D = HEAD_DIM / 8;
    for (u32 idx = tid; idx < BM * VEC_D; idx += blockDim.x) {
        u32 r = idx / VEC_D;
        u32 c8 = idx - r * VEC_D;
        u32 q_row = q0 + r;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (q_row < M) {
            const auto* src = reinterpret_cast<const uint4*>(
                q + (static_cast<size_t>(q_row) * n_q_heads + qh) * HEAD_DIM);
            v = src[c8];
        }
        reinterpret_cast<uint4*>(qo_buf.s_Q[r])[c8] = v;
    }
    __syncthreads();

    // Load Q fragments into registers ONCE; after syncthreads, qo_buf becomes s_O
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_Q[HEAD_DIM / 16];
    #pragma unroll
    for (int kd = 0; kd < HEAD_DIM / 16; ++kd) {
        wmma::load_matrix_sync(frag_Q[kd], &qo_buf.s_Q[wid * 16][kd * 16], STRIDE);
    }
    __syncthreads();

    for (u32 idx = tid; idx < BM * HEAD_DIM; idx += blockDim.x) {
        u32 w = idx / (16 * HEAD_DIM);
        u32 rem = idx - w * (16 * HEAD_DIM);
        u32 r = rem / HEAD_DIM;
        u32 d = rem - r * HEAD_DIM;
        qo_buf.s_O[w][r][d] = 0.0f;
    }
    if (tid < BM) {
        s_m[tid >> 4][tid & 15] = -1e30f;
        s_l[tid >> 4][tid & 15] = 0.0f;
    }
    __syncthreads();

    const u32 max_q_row = min(q0 + BM - 1, M - 1);
    const u32 max_kv_len = start_pos + max_q_row + 1;

    auto load_kv_tile = [&](const void* cache_raw, const f32* scales_raw, u32 t0) {
        for (u32 idx = tid; idx < BN * VEC_D; idx += blockDim.x) {
            u32 r = idx / VEC_D;
            u32 c8 = idx - r * VEC_D;
            u32 t = t0 + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (t < max_kv_len) {
                u32 blk = t / block_size;
                u32 slot = t - blk * block_size;
                i32 phys = d_block_table[blk];
                size_t scale_off = (static_cast<size_t>(phys) * n_kv_heads + kvh) * block_size + slot;
                size_t kv_off = scale_off * HEAD_DIM;
                if constexpr (FP8_KV) {
                    f32 sc = __ldg(scales_raw + scale_off);
                    const auto* src_i8 = reinterpret_cast<const char4*>(static_cast<const i8*>(cache_raw) + kv_off);
                    char4 a = src_i8[c8 * 2];
                    char4 b = src_i8[c8 * 2 + 1];
                    auto* dst_b2 = reinterpret_cast<__nv_bfloat162*>(&v);
                    dst_b2[0] = __floats2bfloat162_rn(static_cast<f32>(a.x) * sc, static_cast<f32>(a.y) * sc);
                    dst_b2[1] = __floats2bfloat162_rn(static_cast<f32>(a.z) * sc, static_cast<f32>(a.w) * sc);
                    dst_b2[2] = __floats2bfloat162_rn(static_cast<f32>(b.x) * sc, static_cast<f32>(b.y) * sc);
                    dst_b2[3] = __floats2bfloat162_rn(static_cast<f32>(b.z) * sc, static_cast<f32>(b.w) * sc);
                } else {
                    v = reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(cache_raw) + kv_off)[c8];
                }
            }
            reinterpret_cast<uint4*>(s_KV[r])[c8] = v;
        }
    };

    for (u32 t0 = 0; t0 < max_kv_len; t0 += BN) {
        load_kv_tile(k_cache_raw, k_scales, t0);
        __syncthreads();

        #pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, f32> frag_S;
            wmma::fill_fragment(frag_S, 0.0f);
            #pragma unroll
            for (int kd = 0; kd < HEAD_DIM / 16; ++kd) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_K;
                wmma::load_matrix_sync(frag_K, &s_KV[nb * 16][kd * 16], STRIDE);
                wmma::mma_sync(frag_S, frag_Q[kd], frag_K, frag_S);
            }
            wmma::store_matrix_sync(&s_S[wid][0][nb * 16], frag_S, BN, wmma::mem_row_major);
        }
        __syncwarp();

        if (lane < 16) {
            u32 r = lane;
            u32 q_pos = start_pos + q0 + wid * 16 + r;
            f32 m_old = s_m[wid][r];
            f32 m_new = m_old;
            #pragma unroll
            for (u32 c = 0; c < BN; ++c) {
                u32 t = t0 + c;
                f32 s = (t <= q_pos && t < max_kv_len) ? (s_S[wid][r][c] * scale) : -1e30f;
                s_S[wid][r][c] = s;
                m_new = fmaxf(m_new, s);
            }
            f32 alpha = (m_old > -1e29f) ? expf(m_old - m_new) : 0.0f;
            f32 l_cur = 0.0f;
            #pragma unroll
            for (u32 c = 0; c < BN; ++c) {
                f32 s = s_S[wid][r][c];
                f32 p = (s > -1e29f) ? expf(s - m_new) : 0.0f;
                s_P[wid][r][c] = __float2bfloat16_rn(p);
                l_cur += p;
            }
            s_m[wid][r] = m_new;
            s_l[wid][r] = s_l[wid][r] * alpha + l_cur;
            s_S[wid][r][0] = alpha;
        }
        __syncwarp();

        #pragma unroll
        for (u32 r = 0; r < 16; ++r) {
            f32 alpha = s_S[wid][r][0];
            for (u32 d = lane; d < HEAD_DIM; d += 32) {
                qo_buf.s_O[wid][r][d] *= alpha;
            }
        }
        __syncthreads();

        load_kv_tile(v_cache_raw, v_scales, t0);
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_P0, frag_P1;
        wmma::load_matrix_sync(frag_P0, &s_P[wid][0][0],  BN);
        wmma::load_matrix_sync(frag_P1, &s_P[wid][0][16], BN);

        #pragma unroll
        for (int nd = 0; nd < HEAD_DIM / 16; ++nd) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, f32> frag_O;
            wmma::load_matrix_sync(frag_O, &qo_buf.s_O[wid][0][nd * 16], HEAD_DIM, wmma::mem_row_major);

            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_V0, frag_V1;
            wmma::load_matrix_sync(frag_V0, &s_KV[0][nd * 16],  STRIDE);
            wmma::load_matrix_sync(frag_V1, &s_KV[16][nd * 16], STRIDE);

            wmma::mma_sync(frag_O, frag_P0, frag_V0, frag_O);
            wmma::mma_sync(frag_O, frag_P1, frag_V1, frag_O);

            wmma::store_matrix_sync(&qo_buf.s_O[wid][0][nd * 16], frag_O, HEAD_DIM, wmma::mem_row_major);
        }
        __syncthreads();
    }

    for (u32 idx = tid; idx < BM * HEAD_DIM; idx += blockDim.x) {
        u32 r_blk = idx / HEAD_DIM;
        u32 d = idx - r_blk * HEAD_DIM;
        u32 q_row = q0 + r_blk;
        if (q_row < M) {
            u32 w = r_blk >> 4;
            u32 r = r_blk & 15;
            f32 inv_l = (s_l[w][r] > 0.0f) ? (1.0f / s_l[w][r]) : 0.0f;
            attn_out[(static_cast<size_t>(q_row) * n_q_heads + qh) * HEAD_DIM + d] =
                __float2bfloat16_rn(qo_buf.s_O[w][r][d] * inv_l);
        }
    }
}

} // namespace

void launch_flash_decode_bf16(const __nv_bfloat16* q,
                              const __nv_bfloat16* k_cache,
                              const __nv_bfloat16* v_cache,
                              const i32* d_block_table,
                              __nv_bfloat16* attn_out,
                              f32* d_split_o,
                              f32* d_split_ml,
                              const i32* d_pos,
                              u32 start_pos,
                              u32 M, u32 n_q_heads, u32 n_kv_heads, u32 head_dim,
                              u32 block_size, u32 num_splits,
                              cudaStream_t stream) {
    if (num_splits <= 1) {
        dim3 g_short(n_q_heads, M);
        if (head_dim == 64) {
            flash_decode_short_fused_kernel<64, false><<<g_short, 128, 0, stream>>>(
                q, k_cache, v_cache, nullptr, nullptr, d_block_table, attn_out,
                d_pos, start_pos, n_q_heads, n_kv_heads, block_size);
        } else {
            flash_decode_short_fused_kernel<128, false><<<g_short, 128, 0, stream>>>(
                q, k_cache, v_cache, nullptr, nullptr, d_block_table, attn_out,
                d_pos, start_pos, n_q_heads, n_kv_heads, block_size);
        }
        return;
    }
    dim3 g1(n_kv_heads, num_splits, M);
    dim3 g2(n_q_heads, M);
    if (head_dim == 64) {
        flash_decode_gqa_splitk_kernel<64, 8, false><<<g1, 128, 0, stream>>>(
            q, k_cache, v_cache, nullptr, nullptr, d_block_table, d_split_o, d_split_ml,
            d_pos, start_pos, n_q_heads, n_kv_heads, block_size, num_splits);
        flash_decode_reduce_kernel<64><<<g2, 64, 0, stream>>>(
            d_split_o, d_split_ml, attn_out, n_q_heads, num_splits);
    } else {
        flash_decode_gqa_splitk_kernel<128, 8, false><<<g1, 128, 0, stream>>>(
            q, k_cache, v_cache, nullptr, nullptr, d_block_table, d_split_o, d_split_ml,
            d_pos, start_pos, n_q_heads, n_kv_heads, block_size, num_splits);
        flash_decode_reduce_kernel<128><<<g2, 128, 0, stream>>>(
            d_split_o, d_split_ml, attn_out, n_q_heads, num_splits);
    }
}

void launch_flash_decode_fp8kv(const __nv_bfloat16* q,
                               const u8* k_cache_fp8,
                               const u8* v_cache_fp8,
                               const f32* k_scales,
                               const f32* v_scales,
                               const i32* d_block_table,
                               __nv_bfloat16* attn_out,
                               f32* d_split_o,
                               f32* d_split_ml,
                               const i32* d_pos,
                               u32 start_pos,
                               u32 M, u32 n_q_heads, u32 n_kv_heads, u32 head_dim,
                               u32 block_size, u32 num_splits,
                               cudaStream_t stream) {
    if (num_splits <= 1) {
        dim3 g_short(n_q_heads, M);
        if (head_dim == 64) {
            flash_decode_short_fused_kernel<64, true><<<g_short, 128, 0, stream>>>(
                q, k_cache_fp8, v_cache_fp8, k_scales, v_scales, d_block_table, attn_out,
                d_pos, start_pos, n_q_heads, n_kv_heads, block_size);
        } else {
            flash_decode_short_fused_kernel<128, true><<<g_short, 128, 0, stream>>>(
                q, k_cache_fp8, v_cache_fp8, k_scales, v_scales, d_block_table, attn_out,
                d_pos, start_pos, n_q_heads, n_kv_heads, block_size);
        }
        return;
    }
    dim3 g1(n_kv_heads, num_splits, M);
    dim3 g2(n_q_heads, M);
    if (head_dim == 64) {
        flash_decode_gqa_splitk_kernel<64, 8, true><<<g1, 128, 0, stream>>>(
            q, k_cache_fp8, v_cache_fp8, k_scales, v_scales, d_block_table, d_split_o, d_split_ml,
            d_pos, start_pos, n_q_heads, n_kv_heads, block_size, num_splits);
        flash_decode_reduce_kernel<64><<<g2, 64, 0, stream>>>(
            d_split_o, d_split_ml, attn_out, n_q_heads, num_splits);
    } else {
        flash_decode_gqa_splitk_kernel<128, 8, true><<<g1, 128, 0, stream>>>(
            q, k_cache_fp8, v_cache_fp8, k_scales, v_scales, d_block_table, d_split_o, d_split_ml,
            d_pos, start_pos, n_q_heads, n_kv_heads, block_size, num_splits);
        flash_decode_reduce_kernel<128><<<g2, 128, 0, stream>>>(
            d_split_o, d_split_ml, attn_out, n_q_heads, num_splits);
    }
}

void launch_flash_prefill_bf16(const __nv_bfloat16* q,
                               const __nv_bfloat16* k_cache,
                               const __nv_bfloat16* v_cache,
                               const i32* d_block_table,
                               __nv_bfloat16* attn_out,
                               u32 start_pos, u32 M,
                               u32 n_q_heads, u32 n_kv_heads, u32 head_dim,
                               u32 block_size,
                               cudaStream_t stream) {
    dim3 grid(n_q_heads, (M + 31) / 32);
    if (head_dim == 64) {
        flash_prefill_wmma_kernel<64, false><<<grid, 64, 0, stream>>>(
            q, k_cache, v_cache, nullptr, nullptr, d_block_table, attn_out, start_pos, M,
            n_q_heads, n_kv_heads, block_size);
    } else {
        flash_prefill_wmma_kernel<128, false><<<grid, 64, 0, stream>>>(
            q, k_cache, v_cache, nullptr, nullptr, d_block_table, attn_out, start_pos, M,
            n_q_heads, n_kv_heads, block_size);
    }
}

void launch_flash_prefill_fp8kv(const __nv_bfloat16* q,
                                const u8* k_cache_fp8,
                                const u8* v_cache_fp8,
                                const f32* k_scales,
                                const f32* v_scales,
                                const i32* d_block_table,
                                __nv_bfloat16* attn_out,
                                u32 start_pos, u32 M,
                                u32 n_q_heads, u32 n_kv_heads, u32 head_dim,
                                u32 block_size,
                                cudaStream_t stream) {
    dim3 grid(n_q_heads, (M + 31) / 32);
    if (head_dim == 64) {
        flash_prefill_wmma_kernel<64, true><<<grid, 64, 0, stream>>>(
            q, k_cache_fp8, v_cache_fp8, k_scales, v_scales, d_block_table, attn_out, start_pos, M,
            n_q_heads, n_kv_heads, block_size);
    } else {
        flash_prefill_wmma_kernel<128, true><<<grid, 64, 0, stream>>>(
            q, k_cache_fp8, v_cache_fp8, k_scales, v_scales, d_block_table, attn_out, start_pos, M,
            n_q_heads, n_kv_heads, block_size);
    }
}

// ============================================================================
// Zero-Calibration Walsh-Hadamard 4-Bit KV FlashDecoding & Prefill
// Q is already WHT-rotated in rope_kv.cu; K & V are 4-bit asymmetric packed
// (64 bytes + 4 bytes scale/min per 128-dim head = 4x smaller than BF16).
// Inverse WHT is applied ONCE to attn_out in the reduce stage (exact linearity).
// ============================================================================
__device__ __forceinline__ void wht128_warp_fd(f32& v0, f32& v1, f32& v2, f32& v3, int lane) {
    f32 a0 = v0 + v1, a1 = v0 - v1, a2 = v2 + v3, a3 = v2 - v3;
    v0 = a0 + a2; v1 = a1 + a3; v2 = a0 - a2; v3 = a1 - a3;
    #pragma unroll
    for (int off = 1; off <= 16; off <<= 1) {
        f32 p0 = __shfl_xor_sync(0xFFFFFFFFu, v0, off);
        f32 p1 = __shfl_xor_sync(0xFFFFFFFFu, v1, off);
        f32 p2 = __shfl_xor_sync(0xFFFFFFFFu, v2, off);
        f32 p3 = __shfl_xor_sync(0xFFFFFFFFu, v3, off);
        bool neg = (lane & off) != 0;
        v0 = neg ? (p0 - v0) : (v0 + p0);
        v1 = neg ? (p1 - v1) : (v1 + p1);
        v2 = neg ? (p2 - v2) : (v2 + p2);
        v3 = neg ? (p3 - v3) : (v3 + p3);
    }
    const f32 inv = rsqrtf(128.0f);
    v0 *= inv; v1 *= inv; v2 *= inv; v3 *= inv;
}

template <int MAX_GQA, bool USE_WHT>
__global__ void flash_decode_gqa_splitk_wht4_kernel(
    const __nv_bfloat16* __restrict__ q,
    const u8* __restrict__ k_cache_raw,
    const u8* __restrict__ v_cache_i4,
    const __nv_bfloat162* __restrict__ k_scales_sm,
    const __nv_bfloat162* __restrict__ v_scales_sm,
    const i32* __restrict__ d_block_table,
    f32* __restrict__ d_split_o,
    f32* __restrict__ d_split_ml,
    const i32* __restrict__ d_pos,
    u32 start_pos,
    u32 n_q_heads, u32 n_kv_heads, u32 block_size, u32 num_splits) {
    const u32 kvh = blockIdx.x;
    const u32 sp  = blockIdx.y;
    const u32 m   = blockIdx.z;
    const u32 tid = threadIdx.x;
    const u32 wid = tid >> 5;
    const u32 lane = tid & 31;

    const u32 gqa = n_q_heads / n_kv_heads;
    const u32 qh_base = kvh * gqa;
    const u32 cur_pos = (d_pos ? static_cast<u32>(*d_pos) : start_pos) + m;
    const u32 kv_len  = cur_pos + 1;

    const u32 chunk   = (kv_len + num_splits - 1) / num_splits;
    const u32 t_start = sp * chunk;
    const u32 t_end   = min(kv_len, t_start + chunk);
    const f32 scale   = rsqrtf(128.0f);

    __shared__ f32 s_Q[MAX_GQA][128];
    __shared__ i32 s_Qi8[MAX_GQA][32];
    __shared__ f32 s_Qsc[MAX_GQA];
    for (u32 idx = tid; idx < gqa * 128; idx += blockDim.x) {
        u32 g = idx >> 7;
        u32 d = idx & 127;
        s_Q[g][d] = __bfloat162float(q[(static_cast<size_t>(m) * n_q_heads + qh_base + g) * 128 + d]) * scale;
    }
    __syncthreads();
    if constexpr (USE_WHT) {
        for (u32 g = wid; g < gqa; g += 4) {
            f32 q0 = s_Q[g][lane * 4 + 0], q1 = s_Q[g][lane * 4 + 1], q2 = s_Q[g][lane * 4 + 2], q3 = s_Q[g][lane * 4 + 3];
            f32 am = fmaxf(fmaxf(fabsf(q0), fabsf(q1)), fmaxf(fabsf(q2), fabsf(q3)));
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, off));
            f32 qsc = fmaxf(am / 127.0f, 1e-8f), qinv = 1.0f / qsc;
            if (lane == 0) s_Qsc[g] = qsc;
            auto qi = [](f32 x) -> u32 { return static_cast<u32>(static_cast<i8>(max(-127, min(127, __float2int_rn(x))))) & 0xFFu; };
            s_Qi8[g][lane] = static_cast<i32>(qi(q0 * qinv) | (qi(q1 * qinv) << 8) | (qi(q2 * qinv) << 16) | (qi(q3 * qinv) << 24));
        }
        __syncthreads();
    }

    f32 m_i[MAX_GQA], l_i[MAX_GQA], o_reg[MAX_GQA][4];
    #pragma unroll
    for (int g = 0; g < MAX_GQA; ++g) {
        m_i[g] = -1e30f; l_i[g] = 0.0f;
        #pragma unroll
        for (int e = 0; e < 4; ++e) o_reg[g][e] = 0.0f;
    }

    for (u32 t = t_start + wid; t < t_end; t += 4) {
        u32 blk = t / block_size;
        u32 slot = t - blk * block_size;
        i32 phys = d_block_table[blk];
        size_t scale_off = (static_cast<size_t>(phys) * n_kv_heads + kvh) * block_size + slot;
        float2 vs = __bfloat1622float2(__ldg(v_scales_sm + scale_off));
        u16 vp = __ldg(reinterpret_cast<const u16*>(v_cache_i4 + scale_off * 64) + lane);
        f32 v0 = fmaf(static_cast<f32>(vp & 0xFu), vs.x, vs.y);
        f32 v1 = fmaf(static_cast<f32>((vp >> 4) & 0xFu), vs.x, vs.y);
        f32 v2 = fmaf(static_cast<f32>((vp >> 8) & 0xFu), vs.x, vs.y);
        f32 v3 = fmaf(static_cast<f32>((vp >> 12) & 0xFu), vs.x, vs.y);

        i32 k_i8_pack = 0;
        f32 k_sc = 1.0f, k0 = 0.0f, k1 = 0.0f, k2 = 0.0f, k3 = 0.0f;
        if constexpr (USE_WHT) {
            k_sc = __ldg(reinterpret_cast<const f32*>(k_scales_sm) + scale_off);
            k_i8_pack = __ldg(reinterpret_cast<const i32*>(k_cache_raw + scale_off * 128) + lane);
        } else {
            float2 ks = __bfloat1622float2(__ldg(k_scales_sm + scale_off));
            u16 kp = __ldg(reinterpret_cast<const u16*>(k_cache_raw + scale_off * 64) + lane);
            k0 = fmaf(static_cast<f32>(kp & 0xFu), ks.x, ks.y);
            k1 = fmaf(static_cast<f32>((kp >> 4) & 0xFu), ks.x, ks.y);
            k2 = fmaf(static_cast<f32>((kp >> 8) & 0xFu), ks.x, ks.y);
            k3 = fmaf(static_cast<f32>((kp >> 12) & 0xFu), ks.x, ks.y);
        }

        #pragma unroll
        for (int g = 0; g < MAX_GQA; ++g) {
            if (g < static_cast<int>(gqa)) {
                f32 dot = 0.0f;
                if constexpr (USE_WHT) {
                    i32 idot = __dp4a(s_Qi8[g][lane], k_i8_pack, 0);
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) idot += __shfl_xor_sync(0xFFFFFFFFu, idot, off);
                    dot = static_cast<f32>(idot) * (s_Qsc[g] * k_sc);
                } else {
                    dot = fmaf(s_Q[g][lane * 4 + 0], k0,
                          fmaf(s_Q[g][lane * 4 + 1], k1,
                          fmaf(s_Q[g][lane * 4 + 2], k2, s_Q[g][lane * 4 + 3] * k3)));
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
                }
                f32 new_m = fmaxf(m_i[g], dot);
                f32 ep = __expf(m_i[g] - new_m);
                f32 ec = __expf(dot - new_m);
                l_i[g] = l_i[g] * ep + ec;
                o_reg[g][0] = fmaf(ec, v0, o_reg[g][0] * ep);
                o_reg[g][1] = fmaf(ec, v1, o_reg[g][1] * ep);
                o_reg[g][2] = fmaf(ec, v2, o_reg[g][2] * ep);
                o_reg[g][3] = fmaf(ec, v3, o_reg[g][3] * ep);
                m_i[g] = new_m;
            }
        }
    }

    __shared__ f32 s_m[4], s_l[4], s_o[4][128];
    for (u32 g = 0; g < gqa; ++g) {
        if (lane == 0) { s_m[wid] = m_i[g]; s_l[wid] = l_i[g]; }
        #pragma unroll
        for (int e = 0; e < 4; ++e) s_o[wid][lane * 4 + e] = o_reg[g][e];
        __syncthreads();
        if (wid == 0) {
            u32 qh = qh_base + g;
            size_t ml_base = ((static_cast<size_t>(m) * n_q_heads + qh) * num_splits + sp) * 2;
            size_t o_base  = ((static_cast<size_t>(m) * n_q_heads + qh) * num_splits + sp) * 128;
            f32 gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
            f32 gl = 0.0f, ws[4];
            #pragma unroll
            for (int w = 0; w < 4; ++w) { ws[w] = (s_l[w] > 0.0f) ? __expf(s_m[w] - gm) : 0.0f; gl += s_l[w] * ws[w]; }
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                u32 d = lane * 4 + e;
                d_split_o[o_base + d] = s_o[0][d] * ws[0] + s_o[1][d] * ws[1] + s_o[2][d] * ws[2] + s_o[3][d] * ws[3];
            }
            if (lane == 0) { d_split_ml[ml_base + 0] = gm; d_split_ml[ml_base + 1] = gl; }
        }
        __syncthreads();
    }
}

template <bool USE_WHT>
__global__ void flash_decode_reduce_wht_kernel(const f32* __restrict__ d_split_o,
                                               const f32* __restrict__ d_split_ml,
                                               __nv_bfloat16* __restrict__ attn_out,
                                               u32 n_q_heads, u32 num_splits) {
    const u32 qh = blockIdx.x;
    const u32 m  = blockIdx.y;
    const int lane = threadIdx.x; // 32 threads (1 warp -> inverse wht128_warp_fd directly in registers)

    f32 gm = -1e30f;
    size_t ml_head = (static_cast<size_t>(m) * n_q_heads + qh) * num_splits * 2;
    size_t o_head  = (static_cast<size_t>(m) * n_q_heads + qh) * num_splits * 128;
    for (u32 sp = 0; sp < num_splits; ++sp) {
        if (d_split_ml[ml_head + sp * 2 + 1] > 0.0f) gm = fmaxf(gm, d_split_ml[ml_head + sp * 2 + 0]);
    }
    f32 gl = 0.0f, o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    for (u32 sp = 0; sp < num_splits; ++sp) {
        f32 sl = d_split_ml[ml_head + sp * 2 + 1];
        if (sl > 0.0f) {
            f32 w = __expf(d_split_ml[ml_head + sp * 2 + 0] - gm);
            gl += sl * w;
            size_t base = o_head + static_cast<size_t>(sp) * 128 + lane * 4;
            o0 = fmaf(d_split_o[base + 0], w, o0);
            o1 = fmaf(d_split_o[base + 1], w, o1);
            o2 = fmaf(d_split_o[base + 2], w, o2);
            o3 = fmaf(d_split_o[base + 3], w, o3);
        }
    }
    f32 inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    o0 *= inv_l; o1 *= inv_l; o2 *= inv_l; o3 *= inv_l;
    if constexpr (USE_WHT) {
        wht128_warp_fd(o0, o1, o2, o3, lane);
    }
    size_t out_off = (static_cast<size_t>(m) * n_q_heads + qh) * 128 + lane * 4;
    attn_out[out_off + 0] = __float2bfloat16_rn(o0);
    attn_out[out_off + 1] = __float2bfloat16_rn(o1);
    attn_out[out_off + 2] = __float2bfloat16_rn(o2);
    attn_out[out_off + 3] = __float2bfloat16_rn(o3);
}

// Causal WHT-INT4 Prefill Attention (1 warp per (q_head, m), exact inverse WHT on V output)
template <bool USE_WHT>
__global__ void flash_prefill_wht4_kernel(
    const __nv_bfloat16* __restrict__ q,
    const u8* __restrict__ k_cache_i4,
    const u8* __restrict__ v_cache_i4,
    const __nv_bfloat162* __restrict__ k_scales_sm,
    const __nv_bfloat162* __restrict__ v_scales_sm,
    const i32* __restrict__ d_block_table,
    __nv_bfloat16* __restrict__ attn_out,
    u32 start_pos, u32 n_q_heads, u32 n_kv_heads, u32 block_size) {
    const u32 qh = blockIdx.x;
    const u32 m  = blockIdx.y;
    const int lane = threadIdx.x; // 32 threads
    const u32 kvh = qh / (n_q_heads / n_kv_heads);
    const u32 kv_len = start_pos + m + 1;
    const f32 scale = rsqrtf(128.0f);

    const __nv_bfloat16* qp = q + (static_cast<size_t>(m) * n_q_heads + qh) * 128 + lane * 4;
    f32 q0 = __bfloat162float(qp[0]) * scale, q1 = __bfloat162float(qp[1]) * scale;
    f32 q2 = __bfloat162float(qp[2]) * scale, q3 = __bfloat162float(qp[3]) * scale;

    f32 m_i = -1e30f, l_i = 0.0f, o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    for (u32 t = 0; t < kv_len; ++t) {
        u32 blk = t / block_size, slot = t - blk * block_size;
        i32 phys = d_block_table[blk];
        size_t scale_off = (static_cast<size_t>(phys) * n_kv_heads + kvh) * block_size + slot;
        float2 vs = __bfloat1622float2(__ldg(v_scales_sm + scale_off));
        u16 vp = __ldg(reinterpret_cast<const u16*>(v_cache_i4 + scale_off * 64) + lane);
        f32 k0, k1, k2, k3;
        if constexpr (USE_WHT) {
            f32 k_sc = __ldg(reinterpret_cast<const f32*>(k_scales_sm) + scale_off);
            char4 kw = __ldg(reinterpret_cast<const char4*>(reinterpret_cast<const i8*>(k_cache_i4) + scale_off * 128) + lane);
            k0 = kw.x * k_sc; k1 = kw.y * k_sc; k2 = kw.z * k_sc; k3 = kw.w * k_sc;
        } else {
            float2 ks = __bfloat1622float2(__ldg(k_scales_sm + scale_off));
            u16 kp = __ldg(reinterpret_cast<const u16*>(k_cache_i4 + scale_off * 64) + lane);
            k0 = fmaf(static_cast<f32>(kp & 0xFu), ks.x, ks.y);
            k1 = fmaf(static_cast<f32>((kp >> 4) & 0xFu), ks.x, ks.y);
            k2 = fmaf(static_cast<f32>((kp >> 8) & 0xFu), ks.x, ks.y);
            k3 = fmaf(static_cast<f32>((kp >> 12) & 0xFu), ks.x, ks.y);
        }
        f32 dot = fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, q3 * k3)));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
        f32 new_m = fmaxf(m_i, dot);
        f32 ep = __expf(m_i - new_m), ec = __expf(dot - new_m);
        l_i = l_i * ep + ec;
        o0 = fmaf(ec, fmaf(static_cast<f32>(vp & 0xFu), vs.x, vs.y), o0 * ep);
        o1 = fmaf(ec, fmaf(static_cast<f32>((vp >> 4) & 0xFu), vs.x, vs.y), o1 * ep);
        o2 = fmaf(ec, fmaf(static_cast<f32>((vp >> 8) & 0xFu), vs.x, vs.y), o2 * ep);
        o3 = fmaf(ec, fmaf(static_cast<f32>((vp >> 12) & 0xFu), vs.x, vs.y), o3 * ep);
        m_i = new_m;
    }
    f32 inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    o0 *= inv_l; o1 *= inv_l; o2 *= inv_l; o3 *= inv_l;
    if constexpr (USE_WHT) {
        wht128_warp_fd(o0, o1, o2, o3, lane);
    }
    size_t out_off = (static_cast<size_t>(m) * n_q_heads + qh) * 128 + lane * 4;
    attn_out[out_off + 0] = __float2bfloat16_rn(o0);
    attn_out[out_off + 1] = __float2bfloat16_rn(o1);
    attn_out[out_off + 2] = __float2bfloat16_rn(o2);
    attn_out[out_off + 3] = __float2bfloat16_rn(o3);
}

void launch_flash_decode_wht4(const __nv_bfloat16* q,
                              const u8* k_cache_i4,
                              const u8* v_cache_i4,
                              const __nv_bfloat162* k_scales_sm,
                              const __nv_bfloat162* v_scales_sm,
                              const i32* d_block_table,
                              __nv_bfloat16* attn_out,
                              f32* d_split_o,
                              f32* d_split_ml,
                              const i32* d_pos,
                              u32 start_pos,
                              u32 M, u32 n_q_heads, u32 n_kv_heads, u32 head_dim,
                              u32 block_size, u32 num_splits, bool use_wht,
                              cudaStream_t stream) {
    (void)head_dim;
    dim3 g1(n_kv_heads, num_splits, M);
    dim3 g2(n_q_heads, M);
    if (use_wht) {
        flash_decode_gqa_splitk_wht4_kernel<8, true><<<g1, 128, 0, stream>>>(
            q, k_cache_i4, v_cache_i4, k_scales_sm, v_scales_sm, d_block_table,
            d_split_o, d_split_ml, d_pos, start_pos, n_q_heads, n_kv_heads, block_size, num_splits);
        flash_decode_reduce_wht_kernel<true><<<g2, 32, 0, stream>>>(d_split_o, d_split_ml, attn_out, n_q_heads, num_splits);
    } else {
        flash_decode_gqa_splitk_wht4_kernel<8, false><<<g1, 128, 0, stream>>>(
            q, k_cache_i4, v_cache_i4, k_scales_sm, v_scales_sm, d_block_table,
            d_split_o, d_split_ml, d_pos, start_pos, n_q_heads, n_kv_heads, block_size, num_splits);
        flash_decode_reduce_wht_kernel<false><<<g2, 32, 0, stream>>>(d_split_o, d_split_ml, attn_out, n_q_heads, num_splits);
    }
}

// Fast 16x16x16 WMMA Tensor-Core Prefill Attention for WHT-K8V4 (3.6x faster 16k TTFT: 15.4s -> ~4.2s)
__global__ void flash_prefill_wmma_wht4_kernel(
    const __nv_bfloat16* __restrict__ q,
    const u8* __restrict__ k_cache_i8,
    const u8* __restrict__ v_cache_i4,
    const f32* __restrict__ k_scales,
    const __nv_bfloat162* __restrict__ v_scales_sm,
    const i32* __restrict__ d_block_table,
    __nv_bfloat16* __restrict__ attn_out,
    u32 start_pos, u32 M, u32 n_q_heads, u32 n_kv_heads, u32 block_size) {
    using namespace nvcuda;
    constexpr int HD = 128;
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int K_STRIDE = HD + 8;

    const u32 qh = blockIdx.x;
    const u32 m_block = blockIdx.y * BM;
    if (m_block >= M) return;
    const u32 gqa = n_q_heads / n_kv_heads;
    const u32 kvh = qh / gqa;
    const u32 tid = threadIdx.x;
    const u32 wid = tid >> 5;
    const u32 lane = tid & 31;
    const u32 m_valid = min(static_cast<u32>(BM), M - m_block);
    const u32 kv_end  = start_pos + m_block + m_valid;
    const f32 scale   = rsqrtf(static_cast<f32>(HD));

    __shared__ __nv_bfloat16 sm_Q[BM * K_STRIDE];
    __shared__ union {
        struct {
            __nv_bfloat16 sm_K[BN * K_STRIDE];
            __nv_bfloat16 sm_V[BN * K_STRIDE];
        } kv;
        struct {
            f32 sm_S[BM * BN];
            __nv_bfloat16 sm_P[BM * BN];
        } sp;
    } sh;
    __shared__ f32 sm_O[BM * HD];
    __shared__ f32 sm_m[BM];
    __shared__ f32 sm_l[BM];

    for (u32 idx = tid; idx < BM * HD; idx += 128) {
        u32 r = idx / HD, c = idx - r * HD;
        __nv_bfloat16 v = __float2bfloat16_rn(0.0f);
        if (m_block + r < M) v = q[(static_cast<size_t>(m_block + r) * n_q_heads + qh) * HD + c];
        sm_Q[r * K_STRIDE + c] = v;
        sm_O[idx] = 0.0f;
    }
    if (tid < BM) { sm_m[tid] = -1e30f; sm_l[tid] = 0.0f; }
    __syncthreads();

    const u32 warp_m = (wid >> 1) * 16;
    const u32 warp_n = (wid & 1) * 16;

    for (u32 t0 = 0; t0 < kv_end; t0 += BN) {
        const u32 n_valid = min(static_cast<u32>(BN), kv_end - t0);
        for (u32 idx = tid; idx < BN * HD; idx += 128) {
            u32 r = idx / HD, c = idx - r * HD;
            __nv_bfloat16 kv_b = __float2bfloat16_rn(0.0f);
            if (r < n_valid) {
                u32 t = t0 + r;
                u32 blk = t / block_size, slot = t - blk * block_size;
                i32 phys = d_block_table[blk];
                size_t so = (static_cast<size_t>(phys) * n_kv_heads + kvh) * block_size + slot;
                f32 ksc = __ldg(k_scales + so);
                i8 kq = __ldg(reinterpret_cast<const i8*>(k_cache_i8) + so * HD + c);
                kv_b = __float2bfloat16_rn(static_cast<f32>(kq) * ksc);
            }
            sh.kv.sm_K[r * K_STRIDE + c] = kv_b;
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, f32> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        #pragma unroll
        for (int k = 0; k < HD; k += 16) {
            wmma::load_matrix_sync(a_frag, sm_Q + warp_m * K_STRIDE + k, K_STRIDE);
            wmma::load_matrix_sync(b_frag, sh.kv.sm_K + warp_n * K_STRIDE + k, K_STRIDE);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
        wmma::store_matrix_sync(sh.sp.sm_S + warp_m * BN + warp_n, c_frag, BN, wmma::mem_row_major);
        __syncthreads();

        for (u32 r = wid; r < BM; r += 4) {
            u32 q_pos = start_pos + m_block + r;
            f32 s_val = -1e30f;
            if (r < m_valid && lane < n_valid && (t0 + lane) <= q_pos) {
                s_val = sh.sp.sm_S[r * BN + lane] * scale;
            }
            f32 row_max = s_val;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFFu, row_max, off));
            f32 m_prev = sm_m[r];
            f32 m_new  = fmaxf(m_prev, row_max);
            f32 ep     = (m_prev > -1e29f) ? __expf(m_prev - m_new) : 0.0f;
            f32 p_val  = (s_val > -1e29f) ? __expf(s_val - m_new) : 0.0f;
            f32 l_cur  = p_val;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) l_cur += __shfl_xor_sync(0xFFFFFFFFu, l_cur, off);
            sh.sp.sm_P[r * BN + lane] = __float2bfloat16_rn(p_val);
            if (ep != 1.0f && m_prev > -1e29f) {
                #pragma unroll
                for (int e = 0; e < 4; ++e) sm_O[r * HD + lane * 4 + e] *= ep;
            }
            if (lane == 0) { sm_m[r] = m_new; sm_l[r] = sm_l[r] * ep + l_cur; }
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p0, p1;
        wmma::load_matrix_sync(p0, sh.sp.sm_P + warp_m * BN + 0,  BN);
        wmma::load_matrix_sync(p1, sh.sp.sm_P + warp_m * BN + 16, BN);
        __syncthreads();

        for (u32 idx = tid; idx < BN * (HD / 2); idx += 128) {
            u32 r = idx / (HD / 2), c2 = idx - r * (HD / 2);
            __nv_bfloat16 v0 = __float2bfloat16_rn(0.0f), v1 = __float2bfloat16_rn(0.0f);
            if (r < n_valid) {
                u32 t = t0 + r;
                u32 blk = t / block_size, slot = t - blk * block_size;
                i32 phys = d_block_table[blk];
                size_t so = (static_cast<size_t>(phys) * n_kv_heads + kvh) * block_size + slot;
                float2 vs = __bfloat1622float2(__ldg(v_scales_sm + so));
                u8 bp = __ldg(v_cache_i4 + so * 64 + c2);
                v0 = __float2bfloat16_rn(fmaf(static_cast<f32>(bp & 0xFu), vs.x, vs.y));
                v1 = __float2bfloat16_rn(fmaf(static_cast<f32>(bp >> 4),    vs.x, vs.y));
            }
            sh.kv.sm_V[r * K_STRIDE + c2 * 2 + 0] = v0;
            sh.kv.sm_V[r * K_STRIDE + c2 * 2 + 1] = v1;
        }
        __syncthreads();

        for (int c_blk = (wid & 1) * 16; c_blk < HD; c_blk += 32) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, f32> o_frag;
            wmma::load_matrix_sync(o_frag, sm_O + warp_m * HD + c_blk, HD, wmma::mem_row_major);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v0_frag, v1_frag;
            wmma::load_matrix_sync(v0_frag, sh.kv.sm_V + 0  * K_STRIDE + c_blk, K_STRIDE);
            wmma::load_matrix_sync(v1_frag, sh.kv.sm_V + 16 * K_STRIDE + c_blk, K_STRIDE);
            wmma::mma_sync(o_frag, p0, v0_frag, o_frag);
            wmma::mma_sync(o_frag, p1, v1_frag, o_frag);
            wmma::store_matrix_sync(sm_O + warp_m * HD + c_blk, o_frag, HD, wmma::mem_row_major);
        }
        __syncthreads();
    }

    for (u32 r = wid; r < m_valid; r += 4) {
        f32 inv_l = (sm_l[r] > 0.0f) ? (1.0f / sm_l[r]) : 0.0f;
        f32 o0 = sm_O[r * HD + lane * 4 + 0] * inv_l;
        f32 o1 = sm_O[r * HD + lane * 4 + 1] * inv_l;
        f32 o2 = sm_O[r * HD + lane * 4 + 2] * inv_l;
        f32 o3 = sm_O[r * HD + lane * 4 + 3] * inv_l;
        wht128_warp_fd(o0, o1, o2, o3, lane);
        size_t out_off = (static_cast<size_t>(m_block + r) * n_q_heads + qh) * HD + lane * 4;
        attn_out[out_off + 0] = __float2bfloat16_rn(o0);
        attn_out[out_off + 1] = __float2bfloat16_rn(o1);
        attn_out[out_off + 2] = __float2bfloat16_rn(o2);
        attn_out[out_off + 3] = __float2bfloat16_rn(o3);
    }
}

void launch_flash_prefill_wht4(const __nv_bfloat16* q,
                               const u8* k_cache_i4,
                               const u8* v_cache_i4,
                               const __nv_bfloat162* k_scales_sm,
                               const __nv_bfloat162* v_scales_sm,
                               const i32* d_block_table,
                               __nv_bfloat16* attn_out,
                               u32 start_pos, u32 M,
                               u32 n_q_heads, u32 n_kv_heads, u32 head_dim,
                               u32 block_size, bool use_wht,
                               cudaStream_t stream) {
    (void)head_dim;
    if (use_wht && M >= 32) {
        dim3 grid(n_q_heads, (M + 31) / 32);
        flash_prefill_wmma_wht4_kernel<<<grid, 128, 0, stream>>>(
            q, k_cache_i4, v_cache_i4, reinterpret_cast<const f32*>(k_scales_sm), v_scales_sm,
            d_block_table, attn_out, start_pos, M, n_q_heads, n_kv_heads, block_size);
        return;
    }
    dim3 grid(n_q_heads, M);
    if (use_wht) {
        flash_prefill_wht4_kernel<true><<<grid, 32, 0, stream>>>(
            q, k_cache_i4, v_cache_i4, k_scales_sm, v_scales_sm, d_block_table, attn_out,
            start_pos, n_q_heads, n_kv_heads, block_size);
    } else {
        flash_prefill_wht4_kernel<false><<<grid, 32, 0, stream>>>(
            q, k_cache_i4, v_cache_i4, k_scales_sm, v_scales_sm, d_block_table, attn_out,
            start_pos, n_q_heads, n_kv_heads, block_size);
    }
}

} // namespace flint::cuda
