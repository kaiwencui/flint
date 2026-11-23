#include "kernels.hpp"

namespace flint::cuda {

namespace {

template <bool FP8_KV>
__global__ void fused_rope_kv_write_kernel(const __nv_bfloat16* __restrict__ qkv,
                                           __nv_bfloat16* __restrict__ q_out,
                                           void* __restrict__ k_cache_raw,
                                           void* __restrict__ v_cache_raw,
                                           f32* __restrict__ k_scales,
                                           f32* __restrict__ v_scales,
                                           const i32* __restrict__ d_block_table,
                                           const f32* __restrict__ d_inv_freq,
                                           const __nv_bfloat16* __restrict__ q_norm_w,
                                           const __nv_bfloat16* __restrict__ k_norm_w,
                                           const i32* __restrict__ d_pos,
                                           u32 start_pos,
                                           u32 n_q_heads, u32 n_kv_heads, u32 head_dim,
                                           u32 block_size, f32 rms_eps) {
    const u32 h_idx = blockIdx.x;
    const u32 m = blockIdx.y;
    const u32 tid = threadIdx.x;
    const u32 half = head_dim / 2;

    const u32 pos = (d_pos ? static_cast<u32>(*d_pos) : start_pos) + m;
    const u32 q_dim = n_q_heads * head_dim;
    const u32 kv_dim = n_kv_heads * head_dim;
    const u32 qkv_stride = q_dim + 2 * kv_dim;
    const __nv_bfloat16* row = qkv + static_cast<size_t>(m) * qkv_stride;

    f32 c = 1.0f, s = 0.0f;
    if (tid < half) {
        f32 theta = static_cast<f32>(pos) * d_inv_freq[tid];
        sincosf(theta, &s, &c);
    }

    __shared__ f32 smem_a[8];
    __shared__ f32 smem_b[8];
    const int lane = static_cast<int>(tid & 31u);
    const int wid  = static_cast<int>(tid >> 5);
    const int nwarps = static_cast<int>((blockDim.x + 31u) >> 5);

    auto apply_head_norm = [&](f32& x0, f32& x1, const __nv_bfloat16* nw) {
        if (!nw) return;
        f32 ss = (tid < half) ? (x0 * x0 + x1 * x1) : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xFFFFFFFFu, ss, off);
        if (lane == 0) smem_a[wid] = ss;
        __syncthreads();
        f32 total = 0.0f;
        for (int w = 0; w < nwarps; ++w) total += smem_a[w];
        f32 inv = rsqrtf(total / static_cast<f32>(head_dim) + rms_eps);
        if (tid < half) {
            x0 = x0 * inv * __bfloat162float(nw[tid]);
            x1 = x1 * inv * __bfloat162float(nw[tid + half]);
        }
        __syncthreads();
    };

    if (h_idx < n_q_heads) {
        const __nv_bfloat16* qh = row + h_idx * head_dim;
        f32 x0 = (tid < half) ? __bfloat162float(qh[tid]) : 0.0f;
        f32 x1 = (tid < half) ? __bfloat162float(qh[tid + half]) : 0.0f;
        apply_head_norm(x0, x1, q_norm_w);

        if (tid < half) {
            __nv_bfloat16* dst = q_out + static_cast<size_t>(m) * q_dim + h_idx * head_dim;
            dst[tid]        = __float2bfloat16_rn(x0 * c - x1 * s);
            dst[tid + half] = __float2bfloat16_rn(x1 * c + x0 * s);
        }
    } else {
        const u32 kvh = h_idx - n_q_heads;
        const __nv_bfloat16* kh = row + q_dim + kvh * head_dim;
        const __nv_bfloat16* vh = row + q_dim + kv_dim + kvh * head_dim;

        f32 k0 = (tid < half) ? __bfloat162float(kh[tid]) : 0.0f;
        f32 k1 = (tid < half) ? __bfloat162float(kh[tid + half]) : 0.0f;
        apply_head_norm(k0, k1, k_norm_w);

        f32 rk0 = k0 * c - k1 * s;
        f32 rk1 = k1 * c + k0 * s;

        const u32 blk_idx = pos / block_size;
        const u32 slot = pos - blk_idx * block_size;
        const i32 phys_blk = d_block_table[blk_idx];
        const size_t scale_off = (static_cast<size_t>(phys_blk) * n_kv_heads + kvh) * block_size + slot;
        const size_t base_off  = scale_off * head_dim;

        if constexpr (FP8_KV) {
            f32 v0 = (tid < half) ? __bfloat162float(vh[tid]) : 0.0f;
            f32 v1 = (tid < half) ? __bfloat162float(vh[tid + half]) : 0.0f;

            // Per-token, per-KV-head dynamic max-abs reduction for outlier-free FP8 E4M3
            f32 ka = (tid < half) ? fmaxf(fabsf(rk0), fabsf(rk1)) : 0.0f;
            f32 va = (tid < half) ? fmaxf(fabsf(v0),  fabsf(v1))  : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                ka = fmaxf(ka, __shfl_xor_sync(0xFFFFFFFFu, ka, off));
                va = fmaxf(va, __shfl_xor_sync(0xFFFFFFFFu, va, off));
            }
            if (lane == 0) {
                smem_a[wid] = ka;
                smem_b[wid] = va;
            }
            __syncthreads();

            f32 k_amax = 0.0f, v_amax = 0.0f;
            for (int w = 0; w < nwarps; ++w) {
                k_amax = fmaxf(k_amax, smem_a[w]);
                v_amax = fmaxf(v_amax, smem_b[w]);
            }
            f32 k_sc = fmaxf(k_amax / 127.0f, 1e-6f);
            f32 v_sc = fmaxf(v_amax / 127.0f, 1e-6f);
            f32 k_inv = 1.0f / k_sc;
            f32 v_inv = 1.0f / v_sc;

            if (tid == 0) {
                k_scales[scale_off] = k_sc;
                v_scales[scale_off] = v_sc;
            }
            if (tid < half) {
                auto* kc = static_cast<i8*>(k_cache_raw);
                auto* vc = static_cast<i8*>(v_cache_raw);
                auto q_i8 = [](f32 x) -> i8 {
                    int q = __float2int_rn(x);
                    return static_cast<i8>(max(-127, min(127, q)));
                };
                kc[base_off + tid]        = q_i8(rk0 * k_inv);
                kc[base_off + tid + half] = q_i8(rk1 * k_inv);
                vc[base_off + tid]        = q_i8(v0  * v_inv);
                vc[base_off + tid + half] = q_i8(v1  * v_inv);
            }
        } else {
            if (tid < half) {
                auto* kc = static_cast<__nv_bfloat16*>(k_cache_raw);
                auto* vc = static_cast<__nv_bfloat16*>(v_cache_raw);
                kc[base_off + tid]        = __float2bfloat16_rn(rk0);
                kc[base_off + tid + half] = __float2bfloat16_rn(rk1);
                vc[base_off + tid]        = vh[tid];
                vc[base_off + tid + half] = vh[tid + half];
            }
        }
    }
}

// Orthonormal 128-point Walsh-Hadamard Transform across 1 warp (32 lanes x 4 elems = 128):
// H = H^T, H^2 = I -> (Hq)·(Hk) == q·k and H(sum p_t (Hv_t)) == sum p_t v_t identically
// Spreads single-channel outliers (e.g. Qwen2.5 Layer 27 k_bias=442.0) uniformly by sqrt(128)=11.3x
// in ~14 cycles with ZERO extra weights and ZERO offline calibration
__device__ __forceinline__ void wht128_warp(f32& v0, f32& v1, f32& v2, f32& v3, int lane) {
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

// Mixed-Precision K-INT8 (PTX dp4a immune to Layer-27 442.0 K-bias outlier) +
// Zero-Calibration Walsh-Hadamard 4-Bit V (4x smaller V -> 2.56x smaller total KV, cos>=0.998)
// When USE_WHT==false (ablation mode): keeps unrotated K4+V4 so we can compare directly
template <bool USE_WHT>
__global__ void fused_rope_kv_write_wht4_kernel(
    __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ q_out,
    u8* __restrict__ k_cache_raw,  // USE_WHT: i8[128] (128B); !USE_WHT: packed u4[64] (64B)
    u8* __restrict__ v_cache_i4,   // packed u4[64] (64B)
    __nv_bfloat162* __restrict__ k_scales_sm, // USE_WHT: f32 scale; !USE_WHT: bf16x2(scale, min)
    __nv_bfloat162* __restrict__ v_scales_sm, // bf16x2(scale, min)
    const i32* __restrict__ d_block_table,
    const f32* __restrict__ d_inv_freq,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const i32* __restrict__ d_pos,
    u32 start_pos,
    u32 n_q_heads, u32 n_kv_heads, u32 block_size, f32 rms_eps) {
    const u32 h = blockIdx.x;
    const u32 m = blockIdx.y;
    const int lane = threadIdx.x;

    const u32 pos = (d_pos ? static_cast<u32>(*d_pos) : start_pos) + m;
    const u32 q_dim  = n_q_heads * 128;
    const u32 kv_dim = n_kv_heads * 128;
    const u32 qkv_stride = q_dim + 2 * kv_dim;
    __nv_bfloat16* tok_qkv = qkv + static_cast<size_t>(m) * qkv_stride;

    __shared__ f32 sm_k[128];
    __shared__ f32 sm_v[128];

    if (h < n_q_heads) {
        __nv_bfloat16* qh = tok_qkv + h * 128;
        __nv_bfloat16* qo = q_out + (static_cast<size_t>(m) * n_q_heads + h) * 128;
        f32 x0 = __bfloat162float(qh[lane * 2 + 0]);
        f32 x1 = __bfloat162float(qh[lane * 2 + 1]);
        f32 y0 = __bfloat162float(qh[lane * 2 + 64]);
        f32 y1 = __bfloat162float(qh[lane * 2 + 65]);
        if (q_norm_w) {
            f32 ss = x0 * x0 + x1 * x1 + y0 * y0 + y1 * y1;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
            f32 rstd = rsqrtf(ss / 128.0f + rms_eps);
            x0 *= rstd * __bfloat162float(q_norm_w[lane * 2 + 0]);
            x1 *= rstd * __bfloat162float(q_norm_w[lane * 2 + 1]);
            y0 *= rstd * __bfloat162float(q_norm_w[lane * 2 + 64]);
            y1 *= rstd * __bfloat162float(q_norm_w[lane * 2 + 65]);
        }
        f32 a0 = static_cast<f32>(pos) * d_inv_freq[lane * 2 + 0];
        f32 a1 = static_cast<f32>(pos) * d_inv_freq[lane * 2 + 1];
        f32 s0, c0, s1, c1;
        __sincosf(a0, &s0, &c0);
        __sincosf(a1, &s1, &c1);
        qo[lane * 2 + 0]  = __float2bfloat16_rn(x0 * c0 - y0 * s0);
        qo[lane * 2 + 1]  = __float2bfloat16_rn(x1 * c1 - y1 * s1);
        qo[lane * 2 + 64] = __float2bfloat16_rn(y0 * c0 + x0 * s0);
        qo[lane * 2 + 65] = __float2bfloat16_rn(y1 * c1 + x1 * s1);
        return;
    }

    const u32 kvh = h - n_q_heads;
    __nv_bfloat16* kh = tok_qkv + q_dim + kvh * 128;
    __nv_bfloat16* vh = tok_qkv + q_dim + kv_dim + kvh * 128;

    f32 x0 = __bfloat162float(kh[lane * 2 + 0]);
    f32 x1 = __bfloat162float(kh[lane * 2 + 1]);
    f32 y0 = __bfloat162float(kh[lane * 2 + 64]);
    f32 y1 = __bfloat162float(kh[lane * 2 + 65]);
    if (k_norm_w) {
        f32 ss = x0 * x0 + x1 * x1 + y0 * y0 + y1 * y1;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
        f32 rstd = rsqrtf(ss / 128.0f + rms_eps);
        x0 *= rstd * __bfloat162float(k_norm_w[lane * 2 + 0]);
        x1 *= rstd * __bfloat162float(k_norm_w[lane * 2 + 1]);
        y0 *= rstd * __bfloat162float(k_norm_w[lane * 2 + 64]);
        y1 *= rstd * __bfloat162float(k_norm_w[lane * 2 + 65]);
    }
    f32 a0 = static_cast<f32>(pos) * d_inv_freq[lane * 2 + 0];
    f32 a1 = static_cast<f32>(pos) * d_inv_freq[lane * 2 + 1];
    f32 s0, c0, s1, c1;
    __sincosf(a0, &s0, &c0);
    __sincosf(a1, &s1, &c1);
    sm_k[lane * 2 + 0]  = x0 * c0 - y0 * s0;
    sm_k[lane * 2 + 1]  = x1 * c1 - y1 * s1;
    sm_k[lane * 2 + 64] = y0 * c0 + x0 * s0;
    sm_k[lane * 2 + 65] = y1 * c1 + x1 * s1;
    sm_v[lane * 2 + 0]  = __bfloat162float(vh[lane * 2 + 0]);
    sm_v[lane * 2 + 1]  = __bfloat162float(vh[lane * 2 + 1]);
    sm_v[lane * 2 + 64] = __bfloat162float(vh[lane * 2 + 64]);
    sm_v[lane * 2 + 65] = __bfloat162float(vh[lane * 2 + 65]);
    __syncwarp();

    f32 k0 = sm_k[lane * 4 + 0], k1 = sm_k[lane * 4 + 1], k2 = sm_k[lane * 4 + 2], k3 = sm_k[lane * 4 + 3];
    f32 v0 = sm_v[lane * 4 + 0], v1 = sm_v[lane * 4 + 1], v2 = sm_v[lane * 4 + 2], v3 = sm_v[lane * 4 + 3];
    if constexpr (USE_WHT) {
        // Rotate Values into Walsh-Hadamard domain (H^2 = I -> exact inverse in reduce stage)
        wht128_warp(v0, v1, v2, v3, lane);
    }

    u32 blk = pos / block_size;
    u32 slot = pos - blk * block_size;
    i32 phys_blk = d_block_table[blk];
    size_t scale_off = (static_cast<size_t>(phys_blk) * n_kv_heads + kvh) * block_size + slot;

    // Write Keys: INT8 (128B, hardware dp4a) for WHT-K8V4; INT4 (64B) for raw INT4 ablation
    if constexpr (USE_WHT) {
        f32 kam = fmaxf(fmaxf(fabsf(k0), fabsf(k1)), fmaxf(fabsf(k2), fabsf(k3)));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) kam = fmaxf(kam, __shfl_xor_sync(0xFFFFFFFFu, kam, off));
        f32 k_sc = fmaxf(kam / 127.0f, 1e-6f);
        f32 k_inv = 1.0f / k_sc;
        if (lane == 0) reinterpret_cast<f32*>(k_scales_sm)[scale_off] = k_sc;
        auto q8 = [](f32 x) -> i8 { return static_cast<i8>(max(-127, min(127, __float2int_rn(x)))); };
        i8* dst = reinterpret_cast<i8*>(k_cache_raw) + scale_off * 128 + lane * 4;
        dst[0] = q8(k0 * k_inv); dst[1] = q8(k1 * k_inv); dst[2] = q8(k2 * k_inv); dst[3] = q8(k3 * k_inv);
    } else {
        f32 kmn = fminf(fminf(k0, k1), fminf(k2, k3)), kmx = fmaxf(fmaxf(k0, k1), fmaxf(k2, k3));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            kmn = fminf(kmn, __shfl_xor_sync(0xFFFFFFFFu, kmn, off));
            kmx = fmaxf(kmx, __shfl_xor_sync(0xFFFFFFFFu, kmx, off));
        }
        f32 k_sc = fmaxf((kmx - kmn) / 15.0f, 1e-6f), k_inv = 1.0f / k_sc;
        if (lane == 0) k_scales_sm[scale_off] = __floats2bfloat162_rn(k_sc, kmn);
        auto q4 = [](f32 x) -> u32 { return static_cast<u32>(max(0, min(15, __float2int_rn(x)))); };
        size_t bo = scale_off * 64 + static_cast<size_t>(lane) * 2;
        k_cache_raw[bo + 0] = static_cast<u8>(q4((k0 - kmn) * k_inv) | (q4((k1 - kmn) * k_inv) << 4));
        k_cache_raw[bo + 1] = static_cast<u8>(q4((k2 - kmn) * k_inv) | (q4((k3 - kmn) * k_inv) << 4));
    }

    // Write Values: Asymmetric 4-bit (64B/head, 4x smaller than BF16)
    f32 vmn = fminf(fminf(v0, v1), fminf(v2, v3)), vmx = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        vmn = fminf(vmn, __shfl_xor_sync(0xFFFFFFFFu, vmn, off));
        vmx = fmaxf(vmx, __shfl_xor_sync(0xFFFFFFFFu, vmx, off));
    }
    f32 v_sc = fmaxf((vmx - vmn) / 15.0f, 1e-6f), v_inv = 1.0f / v_sc;
    if (lane == 0) v_scales_sm[scale_off] = __floats2bfloat162_rn(v_sc, vmn);
    auto q4v = [](f32 x) -> u32 { return static_cast<u32>(max(0, min(15, __float2int_rn(x)))); };
    size_t vbo = scale_off * 64 + static_cast<size_t>(lane) * 2;
    v_cache_i4[vbo + 0] = static_cast<u8>(q4v((v0 - vmn) * v_inv) | (q4v((v1 - vmn) * v_inv) << 4));
    v_cache_i4[vbo + 1] = static_cast<u8>(q4v((v2 - vmn) * v_inv) | (q4v((v3 - vmn) * v_inv) << 4));
}

} // namespace

void launch_fused_rope_kv_write_bf16(__nv_bfloat16* qkv,
                                     __nv_bfloat16* q_out,
                                     __nv_bfloat16* k_cache,
                                     __nv_bfloat16* v_cache,
                                     const i32* d_block_table,
                                     const f32* d_inv_freq,
                                     const __nv_bfloat16* q_norm_w,
                                     const __nv_bfloat16* k_norm_w,
                                     const i32* d_pos,
                                     u32 start_pos,
                                     u32 M, u32 n_q_heads, u32 n_kv_heads, u32 head_dim,
                                     u32 block_size, f32 rms_eps,
                                     cudaStream_t stream) {
    dim3 grid(n_q_heads + n_kv_heads, M);
    u32 threads = ((head_dim / 2 + 31) / 32) * 32;
    fused_rope_kv_write_kernel<false><<<grid, threads, 0, stream>>>(
        qkv, q_out, k_cache, v_cache, nullptr, nullptr, d_block_table, d_inv_freq,
        q_norm_w, k_norm_w, d_pos, start_pos,
        n_q_heads, n_kv_heads, head_dim, block_size, rms_eps);
}

void launch_fused_rope_kv_write_fp8(__nv_bfloat16* qkv,
                                    __nv_bfloat16* q_out,
                                    u8* k_cache_fp8,
                                    u8* v_cache_fp8,
                                    f32* k_scales,
                                    f32* v_scales,
                                    const i32* d_block_table,
                                    const f32* d_inv_freq,
                                    const __nv_bfloat16* q_norm_w,
                                    const __nv_bfloat16* k_norm_w,
                                    const i32* d_pos,
                                    u32 start_pos,
                                    u32 M, u32 n_q_heads, u32 n_kv_heads, u32 head_dim,
                                    u32 block_size, f32 rms_eps,
                                    cudaStream_t stream) {
    dim3 grid(n_q_heads + n_kv_heads, M);
    u32 threads = ((head_dim / 2 + 31) / 32) * 32;
    fused_rope_kv_write_kernel<true><<<grid, threads, 0, stream>>>(
        qkv, q_out, k_cache_fp8, v_cache_fp8, k_scales, v_scales, d_block_table, d_inv_freq,
        q_norm_w, k_norm_w, d_pos, start_pos,
        n_q_heads, n_kv_heads, head_dim, block_size, rms_eps);
}

void launch_fused_rope_kv_write_wht4(__nv_bfloat16* qkv,
                                     __nv_bfloat16* q_out,
                                     u8* k_cache_i4,
                                     u8* v_cache_i4,
                                     __nv_bfloat162* k_scales_sm,
                                     __nv_bfloat162* v_scales_sm,
                                     const i32* d_block_table,
                                     const f32* d_inv_freq,
                                     const __nv_bfloat16* q_norm_w,
                                     const __nv_bfloat16* k_norm_w,
                                     const i32* d_pos,
                                     u32 start_pos,
                                     u32 M, u32 n_q_heads, u32 n_kv_heads, u32 head_dim,
                                     u32 block_size, f32 rms_eps, bool use_wht,
                                     cudaStream_t stream) {
    (void)head_dim; // 128-dim head WHT-INT4
    dim3 grid(n_q_heads + n_kv_heads, M);
    if (use_wht) {
        fused_rope_kv_write_wht4_kernel<true><<<grid, 32, 0, stream>>>(
            qkv, q_out, k_cache_i4, v_cache_i4, k_scales_sm, v_scales_sm, d_block_table, d_inv_freq,
            q_norm_w, k_norm_w, d_pos, start_pos, n_q_heads, n_kv_heads, block_size, rms_eps);
    } else {
        fused_rope_kv_write_wht4_kernel<false><<<grid, 32, 0, stream>>>(
            qkv, q_out, k_cache_i4, v_cache_i4, k_scales_sm, v_scales_sm, d_block_table, d_inv_freq,
            q_norm_w, k_norm_w, d_pos, start_pos, n_q_heads, n_kv_heads, block_size, rms_eps);
    }
}

} // namespace flint::cuda
