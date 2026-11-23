#include "inkling2_common.cuh"

namespace flint::cuda {

namespace {

// ============================================================================
// 1. Block Attention Residual (AttnRes) + Optional Fused RMSNorm
// ============================================================================
// Grid: (num_queries, M), Block: 256 threads
// Sources: stack[0..n_stack-1] (BF16 [M, D]) + last (BF16 [M, D], optionally + addend BF16 [M, D]).
// Queries: [num_queries, D] F32 (pre-scaled by D**-0.5).
// If norm_w != nullptr: writes BF16 RMSNorm(mix, norm_w[q_idx]) to out_bf16[q_idx, m, D].
// Else: writes FP32 mix directly to out_f32[q_idx, m, D].
__global__ void inkling_attn_res_kernel(
    const __nv_bfloat16* const* __restrict__ stack_ptrs, // device array of up to 8 pointers
    u32 n_stack,
    const __nv_bfloat16* __restrict__ last_ptr,
    const __nv_bfloat16* __restrict__ addend_ptr,
    const f32* __restrict__ queries,      // [num_q, D] pre-scaled by D**-0.5
    const __nv_bfloat16* __restrict__ norm_w, // [num_q, D] or nullptr
    __nv_bfloat16* __restrict__ out_bf16, // [num_q, M, D] if norm_w != nullptr
    f32* __restrict__ out_f32,            // [num_q, M, D] if norm_w == nullptr
    u32 M, u32 D, f32 eps, bool normalize_values) {
    const u32 q_idx = blockIdx.x;
    const u32 m     = blockIdx.y;
    const u32 tid   = threadIdx.x;
    const u32 lane  = tid & 31;
    const u32 wid   = tid >> 5;
    if (m >= M) return;

    const f32* q_row = queries + static_cast<size_t>(q_idx) * D;
    const u32 n_src = n_stack + 1; // stack[0..n_stack-1], then last (+ addend)

    __shared__ f32 sm_a[8];
    __shared__ f32 sm_b[8];
    __shared__ f32 sm_w[8];
    __shared__ f32 sm_logits[8];
    __shared__ f32 sm_rstd[8];

    auto load_src = [&](u32 n, u32 d) -> f32 {
        if (n < n_stack) {
            return __bfloat162float(stack_ptrs[n][static_cast<size_t>(m) * D + d]);
        }
        f32 v = __bfloat162float(last_ptr[static_cast<size_t>(m) * D + d]);
        if (addend_ptr) v += __bfloat162float(addend_ptr[static_cast<size_t>(m) * D + d]);
        return v;
    };

    // Pass 1: compute per-source rstd and dot(q, x_n) * rstd
    for (u32 n = 0; n < n_src; ++n) {
        f32 ss = 0.0f, dot = 0.0f;
        for (u32 d = tid; d < D; d += 256) {
            f32 x = load_src(n, d);
            ss  = fmaf(x, x, ss);
            dot = fmaf(q_row[d], x, dot);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            ss  += __shfl_down_sync(0xFFFFFFFFu, ss, off);
            dot += __shfl_down_sync(0xFFFFFFFFu, dot, off);
        }
        if (lane == 0) { sm_a[wid] = ss; sm_b[wid] = dot; }
        __syncthreads();
        if (tid == 0) {
            f32 ts = 0.0f, td = 0.0f;
            for (int w = 0; w < 8; ++w) { ts += sm_a[w]; td += sm_b[w]; }
            f32 rstd = rsqrtf(ts / static_cast<f32>(D) + eps);
            sm_rstd[n] = rstd;
            sm_logits[n] = td * rstd;
        }
        __syncthreads();
    }

    if (tid == 0) {
        f32 mx = -1e30f;
        for (u32 n = 0; n < n_src; ++n) mx = fmaxf(mx, sm_logits[n]);
        f32 sum_p = 0.0f;
        for (u32 n = 0; n < n_src; ++n) {
            sm_w[n] = expf(sm_logits[n] - mx);
            sum_p += sm_w[n];
        }
        f32 inv_sum = 1.0f / sum_p;
        for (u32 n = 0; n < n_src; ++n) {
            sm_w[n] = sm_w[n] * inv_sum * (normalize_values ? sm_rstd[n] : 1.0f);
        }
    }
    __syncthreads();

    if (!norm_w) {
        f32* dst = out_f32 + (static_cast<size_t>(q_idx) * M + m) * D;
        for (u32 d = tid; d < D; d += 256) {
            f32 acc = 0.0f;
            for (u32 n = 0; n < n_src; ++n) acc = fmaf(sm_w[n], load_src(n, d), acc);
            dst[d] = acc;
        }
        return;
    }

    // Pass 2: compute mixed output ONCE into 9 registers (D=2304 / 256 = 9 -> eliminates 3rd global pass)
    f32 mv[10];
    f32 out_ss = 0.0f;
    int cnt = 0;
    for (u32 d = tid; d < D; d += 256) {
        f32 acc = 0.0f;
        for (u32 n = 0; n < n_src; ++n) acc = fmaf(sm_w[n], load_src(n, d), acc);
        mv[cnt++] = acc;
        out_ss = fmaf(acc, acc, out_ss);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) out_ss += __shfl_down_sync(0xFFFFFFFFu, out_ss, off);
    if (lane == 0) sm_a[wid] = out_ss;
    __syncthreads();
    if (tid == 0) {
        f32 ts = 0.0f;
        for (int w = 0; w < 8; ++w) ts += sm_a[w];
        sm_a[0] = rsqrtf(ts / static_cast<f32>(D) + eps);
    }
    __syncthreads();
    f32 out_rstd = sm_a[0];

    const __nv_bfloat16* nw = norm_w + static_cast<size_t>(q_idx) * D;
    __nv_bfloat16* dst = out_bf16 + (static_cast<size_t>(q_idx) * M + m) * D;
    int idx_r = 0;
    for (u32 d = tid; d < D; d += 256) {
        dst[d] = __float2bfloat16_rn(mv[idx_r++] * out_rstd * __bfloat162float(nw[d]));
    }
}

// ============================================================================
// 2. Looped Core Cross-Pass Message Controller (pair_rmsnorm & blend_rmsnorm)
// ============================================================================
__global__ void inkling_pair_rmsnorm_kernel(const f32* __restrict__ dest, // [S_d, M, D]
                                            const f32* __restrict__ msg,  // [S, M, D]
                                            __nv_bfloat16* __restrict__ out, // [S, M, 2*D]
                                            bool broadcast_dest,
                                            u32 M, u32 D, f32 eps) {
    const u32 s = blockIdx.x;
    const u32 m = blockIdx.y;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31;
    const u32 wid  = tid >> 5;

    const f32* d_row = dest + (static_cast<size_t>(broadcast_dest ? 0 : s) * M + m) * D;
    const f32* m_row = msg  + (static_cast<size_t>(s) * M + m) * D;

    f32 ssd = 0.0f, ssm = 0.0f;
    for (u32 d = tid; d < D; d += 256) {
        f32 vd = d_row[d], vm = m_row[d];
        ssd = fmaf(vd, vd, ssd);
        ssm = fmaf(vm, vm, ssm);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ssd += __shfl_down_sync(0xFFFFFFFFu, ssd, off);
        ssm += __shfl_down_sync(0xFFFFFFFFu, ssm, off);
    }
    __shared__ f32 sm_d[8], sm_m[8];
    if (lane == 0) { sm_d[wid] = ssd; sm_m[wid] = ssm; }
    __syncthreads();
    if (tid == 0) {
        f32 td = 0.0f, tm = 0.0f;
        for (int w = 0; w < 8; ++w) { td += sm_d[w]; tm += sm_m[w]; }
        sm_d[0] = rsqrtf(td / static_cast<f32>(D) + eps);
        sm_m[0] = rsqrtf(tm / static_cast<f32>(D) + eps);
    }
    __syncthreads();
    f32 rd = sm_d[0], rm = sm_m[0];
    __nv_bfloat16* o_row = out + (static_cast<size_t>(s) * M + m) * (2 * D);
    for (u32 d = tid; d < D; d += 256) {
        o_row[d]     = __float2bfloat16_rn(d_row[d] * rd);
        o_row[D + d] = __float2bfloat16_rn(m_row[d] * rm);
    }
}

__global__ void inkling_blend_rmsnorm_kernel(const f32* __restrict__ dest,
                                             const f32* __restrict__ msg,
                                             const __nv_bfloat16* __restrict__ gain_bf16,   // [S, M, D]
                                             const __nv_bfloat16* __restrict__ forget_bf16, // [S, M, D]
                                             const __nv_bfloat16* __restrict__ norm_w,      // [S_w, D]
                                             __nv_bfloat16* __restrict__ out,               // [S, M, D]
                                             bool broadcast_dest,
                                             bool broadcast_norm,
                                             u32 M, u32 D, f32 eps) {
    const u32 s = blockIdx.x;
    const u32 m = blockIdx.y;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31;
    const u32 wid  = tid >> 5;

    const f32* d_row = dest + (static_cast<size_t>(broadcast_dest ? 0 : s) * M + m) * D;
    const f32* m_row = msg  + (static_cast<size_t>(s) * M + m) * D;
    const __nv_bfloat16* g_row = gain_bf16   + (static_cast<size_t>(s) * M + m) * D;
    const __nv_bfloat16* f_row = forget_bf16 + (static_cast<size_t>(s) * M + m) * D;
    const __nv_bfloat16* nw    = norm_w      + static_cast<size_t>(broadcast_norm ? 0 : s) * D;

    f32 ss = 0.0f;
    for (u32 d = tid; d < D; d += 256) {
        f32 g = tanhf(__bfloat162float(g_row[d]));
        f32 f = tanhf(__bfloat162float(f_row[d]));
        f32 x = (1.0f - f) * d_row[d] + g * m_row[d];
        ss = fmaf(x, x, ss);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xFFFFFFFFu, ss, off);
    __shared__ f32 sm[8];
    if (lane == 0) sm[wid] = ss;
    __syncthreads();
    if (tid == 0) {
        f32 t = 0.0f;
        for (int w = 0; w < 8; ++w) t += sm[w];
        sm[0] = rsqrtf(t / static_cast<f32>(D) + eps);
    }
    __syncthreads();
    f32 rstd = sm[0];

    __nv_bfloat16* o_row = out + (static_cast<size_t>(s) * M + m) * D;
    for (u32 d = tid; d < D; d += 256) {
        f32 g = tanhf(__bfloat162float(g_row[d]));
        f32 f = tanhf(__bfloat162float(f_row[d]));
        f32 x = (1.0f - f) * d_row[d] + g * m_row[d];
        o_row[d] = __float2bfloat16_rn(x * rstd * __bfloat162float(nw[d]));
    }
}

// ============================================================================
// 3. Causal Short-Conv (W=4) on Hidden Streams (attn_out / mlp_out) + History Update
// ============================================================================
// hist: [4, D] BF16 ring buffer holding x[pos] at hist[pos & 3] (read prior lags, then commit).
// Static formula (oldest tap first): y[t] = (sum_{k=0..3} w[c, k] * x[t - (3-k)] + x[t]) * rsqrt(5)
__global__ void inkling_sconv_hidden_kernel(const __nv_bfloat16* __restrict__ x,    // [M, D]
                                            const __nv_bfloat16* __restrict__ w,    // [D, 4]
                                            __nv_bfloat16* __restrict__ hist,       // [4, D]
                                            __nv_bfloat16* __restrict__ out,        // [M, D]
                                            const __nv_bfloat16* __restrict__ res1, // optional addend 1
                                            const __nv_bfloat16* __restrict__ res2, // optional addend 2
                                            const i32* __restrict__ d_pos,
                                            u32 start_pos, u32 M, u32 D) {
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;

    const u32 p0 = d_pos ? static_cast<u32>(*d_pos) : start_pos;
    const f32 scale = rsqrtf(5.0f);
    f32 w0 = __bfloat162float(w[static_cast<size_t>(c) * 4 + 0]); // lag 3
    f32 w1 = __bfloat162float(w[static_cast<size_t>(c) * 4 + 1]); // lag 2
    f32 w2 = __bfloat162float(w[static_cast<size_t>(c) * 4 + 2]); // lag 1
    f32 w3 = __bfloat162float(w[static_cast<size_t>(c) * 4 + 3]); // lag 0

    // Preload the 3 prior lags into registers BEFORE mutating 16-slot ring hist (lossless spec rollback)
    f32 h_m3 = (p0 >= 3) ? __bfloat162float(hist[((p0 - 3) & 15u) * D + c]) : 0.0f;
    f32 h_m2 = (p0 >= 2) ? __bfloat162float(hist[((p0 - 2) & 15u) * D + c]) : 0.0f;
    f32 h_m1 = (p0 >= 1) ? __bfloat162float(hist[((p0 - 1) & 15u) * D + c]) : 0.0f;

    for (u32 m = 0; m < M; ++m) {
        u32 pos = p0 + m;
        __nv_bfloat16 cur_bf = x[static_cast<size_t>(m) * D + c];
        f32 cur = __bfloat162float(cur_bf);
        f32 y = (w0 * h_m3 + w1 * h_m2 + w2 * h_m1 + (w3 + 1.0f) * cur) * scale;
        f32 yr = __bfloat162float(__float2bfloat16_rn(y));
        if (res1) yr += __bfloat162float(res1[static_cast<size_t>(m) * D + c]);
        if (res2) yr += __bfloat162float(res2[static_cast<size_t>(m) * D + c]);
        out[static_cast<size_t>(m) * D + c] = __float2bfloat16_rn(yr);

        hist[(pos & 15u) * D + c] = cur_bf;
        h_m3 = h_m2;
        h_m2 = h_m1;
        h_m1 = cur;
    }
}

// ============================================================================
// 4. Attention Prologue: Q/R L2-Clip (+ tau) AND K/V Short-Conv + Clip/Norm + KV Write
// ============================================================================
__global__ void inkling_qr_clip_kernel(const __nv_bfloat16* __restrict__ q_in,  // [M, Hq, 128]
                                       const __nv_bfloat16* __restrict__ r_in,  // [M, Hq, 16]
                                       const __nv_bfloat16* __restrict__ q_sc,  // [Hq, 128]
                                       const __nv_bfloat16* __restrict__ r_sc,  // [Hq, 16]
                                       __nv_bfloat16* __restrict__ q_out,
                                       __nv_bfloat16* __restrict__ r_out,
                                       const i32* __restrict__ d_pos,
                                       u32 start_pos, u32 M, u32 n_q_heads,
                                       f32 q_clip, f32 r_clip,
                                       bool use_log_scale, u32 n_floor, f32 alpha) {
    const u32 h = blockIdx.x;
    const u32 m = blockIdx.y;
    const u32 tid = threadIdx.x; // 128 threads
    if (h >= n_q_heads || m >= M) return;

    const u32 pos = (d_pos ? static_cast<u32>(*d_pos) : start_pos) + m;
    f32 tau = 1.0f;
    if (use_log_scale) {
        f32 ratio = static_cast<f32>(pos + 1) / static_cast<f32>(n_floor);
        if (ratio > 1.0f) tau = 1.0f + alpha * logf(ratio);
    }

    // Q L2-clip (128 dims, 1 thread per dim)
    size_t q_off = (static_cast<size_t>(m) * n_q_heads + h) * 128 + tid;
    f32 qv = __bfloat162float(q_in[q_off]) * __bfloat162float(q_sc[h * 128 + tid]);
    f32 qss = qv * qv;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) qss += __shfl_xor_sync(0xFFFFFFFFu, qss, off);
    __shared__ f32 sm[4];
    int wid = tid >> 5, lane = tid & 31;
    if (lane == 0) sm[wid] = qss;
    __syncthreads();
    f32 total_q = sm[0] + sm[1] + sm[2] + sm[3];
    f32 q_norm = sqrtf(total_q);
    f32 q_mul = fminf(q_clip / (q_norm + 1e-6f), 1.0f);
    f32 q_res = qv * q_mul;
    if (use_log_scale) {
        q_res = __bfloat162float(__float2bfloat16_rn(q_res)) * tau;
    }
    q_out[q_off] = __float2bfloat16_rn(q_res);

    // R L2-clip (16 dims, handled by lanes 0..15 of warp 0)
    if (tid < 16) {
        size_t r_off = (static_cast<size_t>(m) * n_q_heads + h) * 16 + tid;
        f32 rv = __bfloat162float(r_in[r_off]) * __bfloat162float(r_sc[h * 16 + tid]);
        f32 rss = rv * rv;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) rss += __shfl_xor_sync(0x0000FFFFu, rss, off);
        f32 r_norm = sqrtf(rss);
        f32 r_mul = fminf(r_clip / (r_norm + 1e-6f), 1.0f);
        f32 r_res = rv * r_mul;
        if (use_log_scale) {
            r_res = __bfloat162float(__float2bfloat16_rn(r_res)) * tau;
        }
        r_out[r_off] = __float2bfloat16_rn(r_res);
    }
}

// K/V Short-Conv (static for local, dynamic for global) + K L2-clip / V RMSNorm + KV Cache Write
// Grid: (n_kv_heads, 2 [0=K, 1=V]), Block: 128 threads (1 thread per head_dim=128)
// Runs sequentially over m=0..M-1 per head so causal conv history is exact and race-free
template <bool DYNAMIC, bool INT8_KV>
__global__ void inkling_kv_sconv_write_kernel(
    const __nv_bfloat16* __restrict__ k_raw,  // [M, Hkv * 128]
    const __nv_bfloat16* __restrict__ v_raw,  // [M, Hkv * 128]
    const __nv_bfloat16* __restrict__ wk,     // static: [Hkv*128, 4]; dynamic: [4, Hkv*128]
    const __nv_bfloat16* __restrict__ wv,
    const __nv_bfloat16* __restrict__ dk_dyn, // [M, Hkv, 4] or nullptr
    const __nv_bfloat16* __restrict__ dv_dyn,
    __nv_bfloat16* __restrict__ k_hist,       // [4, Hkv * 128]
    __nv_bfloat16* __restrict__ v_hist,       // [4, Hkv * 128]
    void* __restrict__ k_cache_raw,
    void* __restrict__ v_cache_raw,
    f32* __restrict__ k_scales,
    f32* __restrict__ v_scales,
    const i32* __restrict__ d_block_table,    // nullptr if local ring buffer
    const i32* __restrict__ d_pos,
    u32 start_pos, u32 M, u32 n_kv_heads, u32 block_size,
    bool is_local_ring, u32 swa_window,
    f32 k_clip, f32 rms_eps) {
    const u32 kvh  = blockIdx.x;
    const bool is_v = (blockIdx.y == 1);
    const u32 d    = threadIdx.x; // 0..127
    const u32 lane = d & 31;
    const u32 wid  = d >> 5;
    const u32 kv_dim = n_kv_heads * 128;
    const u32 c = kvh * 128 + d;

    const __nv_bfloat16* x_src = is_v ? v_raw  : k_raw;
    const __nv_bfloat16* w_src = is_v ? wv     : wk;
    const __nv_bfloat16* dyn   = is_v ? dv_dyn : dk_dyn;
    __nv_bfloat16* hist        = is_v ? v_hist : k_hist;
    void* cache_raw            = is_v ? v_cache_raw : k_cache_raw;
    f32* scales_out            = is_v ? v_scales    : k_scales;

    const u32 p0 = d_pos ? static_cast<u32>(*d_pos) : start_pos;
    const f32 conv_scale = DYNAMIC ? rsqrtf(9.0f) : rsqrtf(5.0f);

    f32 sw0, sw1, sw2, sw3;
    if constexpr (DYNAMIC) {
        // dynamic static_w is [4, kv_dim], newest-tap-first (l=0..3)
        sw0 = __bfloat162float(w_src[0 * kv_dim + c]);
        sw1 = __bfloat162float(w_src[1 * kv_dim + c]);
        sw2 = __bfloat162float(w_src[2 * kv_dim + c]);
        sw3 = __bfloat162float(w_src[3 * kv_dim + c]);
    } else {
        // static weight is [kv_dim, 4], oldest-tap-first (k=0 -> lag 3, k=3 -> lag 0)
        sw3 = __bfloat162float(w_src[static_cast<size_t>(c) * 4 + 0]); // lag 3
        sw2 = __bfloat162float(w_src[static_cast<size_t>(c) * 4 + 1]); // lag 2
        sw1 = __bfloat162float(w_src[static_cast<size_t>(c) * 4 + 2]); // lag 1
        sw0 = __bfloat162float(w_src[static_cast<size_t>(c) * 4 + 3]); // lag 0
    }

    f32 h_m3 = (p0 >= 3) ? __bfloat162float(hist[((p0 - 3) & 15u) * kv_dim + c]) : 0.0f;
    f32 h_m2 = (p0 >= 2) ? __bfloat162float(hist[((p0 - 2) & 15u) * kv_dim + c]) : 0.0f;
    f32 h_m1 = (p0 >= 1) ? __bfloat162float(hist[((p0 - 1) & 15u) * kv_dim + c]) : 0.0f;

    __shared__ f32 sm[4];

    for (u32 m = 0; m < M; ++m) {
        u32 pos = p0 + m;
        __nv_bfloat16 cur_bf = x_src[static_cast<size_t>(m) * kv_dim + c];
        f32 cur = __bfloat162float(cur_bf);

        f32 t0 = sw0, t1 = sw1, t2 = sw2, t3 = sw3;
        if constexpr (DYNAMIC) {
            size_t doff = (static_cast<size_t>(m) * n_kv_heads + kvh) * 4;
            t0 += __bfloat162float(dyn[doff + 0]);
            t1 += __bfloat162float(dyn[doff + 1]);
            t2 += __bfloat162float(dyn[doff + 2]);
            t3 += __bfloat162float(dyn[doff + 3]);
        }
        f32 y = ((t0 + 1.0f) * cur + t1 * h_m1 + t2 * h_m2 + t3 * h_m3) * conv_scale;
        // Round once to BF16 before K-clip / V-norm (matches official reference)
        y = __bfloat162float(__float2bfloat16_rn(y));

        // Per-head L2 norm / RMSNorm across 128 dims
        f32 ss = y * y;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
        if (lane == 0) sm[wid] = ss;
        __syncthreads();
        f32 total_ss = sm[0] + sm[1] + sm[2] + sm[3];
        if (is_v) {
            y *= rsqrtf(total_ss / 128.0f + rms_eps);
        } else {
            f32 norm = sqrtf(total_ss);
            y *= fminf(k_clip / (norm + 1e-6f), 1.0f);
        }

        // Compute target KV cache slot (ring buffer for local SWA, paged block table for global)
        size_t scale_off;
        if (is_local_ring) {
            u32 rslot = pos & (swa_window - 1u); // swa_window = 1024
            scale_off = static_cast<size_t>(kvh) * swa_window + rslot;
        } else {
            u32 blk = pos / block_size;
            u32 slot = pos - blk * block_size;
            i32 phys = d_block_table[blk];
            scale_off = (static_cast<size_t>(phys) * n_kv_heads + kvh) * block_size + slot;
        }
        size_t base_off = scale_off * 128;

        if constexpr (INT8_KV) {
            f32 am = fabsf(y);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, off));
            if (lane == 0) sm[wid] = am;
            __syncthreads();
            f32 amax = fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3]));
            f32 sc = fmaxf(amax / 127.0f, 1e-6f);
            if (d == 0) scales_out[scale_off] = sc;
            int q = __float2int_rn(y / sc);
            static_cast<i8*>(cache_raw)[base_off + d] = static_cast<i8>(max(-127, min(127, q)));
        } else {
            static_cast<__nv_bfloat16*>(cache_raw)[base_off + d] = __float2bfloat16_rn(y);
        }

        hist[(pos & 15u) * kv_dim + c] = cur_bf;
        h_m3 = h_m2;
        h_m2 = h_m1;
        h_m1 = cur;
        __syncthreads();
    }
}

// ============================================================================
// 6. Final Logit Softcapping (30.0 * tanh(x / 30.0)) + Mask Padded Vocab
// ============================================================================
__global__ void inkling_logit_softcap_kernel(__nv_bfloat16* __restrict__ logits,
                                             u32 M, u32 vocab_padded, u32 vocab_unpadded,
                                             f32 cap) {
    u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    u32 row = blockIdx.y;
    if (row >= M || col >= vocab_padded) return;
    size_t off = static_cast<size_t>(row) * vocab_padded + col;
    if (col >= vocab_unpadded) {
        logits[off] = __float2bfloat16_rn(-1e30f);
    } else if (cap > 0.0f) {
        f32 v = __bfloat162float(logits[off]);
        logits[off] = __float2bfloat16_rn(cap * tanhf(v / cap));
    }
}

// ============================================================================
// 8. Zero-Extra-VRAM Multi-Matrix Fused BF16 GEMV (6-way Attn & 2-way Ctrl/KV)
// ============================================================================
template <int MAX_M>
__global__ void inkling_fused_attn_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ wq, const __nv_bfloat16* __restrict__ wr,
    const __nv_bfloat16* __restrict__ wg, const __nv_bfloat16* __restrict__ ws,
    const __nv_bfloat16* __restrict__ wk, const __nv_bfloat16* __restrict__ wv,
    __nv_bfloat16* __restrict__ oq, __nv_bfloat16* __restrict__ or_out,
    __nv_bfloat16* __restrict__ og, __nv_bfloat16* __restrict__ os,
    __nv_bfloat16* __restrict__ ok, __nv_bfloat16* __restrict__ ov,
    u32 M, u32 Qdim, u32 Rdim, u32 Hq, u32 KVdim, u32 K, bool include_kv) {
    const u32 wid  = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 grow = blockIdx.x * 4 + wid;

    const u32 c0 = Qdim;
    const u32 c1 = c0 + Rdim;
    const u32 c2 = c1 + Qdim;
    const u32 c3 = c2 + Hq;
    const u32 c4 = c3 + (include_kv ? KVdim : 0);
    const u32 c5 = c4 + (include_kv ? KVdim : 0);
    if (grow >= c5) return;

    const __nv_bfloat16* W = nullptr;
    __nv_bfloat16* O = nullptr;
    u32 lrow = 0, N = 0;
    if (grow < c0)      { W = wq; O = oq;     lrow = grow;      N = Qdim;  }
    else if (grow < c1) { W = wr; O = or_out; lrow = grow - c0; N = Rdim;  }
    else if (grow < c2) { W = wg; O = og;     lrow = grow - c1; N = Qdim;  }
    else if (grow < c3) { W = ws; O = os;     lrow = grow - c2; N = Hq;    }
    else if (grow < c4) { W = wk; O = ok;     lrow = grow - c3; N = KVdim; }
    else                { W = wv; O = ov;     lrow = grow - c4; N = KVdim; }

    const u32 K8 = K >> 3;
    const auto* w_row = reinterpret_cast<const uint4*>(W + static_cast<size_t>(lrow) * K);
    f32 acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (u32 idx = lane; idx < K8; idx += 32) {
        uint4 w0 = __ldg(w_row + idx);
        const auto* wb0 = reinterpret_cast<const __nv_bfloat162*>(&w0);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < static_cast<int>(M)) {
                uint4 x0 = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K)[idx];
                const auto* xb0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 fw = __bfloat1622float2(wb0[j]);
                    float2 fx = __bfloat1622float2(xb0[j]);
                    acc[m] = fmaf(fw.x, fx.x, fmaf(fw.y, fx.y, acc[m]));
                }
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < static_cast<int>(M)) {
            f32 v = acc[m];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
            if (lane == 0) O[static_cast<size_t>(m) * N + lrow] = __float2bfloat16_rn(v);
        }
    }
}

template <int MAX_M>
__global__ void inkling_fused_2gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w0, const __nv_bfloat16* __restrict__ w1,
    __nv_bfloat16* __restrict__ o0, __nv_bfloat16* __restrict__ o1,
    u32 M, u32 N0, u32 N1, u32 K) {
    const u32 wid  = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 grow = blockIdx.x * 4 + wid;
    if (grow >= N0 + N1) return;
    const bool sec = (grow >= N0);
    const u32 lrow = sec ? (grow - N0) : grow;
    const u32 N    = sec ? N1 : N0;
    const __nv_bfloat16* W = sec ? w1 : w0;
    __nv_bfloat16* O = sec ? o1 : o0;

    const u32 K8 = K >> 3;
    const auto* w_row = reinterpret_cast<const uint4*>(W + static_cast<size_t>(lrow) * K);
    f32 acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (u32 idx = lane; idx < K8; idx += 32) {
        uint4 wv = __ldg(w_row + idx);
        const auto* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < static_cast<int>(M)) {
                uint4 xv = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K)[idx];
                const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 fw = __bfloat1622float2(wb[j]);
                    float2 fx = __bfloat1622float2(xb[j]);
                    acc[m] = fmaf(fw.x, fx.x, fmaf(fw.y, fx.y, acc[m]));
                }
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < static_cast<int>(M)) {
            f32 v = acc[m];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
            if (lane == 0) O[static_cast<size_t>(m) * N + lrow] = __float2bfloat16_rn(v);
        }
    }
}

// 8b. 1-Launch 3-Stream Fused FP8 Attention GEMV (Q/R/Gate/Sink on x_q, K/K_dyn on x_k, V/V_dyn on x_v -> -80 launches/tok)
template <int MAX_M>
__global__ void inkling_fused_attn_gemv_fp8_kernel(
    const __nv_bfloat16* __restrict__ x_q,
    const __nv_bfloat16* __restrict__ x_k,
    const __nv_bfloat16* __restrict__ x_v,
    const u8* __restrict__ wq, const f32* __restrict__ sq,
    const u8* __restrict__ wr, const f32* __restrict__ sr,
    const u8* __restrict__ wg, const f32* __restrict__ sg,
    const __nv_bfloat16* __restrict__ ws_bf16,
    const u8* __restrict__ wk, const f32* __restrict__ sk,
    const u8* __restrict__ wv, const f32* __restrict__ sv,
    const __nv_bfloat16* __restrict__ wk_dyn,
    const __nv_bfloat16* __restrict__ wv_dyn,
    __nv_bfloat16* __restrict__ oq, __nv_bfloat16* __restrict__ or_out,
    __nv_bfloat16* __restrict__ og, __nv_bfloat16* __restrict__ os,
    __nv_bfloat16* __restrict__ ok, __nv_bfloat16* __restrict__ ov,
    __nv_bfloat16* __restrict__ ok_dyn, __nv_bfloat16* __restrict__ ov_dyn,
    u32 M, u32 Qdim, u32 Rdim, u32 Hq, u32 KVdim, u32 DynDim, u32 K) {
    const u32 wid  = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 grow = blockIdx.x * 4 + wid;

    const u32 c0 = Qdim;
    const u32 c1 = c0 + Rdim;
    const u32 c2 = c1 + Qdim;
    const u32 c3 = c2 + Hq;
    const u32 c4 = c3 + KVdim;
    const u32 c5 = c4 + KVdim;
    const u32 c6 = c5 + DynDim;
    const u32 c7 = c6 + DynDim;
    if (grow >= c7) return;

    // Small BF16 projections: sink (24 rows on x_q), k_sconv_proj (32 rows on x_k), v_sconv_proj (32 rows on x_v)
    if ((grow >= c2 && grow < c3) || grow >= c5) {
        const __nv_bfloat16* x_src = nullptr;
        const __nv_bfloat16* w_bf  = nullptr;
        __nv_bfloat16* O = nullptr;
        u32 lrow = 0, N = 0;
        if (grow < c3)      { x_src = x_q; w_bf = ws_bf16; O = os;     lrow = grow - c2; N = Hq; }
        else if (grow < c6) { x_src = x_k; w_bf = wk_dyn;  O = ok_dyn; lrow = grow - c5; N = DynDim; }
        else                { x_src = x_v; w_bf = wv_dyn;  O = ov_dyn; lrow = grow - c6; N = DynDim; }
        const u32 K8 = K >> 3;
        const auto* w_row = reinterpret_cast<const uint4*>(w_bf + static_cast<size_t>(lrow) * K);
        f32 acc[MAX_M];
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;
        for (u32 idx = lane; idx < K8; idx += 32) {
            uint4 w0 = __ldg(w_row + idx);
            const auto* wb0 = reinterpret_cast<const __nv_bfloat162*>(&w0);
            #pragma unroll 1
            for (int m = 0; m < MAX_M; ++m) {
                if (m < static_cast<int>(M)) {
                    uint4 x0 = reinterpret_cast<const uint4*>(x_src + static_cast<size_t>(m) * K)[idx];
                    const auto* xb0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        float2 fw = __bfloat1622float2(wb0[j]);
                        float2 fx = __bfloat1622float2(xb0[j]);
                        acc[m] = fmaf(fw.x, fx.x, fmaf(fw.y, fx.y, acc[m]));
                    }
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < static_cast<int>(M)) {
                f32 v = acc[m];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
                if (lane == 0) O[static_cast<size_t>(m) * N + lrow] = __float2bfloat16_rn(v);
            }
        }
        return;
    }

    const __nv_bfloat16* x_src = x_q;
    const u8* W = nullptr;
    const f32* S = nullptr;
    __nv_bfloat16* O = nullptr;
    u32 lrow = 0, N = 0;
    if (grow < c0)      { x_src = x_q; W = wq; S = sq; O = oq;     lrow = grow;      N = Qdim;  }
    else if (grow < c1) { x_src = x_q; W = wr; S = sr; O = or_out; lrow = grow - c0; N = Rdim;  }
    else if (grow < c2) { x_src = x_q; W = wg; S = sg; O = og;     lrow = grow - c1; N = Qdim;  }
    else if (grow < c4) { x_src = x_k; W = wk; S = sk; O = ok;     lrow = grow - c3; N = KVdim; }
    else                { x_src = x_v; W = wv; S = sv; O = ov;     lrow = grow - c4; N = KVdim; }

    const u32 K16 = K >> 4;
    const auto* w16_row = reinterpret_cast<const uint4*>(W + static_cast<size_t>(lrow) * K);
    f32 acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (u32 idx = lane; idx < K16; idx += 32) {
        uint4 wv16 = __ldg(w16_row + idx);
        const auto* wf = reinterpret_cast<const __nv_fp8x2_e4m3*>(&wv16);
        #pragma unroll 1
        for (int m = 0; m < MAX_M; ++m) {
            if (m < static_cast<int>(M)) {
                const auto* x8 = reinterpret_cast<const uint4*>(x_src + static_cast<size_t>(m) * K);
                uint4 x0 = x8[idx * 2 + 0];
                uint4 x1 = x8[idx * 2 + 1];
                const auto* xb0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
                const auto* xb1 = reinterpret_cast<const __nv_bfloat162*>(&x1);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 fw = static_cast<float2>(wf[j]);
                    float2 fx = __bfloat1622float2(xb0[j]);
                    acc[m] = fmaf(fw.x, fx.x, fmaf(fw.y, fx.y, acc[m]));
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 fw = static_cast<float2>(wf[4 + j]);
                    float2 fx = __bfloat1622float2(xb1[j]);
                    acc[m] = fmaf(fw.x, fx.x, fmaf(fw.y, fx.y, acc[m]));
                }
            }
        }
    }
    f32 sc = __ldg(S + lrow);
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < static_cast<int>(M)) {
            f32 v = acc[m];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
            if (lane == 0) O[static_cast<size_t>(m) * N + lrow] = __float2bfloat16_rn(v * sc);
        }
    }
}

// 8c. Fused M=1..8 FP8 wo_proj + attn_sconv_hidden (eliminates 48 launches & 48 round-trips on BOTH base & MTP verify)
template <int MAX_M>
__global__ void inkling_wo_sconv_fp8_kernel(
    const __nv_bfloat16* __restrict__ x,      // [M, Qdim=3072]
    const u8* __restrict__ wo_fp8,            // [H=2304, Qdim=3072]
    const f32* __restrict__ so_fp8,           // [H]
    const __nv_bfloat16* __restrict__ sconv_w,// [H, 4]
    __nv_bfloat16* __restrict__ hist_ring,    // [16, H]
    __nv_bfloat16* __restrict__ out,          // [M, H]
    const i32* __restrict__ d_pos, u32 start_pos, u32 M, u32 H, u32 Qdim) {
    const u32 wid  = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 h    = blockIdx.x * 4 + wid;
    if (h >= H) return;

    const u32 K16 = Qdim >> 4;
    const auto* w16 = reinterpret_cast<const uint4*>(wo_fp8 + static_cast<size_t>(h) * Qdim);
    f32 acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (u32 idx = lane; idx < K16; idx += 32) {
        uint4 wv = __ldg(w16 + idx);
        const auto* wf = reinterpret_cast<const __nv_fp8x2_e4m3*>(&wv);
        #pragma unroll 1
        for (int m = 0; m < MAX_M; ++m) {
            if (m < static_cast<int>(M)) {
                const auto* x8 = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * Qdim);
                uint4 x0 = x8[idx * 2 + 0];
                uint4 x1 = x8[idx * 2 + 1];
                const auto* xb0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
                const auto* xb1 = reinterpret_cast<const __nv_bfloat162*>(&x1);
                f32 s = 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 fw = static_cast<float2>(wf[j]);
                    float2 fx = __bfloat1622float2(xb0[j]);
                    s = fmaf(fw.x, fx.x, fmaf(fw.y, fx.y, s));
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 fw = static_cast<float2>(wf[4 + j]);
                    float2 fx = __bfloat1622float2(xb1[j]);
                    s = fmaf(fw.x, fx.x, fmaf(fw.y, fx.y, s));
                }
                acc[m] += s;
            }
        }
    }
    f32 sc = __ldg(so_fp8 + h);
    f32 cur_vals[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        f32 v = acc[m];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
        cur_vals[m] = v * sc;
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
            out[static_cast<size_t>(m) * H + h] = __float2bfloat16_rn(y);
            h_m3 = h_m2; h_m2 = h_m1; h_m1 = c0;
        }
    }
}

// 8d. Fused 3-Stream Looped-Core Controller GEMVs (cuts 24 x 4 = 96 launches/tok, 0 extra VRAM)
template <int MAX_M>
__global__ void inkling_fused_3ctrl_down_kernel(
    const __nv_bfloat16* __restrict__ pair_3x2H, // [3, M, 2H]
    const __nv_bfloat16* __restrict__ wd0,
    const __nv_bfloat16* __restrict__ wd1,
    const __nv_bfloat16* __restrict__ wd2,
    __nv_bfloat16* __restrict__ out_3xR,         // [3, M, rank]
    u32 M, u32 rank, u32 K2H) {
    const u32 wid  = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 grow = blockIdx.x * 4 + wid;
    const u32 s    = grow / rank;
    const u32 r    = grow - s * rank;
    if (s >= 3) return;

    const __nv_bfloat16* W = (s == 0) ? wd0 : ((s == 1) ? wd1 : wd2);
    const __nv_bfloat16* X = pair_3x2H + static_cast<size_t>(s) * M * K2H;
    __nv_bfloat16* O       = out_3xR   + static_cast<size_t>(s) * M * rank;

    const u32 K8 = K2H >> 3;
    const auto* w_row = reinterpret_cast<const uint4*>(W + static_cast<size_t>(r) * K2H);
    f32 acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (u32 idx = lane; idx < K8; idx += 32) {
        uint4 wv = __ldg(w_row + idx);
        const auto* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
        for (int m = 0; m < MAX_M; ++m) {
            if (m < static_cast<int>(M)) {
                uint4 xv = reinterpret_cast<const uint4*>(X + static_cast<size_t>(m) * K2H)[idx];
                const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 fw = __bfloat1622float2(wb[j]);
                    float2 fx = __bfloat1622float2(xb[j]);
                    acc[m] = fmaf(fw.x, fx.x, fmaf(fw.y, fx.y, acc[m]));
                }
            }
        }
    }
    for (int m = 0; m < MAX_M; ++m) {
        if (m < static_cast<int>(M)) {
            f32 v = acc[m];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
            if (lane == 0) O[static_cast<size_t>(m) * rank + r] = __float2bfloat16_rn(v);
        }
    }
}

template <int MAX_M>
__global__ void inkling_fused_3ctrl_gf_kernel(
    const __nv_bfloat16* __restrict__ h_3xR,   // [3, M, rank]
    const __nv_bfloat16* __restrict__ wg0, const __nv_bfloat16* __restrict__ wf0,
    const __nv_bfloat16* __restrict__ wg1, const __nv_bfloat16* __restrict__ wf1,
    const __nv_bfloat16* __restrict__ wg2, const __nv_bfloat16* __restrict__ wf2,
    __nv_bfloat16* __restrict__ out_3x2H,      // [3][gain(M*H), forget(M*H)]
    u32 M, u32 H, u32 rank) {
    const u32 wid  = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 grow = blockIdx.x * 4 + wid;
    const u32 twoH = 2 * H;
    const u32 s    = grow / twoH;
    const u32 rem  = grow - s * twoH;
    if (s >= 3) return;
    const bool is_f = (rem >= H);
    const u32 h     = is_f ? (rem - H) : rem;

    const __nv_bfloat16* W = (s == 0) ? (is_f ? wf0 : wg0)
                           : ((s == 1) ? (is_f ? wf1 : wg1)
                                       : (is_f ? wf2 : wg2));
    const __nv_bfloat16* X = h_3xR + static_cast<size_t>(s) * M * rank;
    __nv_bfloat16* O       = out_3x2H + static_cast<size_t>(s) * M * twoH + (is_f ? static_cast<size_t>(M) * H : 0);

    const u32 R8 = rank >> 3; // 64 / 8 = 8
    const auto* w_row = reinterpret_cast<const uint4*>(W + static_cast<size_t>(h) * rank);
    f32 acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    if (lane < R8) {
        uint4 wv = __ldg(w_row + lane);
        const auto* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
        for (int m = 0; m < MAX_M; ++m) {
            if (m < static_cast<int>(M)) {
                uint4 xv = reinterpret_cast<const uint4*>(X + static_cast<size_t>(m) * rank)[lane];
                const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 fw = __bfloat1622float2(wb[j]);
                    float2 fx = __bfloat1622float2(xb[j]);
                    acc[m] = fmaf(fw.x, fx.x, fmaf(fw.y, fx.y, acc[m]));
                }
            }
        }
    }
    for (int m = 0; m < MAX_M; ++m) {
        if (m < static_cast<int>(M)) {
            f32 v = acc[m];
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
            if (lane == 0) O[static_cast<size_t>(m) * H + h] = __float2bfloat16_rn(v);
        }
    }
}

// 8e. 1-launch 3-stream blend_rmsnorm (replaces 3 sequential launches per looped core visit -> -48 launches/tok)
// gf_3x2H is laid out as [s][gain(M*D), forget(M*D)], matching inkling_fused_3ctrl_gf_kernel
__global__ void inkling_blend_rmsnorm_3stream_kernel(
    const f32* __restrict__ dest,
    const f32* __restrict__ msg,
    const __nv_bfloat16* __restrict__ gf_3x2H,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ out,
    bool is_local, u32 M, u32 D, f32 eps) {
    const u32 s = blockIdx.x; // 0..2
    const u32 m = blockIdx.y;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31, wid = tid >> 5;

    const f32* d_row = dest + (static_cast<size_t>(is_local ? 0 : s) * M + m) * D;
    const f32* m_row = msg  + (static_cast<size_t>(s) * M + m) * D;
    const __nv_bfloat16* gs_base = gf_3x2H + static_cast<size_t>(s) * M * (2 * D);
    const __nv_bfloat16* g_row = gs_base + static_cast<size_t>(m) * D;
    const __nv_bfloat16* f_row = gs_base + static_cast<size_t>(M + m) * D;
    const __nv_bfloat16* nw    = norm_w  + static_cast<size_t>(is_local ? 0 : s) * D;

    f32 ss = 0.0f;
    for (u32 d = tid; d < D; d += 256) {
        f32 g = tanhf(__bfloat162float(g_row[d]));
        f32 f = tanhf(__bfloat162float(f_row[d]));
        f32 x = (1.0f - f) * d_row[d] + g * m_row[d];
        ss = fmaf(x, x, ss);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xFFFFFFFFu, ss, off);
    __shared__ f32 sm[8];
    if (lane == 0) sm[wid] = ss;
    __syncthreads();
    if (tid == 0) {
        f32 t = 0.0f;
        for (int w = 0; w < 8; ++w) t += sm[w];
        sm[0] = rsqrtf(t / static_cast<f32>(D) + eps);
    }
    __syncthreads();
    f32 rstd = sm[0];
    __nv_bfloat16* o_row = out + (static_cast<size_t>(s) * M + m) * D;
    for (u32 d = tid; d < D; d += 256) {
        f32 g = tanhf(__bfloat162float(g_row[d]));
        f32 f = tanhf(__bfloat162float(f_row[d]));
        f32 x = (1.0f - f) * d_row[d] + g * m_row[d];
        o_row[d] = __float2bfloat16_rn(x * rstd * __bfloat162float(nw[d]));
    }
}

// 8f. 1-launch concurrent qr_clip + kv_sconv_write (blockIdx.x < n_q_heads -> Q/R, else K/V -> -48 launches/tok)
// Grid: (n_q_heads + 2 * n_kv_heads) = 40 blocks x 128 threads (for M<=8, each Q/R block loops m=0..M-1 too)
template <bool DYNAMIC, bool INT8_KV>
__global__ void inkling_qr_kv_fused_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ r_in,
    const __nv_bfloat16* __restrict__ q_sc,
    const __nv_bfloat16* __restrict__ r_sc,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ r_out,
    const __nv_bfloat16* __restrict__ k_raw,
    const __nv_bfloat16* __restrict__ v_raw,
    const __nv_bfloat16* __restrict__ wk,
    const __nv_bfloat16* __restrict__ wv,
    const __nv_bfloat16* __restrict__ dk_dyn,
    const __nv_bfloat16* __restrict__ dv_dyn,
    __nv_bfloat16* __restrict__ k_hist,
    __nv_bfloat16* __restrict__ v_hist,
    void* __restrict__ k_cache_raw,
    void* __restrict__ v_cache_raw,
    f32* __restrict__ k_scales,
    f32* __restrict__ v_scales,
    const i32* __restrict__ d_block_table,
    const i32* __restrict__ d_pos,
    u32 start_pos, u32 M, u32 n_q_heads, u32 n_kv_heads, u32 block_size,
    bool is_local_ring, u32 swa_window,
    f32 q_clip, f32 r_clip, bool use_log_scale, u32 n_floor, f32 alpha, f32 rms_eps) {
    const u32 bidx = blockIdx.x;
    const u32 tid  = threadIdx.x; // 0..127
    const u32 lane = tid & 31, wid = tid >> 5;
    const u32 p0   = d_pos ? static_cast<u32>(*d_pos) : start_pos;
    __shared__ f32 sm[4];

    if (bidx < n_q_heads) {
        // Branch A: Q/R L2-clip (+ tau) for Q head h = bidx across m = 0..M-1
        const u32 h = bidx;
        f32 qsc_v = __bfloat162float(q_sc[h * 128 + tid]);
        f32 rsc_v = (tid < 16) ? __bfloat162float(r_sc[h * 16 + tid]) : 0.0f;
        for (u32 m = 0; m < M; ++m) {
            u32 pos = p0 + m;
            f32 tau = 1.0f;
            if (use_log_scale) {
                f32 ratio = static_cast<f32>(pos + 1) / static_cast<f32>(n_floor);
                if (ratio > 1.0f) tau = 1.0f + alpha * logf(ratio);
            }
            size_t q_off = (static_cast<size_t>(m) * n_q_heads + h) * 128 + tid;
            f32 qv = __bfloat162float(q_in[q_off]) * qsc_v;
            f32 qss = qv * qv;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) qss += __shfl_xor_sync(0xFFFFFFFFu, qss, off);
            if (lane == 0) sm[wid] = qss;
            __syncthreads();
            f32 q_norm = sqrtf(sm[0] + sm[1] + sm[2] + sm[3]);
            f32 q_res = qv * fminf(q_clip / (q_norm + 1e-6f), 1.0f);
            if (use_log_scale) q_res = __bfloat162float(__float2bfloat16_rn(q_res)) * tau;
            q_out[q_off] = __float2bfloat16_rn(q_res);

            if (tid < 16) {
                size_t r_off = (static_cast<size_t>(m) * n_q_heads + h) * 16 + tid;
                f32 rv = __bfloat162float(r_in[r_off]) * rsc_v;
                f32 rss = rv * rv;
                #pragma unroll
                for (int off = 8; off > 0; off >>= 1) rss += __shfl_xor_sync(0x0000FFFFu, rss, off);
                f32 r_norm = sqrtf(rss);
                f32 r_res = rv * fminf(r_clip / (r_norm + 1e-6f), 1.0f);
                if (use_log_scale) r_res = __bfloat162float(__float2bfloat16_rn(r_res)) * tau;
                r_out[r_off] = __float2bfloat16_rn(r_res);
            }
            __syncthreads();
        }
        return;
    }

    // Branch B: K/V Short-Conv + Clip/Norm + KV Cache Write (identical to inkling_kv_sconv_write_kernel)
    const u32 kv_blk = bidx - n_q_heads;
    const bool is_v  = (kv_blk >= n_kv_heads);
    const u32 kvh    = is_v ? (kv_blk - n_kv_heads) : kv_blk;
    const u32 d      = tid;
    const u32 kv_dim = n_kv_heads * 128;
    const u32 c      = kvh * 128 + d;

    const __nv_bfloat16* x_src = is_v ? v_raw  : k_raw;
    const __nv_bfloat16* w_src = is_v ? wv     : wk;
    const __nv_bfloat16* dyn   = is_v ? dv_dyn : dk_dyn;
    __nv_bfloat16* hist        = is_v ? v_hist : k_hist;
    void* cache_raw            = is_v ? v_cache_raw : k_cache_raw;
    f32* scales_out            = is_v ? v_scales    : k_scales;
    const f32 conv_scale       = DYNAMIC ? rsqrtf(9.0f) : rsqrtf(5.0f);

    f32 sw0, sw1, sw2, sw3;
    if constexpr (DYNAMIC) {
        sw0 = __bfloat162float(w_src[0 * kv_dim + c]);
        sw1 = __bfloat162float(w_src[1 * kv_dim + c]);
        sw2 = __bfloat162float(w_src[2 * kv_dim + c]);
        sw3 = __bfloat162float(w_src[3 * kv_dim + c]);
    } else {
        sw3 = __bfloat162float(w_src[static_cast<size_t>(c) * 4 + 0]);
        sw2 = __bfloat162float(w_src[static_cast<size_t>(c) * 4 + 1]);
        sw1 = __bfloat162float(w_src[static_cast<size_t>(c) * 4 + 2]);
        sw0 = __bfloat162float(w_src[static_cast<size_t>(c) * 4 + 3]);
    }
    f32 h_m3 = (p0 >= 3) ? __bfloat162float(hist[((p0 - 3) & 15u) * kv_dim + c]) : 0.0f;
    f32 h_m2 = (p0 >= 2) ? __bfloat162float(hist[((p0 - 2) & 15u) * kv_dim + c]) : 0.0f;
    f32 h_m1 = (p0 >= 1) ? __bfloat162float(hist[((p0 - 1) & 15u) * kv_dim + c]) : 0.0f;

    for (u32 m = 0; m < M; ++m) {
        u32 pos = p0 + m;
        __nv_bfloat16 cur_bf = x_src[static_cast<size_t>(m) * kv_dim + c];
        f32 cur = __bfloat162float(cur_bf);
        f32 t0 = sw0, t1 = sw1, t2 = sw2, t3 = sw3;
        if constexpr (DYNAMIC) {
            size_t doff = (static_cast<size_t>(m) * n_kv_heads + kvh) * 4;
            t0 += __bfloat162float(dyn[doff + 0]);
            t1 += __bfloat162float(dyn[doff + 1]);
            t2 += __bfloat162float(dyn[doff + 2]);
            t3 += __bfloat162float(dyn[doff + 3]);
        }
        f32 y = ((t0 + 1.0f) * cur + t1 * h_m1 + t2 * h_m2 + t3 * h_m3) * conv_scale;
        y = __bfloat162float(__float2bfloat16_rn(y));

        f32 ss = y * y;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
        if (lane == 0) sm[wid] = ss;
        __syncthreads();
        f32 total_ss = sm[0] + sm[1] + sm[2] + sm[3];
        if (is_v) y *= rsqrtf(total_ss / 128.0f + rms_eps);
        else      y *= fminf(q_clip / (sqrtf(total_ss) + 1e-6f), 1.0f);

        size_t scale_off;
        if (is_local_ring) {
            scale_off = static_cast<size_t>(kvh) * swa_window + (pos & (swa_window - 1u));
        } else {
            u32 blk = pos / block_size;
            u32 slot = pos - blk * block_size;
            scale_off = (static_cast<size_t>(d_block_table[blk]) * n_kv_heads + kvh) * block_size + slot;
        }
        size_t base_off = scale_off * 128;
        if constexpr (INT8_KV) {
            f32 am = fabsf(y);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, off));
            if (lane == 0) sm[wid] = am;
            __syncthreads();
            f32 amax = fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3]));
            f32 sc = fmaxf(amax / 127.0f, 1e-6f);
            if (d == 0) scales_out[scale_off] = sc;
            int q = __float2int_rn(y / sc);
            static_cast<i8*>(cache_raw)[base_off + d] = static_cast<i8>(max(-127, min(127, q)));
        } else {
            static_cast<__nv_bfloat16*>(cache_raw)[base_off + d] = __float2bfloat16_rn(y);
        }
        hist[(pos & 15u) * kv_dim + c] = cur_bf;
        h_m3 = h_m2; h_m2 = h_m1; h_m1 = cur;
        __syncthreads();
    }
}

} // namespace


void launch_inkling_attn_res(const __nv_bfloat16* const* d_stack_ptrs,
                             u32 n_stack,
                             const __nv_bfloat16* last_ptr,
                             const __nv_bfloat16* addend_ptr,
                             const f32* queries,
                             const __nv_bfloat16* norm_w,
                             __nv_bfloat16* out_bf16,
                             f32* out_f32,
                             u32 num_q, u32 M, u32 D, f32 eps, bool normalize_values,
                             cudaStream_t stream) {
    dim3 grid(num_q, M);
    inkling_attn_res_kernel<<<grid, 256, 0, stream>>>(
        d_stack_ptrs, n_stack, last_ptr, addend_ptr, queries, norm_w,
        out_bf16, out_f32, M, D, eps, normalize_values);
}

void launch_inkling_pair_rmsnorm(const f32* dest, const f32* msg,
                                 __nv_bfloat16* out,
                                 bool broadcast_dest, u32 S, u32 M, u32 D, f32 eps,
                                 cudaStream_t stream) {
    dim3 grid(S, M);
    inkling_pair_rmsnorm_kernel<<<grid, 256, 0, stream>>>(dest, msg, out, broadcast_dest, M, D, eps);
}

void launch_inkling_blend_rmsnorm(const f32* dest, const f32* msg,
                                  const __nv_bfloat16* gain_bf16,
                                  const __nv_bfloat16* forget_bf16,
                                  const __nv_bfloat16* norm_w,
                                  __nv_bfloat16* out,
                                  bool broadcast_dest, bool broadcast_norm,
                                  u32 S, u32 M, u32 D, f32 eps,
                                  cudaStream_t stream) {
    dim3 grid(S, M);
    inkling_blend_rmsnorm_kernel<<<grid, 256, 0, stream>>>(
        dest, msg, gain_bf16, forget_bf16, norm_w, out, broadcast_dest, broadcast_norm, M, D, eps);
}

void launch_inkling_sconv_hidden(const __nv_bfloat16* x, const __nv_bfloat16* w,
                                 __nv_bfloat16* hist, __nv_bfloat16* out,
                                 const __nv_bfloat16* res1, const __nv_bfloat16* res2,
                                 const i32* d_pos, u32 start_pos, u32 M, u32 D,
                                 cudaStream_t stream) {
    inkling_sconv_hidden_kernel<<<(D + 255) / 256, 256, 0, stream>>>(
        x, w, hist, out, res1, res2, d_pos, start_pos, M, D);
}

void launch_inkling_qr_clip(const __nv_bfloat16* q_in, const __nv_bfloat16* r_in,
                            const __nv_bfloat16* q_sc, const __nv_bfloat16* r_sc,
                            __nv_bfloat16* q_out, __nv_bfloat16* r_out,
                            const i32* d_pos, u32 start_pos, u32 M, u32 n_q_heads,
                            f32 q_clip, f32 r_clip,
                            bool use_log_scale, u32 n_floor, f32 alpha,
                            cudaStream_t stream) {
    dim3 grid(n_q_heads, M);
    inkling_qr_clip_kernel<<<grid, 128, 0, stream>>>(
        q_in, r_in, q_sc, r_sc, q_out, r_out, d_pos, start_pos, M, n_q_heads,
        q_clip, r_clip, use_log_scale, n_floor, alpha);
}

void launch_inkling_kv_sconv_write(const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
                                   const __nv_bfloat16* wk, const __nv_bfloat16* wv,
                                   const __nv_bfloat16* dk_dyn, const __nv_bfloat16* dv_dyn,
                                   __nv_bfloat16* k_hist, __nv_bfloat16* v_hist,
                                   void* k_cache_raw, void* v_cache_raw,
                                   f32* k_scales, f32* v_scales,
                                   const i32* d_block_table,
                                   const i32* d_pos, u32 start_pos, u32 M, u32 n_kv_heads,
                                   u32 block_size, bool is_local_ring, u32 swa_window,
                                   bool dynamic, bool int8_kv, f32 k_clip, f32 rms_eps,
                                   cudaStream_t stream) {
    dim3 grid(n_kv_heads, 2);
    if (dynamic && int8_kv) {
        inkling_kv_sconv_write_kernel<true, true><<<grid, 128, 0, stream>>>(
            k_raw, v_raw, wk, wv, dk_dyn, dv_dyn, k_hist, v_hist,
            k_cache_raw, v_cache_raw, k_scales, v_scales, d_block_table,
            d_pos, start_pos, M, n_kv_heads, block_size, is_local_ring, swa_window, k_clip, rms_eps);
    } else if (dynamic && !int8_kv) {
        inkling_kv_sconv_write_kernel<true, false><<<grid, 128, 0, stream>>>(
            k_raw, v_raw, wk, wv, dk_dyn, dv_dyn, k_hist, v_hist,
            k_cache_raw, v_cache_raw, k_scales, v_scales, d_block_table,
            d_pos, start_pos, M, n_kv_heads, block_size, is_local_ring, swa_window, k_clip, rms_eps);
    } else if (!dynamic && int8_kv) {
        inkling_kv_sconv_write_kernel<false, true><<<grid, 128, 0, stream>>>(
            k_raw, v_raw, wk, wv, dk_dyn, dv_dyn, k_hist, v_hist,
            k_cache_raw, v_cache_raw, k_scales, v_scales, d_block_table,
            d_pos, start_pos, M, n_kv_heads, block_size, is_local_ring, swa_window, k_clip, rms_eps);
    } else {
        inkling_kv_sconv_write_kernel<false, false><<<grid, 128, 0, stream>>>(
            k_raw, v_raw, wk, wv, dk_dyn, dv_dyn, k_hist, v_hist,
            k_cache_raw, v_cache_raw, k_scales, v_scales, d_block_table,
            d_pos, start_pos, M, n_kv_heads, block_size, is_local_ring, swa_window, k_clip, rms_eps);
    }
}

void launch_inkling_logit_softcap(__nv_bfloat16* logits, u32 M,
                                  u32 vocab_padded, u32 vocab_unpadded, f32 cap,
                                  cudaStream_t stream) {
    dim3 grid((vocab_padded + 255) / 256, M);
    inkling_logit_softcap_kernel<<<grid, 256, 0, stream>>>(logits, M, vocab_padded, vocab_unpadded, cap);
}

void launch_inkling_fused_attn_gemv(const __nv_bfloat16* x,
                                    const __nv_bfloat16* wq, const __nv_bfloat16* wr,
                                    const __nv_bfloat16* wg, const __nv_bfloat16* ws,
                                    const __nv_bfloat16* wk, const __nv_bfloat16* wv,
                                    __nv_bfloat16* oq, __nv_bfloat16* or_out,
                                    __nv_bfloat16* og, __nv_bfloat16* os,
                                    __nv_bfloat16* ok, __nv_bfloat16* ov,
                                    u32 M, u32 Qdim, u32 Rdim, u32 Hq, u32 KVdim, u32 K,
                                    bool include_kv, cudaStream_t stream) {
    u32 total_n = Qdim + Rdim + Qdim + Hq + (include_kv ? 2 * KVdim : 0);
    u32 blocks = (total_n + 3) / 4;
    if (M == 1) {
        inkling_fused_attn_gemv_kernel<1><<<blocks, 128, 0, stream>>>(
            x, wq, wr, wg, ws, wk, wv, oq, or_out, og, os, ok, ov, M, Qdim, Rdim, Hq, KVdim, K, include_kv);
    } else if (M <= 4) {
        inkling_fused_attn_gemv_kernel<4><<<blocks, 128, 0, stream>>>(
            x, wq, wr, wg, ws, wk, wv, oq, or_out, og, os, ok, ov, M, Qdim, Rdim, Hq, KVdim, K, include_kv);
    } else {
        inkling_fused_attn_gemv_kernel<8><<<blocks, 128, 0, stream>>>(
            x, wq, wr, wg, ws, wk, wv, oq, or_out, og, os, ok, ov, M, Qdim, Rdim, Hq, KVdim, K, include_kv);
    }
}

void launch_inkling_fused_attn_gemv_fp8(const __nv_bfloat16* x_q,
                                        const __nv_bfloat16* x_k,
                                        const __nv_bfloat16* x_v,
                                        const u8* wq, const f32* sq,
                                        const u8* wr, const f32* sr,
                                        const u8* wg, const f32* sg,
                                        const __nv_bfloat16* ws_bf16,
                                        const u8* wk, const f32* sk,
                                        const u8* wv, const f32* sv,
                                        const __nv_bfloat16* wk_dyn,
                                        const __nv_bfloat16* wv_dyn,
                                        __nv_bfloat16* oq, __nv_bfloat16* or_out,
                                        __nv_bfloat16* og, __nv_bfloat16* os,
                                        __nv_bfloat16* ok, __nv_bfloat16* ov,
                                        __nv_bfloat16* ok_dyn, __nv_bfloat16* ov_dyn,
                                        u32 M, u32 Qdim, u32 Rdim, u32 Hq, u32 KVdim, u32 DynDim, u32 K,
                                        cudaStream_t stream) {
    u32 total_n = Qdim + Rdim + Qdim + Hq + 2 * KVdim + 2 * DynDim;
    u32 blocks = (total_n + 3) / 4;
    if (M == 1) {
        inkling_fused_attn_gemv_fp8_kernel<1><<<blocks, 128, 0, stream>>>(
            x_q, x_k, x_v, wq, sq, wr, sr, wg, sg, ws_bf16, wk, sk, wv, sv, wk_dyn, wv_dyn,
            oq, or_out, og, os, ok, ov, ok_dyn, ov_dyn, M, Qdim, Rdim, Hq, KVdim, DynDim, K);
    } else if (M <= 4) {
        inkling_fused_attn_gemv_fp8_kernel<4><<<blocks, 128, 0, stream>>>(
            x_q, x_k, x_v, wq, sq, wr, sr, wg, sg, ws_bf16, wk, sk, wv, sv, wk_dyn, wv_dyn,
            oq, or_out, og, os, ok, ov, ok_dyn, ov_dyn, M, Qdim, Rdim, Hq, KVdim, DynDim, K);
    } else {
        inkling_fused_attn_gemv_fp8_kernel<8><<<blocks, 128, 0, stream>>>(
            x_q, x_k, x_v, wq, sq, wr, sr, wg, sg, ws_bf16, wk, sk, wv, sv, wk_dyn, wv_dyn,
            oq, or_out, og, os, ok, ov, ok_dyn, ov_dyn, M, Qdim, Rdim, Hq, KVdim, DynDim, K);
    }
}

void launch_inkling_wo_sconv_fp8(const __nv_bfloat16* attn_out,
                                 const u8* wo_fp8, const f32* so_fp8,
                                 const __nv_bfloat16* sconv_w,
                                 __nv_bfloat16* hist_ring,
                                 __nv_bfloat16* out,
                                 const i32* d_pos, u32 start_pos,
                                 u32 M, u32 H, u32 Qdim, cudaStream_t stream) {
    u32 blocks = (H + 3) / 4;
    if (M == 1) {
        inkling_wo_sconv_fp8_kernel<1><<<blocks, 128, 0, stream>>>(
            attn_out, wo_fp8, so_fp8, sconv_w, hist_ring, out, d_pos, start_pos, M, H, Qdim);
    } else if (M <= 4) {
        inkling_wo_sconv_fp8_kernel<4><<<blocks, 128, 0, stream>>>(
            attn_out, wo_fp8, so_fp8, sconv_w, hist_ring, out, d_pos, start_pos, M, H, Qdim);
    } else {
        inkling_wo_sconv_fp8_kernel<8><<<blocks, 128, 0, stream>>>(
            attn_out, wo_fp8, so_fp8, sconv_w, hist_ring, out, d_pos, start_pos, M, H, Qdim);
    }
}

void launch_inkling_fused_3ctrl_down(const __nv_bfloat16* pair_3x2H,
                                     const __nv_bfloat16* wd0, const __nv_bfloat16* wd1, const __nv_bfloat16* wd2,
                                     __nv_bfloat16* out_3xR, u32 M, u32 rank, u32 K2H, cudaStream_t stream) {
    u32 blocks = (3 * rank + 3) / 4;
    if (M == 1) inkling_fused_3ctrl_down_kernel<1><<<blocks, 128, 0, stream>>>(pair_3x2H, wd0, wd1, wd2, out_3xR, M, rank, K2H);
    else if (M <= 4) inkling_fused_3ctrl_down_kernel<4><<<blocks, 128, 0, stream>>>(pair_3x2H, wd0, wd1, wd2, out_3xR, M, rank, K2H);
    else inkling_fused_3ctrl_down_kernel<8><<<blocks, 128, 0, stream>>>(pair_3x2H, wd0, wd1, wd2, out_3xR, M, rank, K2H);
}

void launch_inkling_fused_3ctrl_gf(const __nv_bfloat16* h_3xR,
                                   const __nv_bfloat16* wg0, const __nv_bfloat16* wf0,
                                   const __nv_bfloat16* wg1, const __nv_bfloat16* wf1,
                                   const __nv_bfloat16* wg2, const __nv_bfloat16* wf2,
                                   __nv_bfloat16* out_3x2H, u32 M, u32 H, u32 rank, cudaStream_t stream) {
    u32 blocks = (3 * 2 * H + 3) / 4;
    if (M == 1) inkling_fused_3ctrl_gf_kernel<1><<<blocks, 128, 0, stream>>>(h_3xR, wg0, wf0, wg1, wf1, wg2, wf2, out_3x2H, M, H, rank);
    else if (M <= 4) inkling_fused_3ctrl_gf_kernel<4><<<blocks, 128, 0, stream>>>(h_3xR, wg0, wf0, wg1, wf1, wg2, wf2, out_3x2H, M, H, rank);
    else inkling_fused_3ctrl_gf_kernel<8><<<blocks, 128, 0, stream>>>(h_3xR, wg0, wf0, wg1, wf1, wg2, wf2, out_3x2H, M, H, rank);
}

void launch_inkling_blend_rmsnorm_3stream(const f32* dest, const f32* msg,
                                          const __nv_bfloat16* gf_3x2H,
                                          const __nv_bfloat16* norm_w,
                                          __nv_bfloat16* out,
                                          bool is_local, u32 M, u32 D, f32 eps,
                                          cudaStream_t stream) {
    dim3 grid(3, M);
    inkling_blend_rmsnorm_3stream_kernel<<<grid, 256, 0, stream>>>(
        dest, msg, gf_3x2H, norm_w, out, is_local, M, D, eps);
}

void launch_inkling_qr_kv_fused(
    const __nv_bfloat16* q_in, const __nv_bfloat16* r_in,
    const __nv_bfloat16* q_sc, const __nv_bfloat16* r_sc,
    __nv_bfloat16* q_out, __nv_bfloat16* r_out,
    const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
    const __nv_bfloat16* wk, const __nv_bfloat16* wv,
    const __nv_bfloat16* dk_dyn, const __nv_bfloat16* dv_dyn,
    __nv_bfloat16* k_hist, __nv_bfloat16* v_hist,
    void* k_cache_raw, void* v_cache_raw,
    f32* k_scales, f32* v_scales,
    const i32* d_block_table, const i32* d_pos, u32 start_pos,
    u32 M, u32 n_q_heads, u32 n_kv_heads, u32 block_size,
    bool is_local_ring, u32 swa_window, bool is_dynamic_sconv, bool int8_kv,
    f32 q_clip, f32 r_clip, bool use_log_scale, u32 n_floor, f32 alpha, f32 rms_eps,
    cudaStream_t stream) {
    u32 blocks = n_q_heads + 2 * n_kv_heads;
    if (is_dynamic_sconv) {
        if (int8_kv)
            inkling_qr_kv_fused_kernel<true, true><<<blocks, 128, 0, stream>>>(
                q_in, r_in, q_sc, r_sc, q_out, r_out, k_raw, v_raw, wk, wv, dk_dyn, dv_dyn,
                k_hist, v_hist, k_cache_raw, v_cache_raw, k_scales, v_scales, d_block_table,
                d_pos, start_pos, M, n_q_heads, n_kv_heads, block_size, is_local_ring, swa_window,
                q_clip, r_clip, use_log_scale, n_floor, alpha, rms_eps);
        else
            inkling_qr_kv_fused_kernel<true, false><<<blocks, 128, 0, stream>>>(
                q_in, r_in, q_sc, r_sc, q_out, r_out, k_raw, v_raw, wk, wv, dk_dyn, dv_dyn,
                k_hist, v_hist, k_cache_raw, v_cache_raw, k_scales, v_scales, d_block_table,
                d_pos, start_pos, M, n_q_heads, n_kv_heads, block_size, is_local_ring, swa_window,
                q_clip, r_clip, use_log_scale, n_floor, alpha, rms_eps);
    } else {
        if (int8_kv)
            inkling_qr_kv_fused_kernel<false, true><<<blocks, 128, 0, stream>>>(
                q_in, r_in, q_sc, r_sc, q_out, r_out, k_raw, v_raw, wk, wv, dk_dyn, dv_dyn,
                k_hist, v_hist, k_cache_raw, v_cache_raw, k_scales, v_scales, d_block_table,
                d_pos, start_pos, M, n_q_heads, n_kv_heads, block_size, is_local_ring, swa_window,
                q_clip, r_clip, use_log_scale, n_floor, alpha, rms_eps);
        else
            inkling_qr_kv_fused_kernel<false, false><<<blocks, 128, 0, stream>>>(
                q_in, r_in, q_sc, r_sc, q_out, r_out, k_raw, v_raw, wk, wv, dk_dyn, dv_dyn,
                k_hist, v_hist, k_cache_raw, v_cache_raw, k_scales, v_scales, d_block_table,
                d_pos, start_pos, M, n_q_heads, n_kv_heads, block_size, is_local_ring, swa_window,
                q_clip, r_clip, use_log_scale, n_floor, alpha, rms_eps);
    }
}

void launch_inkling_fused_2gemv(const __nv_bfloat16* x,
                                const __nv_bfloat16* w0, const __nv_bfloat16* w1,
                                __nv_bfloat16* o0, __nv_bfloat16* o1,
                                u32 M, u32 N0, u32 N1, u32 K,
                                cudaStream_t stream) {
    u32 blocks = (N0 + N1 + 3) / 4;
    if (M == 1) {
        inkling_fused_2gemv_kernel<1><<<blocks, 128, 0, stream>>>(x, w0, w1, o0, o1, M, N0, N1, K);
    } else if (M <= 4) {
        inkling_fused_2gemv_kernel<4><<<blocks, 128, 0, stream>>>(x, w0, w1, o0, o1, M, N0, N1, K);
    } else {
        inkling_fused_2gemv_kernel<8><<<blocks, 128, 0, stream>>>(x, w0, w1, o0, o1, M, N0, N1, K);
    }
}


} // namespace flint::cuda
