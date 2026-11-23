#include "kernels.hpp"

namespace flint::cuda {

namespace {

// 1. DFlare Mix + RMSNorm: out[m, d] = rms_norm(sum_{t=0..3} alpha[t] * feat[t][m, d], weight[d])
__global__ void inkling_mtp_mix_rmsnorm_kernel(
    const __nv_bfloat16* const* __restrict__ feat_ptrs, // [4] pointers to [M, D]
    const f32* __restrict__ alpha,                      // [4] softmax weights
    const __nv_bfloat16* __restrict__ norm_w,           // [D]
    __nv_bfloat16* __restrict__ out,                    // [M, D]
    u32 M, u32 D, f32 eps) {
    const u32 m = blockIdx.x;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31, wid = tid >> 5;
    if (m >= M) return;

    f32 a0 = alpha[0], a1 = alpha[1], a2 = alpha[2], a3 = alpha[3];
    const __nv_bfloat16* f0 = feat_ptrs[0] + static_cast<size_t>(m) * D;
    const __nv_bfloat16* f1 = feat_ptrs[1] + static_cast<size_t>(m) * D;
    const __nv_bfloat16* f2 = feat_ptrs[2] + static_cast<size_t>(m) * D;
    const __nv_bfloat16* f3 = feat_ptrs[3] + static_cast<size_t>(m) * D;

    f32 ss = 0.0f;
    for (u32 d = tid; d < D; d += 256) {
        f32 v = a0 * __bfloat162float(f0[d]) + a1 * __bfloat162float(f1[d]) +
                a2 * __bfloat162float(f2[d]) + a3 * __bfloat162float(f3[d]);
        ss = fmaf(v, v, ss);
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
    __nv_bfloat16* dst = out + static_cast<size_t>(m) * D;
    for (u32 d = tid; d < D; d += 256) {
        f32 v = a0 * __bfloat162float(f0[d]) + a1 * __bfloat162float(f1[d]) +
                a2 * __bfloat162float(f2[d]) + a3 * __bfloat162float(f3[d]);
        dst[d] = __float2bfloat16_rn(v * rstd * __bfloat162float(norm_w[d]));
    }
}

// 2. Per-head RMSNorm (rounded to BF16) + NeoX RoPE (in-place on [M, H, 128])
__global__ void inkling_mtp_norm_rope_kernel(
    __nv_bfloat16* __restrict__ x,          // [M, H, 128]
    const __nv_bfloat16* __restrict__ w,    // [128] or nullptr
    const f32* __restrict__ inv_freq,       // [64]
    u32 start_pos, u32 M, u32 H, f32 eps) {
    const u32 h = blockIdx.x;
    const u32 m = blockIdx.y;
    const u32 tid = threadIdx.x; // 64 threads (pairs tid, tid+64)
    if (h >= H || m >= M) return;

    __nv_bfloat16* row = x + (static_cast<size_t>(m) * H + h) * 128;
    f32 x0 = __bfloat162float(row[tid]);
    f32 x1 = __bfloat162float(row[tid + 64]);
    if (w) {
        f32 ss = x0 * x0 + x1 * x1;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
        __shared__ f32 sm[2];
        if ((tid & 31) == 0) sm[tid >> 5] = ss;
        __syncthreads();
        f32 rstd = rsqrtf((sm[0] + sm[1]) / 128.0f + eps);
        x0 = __bfloat162float(__float2bfloat16_rn(x0 * rstd * __bfloat162float(w[tid])));
        x1 = __bfloat162float(__float2bfloat16_rn(x1 * rstd * __bfloat162float(w[tid + 64])));
    }
    f32 ang = static_cast<f32>(start_pos + m) * inv_freq[tid];
    f32 s, c;
    __sincosf(ang, &s, &c);
    row[tid]      = __float2bfloat16_rn(x0 * c - x1 * s);
    row[tid + 64] = __float2bfloat16_rn(x1 * c + x0 * s);
}

// 3. MTP Bidirectional Block Attention over Context KV [ctx_len, 8, 128] + Block KV [B, 8, 128]
// Grid: (24 Q heads, B=8 block tokens), Block: 128 threads (4 warps)
__global__ void inkling_mtp_block_attn_kernel(
    const __nv_bfloat16* __restrict__ q_blk,   // [B, 24, 128]
    const __nv_bfloat16* __restrict__ k_ctx,   // [max_seq, 8, 128]
    const __nv_bfloat16* __restrict__ v_ctx,   // [max_seq, 8, 128]
    const __nv_bfloat16* __restrict__ k_blk,   // [B, 8, 128]
    const __nv_bfloat16* __restrict__ v_blk,   // [B, 8, 128]
    __nv_bfloat16* __restrict__ out,           // [B, 24, 128]
    u32 ctx_start, u32 ctx_len, u32 B, u32 n_q_heads, u32 n_kv_heads) {
    const u32 qh   = blockIdx.x;
    const u32 b    = blockIdx.y;
    const u32 tid  = threadIdx.x;
    const u32 wid  = tid >> 5;
    const u32 lane = tid & 31;
    const u32 kvh  = qh / (n_q_heads / n_kv_heads);
    const f32 scale = rsqrtf(128.0f);

    const __nv_bfloat16* qp = q_blk + (static_cast<size_t>(b) * n_q_heads + qh) * 128 + lane * 4;
    f32 q0 = __bfloat162float(qp[0]) * scale, q1 = __bfloat162float(qp[1]) * scale;
    f32 q2 = __bfloat162float(qp[2]) * scale, q3 = __bfloat162float(qp[3]) * scale;

    f32 m_i = -1e30f, l_i = 0.0f, o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    const u32 total_k = (ctx_len - ctx_start) + B;

    for (u32 idx = wid; idx < total_k; idx += 4) {
        const __nv_bfloat16* kp;
        const __nv_bfloat16* vp;
        if (idx < (ctx_len - ctx_start)) {
            u32 t = ctx_start + idx;
            kp = k_ctx + (static_cast<size_t>(t) * n_kv_heads + kvh) * 128;
            vp = v_ctx + (static_cast<size_t>(t) * n_kv_heads + kvh) * 128;
        } else {
            u32 j = idx - (ctx_len - ctx_start);
            kp = k_blk + (static_cast<size_t>(j) * n_kv_heads + kvh) * 128;
            vp = v_blk + (static_cast<size_t>(j) * n_kv_heads + kvh) * 128;
        }
        uint2 ku = __ldg(reinterpret_cast<const uint2*>(kp) + lane);
        uint2 vu = __ldg(reinterpret_cast<const uint2*>(vp) + lane);
        const auto* k2 = reinterpret_cast<const __nv_bfloat162*>(&ku);
        const auto* v2 = reinterpret_cast<const __nv_bfloat162*>(&vu);
        float2 fk0 = __bfloat1622float2(k2[0]), fk1 = __bfloat1622float2(k2[1]);
        float2 fv0 = __bfloat1622float2(v2[0]), fv1 = __bfloat1622float2(v2[1]);
        f32 dot = fmaf(q0, fk0.x, fmaf(q1, fk0.y, fmaf(q2, fk1.x, q3 * fk1.y)));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
        f32 nm = fmaxf(m_i, dot);
        f32 ep = __expf(m_i - nm), ec = __expf(dot - nm);
        l_i = l_i * ep + ec;
        o0 = fmaf(ec, fv0.x, o0 * ep); o1 = fmaf(ec, fv0.y, o1 * ep);
        o2 = fmaf(ec, fv1.x, o2 * ep); o3 = fmaf(ec, fv1.y, o3 * ep);
        m_i = nm;
    }

    __shared__ f32 sm_m[4], sm_l[4], sm_o[4][128];
    if (lane == 0) { sm_m[wid] = m_i; sm_l[wid] = l_i; }
    sm_o[wid][lane * 4 + 0] = o0; sm_o[wid][lane * 4 + 1] = o1;
    sm_o[wid][lane * 4 + 2] = o2; sm_o[wid][lane * 4 + 3] = o3;
    __syncthreads();
    if (tid < 128) {
        f32 mg = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
        f32 w0 = (sm_l[0] > 0.0f) ? __expf(sm_m[0] - mg) : 0.0f;
        f32 w1 = (sm_l[1] > 0.0f) ? __expf(sm_m[1] - mg) : 0.0f;
        f32 w2 = (sm_l[2] > 0.0f) ? __expf(sm_m[2] - mg) : 0.0f;
        f32 w3 = (sm_l[3] > 0.0f) ? __expf(sm_m[3] - mg) : 0.0f;
        f32 lg = w0 * sm_l[0] + w1 * sm_l[1] + w2 * sm_l[2] + w3 * sm_l[3];
        f32 acc = w0 * sm_o[0][tid] + w1 * sm_o[1][tid] + w2 * sm_o[2][tid] + w3 * sm_o[3][tid];
        out[(static_cast<size_t>(b) * n_q_heads + qh) * 128 + tid] = __float2bfloat16_rn((lg > 0.0f) ? (acc / lg) : 0.0f);
    }
}

// 4. Exact Markov Gate (matches ops/dspark.py:_markov_gate_kernel)
// raw: [3*rank] BF16/F32, state: [rank] BF16 -> new_state: [rank], latent: [rank]
__device__ __forceinline__ f32 rbf16(f32 x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

__global__ void inkling_mtp_markov_gate_kernel(
    const __nv_bfloat16* __restrict__ raw_3r, // [3 * rank]
    __nv_bfloat16* __restrict__ state,        // [rank] (in/out)
    __nv_bfloat16* __restrict__ latent,       // [rank] (out)
    u32 rank) {
    u32 c = threadIdx.x;
    if (c >= rank) return;
    f32 g = rbf16(__bfloat162float(raw_3r[c]));
    f32 cand = rbf16(__bfloat162float(raw_3r[rank + c]));
    f32 outv = rbf16(__bfloat162float(raw_3r[2 * rank + c]));
    f32 st = __bfloat162float(state[c]);
    g = rbf16(1.0f / (1.0f + expf(-g)));
    cand = rbf16(tanhf(cand));
    f32 keep = rbf16(g * st);
    f32 upd  = rbf16(rbf16(1.0f - g) * cand);
    state[c]  = __float2bfloat16_rn(keep + upd);
    latent[c] = __float2bfloat16_rn(tanhf(outv));
}

// 5. Add base_logits + markov_bias + softcap + argmax (1-shot for step i)
__global__ void inkling_mtp_add_softcap_kernel(
    const __nv_bfloat16* __restrict__ base_logits,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    u32 vocab_padded, u32 vocab_unpadded, f32 cap) {
    u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= vocab_padded) return;
    if (c >= vocab_unpadded) {
        out[c] = __float2bfloat16_rn(-1e30f);
        return;
    }
    f32 v = rbf16(__bfloat162float(base_logits[c]) + __bfloat162float(bias[c]));
    if (cap > 0.0f) v = cap * tanhf(v / cap);
    out[c] = __float2bfloat16_rn(v);
}

// 6. Elementwise SiLU(gate) * up for MTP dense MLP (gate [B, inter], up [B, inter] -> out [B, inter])
__global__ void inkling_mtp_silu_mul_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    u32 total) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    f32 g = __bfloat162float(gate[i]);
    f32 u = __bfloat162float(up[i]);
    out[i] = __float2bfloat16_rn((g / (1.0f + expf(-g))) * u);
}

// 7. Zero-Host-Sync GPU-resident RNN input packer: [state(rank), markov_w1[*d_prev_tok](rank), h_step(H), f_rnn(H)]
__global__ void inkling_mtp_pack_rnn_in_kernel(
    const __nv_bfloat16* __restrict__ state,
    const __nv_bfloat16* __restrict__ markov_w1,
    const i32* __restrict__ d_prev_tok,
    const __nv_bfloat16* __restrict__ h_step,
    const __nv_bfloat16* __restrict__ f_rnn,
    __nv_bfloat16* __restrict__ rnn_in,
    u32 rank, u32 H) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    u32 total = 2 * rank + 2 * H;
    if (idx >= total) return;
    if (idx < rank) {
        rnn_in[idx] = state[idx];
    } else if (idx < 2 * rank) {
        i32 tok = *d_prev_tok;
        rnn_in[idx] = markov_w1[static_cast<size_t>(tok) * rank + (idx - rank)];
    } else if (idx < 2 * rank + H) {
        rnn_in[idx] = h_step[idx - 2 * rank];
    } else {
        rnn_in[idx] = f_rnn[idx - (2 * rank + H)];
    }
}

// 8. CUDA-Graph-capturable 8-token draft block builder: x[0]=rmsnorm(embed[*d_anchor]), x[1..B-1]=mask_embed
__global__ void inkling_mtp_build_block_kernel(
    const __nv_bfloat16* __restrict__ embed,
    const __nv_bfloat16* __restrict__ embed_norm,
    const __nv_bfloat16* __restrict__ mask_embed,
    const i32* __restrict__ d_anchor_tok,
    __nv_bfloat16* __restrict__ x_blk,
    u32 B, u32 H, f32 eps) {
    const u32 b   = blockIdx.x;
    const u32 tid = threadIdx.x;
    if (b >= B) return;
    __nv_bfloat16* dst = x_blk + static_cast<size_t>(b) * H;
    if (b > 0) {
        for (u32 d = tid; d < H; d += 256) dst[d] = mask_embed[d];
        return;
    }
    const i32 tok = *d_anchor_tok;
    const __nv_bfloat16* erow = embed + static_cast<size_t>(tok) * H;
    const u32 lane = tid & 31, wid = tid >> 5;
    f32 ss = 0.0f;
    for (u32 d = tid; d < H; d += 256) {
        f32 v = __bfloat162float(erow[d]);
        ss = fmaf(v, v, ss);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xFFFFFFFFu, ss, off);
    __shared__ f32 sm[8];
    if (lane == 0) sm[wid] = ss;
    __syncthreads();
    if (tid == 0) {
        f32 t = 0.0f;
        for (int w = 0; w < 8; ++w) t += sm[w];
        sm[0] = rsqrtf(t / static_cast<f32>(H) + eps);
    }
    __syncthreads();
    f32 rstd = sm[0];
    for (u32 d = tid; d < H; d += 256) {
        dst[d] = __float2bfloat16_rn(__bfloat162float(erow[d]) * rstd * __bfloat162float(embed_norm[d]));
    }
}

__global__ void inkling_mtp_build_block_fp8_kernel(
    const u8* __restrict__ embed_fp8, const f32* __restrict__ embed_scale,
    const __nv_bfloat16* __restrict__ embed_norm, const __nv_bfloat16* __restrict__ mask_embed,
    const i32* __restrict__ d_anchor_tok, __nv_bfloat16* __restrict__ x_blk, u32 B, u32 H, f32 eps) {
    const u32 b = blockIdx.x, tid = threadIdx.x;
    if (b >= B) return;
    __nv_bfloat16* dst = x_blk + static_cast<size_t>(b) * H;
    if (b > 0) { for (u32 d = tid; d < H; d += 256) dst[d] = mask_embed[d]; return; }
    const i32 tok = *d_anchor_tok;
    const u8* erow = embed_fp8 + static_cast<size_t>(tok) * H;
    const f32 sc = __ldg(embed_scale + tok);
    const u32 lane = tid & 31, wid = tid >> 5;
    f32 ss = 0.0f;
    for (u32 d = tid; d < H; d += 256) {
        __nv_fp8_e4m3 fv; *reinterpret_cast<u8*>(&fv) = __ldg(erow + d);
        f32 v = static_cast<f32>(fv) * sc;
        ss = fmaf(v, v, ss);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xFFFFFFFFu, ss, off);
    __shared__ f32 sm[8];
    if (lane == 0) sm[wid] = ss;
    __syncthreads();
    if (tid == 0) { f32 t = 0.0f; for (int w = 0; w < 8; ++w) t += sm[w]; sm[0] = rsqrtf(t / static_cast<f32>(H) + eps); }
    __syncthreads();
    f32 rstd = sm[0];
    for (u32 d = tid; d < H; d += 256) {
        __nv_fp8_e4m3 fv; *reinterpret_cast<u8*>(&fv) = __ldg(erow + d);
        dst[d] = __float2bfloat16_rn((static_cast<f32>(fv) * sc) * rstd * __bfloat162float(embed_norm[d]));
    }
}

__global__ void inkling_mtp_build_block_w4a16_kernel(
    const u32* __restrict__ embed_qw, const u32* __restrict__ embed_sc,
    const __nv_bfloat16* __restrict__ embed_norm, const __nv_bfloat16* __restrict__ mask_embed,
    const i32* __restrict__ d_anchor_tok, __nv_bfloat16* __restrict__ x_blk, u32 B, u32 H, f32 eps) {
    const u32 b = blockIdx.x, tid = threadIdx.x;
    if (b >= B) return;
    __nv_bfloat16* dst = x_blk + static_cast<size_t>(b) * H;
    if (b > 0) { for (u32 d = tid; d < H; d += 256) dst[d] = mask_embed[d]; return; }
    const i32 tok = *d_anchor_tok;
    const u32* wrow = embed_qw + static_cast<size_t>(tok) * (H >> 3);
    const u32* srow = embed_sc + static_cast<size_t>(tok) * (H >> 7);
    const u32 lane = tid & 31, wid = tid >> 5;
    f32 ss = 0.0f;
    for (u32 d = tid; d < H; d += 256) {
        u32 g = d >> 7, w = d >> 3, pos8 = d & 7u;
        u32 shift = ((pos8 & 1u) ? 16u : 0u) + (pos8 >> 1) * 4u;
        u32 sm_v = __ldg(srow + g);
        float2 fs = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sm_v));
        f32 v = fmaf(static_cast<f32>((__ldg(wrow + w) >> shift) & 0xFu), fs.x, fs.y);
        ss = fmaf(v, v, ss);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xFFFFFFFFu, ss, off);
    __shared__ f32 sm[8];
    if (lane == 0) sm[wid] = ss;
    __syncthreads();
    if (tid == 0) { f32 t = 0.0f; for (int w = 0; w < 8; ++w) t += sm[w]; sm[0] = rsqrtf(t / static_cast<f32>(H) + eps); }
    __syncthreads();
    f32 rstd = sm[0];
    for (u32 d = tid; d < H; d += 256) {
        u32 g = d >> 7, w = d >> 3, pos8 = d & 7u;
        u32 shift = ((pos8 & 1u) ? 16u : 0u) + (pos8 >> 1) * 4u;
        u32 sm_v = __ldg(srow + g);
        float2 fs = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sm_v));
        f32 v = fmaf(static_cast<f32>((__ldg(wrow + w) >> shift) & 0xFu), fs.x, fs.y);
        dst[d] = __float2bfloat16_rn(v * rstd * __bfloat162float(embed_norm[d]));
    }
}

// 9. Device-position RoPE + per-head RMSNorm (100% CUDA-Graph capturable)
__global__ void inkling_mtp_norm_rope_dpos_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const f32* __restrict__ inv_freq,
    const i32* __restrict__ d_pos,
    u32 M, u32 H, f32 eps) {
    const u32 h = blockIdx.x;
    const u32 m = blockIdx.y;
    const u32 tid = threadIdx.x;
    if (h >= H || m >= M) return;
    const u32 start_pos = static_cast<u32>(*d_pos);
    __nv_bfloat16* row = x + (static_cast<size_t>(m) * H + h) * 128;
    f32 x0 = __bfloat162float(row[tid]);
    f32 x1 = __bfloat162float(row[tid + 64]);
    if (w) {
        f32 ss = x0 * x0 + x1 * x1;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
        __shared__ f32 sm[2];
        if ((tid & 31) == 0) sm[tid >> 5] = ss;
        __syncthreads();
        f32 rstd = rsqrtf((sm[0] + sm[1]) / 128.0f + eps);
        x0 = __bfloat162float(__float2bfloat16_rn(x0 * rstd * __bfloat162float(w[tid])));
        x1 = __bfloat162float(__float2bfloat16_rn(x1 * rstd * __bfloat162float(w[tid + 64])));
    }
    f32 ang = static_cast<f32>(start_pos + m) * inv_freq[tid];
    f32 s, c;
    __sincosf(ang, &s, &c);
    row[tid]      = __float2bfloat16_rn(x0 * c - x1 * s);
    row[tid + 64] = __float2bfloat16_rn(x1 * c + x0 * s);
}

// 10. Device-position Bidirectional Block Attention (100% CUDA-Graph capturable)
__global__ void inkling_mtp_block_attn_dpos_kernel(
    const __nv_bfloat16* __restrict__ q_blk,
    const __nv_bfloat16* __restrict__ k_ctx,
    const __nv_bfloat16* __restrict__ v_ctx,
    const __nv_bfloat16* __restrict__ k_blk,
    const __nv_bfloat16* __restrict__ v_blk,
    __nv_bfloat16* __restrict__ out,
    const i32* __restrict__ d_ctx_len, bool is_local, u32 swa_window,
    u32 B, u32 n_q_heads, u32 n_kv_heads) {
    const u32 qh   = blockIdx.x;
    const u32 b    = blockIdx.y;
    const u32 tid  = threadIdx.x;
    const u32 wid  = tid >> 5;
    const u32 lane = tid & 31;
    const u32 kvh  = qh / (n_q_heads / n_kv_heads);
    const f32 scale = rsqrtf(128.0f);
    const u32 ctx_len   = static_cast<u32>(*d_ctx_len);
    const u32 ctx_start = (is_local && ctx_len > swa_window) ? (ctx_len - swa_window) : 0;
    const u32 ctx_span  = ctx_len - ctx_start;

    const __nv_bfloat16* qp = q_blk + (static_cast<size_t>(b) * n_q_heads + qh) * 128 + lane * 4;
    f32 q0 = __bfloat162float(qp[0]) * scale, q1 = __bfloat162float(qp[1]) * scale;
    f32 q2 = __bfloat162float(qp[2]) * scale, q3 = __bfloat162float(qp[3]) * scale;

    f32 m_i = -1e30f, l_i = 0.0f, o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    const u32 total_k = ctx_span + B;
    for (u32 idx = wid; idx < total_k; idx += 4) {
        const __nv_bfloat16* kp;
        const __nv_bfloat16* vp;
        if (idx < ctx_span) {
            u32 t = ctx_start + idx;
            kp = k_ctx + (static_cast<size_t>(t) * n_kv_heads + kvh) * 128;
            vp = v_ctx + (static_cast<size_t>(t) * n_kv_heads + kvh) * 128;
        } else {
            u32 j = idx - ctx_span;
            kp = k_blk + (static_cast<size_t>(j) * n_kv_heads + kvh) * 128;
            vp = v_blk + (static_cast<size_t>(j) * n_kv_heads + kvh) * 128;
        }
        uint2 ku = __ldg(reinterpret_cast<const uint2*>(kp) + lane);
        uint2 vu = __ldg(reinterpret_cast<const uint2*>(vp) + lane);
        const auto* k2 = reinterpret_cast<const __nv_bfloat162*>(&ku);
        const auto* v2 = reinterpret_cast<const __nv_bfloat162*>(&vu);
        float2 fk0 = __bfloat1622float2(k2[0]), fk1 = __bfloat1622float2(k2[1]);
        float2 fv0 = __bfloat1622float2(v2[0]), fv1 = __bfloat1622float2(v2[1]);
        f32 dot = fmaf(q0, fk0.x, fmaf(q1, fk0.y, fmaf(q2, fk1.x, q3 * fk1.y)));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
        f32 nm = fmaxf(m_i, dot);
        f32 ep = __expf(m_i - nm), ec = __expf(dot - nm);
        l_i = l_i * ep + ec;
        o0 = fmaf(ec, fv0.x, o0 * ep); o1 = fmaf(ec, fv0.y, o1 * ep);
        o2 = fmaf(ec, fv1.x, o2 * ep); o3 = fmaf(ec, fv1.y, o3 * ep);
        m_i = nm;
    }
    __shared__ f32 sm_m[4], sm_l[4], sm_o[4][128];
    if (lane == 0) { sm_m[wid] = m_i; sm_l[wid] = l_i; }
    sm_o[wid][lane * 4 + 0] = o0; sm_o[wid][lane * 4 + 1] = o1;
    sm_o[wid][lane * 4 + 2] = o2; sm_o[wid][lane * 4 + 3] = o3;
    __syncthreads();
    if (tid < 128) {
        f32 mg = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
        f32 w0 = (sm_l[0] > 0.0f) ? __expf(sm_m[0] - mg) : 0.0f;
        f32 w1 = (sm_l[1] > 0.0f) ? __expf(sm_m[1] - mg) : 0.0f;
        f32 w2 = (sm_l[2] > 0.0f) ? __expf(sm_m[2] - mg) : 0.0f;
        f32 w3 = (sm_l[3] > 0.0f) ? __expf(sm_m[3] - mg) : 0.0f;
        f32 lg = w0 * sm_l[0] + w1 * sm_l[1] + w2 * sm_l[2] + w3 * sm_l[3];
        f32 acc = w0 * sm_o[0][tid] + w1 * sm_o[1][tid] + w2 * sm_o[2][tid] + w3 * sm_o[3][tid];
        out[(static_cast<size_t>(b) * n_q_heads + qh) * 128 + tid] = __float2bfloat16_rn((lg > 0.0f) ? (acc / lg) : 0.0f);
    }
}

// 11. Precompute [n_steps, 3*rank] F32 Markov context ONCE outside the 7-step serial loop
// Matches dspark.py:363: ctx[s, r] = rnn_b[r] + dot(h_steps[s], rnn_w[r, 2*rank : 2*rank+H]) + dot(f_rnn, rnn_w[r, 2*rank+H : ])
__global__ void inkling_mtp_rnn_ctx_kernel(
    const __nv_bfloat16* __restrict__ h_steps, // [n_steps, H]
    const __nv_bfloat16* __restrict__ f_rnn,   // [H]
    const __nv_bfloat16* __restrict__ rnn_w,   // [3*rank, 2*rank + 2*H]
    const __nv_bfloat16* __restrict__ rnn_b,   // [3*rank]
    f32* __restrict__ out_ctx_f32,             // [n_steps, 3*rank]
    u32 n_steps, u32 rank, u32 H) {
    const u32 wid  = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 row  = blockIdx.x * 4 + wid;
    const u32 N    = 3 * rank;
    if (row >= N) return;

    const u32 stride = 2 * rank + 2 * H;
    const u32 H8 = H >> 3;
    const auto* wh = reinterpret_cast<const uint4*>(rnn_w + static_cast<size_t>(row) * stride + 2 * rank);
    const auto* wf = reinterpret_cast<const uint4*>(rnn_w + static_cast<size_t>(row) * stride + 2 * rank + H);
    const auto* xf = reinterpret_cast<const uint4*>(f_rnn);

    // 1. Compute shared f_rnn contribution ONCE for all 7 steps
    f32 f_acc = 0.0f;
    for (u32 idx = lane; idx < H8; idx += 32) {
        uint4 wv = __ldg(wf + idx);
        uint4 xv = xf[idx];
        const auto* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
        const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 fw = __bfloat1622float2(wb[j]);
            float2 fx = __bfloat1622float2(xb[j]);
            f_acc = fmaf(fw.x, fx.x, fmaf(fw.y, fx.y, f_acc));
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) f_acc += __shfl_down_sync(0xFFFFFFFFu, f_acc, off);
    f32 base_f = __shfl_sync(0xFFFFFFFFu, f_acc, 0) + __bfloat162float(rnn_b[row]);

    // 2. Compute h_steps[0..n_steps-1] sharing the SAME wh weight load
    f32 acc[7] = {};
    for (u32 idx = lane; idx < H8; idx += 32) {
        uint4 wv = __ldg(wh + idx);
        const auto* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
        for (u32 s = 0; s < n_steps; ++s) {
            uint4 xv = reinterpret_cast<const uint4*>(h_steps + static_cast<size_t>(s) * H)[idx];
            const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 fw = __bfloat1622float2(wb[j]);
                float2 fx = __bfloat1622float2(xb[j]);
                acc[s] = fmaf(fw.x, fx.x, fmaf(fw.y, fx.y, acc[s]));
            }
        }
    }
    for (u32 s = 0; s < n_steps; ++s) {
        f32 v = acc[s];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
        if (lane == 0) out_ctx_f32[static_cast<size_t>(s) * N + row] = base_f + v;
    }
}

// 12. Fused 1-kernel Markov RNN Step + Exact BF16-Rounded Gate (10x less work than 5120-col GEMV)
// Grid: 1 block x 256 threads (rank=256). Each thread c computes gate/cand/out row c, c+rank, c+2*rank
__global__ void inkling_mtp_rnn_step_gate_kernel(
    const f32* __restrict__ ctx_s_f32,          // [3*rank]
    const __nv_bfloat16* __restrict__ rnn_w,    // [3*rank, 2*rank + 2*H]
    const __nv_bfloat16* __restrict__ markov_w1,// [V, rank]
    const i32* __restrict__ d_prev_tok,
    __nv_bfloat16* __restrict__ state,          // [rank] (in/out)
    __nv_bfloat16* __restrict__ latent,         // [rank] (out)
    u32 rank, u32 H) {
    const u32 c = threadIdx.x; // 0..255 (rank=256)
    if (c >= rank) return;
    __shared__ __nv_bfloat16 sm_st[256];
    __shared__ __nv_bfloat16 sm_pw[256];
    const i32 prev_tok = *d_prev_tok;
    sm_st[c] = state[c];
    sm_pw[c] = markov_w1[static_cast<size_t>(prev_tok) * rank + c];
    __syncthreads();

    const u32 stride = 2 * rank + 2 * H;
    const u32 R8 = rank >> 3;
    const auto* st4 = reinterpret_cast<const uint4*>(sm_st);
    const auto* pw4 = reinterpret_cast<const uint4*>(sm_pw);

    f32 raw3[3];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        u32 row = g * rank + c;
        const auto* w_st = reinterpret_cast<const uint4*>(rnn_w + static_cast<size_t>(row) * stride);
        const auto* w_pw = reinterpret_cast<const uint4*>(rnn_w + static_cast<size_t>(row) * stride + rank);
        f32 acc = ctx_s_f32[row];
        for (u32 i = 0; i < R8; ++i) {
            uint4 ws = __ldg(w_st + i), wp = __ldg(w_pw + i);
            uint4 xs = st4[i],          xp = pw4[i];
            const auto* wsb = reinterpret_cast<const __nv_bfloat162*>(&ws);
            const auto* xsb = reinterpret_cast<const __nv_bfloat162*>(&xs);
            const auto* wpb = reinterpret_cast<const __nv_bfloat162*>(&wp);
            const auto* xpb = reinterpret_cast<const __nv_bfloat162*>(&xp);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 fws = __bfloat1622float2(wsb[j]), fxs = __bfloat1622float2(xsb[j]);
                float2 fwp = __bfloat1622float2(wpb[j]), fxp = __bfloat1622float2(xpb[j]);
                acc = fmaf(fws.x, fxs.x, fmaf(fws.y, fxs.y, acc));
                acc = fmaf(fwp.x, fxp.x, fmaf(fwp.y, fxp.y, acc));
            }
        }
        raw3[g] = rbf16(acc);
    }

    f32 gval = rbf16(1.0f / (1.0f + expf(-raw3[0])));
    f32 cand = rbf16(tanhf(raw3[1]));
    f32 st   = __bfloat162float(sm_st[c]);
    f32 keep = rbf16(gval * st);
    f32 upd  = rbf16(rbf16(1.0f - gval) * cand);
    state[c]  = __float2bfloat16_rn(keep + upd);
    latent[c] = __float2bfloat16_rn(tanhf(raw3[2]));
}

} // namespace

void launch_inkling_mtp_mix_rmsnorm(const __nv_bfloat16* const* d_feat_ptrs,
                                    const f32* d_alpha,
                                    const __nv_bfloat16* norm_w,
                                    __nv_bfloat16* out,
                                    u32 M, u32 D, f32 eps, cudaStream_t stream) {
    inkling_mtp_mix_rmsnorm_kernel<<<M, 256, 0, stream>>>(d_feat_ptrs, d_alpha, norm_w, out, M, D, eps);
}

void launch_inkling_mtp_norm_rope(__nv_bfloat16* x, const __nv_bfloat16* w,
                                  const f32* inv_freq, u32 start_pos, u32 M, u32 H, f32 eps,
                                  cudaStream_t stream) {
    dim3 grid(H, M);
    inkling_mtp_norm_rope_kernel<<<grid, 64, 0, stream>>>(x, w, inv_freq, start_pos, M, H, eps);
}

void launch_inkling_mtp_block_attn(const __nv_bfloat16* q_blk,
                                   const __nv_bfloat16* k_ctx, const __nv_bfloat16* v_ctx,
                                   const __nv_bfloat16* k_blk, const __nv_bfloat16* v_blk,
                                   __nv_bfloat16* out,
                                   u32 ctx_start, u32 ctx_len, u32 B, u32 n_q_heads, u32 n_kv_heads,
                                   cudaStream_t stream) {
    dim3 grid(n_q_heads, B);
    inkling_mtp_block_attn_kernel<<<grid, 128, 0, stream>>>(
        q_blk, k_ctx, v_ctx, k_blk, v_blk, out, ctx_start, ctx_len, B, n_q_heads, n_kv_heads);
}

void launch_inkling_mtp_markov_gate(const __nv_bfloat16* raw_3r, __nv_bfloat16* state,
                                    __nv_bfloat16* latent, u32 rank, cudaStream_t stream) {
    inkling_mtp_markov_gate_kernel<<<1, rank, 0, stream>>>(raw_3r, state, latent, rank);
}

void launch_inkling_mtp_add_softcap(const __nv_bfloat16* base_logits, const __nv_bfloat16* bias,
                                    __nv_bfloat16* out, u32 vocab_padded, u32 vocab_unpadded,
                                    f32 cap, cudaStream_t stream) {
    inkling_mtp_add_softcap_kernel<<<(vocab_padded + 255) / 256, 256, 0, stream>>>(
        base_logits, bias, out, vocab_padded, vocab_unpadded, cap);
}

void launch_inkling_mtp_silu_mul(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                 __nv_bfloat16* out, u32 total, cudaStream_t stream) {
    inkling_mtp_silu_mul_kernel<<<(total + 255) / 256, 256, 0, stream>>>(gate, up, out, total);
}

void launch_inkling_mtp_pack_rnn_in(const __nv_bfloat16* state,
                                    const __nv_bfloat16* markov_w1,
                                    const i32* d_prev_tok,
                                    const __nv_bfloat16* h_step,
                                    const __nv_bfloat16* f_rnn,
                                    __nv_bfloat16* rnn_in,
                                    u32 rank, u32 H, cudaStream_t stream) {
    u32 total = 2 * rank + 2 * H;
    inkling_mtp_pack_rnn_in_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        state, markov_w1, d_prev_tok, h_step, f_rnn, rnn_in, rank, H);
}

void launch_inkling_mtp_build_block(const __nv_bfloat16* embed, const __nv_bfloat16* embed_norm,
                                    const __nv_bfloat16* mask_embed, const i32* d_anchor_tok,
                                    __nv_bfloat16* x_blk, u32 B, u32 H, f32 eps, cudaStream_t stream) {
    inkling_mtp_build_block_kernel<<<B, 256, 0, stream>>>(embed, embed_norm, mask_embed, d_anchor_tok, x_blk, B, H, eps);
}

void launch_inkling_mtp_build_block_fp8(const u8* embed_fp8, const f32* embed_scale,
                                        const __nv_bfloat16* embed_norm,
                                        const __nv_bfloat16* mask_embed, const i32* d_anchor_tok,
                                        __nv_bfloat16* x_blk, u32 B, u32 H, f32 eps, cudaStream_t stream) {
    inkling_mtp_build_block_fp8_kernel<<<B, 256, 0, stream>>>(
        embed_fp8, embed_scale, embed_norm, mask_embed, d_anchor_tok, x_blk, B, H, eps);
}

void launch_inkling_mtp_build_block_w4a16(const u32* embed_qw, const u32* embed_sc,
                                          const __nv_bfloat16* embed_norm,
                                          const __nv_bfloat16* mask_embed, const i32* d_anchor_tok,
                                          __nv_bfloat16* x_blk, u32 B, u32 H, f32 eps, cudaStream_t stream) {
    inkling_mtp_build_block_w4a16_kernel<<<B, 256, 0, stream>>>(
        embed_qw, embed_sc, embed_norm, mask_embed, d_anchor_tok, x_blk, B, H, eps);
}

void launch_inkling_mtp_norm_rope_dpos(__nv_bfloat16* x, const __nv_bfloat16* w,
                                       const f32* inv_freq, const i32* d_pos, u32 M, u32 H, f32 eps,
                                       cudaStream_t stream) {
    dim3 grid(H, M);
    inkling_mtp_norm_rope_dpos_kernel<<<grid, 64, 0, stream>>>(x, w, inv_freq, d_pos, M, H, eps);
}

void launch_inkling_mtp_block_attn_dpos(const __nv_bfloat16* q_blk,
                                        const __nv_bfloat16* k_ctx, const __nv_bfloat16* v_ctx,
                                        const __nv_bfloat16* k_blk, const __nv_bfloat16* v_blk,
                                        __nv_bfloat16* out,
                                        const i32* d_ctx_len, bool is_local, u32 swa_window,
                                        u32 B, u32 n_q_heads, u32 n_kv_heads, cudaStream_t stream) {
    dim3 grid(n_q_heads, B);
    inkling_mtp_block_attn_dpos_kernel<<<grid, 128, 0, stream>>>(
        q_blk, k_ctx, v_ctx, k_blk, v_blk, out, d_ctx_len, is_local, swa_window, B, n_q_heads, n_kv_heads);
}

void launch_inkling_mtp_rnn_ctx(const __nv_bfloat16* h_steps, const __nv_bfloat16* f_rnn,
                                const __nv_bfloat16* rnn_w, const __nv_bfloat16* rnn_b,
                                f32* out_ctx_f32, u32 n_steps, u32 rank, u32 H, cudaStream_t stream) {
    u32 N = 3 * rank;
    inkling_mtp_rnn_ctx_kernel<<<(N + 3) / 4, 128, 0, stream>>>(
        h_steps, f_rnn, rnn_w, rnn_b, out_ctx_f32, n_steps, rank, H);
}

void launch_inkling_mtp_rnn_step_gate(const f32* ctx_s_f32, const __nv_bfloat16* rnn_w,
                                      const __nv_bfloat16* markov_w1, const i32* d_prev_tok,
                                      __nv_bfloat16* state, __nv_bfloat16* latent,
                                      u32 rank, u32 H, cudaStream_t stream) {
    inkling_mtp_rnn_step_gate_kernel<<<1, rank, 0, stream>>>(
        ctx_s_f32, rnn_w, markov_w1, d_prev_tok, state, latent, rank, H);
}

} // namespace flint::cuda
