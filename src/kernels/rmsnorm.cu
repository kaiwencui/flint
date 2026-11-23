#include "kernels.hpp"

namespace flint::cuda {

namespace {

__global__ void embed_lookup_bf16_kernel(const __nv_bfloat16* __restrict__ embed_table,
                                         const i32* __restrict__ d_token_ids,
                                         __nv_bfloat16* __restrict__ out,
                                         u32 hidden_size) {
    const u32 m = blockIdx.y;
    const i32 tok = d_token_ids[m];
    const u32 vec_elems = hidden_size / 8;
    const u32 idx = blockIdx.x * blockDim.x + threadIdx.x;

    const auto* src = reinterpret_cast<const uint4*>(embed_table + static_cast<size_t>(tok) * hidden_size);
    auto* dst = reinterpret_cast<uint4*>(out + static_cast<size_t>(m) * hidden_size);
    if (idx < vec_elems) {
        dst[idx] = src[idx];
    }
}

__device__ __forceinline__ f32 block_reduce_sum(f32 val) {
    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
    }
    __shared__ f32 smem[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
        }
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

__global__ void rmsnorm_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ w,
                                    __nv_bfloat16* __restrict__ out,
                                    u32 hidden_size, f32 eps) {
    const u32 row = blockIdx.x;
    const u32 vec_n = hidden_size / 8;
    const auto* x4 = reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * hidden_size);
    const auto* w4 = reinterpret_cast<const uint4*>(w);
    auto* o4 = reinterpret_cast<uint4*>(out + static_cast<size_t>(row) * hidden_size);

    f32 ss = 0.0f;
    for (u32 i = threadIdx.x; i < vec_n; i += blockDim.x) {
        uint4 v = x4[i];
        const auto* b2 = reinterpret_cast<const __nv_bfloat162*>(&v);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 f = __bfloat1622float2(b2[k]);
            ss = fmaf(f.x, f.x, ss);
            ss = fmaf(f.y, f.y, ss);
        }
    }
    f32 total_ss = block_reduce_sum(ss);
    f32 inv_rms = rsqrtf(total_ss / static_cast<f32>(hidden_size) + eps);

    for (u32 i = threadIdx.x; i < vec_n; i += blockDim.x) {
        uint4 xv = x4[i];
        uint4 wv = w4[i];
        uint4 ov;
        const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
        const auto* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
        auto* ob = reinterpret_cast<__nv_bfloat162*>(&ov);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 fx = __bfloat1622float2(xb[k]);
            float2 fw = __bfloat1622float2(wb[k]);
            ob[k] = __floats2bfloat162_rn(fx.x * inv_rms * fw.x, fx.y * inv_rms * fw.y);
        }
        o4[i] = ov;
    }
}

__global__ void fused_residual_rmsnorm_bf16_kernel(__nv_bfloat16* __restrict__ residual,
                                                   const __nv_bfloat16* __restrict__ delta,
                                                   const __nv_bfloat16* __restrict__ w,
                                                   __nv_bfloat16* __restrict__ norm_out,
                                                   u32 hidden_size, f32 eps) {
    const u32 row = blockIdx.x;
    const u32 vec_n = hidden_size / 8;
    auto* r4 = reinterpret_cast<uint4*>(residual + static_cast<size_t>(row) * hidden_size);
    const auto* d4 = reinterpret_cast<const uint4*>(delta + static_cast<size_t>(row) * hidden_size);
    const auto* w4 = reinterpret_cast<const uint4*>(w);
    auto* o4 = reinterpret_cast<uint4*>(norm_out + static_cast<size_t>(row) * hidden_size);

    f32 ss = 0.0f;
    for (u32 i = threadIdx.x; i < vec_n; i += blockDim.x) {
        uint4 rv = r4[i];
        uint4 dv = d4[i];
        uint4 sumv;
        const auto* rb = reinterpret_cast<const __nv_bfloat162*>(&rv);
        const auto* db = reinterpret_cast<const __nv_bfloat162*>(&dv);
        auto* sb = reinterpret_cast<__nv_bfloat162*>(&sumv);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 fr = __bfloat1622float2(rb[k]);
            float2 fd = __bfloat1622float2(db[k]);
            f32 sx = fr.x + fd.x;
            f32 sy = fr.y + fd.y;
            ss = fmaf(sx, sx, ss);
            ss = fmaf(sy, sy, ss);
            sb[k] = __floats2bfloat162_rn(sx, sy);
        }
        r4[i] = sumv;
    }

    f32 total_ss = block_reduce_sum(ss);
    f32 inv_rms = rsqrtf(total_ss / static_cast<f32>(hidden_size) + eps);

    for (u32 i = threadIdx.x; i < vec_n; i += blockDim.x) {
        uint4 rv = r4[i];
        uint4 wv = w4[i];
        uint4 ov;
        const auto* rb = reinterpret_cast<const __nv_bfloat162*>(&rv);
        const auto* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
        auto* ob = reinterpret_cast<__nv_bfloat162*>(&ov);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 fx = __bfloat1622float2(rb[k]);
            float2 fw = __bfloat1622float2(wb[k]);
            ob[k] = __floats2bfloat162_rn(fx.x * inv_rms * fw.x, fx.y * inv_rms * fw.y);
        }
        o4[i] = ov;
    }
}

__global__ void add_residual_bf16_kernel(__nv_bfloat16* __restrict__ residual,
                                         const __nv_bfloat16* __restrict__ delta,
                                         u32 vec_count) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vec_count) return;
    auto* r4 = reinterpret_cast<uint4*>(residual);
    const auto* d4 = reinterpret_cast<const uint4*>(delta);
    uint4 rv = r4[idx];
    uint4 dv = d4[idx];
    auto* rb = reinterpret_cast<__nv_bfloat162*>(&rv);
    const auto* db = reinterpret_cast<const __nv_bfloat162*>(&dv);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        float2 fr = __bfloat1622float2(rb[k]);
        float2 fd = __bfloat1622float2(db[k]);
        rb[k] = __floats2bfloat162_rn(fr.x + fd.x, fr.y + fd.y);
    }
    r4[idx] = rv;
}

__global__ void silu_mul_prefill_bf16_kernel(const __nv_bfloat16* __restrict__ gate_up,
                                             __nv_bfloat16* __restrict__ out,
                                             u32 M, u32 inter) {
    u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    u32 row = blockIdx.y;
    if (row >= M || col >= inter) return;
    const __nv_bfloat16* row_in = gate_up + static_cast<size_t>(row) * (2 * inter);
    f32 g = __bfloat162float(row_in[col]);
    f32 u = __bfloat162float(row_in[inter + col]);
    f32 s = g / (1.0f + expf(-g));
    out[static_cast<size_t>(row) * inter + col] = __float2bfloat16_rn(s * u);
}

__global__ void add_bias_bf16_kernel(__nv_bfloat16* __restrict__ out,
                                     const __nv_bfloat16* __restrict__ bias,
                                     u32 M, u32 N) {
    u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    u32 row = blockIdx.y;
    if (row >= M || col >= N) return;
    size_t idx = static_cast<size_t>(row) * N + col;
    out[idx] = __float2bfloat16_rn(__bfloat162float(out[idx]) + __bfloat162float(bias[col]));
}

__global__ void embed_lookup_fp8_kernel(const u8* __restrict__ embed_fp8, const f32* __restrict__ embed_scale,
                                        const i32* __restrict__ d_token_ids, __nv_bfloat16* __restrict__ out, u32 H) {
    const i32 tid = d_token_ids[blockIdx.x];
    const u8* src = embed_fp8 + static_cast<size_t>(tid) * H;
    const f32 sc  = __ldg(embed_scale + tid);
    __nv_bfloat16* dst = out + static_cast<size_t>(blockIdx.x) * H;
    for (u32 i = threadIdx.x; i < H; i += blockDim.x) {
        __nv_fp8_e4m3 fv; *reinterpret_cast<u8*>(&fv) = __ldg(src + i);
        dst[i] = __float2bfloat16_rn(static_cast<f32>(fv) * sc);
    }
}

__global__ void embed_lookup_w4a16_kernel(const u32* __restrict__ qw, const u32* __restrict__ sc,
                                          const i32* __restrict__ d_token_ids,
                                          __nv_bfloat16* __restrict__ out, u32 H) {
    const i32 tok = d_token_ids[blockIdx.x];
    const u32* wrow = qw + static_cast<size_t>(tok) * (H >> 3);
    const u32* srow = sc + static_cast<size_t>(tok) * (H >> 7);
    __nv_bfloat16* dst = out + static_cast<size_t>(blockIdx.x) * H;
    const u32 H8 = H >> 3;
    for (u32 w = threadIdx.x; w < H8; w += blockDim.x) {
        u32 g = w >> 4;
        u32 sm = __ldg(srow + g);
        float2 fs = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sm));
        f32 s = fs.x, z = fs.y;
        u32 packed = __ldg(wrow + w);
        // Marlin nibble order: q0,q2,q4,q6 in low 16b; q1,q3,q5,q7 in high 16b
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            f32 qe = static_cast<f32>((packed >> (k * 4)) & 0xFu);
            f32 qo = static_cast<f32>((packed >> (16 + k * 4)) & 0xFu);
            dst[w * 8 + 2 * k + 0] = __float2bfloat16_rn(fmaf(qe, s, z));
            dst[w * 8 + 2 * k + 1] = __float2bfloat16_rn(fmaf(qo, s, z));
        }
    }
}

__global__ void quant_bf16_to_fp8_rowwise_kernel(const __nv_bfloat16* __restrict__ src,
                                                 u8* __restrict__ dst_fp8, f32* __restrict__ dst_scales,
                                                 u32 rows, u32 cols) {
    const u32 wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const u32 r = blockIdx.x * 4 + wid;
    if (r >= rows) return;
    const __nv_bfloat16* srow = src + static_cast<size_t>(r) * cols;
    u8* drow = dst_fp8 + static_cast<size_t>(r) * cols;
    f32 am = 0.0f;
    for (u32 c = lane; c < cols; c += 32) am = fmaxf(am, fabsf(__bfloat162float(srow[c])));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, off));
    f32 sc = fmaxf(am / 448.0f, 1e-8f);
    f32 inv_sc = 1.0f / sc;
    if (lane == 0) dst_scales[r] = sc;
    for (u32 c = lane; c < cols; c += 32) {
        __nv_fp8_e4m3 fv(__bfloat162float(srow[c]) * inv_sc);
        drow[c] = *reinterpret_cast<const u8*>(&fv);
    }
}

} // namespace

void launch_embed_lookup_bf16(const __nv_bfloat16* embed_table,
                              const i32* d_token_ids,
                              __nv_bfloat16* out,
                              u32 M, u32 hidden_size,
                              cudaStream_t stream) {
    u32 vecs = hidden_size / 8;
    u32 threads = 256;
    dim3 grid((vecs + threads - 1) / threads, M);
    embed_lookup_bf16_kernel<<<grid, threads, 0, stream>>>(embed_table, d_token_ids, out, hidden_size);
}

void launch_embed_lookup_fp8(const u8* embed_fp8, const f32* embed_scale,
                             const i32* d_token_ids, __nv_bfloat16* out,
                             u32 M, u32 hidden_size, cudaStream_t stream) {
    embed_lookup_fp8_kernel<<<M, 256, 0, stream>>>(embed_fp8, embed_scale, d_token_ids, out, hidden_size);
}

void launch_embed_lookup_w4a16(const u32* embed_qweight, const u32* embed_scales,
                               const i32* d_token_ids, __nv_bfloat16* out,
                               u32 M, u32 hidden_size, cudaStream_t stream) {
    embed_lookup_w4a16_kernel<<<M, 256, 0, stream>>>(embed_qweight, embed_scales, d_token_ids, out, hidden_size);
}

void launch_quant_bf16_to_fp8_rowwise(const __nv_bfloat16* src,
                                      u8* dst_fp8, f32* dst_scales,
                                      u32 rows, u32 cols, cudaStream_t stream) {
    quant_bf16_to_fp8_rowwise_kernel<<<(rows + 3) / 4, 128, 0, stream>>>(src, dst_fp8, dst_scales, rows, cols);
}

void launch_rmsnorm_bf16(const __nv_bfloat16* x,
                         const __nv_bfloat16* w,
                         __nv_bfloat16* out,
                         u32 M, u32 hidden_size, f32 eps,
                         cudaStream_t stream) {
    u32 threads = std::min<u32>(256, std::max<u32>(32, ((hidden_size / 8 + 31) / 32) * 32));
    rmsnorm_bf16_kernel<<<M, threads, 0, stream>>>(x, w, out, hidden_size, eps);
}

void launch_fused_residual_rmsnorm_bf16(__nv_bfloat16* residual,
                                        const __nv_bfloat16* delta,
                                        const __nv_bfloat16* w,
                                        __nv_bfloat16* norm_out,
                                        u32 M, u32 hidden_size, f32 eps,
                                        cudaStream_t stream) {
    u32 threads = std::min<u32>(256, std::max<u32>(32, ((hidden_size / 8 + 31) / 32) * 32));
    fused_residual_rmsnorm_bf16_kernel<<<M, threads, 0, stream>>>(
        residual, delta, w, norm_out, hidden_size, eps);
}

void launch_add_residual_bf16(__nv_bfloat16* residual,
                              const __nv_bfloat16* delta,
                              u32 count,
                              cudaStream_t stream) {
    u32 vecs = count / 8;
    u32 threads = 256;
    u32 blocks = (vecs + threads - 1) / threads;
    add_residual_bf16_kernel<<<blocks, threads, 0, stream>>>(residual, delta, vecs);
}

void launch_silu_mul_prefill_bf16(const __nv_bfloat16* gate_up,
                                  __nv_bfloat16* out,
                                  u32 M, u32 inter,
                                  cudaStream_t stream) {
    u32 threads = 256;
    dim3 grid((inter + threads - 1) / threads, M);
    silu_mul_prefill_bf16_kernel<<<grid, threads, 0, stream>>>(gate_up, out, M, inter);
}

void launch_add_bias_bf16(__nv_bfloat16* out,
                          const __nv_bfloat16* bias,
                          u32 M, u32 N,
                          cudaStream_t stream) {
    u32 threads = 256;
    dim3 grid((N + threads - 1) / threads, M);
    add_bias_bf16_kernel<<<grid, threads, 0, stream>>>(out, bias, M, N);
}

} // namespace flint::cuda
