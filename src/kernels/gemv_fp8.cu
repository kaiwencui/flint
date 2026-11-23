#include "kernels.hpp"

namespace flint::cuda {

namespace {

__device__ __forceinline__ f32 dot_fp8x16_bf16x16(uint4 w16, uint4 x_lo, uint4 x_hi) {
    const auto* fp8_2 = reinterpret_cast<const __nv_fp8x2_e4m3*>(&w16);
    const auto* xb0 = reinterpret_cast<const __nv_bfloat162*>(&x_lo);
    const auto* xb1 = reinterpret_cast<const __nv_bfloat162*>(&x_hi);
    f32 s = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        float2 fw = static_cast<float2>(fp8_2[k]);
        float2 fx = __bfloat1622float2(xb0[k]);
        s = fmaf(fw.x, fx.x, s);
        s = fmaf(fw.y, fx.y, s);
    }
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        float2 fw = static_cast<float2>(fp8_2[4 + k]);
        float2 fx = __bfloat1622float2(xb1[k]);
        s = fmaf(fw.x, fx.x, s);
        s = fmaf(fw.y, fx.y, s);
    }
    return s;
}

// Roofline FP8 E4M3 GEMV for M=1..8 (Ada SM_89 / Hopper SM_90 hardware FP8 conversion)
template <int MAX_M>
__global__ void gemv_fp8_kernel(const __nv_bfloat16* __restrict__ x,
                                const u8* __restrict__ qweight,
                                const f32* __restrict__ scales,
                                const __nv_bfloat16* __restrict__ bias,
                                __nv_bfloat16* __restrict__ out,
                                u32 M, u32 N, u32 K, bool add_to_out) {
    const u32 warp_id = threadIdx.x >> 5;
    const u32 lane    = threadIdx.x & 31;
    const u32 row     = blockIdx.x * 4 + warp_id;
    if (row >= N) return;

    const u32 K16 = K >> 4;
    const auto* w16_row = reinterpret_cast<const uint4*>(qweight + static_cast<size_t>(row) * K);

    f32 acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (u32 idx = lane; idx < K16; idx += 32) {
        uint4 wv = __ldg(w16_row + idx);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < static_cast<int>(M)) {
                const auto* x8_row = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K);
                uint4 x0 = x8_row[idx * 2];
                uint4 x1 = x8_row[idx * 2 + 1];
                acc[m] += dot_fp8x16_bf16x16(wv, x0, x1);
            }
        }
    }

    f32 sc = scales[row];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < static_cast<int>(M)) {
            f32 v = acc[m];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xFFFFFFFFu, v, off);
            }
            if (lane == 0) {
                v *= sc;
                if (bias) v += __bfloat162float(bias[row]);
                size_t off = static_cast<size_t>(m) * N + row;
                if (add_to_out) v += __bfloat162float(out[off]);
                out[off] = __float2bfloat16_rn(v);
            }
        }
    }
}

template <int MAX_M>
__global__ void fused_gate_up_silu_fp8_kernel(const __nv_bfloat16* __restrict__ x,
                                              const u8* __restrict__ qweight_gu,
                                              const f32* __restrict__ scales_gu,
                                              __nv_bfloat16* __restrict__ out,
                                              u32 M, u32 inter, u32 K) {
    const u32 warp_id = threadIdx.x >> 5;
    const u32 lane    = threadIdx.x & 31;
    const u32 row     = blockIdx.x * 4 + warp_id;
    if (row >= inter) return;

    const u32 K16 = K >> 4;
    const auto* wg16 = reinterpret_cast<const uint4*>(qweight_gu + static_cast<size_t>(row) * K);
    const auto* wu16 = reinterpret_cast<const uint4*>(qweight_gu + static_cast<size_t>(inter + row) * K);

    f32 acc_g[MAX_M];
    f32 acc_u[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) { acc_g[m] = 0.0f; acc_u[m] = 0.0f; }

    for (u32 idx = lane; idx < K16; idx += 32) {
        uint4 wvg = __ldg(wg16 + idx);
        uint4 wvu = __ldg(wu16 + idx);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < static_cast<int>(M)) {
                const auto* x8_row = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K);
                uint4 x0 = x8_row[idx * 2];
                uint4 x1 = x8_row[idx * 2 + 1];
                acc_g[m] += dot_fp8x16_bf16x16(wvg, x0, x1);
                acc_u[m] += dot_fp8x16_bf16x16(wvu, x0, x1);
            }
        }
    }

    f32 scg = scales_gu[row];
    f32 scu = scales_gu[inter + row];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < static_cast<int>(M)) {
            f32 g = acc_g[m];
            f32 u = acc_u[m];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                g += __shfl_down_sync(0xFFFFFFFFu, g, off);
                u += __shfl_down_sync(0xFFFFFFFFu, u, off);
            }
            if (lane == 0) {
                g *= scg;
                u *= scu;
                f32 s = g / (1.0f + expf(-g));
                out[static_cast<size_t>(m) * inter + row] = __float2bfloat16_rn(s * u);
            }
        }
    }
}

__global__ void dequant_fp8_to_bf16_kernel(const u8* __restrict__ qweight,
                                           const f32* __restrict__ scales,
                                           __nv_bfloat16* __restrict__ out_bf16,
                                           u32 N, u32 K) {
    u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    u32 row = blockIdx.y;
    if (row >= N || col >= K) return;
    __nv_fp8_e4m3 v;
    *reinterpret_cast<u8*>(&v) = qweight[static_cast<size_t>(row) * K + col];
    f32 f = static_cast<f32>(v) * scales[row];
    out_bf16[static_cast<size_t>(row) * K + col] = __float2bfloat16_rn(f);
}

__global__ void bf16_amax_kernel(const __nv_bfloat16* __restrict__ x,
                                 u32* __restrict__ d_amax_u32,
                                 size_t total_elems) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    f32 local_max = 0.0f;
    const auto* x4 = reinterpret_cast<const uint4*>(x);
    size_t n4 = total_elems >> 3;
    for (size_t i = idx; i < n4; i += stride) {
        uint4 v = x4[i];
        const auto* b2 = reinterpret_cast<const __nv_bfloat162*>(&v);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 f = __bfloat1622float2(b2[k]);
            local_max = fmaxf(local_max, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFFu, local_max, off));
    }
    __shared__ f32 sm[8];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) sm[wid] = local_max;
    __syncthreads();
    if (threadIdx.x == 0) {
        f32 bmax = 0.0f;
        for (int w = 0; w < 8; ++w) bmax = fmaxf(bmax, sm[w]);
        atomicMax(d_amax_u32, __float_as_uint(bmax));
    }
}

__global__ void quant_bf16_to_fp8_kernel(const __nv_bfloat16* __restrict__ x,
                                         u8* __restrict__ x_fp8,
                                         f32* __restrict__ d_amax_and_scale,
                                         u32 M, u32 M_pad, u32 K) {
    f32 amax = __uint_as_float(*reinterpret_cast<const u32*>(d_amax_and_scale));
    f32 scale = fmaxf(amax / 448.0f, 1e-6f);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        d_amax_and_scale[1] = scale;
    }
    f32 inv_scale = 1.0f / scale;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    size_t real_n8 = (static_cast<size_t>(M) * K) >> 3;
    size_t pad_n8  = (static_cast<size_t>(M_pad) * K) >> 3;
    const auto* x4 = reinterpret_cast<const uint4*>(x);
    auto* dst2 = reinterpret_cast<uint2*>(x_fp8);
    for (size_t i = idx; i < pad_n8; i += stride) {
        if (i < real_n8) {
            uint4 v = x4[i];
            const auto* b2 = reinterpret_cast<const __nv_bfloat162*>(&v);
            uint2 out8;
            auto* o2 = reinterpret_cast<__nv_fp8x2_e4m3*>(&out8);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f = __bfloat1622float2(b2[k]);
                f.x *= inv_scale;
                f.y *= inv_scale;
                o2[k] = __nv_fp8x2_e4m3(f);
            }
            dst2[i] = out8;
        } else {
            dst2[i] = make_uint2(0, 0);
        }
    }
}

__global__ void apply_row_scales_bf16_kernel(__nv_bfloat16* __restrict__ out,
                                             const f32* __restrict__ row_scales,
                                             const __nv_bfloat16* __restrict__ bias,
                                             u32 M, u32 N) {
    u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    u32 row = blockIdx.y;
    if (row >= M || col >= N) return;
    size_t off = static_cast<size_t>(row) * N + col;
    f32 v = __bfloat162float(out[off]) * __ldg(row_scales + col);
    if (bias) v += __bfloat162float(__ldg(bias + col));
    out[off] = __float2bfloat16_rn(v);
}

} // namespace

void launch_gemv_fp8(const __nv_bfloat16* x,
                     const u8* qweight,
                     const f32* scales,
                     const __nv_bfloat16* bias,
                     __nv_bfloat16* out,
                     u32 M, u32 N, u32 K,
                     cudaStream_t stream,
                     bool add_to_out) {
    u32 blocks = (N + 3) / 4;
    if (M == 1) {
        gemv_fp8_kernel<1><<<blocks, 128, 0, stream>>>(x, qweight, scales, bias, out, M, N, K, add_to_out);
    } else if (M <= 4) {
        gemv_fp8_kernel<4><<<blocks, 128, 0, stream>>>(x, qweight, scales, bias, out, M, N, K, add_to_out);
    } else {
        gemv_fp8_kernel<8><<<blocks, 128, 0, stream>>>(x, qweight, scales, bias, out, M, N, K, add_to_out);
    }
}

void launch_fused_gate_up_silu_fp8(const __nv_bfloat16* x,
                                   const u8* qweight_gu,
                                   const f32* scales_gu,
                                   __nv_bfloat16* out,
                                   u32 M, u32 inter, u32 K,
                                   cudaStream_t stream) {
    u32 blocks = (inter + 3) / 4;
    if (M == 1) {
        fused_gate_up_silu_fp8_kernel<1><<<blocks, 128, 0, stream>>>(x, qweight_gu, scales_gu, out, M, inter, K);
    } else if (M <= 4) {
        fused_gate_up_silu_fp8_kernel<4><<<blocks, 128, 0, stream>>>(x, qweight_gu, scales_gu, out, M, inter, K);
    } else {
        fused_gate_up_silu_fp8_kernel<8><<<blocks, 128, 0, stream>>>(x, qweight_gu, scales_gu, out, M, inter, K);
    }
}

void launch_dequant_fp8_to_bf16(const u8* qweight,
                                const f32* scales,
                                __nv_bfloat16* out_bf16,
                                u32 N, u32 K,
                                cudaStream_t stream) {
    dim3 grid((K + 255) / 256, N);
    dequant_fp8_to_bf16_kernel<<<grid, 256, 0, stream>>>(qweight, scales, out_bf16, N, K);
}

void launch_quant_bf16_to_fp8_pertensor(const __nv_bfloat16* x,
                                        u8* x_fp8,
                                        f32* d_amax_and_scale,
                                        u32 M, u32 M_pad, u32 K,
                                        cudaStream_t stream) {
    cudaMemsetAsync(d_amax_and_scale, 0, sizeof(u32), stream);
    size_t total = static_cast<size_t>(M) * K;
    u32 blocks = static_cast<u32>(std::min<size_t>(256, (total / 8 + 255) / 256));
    if (blocks == 0) blocks = 1;
    bf16_amax_kernel<<<blocks, 256, 0, stream>>>(x, reinterpret_cast<u32*>(d_amax_and_scale), total);
    u32 qblocks = static_cast<u32>(std::min<size_t>(512, (static_cast<size_t>(M_pad) * K / 8 + 255) / 256));
    if (qblocks == 0) qblocks = 1;
    quant_bf16_to_fp8_kernel<<<qblocks, 256, 0, stream>>>(x, x_fp8, d_amax_and_scale, M, M_pad, K);
}

void launch_apply_row_scales_bf16(__nv_bfloat16* out,
                                  const f32* row_scales,
                                  const __nv_bfloat16* bias,
                                  u32 M, u32 N,
                                  cudaStream_t stream) {
    dim3 grid((N + 255) / 256, M);
    apply_row_scales_bf16_kernel<<<grid, 256, 0, stream>>>(out, row_scales, bias, M, N);
}

} // namespace flint::cuda
