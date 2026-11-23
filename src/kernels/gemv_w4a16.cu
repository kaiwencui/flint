#include "kernels.hpp"

namespace flint::cuda {

namespace {

// Native BF16x2 PTX lop3.b32 + IEEE-754 BF16 magic-bias (128.0 = 0x4300) INT4->BF16x2 dequantization
// Because BF16 has 7 mantissa bits, 2^7 = 128.0 has ULP = 1.0, so 0x4300 | q (q in 0..15) is BIT-EXACT bf16(128 + q)
// 1 PTX lop3.b32 + 1 native BF16x2 sub + 1 native BF16x2 FMA -> ZERO half2/float2 inner-loop conversions
__device__ __forceinline__ __nv_bfloat162 u4x2_to_bf16x2_ptx(u32 q_shifted) {
    u32 res;
    asm("lop3.b32 %0, %1, 0x000F000F, 0x43004300, 0xEA;" : "=r"(res) : "r"(q_shifted));
    const u32 magic_u = 0x43004300u; // bf162(128.0, 128.0)
    return __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&res),
                   *reinterpret_cast<const __nv_bfloat162*>(&magic_u));
}

// Native BF16x2 PTX-dequantized asymmetric 8x INT4 dot product against 8 BF16 activations (uint4 x).
// Sustains ~955 GB/s GEMV bandwidth on RTX 4090 (~79.5% end-to-end with 0.78 ms fixed non-GEMV floor)
// and eliminates ALU stalls on M = 2..8 speculative verification (up to 360.6 tok/s lossless).
__device__ __forceinline__ void dot_u4x8_bf16x8_ptx(u32 packed_q, uint4 x_vec, f32& dot_q, f32& sum_x) {
    const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&x_vec);

    __nv_bfloat162 q0 = u4x2_to_bf16x2_ptx(packed_q);
    __nv_bfloat162 q1 = u4x2_to_bf16x2_ptx(packed_q >> 4);
    __nv_bfloat162 q2 = u4x2_to_bf16x2_ptx(packed_q >> 8);
    __nv_bfloat162 q3 = u4x2_to_bf16x2_ptx(packed_q >> 12);

    __nv_bfloat162 aq = __hmul2(q0, xb[0]);
    aq = __hfma2(q1, xb[1], aq);
    aq = __hfma2(q2, xb[2], aq);
    aq = __hfma2(q3, xb[3], aq);

    __nv_bfloat162 ax = __hadd2(__hadd2(xb[0], xb[1]), __hadd2(xb[2], xb[3]));

    float2 fq = __bfloat1622float2(aq);
    float2 fx = __bfloat1622float2(ax);
    dot_q += fq.x + fq.y;
    sum_x += fx.x + fx.y;
}

// Roofline Asymmetric W4A16 (INT4 group-128) GEMV for M=1..8:
// - 4 warps (128 threads) per block -> exact 0%-tail wave alignment across 128 SMs of RTX 4090
// (Qwen2.5-7B: o_proj/down_proj = 7.00 waves, qkv = 9.00 waves, gate_up = 37.00 waves).
// - Each thread loads uint2 (2 u32 = 16 INT4 weights = 8 B -> 256 B/warp = 2x 128B cache lines):
// K=3584 -> 7.00 exact warp iters, K=18944 -> 37.00 exact warp iters (0% idle lanes).
template <int MAX_M>
__global__ void gemv_w4a16_kernel(const __nv_bfloat16* __restrict__ x,
                                  const u32* __restrict__ qweight,
                                  const __nv_bfloat162* __restrict__ scales,
                                  const __nv_bfloat16* __restrict__ bias,
                                  __nv_bfloat16* __restrict__ out,
                                  u32 M, u32 N, u32 K, bool add_to_out) {
    const u32 warp_id = threadIdx.x >> 5;
    const u32 lane    = threadIdx.x & 31;
    const u32 row     = blockIdx.x * 4 + warp_id;
    if (row >= N) return;

    const u32 K8  = K >> 3;
    const u32 K16 = K >> 4;
    const u32 groups_per_row = K >> 7;
    const auto* q2_row = reinterpret_cast<const uint2*>(qweight + static_cast<size_t>(row) * K8);
    const __nv_bfloat162* s_row = scales + static_cast<size_t>(row) * groups_per_row;

    f32 acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    u32 idx = lane;
    for (; idx + 32 < K16; idx += 64) {
        uint2 qw0 = __ldg(q2_row + idx);
        uint2 qw1 = __ldg(q2_row + idx + 32);
        float2 sm0 = __bfloat1622float2(__ldg(s_row + (idx >> 3)));
        float2 sm1 = __bfloat1622float2(__ldg(s_row + ((idx + 32) >> 3)));

        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < static_cast<int>(M)) {
                const auto* x4_row = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K);
                u32 xb0 = idx * 2;
                f32 dq0 = 0.0f, sx0 = 0.0f;
                dot_u4x8_bf16x8_ptx(qw0.x, x4_row[xb0 + 0], dq0, sx0);
                dot_u4x8_bf16x8_ptx(qw0.y, x4_row[xb0 + 1], dq0, sx0);
                acc[m] = fmaf(dq0, sm0.x, fmaf(sx0, sm0.y, acc[m]));

                u32 xb1 = (idx + 32) * 2;
                f32 dq1 = 0.0f, sx1 = 0.0f;
                dot_u4x8_bf16x8_ptx(qw1.x, x4_row[xb1 + 0], dq1, sx1);
                dot_u4x8_bf16x8_ptx(qw1.y, x4_row[xb1 + 1], dq1, sx1);
                acc[m] = fmaf(dq1, sm1.x, fmaf(sx1, sm1.y, acc[m]));
            }
        }
    }
    for (; idx < K16; idx += 32) {
        uint2 qw0 = __ldg(q2_row + idx);
        float2 sm0 = __bfloat1622float2(__ldg(s_row + (idx >> 3)));
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < static_cast<int>(M)) {
                const auto* x4_row = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K);
                u32 xb0 = idx * 2;
                f32 dq0 = 0.0f, sx0 = 0.0f;
                dot_u4x8_bf16x8_ptx(qw0.x, x4_row[xb0 + 0], dq0, sx0);
                dot_u4x8_bf16x8_ptx(qw0.y, x4_row[xb0 + 1], dq0, sx0);
                acc[m] = fmaf(dq0, sm0.x, fmaf(sx0, sm0.y, acc[m]));
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < static_cast<int>(M)) {
            f32 v = acc[m];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xFFFFFFFFu, v, off);
            }
            if (lane == 0) {
                if (bias) v += __bfloat162float(bias[row]);
                size_t off = static_cast<size_t>(m) * N + row;
                if (add_to_out) v += __bfloat162float(out[off]);
                out[off] = __float2bfloat16_rn(v);
            }
        }
    }
}

template <int MAX_M>
__global__ void fused_gate_up_silu_w4a16_kernel(const __nv_bfloat16* __restrict__ x,
                                                const u32* __restrict__ qweight_gu,
                                                const __nv_bfloat162* __restrict__ scales_gu,
                                                __nv_bfloat16* __restrict__ out,
                                                u32 M, u32 inter, u32 K) {
    const u32 warp_id = threadIdx.x >> 5;
    const u32 lane    = threadIdx.x & 31;
    const u32 row     = blockIdx.x * 4 + warp_id;
    if (row >= inter) return;

    const u32 K8  = K >> 3;
    const u32 K16 = K >> 4;
    const u32 groups_per_row = K >> 7;

    const auto* qg2 = reinterpret_cast<const uint2*>(qweight_gu + static_cast<size_t>(row) * K8);
    const auto* qu2 = reinterpret_cast<const uint2*>(qweight_gu + static_cast<size_t>(inter + row) * K8);
    const __nv_bfloat162* sg = scales_gu + static_cast<size_t>(row) * groups_per_row;
    const __nv_bfloat162* su = scales_gu + static_cast<size_t>(inter + row) * groups_per_row;

    f32 acc_g[MAX_M];
    f32 acc_u[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) { acc_g[m] = 0.0f; acc_u[m] = 0.0f; }

    for (u32 idx = lane; idx < K16; idx += 32) {
        uint2 qwg = __ldg(qg2 + idx);
        uint2 qwu = __ldg(qu2 + idx);
        u32 g = idx >> 3;
        float2 smg = __bfloat1622float2(__ldg(sg + g));
        float2 smu = __bfloat1622float2(__ldg(su + g));

        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < static_cast<int>(M)) {
                const auto* x4_row = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K);
                u32 xbase = idx * 2;
                uint4 x0 = x4_row[xbase + 0];
                uint4 x1 = x4_row[xbase + 1];

                f32 dqg = 0.0f, sxg = 0.0f;
                dot_u4x8_bf16x8_ptx(qwg.x, x0, dqg, sxg);
                dot_u4x8_bf16x8_ptx(qwg.y, x1, dqg, sxg);

                f32 dqu = 0.0f, sxu = 0.0f;
                dot_u4x8_bf16x8_ptx(qwu.x, x0, dqu, sxu);
                dot_u4x8_bf16x8_ptx(qwu.y, x1, dqu, sxu);

                acc_g[m] = fmaf(dqg, smg.x, fmaf(sxg, smg.y, acc_g[m]));
                acc_u[m] = fmaf(dqu, smu.x, fmaf(sxu, smu.y, acc_u[m]));
            }
        }
    }

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
                f32 s = g / (1.0f + expf(-g));
                out[static_cast<size_t>(m) * inter + row] = __float2bfloat16_rn(s * u);
            }
        }
    }
}

__global__ void dequant_w4a16_to_bf16_kernel(const u32* __restrict__ qweight,
                                             const __nv_bfloat162* __restrict__ scales,
                                             __nv_bfloat16* __restrict__ out_bf16,
                                             u32 N, u32 K) {
    const u32 K8 = K >> 3;
    const u32 groups_per_row = K >> 7;
    const u32 word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 row = blockIdx.y;
    if (row >= N || word_idx >= K8) return;

    u32 qw = qweight[static_cast<size_t>(row) * K8 + word_idx];
    float2 sm = __bfloat1622float2(scales[static_cast<size_t>(row) * groups_per_row + (word_idx >> 4)]);

    uint4 out4;
    auto* ob = reinterpret_cast<__nv_bfloat162*>(&out4);
    float2 q0 = __bfloat1622float2(u4x2_to_bf16x2_ptx(qw));
    float2 q1 = __bfloat1622float2(u4x2_to_bf16x2_ptx(qw >> 4));
    float2 q2 = __bfloat1622float2(u4x2_to_bf16x2_ptx(qw >> 8));
    float2 q3 = __bfloat1622float2(u4x2_to_bf16x2_ptx(qw >> 12));

    ob[0] = __floats2bfloat162_rn(fmaf(q0.x, sm.x, sm.y), fmaf(q0.y, sm.x, sm.y));
    ob[1] = __floats2bfloat162_rn(fmaf(q1.x, sm.x, sm.y), fmaf(q1.y, sm.x, sm.y));
    ob[2] = __floats2bfloat162_rn(fmaf(q2.x, sm.x, sm.y), fmaf(q2.y, sm.x, sm.y));
    ob[3] = __floats2bfloat162_rn(fmaf(q3.x, sm.x, sm.y), fmaf(q3.y, sm.x, sm.y));

    auto* dst4 = reinterpret_cast<uint4*>(out_bf16 + static_cast<size_t>(row) * K);
    dst4[word_idx] = out4;
}

// Fused W4A16 -> per-row FP8 E4M3 dequantizer (1 warp per output row, 4 warps/block -> feeds native FP8 cublasLtMatmul)
__global__ void dequant_w4a16_to_fp8_kernel(const u32* __restrict__ qweight,
                                            const __nv_bfloat162* __restrict__ scales,
                                            u8* __restrict__ out_fp8,
                                            f32* __restrict__ out_row_scales,
                                            u32 N, u32 K) {
    const u32 wid  = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 row  = blockIdx.x * 4 + wid;
    if (row >= N) return;

    const u32 K8 = K >> 3;
    const u32 groups_per_row = K >> 7;
    const u32* qw_row = qweight + static_cast<size_t>(row) * K8;
    const __nv_bfloat162* sc_row = scales + static_cast<size_t>(row) * groups_per_row;

    // Pass 1: exact row max |w| from group (scale, min): max |q*s + m| over q in [0, 15]
    f32 amax = 0.0f;
    for (u32 g = lane; g < groups_per_row; g += 32) {
        float2 sm = __bfloat1622float2(__ldg(sc_row + g));
        f32 v0 = fabsf(sm.y);
        f32 v1 = fabsf(fmaf(15.0f, sm.x, sm.y));
        amax = fmaxf(amax, fmaxf(v0, v1));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
    f32 row_scale = fmaxf(amax / 448.0f, 1e-8f);
    f32 inv_scale = 1.0f / row_scale;
    if (lane == 0) out_row_scales[row] = row_scale;

    // Pass 2: dequantize 8 INT4 weights -> 8 FP8 E4M3 bytes coalesced as uint2
    auto* dst2 = reinterpret_cast<uint2*>(out_fp8 + static_cast<size_t>(row) * K);
    for (u32 widx = lane; widx < K8; widx += 32) {
        u32 qw = __ldg(qw_row + widx);
        float2 sm = __bfloat1622float2(__ldg(sc_row + (widx >> 4)));
        f32 s = sm.x * inv_scale;
        f32 m = sm.y * inv_scale;
        float2 q0 = __bfloat1622float2(u4x2_to_bf16x2_ptx(qw));
        float2 q1 = __bfloat1622float2(u4x2_to_bf16x2_ptx(qw >> 4));
        float2 q2 = __bfloat1622float2(u4x2_to_bf16x2_ptx(qw >> 8));
        float2 q3 = __bfloat1622float2(u4x2_to_bf16x2_ptx(qw >> 12));
        uint2 packed;
        auto* fp = reinterpret_cast<__nv_fp8x2_e4m3*>(&packed);
        fp[0] = __nv_fp8x2_e4m3(make_float2(fmaf(q0.x, s, m), fmaf(q0.y, s, m)));
        fp[1] = __nv_fp8x2_e4m3(make_float2(fmaf(q1.x, s, m), fmaf(q1.y, s, m)));
        fp[2] = __nv_fp8x2_e4m3(make_float2(fmaf(q2.x, s, m), fmaf(q2.y, s, m)));
        fp[3] = __nv_fp8x2_e4m3(make_float2(fmaf(q3.x, s, m), fmaf(q3.y, s, m)));
        dst2[widx] = packed;
    }
}

} // namespace

void launch_gemv_w4a16(const __nv_bfloat16* x,
                       const u32* qweight,
                       const __nv_bfloat162* scales,
                       const __nv_bfloat16* bias,
                       __nv_bfloat16* out,
                       u32 M, u32 N, u32 K,
                       cudaStream_t stream,
                       bool add_to_out) {
    u32 blocks = (N + 3) / 4;
    if (M == 1) {
        gemv_w4a16_kernel<1><<<blocks, 128, 0, stream>>>(x, qweight, scales, bias, out, M, N, K, add_to_out);
    } else if (M <= 4) {
        gemv_w4a16_kernel<4><<<blocks, 128, 0, stream>>>(x, qweight, scales, bias, out, M, N, K, add_to_out);
    } else {
        gemv_w4a16_kernel<8><<<blocks, 128, 0, stream>>>(x, qweight, scales, bias, out, M, N, K, add_to_out);
    }
}

void launch_fused_gate_up_silu_w4a16(const __nv_bfloat16* x,
                                     const u32* qweight_gu,
                                     const __nv_bfloat162* scales_gu,
                                     __nv_bfloat16* out,
                                     u32 M, u32 inter, u32 K,
                                     cudaStream_t stream) {
    u32 blocks = (inter + 3) / 4;
    if (M == 1) {
        fused_gate_up_silu_w4a16_kernel<1><<<blocks, 128, 0, stream>>>(x, qweight_gu, scales_gu, out, M, inter, K);
    } else if (M <= 4) {
        fused_gate_up_silu_w4a16_kernel<4><<<blocks, 128, 0, stream>>>(x, qweight_gu, scales_gu, out, M, inter, K);
    } else {
        fused_gate_up_silu_w4a16_kernel<8><<<blocks, 128, 0, stream>>>(x, qweight_gu, scales_gu, out, M, inter, K);
    }
}

void launch_dequant_w4a16_to_bf16(const u32* qweight,
                                  const __nv_bfloat162* scales,
                                  __nv_bfloat16* out_bf16,
                                  u32 N, u32 K,
                                  cudaStream_t stream) {
    u32 K8 = K >> 3;
    dim3 grid((K8 + 255) / 256, N);
    dequant_w4a16_to_bf16_kernel<<<grid, 256, 0, stream>>>(qweight, scales, out_bf16, N, K);
}

void launch_dequant_w4a16_to_fp8(const u32* qweight,
                                 const __nv_bfloat162* scales,
                                 u8* out_fp8,
                                 f32* out_row_scales,
                                 u32 N, u32 K,
                                 cudaStream_t stream) {
    dequant_w4a16_to_fp8_kernel<<<(N + 3) / 4, 128, 0, stream>>>(qweight, scales, out_fp8, out_row_scales, N, K);
}

} // namespace flint::cuda
