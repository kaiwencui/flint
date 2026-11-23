#include "kernels.hpp"

namespace flint::cuda {

namespace {

// Roofline BF16 GEMV for M=1..8:
// 1 warp computes 1 output row, 4 warps (128 threads) per block.
// Each thread loads uint4 (8 BF16 weights = 16 bytes) coalesced across 32 threads (512 bytes/iter).
// For M > 1 (speculative verification batch), the SAME weight load from HBM is multiplied against
// all M activation vectors in registers -> M tokens verified for nearly the cost of 1
template <int MAX_M>
__global__ void gemv_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                 const __nv_bfloat16* __restrict__ W,
                                 const __nv_bfloat16* __restrict__ bias,
                                 __nv_bfloat16* __restrict__ out,
                                 u32 M, u32 N, u32 K, bool add_to_out) {
    const u32 warp_id = threadIdx.x >> 5;
    const u32 lane    = threadIdx.x & 31;
    const u32 row     = blockIdx.x * 4 + warp_id;
    if (row >= N) return;

    const u32 K8 = K >> 3;
    const auto* w_row = reinterpret_cast<const uint4*>(W + static_cast<size_t>(row) * K);

    f32 acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    // Unroll 2x along K8 to hide HBM latency with dual outstanding 128-bit loads
    u32 idx = lane;
    for (; idx + 32 < K8; idx += 64) {
        uint4 w0 = __ldg(w_row + idx);
        uint4 w1 = __ldg(w_row + idx + 32);
        const auto* wb0 = reinterpret_cast<const __nv_bfloat162*>(&w0);
        const auto* wb1 = reinterpret_cast<const __nv_bfloat162*>(&w1);

        // Prevent nvcc 12.8 sm_120 auto-unroller from truncating 64-bit x_row pointers
        #pragma unroll 1
        for (int m = 0; m < MAX_M; ++m) {
            if (m < static_cast<int>(M)) {
                const auto* x_row = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K);
                uint4 x0 = x_row[idx];
                uint4 x1 = x_row[idx + 32];
                const auto* xb0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
                const auto* xb1 = reinterpret_cast<const __nv_bfloat162*>(&x1);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 fw0 = __bfloat1622float2(wb0[j]);
                    float2 fx0 = __bfloat1622float2(xb0[j]);
                    acc[m] = fmaf(fw0.x, fx0.x, acc[m]);
                    acc[m] = fmaf(fw0.y, fx0.y, acc[m]);

                    float2 fw1 = __bfloat1622float2(wb1[j]);
                    float2 fx1 = __bfloat1622float2(xb1[j]);
                    acc[m] = fmaf(fw1.x, fx1.x, acc[m]);
                    acc[m] = fmaf(fw1.y, fx1.y, acc[m]);
                }
            }
        }
    }
    for (; idx < K8; idx += 32) {
        uint4 w0 = __ldg(w_row + idx);
        const auto* wb0 = reinterpret_cast<const __nv_bfloat162*>(&w0);
        // Prevent nvcc 12.8 sm_120 auto-unroller from truncating 64-bit x_row pointers
        #pragma unroll 1
        for (int m = 0; m < MAX_M; ++m) {
            if (m < static_cast<int>(M)) {
                const auto* x_row = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K);
                uint4 x0 = x_row[idx];
                const auto* xb0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 fw0 = __bfloat1622float2(wb0[j]);
                    float2 fx0 = __bfloat1622float2(xb0[j]);
                    acc[m] = fmaf(fw0.x, fx0.x, acc[m]);
                    acc[m] = fmaf(fw0.y, fx0.y, acc[m]);
                }
            }
        }
    }

    // Warp shuffle reduction
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

// Fused Gate + Up + SiLU*Mul GEMV (eliminates intermediate 2*inter HBM write + read)
template <int MAX_M>
__global__ void fused_gate_up_silu_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                               const __nv_bfloat16* __restrict__ W_gu,
                                               __nv_bfloat16* __restrict__ out,
                                               u32 M, u32 inter, u32 K) {
    const u32 warp_id = threadIdx.x >> 5;
    const u32 lane    = threadIdx.x & 31;
    const u32 row     = blockIdx.x * 4 + warp_id;
    if (row >= inter) return;

    const u32 K8 = K >> 3;
    const auto* wg_row = reinterpret_cast<const uint4*>(W_gu + static_cast<size_t>(row) * K);
    const auto* wu_row = reinterpret_cast<const uint4*>(W_gu + static_cast<size_t>(inter + row) * K);

    f32 acc_g[MAX_M];
    f32 acc_u[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) { acc_g[m] = 0.0f; acc_u[m] = 0.0f; }

    for (u32 idx = lane; idx < K8; idx += 32) {
        uint4 wg = __ldg(wg_row + idx);
        uint4 wu = __ldg(wu_row + idx);
        const auto* wgb = reinterpret_cast<const __nv_bfloat162*>(&wg);
        const auto* wub = reinterpret_cast<const __nv_bfloat162*>(&wu);

        // Prevent nvcc 12.8 sm_120 auto-unroller from truncating 64-bit x_row pointers
        #pragma unroll 1
        for (int m = 0; m < MAX_M; ++m) {
            if (m < static_cast<int>(M)) {
                const auto* x_row = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K);
                uint4 xv = x_row[idx];
                const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 fx = __bfloat1622float2(xb[j]);
                    float2 fg = __bfloat1622float2(wgb[j]);
                    float2 fu = __bfloat1622float2(wub[j]);
                    acc_g[m] = fmaf(fg.x, fx.x, acc_g[m]);
                    acc_g[m] = fmaf(fg.y, fx.y, acc_g[m]);
                    acc_u[m] = fmaf(fu.x, fx.x, acc_u[m]);
                    acc_u[m] = fmaf(fu.y, fx.y, acc_u[m]);
                }
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

} // namespace

void launch_gemv_bf16(const __nv_bfloat16* x,
                      const __nv_bfloat16* W,
                      const __nv_bfloat16* bias,
                      __nv_bfloat16* out,
                      u32 M, u32 N, u32 K,
                      cudaStream_t stream,
                      bool add_to_out) {
    u32 blocks = (N + 3) / 4;
    if (M == 1) {
        gemv_bf16_kernel<1><<<blocks, 128, 0, stream>>>(x, W, bias, out, M, N, K, add_to_out);
    } else if (M <= 4) {
        gemv_bf16_kernel<4><<<blocks, 128, 0, stream>>>(x, W, bias, out, M, N, K, add_to_out);
    } else {
        gemv_bf16_kernel<8><<<blocks, 128, 0, stream>>>(x, W, bias, out, M, N, K, add_to_out);
    }
}

void launch_fused_gate_up_silu_bf16(const __nv_bfloat16* x,
                                    const __nv_bfloat16* W_gate_up,
                                    __nv_bfloat16* out,
                                    u32 M, u32 inter, u32 K,
                                    cudaStream_t stream) {
    u32 blocks = (inter + 3) / 4;
    if (M == 1) {
        fused_gate_up_silu_bf16_kernel<1><<<blocks, 128, 0, stream>>>(x, W_gate_up, out, M, inter, K);
    } else if (M <= 4) {
        fused_gate_up_silu_bf16_kernel<4><<<blocks, 128, 0, stream>>>(x, W_gate_up, out, M, inter, K);
    } else {
        fused_gate_up_silu_bf16_kernel<8><<<blocks, 128, 0, stream>>>(x, W_gate_up, out, M, inter, K);
    }
}

} // namespace flint::cuda
