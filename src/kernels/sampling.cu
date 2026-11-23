#include "kernels.hpp"

namespace flint::cuda {

namespace {

template <bool COMPUTE_CONF>
__global__ void argmax_bf16_kernel(const __nv_bfloat16* __restrict__ logits,
                                   i32* __restrict__ d_out_tokens,
                                   f32* __restrict__ d_out_conf,
                                   u32 vocab_size) {
    const u32 m = blockIdx.x;
    const __nv_bfloat16* row = logits + static_cast<size_t>(m) * vocab_size;

    f32 best_val = -1e30f;
    i32 best_idx = 0;

    for (u32 i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        f32 v = __bfloat162float(row[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = static_cast<i32>(i);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        f32 ov = __shfl_down_sync(0xFFFFFFFFu, best_val, off);
        i32 oi = __shfl_down_sync(0xFFFFFFFFu, best_idx, off);
        if (ov > best_val || (ov == best_val && oi < best_idx)) {
            best_val = ov;
            best_idx = oi;
        }
    }

    __shared__ f32 s_val[32];
    __shared__ i32 s_idx[32];
    __shared__ f32 s_max;
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) {
        s_val[wid] = best_val;
        s_idx[wid] = best_idx;
    }
    __syncthreads();

    if (wid == 0) {
        u32 nwarps = blockDim.x >> 5;
        best_val = (lane < static_cast<int>(nwarps)) ? s_val[lane] : -1e30f;
        best_idx = (lane < static_cast<int>(nwarps)) ? s_idx[lane] : 0;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            f32 ov = __shfl_down_sync(0xFFFFFFFFu, best_val, off);
            i32 oi = __shfl_down_sync(0xFFFFFFFFu, best_idx, off);
            if (ov > best_val || (ov == best_val && oi < best_idx)) {
                best_val = ov;
                best_idx = oi;
            }
        }
        if (lane == 0) {
            d_out_tokens[m] = best_idx;
            s_max = best_val;
        }
    }

    if constexpr (COMPUTE_CONF) {
        __syncthreads();
        f32 mx = s_max;
        f32 sum_e = 0.0f;
        for (u32 i = threadIdx.x; i < vocab_size; i += blockDim.x) {
            f32 d = __bfloat162float(row[i]) - mx;
            if (d > -20.0f) sum_e += expf(d);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_e += __shfl_down_sync(0xFFFFFFFFu, sum_e, off);
        }
        if (lane == 0) s_val[wid] = sum_e;
        __syncthreads();
        if (wid == 0) {
            u32 nwarps = blockDim.x >> 5;
            f32 tot = (lane < static_cast<int>(nwarps)) ? s_val[lane] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                tot += __shfl_down_sync(0xFFFFFFFFu, tot, off);
            }
            if (lane == 0) {
                d_out_conf[m] = (tot > 0.0f) ? (1.0f / tot) : 0.0f;
            }
        }
    }
}

__global__ void advance_decode_state_kernel(i32* d_pos, i32* d_cur_token, const i32* d_next_token) {
    *d_pos = *d_pos + 1;
    *d_cur_token = *d_next_token;
}

} // namespace

void launch_argmax_bf16(const __nv_bfloat16* logits,
                        i32* d_out_tokens,
                        u32 M, u32 vocab_size,
                        cudaStream_t stream) {
    argmax_bf16_kernel<false><<<M, 512, 0, stream>>>(logits, d_out_tokens, nullptr, vocab_size);
}

void launch_argmax_conf_bf16(const __nv_bfloat16* logits,
                             i32* d_out_tokens,
                             f32* d_out_conf,
                             u32 M, u32 vocab_size,
                             cudaStream_t stream) {
    argmax_bf16_kernel<true><<<M, 512, 0, stream>>>(logits, d_out_tokens, d_out_conf, vocab_size);
}

void launch_advance_decode_state(i32* d_pos, i32* d_cur_token, const i32* d_next_token,
                                 cudaStream_t stream) {
    advance_decode_state_kernel<<<1, 1, 0, stream>>>(d_pos, d_cur_token, d_next_token);
}

} // namespace flint::cuda
