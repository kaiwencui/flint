#pragma once

#include "kernels.hpp"

namespace flint::cuda {

__device__ __forceinline__ f32 sigmoid_f32(f32 x) {
    return 1.0f / (1.0f + expf(-x));
}

// Orthonormal 128-point Walsh-Hadamard transform across one warp (32 lanes x 4 elements = 128).
// H = H^T and H^2 = I, so (Hq).(Hk) = q.k and H(sum p_t (Hv_t)) = sum p_t v_t.
__device__ __forceinline__ void ink_wht128_warp(f32& v0, f32& v1, f32& v2, f32& v3, int lane) {
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

} // namespace flint::cuda
