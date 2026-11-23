#pragma once

#include "common.hpp"

#if defined(FLINT_CUDA) && FLINT_CUDA == 1

namespace flint::cuda {

void launch_embed_lookup_bf16(const __nv_bfloat16* embed_table,
                              const i32* d_token_ids,
                              __nv_bfloat16* out,
                              u32 M, u32 hidden_size,
                              cudaStream_t stream);

void launch_embed_lookup_fp8(const u8* embed_fp8, const f32* embed_scale,
                             const i32* d_token_ids, __nv_bfloat16* out,
                             u32 M, u32 hidden_size, cudaStream_t stream);

// W4A16 group-128 embed lookup (-651 MiB VRAM vs BF16 for true desktop-safe 16 GB Inkling-2-Mini)
void launch_embed_lookup_w4a16(const u32* embed_qweight, const u32* embed_scales,
                               const i32* d_token_ids, __nv_bfloat16* out,
                               u32 M, u32 hidden_size, cudaStream_t stream);

void launch_quant_bf16_to_fp8_rowwise(const __nv_bfloat16* src,
                                      u8* dst_fp8, f32* dst_scales,
                                      u32 rows, u32 cols, cudaStream_t stream);

void launch_rmsnorm_bf16(const __nv_bfloat16* x,
                         const __nv_bfloat16* w,
                         __nv_bfloat16* out,
                         u32 M, u32 hidden_size, f32 eps,
                         cudaStream_t stream);

void launch_fused_residual_rmsnorm_bf16(__nv_bfloat16* residual,
                                        const __nv_bfloat16* delta,
                                        const __nv_bfloat16* w,
                                        __nv_bfloat16* norm_out,
                                        u32 M, u32 hidden_size, f32 eps,
                                        cudaStream_t stream);

void launch_add_residual_bf16(__nv_bfloat16* residual,
                              const __nv_bfloat16* delta,
                              u32 count,
                              cudaStream_t stream);

// Fused RoPE (+ optional Qwen3 QK-norm) + Paged BF16 KV-Cache Write
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
                                     cudaStream_t stream);

// Fused RoPE (+ optional Qwen3 QK-norm) + Paged Per-Token/Per-Head Dynamic-Scaled FP8 E4M3 KV-Cache Write
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
                                    cudaStream_t stream);

// Fused RoPE (+ optional Qwen3 QK-norm) + Zero-Calibration Walsh-Hadamard 4-Bit KV-Cache Write (4x smaller)
void launch_fused_rope_kv_write_wht4(__nv_bfloat16* qkv,
                                     __nv_bfloat16* q_out,
                                     u8* k_cache_i4,
                                     u8* v_cache_i4,
                                     __nv_bfloat162* k_scales_sm, // (scale, min_val)
                                     __nv_bfloat162* v_scales_sm,
                                     const i32* d_block_table,
                                     const f32* d_inv_freq,
                                     const __nv_bfloat16* q_norm_w,
                                     const __nv_bfloat16* k_norm_w,
                                     const i32* d_pos,
                                     u32 start_pos,
                                     u32 M, u32 n_q_heads, u32 n_kv_heads, u32 head_dim,
                                     u32 block_size, f32 rms_eps, bool use_wht,
                                     cudaStream_t stream);

// Roofline BF16 GEMV for M=1..8 (optional fused += residual epilogue when add_to_out=true)
void launch_gemv_bf16(const __nv_bfloat16* x,
                      const __nv_bfloat16* W,
                      const __nv_bfloat16* bias,
                      __nv_bfloat16* out,
                      u32 M, u32 N, u32 K,
                      cudaStream_t stream,
                      bool add_to_out = false);

void launch_fused_gate_up_silu_bf16(const __nv_bfloat16* x,
                                    const __nv_bfloat16* W_gate_up,
                                    __nv_bfloat16* out,
                                    u32 M, u32 inter, u32 K,
                                    cudaStream_t stream);

// Roofline W4A16 (Asymmetric INT4 group-128, PTX lop3.b32 FP16-magic dequant) GEMV for M=1..8
void launch_gemv_w4a16(const __nv_bfloat16* x,
                       const u32* qweight,
                       const __nv_bfloat162* scales,
                       const __nv_bfloat16* bias,
                       __nv_bfloat16* out,
                       u32 M, u32 N, u32 K,
                       cudaStream_t stream,
                       bool add_to_out = false);

void launch_fused_gate_up_silu_w4a16(const __nv_bfloat16* x,
                                     const u32* qweight_gu,
                                     const __nv_bfloat162* scales_gu,
                                     __nv_bfloat16* out,
                                     u32 M, u32 inter, u32 K,
                                     cudaStream_t stream);

void launch_dequant_w4a16_to_bf16(const u32* qweight,
                                  const __nv_bfloat162* scales,
                                  __nv_bfloat16* out_bf16,
                                  u32 N, u32 K,
                                  cudaStream_t stream);

// Fused W4A16 -> per-row FP8 E4M3 dequantization for native 330+ TFLOPS cublasLtMatmul prefill
void launch_dequant_w4a16_to_fp8(const u32* qweight,
                                 const __nv_bfloat162* scales,
                                 u8* out_fp8,
                                 f32* out_row_scales,
                                 u32 N, u32 K,
                                 cudaStream_t stream);

// Roofline FP8 E4M3 GEMV for M=1..8
void launch_gemv_fp8(const __nv_bfloat16* x,
                     const u8* qweight,
                     const f32* scales,
                     const __nv_bfloat16* bias,
                     __nv_bfloat16* out,
                     u32 M, u32 N, u32 K,
                     cudaStream_t stream,
                     bool add_to_out = false);

void launch_fused_gate_up_silu_fp8(const __nv_bfloat16* x,
                                   const u8* qweight_gu,
                                   const f32* scales_gu,
                                   __nv_bfloat16* out,
                                   u32 M, u32 inter, u32 K,
                                   cudaStream_t stream);

void launch_dequant_fp8_to_bf16(const u8* qweight,
                                const f32* scales,
                                __nv_bfloat16* out_bf16,
                                u32 N, u32 K,
                                cudaStream_t stream);

// Native FP8 Tensor-Core Prefill helpers (quantize BF16 activations -> FP8 E4M3 per-tensor + apply per-row weight scales)
void launch_quant_bf16_to_fp8_pertensor(const __nv_bfloat16* x,
                                        u8* x_fp8,
                                        f32* d_amax_and_scale, // [0]=amax scratch, [1]=scale=amax/448
                                        u32 M, u32 M_pad, u32 K,
                                        cudaStream_t stream);

void launch_apply_row_scales_bf16(__nv_bfloat16* out,
                                  const f32* row_scales,
                                  const __nv_bfloat16* bias,
                                  u32 M, u32 N,
                                  cudaStream_t stream);

void launch_silu_mul_prefill_bf16(const __nv_bfloat16* gate_up,
                                  __nv_bfloat16* out,
                                  u32 M, u32 inter,
                                  cudaStream_t stream);

void launch_add_bias_bf16(__nv_bfloat16* out,
                          const __nv_bfloat16* bias,
                          u32 M, u32 N,
                          cudaStream_t stream);

// GQA-Grouped Split-K FlashDecoding (BF16 KV cache)
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
                              cudaStream_t stream);

// GQA-Grouped Split-K FlashDecoding (Per-Token/Per-Head Dynamic-Scaled FP8 E4M3 KV cache)
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
                               cudaStream_t stream);

// GQA-Grouped Split-K FlashDecoding (Zero-Calibration Walsh-Hadamard 4-Bit KV cache, 4x smaller than BF16)
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
                              cudaStream_t stream);

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
                               cudaStream_t stream);

// Union-smem 2x-Occupancy WMMA Tensor-Core FlashAttention-2 Prefill (BF16 or Scaled FP8 KV)
void launch_flash_prefill_bf16(const __nv_bfloat16* q,
                               const __nv_bfloat16* k_cache,
                               const __nv_bfloat16* v_cache,
                               const i32* d_block_table,
                               __nv_bfloat16* attn_out,
                               u32 start_pos, u32 M,
                               u32 n_q_heads, u32 n_kv_heads, u32 head_dim,
                               u32 block_size,
                               cudaStream_t stream);

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
                                cudaStream_t stream);

// GPU Argmax (plus optional softmax top-1 confidence for confidence-gated speculative drafting)
void launch_argmax_bf16(const __nv_bfloat16* logits,
                        i32* d_out_tokens,
                        u32 M, u32 vocab_size,
                        cudaStream_t stream);

void launch_argmax_conf_bf16(const __nv_bfloat16* logits,
                             i32* d_out_tokens,
                             f32* d_out_conf,
                             u32 M, u32 vocab_size,
                             cudaStream_t stream);

void launch_advance_decode_state(i32* d_pos, i32* d_cur_token, const i32* d_next_token,
                                 cudaStream_t stream);

// ============================================================================
// Inkling-2-Mini (Looped MoE + NVFP4 + Rel-Bias Hybrid Attn + Block AttnRes)
// ============================================================================
void launch_inkling_moe_router(const __nv_bfloat16* x,
                               const __nv_bfloat16* gate_w,
                               const f32* gate_bias,
                               i32* out_ids,
                               f32* out_weights,
                               u32 M, u32 H, u32 E, u32 topk, f32 route_scale,
                               cudaStream_t stream);

// Fast parallel 32-SM router stage-2: takes precomputed router logits [M, 128] BF16 -> top-8 ids & weights
void launch_inkling_moe_topk_from_logits(const __nv_bfloat16* router_logits,
                                         const f32* gate_bias,
                                         i32* out_ids,
                                         f32* out_weights,
                                         u32 M, u32 E, u32 topk, f32 route_scale,
                                         cudaStream_t stream);

// Expert-Grouped Batched Tensor-Core MoE Prefill helpers (NVFP4 & MXFP8 single-expert dequant + SiTU + gather/scatter)
void launch_dequant_nvfp4_expert(const u8* w_e, const u8* s_e, const f32* amax_e, const f32* denom,
                                 __nv_bfloat16* out_bf16, u32 rows, u32 cols, cudaStream_t stream);
void launch_dequant_nvfp4_expert_to_fp8(const u8* w_e, const u8* s_e, const f32* amax_e, const f32* denom,
                                        u8* out_fp8, f32* out_row_scales, u32 rows, u32 cols, cudaStream_t stream);

// Blackwell sm_120a Hardware PTX mxf4nvf4.m16n8k64 FP4 Tensor-Core Zero-Host-Sync Grouped MoE
void launch_inkling_moe_grouped_nvfp4_fp4tc(
    const __nv_bfloat16* x_in,                 // [M, H]
    const u8* w13_weight, const u8* w13_scale, const f32* w13_amax, const f32* w13_denom,
    const u8* w2_weight,  const u8* w2_scale,  const f32* w2_amax,  const f32* w2_denom,
    const i32* expert_ids, const f32* expert_weights,
    // Persistent GPU worklists & NVFP4 activation scratch (zero host sync)
    i32* d_pack_idx, f32* d_pack_gam,
    u32* d_tile_meta, u32* d_tile_cnt, u32* d_num_tiles,
    u8* d_x_fp4, u8* d_x_fp4_sc, f32* d_x_fp4_gs,
    __nv_bfloat16* d_act_bf16, u8* d_act_fp4, u8* d_act_fp4_sc, f32* d_act_fp4_gs,
    f32* d_moe_out_f32, __nv_bfloat16* d_moe_out,
    u32 M, u32 H, u32 inter, u32 E, u32 topk,
    f32 gate_cap, f32 up_cap, cudaStream_t stream);
void launch_dequant_mxfp8_expert(const u8* w_e, const u8* s_e,
                                 __nv_bfloat16* out_bf16, u32 rows, u32 cols, cudaStream_t stream);
void launch_inkling_moe_gather(const __nv_bfloat16* x, const i32* tok_idx, __nv_bfloat16* out,
                               u32 cnt, u32 H, cudaStream_t stream);
void launch_inkling_moe_situ_batched(const __nv_bfloat16* w13_out, const f32* gammas,
                                     __nv_bfloat16* act_out, u32 cnt, u32 inter,
                                     f32 gate_cap, f32 up_cap, cudaStream_t stream);
void launch_inkling_moe_scatter_add(const __nv_bfloat16* y, const i32* tok_idx,
                                    f32* moe_out_f32, u32 cnt, u32 H, cudaStream_t stream);
void launch_f32_to_bf16(const f32* src, __nv_bfloat16* dst, u32 total, cudaStream_t stream);

void launch_inkling_moe_nvfp4_gate_up_situ(const __nv_bfloat16* x,
                                           const u8* w13_weight,
                                           const u8* w13_scale,
                                           const f32* w13_amax,
                                           const f32* w13_outer_denom,
                                           const i32* expert_ids,
                                           const f32* expert_weights,
                                           __nv_bfloat16* act_out,
                                           u32 M, u32 inter, u32 K, u32 topk,
                                           f32 gate_cap, f32 up_cap,
                                           cudaStream_t stream);

void launch_inkling_moe_nvfp4_down_combine(const __nv_bfloat16* act_in,
                                           const u8* w2_weight,
                                           const u8* w2_scale,
                                           const f32* w2_amax,
                                           const f32* w2_outer_denom,
                                           const i32* expert_ids,
                                           __nv_bfloat16* out,
                                           u32 M, u32 H, u32 inter, u32 topk,
                                           cudaStream_t stream);

void launch_inkling_moe_mxfp8_gate_up_situ(const __nv_bfloat16* x,
                                           const u8* w13_weight,
                                           const u8* w13_scale_e8m0,
                                           const i32* expert_ids,
                                           const f32* expert_weights,
                                           __nv_bfloat16* act_out,
                                           u32 M, u32 inter, u32 K, u32 topk,
                                           f32 gate_cap, f32 up_cap,
                                           cudaStream_t stream);

void launch_inkling_moe_mxfp8_down_combine(const __nv_bfloat16* act_in,
                                           const u8* w2_weight,
                                           const u8* w2_scale_e8m0,
                                           const i32* expert_ids,
                                           __nv_bfloat16* out,
                                           u32 M, u32 H, u32 inter, u32 topk,
                                           cudaStream_t stream);

void launch_inkling_attn_res(const __nv_bfloat16* const* d_stack_ptrs,
                             u32 n_stack,
                             const __nv_bfloat16* last_ptr,
                             const __nv_bfloat16* addend_ptr,
                             const f32* queries,
                             const __nv_bfloat16* norm_w,
                             __nv_bfloat16* out_bf16,
                             f32* out_f32,
                             u32 num_q, u32 M, u32 D, f32 eps, bool normalize_values,
                             cudaStream_t stream);

void launch_inkling_pair_rmsnorm(const f32* dest, const f32* msg,
                                 __nv_bfloat16* out,
                                 bool broadcast_dest, u32 S, u32 M, u32 D, f32 eps,
                                 cudaStream_t stream);

void launch_inkling_blend_rmsnorm(const f32* dest, const f32* msg,
                                  const __nv_bfloat16* gain_bf16,
                                  const __nv_bfloat16* forget_bf16,
                                  const __nv_bfloat16* norm_w,
                                  __nv_bfloat16* out,
                                  bool broadcast_dest, bool broadcast_norm,
                                  u32 S, u32 M, u32 D, f32 eps,
                                  cudaStream_t stream);

void launch_inkling_sconv_hidden(const __nv_bfloat16* x, const __nv_bfloat16* w,
                                 __nv_bfloat16* hist, __nv_bfloat16* out,
                                 const __nv_bfloat16* res1, const __nv_bfloat16* res2,
                                 const i32* d_pos, u32 start_pos, u32 M, u32 D,
                                 cudaStream_t stream);

void launch_inkling_qr_clip(const __nv_bfloat16* q_in, const __nv_bfloat16* r_in,
                            const __nv_bfloat16* q_sc, const __nv_bfloat16* r_sc,
                            __nv_bfloat16* q_out, __nv_bfloat16* r_out,
                            const i32* d_pos, u32 start_pos, u32 M, u32 n_q_heads,
                            f32 q_clip, f32 r_clip,
                            bool use_log_scale, u32 n_floor, f32 alpha,
                            cudaStream_t stream);

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
                                   cudaStream_t stream);

void launch_inkling_rel_attention(const __nv_bfloat16* q, const __nv_bfloat16* r,
                                  const __nv_bfloat16* gate, const __nv_bfloat16* sink,
                                  const __nv_bfloat16* rel_proj_T,
                                  const void* k_cache_raw, const void* v_cache_raw,
                                  const f32* k_scales, const f32* v_scales,
                                  const i32* d_block_table,
                                  __nv_bfloat16* attn_out,
                                  const i32* d_pos, u32 start_pos, u32 M,
                                  u32 n_q_heads, u32 n_kv_heads, u32 block_size,
                                  bool is_local_ring, u32 swa_window, bool int8_kv,
                                  cudaStream_t stream);

// GQA-Grouped Split-K Relative-Bias Attention (saturates all 128 SMs + 3x KV reuse -> 2.5-3x faster at 4k-16k)
void launch_inkling_rel_attention_splitk(const __nv_bfloat16* q, const __nv_bfloat16* r,
                                         const __nv_bfloat16* gate, const __nv_bfloat16* sink,
                                         const __nv_bfloat16* rel_proj_T,
                                         const void* k_cache_raw, const void* v_cache_raw,
                                         const f32* k_scales, const f32* v_scales,
                                         const i32* d_block_table,
                                         __nv_bfloat16* attn_out,
                                         f32* d_split_o, f32* d_split_ml,
                                         const i32* d_pos, u32 start_pos, u32 M,
                                         u32 n_q_heads, u32 n_kv_heads, u32 block_size,
                                         u32 num_splits, bool is_local_ring, u32 swa_window, bool int8_kv,
                                         cudaStream_t stream);

// Zero-extra-VRAM multi-matrix fused BF16 GEMV (fuses 6 local attn GEMVs or 4 global Q/R/Gate/Sink GEMVs into 1 launch)
void launch_inkling_fused_attn_gemv(const __nv_bfloat16* x,
                                    const __nv_bfloat16* wq, const __nv_bfloat16* wr,
                                    const __nv_bfloat16* wg, const __nv_bfloat16* ws,
                                    const __nv_bfloat16* wk, const __nv_bfloat16* wv,
                                    __nv_bfloat16* oq, __nv_bfloat16* or_out,
                                    __nv_bfloat16* og, __nv_bfloat16* os,
                                    __nv_bfloat16* ok, __nv_bfloat16* ov,
                                    u32 M, u32 Qdim, u32 Rdim, u32 Hq, u32 KVdim, u32 K,
                                    bool include_kv, cudaStream_t stream);

// Zero-extra-VRAM multi-matrix fused FP8 Attention GEMV (fuses 5 FP8 projections + BF16 sink into 1 launch)
// 1-launch 3-stream FP8 Attention GEMV (WQ+WR+WGate+WSink on x_q, WK+WK_dyn on x_k, WV+WV_dyn on x_v -> -80 launches/tok)
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
                                        cudaStream_t stream);

// Fused M=1..8 FP8 wo_proj + attn_sconv_hidden (eliminates 48 launches & 48 round-trips on BOTH base & MTP verify)
void launch_inkling_wo_sconv_fp8(const __nv_bfloat16* attn_out,
                                 const u8* wo_fp8, const f32* so_fp8,
                                 const __nv_bfloat16* sconv_w,
                                 __nv_bfloat16* hist_ring,
                                 __nv_bfloat16* out,
                                 const i32* d_pos, u32 start_pos,
                                 u32 M, u32 H, u32 Qdim, cudaStream_t stream);

// 16x16x16 WMMA Tensor-Core Relative-Bias Prefill Attention (3-4x faster pp1024/pp4096/pp16k)
void launch_inkling_rel_attention_prefill_wmma(
    const __nv_bfloat16* q, const __nv_bfloat16* r,
    const __nv_bfloat16* gate, const __nv_bfloat16* sink,
    const __nv_bfloat16* rel_proj_T,
    const void* k_cache_raw, const void* v_cache_raw,
    const f32* k_scales, const f32* v_scales,
    const i32* d_block_table, __nv_bfloat16* attn_out,
    u32 start_pos, u32 M, u32 n_q_heads, u32 n_kv_heads, u32 block_size,
    bool is_local_ring, u32 swa_window, bool int8_kv, cudaStream_t stream);

// 1M-Context Global KV WHT-K4V4 (8.25 GiB @ 1M -> 24 GB RTX 4090) & WHT-K8V4 (12.25 GiB @ 1M -> 32 GB RTX 5090)
void launch_inkling_kv_sconv_write_wht(
    const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
    const __nv_bfloat16* wk, const __nv_bfloat16* wv,
    const __nv_bfloat16* dk_dyn, const __nv_bfloat16* dv_dyn,
    __nv_bfloat16* k_hist, __nv_bfloat16* v_hist,
    u8* k_cache, u8* v_cache,
    __nv_bfloat16* k_scales_bf16, __nv_bfloat16* v_scales_bf16,
    const i32* d_block_table, const i32* d_pos,
    u32 start_pos, u32 M, u32 n_kv_heads, u32 block_size,
    bool k_is_4bit, f32 k_clip, f32 rms_eps, cudaStream_t stream);

void launch_inkling_global_attn_splitk_wht(
    const __nv_bfloat16* q, const __nv_bfloat16* r,
    const __nv_bfloat16* gate, const __nv_bfloat16* sink,
    const __nv_bfloat16* rel_proj_T,
    const u8* k_cache, const u8* v_cache,
    const __nv_bfloat16* k_scales_bf16, const __nv_bfloat16* v_scales_bf16,
    const i32* d_block_table, __nv_bfloat16* attn_out,
    f32* d_split_o, f32* d_split_ml,
    const i32* d_pos, u32 start_pos, u32 M,
    u32 n_q_heads, u32 n_kv_heads, u32 block_size,
    u32 num_splits, bool k_is_4bit, cudaStream_t stream);

void launch_inkling_global_attn_prefill_wmma_wht(
    const __nv_bfloat16* q, const __nv_bfloat16* r,
    const __nv_bfloat16* gate, const __nv_bfloat16* sink,
    const __nv_bfloat16* rel_proj_T,
    const u8* k_cache, const u8* v_cache,
    const __nv_bfloat16* k_scales_bf16, const __nv_bfloat16* v_scales_bf16,
    const i32* d_block_table, __nv_bfloat16* attn_out,
    u32 start_pos, u32 M, u32 n_q_heads, u32 n_kv_heads, u32 block_size,
    bool k_is_4bit, cudaStream_t stream);

// Fused M=1..8 NVFP4 down_combine + mlp_sconv_hidden + residual add (-48 launches & -48 round-trips on BOTH base & MTP verify)
void launch_inkling_moe_nvfp4_down_sconv(const __nv_bfloat16* act_in,
                                         const u8* w2_weight,
                                         const u8* w2_scale,
                                         const f32* w2_amax,
                                         const f32* w2_outer_denom,
                                         const i32* expert_ids,
                                         const __nv_bfloat16* sconv_w,
                                         __nv_bfloat16* hist_ring,
                                         __nv_bfloat16* out_dst,
                                         const __nv_bfloat16* add_a,
                                         const __nv_bfloat16* add_b,
                                         const i32* d_pos, u32 start_pos,
                                         u32 M, u32 H, u32 inter, u32 topk,
                                         cudaStream_t stream);

// Fused 3-stream Looped-Core Controller GEMVs (cuts 24 x 4 = 96 kernel launches/tok on looped core, 0 extra VRAM)
void launch_inkling_fused_3ctrl_down(const __nv_bfloat16* pair_3x2H,
                                     const __nv_bfloat16* wd0, const __nv_bfloat16* wd1, const __nv_bfloat16* wd2,
                                     __nv_bfloat16* out_3xR, u32 M, u32 rank, u32 K2H, cudaStream_t stream);

void launch_inkling_fused_3ctrl_gf(const __nv_bfloat16* h_3xR,
                                   const __nv_bfloat16* wg0, const __nv_bfloat16* wf0,
                                   const __nv_bfloat16* wg1, const __nv_bfloat16* wf1,
                                   const __nv_bfloat16* wg2, const __nv_bfloat16* wf2,
                                   __nv_bfloat16* out_3x2H, u32 M, u32 H, u32 rank, cudaStream_t stream);

// 1-launch 3-stream blend_rmsnorm (replaces 3 sequential launches per looped core visit -> -48 launches/tok)
void launch_inkling_blend_rmsnorm_3stream(const f32* dest, const f32* msg,
                                          const __nv_bfloat16* gf_3x2H,
                                          const __nv_bfloat16* norm_w,
                                          __nv_bfloat16* out,
                                          bool is_local, u32 M, u32 D, f32 eps,
                                          cudaStream_t stream);

// Compact-list active-expert NVFP4 Gate+Up+SiTU for M=2..8 (1-block compact list -> ZERO wasted blocks, ZERO atomicAdd)
void launch_inkling_moe_compact_nvfp4_w13(
    const __nv_bfloat16* x, const u8* w13_weight, const u8* w13_scale,
    const f32* w13_amax, const f32* w13_outer_denom,
    const i32* expert_ids, const f32* expert_weights, __nv_bfloat16* act_out,
    u32* d_uniq_e, u64* d_uniq_mask, u32* d_uniq_count,
    u32 M, u32 inter, u32 H, u32 topk, f32 gate_cap, f32 up_cap, cudaStream_t stream);

// 1-launch concurrent qr_clip + kv_sconv_write (40 blocks = 24 Q/R + 16 K/V -> -48 launches/tok)
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
    cudaStream_t stream);

// Fused 2-output BF16 GEMV (fuses controller gain+forget or global K+V into 1 launch)
void launch_inkling_fused_2gemv(const __nv_bfloat16* x,
                                const __nv_bfloat16* w0, const __nv_bfloat16* w1,
                                __nv_bfloat16* o0, __nv_bfloat16* o1,
                                u32 M, u32 N0, u32 N1, u32 K,
                                cudaStream_t stream);

void launch_inkling_logit_softcap(__nv_bfloat16* logits, u32 M,
                                  u32 vocab_padded, u32 vocab_unpadded, f32 cap,
                                  cudaStream_t stream);

// Built-in Inkling-2-Mini MTP (DSpark 3-layer + Rank-256 Markov Head)
void launch_inkling_mtp_mix_rmsnorm(const __nv_bfloat16* const* d_feat_ptrs,
                                    const f32* d_alpha,
                                    const __nv_bfloat16* norm_w,
                                    __nv_bfloat16* out,
                                    u32 M, u32 D, f32 eps, cudaStream_t stream);

void launch_inkling_mtp_norm_rope(__nv_bfloat16* x, const __nv_bfloat16* w,
                                  const f32* inv_freq, u32 start_pos, u32 M, u32 H, f32 eps,
                                  cudaStream_t stream);

void launch_inkling_mtp_block_attn(const __nv_bfloat16* q_blk,
                                   const __nv_bfloat16* k_ctx, const __nv_bfloat16* v_ctx,
                                   const __nv_bfloat16* k_blk, const __nv_bfloat16* v_blk,
                                   __nv_bfloat16* out,
                                   u32 ctx_start, u32 ctx_len, u32 B, u32 n_q_heads, u32 n_kv_heads,
                                   cudaStream_t stream);

void launch_inkling_mtp_markov_gate(const __nv_bfloat16* raw_3r, __nv_bfloat16* state,
                                    __nv_bfloat16* latent, u32 rank, cudaStream_t stream);

void launch_inkling_mtp_add_softcap(const __nv_bfloat16* base_logits, const __nv_bfloat16* bias,
                                    __nv_bfloat16* out, u32 vocab_padded, u32 vocab_unpadded,
                                    f32 cap, cudaStream_t stream);

void launch_inkling_mtp_silu_mul(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                 __nv_bfloat16* out, u32 total, cudaStream_t stream);

// Zero-host-sync GPU-resident RNN input packer: packs [state(256), markov_w1[*d_prev_tok](256), h_step(2304), f_rnn(2304)]
void launch_inkling_mtp_pack_rnn_in(const __nv_bfloat16* state,
                                    const __nv_bfloat16* markov_w1,
                                    const i32* d_prev_tok,
                                    const __nv_bfloat16* h_step,
                                    const __nv_bfloat16* f_rnn,
                                    __nv_bfloat16* rnn_in,
                                    u32 rank, u32 H, cudaStream_t stream);

// GPU-position-driven MTP helpers (supports BF16, FP8, and W4A16 embed tables)
void launch_inkling_mtp_build_block(const __nv_bfloat16* embed, const __nv_bfloat16* embed_norm,
                                    const __nv_bfloat16* mask_embed, const i32* d_anchor_tok,
                                    __nv_bfloat16* x_blk, u32 B, u32 H, f32 eps, cudaStream_t stream);

void launch_inkling_mtp_build_block_fp8(const u8* embed_fp8, const f32* embed_scale,
                                        const __nv_bfloat16* embed_norm,
                                        const __nv_bfloat16* mask_embed, const i32* d_anchor_tok,
                                        __nv_bfloat16* x_blk, u32 B, u32 H, f32 eps, cudaStream_t stream);

void launch_inkling_mtp_build_block_w4a16(const u32* embed_qw, const u32* embed_sc,
                                          const __nv_bfloat16* embed_norm,
                                          const __nv_bfloat16* mask_embed, const i32* d_anchor_tok,
                                          __nv_bfloat16* x_blk, u32 B, u32 H, f32 eps, cudaStream_t stream);

void launch_inkling_mtp_norm_rope_dpos(__nv_bfloat16* x, const __nv_bfloat16* w,
                                       const f32* inv_freq, const i32* d_pos, u32 M, u32 H, f32 eps,
                                       cudaStream_t stream);

void launch_inkling_mtp_block_attn_dpos(const __nv_bfloat16* q_blk,
                                        const __nv_bfloat16* k_ctx, const __nv_bfloat16* v_ctx,
                                        const __nv_bfloat16* k_blk, const __nv_bfloat16* v_blk,
                                        __nv_bfloat16* out,
                                        const i32* d_ctx_len, bool is_local, u32 swa_window,
                                        u32 B, u32 n_q_heads, u32 n_kv_heads, cudaStream_t stream);

// Precompute [n_steps, 3*rank] F32 Markov context ONCE outside the 7-step loop (matches dspark.py:363)
void launch_inkling_mtp_rnn_ctx(const __nv_bfloat16* h_steps, const __nv_bfloat16* f_rnn,
                                const __nv_bfloat16* rnn_w, const __nv_bfloat16* rnn_b,
                                f32* out_ctx_f32, u32 n_steps, u32 rank, u32 H, cudaStream_t stream);

// Fused 1-kernel Markov RNN step + exact BF16-rounded gate (multiplies ONLY 512 cols instead of 5120 -> 10x less work)
void launch_inkling_mtp_rnn_step_gate(const f32* ctx_s_f32, const __nv_bfloat16* rnn_w,
                                      const __nv_bfloat16* markov_w1, const i32* d_prev_tok,
                                      __nv_bfloat16* state, __nv_bfloat16* latent,
                                      u32 rank, u32 H, cudaStream_t stream);

// Single-Launch Persistent Cooperative Megakernel (1 launch = all 28 layers + LM head + Argmax)
struct MegakernelLayerDesc {
    const __nv_bfloat16* input_norm;
    const __nv_bfloat16* post_attn_norm;
    const __nv_bfloat16* qkv_bias;
    const u32* w_qkv;
    const __nv_bfloat162* s_qkv;
    const u32* w_o;
    const __nv_bfloat162* s_o;
    const u32* w_gu;
    const __nv_bfloat162* s_gu;
    const u32* w_down;
    const __nv_bfloat162* s_down;
    void* k_cache;
    void* v_cache;
    f32* k_scale;
    f32* v_scale;
};

bool launch_megakernel_w4a16_decode(const MegakernelLayerDesc* d_layers,
                                    const __nv_bfloat16* d_embed,
                                    const __nv_bfloat16* d_final_norm,
                                    const u8* d_lm_head_fp8,
                                    const f32* d_lm_head_scales,
                                    const f32* d_inv_freq,
                                    const i32* d_block_table,
                                    __nv_bfloat16* d_x,
                                    __nv_bfloat16* d_norm,
                                    __nv_bfloat16* d_qkv,
                                    __nv_bfloat16* d_q,
                                    __nv_bfloat16* d_attn_out,
                                    __nv_bfloat16* d_mlp_act,
                                    f32* d_split_o,
                                    f32* d_split_ml,
                                    f32* d_argmax_val,
                                    i32* d_argmax_idx,
                                    i32* d_pos,
                                    i32* d_token_ids,
                                    i32* d_out_tokens,
                                    u32 num_layers, u32 H, u32 inter, u32 vocab_size,
                                    u32 n_q_heads, u32 n_kv_heads, u32 block_size, f32 rms_eps,
                                    bool int8_kv, cudaStream_t stream);

} // namespace flint::cuda

#endif // FLINT_CUDA
