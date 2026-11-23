#include "inkling2_engine.hpp"
#include "engine.hpp"
#include "kernels/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace flint {

#if defined(FLINT_CUDA) && FLINT_CUDA == 1
#define INK_CUDA_CHECK(call)                                                        \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error %s:%d: %s (%d)\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__), static_cast<int>(err__));       \
            std::abort();                                                           \
        }                                                                           \
    } while (0)

void Inkling2Engine::destroy_decode_graph() {
    if (decode_graph_exec_) { cudaGraphExecDestroy(decode_graph_exec_); decode_graph_exec_ = nullptr; }
    if (decode_graph_) { cudaGraphDestroy(decode_graph_); decode_graph_ = nullptr; }
    graph_captured_ = false;
}

void Inkling2Engine::destroy_mtp_graphs() {
    if (mtp_draft_graph_exec_)  { cudaGraphExecDestroy(mtp_draft_graph_exec_);  mtp_draft_graph_exec_ = nullptr; }
    if (mtp_draft_graph_)       { cudaGraphDestroy(mtp_draft_graph_);           mtp_draft_graph_ = nullptr; }
    if (mtp_verify_graph_exec_) { cudaGraphExecDestroy(mtp_verify_graph_exec_); mtp_verify_graph_exec_ = nullptr; }
    if (mtp_verify_graph_)      { cudaGraphDestroy(mtp_verify_graph_);          mtp_verify_graph_ = nullptr; }
    mtp_graphs_captured_ = false;
}

void Inkling2Engine::run_bf16_linear(const __nv_bfloat16* x, const __nv_bfloat16* w,
                                     __nv_bfloat16* out, u32 M, u32 N, u32 K, bool add_to_out) {
    if (M <= 8 && (K % 8 == 0)) {
        cuda::launch_gemv_bf16(x, w, nullptr, out, M, N, K, stream_, add_to_out);
        return;
    }
    f32 alpha = 1.0f, beta = add_to_out ? 1.0f : 0.0f;
    cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                 static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                 &alpha, w, CUDA_R_16BF, static_cast<int>(K),
                 x, CUDA_R_16BF, static_cast<int>(K),
                 &beta, out, CUDA_R_16BF, static_cast<int>(N),
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

bool Inkling2Engine::run_fp8_cublaslt_gemm(const __nv_bfloat16* x, const u8* w_fp8, const f32* row_scales,
                                           __nv_bfloat16* out, u32 M, u32 N, u32 K) {
    if (!cublaslt_ || M < 16 || (K & 15) != 0 || (N & 15) != 0 || N > std::max<u32>({cfg_.hidden_size, cfg_.q_dim(), 2 * cfg_.intermediate_size})) {
        return false;
    }
    const u32 M_pad = (M + 15u) & ~15u;
    if (M_pad > max_batch_tokens_ + 16) return false;

    cuda::launch_quant_bf16_to_fp8_pertensor(x, d_fp8_x_, d_fp8_scale_, M, M_pad, K, stream_);

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    bool ok = false;
    do {
        if (cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) break;
        cublasOperation_t trans_a = CUBLAS_OP_T;
        cublasOperation_t trans_b = CUBLAS_OP_N;
        int8_t fast_acc = 1;
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_acc, sizeof(fast_acc));
        const f32* a_scale_ptr = d_fp8_scale_ + 2; // 1.0f
        const f32* b_scale_ptr = d_fp8_scale_ + 1; // dynamic x scale
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr));
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr));

        if (cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, K, N, K) != CUBLAS_STATUS_SUCCESS) break;
        if (cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, K, M_pad, K) != CUBLAS_STATUS_SUCCESS) break;
        if (cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, N, M_pad, N) != CUBLAS_STATUS_SUCCESS) break;

        // O(1) cublasLtMatmulAlgo_t cache by (M_pad, N, K) -> kills 12,288 CPU heuristic calls/chunk
        u64 algo_key = (static_cast<u64>(M_pad) << 36) | (static_cast<u64>(N) << 18) | static_cast<u64>(K);
        cublasLtMatmulAlgo_t algo{};
        auto it = lt_algo_cache_.find(algo_key);
        if (it != lt_algo_cache_.end()) {
            algo = it->second;
        } else {
            if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) break;
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &cublaslt_ws_bytes_, sizeof(cublaslt_ws_bytes_));
            cublasLtMatmulHeuristicResult_t heur{};
            int returned = 0;
            if (cublasLtMatmulAlgoGetHeuristic(cublaslt_, op_desc, a_desc, b_desc, c_desc, c_desc,
                                               pref, 1, &heur, &returned) != CUBLAS_STATUS_SUCCESS || returned == 0) {
                break;
            }
            algo = heur.algo;
            lt_algo_cache_[algo_key] = algo;
        }
        __nv_bfloat16* dst_buf = (M_pad == M) ? out : d_fp8_out_pad_;
        f32 alpha = 1.0f, beta = 0.0f;
        if (cublasLtMatmul(cublaslt_, op_desc, &alpha,
                           w_fp8, a_desc, d_fp8_x_, b_desc,
                           &beta, dst_buf, c_desc, dst_buf, c_desc,
                           &algo, d_cublaslt_ws_, cublaslt_ws_bytes_, stream_) != CUBLAS_STATUS_SUCCESS) {
            break;
        }
        if (M_pad != M) {
            INK_CUDA_CHECK(cudaMemcpyAsync(out, dst_buf, static_cast<size_t>(M) * N * sizeof(__nv_bfloat16),
                                           cudaMemcpyDeviceToDevice, stream_));
        }
        cuda::launch_apply_row_scales_bf16(out, row_scales, nullptr, M, N, stream_);
        ok = true;
    } while (false);

    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
    if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
    if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
    if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    return ok;
}

void Inkling2Engine::run_fp8_linear(const __nv_bfloat16* x, const u8* w_fp8, const f32* s_fp8,
                                    __nv_bfloat16* out, u32 M, u32 N, u32 K, bool add_to_out) {
    if (M <= 8 && (K % 16 == 0)) {
        cuda::launch_gemv_fp8(x, w_fp8, s_fp8, nullptr, out, M, N, K, stream_, add_to_out);
        return;
    }
    if (!add_to_out && run_fp8_cublaslt_gemm(x, w_fp8, s_fp8, out, M, N, K)) {
        return;
    }
    cuda::launch_dequant_fp8_to_bf16(w_fp8, s_fp8, d_moe_dequant_, N, K, stream_);
    run_bf16_linear(x, d_moe_dequant_, out, M, N, K, add_to_out);
}

void Inkling2Engine::mtp_store_context(u32 M, u32 start_pos) {
    if (!mtp_loaded_ || M == 0 || start_pos + M > mtp_max_seq_) return;
    const u32 H = cfg_.hidden_size;
    const u32 KVdim = cfg_.kv_dim();
    for (int l = 0; l < 3; ++l) {
        auto& ml = mtp_layers_[l];
        cuda::launch_inkling_mtp_mix_rmsnorm(d_mtp_feat_ptrs_, d_mtp_alpha_ + l * 4,
                                             ml.dflare_norm, d_mtp_norm_x_, M, H, cfg_.rms_norm_eps, stream_);
        __nv_bfloat16* k_dst = ml.d_k_ctx + static_cast<size_t>(start_pos) * KVdim;
        __nv_bfloat16* v_dst = ml.d_v_ctx + static_cast<size_t>(start_pos) * KVdim;
        if (mtp_fp8_) {
            run_fp8_linear(d_mtp_norm_x_, ml.wk_ctx_fp8, ml.sk_ctx_fp8, k_dst, M, KVdim, H);
            run_fp8_linear(d_mtp_norm_x_, ml.wv_ctx_fp8, ml.sv_ctx_fp8, v_dst, M, KVdim, H);
        } else {
            run_bf16_linear(d_mtp_norm_x_, ml.wk_ctx, k_dst, M, KVdim, H);
            run_bf16_linear(d_mtp_norm_x_, ml.wv_ctx, v_dst, M, KVdim, H);
        }
        cuda::launch_inkling_mtp_norm_rope(k_dst, ml.k_norm_ctx, d_mtp_inv_freq_, start_pos, M, cfg_.num_kv_heads, cfg_.rms_norm_eps, stream_);
    }
    // Save last committed token's 4 features for f_rnn
    size_t last_off = static_cast<size_t>(M - 1) * H;
    for (int f = 0; f < 4; ++f) {
        INK_CUDA_CHECK(cudaMemcpyAsync(d_mtp_last_feats_[f], d_mtp_feats_[f] + last_off,
                                       H * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_));
    }
}

void Inkling2Engine::mtp_draft_block(i32 anchor_tok, u32 cur_pos, u32 K, std::vector<i32>& out_drafts) {
    out_drafts.clear();
    if (!mtp_loaded_ || K == 0 || cur_pos == 0) return;
    const u32 B = cfg_.ink.mtp_block_size; // 8
    const u32 H = cfg_.hidden_size;        // 2304
    const u32 Qdim = cfg_.q_dim();         // 3072
    const u32 KVdim = cfg_.kv_dim();       // 1024
    const u32 Inter = cfg_.ink.mtp_intermediate_size; // 9216
    const u32 rank = cfg_.ink.mtp_markov_rank;        // 256
    const u32 V = cfg_.vocab_size;
    const f32 eps = cfg_.rms_norm_eps;
    const u32 n_steps = std::min<u32>(K, B - 1); // up to 7 drafts

    // 1. Build 8-token block: [rms_norm(embed[anchor_tok], embed_norm), mask_embed x 7]
    INK_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, &anchor_tok, sizeof(i32), cudaMemcpyHostToDevice, stream_));
    if (w4_embed_)       cuda::launch_embed_lookup_w4a16(d_embed_qw_, d_embed_sc_, d_token_ids_, d_proj_out_, 1, H, stream_);
    else if (fp8_embed_) cuda::launch_embed_lookup_fp8(d_embed_fp8_, d_embed_scale_, d_token_ids_, d_proj_out_, 1, H, stream_);
    else                 cuda::launch_embed_lookup_bf16(d_embed_, d_token_ids_, d_proj_out_, 1, H, stream_);
    cuda::launch_rmsnorm_bf16(d_proj_out_, d_embed_norm_, d_mtp_x_, 1, H, eps, stream_);
    for (u32 b = 1; b < B; ++b) {
        INK_CUDA_CHECK(cudaMemcpyAsync(d_mtp_x_ + static_cast<size_t>(b) * H, d_mtp_mask_embed_,
                                       H * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_));
    }

    // 2. Run 3 MTP Transformer Layers on B=8 tokens (BF16 or FP8 -> 2x faster draft)
    for (int l = 0; l < 3; ++l) {
        auto& ml = mtp_layers_[l];
        cuda::launch_rmsnorm_bf16(d_mtp_x_, ml.input_norm, d_mtp_norm_x_, B, H, eps, stream_);
        if (mtp_fp8_) {
            run_fp8_linear(d_mtp_norm_x_, ml.wq_fp8, ml.sq_fp8, d_mtp_q_, B, Qdim, H);
            run_fp8_linear(d_mtp_norm_x_, ml.wk_fp8, ml.sk_fp8, d_mtp_k_, B, KVdim, H);
            run_fp8_linear(d_mtp_norm_x_, ml.wv_fp8, ml.sv_fp8, d_mtp_v_, B, KVdim, H);
        } else {
            run_bf16_linear(d_mtp_norm_x_, ml.wq, d_mtp_q_, B, Qdim, H);
            run_bf16_linear(d_mtp_norm_x_, ml.wk, d_mtp_k_, B, KVdim, H);
            run_bf16_linear(d_mtp_norm_x_, ml.wv, d_mtp_v_, B, KVdim, H);
        }
        cuda::launch_inkling_mtp_norm_rope(d_mtp_q_, ml.q_norm, d_mtp_inv_freq_, cur_pos, B, cfg_.num_q_heads, eps, stream_);
        cuda::launch_inkling_mtp_norm_rope(d_mtp_k_, ml.k_norm, d_mtp_inv_freq_, cur_pos, B, cfg_.num_kv_heads, eps, stream_);
        u32 ctx_start = (ml.is_local && cur_pos > 1024) ? (cur_pos - 1024) : 0;
        cuda::launch_inkling_mtp_block_attn(d_mtp_q_, ml.d_k_ctx, ml.d_v_ctx, d_mtp_k_, d_mtp_v_, d_mtp_attn_,
                                            ctx_start, cur_pos, B, cfg_.num_q_heads, cfg_.num_kv_heads, stream_);
        if (mtp_fp8_) run_fp8_linear(d_mtp_attn_, ml.wo_fp8, ml.so_fp8, d_mtp_x_, B, H, Qdim, /*add_to_out=*/true);
        else          run_bf16_linear(d_mtp_attn_, ml.wo, d_mtp_x_, B, H, Qdim, /*add_to_out=*/true);

        cuda::launch_rmsnorm_bf16(d_mtp_x_, ml.post_attn_norm, d_mtp_norm_x_, B, H, eps, stream_);
        if (mtp_fp8_) {
            run_fp8_linear(d_mtp_norm_x_, ml.w_gate_fp8, ml.s_gate_fp8, d_mtp_mlp_g_, B, Inter, H);
            run_fp8_linear(d_mtp_norm_x_, ml.w_up_fp8,   ml.s_up_fp8,   d_mtp_mlp_u_, B, Inter, H);
        } else {
            run_bf16_linear(d_mtp_norm_x_, ml.w_gate, d_mtp_mlp_g_, B, Inter, H);
            run_bf16_linear(d_mtp_norm_x_, ml.w_up,   d_mtp_mlp_u_, B, Inter, H);
        }
        cuda::launch_inkling_mtp_silu_mul(d_mtp_mlp_g_, d_mtp_mlp_u_, d_mtp_mlp_g_, B * Inter, stream_);
        if (mtp_fp8_) run_fp8_linear(d_mtp_mlp_g_, ml.w_down_fp8, ml.s_down_fp8, d_mtp_x_, B, H, Inter, /*add_to_out=*/true);
        else          run_bf16_linear(d_mtp_mlp_g_, ml.w_down, d_mtp_x_, B, H, Inter, /*add_to_out=*/true);
    }

    // 3. Final MTP norm -> base logits for steps 0..n_steps-1 (positions 1..n_steps of block)
    cuda::launch_rmsnorm_bf16(d_mtp_x_, d_mtp_norm_, d_mtp_norm_x_, B, H, eps, stream_);
    const __nv_bfloat16* h_steps = d_mtp_norm_x_ + static_cast<size_t>(1) * H;
    if (fp8_attn_) {
        cuda::launch_gemv_fp8(h_steps, d_unembed_fp8_, d_unembed_scale_, nullptr, d_logits_, n_steps, V, H, stream_);
    } else {
        cuda::launch_gemv_bf16(h_steps, d_unembed_, nullptr, d_logits_, n_steps, V, H, stream_);
    }

    // 4. Precomputed Markov RNN Context (10x less serial work) + 1-Kernel Fused Step+Gate
    cuda::launch_inkling_mtp_mix_rmsnorm(d_mtp_feat_ptrs_ + 4, d_mtp_alpha_ + 3 * 4,
                                         d_mtp_mh_norm_, d_mtp_f_rnn_, 1, H, eps, stream_);
    cuda::launch_inkling_mtp_rnn_ctx(h_steps, d_mtp_f_rnn_, d_mtp_rnn_w_, d_mtp_rnn_b_,
                                     d_mtp_rnn_ctx_f32_, n_steps, rank, H, stream_);
    INK_CUDA_CHECK(cudaMemsetAsync(d_mtp_state_, 0, rank * sizeof(__nv_bfloat16), stream_));
    INK_CUDA_CHECK(cudaMemcpyAsync(d_out_tokens_, &anchor_tok, sizeof(i32), cudaMemcpyHostToDevice, stream_));

    for (u32 s = 0; s < n_steps; ++s) {
        cuda::launch_inkling_mtp_rnn_step_gate(d_mtp_rnn_ctx_f32_ + static_cast<size_t>(s) * (3 * rank),
                                               d_mtp_rnn_w_, d_mtp_markov_w1_, d_out_tokens_ + s,
                                               d_mtp_state_, d_mtp_latent_, rank, H, stream_);
        if (d_mtp_fold_fp8_) {
            cuda::launch_gemv_fp8(d_mtp_latent_, d_mtp_fold_fp8_, d_mtp_fold_scale_, d_mtp_fold_b_, d_mtp_bias_, 1, V, rank, stream_);
        } else {
            cuda::launch_gemv_bf16(d_mtp_latent_, d_mtp_fold_w_, d_mtp_fold_b_, d_mtp_bias_, 1, V, rank, stream_);
        }
        cuda::launch_inkling_mtp_add_softcap(d_logits_ + static_cast<size_t>(s) * V, d_mtp_bias_, d_mtp_logits_tmp_,
                                             V, cfg_.ink.unpadded_vocab_size, cfg_.ink.final_logit_softcapping, stream_);
        cuda::launch_argmax_bf16(d_mtp_logits_tmp_, d_out_tokens_ + (s + 1), 1, V, stream_);
    }
    INK_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_ + 1, n_steps * sizeof(i32), cudaMemcpyDeviceToHost, stream_));
    INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
    for (u32 s = 0; s < n_steps; ++s) out_drafts.push_back(h_pinned_token_[s]);
}

void Inkling2Engine::ensure_mtp_graphs() {
    if (mtp_graphs_captured_ || !mtp_loaded_) return;
    const u32 B = cfg_.ink.mtp_block_size; // 8
    const u32 K = B - 1;                   // 7
    const u32 H = cfg_.hidden_size;
    const u32 Qdim = cfg_.q_dim();
    const u32 KVdim = cfg_.kv_dim();
    const u32 Inter = cfg_.ink.mtp_intermediate_size;
    const u32 rank = cfg_.ink.mtp_markov_rank;
    const u32 V = cfg_.vocab_size;
    const f32 eps = cfg_.rms_norm_eps;

    // 1. Capture Full 3-Layer MTP Draft + Precomputed Context + 7-Step Markov Graph
    // Inputs before launch: *d_decode_pos_ = cur_pos, *d_out_tokens_ = anchor_tok
    // Outputs after launch: d_out_tokens_[1..7] = 7 drafted tokens
    INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
    INK_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    if (w4_embed_) {
        cuda::launch_inkling_mtp_build_block_w4a16(d_embed_qw_, d_embed_sc_, d_embed_norm_,
                                                   d_mtp_mask_embed_, d_out_tokens_, d_mtp_x_, B, H, eps, stream_);
    } else if (fp8_embed_) {
        cuda::launch_inkling_mtp_build_block_fp8(d_embed_fp8_, d_embed_scale_, d_embed_norm_, d_mtp_mask_embed_,
                                                 d_out_tokens_, d_mtp_x_, B, H, eps, stream_);
    } else {
        cuda::launch_inkling_mtp_build_block(d_embed_, d_embed_norm_, d_mtp_mask_embed_,
                                             d_out_tokens_, d_mtp_x_, B, H, eps, stream_);
    }
    for (int l = 0; l < 3; ++l) {
        auto& ml = mtp_layers_[l];
        cuda::launch_rmsnorm_bf16(d_mtp_x_, ml.input_norm, d_mtp_norm_x_, B, H, eps, stream_);
        if (mtp_fp8_) {
            run_fp8_linear(d_mtp_norm_x_, ml.wq_fp8, ml.sq_fp8, d_mtp_q_, B, Qdim, H);
            run_fp8_linear(d_mtp_norm_x_, ml.wk_fp8, ml.sk_fp8, d_mtp_k_, B, KVdim, H);
            run_fp8_linear(d_mtp_norm_x_, ml.wv_fp8, ml.sv_fp8, d_mtp_v_, B, KVdim, H);
        } else {
            run_bf16_linear(d_mtp_norm_x_, ml.wq, d_mtp_q_, B, Qdim, H);
            run_bf16_linear(d_mtp_norm_x_, ml.wk, d_mtp_k_, B, KVdim, H);
            run_bf16_linear(d_mtp_norm_x_, ml.wv, d_mtp_v_, B, KVdim, H);
        }
        cuda::launch_inkling_mtp_norm_rope_dpos(d_mtp_q_, ml.q_norm, d_mtp_inv_freq_, d_decode_pos_, B, cfg_.num_q_heads, eps, stream_);
        cuda::launch_inkling_mtp_norm_rope_dpos(d_mtp_k_, ml.k_norm, d_mtp_inv_freq_, d_decode_pos_, B, cfg_.num_kv_heads, eps, stream_);
        cuda::launch_inkling_mtp_block_attn_dpos(d_mtp_q_, ml.d_k_ctx, ml.d_v_ctx, d_mtp_k_, d_mtp_v_, d_mtp_attn_,
                                                 d_decode_pos_, ml.is_local, 1024u, B, cfg_.num_q_heads, cfg_.num_kv_heads, stream_);
        if (mtp_fp8_) run_fp8_linear(d_mtp_attn_, ml.wo_fp8, ml.so_fp8, d_mtp_x_, B, H, Qdim, true);
        else          run_bf16_linear(d_mtp_attn_, ml.wo, d_mtp_x_, B, H, Qdim, true);

        cuda::launch_rmsnorm_bf16(d_mtp_x_, ml.post_attn_norm, d_mtp_norm_x_, B, H, eps, stream_);
        if (mtp_fp8_) {
            run_fp8_linear(d_mtp_norm_x_, ml.w_gate_fp8, ml.s_gate_fp8, d_mtp_mlp_g_, B, Inter, H);
            run_fp8_linear(d_mtp_norm_x_, ml.w_up_fp8,   ml.s_up_fp8,   d_mtp_mlp_u_, B, Inter, H);
        } else {
            run_bf16_linear(d_mtp_norm_x_, ml.w_gate, d_mtp_mlp_g_, B, Inter, H);
            run_bf16_linear(d_mtp_norm_x_, ml.w_up,   d_mtp_mlp_u_, B, Inter, H);
        }
        cuda::launch_inkling_mtp_silu_mul(d_mtp_mlp_g_, d_mtp_mlp_u_, d_mtp_mlp_g_, B * Inter, stream_);
        if (mtp_fp8_) run_fp8_linear(d_mtp_mlp_g_, ml.w_down_fp8, ml.s_down_fp8, d_mtp_x_, B, H, Inter, true);
        else          run_bf16_linear(d_mtp_mlp_g_, ml.w_down, d_mtp_x_, B, H, Inter, true);
    }
    cuda::launch_rmsnorm_bf16(d_mtp_x_, d_mtp_norm_, d_mtp_norm_x_, B, H, eps, stream_);
    const __nv_bfloat16* h_steps = d_mtp_norm_x_ + static_cast<size_t>(1) * H;
    if (fp8_attn_) cuda::launch_gemv_fp8(h_steps, d_unembed_fp8_, d_unembed_scale_, nullptr, d_logits_, K, V, H, stream_);
    else           cuda::launch_gemv_bf16(h_steps, d_unembed_, nullptr, d_logits_, K, V, H, stream_);

    cuda::launch_inkling_mtp_mix_rmsnorm(d_mtp_feat_ptrs_ + 4, d_mtp_alpha_ + 3 * 4,
                                         d_mtp_mh_norm_, d_mtp_f_rnn_, 1, H, eps, stream_);
    cuda::launch_inkling_mtp_rnn_ctx(h_steps, d_mtp_f_rnn_, d_mtp_rnn_w_, d_mtp_rnn_b_,
                                     d_mtp_rnn_ctx_f32_, K, rank, H, stream_);
    INK_CUDA_CHECK(cudaMemsetAsync(d_mtp_state_, 0, rank * sizeof(__nv_bfloat16), stream_));
    for (u32 s = 0; s < K; ++s) {
        cuda::launch_inkling_mtp_rnn_step_gate(d_mtp_rnn_ctx_f32_ + static_cast<size_t>(s) * (3 * rank),
                                               d_mtp_rnn_w_, d_mtp_markov_w1_, d_out_tokens_ + s,
                                               d_mtp_state_, d_mtp_latent_, rank, H, stream_);
        if (d_mtp_fold_fp8_) {
            cuda::launch_gemv_fp8(d_mtp_latent_, d_mtp_fold_fp8_, d_mtp_fold_scale_, d_mtp_fold_b_, d_mtp_bias_, 1, V, rank, stream_);
        } else {
            cuda::launch_gemv_bf16(d_mtp_latent_, d_mtp_fold_w_, d_mtp_fold_b_, d_mtp_bias_, 1, V, rank, stream_);
        }
        cuda::launch_inkling_mtp_add_softcap(d_logits_ + static_cast<size_t>(s) * V, d_mtp_bias_, d_mtp_logits_tmp_,
                                             V, cfg_.ink.unpadded_vocab_size, cfg_.ink.final_logit_softcapping, stream_);
        cuda::launch_argmax_bf16(d_mtp_logits_tmp_, d_out_tokens_ + (s + 1), 1, V, stream_);
    }
    INK_CUDA_CHECK(cudaStreamEndCapture(stream_, &mtp_draft_graph_));
    INK_CUDA_CHECK(cudaGraphInstantiate(&mtp_draft_graph_exec_, mtp_draft_graph_, nullptr, nullptr, 0));

    // 2. Capture Full M=8 Target Verification Pass + Argmax Graph
    // Inputs before launch: d_token_ids_[0..7] = [anchor, d0..d6], *d_decode_pos_ = cur_pos
    // Outputs after launch: d_out_tokens_[0..7] = verified target argmax tokens, d_mtp_feats_[0..3] captured
    INK_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    run_forward(B, d_decode_pos_, 0, false, false, /*capture_mtp=*/true);
    cuda::launch_argmax_bf16(d_logits_, d_out_tokens_, B, V, stream_);
    INK_CUDA_CHECK(cudaStreamEndCapture(stream_, &mtp_verify_graph_));
    INK_CUDA_CHECK(cudaGraphInstantiate(&mtp_verify_graph_exec_, mtp_verify_graph_, nullptr, nullptr, 0));

    mtp_graphs_captured_ = true;
}

void Inkling2Engine::run_forward(u32 M, const i32* d_pos_ptr, u32 host_start_pos, bool is_prefill,
                                 bool draft_1pass, bool capture_mtp) {
    const u32 H = cfg_.hidden_size;
    const u32 Qdim = cfg_.q_dim();
    const u32 KVdim = cfg_.kv_dim();
    const u32 Rdim = cfg_.num_q_heads * cfg_.ink.d_rel;
    const u32 rank = cfg_.ink.gate_rank;
    const f32 eps = cfg_.rms_norm_eps;
    const u32 blk_sz = cfg_.ink.n_layers_per_attn_res_block; // 6

    // 1. Token embedding + embed_norm -> stack[0]
    if (w4_embed_)       cuda::launch_embed_lookup_w4a16(d_embed_qw_, d_embed_sc_, d_token_ids_, d_proj_out_, M, H, stream_);
    else if (fp8_embed_) cuda::launch_embed_lookup_fp8(d_embed_fp8_, d_embed_scale_, d_token_ids_, d_proj_out_, M, H, stream_);
    else                 cuda::launch_embed_lookup_bf16(d_embed_, d_token_ids_, d_proj_out_, M, H, stream_);
    cuda::launch_rmsnorm_bf16(d_proj_out_, d_embed_norm_, d_stack_[0], M, H, eps, stream_);

    u32 num_blocks = 0;
    u32 prelude_blocks = 0;
    bool has_prelude_dests = false;
    bool inputs_are_f32 = false; // true when next visit is a looped pass > 0
    bool inputs_initialized = false;

    for (size_t v_idx = 0; v_idx < visits_.size(); ++v_idx) {
        auto& vs = visits_[v_idx];
        const u32 i = vs.phys_layer;
        const u32 pass_idx = vs.pass_idx;
        if (draft_1pass && pass_idx > 0) continue; // 1-pass looped-depth self-draft skips passes 1 & 2 (2x faster)
        const bool looped = (pass_idx > 0);
        const auto& pl = layers_[i];

        if (i == cfg_.ink.core_start && !looped) {
            prelude_blocks = num_blocks;
            INK_CUDA_CHECK(cudaMemcpyAsync(d_prelude_partial_, d_stack_[num_blocks],
                                           static_cast<size_t>(M) * H * sizeof(__nv_bfloat16),
                                           cudaMemcpyDeviceToDevice, stream_));
        } else if (i == cfg_.ink.core_start && looped) {
            // Copy previous pass's 2 blocks into bank[0..1]
            INK_CUDA_CHECK(cudaMemcpyAsync(d_bank_[0], d_stack_[prelude_blocks + 1],
                                           static_cast<size_t>(M) * H * sizeof(__nv_bfloat16),
                                           cudaMemcpyDeviceToDevice, stream_));
            INK_CUDA_CHECK(cudaMemcpyAsync(d_bank_[1], d_stack_[prelude_blocks + 2],
                                           static_cast<size_t>(M) * H * sizeof(__nv_bfloat16),
                                           cudaMemcpyDeviceToDevice, stream_));
            num_blocks = prelude_blocks;
            INK_CUDA_CHECK(cudaMemcpyAsync(d_stack_[num_blocks], d_prelude_partial_,
                                           static_cast<size_t>(M) * H * sizeof(__nv_bfloat16),
                                           cudaMemcpyDeviceToDevice, stream_));
            if (!has_prelude_dests) {
                cuda::launch_inkling_attn_res(d_ptr_table_, num_blocks, d_prelude_partial_, nullptr,
                                              d_attn_queries_[i - 1], nullptr, nullptr, d_prelude_dests_f32_,
                                              pl.is_local ? 1u : 3u, M, H, eps, cfg_.ink.attn_res_normalize_values, stream_);
                has_prelude_dests = true;
            }
            INK_CUDA_CHECK(cudaMemcpyAsync(d_mix_f32_, d_prelude_dests_f32_,
                                           static_cast<size_t>(pl.is_local ? 1u : 3u) * M * H * sizeof(f32),
                                           cudaMemcpyDeviceToDevice, stream_));
            inputs_are_f32 = true;
            inputs_initialized = true;
        }

        // Capture MTP target hidden states at target_layer_ids = [5, 23, 41] (matches loop.py:372)
        if (capture_mtp && mtp_loaded_) {
            int cap_idx = (v_idx == 5) ? 0 : (v_idx == 23 ? 1 : (v_idx == 41 ? 2 : -1));
            if (cap_idx >= 0) {
                INK_CUDA_CHECK(cudaMemcpyAsync(d_mtp_feats_[cap_idx], d_stack_[num_blocks],
                                               static_cast<size_t>(M) * H * sizeof(__nv_bfloat16),
                                               cudaMemcpyDeviceToDevice, stream_));
            }
        }

        // Prepare attention inputs (q, k, v streams)
        const __nv_bfloat16* x_q = nullptr;
        const __nv_bfloat16* x_k = nullptr;
        const __nv_bfloat16* x_v = nullptr;
        if (looped) {
            // Read all 4 cross-pass messages from bank[0..1]
            u32 c_idx = i - cfg_.ink.core_start;
            cuda::launch_inkling_attn_res(d_ptr_table_ + 5, 1, d_bank_[1], nullptr,
                                          d_msg_queries_[c_idx], nullptr, nullptr, d_msg_f32_,
                                          4, M, H, eps, true, stream_);
            // Blend streams 0..2 (attn/q, k, v) with controllers (fused 3-stream GEMV -> cuts 96 launches/tok)
            cuda::launch_inkling_pair_rmsnorm(d_mix_f32_, d_msg_f32_, d_pair_bf16_,
                                              pl.is_local, 3, M, H, eps, stream_);
            if (M <= 8) {
                cuda::launch_inkling_fused_3ctrl_down(d_pair_bf16_,
                                                      pl.ctrl[0].down, pl.ctrl[1].down, pl.ctrl[2].down,
                                                      d_ctrl_h_, M, rank, 2 * H, stream_);
                cuda::launch_inkling_fused_3ctrl_gf(d_ctrl_h_,
                                                    pl.ctrl[0].gain, pl.ctrl[0].forget,
                                                    pl.ctrl[1].gain, pl.ctrl[1].forget,
                                                    pl.ctrl[2].gain, pl.ctrl[2].forget,
                                                    d_ctrl_gf_, M, H, rank, stream_);
            } else {
                for (int s = 0; s < 3; ++s) {
                    __nv_bfloat16* hs = d_ctrl_h_ + static_cast<size_t>(s) * M * rank;
                    __nv_bfloat16* gs = d_ctrl_gf_ + static_cast<size_t>(s) * M * (2 * H);
                    __nv_bfloat16* fs = gs + static_cast<size_t>(M) * H;
                    run_bf16_linear(d_pair_bf16_ + static_cast<size_t>(s) * M * (2 * H),
                                    pl.ctrl[s].down, hs, M, rank, 2 * H);
                    run_bf16_linear(hs, pl.ctrl[s].gain,   gs, M, H, rank);
                    run_bf16_linear(hs, pl.ctrl[s].forget, fs, M, H, rank);
                }
            }
            // 1-launch 3-stream blend_rmsnorm (replaces 3 sequential launches -> -48 launches/tok)
            cuda::launch_inkling_blend_rmsnorm_3stream(
                d_mix_f32_, d_msg_f32_, d_ctrl_gf_, pl.attn_norm, d_attn_in_,
                pl.is_local, M, H, eps, stream_);
            x_q = d_attn_in_ + 0ULL * M * H;
            x_k = d_attn_in_ + 1ULL * M * H;
            x_v = d_attn_in_ + 2ULL * M * H;
        } else if (!inputs_initialized) {
            // Layer 0 (num_blocks == 0): direct RMSNorm of stack[0]
            cuda::launch_rmsnorm_bf16(d_stack_[0], pl.attn_norm, d_attn_in_, M, H, eps, stream_);
            x_q = x_k = x_v = d_attn_in_;
        } else {
            x_q = d_attn_in_ + 0ULL * M * H;
            x_k = pl.is_local ? x_q : (d_attn_in_ + 1ULL * M * H);
            x_v = pl.is_local ? x_q : (d_attn_in_ + 2ULL * M * H);
        }

        // Attention Projections: 1-launch 3-stream FP8 GEMV covers ALL 6-8 projections on ALL 48 visits (-80 launches/tok)
        bool dyn_fused = false;
        if (fp8_attn_) {
            if (M <= 8) {
                u32 dyn_dim = pl.is_local ? 0u : (cfg_.num_kv_heads * 4);
                cuda::launch_inkling_fused_attn_gemv_fp8(
                    x_q, x_k, x_v,
                    pl.wq_fp8, pl.sq_fp8, pl.wr_fp8, pl.sr_fp8, pl.wg_fp8, pl.sg_fp8, pl.w_sink,
                    pl.wk_fp8, pl.sk_fp8, pl.wv_fp8, pl.sv_fp8,
                    pl.is_local ? nullptr : pl.k_sconv_proj,
                    pl.is_local ? nullptr : pl.v_sconv_proj,
                    d_q_raw_, d_r_raw_, d_gate_, d_sink_, d_k_raw_, d_v_raw_,
                    pl.is_local ? nullptr : d_k_dyn_,
                    pl.is_local ? nullptr : d_v_dyn_,
                    M, Qdim, Rdim, cfg_.num_q_heads, KVdim, dyn_dim, H, stream_);
                dyn_fused = !pl.is_local;
            } else {
                run_fp8_linear(x_q, pl.wq_fp8, pl.sq_fp8, d_q_raw_, M, Qdim, H);
                run_fp8_linear(x_q, pl.wr_fp8, pl.sr_fp8, d_r_raw_, M, Rdim, H);
                run_fp8_linear(x_q, pl.wg_fp8, pl.sg_fp8, d_gate_,  M, Qdim, H);
                run_bf16_linear(x_q, pl.w_sink, d_sink_, M, cfg_.num_q_heads, H);
                run_fp8_linear(x_k, pl.wk_fp8, pl.sk_fp8, d_k_raw_, M, KVdim, H);
                run_fp8_linear(x_v, pl.wv_fp8, pl.sv_fp8, d_v_raw_, M, KVdim, H);
            }
        } else if (M <= 8 && x_q == x_k && x_q == x_v) {
            cuda::launch_inkling_fused_attn_gemv(
                x_q, pl.wq, pl.wr, pl.w_gate, pl.w_sink, pl.wk, pl.wv,
                d_q_raw_, d_r_raw_, d_gate_, d_sink_, d_k_raw_, d_v_raw_,
                M, Qdim, Rdim, cfg_.num_q_heads, KVdim, H, true, stream_);
        } else if (M <= 8) {
            cuda::launch_inkling_fused_attn_gemv(
                x_q, pl.wq, pl.wr, pl.w_gate, pl.w_sink, nullptr, nullptr,
                d_q_raw_, d_r_raw_, d_gate_, d_sink_, nullptr, nullptr,
                M, Qdim, Rdim, cfg_.num_q_heads, KVdim, H, false, stream_);
            run_bf16_linear(x_k, pl.wk, d_k_raw_, M, KVdim, H);
            run_bf16_linear(x_v, pl.wv, d_v_raw_, M, KVdim, H);
        } else {
            run_bf16_linear(x_q, pl.wq,     d_q_raw_, M, Qdim, H);
            run_bf16_linear(x_q, pl.wr,     d_r_raw_, M, Rdim, H);
            run_bf16_linear(x_q, pl.w_gate, d_gate_,  M, Qdim, H);
            run_bf16_linear(x_q, pl.w_sink, d_sink_,  M, cfg_.num_q_heads, H);
            run_bf16_linear(x_k, pl.wk,     d_k_raw_, M, KVdim, H);
            run_bf16_linear(x_v, pl.wv,     d_v_raw_, M, KVdim, H);
        }
        if (!pl.is_local && !dyn_fused) {
            run_bf16_linear(x_k, pl.k_sconv_proj, d_k_dyn_, M, cfg_.num_kv_heads * 4, H);
            run_bf16_linear(x_v, pl.v_sconv_proj, d_v_dyn_, M, cfg_.num_kv_heads * 4, H);
        }

        // Q/R L2-clip (+ tau on global layers) + K/V Short-Conv + K-clip / V-norm + KV write
        const bool global_wht = (!pl.is_local && (kv_mode_ == KvQuantMode::WHT4 || kv_mode_ == KvQuantMode::WHT84));
        const bool global_k4  = (kv_mode_ == KvQuantMode::WHT4);
        if (global_wht) {
            cuda::launch_inkling_qr_clip(
                d_q_raw_, d_r_raw_, pl.q_scale, pl.r_scale, d_q_clip_, d_r_clip_,
                d_pos_ptr, host_start_pos, M, cfg_.num_q_heads,
                cfg_.ink.qk_l2_clip_threshold, cfg_.ink.rel_l2_clip_threshold,
                true, cfg_.ink.log_scaling_n_floor, cfg_.ink.log_scaling_alpha, stream_);
            cuda::launch_inkling_kv_sconv_write_wht(
                d_k_raw_, d_v_raw_, pl.k_sconv, pl.v_sconv, d_k_dyn_, d_v_dyn_,
                vs.d_k_hist, vs.d_v_hist,
                static_cast<u8*>(vs.d_k_cache), static_cast<u8*>(vs.d_v_cache),
                reinterpret_cast<__nv_bfloat16*>(vs.d_k_scale),
                reinterpret_cast<__nv_bfloat16*>(vs.d_v_scale),
                d_block_table_, d_pos_ptr, host_start_pos, M, cfg_.num_kv_heads, kKvBlockSize,
                global_k4, cfg_.ink.qk_l2_clip_threshold, eps, stream_);
        } else if (M == 1) {
            cuda::launch_inkling_qr_kv_fused(
                d_q_raw_, d_r_raw_, pl.q_scale, pl.r_scale, d_q_clip_, d_r_clip_,
                d_k_raw_, d_v_raw_, pl.k_sconv, pl.v_sconv,
                pl.is_local ? nullptr : d_k_dyn_, pl.is_local ? nullptr : d_v_dyn_,
                vs.d_k_hist, vs.d_v_hist,
                vs.d_k_cache, vs.d_v_cache, vs.d_k_scale, vs.d_v_scale,
                pl.is_local ? nullptr : d_block_table_, d_pos_ptr, host_start_pos,
                M, cfg_.num_q_heads, cfg_.num_kv_heads, kKvBlockSize,
                pl.is_local, cfg_.ink.sliding_window_size, !pl.is_local, kv_int8_,
                cfg_.ink.qk_l2_clip_threshold, cfg_.ink.rel_l2_clip_threshold,
                !pl.is_local, cfg_.ink.log_scaling_n_floor, cfg_.ink.log_scaling_alpha, eps, stream_);
        } else {
            cuda::launch_inkling_qr_clip(
                d_q_raw_, d_r_raw_, pl.q_scale, pl.r_scale, d_q_clip_, d_r_clip_,
                d_pos_ptr, host_start_pos, M, cfg_.num_q_heads,
                cfg_.ink.qk_l2_clip_threshold, cfg_.ink.rel_l2_clip_threshold,
                !pl.is_local, cfg_.ink.log_scaling_n_floor, cfg_.ink.log_scaling_alpha, stream_);
            cuda::launch_inkling_kv_sconv_write(
                d_k_raw_, d_v_raw_, pl.k_sconv, pl.v_sconv,
                pl.is_local ? nullptr : d_k_dyn_, pl.is_local ? nullptr : d_v_dyn_,
                vs.d_k_hist, vs.d_v_hist,
                vs.d_k_cache, vs.d_v_cache, vs.d_k_scale, vs.d_v_scale,
                pl.is_local ? nullptr : d_block_table_,
                d_pos_ptr, host_start_pos, M, cfg_.num_kv_heads, kKvBlockSize,
                pl.is_local, cfg_.ink.sliding_window_size, !pl.is_local, kv_int8_,
                cfg_.ink.qk_l2_clip_threshold, eps, stream_);
        }

        // Relative-Position-Bias GQA Attention + Gate & Sink:
        if (global_wht) {
            if (!is_prefill) {
                u32 splits = (M == 1) ? splitk_max_ : std::min<u32>(64u, splitk_max_);
                cuda::launch_inkling_global_attn_splitk_wht(
                    d_q_clip_, d_r_clip_, d_gate_, d_sink_, pl.rel_proj_T,
                    static_cast<const u8*>(vs.d_k_cache), static_cast<const u8*>(vs.d_v_cache),
                    reinterpret_cast<const __nv_bfloat16*>(vs.d_k_scale),
                    reinterpret_cast<const __nv_bfloat16*>(vs.d_v_scale),
                    d_block_table_, d_attn_out_, d_split_o_, d_split_ml_,
                    d_pos_ptr, host_start_pos, M, cfg_.num_q_heads, cfg_.num_kv_heads, kKvBlockSize,
                    splits, global_k4, stream_);
            } else {
                cuda::launch_inkling_global_attn_prefill_wmma_wht(
                    d_q_clip_, d_r_clip_, d_gate_, d_sink_, pl.rel_proj_T,
                    static_cast<const u8*>(vs.d_k_cache), static_cast<const u8*>(vs.d_v_cache),
                    reinterpret_cast<const __nv_bfloat16*>(vs.d_k_scale),
                    reinterpret_cast<const __nv_bfloat16*>(vs.d_v_scale),
                    d_block_table_, d_attn_out_, host_start_pos, M,
                    cfg_.num_q_heads, cfg_.num_kv_heads, kKvBlockSize, global_k4, stream_);
            }
        } else if (!is_prefill && M <= 8) {
            u32 splits = pl.is_local ? 16u : ((M == 1 && max_seq_ >= 65536) ? splitk_max_ : 64u);
            cuda::launch_inkling_rel_attention_splitk(
                d_q_clip_, d_r_clip_, d_gate_, d_sink_, pl.rel_proj_T,
                vs.d_k_cache, vs.d_v_cache, vs.d_k_scale, vs.d_v_scale,
                pl.is_local ? nullptr : d_block_table_, d_attn_out_,
                d_split_o_, d_split_ml_,
                d_pos_ptr, host_start_pos, M, cfg_.num_q_heads, cfg_.num_kv_heads, kKvBlockSize,
                splits, pl.is_local, cfg_.ink.sliding_window_size, kv_int8_, stream_);
        } else if (is_prefill && M >= 16) {
            cuda::launch_inkling_rel_attention_prefill_wmma(
                d_q_clip_, d_r_clip_, d_gate_, d_sink_, pl.rel_proj_T,
                vs.d_k_cache, vs.d_v_cache, vs.d_k_scale, vs.d_v_scale,
                pl.is_local ? nullptr : d_block_table_, d_attn_out_,
                host_start_pos, M, cfg_.num_q_heads, cfg_.num_kv_heads, kKvBlockSize,
                pl.is_local, cfg_.ink.sliding_window_size, kv_int8_, stream_);
        } else {
            cuda::launch_inkling_rel_attention(
                d_q_clip_, d_r_clip_, d_gate_, d_sink_, pl.rel_proj_T,
                vs.d_k_cache, vs.d_v_cache, vs.d_k_scale, vs.d_v_scale,
                pl.is_local ? nullptr : d_block_table_, d_attn_out_,
                d_pos_ptr, host_start_pos, M, cfg_.num_q_heads, cfg_.num_kv_heads, kKvBlockSize,
                pl.is_local, cfg_.ink.sliding_window_size, kv_int8_, stream_);
        }

        // Output projection + Attn Short-Conv: fused M=1..8 FP8 wo_sconv (-48 launches on BOTH base & MTP verify)
        if (fp8_attn_ && M <= 8) {
            cuda::launch_inkling_wo_sconv_fp8(d_attn_out_, pl.wo_fp8, pl.so_fp8, pl.attn_sconv,
                                              vs.d_attn_hist, d_attn_conv_, d_pos_ptr, host_start_pos, M, H, Qdim, stream_);
        } else {
            if (fp8_attn_) run_fp8_linear(d_attn_out_, pl.wo_fp8, pl.so_fp8, d_proj_out_, M, H, Qdim);
            else           run_bf16_linear(d_attn_out_, pl.wo, d_proj_out_, M, H, Qdim);
            cuda::launch_inkling_sconv_hidden(d_proj_out_, pl.attn_sconv, vs.d_attn_hist, d_attn_conv_,
                                              nullptr, nullptr, d_pos_ptr, host_start_pos, M, H, stream_);
        }

        // MLP Input AttnRes Mix
        const bool first = (i % blk_sz == 0);
        const __nv_bfloat16* partial = d_stack_[num_blocks];
        if (looped) {
            cuda::launch_inkling_attn_res(
                d_ptr_table_, num_blocks + (first ? 1u : 0u),
                first ? d_attn_conv_ : partial,
                first ? nullptr : d_attn_conv_,
                d_mlp_queries_[i], nullptr, nullptr, d_mix_f32_,
                1, M, H, eps, cfg_.ink.attn_res_normalize_values, stream_);
            // Blend MLP stream (index 3) with controller[3]
            const f32* m3 = d_msg_f32_ + 3ULL * M * H;
            cuda::launch_inkling_pair_rmsnorm(d_mix_f32_, m3, d_pair_bf16_, false, 1, M, H, eps, stream_);
            run_bf16_linear(d_pair_bf16_, pl.ctrl[3].down, d_ctrl_h_, M, rank, 2 * H);
            if (M <= 8) {
                cuda::launch_inkling_fused_2gemv(d_ctrl_h_, pl.ctrl[3].gain, pl.ctrl[3].forget,
                                                 d_ctrl_gf_, d_ctrl_gf_ + static_cast<size_t>(M) * H,
                                                 M, H, H, rank, stream_);
            } else {
                run_bf16_linear(d_ctrl_h_, pl.ctrl[3].gain,   d_ctrl_gf_, M, H, rank);
                run_bf16_linear(d_ctrl_h_, pl.ctrl[3].forget, d_ctrl_gf_ + static_cast<size_t>(M) * H, M, H, rank);
            }
            cuda::launch_inkling_blend_rmsnorm(d_mix_f32_, m3, d_ctrl_gf_, d_ctrl_gf_ + static_cast<size_t>(M) * H,
                                               pl.mlp_norm, d_mlp_in_, false, false, 1, M, H, eps, stream_);
        } else {
            cuda::launch_inkling_attn_res(
                d_ptr_table_, num_blocks + (first ? 1u : 0u),
                first ? d_attn_conv_ : partial,
                first ? nullptr : d_attn_conv_,
                d_mlp_queries_[i], pl.mlp_norm, d_mlp_in_, nullptr,
                1, M, H, eps, cfg_.ink.attn_res_normalize_values, stream_);
        }

        // Sparse MoE:
        // Parallel 32-SM Router (empirically beats 1-block 1024-thread fused router by +12.7 tok/s on RTX 5090)
        if (pl.moe_gate_fp8) {
            run_fp8_linear(d_mlp_in_, pl.moe_gate_fp8, pl.moe_gate_scale, d_router_logits_, M, cfg_.ink.n_routed_experts, H);
        } else {
            run_bf16_linear(d_mlp_in_, pl.moe_gate_w, d_router_logits_, M, cfg_.ink.n_routed_experts, H);
        }
        cuda::launch_inkling_moe_topk_from_logits(
            d_router_logits_, pl.moe_gate_bias, d_expert_ids_, d_expert_weights_,
            M, cfg_.ink.n_routed_experts, cfg_.ink.num_experts_per_tok, cfg_.ink.route_scale, stream_);

        bool fused_mlp_sconv_done = false;
#if defined(FLINT_HAS_FP4_TC) && FLINT_HAS_FP4_TC == 1
        if (!pl.is_mxfp8 && M >= 16) {
            // Blackwell sm_120a Hardware PTX mxf4nvf4.m16n8k64 FP4 Tensor-Core Zero-Host-Sync Grouped MoE Prefill
            // Keeps exact BF16xNVFP4 on M=2..8 MTP verify for max speed (376-387 tok/s) & 256/256 lossless parity
            cuda::launch_inkling_moe_grouped_nvfp4_fp4tc(
                d_mlp_in_,
                pl.w13_weight, pl.w13_scale, pl.w13_amax, pl.w13_outer_denom,
                pl.w2_weight,  pl.w2_scale,  pl.w2_amax,  pl.w2_outer_denom,
                d_expert_ids_, d_expert_weights_,
                d_moe_pack_idx_, d_moe_pack_gam_,
                d_tile_meta_, d_tile_cnt_, d_num_tiles_,
                d_x_fp4_, d_x_fp4_sc_, d_x_fp4_gs_,
                d_moe_act_, d_act_fp4_, d_act_fp4_sc_, d_act_fp4_gs_,
                d_moe_out_f32_, d_moe_out_,
                M, H, cfg_.intermediate_size, cfg_.ink.n_routed_experts, cfg_.ink.num_experts_per_tok,
                cfg_.ink.mlp_gate_softcap, cfg_.ink.mlp_up_softcap, stream_);
        } else
#endif
        if (M >= 32 && !d_pos_ptr) {
            // Single-Pinned-H2D + Native NVFP4->FP8 cublasLtMatmul Expert-Grouped MoE Prefill
            // Eliminates 12,288 synchronous pageable H2D copies/chunk & doubles MoE prefill TFLOPS
            const u32 E = cfg_.ink.n_routed_experts;
            const u32 topk = cfg_.ink.num_experts_per_tok;
            const u32 inter = cfg_.intermediate_size;
            const u32 total_slots = M * topk;
            INK_CUDA_CHECK(cudaMemcpyAsync(h_moe_pinned_ids_, d_expert_ids_, total_slots * sizeof(i32), cudaMemcpyDeviceToHost, stream_));
            INK_CUDA_CHECK(cudaMemcpyAsync(h_moe_pinned_wts_, d_expert_weights_, total_slots * sizeof(f32), cudaMemcpyDeviceToHost, stream_));
            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));

            u32 counts[128] = {};
            for (u32 idx = 0; idx < total_slots; ++idx) counts[h_moe_pinned_ids_[idx]]++;
            u32 offsets[128];
            u32 cur_off = 0;
            for (u32 e = 0; e < E; ++e) { offsets[e] = cur_off; cur_off += counts[e]; }
            u32 write_pos[128];
            std::memcpy(write_pos, offsets, E * sizeof(u32));
            for (u32 m = 0; m < M; ++m) {
                for (u32 k = 0; k < topk; ++k) {
                    i32 eid = h_moe_pinned_ids_[m * topk + k];
                    u32 p = write_pos[eid]++;
                    h_moe_pack_idx_[p] = static_cast<i32>(m);
                    h_moe_pack_gam_[p] = h_moe_pinned_wts_[m * topk + k];
                }
            }
            // ONE single async pinned H2D upload for ALL 128 experts
            INK_CUDA_CHECK(cudaMemcpyAsync(d_moe_pack_idx_, h_moe_pack_idx_, total_slots * sizeof(i32), cudaMemcpyHostToDevice, stream_));
            INK_CUDA_CHECK(cudaMemcpyAsync(d_moe_pack_gam_, h_moe_pack_gam_, total_slots * sizeof(f32), cudaMemcpyHostToDevice, stream_));

            INK_CUDA_CHECK(cudaMemsetAsync(d_moe_out_f32_, 0, static_cast<size_t>(M) * H * sizeof(f32), stream_));
            auto* d_moe_fp8_buf = reinterpret_cast<u8*>(d_moe_dequant_);
            for (u32 e = 0; e < E; ++e) {
                u32 cnt = counts[e];
                if (cnt == 0) continue;
                const i32* e_idx = d_moe_pack_idx_ + offsets[e];
                const f32* e_gam = d_moe_pack_gam_ + offsets[e];
                cuda::launch_inkling_moe_gather(d_mlp_in_, e_idx, d_moe_gather_x_, cnt, H, stream_);

                if (pl.is_mxfp8) {
                    size_t w13_off = static_cast<size_t>(e) * (2 * inter) * H;
                    size_t s13_off = static_cast<size_t>(e) * (2 * inter) * (H >> 5);
                    cuda::launch_dequant_mxfp8_expert(pl.w13_weight + w13_off, pl.w13_scale + s13_off,
                                                      d_moe_dequant_, 2 * inter, H, stream_);
                    run_bf16_linear(d_moe_gather_x_, d_moe_dequant_, d_moe_w13_out_, cnt, 2 * inter, H);
                } else {
                    size_t w13_off = static_cast<size_t>(e) * (2 * inter) * (H >> 1);
                    size_t s13_off = static_cast<size_t>(e) * (2 * inter) * (H >> 4);
                    if (cnt >= 16) {
                        cuda::launch_dequant_nvfp4_expert_to_fp8(pl.w13_weight + w13_off, pl.w13_scale + s13_off,
                                                                 pl.w13_amax + e, pl.w13_outer_denom,
                                                                 d_moe_fp8_buf, d_moe_fp8_scales_, 2 * inter, H, stream_);
                        if (!run_fp8_cublaslt_gemm(d_moe_gather_x_, d_moe_fp8_buf, d_moe_fp8_scales_, d_moe_w13_out_, cnt, 2 * inter, H)) {
                            cuda::launch_dequant_nvfp4_expert(pl.w13_weight + w13_off, pl.w13_scale + s13_off,
                                                              pl.w13_amax + e, pl.w13_outer_denom,
                                                              d_moe_dequant_, 2 * inter, H, stream_);
                            run_bf16_linear(d_moe_gather_x_, d_moe_dequant_, d_moe_w13_out_, cnt, 2 * inter, H);
                        }
                    } else {
                        cuda::launch_dequant_nvfp4_expert(pl.w13_weight + w13_off, pl.w13_scale + s13_off,
                                                          pl.w13_amax + e, pl.w13_outer_denom,
                                                          d_moe_dequant_, 2 * inter, H, stream_);
                        run_bf16_linear(d_moe_gather_x_, d_moe_dequant_, d_moe_w13_out_, cnt, 2 * inter, H);
                    }
                }
                cuda::launch_inkling_moe_situ_batched(d_moe_w13_out_, e_gam, d_moe_act_b_,
                                                      cnt, inter, cfg_.ink.mlp_gate_softcap, cfg_.ink.mlp_up_softcap, stream_);

                if (pl.is_mxfp8) {
                    size_t w2_off = static_cast<size_t>(e) * H * inter;
                    size_t s2_off = static_cast<size_t>(e) * H * (inter >> 5);
                    cuda::launch_dequant_mxfp8_expert(pl.w2_weight + w2_off, pl.w2_scale + s2_off,
                                                      d_moe_dequant_, H, inter, stream_);
                    run_bf16_linear(d_moe_act_b_, d_moe_dequant_, d_moe_y_b_, cnt, H, inter);
                } else {
                    size_t w2_off = static_cast<size_t>(e) * H * (inter >> 1);
                    size_t s2_off = static_cast<size_t>(e) * H * (inter >> 4);
                    if (cnt >= 16) {
                        cuda::launch_dequant_nvfp4_expert_to_fp8(pl.w2_weight + w2_off, pl.w2_scale + s2_off,
                                                                 pl.w2_amax + e, pl.w2_outer_denom,
                                                                 d_moe_fp8_buf, d_moe_fp8_scales_, H, inter, stream_);
                        if (!run_fp8_cublaslt_gemm(d_moe_act_b_, d_moe_fp8_buf, d_moe_fp8_scales_, d_moe_y_b_, cnt, H, inter)) {
                            cuda::launch_dequant_nvfp4_expert(pl.w2_weight + w2_off, pl.w2_scale + s2_off,
                                                              pl.w2_amax + e, pl.w2_outer_denom,
                                                              d_moe_dequant_, H, inter, stream_);
                            run_bf16_linear(d_moe_act_b_, d_moe_dequant_, d_moe_y_b_, cnt, H, inter);
                        }
                    } else {
                        cuda::launch_dequant_nvfp4_expert(pl.w2_weight + w2_off, pl.w2_scale + s2_off,
                                                          pl.w2_amax + e, pl.w2_outer_denom,
                                                          d_moe_dequant_, H, inter, stream_);
                        run_bf16_linear(d_moe_act_b_, d_moe_dequant_, d_moe_y_b_, cnt, H, inter);
                    }
                }
                cuda::launch_inkling_moe_scatter_add(d_moe_y_b_, e_idx, d_moe_out_f32_, cnt, H, stream_);
            }
            cuda::launch_f32_to_bf16(d_moe_out_f32_, d_moe_out_, M * H, stream_);
        } else if (pl.is_mxfp8) {
            cuda::launch_inkling_moe_mxfp8_gate_up_situ(
                d_mlp_in_, pl.w13_weight, pl.w13_scale,
                d_expert_ids_, d_expert_weights_, d_moe_act_, M, cfg_.intermediate_size, H,
                cfg_.ink.num_experts_per_tok, cfg_.ink.mlp_gate_softcap, cfg_.ink.mlp_up_softcap, stream_);
            cuda::launch_inkling_moe_mxfp8_down_combine(
                d_moe_act_, pl.w2_weight, pl.w2_scale,
                d_expert_ids_, d_moe_out_, M, H, cfg_.intermediate_size,
                cfg_.ink.num_experts_per_tok, stream_);
        } else {
            if (M == 1) {
                cuda::launch_inkling_moe_nvfp4_gate_up_situ(
                    d_mlp_in_, pl.w13_weight, pl.w13_scale, pl.w13_amax, pl.w13_outer_denom,
                    d_expert_ids_, d_expert_weights_, d_moe_act_, M, cfg_.intermediate_size, H,
                    cfg_.ink.num_experts_per_tok, cfg_.ink.mlp_gate_softcap, cfg_.ink.mlp_up_softcap, stream_);
            } else if (M <= 8) {
                // Compact-list active-expert NVFP4 w13 (reads each unique active expert ONCE, 0 atomicAdd, 0 spills)
                cuda::launch_inkling_moe_compact_nvfp4_w13(
                    d_mlp_in_, pl.w13_weight, pl.w13_scale, pl.w13_amax, pl.w13_outer_denom,
                    d_expert_ids_, d_expert_weights_, d_moe_act_,
                    d_uniq_e_, d_uniq_mask_, d_uniq_count_,
                    M, cfg_.intermediate_size, H, cfg_.ink.num_experts_per_tok,
                    cfg_.ink.mlp_gate_softcap, cfg_.ink.mlp_up_softcap, stream_);
            } else {
                cuda::launch_inkling_moe_nvfp4_gate_up_situ(
                    d_mlp_in_, pl.w13_weight, pl.w13_scale, pl.w13_amax, pl.w13_outer_denom,
                    d_expert_ids_, d_expert_weights_, d_moe_act_, M, cfg_.intermediate_size, H,
                    cfg_.ink.num_experts_per_tok, cfg_.ink.mlp_gate_softcap, cfg_.ink.mlp_up_softcap, stream_);
            }
            if (M <= 8) {
                // Fused M=1..8 NVFP4 down + mlp_sconv + residual add (-48 launches & -48 round-trips on BOTH base & MTP)
                if (first) {
                    num_blocks++;
                    cuda::launch_inkling_moe_nvfp4_down_sconv(
                        d_moe_act_, pl.w2_weight, pl.w2_scale, pl.w2_amax, pl.w2_outer_denom,
                        d_expert_ids_, pl.mlp_sconv, vs.d_mlp_hist, d_stack_[num_blocks],
                        d_attn_conv_, nullptr, d_pos_ptr, host_start_pos,
                        M, H, cfg_.intermediate_size, cfg_.ink.num_experts_per_tok, stream_);
                } else {
                    cuda::launch_inkling_moe_nvfp4_down_sconv(
                        d_moe_act_, pl.w2_weight, pl.w2_scale, pl.w2_amax, pl.w2_outer_denom,
                        d_expert_ids_, pl.mlp_sconv, vs.d_mlp_hist, d_stack_[num_blocks],
                        d_attn_conv_, partial, d_pos_ptr, host_start_pos,
                        M, H, cfg_.intermediate_size, cfg_.ink.num_experts_per_tok, stream_);
                }
                fused_mlp_sconv_done = true;
            } else {
                cuda::launch_inkling_moe_nvfp4_down_combine(
                    d_moe_act_, pl.w2_weight, pl.w2_scale, pl.w2_amax, pl.w2_outer_denom,
                    d_expert_ids_, d_moe_out_, M, H, cfg_.intermediate_size,
                    cfg_.ink.num_experts_per_tok, stream_);
            }
        }

        // MLP Short-Conv + Residual Stack Update (skipped when fused into NVFP4 down_sconv)
        if (!fused_mlp_sconv_done) {
            if (first) {
                num_blocks++;
                cuda::launch_inkling_sconv_hidden(d_moe_out_, pl.mlp_sconv, vs.d_mlp_hist, d_stack_[num_blocks],
                                                  d_attn_conv_, nullptr, d_pos_ptr, host_start_pos, M, H, stream_);
            } else {
                cuda::launch_inkling_sconv_hidden(d_moe_out_, pl.mlp_sconv, vs.d_mlp_hist, d_stack_[num_blocks],
                                                  d_attn_conv_, partial, d_pos_ptr, host_start_pos, M, H, stream_);
            }
        }

        // Next visit's AttnRes mix (or final output mix after last executed visit)
        size_t next_v = v_idx + 1;
        while (draft_1pass && next_v < visits_.size() && visits_[next_v].pass_idx > 0) next_v++;
        if (next_v >= visits_.size()) {
            cuda::launch_inkling_attn_res(
                d_ptr_table_, num_blocks, d_stack_[num_blocks], nullptr,
                d_output_query_, d_final_norm_, d_attn_in_, nullptr,
                1, M, H, eps, cfg_.ink.attn_res_normalize_values, stream_);
        } else {
            u32 next_i = visits_[next_v].phys_layer;
            u32 next_pass = visits_[next_v].pass_idx;
            bool next_is_local = layers_[next_i].is_local;
            u32 num_q = next_is_local ? 1u : 3u;
            if (next_pass > 0) {
                cuda::launch_inkling_attn_res(
                    d_ptr_table_, num_blocks, d_stack_[num_blocks], nullptr,
                    d_attn_queries_[next_i - 1], nullptr, nullptr, d_mix_f32_,
                    num_q, M, H, eps, cfg_.ink.attn_res_normalize_values, stream_);
                inputs_are_f32 = true;
            } else {
                cuda::launch_inkling_attn_res(
                    d_ptr_table_, num_blocks, d_stack_[num_blocks], nullptr,
                    d_attn_queries_[next_i - 1], layers_[next_i].attn_norm, d_attn_in_, nullptr,
                    num_q, M, H, eps, cfg_.ink.attn_res_normalize_values, stream_);
                inputs_are_f32 = false;
            }
            inputs_initialized = true;
        }
    }
    (void)inputs_are_f32;
    if (capture_mtp && mtp_loaded_) {
        INK_CUDA_CHECK(cudaMemcpyAsync(d_mtp_feats_[3], d_attn_in_,
                                       static_cast<size_t>(M) * H * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToDevice, stream_));
    }

    // Final Unembed LM Head + Softcap(30.0) + Mask Padded Vocab (>= 200058)
    const __nv_bfloat16* lm_in = is_prefill ? (d_attn_in_ + static_cast<size_t>(M - 1) * H) : d_attn_in_;
    u32 lm_M = is_prefill ? 1 : M;
    if (fp8_attn_) {
        cuda::launch_gemv_fp8(lm_in, d_unembed_fp8_, d_unembed_scale_, nullptr, d_logits_, lm_M, cfg_.vocab_size, H, stream_);
    } else {
        cuda::launch_gemv_bf16(lm_in, d_unembed_, nullptr, d_logits_, lm_M, cfg_.vocab_size, H, stream_);
    }
    cuda::launch_inkling_logit_softcap(d_logits_, lm_M, cfg_.vocab_size, cfg_.ink.unpadded_vocab_size,
                                       cfg_.ink.final_logit_softcapping, stream_);
}

void Inkling2Engine::ensure_decode_graph() {
    if (graph_captured_) return;
    INK_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    run_forward(1, d_decode_pos_, 0, false);
    cuda::launch_argmax_bf16(d_logits_, d_out_tokens_, 1, cfg_.vocab_size, stream_);
    cuda::launch_advance_decode_state(d_decode_pos_, d_token_ids_, d_out_tokens_, stream_);
    INK_CUDA_CHECK(cudaStreamEndCapture(stream_, &decode_graph_));
    INK_CUDA_CHECK(cudaGraphInstantiate(&decode_graph_exec_, decode_graph_, nullptr, nullptr, 0));
    graph_captured_ = true;
}

#endif

} // namespace flint
