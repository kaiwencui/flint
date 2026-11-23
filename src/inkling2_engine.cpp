#include "inkling2_engine.hpp"
#include "engine.hpp"
#include "kernels/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <thread>

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
#endif

Inkling2Engine::~Inkling2Engine() {
#if defined(FLINT_CUDA) && FLINT_CUDA == 1
    free_resources();
#endif
}

void Inkling2Engine::clear_kv_cache() {
#if defined(FLINT_CUDA) && FLINT_CUDA == 1
    kv_mgr_.clear();
    prefix_snap_len_ = 0;
    for (auto& v : visits_) {
        if (v.d_k_hist)    cudaMemsetAsync(v.d_k_hist,    0, 16ULL * cfg_.kv_dim() * sizeof(__nv_bfloat16), stream_);
        if (v.d_v_hist)    cudaMemsetAsync(v.d_v_hist,    0, 16ULL * cfg_.kv_dim() * sizeof(__nv_bfloat16), stream_);
        if (v.d_attn_hist) cudaMemsetAsync(v.d_attn_hist, 0, 16ULL * cfg_.hidden_size * sizeof(__nv_bfloat16), stream_);
        if (v.d_mlp_hist)  cudaMemsetAsync(v.d_mlp_hist,  0, 16ULL * cfg_.hidden_size * sizeof(__nv_bfloat16), stream_);
    }
    if (stream_) cudaStreamSynchronize(stream_);
#endif
}

#if defined(FLINT_CUDA) && FLINT_CUDA == 1

void Inkling2Engine::free_resources() {
    destroy_decode_graph();
    destroy_mtp_graphs();
    auto dfree = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
    if (d_weight_slab_) {
        dfree(d_weight_slab_);
        // Also free transformed/derived buffers that were separately allocated
        for (auto& l : layers_) {
            dfree(l.rel_proj_T);
            if (!l.is_local) dfree(l.attn_norm); // stacked [3, H]
        }
        layers_.clear();
        d_embed_ = nullptr;
        d_embed_fp8_ = nullptr;
        d_embed_scale_ = nullptr;
        d_embed_qw_ = nullptr;
        d_embed_sc_ = nullptr;
        d_embed_norm_ = nullptr;
        d_final_norm_ = nullptr;
        d_unembed_ = nullptr;
    } else {
        dfree(d_embed_);
        dfree(d_embed_fp8_);
        dfree(d_embed_scale_);
        dfree(d_embed_qw_);
        dfree(d_embed_sc_);
        dfree(d_embed_norm_);
        dfree(d_final_norm_);
        dfree(d_unembed_);
        for (auto& l : layers_) {
            dfree(l.attn_norm); dfree(l.mlp_norm);
            dfree(l.wq); dfree(l.wr); dfree(l.w_gate); dfree(l.w_sink);
            dfree(l.wk); dfree(l.wv); dfree(l.wo);
            dfree(l.q_scale); dfree(l.r_scale); dfree(l.rel_proj_T);
            dfree(l.k_sconv); dfree(l.v_sconv); dfree(l.k_sconv_proj); dfree(l.v_sconv_proj);
            dfree(l.attn_sconv); dfree(l.mlp_sconv);
            dfree(l.moe_gate_w); dfree(l.moe_gate_bias);
            dfree(l.w13_weight); dfree(l.w13_scale); dfree(l.w13_amax); dfree(l.w13_outer_denom);
            dfree(l.w2_weight);  dfree(l.w2_scale);  dfree(l.w2_amax);  dfree(l.w2_outer_denom);
            for (int c = 0; c < 4; ++c) {
                dfree(l.ctrl[c].down); dfree(l.ctrl[c].gain); dfree(l.ctrl[c].forget);
            }
        }
        layers_.clear();
    }
    for (auto*& p : d_attn_queries_) dfree(p);
    for (auto*& p : d_mlp_queries_) dfree(p);
    for (auto*& p : d_msg_queries_) dfree(p);
    d_attn_queries_.clear(); d_mlp_queries_.clear(); d_msg_queries_.clear();
    dfree(d_output_query_);

    for (auto& v : visits_) {
        dfree(v.d_k_cache); dfree(v.d_v_cache);
        dfree(v.d_k_scale); dfree(v.d_v_scale);
        dfree(v.d_k_hist);  dfree(v.d_v_hist);
        dfree(v.d_attn_hist); dfree(v.d_mlp_hist);
    }
    visits_.clear();
    for (auto*& p : d_hist_backup_) dfree(p);
    d_hist_backup_.clear();

    dfree(d_block_table_);
    for (int i = 0; i < 5; ++i) dfree(d_stack_[i]);
    dfree(d_prelude_partial_);
    for (int i = 0; i < 2; ++i) dfree(d_bank_[i]);
    dfree(d_ptr_table_);
    dfree(d_mix_f32_); dfree(d_prelude_dests_f32_); dfree(d_msg_f32_);
    dfree(d_pair_bf16_); dfree(d_ctrl_h_); dfree(d_ctrl_gf_);
    dfree(d_attn_in_); dfree(d_mlp_in_);
    dfree(d_q_raw_); dfree(d_q_clip_); dfree(d_r_raw_); dfree(d_r_clip_);
    dfree(d_gate_); dfree(d_sink_);
    dfree(d_k_raw_); dfree(d_v_raw_); dfree(d_k_dyn_); dfree(d_v_dyn_);
    dfree(d_attn_out_); dfree(d_proj_out_); dfree(d_attn_conv_);
    dfree(d_moe_out_); dfree(d_moe_act_);
    dfree(d_router_logits_); dfree(d_moe_dequant_); dfree(d_moe_gather_x_);
    dfree(d_moe_w13_out_); dfree(d_moe_act_b_); dfree(d_moe_y_b_);
    dfree(d_moe_out_f32_); dfree(d_moe_tok_idx_); dfree(d_moe_gammas_);
    dfree(d_expert_ids_); dfree(d_expert_weights_);
    dfree(d_uniq_e_); dfree(d_uniq_mask_); dfree(d_uniq_count_);
    dfree(d_logits_); dfree(d_split_o_); dfree(d_split_ml_);
    dfree(d_token_ids_); dfree(d_out_tokens_); dfree(d_decode_pos_);
    for (int l = 0; l < 3; ++l) { dfree(mtp_layers_[l].d_k_ctx); dfree(mtp_layers_[l].d_v_ctx); }
    dfree(d_mtp_alpha_); dfree(d_mtp_inv_freq_);
    dfree(d_mtp_fold_w_); dfree(d_mtp_fold_fp8_); dfree(d_mtp_fold_scale_); dfree(d_mtp_fold_b_);
    for (int f = 0; f < 4; ++f) { dfree(d_mtp_feats_[f]); dfree(d_mtp_last_feats_[f]); }
    dfree(d_mtp_feat_ptrs_);
    dfree(d_mtp_x_); dfree(d_mtp_norm_x_); dfree(d_mtp_q_); dfree(d_mtp_k_); dfree(d_mtp_v_);
    dfree(d_mtp_attn_); dfree(d_mtp_mlp_g_); dfree(d_mtp_mlp_u_);
    dfree(d_mtp_f_rnn_); dfree(d_mtp_state_); dfree(d_mtp_latent_);
    dfree(d_mtp_rnn_in_); dfree(d_mtp_raw3r_); dfree(d_mtp_rnn_ctx_f32_); dfree(d_mtp_bias_); dfree(d_mtp_logits_tmp_);
    for (auto& ps : prefix_snap_) {
        dfree(ps.d_k_snap); dfree(ps.d_v_snap); dfree(ps.d_ks_snap); dfree(ps.d_vs_snap);
        dfree(ps.d_kh_snap); dfree(ps.d_vh_snap); dfree(ps.d_ah_snap); dfree(ps.d_mh_snap);
    }
    prefix_snap_.clear();
    prefix_tokens_.clear();
    prefix_bt_.clear();
    prefix_snap_alloc_ = false;
    prefix_snap_len_ = 0;
    mtp_loaded_ = false;
    dfree(d_fp8_x_); dfree(d_fp8_scale_); dfree(d_fp8_out_pad_);
    dfree(d_moe_fp8_scales_); dfree(d_cublaslt_ws_);
    dfree(d_moe_pack_idx_); dfree(d_moe_pack_gam_);
    dfree(d_tile_meta_); dfree(d_tile_cnt_); dfree(d_num_tiles_);
    dfree(d_x_fp4_); dfree(d_x_fp4_sc_); dfree(d_x_fp4_gs_);
    dfree(d_act_fp4_); dfree(d_act_fp4_sc_); dfree(d_act_fp4_gs_);
    if (h_moe_pinned_ids_) { cudaFreeHost(h_moe_pinned_ids_); h_moe_pinned_ids_ = nullptr; }
    if (h_moe_pinned_wts_) { cudaFreeHost(h_moe_pinned_wts_); h_moe_pinned_wts_ = nullptr; }
    if (h_moe_pack_idx_)   { cudaFreeHost(h_moe_pack_idx_);   h_moe_pack_idx_   = nullptr; }
    if (h_moe_pack_gam_)   { cudaFreeHost(h_moe_pack_gam_);   h_moe_pack_gam_   = nullptr; }
    if (h_pinned_token_) { cudaFreeHost(h_pinned_token_); h_pinned_token_ = nullptr; }
    lt_algo_cache_.clear();
    if (cublaslt_) { cublasLtDestroy(cublaslt_); cublaslt_ = nullptr; }
    if (cublas_) { cublasDestroy(cublas_); cublas_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

bool Inkling2Engine::load(const ModelConfig& cfg,
                          const BpeTokenizer& tok,
                          const EngineOptions& opts,
                          std::string& err) {
    free_resources();
    cfg_ = cfg;
    tok_ = &tok;
    kv_mode_ = opts.kv_mode;
    if (opts.kv_fp8 && kv_mode_ == KvQuantMode::BF16) kv_mode_ = KvQuantMode::INT8;
    kv_int8_ = (kv_mode_ != KvQuantMode::BF16); // Local SWA rings stay exact INT8 (only 85 MiB)
    use_cuda_graph_ = opts.use_cuda_graph;
    spec_mode_ = opts.spec_mode;

    cudaDeviceProp prop{};
    INK_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    gpu_name_ = prop.name;
    int mem_clk_khz = 0, bus_bits = 0;
    cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    gpu_peak_hbm_gbs_ = 2.0 * static_cast<double>(mem_clk_khz) * static_cast<double>(bus_bits) / 8.0 / 1e6;

    INK_CUDA_CHECK(cudaStreamCreate(&stream_));
    if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) {
        err = "cublasCreate failed for Inkling2Engine";
        return false;
    }
    cublasSetStream(cublas_, stream_);

    std::error_code ec;
    bool is_pack = std::filesystem::is_regular_file(opts.model_path, ec) &&
                   opts.model_path.extension() == ".pack";

    // Double-buffered 64 MiB pinned H2D staging pool with 4-thread parallel memcpy
    constexpr size_t kStageBytes = 64ULL * 1024 * 1024;
    u8* h_stage[2] = {nullptr, nullptr};
    cudaEvent_t ev_stage[2] = {nullptr, nullptr};
    INK_CUDA_CHECK(cudaHostAlloc(&h_stage[0], kStageBytes, cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaHostAlloc(&h_stage[1], kStageBytes, cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaEventCreateWithFlags(&ev_stage[0], cudaEventDisableTiming));
    INK_CUDA_CHECK(cudaEventCreateWithFlags(&ev_stage[1], cudaEventDisableTiming));
    int stage_idx = 0;

    auto cleanup_stage = [&]() {
        cudaStreamSynchronize(stream_);
        if (ev_stage[0]) cudaEventDestroy(ev_stage[0]);
        if (ev_stage[1]) cudaEventDestroy(ev_stage[1]);
        if (h_stage[0]) cudaFreeHost(h_stage[0]);
        if (h_stage[1]) cudaFreeHost(h_stage[1]);
    };

    auto parallel_memcpy = [](u8* dst, const u8* src, size_t nbytes) {
        if (nbytes < 8ULL * 1024 * 1024) {
            std::memcpy(dst, src, nbytes);
            return;
        }
        constexpr size_t kThreads = 4;
        std::thread ts[kThreads - 1];
        size_t part = (nbytes / kThreads) & ~63ULL;
        for (size_t t = 0; t < kThreads - 1; ++t) {
            ts[t] = std::thread([=]() { std::memcpy(dst + t * part, src + t * part, part); });
        }
        size_t tail_off = (kThreads - 1) * part;
        std::memcpy(dst + tail_off, src + tail_off, nbytes - tail_off);
        for (auto& th : ts) th.join();
    };

    auto stream_h2d_into = [&](const void* src, size_t nbytes, void* dst_dev) {
        const auto* s = static_cast<const u8*>(src);
        auto* d = static_cast<u8*>(dst_dev);
        size_t off = 0;
        while (off < nbytes) {
            size_t chunk = std::min(kStageBytes, nbytes - off);
            int cur = stage_idx;
            stage_idx ^= 1;
            INK_CUDA_CHECK(cudaEventSynchronize(ev_stage[cur]));
            parallel_memcpy(h_stage[cur], s + off, chunk);
            INK_CUDA_CHECK(cudaMemcpyAsync(d + off, h_stage[cur], chunk, cudaMemcpyHostToDevice, stream_));
            INK_CUDA_CHECK(cudaEventRecord(ev_stage[cur], stream_));
            off += chunk;
        }
    };

    auto upload_raw = [&](const void* src, size_t nbytes, void** dst) {
        INK_CUDA_CHECK(cudaMalloc(dst, nbytes));
        stream_h2d_into(src, nbytes, *dst);
    };

    PackedModelReader pr;
    SafetensorsLoader st;
    const u8* host_slab = nullptr;
    if (is_pack) {
        if (!pr.open(opts.model_path, err)) { cleanup_stage(); return false; }
        host_slab = pr.base_ptr() + pr.data_offset();
        u64 slab_bytes = pr.weight_slab_bytes();
        // 16GB optimization: mtp.* tensors are packed last -> skip uploading them (~295 MiB) when spec != mtp
        if (spec_mode_ != SpecMode::Mtp) {
            u64 non_mtp_end = 0;
            for (const auto& [tname, tv] : pr.tensors()) {
                if (tname.rfind("mtp.", 0) == 0) continue;
                u64 end_off = static_cast<u64>(tv.data - host_slab) + tv.nbytes;
                if (end_off > non_mtp_end) non_mtp_end = end_off;
            }
            if (non_mtp_end > 0 && non_mtp_end < slab_bytes) {
                slab_bytes = (non_mtp_end + 255ULL) & ~255ULL;
            }
        }
        INK_CUDA_CHECK(cudaMalloc(&d_weight_slab_, slab_bytes));
        stream_h2d_into(host_slab, slab_bytes, d_weight_slab_);
    } else {
        if (!st.open_dir(opts.model_path, err)) { cleanup_stage(); return false; }
    }

    auto get_tv = [&](const std::string& name) -> const TensorView* {
        return is_pack ? pr.get(name) : st.get(name);
    };

    auto bind_or_upload = [&](const std::string& name, void** dst, bool required = true) -> bool {
        const auto* tv = get_tv(name);
        if (!tv) {
            if (required) { err = "Missing Inkling-2 tensor: " + name; return false; }
            *dst = nullptr;
            return true;
        }
        if (is_pack) {
            *dst = d_weight_slab_ + static_cast<size_t>(tv->data - host_slab);
        } else {
            upload_raw(tv->data, tv->nbytes, dst);
        }
        return true;
    };

    const u32 H = cfg_.hidden_size; // 2304
    const f32 q_scale_inv_sqrt = 1.0f / std::sqrt(static_cast<f32>(H));

    // Helper: read F32/BF16 query tensor, multiply by H**-0.5, and upload as F32
    auto upload_scaled_f32_query = [&](const std::string& name, f32** dst) -> bool {
        const auto* tv = get_tv(name);
        if (!tv) { err = "Missing Inkling-2 query: " + name; return false; }
        std::vector<f32> tmp;
        if (!SafetensorsLoader::read_as_f32(*tv, tmp)) { err = "Bad query dtype: " + name; return false; }
        for (f32& v : tmp) v *= q_scale_inv_sqrt;
        upload_raw(tmp.data(), tmp.size() * sizeof(f32), reinterpret_cast<void**>(dst));
        return true;
    };

    fp8_attn_  = (get_tv("model.llm.unembed.weight.fp8") != nullptr);
    w4_embed_  = (get_tv("model.llm.embed.weight.qweight") != nullptr);
    fp8_embed_ = (!w4_embed_ && get_tv("model.llm.embed.weight.fp8") != nullptr);
    if (w4_embed_) {
        if (!bind_or_upload("model.llm.embed.weight.qweight", reinterpret_cast<void**>(&d_embed_qw_))) { cleanup_stage(); return false; }
        if (!bind_or_upload("model.llm.embed.weight.scales",  reinterpret_cast<void**>(&d_embed_sc_))) { cleanup_stage(); return false; }
    } else if (fp8_embed_) {
        if (!bind_or_upload("model.llm.embed.weight.fp8",   reinterpret_cast<void**>(&d_embed_fp8_)))   { cleanup_stage(); return false; }
        if (!bind_or_upload("model.llm.embed.weight.scale", reinterpret_cast<void**>(&d_embed_scale_))) { cleanup_stage(); return false; }
    } else {
        if (!bind_or_upload("model.llm.embed.weight", reinterpret_cast<void**>(&d_embed_))) { cleanup_stage(); return false; }
    }
    if (!bind_or_upload("model.llm.embed_norm.weight", reinterpret_cast<void**>(&d_embed_norm_))) { cleanup_stage(); return false; }
    if (!bind_or_upload("model.llm.norm.weight", reinterpret_cast<void**>(&d_final_norm_))) { cleanup_stage(); return false; }
    if (fp8_attn_) {
        if (!bind_or_upload("model.llm.unembed.weight.fp8", reinterpret_cast<void**>(&d_unembed_fp8_))) { cleanup_stage(); return false; }
        if (!bind_or_upload("model.llm.unembed.weight.scale", reinterpret_cast<void**>(&d_unembed_scale_))) { cleanup_stage(); return false; }
    } else {
        if (!bind_or_upload("model.llm.unembed.weight", reinterpret_cast<void**>(&d_unembed_))) { cleanup_stage(); return false; }
    }

    // Pre-scaled AttnRes queries
    d_attn_queries_.resize(cfg_.num_layers - 1, nullptr);
    for (u32 i = 0; i < cfg_.num_layers - 1; ++i) {
        if (!upload_scaled_f32_query("model.llm.attn_res.attn_queries." + std::to_string(i), &d_attn_queries_[i])) {
            cleanup_stage(); return false;
        }
    }
    d_mlp_queries_.resize(cfg_.num_layers, nullptr);
    for (u32 i = 0; i < cfg_.num_layers; ++i) {
        if (!upload_scaled_f32_query("model.llm.attn_res.mlp_queries." + std::to_string(i), &d_mlp_queries_[i])) {
            cleanup_stage(); return false;
        }
    }
    if (!upload_scaled_f32_query("model.llm.attn_res.output_query", &d_output_query_)) {
        cleanup_stage(); return false;
    }

    // Cross-pass message queries: stack (queries[2c], kv[2c], kv[2c+1], queries[2c+1]) * H**-0.5 per core layer
    {
        const auto* tv_q  = get_tv("model.llm.middle_cycle_cross_pass_queries");
        const auto* tv_kv = get_tv("model.llm.middle_cycle_cross_pass_kv_queries");
        if (!tv_q || !tv_kv) { err = "Missing middle_cycle_cross_pass_queries"; cleanup_stage(); return false; }
        std::vector<f32> fq, fkv;
        SafetensorsLoader::read_as_f32(*tv_q, fq);
        SafetensorsLoader::read_as_f32(*tv_kv, fkv);
        u32 n_core = cfg_.ink.core_end - cfg_.ink.core_start;
        d_msg_queries_.resize(n_core, nullptr);
        std::vector<f32> stacked(4 * H);
        for (u32 c = 0; c < n_core; ++c) {
            for (u32 d = 0; d < H; ++d) {
                stacked[0 * H + d] = fq[(2 * c + 0) * H + d] * q_scale_inv_sqrt;  // attn
                stacked[1 * H + d] = fkv[(2 * c + 0) * H + d] * q_scale_inv_sqrt; // k
                stacked[2 * H + d] = fkv[(2 * c + 1) * H + d] * q_scale_inv_sqrt; // v
                stacked[3 * H + d] = fq[(2 * c + 1) * H + d] * q_scale_inv_sqrt;  // mlp
            }
            upload_raw(stacked.data(), stacked.size() * sizeof(f32), reinterpret_cast<void**>(&d_msg_queries_[c]));
        }
    }

    // Load all 24 physical layers
    layers_.resize(cfg_.num_layers);
    for (u32 l = 0; l < cfg_.num_layers; ++l) {
        auto& pl = layers_[l];
        pl.is_local = cfg_.ink.is_local(l);
        std::string pre = "model.llm.layers." + std::to_string(l) + ".";

        if (pl.is_local) {
            if (!bind_or_upload(pre + "attn_norm.weight", reinterpret_cast<void**>(&pl.attn_norm))) { cleanup_stage(); return false; }
        } else {
            // Stack q_input_norm, k_input_norm, v_input_norm into contiguous [3, H] BF16
            const auto* tq = get_tv(pre + "q_input_norm.weight");
            const auto* tk = get_tv(pre + "k_input_norm.weight");
            const auto* tv = get_tv(pre + "v_input_norm.weight");
            if (!tq || !tk || !tv) { err = "Missing global input norms at layer " + std::to_string(l); cleanup_stage(); return false; }
            std::vector<u16> bq, bk, bv, stacked(3 * H);
            SafetensorsLoader::read_as_bf16(*tq, bq);
            SafetensorsLoader::read_as_bf16(*tk, bk);
            SafetensorsLoader::read_as_bf16(*tv, bv);
            std::memcpy(stacked.data() + 0 * H, bq.data(), H * 2);
            std::memcpy(stacked.data() + 1 * H, bk.data(), H * 2);
            std::memcpy(stacked.data() + 2 * H, bv.data(), H * 2);
            upload_raw(stacked.data(), stacked.size() * 2, reinterpret_cast<void**>(&pl.attn_norm));
        }

        if (!bind_or_upload(pre + "mlp_norm.weight", reinterpret_cast<void**>(&pl.mlp_norm))) { cleanup_stage(); return false; }
        if (!bind_or_upload(pre + "attn.attn_sink_proj.weight", reinterpret_cast<void**>(&pl.w_sink))) { cleanup_stage(); return false; }
        if (fp8_attn_) {
            if (!bind_or_upload(pre + "attn.wq_du.weight.fp8",      reinterpret_cast<void**>(&pl.wq_fp8))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.wq_du.weight.scale",    reinterpret_cast<void**>(&pl.sq_fp8))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.wr_du.weight.fp8",      reinterpret_cast<void**>(&pl.wr_fp8))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.wr_du.weight.scale",    reinterpret_cast<void**>(&pl.sr_fp8))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.gate_attn.weight.fp8",  reinterpret_cast<void**>(&pl.wg_fp8))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.gate_attn.weight.scale",reinterpret_cast<void**>(&pl.sg_fp8))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.wk_dv.weight.fp8",      reinterpret_cast<void**>(&pl.wk_fp8))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.wk_dv.weight.scale",    reinterpret_cast<void**>(&pl.sk_fp8))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.wv_dv.weight.fp8",      reinterpret_cast<void**>(&pl.wv_fp8))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.wv_dv.weight.scale",    reinterpret_cast<void**>(&pl.sv_fp8))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.wo_ud.weight.fp8",      reinterpret_cast<void**>(&pl.wo_fp8))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.wo_ud.weight.scale",    reinterpret_cast<void**>(&pl.so_fp8))) { cleanup_stage(); return false; }
        } else {
            if (!bind_or_upload(pre + "attn.wq_du.weight", reinterpret_cast<void**>(&pl.wq))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.wr_du.weight", reinterpret_cast<void**>(&pl.wr))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.gate_attn.weight", reinterpret_cast<void**>(&pl.w_gate))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.wk_dv.weight", reinterpret_cast<void**>(&pl.wk))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.wv_dv.weight", reinterpret_cast<void**>(&pl.wv))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.wo_ud.weight", reinterpret_cast<void**>(&pl.wo))) { cleanup_stage(); return false; }
        }
        if (!bind_or_upload(pre + "attn.q_l2_clip.scale", reinterpret_cast<void**>(&pl.q_scale))) { cleanup_stage(); return false; }
        if (!bind_or_upload(pre + "attn.rel_l2_clip.scale", reinterpret_cast<void**>(&pl.r_scale))) { cleanup_stage(); return false; }

        // Transpose rel_logits_proj.proj [16, 1024] -> [1024, 16] for coalesced L1-cached reads
        {
            const auto* tr = get_tv(pre + "attn.rel_logits_proj.proj");
            if (!tr) { err = "Missing rel_logits_proj at layer " + std::to_string(l); cleanup_stage(); return false; }
            std::vector<u16> raw_r;
            SafetensorsLoader::read_as_bf16(*tr, raw_r);
            u32 d_rel = cfg_.ink.d_rel;
            u32 ext   = cfg_.ink.rel_extent;
            std::vector<u16> trans(static_cast<size_t>(ext) * d_rel);
            for (u32 j = 0; j < d_rel; ++j) {
                for (u32 d = 0; d < ext; ++d) {
                    trans[static_cast<size_t>(d) * d_rel + j] = raw_r[static_cast<size_t>(j) * ext + d];
                }
            }
            upload_raw(trans.data(), trans.size() * 2, reinterpret_cast<void**>(&pl.rel_proj_T));
        }

        if (pl.is_local) {
            if (!bind_or_upload(pre + "attn.k_sconv.weight", reinterpret_cast<void**>(&pl.k_sconv))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.v_sconv.weight", reinterpret_cast<void**>(&pl.v_sconv))) { cleanup_stage(); return false; }
        } else {
            if (!bind_or_upload(pre + "attn.k_sconv.static_w", reinterpret_cast<void**>(&pl.k_sconv))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.v_sconv.static_w", reinterpret_cast<void**>(&pl.v_sconv))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.k_sconv.weight_proj", reinterpret_cast<void**>(&pl.k_sconv_proj))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "attn.v_sconv.weight_proj", reinterpret_cast<void**>(&pl.v_sconv_proj))) { cleanup_stage(); return false; }
        }
        if (!bind_or_upload(pre + "attn_sconv.weight", reinterpret_cast<void**>(&pl.attn_sconv))) { cleanup_stage(); return false; }
        if (!bind_or_upload(pre + "mlp_sconv.weight", reinterpret_cast<void**>(&pl.mlp_sconv))) { cleanup_stage(); return false; }

        // MoE router + NVFP4 (L0..19) or MXFP8 (L20..23) experts
        if (fp8_attn_ && get_tv(pre + "mlp.gate.weight.fp8") != nullptr) {
            if (!bind_or_upload(pre + "mlp.gate.weight.fp8", reinterpret_cast<void**>(&pl.moe_gate_fp8))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "mlp.gate.weight.scale", reinterpret_cast<void**>(&pl.moe_gate_scale))) { cleanup_stage(); return false; }
        } else {
            if (!bind_or_upload(pre + "mlp.gate.weight", reinterpret_cast<void**>(&pl.moe_gate_w))) { cleanup_stage(); return false; }
        }
        if (!bind_or_upload(pre + "mlp.gate.bias", reinterpret_cast<void**>(&pl.moe_gate_bias))) { cleanup_stage(); return false; }
        if (!bind_or_upload(pre + "mlp.experts.w13_weight", reinterpret_cast<void**>(&pl.w13_weight))) { cleanup_stage(); return false; }
        if (!bind_or_upload(pre + "mlp.experts.w13_weight_scale", reinterpret_cast<void**>(&pl.w13_scale))) { cleanup_stage(); return false; }
        if (!bind_or_upload(pre + "mlp.experts.w2_weight", reinterpret_cast<void**>(&pl.w2_weight))) { cleanup_stage(); return false; }
        if (!bind_or_upload(pre + "mlp.experts.w2_weight_scale", reinterpret_cast<void**>(&pl.w2_scale))) { cleanup_stage(); return false; }
        const auto* tv_w13 = get_tv(pre + "mlp.experts.w13_weight");
        pl.is_mxfp8 = (tv_w13 && tv_w13->dtype == DType::FP8_E4M3);
        if (!pl.is_mxfp8) {
            if (!bind_or_upload(pre + "mlp.experts.w13_weight_amax", reinterpret_cast<void**>(&pl.w13_amax))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "mlp.experts.w13_weight_outer_scale_denominator", reinterpret_cast<void**>(&pl.w13_outer_denom))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "mlp.experts.w2_weight_amax", reinterpret_cast<void**>(&pl.w2_amax))) { cleanup_stage(); return false; }
            if (!bind_or_upload(pre + "mlp.experts.w2_weight_outer_scale_denominator", reinterpret_cast<void**>(&pl.w2_outer_denom))) { cleanup_stage(); return false; }
        }

        if (l >= cfg_.ink.core_start && l < cfg_.ink.core_end) {
            pl.has_controllers = true;
            const char* cnames[4] = {"attn", "k", "v", "mlp"};
            for (int c = 0; c < 4; ++c) {
                std::string cp = pre + "controllers." + cnames[c] + ".";
                if (!bind_or_upload(cp + "down_weight", reinterpret_cast<void**>(&pl.ctrl[c].down))) { cleanup_stage(); return false; }
                if (!bind_or_upload(cp + "gain_weight", reinterpret_cast<void**>(&pl.ctrl[c].gain))) { cleanup_stage(); return false; }
                if (!bind_or_upload(cp + "forget_weight", reinterpret_cast<void**>(&pl.ctrl[c].forget))) { cleanup_stage(); return false; }
            }
        }
    }

    // Load Built-In MTP (DSpark 3-Layer + Rank-256 Markov Head) when --spec mtp is requested
    if (spec_mode_ == SpecMode::Mtp && get_tv("mtp.fuse_weight") != nullptr) {
        bool ok_mtp = true;
        auto bm = [&](const std::string& n, void** dst) {
            if (!ok_mtp) return;
            if (!bind_or_upload("mtp." + n, dst, true)) ok_mtp = false;
        };
        bm("mask_embed", reinterpret_cast<void**>(&d_mtp_mask_embed_));
        bm("norm.weight", reinterpret_cast<void**>(&d_mtp_norm_));
        bm("markov_head.dflare_norm.weight", reinterpret_cast<void**>(&d_mtp_mh_norm_));
        bm("markov_head.markov_w1.weight", reinterpret_cast<void**>(&d_mtp_markov_w1_));
        bm("markov_head.rnn_proj.weight", reinterpret_cast<void**>(&d_mtp_rnn_w_));
        bm("markov_head.rnn_proj.bias", reinterpret_cast<void**>(&d_mtp_rnn_b_));
        bm("markov_head.hidden_out.weight", reinterpret_cast<void**>(&d_mtp_hout_w_));
        bm("markov_head.hidden_out.bias", reinterpret_cast<void**>(&d_mtp_hout_b_));
        mtp_fp8_ = (get_tv("mtp.layers.0.self_attn.q_proj.weight.fp8") != nullptr);
        for (int l = 0; l < 3; ++l) {
            auto& ml = mtp_layers_[l];
            ml.is_local = (l == 0 || l == 2); // config.json mtp_config.local_layer_ids = [0, 2]
            std::string lp = "layers." + std::to_string(l) + ".";
            bm(lp + "dflare_norm.weight", reinterpret_cast<void**>(&ml.dflare_norm));
            bm(lp + "input_layernorm.weight", reinterpret_cast<void**>(&ml.input_norm));
            bm(lp + "post_attention_layernorm.weight", reinterpret_cast<void**>(&ml.post_attn_norm));
            bm(lp + "self_attn.q_norm.weight", reinterpret_cast<void**>(&ml.q_norm));
            bm(lp + "self_attn.k_norm.weight", reinterpret_cast<void**>(&ml.k_norm));
            bm(lp + "self_attn.k_norm_ctx.weight", reinterpret_cast<void**>(&ml.k_norm_ctx));
            if (mtp_fp8_) {
                bm(lp + "self_attn.q_proj.weight.fp8",      reinterpret_cast<void**>(&ml.wq_fp8));
                bm(lp + "self_attn.q_proj.weight.scale",    reinterpret_cast<void**>(&ml.sq_fp8));
                bm(lp + "self_attn.k_proj.weight.fp8",      reinterpret_cast<void**>(&ml.wk_fp8));
                bm(lp + "self_attn.k_proj.weight.scale",    reinterpret_cast<void**>(&ml.sk_fp8));
                bm(lp + "self_attn.v_proj.weight.fp8",      reinterpret_cast<void**>(&ml.wv_fp8));
                bm(lp + "self_attn.v_proj.weight.scale",    reinterpret_cast<void**>(&ml.sv_fp8));
                bm(lp + "self_attn.k_proj_ctx.weight.fp8",  reinterpret_cast<void**>(&ml.wk_ctx_fp8));
                bm(lp + "self_attn.k_proj_ctx.weight.scale",reinterpret_cast<void**>(&ml.sk_ctx_fp8));
                bm(lp + "self_attn.v_proj_ctx.weight.fp8",  reinterpret_cast<void**>(&ml.wv_ctx_fp8));
                bm(lp + "self_attn.v_proj_ctx.weight.scale",reinterpret_cast<void**>(&ml.sv_ctx_fp8));
                bm(lp + "self_attn.o_proj.weight.fp8",      reinterpret_cast<void**>(&ml.wo_fp8));
                bm(lp + "self_attn.o_proj.weight.scale",    reinterpret_cast<void**>(&ml.so_fp8));
                bm(lp + "mlp.gate_proj.weight.fp8",         reinterpret_cast<void**>(&ml.w_gate_fp8));
                bm(lp + "mlp.gate_proj.weight.scale",       reinterpret_cast<void**>(&ml.s_gate_fp8));
                bm(lp + "mlp.up_proj.weight.fp8",           reinterpret_cast<void**>(&ml.w_up_fp8));
                bm(lp + "mlp.up_proj.weight.scale",         reinterpret_cast<void**>(&ml.s_up_fp8));
                bm(lp + "mlp.down_proj.weight.fp8",         reinterpret_cast<void**>(&ml.w_down_fp8));
                bm(lp + "mlp.down_proj.weight.scale",       reinterpret_cast<void**>(&ml.s_down_fp8));
            } else {
                bm(lp + "self_attn.q_proj.weight", reinterpret_cast<void**>(&ml.wq));
                bm(lp + "self_attn.k_proj.weight", reinterpret_cast<void**>(&ml.wk));
                bm(lp + "self_attn.v_proj.weight", reinterpret_cast<void**>(&ml.wv));
                bm(lp + "self_attn.k_proj_ctx.weight", reinterpret_cast<void**>(&ml.wk_ctx));
                bm(lp + "self_attn.v_proj_ctx.weight", reinterpret_cast<void**>(&ml.wv_ctx));
                bm(lp + "self_attn.o_proj.weight", reinterpret_cast<void**>(&ml.wo));
                bm(lp + "mlp.gate_proj.weight", reinterpret_cast<void**>(&ml.w_gate));
                bm(lp + "mlp.up_proj.weight", reinterpret_cast<void**>(&ml.w_up));
                bm(lp + "mlp.down_proj.weight", reinterpret_cast<void**>(&ml.w_down));
            }
        }
        if (ok_mtp) {
            std::vector<f32> fw, alpha(16);
            SafetensorsLoader::read_as_f32(*get_tv("mtp.fuse_weight"), fw);
            for (int r = 0; r < 4; ++r) {
                f32 mx = -1e30f;
                for (int c = 0; c < 4; ++c) mx = std::max(mx, fw[r * 4 + c]);
                f32 sm = 0.0f;
                for (int c = 0; c < 4; ++c) { alpha[r * 4 + c] = std::exp(fw[r * 4 + c] - mx); sm += alpha[r * 4 + c]; }
                for (int c = 0; c < 4; ++c) alpha[r * 4 + c] /= sm;
            }
            upload_raw(alpha.data(), alpha.size() * sizeof(f32), reinterpret_cast<void**>(&d_mtp_alpha_));
            std::vector<f32> invf(64);
            for (u32 i = 0; i < 64; ++i) invf[i] = static_cast<f32>(1.0 / std::pow(cfg_.ink.mtp_rope_theta, (2.0 * i) / 128.0));
            upload_raw(invf.data(), invf.size() * sizeof(f32), reinterpret_cast<void**>(&d_mtp_inv_freq_));
            mtp_loaded_ = true;
        }
    }
    cleanup_stage();

    // Build the exact 48 logical layer visits: prelude [0..6), core [6..18) x 3, coda [18..24)
    visits_.clear();
    for (u32 i = 0; i < cfg_.ink.core_start; ++i) visits_.push_back({i, 0, cfg_.ink.is_local(i)});
    for (u32 p = 0; p < cfg_.ink.num_passes; ++p) {
        for (u32 i = cfg_.ink.core_start; i < cfg_.ink.core_end; ++i) {
            visits_.push_back({i, p, cfg_.ink.is_local(i)});
        }
    }
    for (u32 i = cfg_.ink.core_end; i < cfg_.num_layers; ++i) visits_.push_back({i, 0, cfg_.ink.is_local(i)});

    // Exact honest active-parameter bytes per decode step (ONLY 8/128 routed experts per visit + BF16/FP8 attn/ctrl + unembed, NEVER embed.weight)
    u64 bpe_attn = fp8_attn_ ? 1ULL : 2ULL;
    u64 per_visit_attn_bytes =
        (static_cast<u64>(cfg_.q_dim()) * H + static_cast<u64>(cfg_.num_q_heads * cfg_.ink.d_rel) * H +
         static_cast<u64>(cfg_.q_dim()) * H +
         static_cast<u64>(2 * cfg_.kv_dim()) * H + static_cast<u64>(H) * cfg_.q_dim()) * bpe_attn +
        static_cast<u64>(cfg_.num_q_heads) * H * 2ULL;
    u64 per_visit_moe_active_bytes =
        static_cast<u64>(cfg_.ink.n_routed_experts) * H * 2ULL + // router BF16
        static_cast<u64>(cfg_.ink.num_experts_per_tok) * (
            static_cast<u64>(2 * cfg_.intermediate_size) * (H / 2 + H / 16) +
            static_cast<u64>(H) * (cfg_.intermediate_size / 2 + cfg_.intermediate_size / 16)
        );
    u64 per_visit_moe_mxfp8_bytes =
        static_cast<u64>(cfg_.ink.n_routed_experts) * H * 2ULL +
        static_cast<u64>(cfg_.ink.num_experts_per_tok) * (
            static_cast<u64>(2 * cfg_.intermediate_size) * (H + H / 32) +
            static_cast<u64>(H) * (cfg_.intermediate_size + cfg_.intermediate_size / 32)
        );
    u64 per_ctrl_visit_bytes = 4ULL * (static_cast<u64>(cfg_.ink.gate_rank) * (2 * H) + 2ULL * H * cfg_.ink.gate_rank) * 2ULL;
    active_weight_bytes_ = static_cast<u64>(cfg_.vocab_size) * H * bpe_attn; // unembed.weight (BF16 or FP8)
    for (const auto& v : visits_) {
        active_weight_bytes_ += per_visit_attn_bytes + (layers_[v.phys_layer].is_mxfp8 ? per_visit_moe_mxfp8_bytes : per_visit_moe_active_bytes);
        if (v.pass_idx > 0) active_weight_bytes_ += per_ctrl_visit_bytes;
    }

    // Allocate KV caches:
    // - 40 Local SWA visits: tiny 1024-token ring buffers (INT8 = 85 MiB total)
    // - 8 Global visits: paged block pool
    // * WHT4 (--kv-wht4): WHT-K4V4 (64B K + 64B V + 4B BF16 scales = 132 B/head/tok -> 8.25 GiB @ 1,048,576)
    // * WHT84 (--kv-wht84): WHT-K8V4 (128B K + 64B V + 4B BF16 scales = 196 B/head/tok -> 12.25 GiB @ 1,048,576)
    max_seq_ = std::min(opts.max_seq_len, cfg_.max_position_embeddings);
    u32 base_blocks = (max_seq_ + kKvBlockSize - 1) / kKvBlockSize;
    u32 num_blocks = opts.max_kv_blocks > 0 ? opts.max_kv_blocks : (max_seq_ >= 262144 ? base_blocks : (base_blocks + 32));
    kv_mgr_.init(num_blocks);

    const bool global_wht = (kv_mode_ == KvQuantMode::WHT4 || kv_mode_ == KvQuantMode::WHT84);
    const bool global_k4  = (kv_mode_ == KvQuantMode::WHT4);
    size_t local_kv_eb    = kv_int8_ ? sizeof(u8) : sizeof(__nv_bfloat16);
    size_t local_scale_n  = static_cast<size_t>(cfg_.num_kv_heads) * cfg_.ink.sliding_window_size;
    size_t global_scale_n = static_cast<size_t>(num_blocks) * cfg_.num_kv_heads * kKvBlockSize;

    for (auto& v : visits_) {
        if (v.is_local) {
            INK_CUDA_CHECK(cudaMalloc(&v.d_k_cache, local_scale_n * cfg_.head_dim * local_kv_eb));
            INK_CUDA_CHECK(cudaMalloc(&v.d_v_cache, local_scale_n * cfg_.head_dim * local_kv_eb));
            if (kv_int8_) {
                INK_CUDA_CHECK(cudaMalloc(&v.d_k_scale, local_scale_n * sizeof(f32)));
                INK_CUDA_CHECK(cudaMalloc(&v.d_v_scale, local_scale_n * sizeof(f32)));
            }
        } else if (global_wht) {
            size_t k_bph = global_k4 ? (cfg_.head_dim / 2) : cfg_.head_dim;
            size_t v_bph = cfg_.head_dim / 2;
            INK_CUDA_CHECK(cudaMalloc(&v.d_k_cache, global_scale_n * k_bph));
            INK_CUDA_CHECK(cudaMalloc(&v.d_v_cache, global_scale_n * v_bph));
            // BF16 per-head scales (2 B each instead of 4 B f32 -> saves 256 MiB @ 1,048,576 tokens)
            INK_CUDA_CHECK(cudaMalloc(&v.d_k_scale, global_scale_n * sizeof(__nv_bfloat16)));
            INK_CUDA_CHECK(cudaMalloc(&v.d_v_scale, global_scale_n * sizeof(__nv_bfloat16)));
        } else {
            INK_CUDA_CHECK(cudaMalloc(&v.d_k_cache, global_scale_n * cfg_.head_dim * local_kv_eb));
            INK_CUDA_CHECK(cudaMalloc(&v.d_v_cache, global_scale_n * cfg_.head_dim * local_kv_eb));
            if (kv_int8_) {
                INK_CUDA_CHECK(cudaMalloc(&v.d_k_scale, global_scale_n * sizeof(f32)));
                INK_CUDA_CHECK(cudaMalloc(&v.d_v_scale, global_scale_n * sizeof(f32)));
            }
        }
        INK_CUDA_CHECK(cudaMalloc(&v.d_k_hist,    16ULL * cfg_.kv_dim() * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&v.d_v_hist,    16ULL * cfg_.kv_dim() * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&v.d_attn_hist, 16ULL * H * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&v.d_mlp_hist,  16ULL * H * sizeof(__nv_bfloat16)));
    }
    INK_CUDA_CHECK(cudaMalloc(&d_block_table_, (static_cast<size_t>(num_blocks) + 64) * sizeof(i32)));

    // VRAM-adaptive prefill chunk size via cudaMemGetInfo
    // >= 3.0 GiB free (24 GB up to 512k, 32 GB up to 1M): 1024-token chunks
    // >= 750 MiB free (24 GB RTX 4090 @ 1,048,576 tokens): 512-token chunks (2x faster 1M prefill, ~600 MiB headroom)
    // < 750 MiB free (16 GB RTX 4060 Ti): 256-token chunks
    size_t free_vram = 0, total_vram = 0;
    cudaMemGetInfo(&free_vram, &total_vram);
    if (free_vram >= 3ULL * 1024 * 1024 * 1024) max_batch_tokens_ = 1024u;
    else if (free_vram >= 750ULL * 1024 * 1024) max_batch_tokens_ = 512u;
    else max_batch_tokens_ = 256u;
    const u32 M = max_batch_tokens_;
    for (int i = 0; i < 5; ++i) INK_CUDA_CHECK(cudaMalloc(&d_stack_[i], static_cast<size_t>(M) * H * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_prelude_partial_, static_cast<size_t>(M) * H * sizeof(__nv_bfloat16)));
    for (int i = 0; i < 2; ++i) INK_CUDA_CHECK(cudaMalloc(&d_bank_[i], static_cast<size_t>(M) * H * sizeof(__nv_bfloat16)));

    // Device pointer table: [0..4] = d_stack_[0..4], [5..6] = d_bank_[0..1]
    INK_CUDA_CHECK(cudaMalloc(&d_ptr_table_, 8 * sizeof(__nv_bfloat16*)));
    const __nv_bfloat16* h_ptrs[8] = {d_stack_[0], d_stack_[1], d_stack_[2], d_stack_[3], d_stack_[4], d_bank_[0], d_bank_[1], nullptr};
    INK_CUDA_CHECK(cudaMemcpyAsync(d_ptr_table_, h_ptrs, 8 * sizeof(__nv_bfloat16*), cudaMemcpyHostToDevice, stream_));

    INK_CUDA_CHECK(cudaMalloc(&d_mix_f32_,           3ULL * M * H * sizeof(f32)));
    INK_CUDA_CHECK(cudaMalloc(&d_prelude_dests_f32_, 3ULL * M * H * sizeof(f32)));
    INK_CUDA_CHECK(cudaMalloc(&d_msg_f32_,           4ULL * M * H * sizeof(f32)));
    INK_CUDA_CHECK(cudaMalloc(&d_pair_bf16_,         3ULL * M * (2 * H) * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_ctrl_h_,            3ULL * M * cfg_.ink.gate_rank * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_ctrl_gf_,           3ULL * M * (2 * H) * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_attn_in_,           3ULL * M * H * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_mlp_in_,            1ULL * M * H * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_q_raw_,             static_cast<size_t>(M) * cfg_.q_dim() * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_q_clip_,            static_cast<size_t>(M) * cfg_.q_dim() * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_r_raw_,             static_cast<size_t>(M) * cfg_.num_q_heads * cfg_.ink.d_rel * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_r_clip_,            static_cast<size_t>(M) * cfg_.num_q_heads * cfg_.ink.d_rel * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_gate_,              static_cast<size_t>(M) * cfg_.q_dim() * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_sink_,              static_cast<size_t>(M) * cfg_.num_q_heads * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_k_raw_,             static_cast<size_t>(M) * cfg_.kv_dim() * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_v_raw_,             static_cast<size_t>(M) * cfg_.kv_dim() * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_k_dyn_,             static_cast<size_t>(M) * cfg_.num_kv_heads * 4 * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_v_dyn_,             static_cast<size_t>(M) * cfg_.num_kv_heads * 4 * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_attn_out_,          static_cast<size_t>(M) * cfg_.q_dim() * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_proj_out_,          static_cast<size_t>(M) * H * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_attn_conv_,         static_cast<size_t>(M) * H * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_moe_out_,           static_cast<size_t>(M) * H * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_moe_act_,           static_cast<size_t>(M) * cfg_.ink.num_experts_per_tok * cfg_.intermediate_size * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_router_logits_,     static_cast<size_t>(M) * cfg_.ink.n_routed_experts * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_moe_dequant_,       static_cast<size_t>(std::max<u32>(2 * cfg_.intermediate_size, cfg_.q_dim())) * std::max<u32>(H, cfg_.q_dim()) * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_moe_gather_x_,      static_cast<size_t>(M) * H * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_moe_w13_out_,       static_cast<size_t>(M) * (2 * cfg_.intermediate_size) * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_moe_act_b_,         static_cast<size_t>(M) * cfg_.intermediate_size * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_moe_y_b_,           static_cast<size_t>(M) * H * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_moe_out_f32_,       static_cast<size_t>(M) * H * sizeof(f32)));
    INK_CUDA_CHECK(cudaMalloc(&d_moe_tok_idx_,       static_cast<size_t>(M) * sizeof(i32)));
    INK_CUDA_CHECK(cudaMalloc(&d_moe_gammas_,        static_cast<size_t>(M) * sizeof(f32)));
    INK_CUDA_CHECK(cudaMalloc(&d_expert_ids_,        static_cast<size_t>(M) * cfg_.ink.num_experts_per_tok * sizeof(i32)));
    INK_CUDA_CHECK(cudaMalloc(&d_expert_weights_,    static_cast<size_t>(M) * cfg_.ink.num_experts_per_tok * sizeof(f32)));
    INK_CUDA_CHECK(cudaMalloc(&d_uniq_e_,            64 * sizeof(u32)));
    INK_CUDA_CHECK(cudaMalloc(&d_uniq_mask_,         64 * sizeof(u64)));
    INK_CUDA_CHECK(cudaMalloc(&d_uniq_count_,        sizeof(u32)));
    const u32 decode_m_cap = (spec_mode_ == SpecMode::Mtp || spec_mode_ == SpecMode::Ngram) ? 8u : 1u;
    INK_CUDA_CHECK(cudaMalloc(&d_logits_, static_cast<size_t>(decode_m_cap) * cfg_.vocab_size * sizeof(__nv_bfloat16)));
    splitk_max_ = (decode_m_cap == 1u) ? 256u : 64u;
    INK_CUDA_CHECK(cudaMalloc(&d_split_o_,  static_cast<size_t>(decode_m_cap) * cfg_.num_q_heads * splitk_max_ * cfg_.head_dim * sizeof(f32)));
    INK_CUDA_CHECK(cudaMalloc(&d_split_ml_, static_cast<size_t>(decode_m_cap) * cfg_.num_q_heads * splitk_max_ * 2 * sizeof(f32)));

    // Hardware L2 Cache Pinning (cudaAccessPolicyWindow) for 3x-reused Looped Core weights
    void* core_pin_ptr = layers_.size() > cfg_.ink.core_start
        ? (fp8_attn_ ? static_cast<void*>(layers_[cfg_.ink.core_start].wq_fp8)
                     : static_cast<void*>(layers_[cfg_.ink.core_start].wq))
        : nullptr;
    if (d_weight_slab_ && prop.persistingL2CacheMaxSize > 0 && core_pin_ptr) {
        size_t pin_bytes = std::min<size_t>(static_cast<size_t>(prop.persistingL2CacheMaxSize), 48ULL * 1024 * 1024);
        if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, pin_bytes) == cudaSuccess) {
            cudaStreamAttrValue attr{};
            attr.accessPolicyWindow.base_ptr  = core_pin_ptr;
            attr.accessPolicyWindow.num_bytes = pin_bytes;
            attr.accessPolicyWindow.hitRatio  = 1.0f;
            attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(stream_, cudaStreamAttributeAccessPolicyWindow, &attr);
        }
    }

    INK_CUDA_CHECK(cudaMalloc(&d_token_ids_,  M * sizeof(i32)));
    INK_CUDA_CHECK(cudaMalloc(&d_out_tokens_, 16 * sizeof(i32)));
    INK_CUDA_CHECK(cudaMalloc(&d_decode_pos_, sizeof(i32)));
    INK_CUDA_CHECK(cudaHostAlloc(&h_pinned_token_, 16 * sizeof(i32), cudaHostAllocDefault));

    // Native FP8 cublasLtMatmul workspace (~4 MiB) + Pinned MoE Prefill Buffers (eliminates 12,288 pageable H2D syncs/chunk)
    cublasLtCreate(&cublaslt_);
    u32 max_fp8_k = std::max<u32>({H, cfg_.q_dim(), 2 * cfg_.intermediate_size});
    u32 max_fp8_n = std::max<u32>({H, cfg_.q_dim(), 2 * cfg_.intermediate_size});
    INK_CUDA_CHECK(cudaMalloc(&d_fp8_x_,          static_cast<size_t>(M + 16) * max_fp8_k));
    INK_CUDA_CHECK(cudaMalloc(&d_fp8_scale_,      3 * sizeof(f32)));
    INK_CUDA_CHECK(cudaMalloc(&d_fp8_out_pad_,    static_cast<size_t>(M + 16) * max_fp8_n * sizeof(__nv_bfloat16)));
    INK_CUDA_CHECK(cudaMalloc(&d_moe_fp8_scales_, static_cast<size_t>(max_fp8_n) * sizeof(f32)));
    f32 one_f = 1.0f;
    INK_CUDA_CHECK(cudaMemcpyAsync(d_fp8_scale_ + 2, &one_f, sizeof(f32), cudaMemcpyHostToDevice, stream_));
    cublaslt_ws_bytes_ = 4ULL * 1024 * 1024;
    INK_CUDA_CHECK(cudaMalloc(&d_cublaslt_ws_, cublaslt_ws_bytes_));
    size_t moe_slots = static_cast<size_t>(M) * cfg_.ink.num_experts_per_tok;
    INK_CUDA_CHECK(cudaHostAlloc(&h_moe_pinned_ids_, moe_slots * sizeof(i32), cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaHostAlloc(&h_moe_pinned_wts_, moe_slots * sizeof(f32), cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaHostAlloc(&h_moe_pack_idx_,   moe_slots * sizeof(i32), cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaHostAlloc(&h_moe_pack_gam_,   moe_slots * sizeof(f32), cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaMalloc(&d_moe_pack_idx_,      moe_slots * sizeof(i32)));
    INK_CUDA_CHECK(cudaMalloc(&d_moe_pack_gam_,      moe_slots * sizeof(f32)));
#if defined(FLINT_HAS_FP4_TC) && FLINT_HAS_FP4_TC == 1
    size_t max_tiles_alloc = cfg_.ink.n_routed_experts + moe_slots / 16 + 16;
    INK_CUDA_CHECK(cudaMalloc(&d_tile_meta_,  max_tiles_alloc * sizeof(u32)));
    INK_CUDA_CHECK(cudaMalloc(&d_tile_cnt_,   max_tiles_alloc * sizeof(u32)));
    INK_CUDA_CHECK(cudaMalloc(&d_num_tiles_,  sizeof(u32)));
    INK_CUDA_CHECK(cudaMalloc(&d_x_fp4_,      static_cast<size_t>(M) * (H >> 1)));
    INK_CUDA_CHECK(cudaMalloc(&d_x_fp4_sc_,   static_cast<size_t>(M) * (H >> 4)));
    INK_CUDA_CHECK(cudaMalloc(&d_x_fp4_gs_,   static_cast<size_t>(M) * sizeof(f32)));
    INK_CUDA_CHECK(cudaMalloc(&d_act_fp4_,    moe_slots * (cfg_.intermediate_size >> 1)));
    INK_CUDA_CHECK(cudaMalloc(&d_act_fp4_sc_, moe_slots * (cfg_.intermediate_size >> 4)));
    INK_CUDA_CHECK(cudaMalloc(&d_act_fp4_gs_, moe_slots * sizeof(f32)));
#endif

    if (mtp_loaded_) {
        const u32 rank = cfg_.ink.mtp_markov_rank; // 256
        const u32 B    = cfg_.ink.mtp_block_size;  // 8
        const u32 V    = cfg_.vocab_size;          // 201024
        mtp_max_seq_   = std::min<u32>(max_seq_ + 64, 18432);
        for (int l = 0; l < 3; ++l) {
            size_t kv_sz = static_cast<size_t>(mtp_max_seq_) * cfg_.kv_dim() * sizeof(__nv_bfloat16);
            INK_CUDA_CHECK(cudaMalloc(&mtp_layers_[l].d_k_ctx, kv_sz));
            INK_CUDA_CHECK(cudaMalloc(&mtp_layers_[l].d_v_ctx, kv_sz));
        }
        for (int f = 0; f < 4; ++f) {
            INK_CUDA_CHECK(cudaMalloc(&d_mtp_feats_[f], static_cast<size_t>(M) * H * sizeof(__nv_bfloat16)));
            INK_CUDA_CHECK(cudaMalloc(&d_mtp_last_feats_[f], static_cast<size_t>(H) * sizeof(__nv_bfloat16)));
        }
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_feat_ptrs_, 8 * sizeof(__nv_bfloat16*)));
        const __nv_bfloat16* h_mp[8] = {
            d_mtp_feats_[0], d_mtp_feats_[1], d_mtp_feats_[2], d_mtp_feats_[3],
            d_mtp_last_feats_[0], d_mtp_last_feats_[1], d_mtp_last_feats_[2], d_mtp_last_feats_[3]
        };
        INK_CUDA_CHECK(cudaMemcpyAsync(d_mtp_feat_ptrs_, h_mp, 8 * sizeof(__nv_bfloat16*), cudaMemcpyHostToDevice, stream_));

        INK_CUDA_CHECK(cudaMalloc(&d_mtp_x_,       static_cast<size_t>(B) * H * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_norm_x_,  static_cast<size_t>(M) * H * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_q_,       static_cast<size_t>(B) * cfg_.q_dim() * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_k_,       static_cast<size_t>(B) * cfg_.kv_dim() * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_v_,       static_cast<size_t>(B) * cfg_.kv_dim() * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_attn_,    static_cast<size_t>(B) * cfg_.q_dim() * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_mlp_g_,   static_cast<size_t>(B) * cfg_.ink.mtp_intermediate_size * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_mlp_u_,   static_cast<size_t>(B) * cfg_.ink.mtp_intermediate_size * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_f_rnn_,   static_cast<size_t>(H) * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_state_,   static_cast<size_t>(rank) * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_latent_,  static_cast<size_t>(rank) * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_rnn_in_,  static_cast<size_t>(2 * rank + 2 * H) * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_raw3r_,   static_cast<size_t>(3 * rank) * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_rnn_ctx_f32_, static_cast<size_t>(B - 1) * (3 * rank) * sizeof(f32)));
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_bias_,    static_cast<size_t>(V) * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_logits_tmp_, static_cast<size_t>(V) * sizeof(__nv_bfloat16)));

        // Pre-fold hidden_out into unembed once on GPU (works for both BF16 & FP8 unembed):
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_fold_w_, static_cast<size_t>(V) * rank * sizeof(__nv_bfloat16)));
        INK_CUDA_CHECK(cudaMalloc(&d_mtp_fold_b_, static_cast<size_t>(V) * sizeof(__nv_bfloat16)));
        f32 a1 = 1.0f, b0 = 0.0f;
        if (fp8_attn_) {
            // Chunked FP8 unembed dequant into d_moe_dequant_ (~18 MB) -> zero extra VRAM
            const u32 chunk_r = 2048;
            for (u32 r0 = 0; r0 < V; r0 += chunk_r) {
                u32 nr = std::min<u32>(chunk_r, V - r0);
                cuda::launch_dequant_fp8_to_bf16(d_unembed_fp8_ + static_cast<size_t>(r0) * H,
                                                 d_unembed_scale_ + r0, d_moe_dequant_, nr, H, stream_);
                cublasGemmEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                             static_cast<int>(rank), static_cast<int>(nr), static_cast<int>(H),
                             &a1, d_mtp_hout_w_, CUDA_R_16BF, static_cast<int>(rank),
                             d_moe_dequant_, CUDA_R_16BF, static_cast<int>(H),
                             &b0, d_mtp_fold_w_ + static_cast<size_t>(r0) * rank, CUDA_R_16BF, static_cast<int>(rank),
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }
            cuda::launch_gemv_fp8(d_mtp_hout_b_, d_unembed_fp8_, d_unembed_scale_, nullptr, d_mtp_fold_b_, 1, V, H, stream_);
        } else {
            cublasGemmEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                         static_cast<int>(rank), static_cast<int>(V), static_cast<int>(H),
                         &a1, d_mtp_hout_w_, CUDA_R_16BF, static_cast<int>(rank),
                         d_unembed_, CUDA_R_16BF, static_cast<int>(H),
                         &b0, d_mtp_fold_w_, CUDA_R_16BF, static_cast<int>(rank),
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cuda::launch_gemv_bf16(d_mtp_hout_b_, d_unembed_, nullptr, d_mtp_fold_b_, 1, V, H, stream_);
        }
        // Quantize d_mtp_fold_w_ to per-row FP8 E4M3 -> -51 MiB VRAM + 2x faster 7-step Markov head
        if (fp8_attn_) {
            INK_CUDA_CHECK(cudaMalloc(&d_mtp_fold_fp8_,   static_cast<size_t>(V) * rank));
            INK_CUDA_CHECK(cudaMalloc(&d_mtp_fold_scale_, static_cast<size_t>(V) * sizeof(f32)));
            cuda::launch_quant_bf16_to_fp8_rowwise(d_mtp_fold_w_, d_mtp_fold_fp8_, d_mtp_fold_scale_, V, rank, stream_);
            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            cudaFree(d_mtp_fold_w_);
            d_mtp_fold_w_ = nullptr;
        }
    }

    clear_kv_cache();
    INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
    return true;
}

void Inkling2Engine::save_prefix_snapshot(std::span<const i32> prefix_ids, const std::vector<i32>& bt) {
    if (prefix_ids.size() < 256) return;
    const u32 H = cfg_.hidden_size;
    size_t kv_eb = kv_int8_ ? sizeof(u8) : sizeof(__nv_bfloat16);
    size_t local_sn = static_cast<size_t>(cfg_.num_kv_heads) * cfg_.ink.sliding_window_size;
    if (!prefix_snap_alloc_) {
        prefix_snap_.resize(visits_.size());
        for (size_t v = 0; v < visits_.size(); ++v) {
            auto& ps = prefix_snap_[v];
            if (visits_[v].is_local) {
                INK_CUDA_CHECK(cudaMalloc(&ps.d_k_snap, local_sn * cfg_.head_dim * kv_eb));
                INK_CUDA_CHECK(cudaMalloc(&ps.d_v_snap, local_sn * cfg_.head_dim * kv_eb));
                if (kv_int8_) {
                    INK_CUDA_CHECK(cudaMalloc(&ps.d_ks_snap, local_sn * sizeof(f32)));
                    INK_CUDA_CHECK(cudaMalloc(&ps.d_vs_snap, local_sn * sizeof(f32)));
                }
            }
            INK_CUDA_CHECK(cudaMalloc(&ps.d_kh_snap, 16ULL * cfg_.kv_dim() * sizeof(__nv_bfloat16)));
            INK_CUDA_CHECK(cudaMalloc(&ps.d_vh_snap, 16ULL * cfg_.kv_dim() * sizeof(__nv_bfloat16)));
            INK_CUDA_CHECK(cudaMalloc(&ps.d_ah_snap, 16ULL * H * sizeof(__nv_bfloat16)));
            INK_CUDA_CHECK(cudaMalloc(&ps.d_mh_snap, 16ULL * H * sizeof(__nv_bfloat16)));
        }
        prefix_snap_alloc_ = true;
    }
    for (size_t v = 0; v < visits_.size(); ++v) {
        const auto& vs = visits_[v];
        auto& ps = prefix_snap_[v];
        if (vs.is_local) {
            INK_CUDA_CHECK(cudaMemcpyAsync(ps.d_k_snap, vs.d_k_cache, local_sn * cfg_.head_dim * kv_eb, cudaMemcpyDeviceToDevice, stream_));
            INK_CUDA_CHECK(cudaMemcpyAsync(ps.d_v_snap, vs.d_v_cache, local_sn * cfg_.head_dim * kv_eb, cudaMemcpyDeviceToDevice, stream_));
            if (kv_int8_) {
                INK_CUDA_CHECK(cudaMemcpyAsync(ps.d_ks_snap, vs.d_k_scale, local_sn * sizeof(f32), cudaMemcpyDeviceToDevice, stream_));
                INK_CUDA_CHECK(cudaMemcpyAsync(ps.d_vs_snap, vs.d_v_scale, local_sn * sizeof(f32), cudaMemcpyDeviceToDevice, stream_));
            }
        }
        INK_CUDA_CHECK(cudaMemcpyAsync(ps.d_kh_snap, vs.d_k_hist,    16ULL * cfg_.kv_dim() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_));
        INK_CUDA_CHECK(cudaMemcpyAsync(ps.d_vh_snap, vs.d_v_hist,    16ULL * cfg_.kv_dim() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_));
        INK_CUDA_CHECK(cudaMemcpyAsync(ps.d_ah_snap, vs.d_attn_hist, 16ULL * H * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_));
        INK_CUDA_CHECK(cudaMemcpyAsync(ps.d_mh_snap, vs.d_mlp_hist,  16ULL * H * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_));
    }
    prefix_tokens_.assign(prefix_ids.begin(), prefix_ids.end());
    prefix_snap_len_ = static_cast<u32>(prefix_ids.size());
    u32 keep_blks = (prefix_snap_len_ + kKvBlockSize - 1) / kKvBlockSize;
    prefix_bt_.assign(bt.begin(), bt.begin() + std::min<size_t>(keep_blks, bt.size()));
}

bool Inkling2Engine::try_restore_prefix_snapshot(std::span<const i32> prompt_ids, u32& out_start_pos, std::vector<i32>& out_bt) {
    if (!prefix_snap_alloc_ || prefix_snap_len_ == 0 || prompt_ids.size() <= prefix_snap_len_) return false;
    for (u32 i = 0; i < prefix_snap_len_; ++i) {
        if (prompt_ids[i] != prefix_tokens_[i]) return false;
    }
    const u32 H = cfg_.hidden_size;
    size_t kv_eb = kv_int8_ ? sizeof(u8) : sizeof(__nv_bfloat16);
    size_t local_sn = static_cast<size_t>(cfg_.num_kv_heads) * cfg_.ink.sliding_window_size;
    for (size_t v = 0; v < visits_.size(); ++v) {
        auto& vs = visits_[v];
        const auto& ps = prefix_snap_[v];
        if (vs.is_local) {
            INK_CUDA_CHECK(cudaMemcpyAsync(vs.d_k_cache, ps.d_k_snap, local_sn * cfg_.head_dim * kv_eb, cudaMemcpyDeviceToDevice, stream_));
            INK_CUDA_CHECK(cudaMemcpyAsync(vs.d_v_cache, ps.d_v_snap, local_sn * cfg_.head_dim * kv_eb, cudaMemcpyDeviceToDevice, stream_));
            if (kv_int8_) {
                INK_CUDA_CHECK(cudaMemcpyAsync(vs.d_k_scale, ps.d_ks_snap, local_sn * sizeof(f32), cudaMemcpyDeviceToDevice, stream_));
                INK_CUDA_CHECK(cudaMemcpyAsync(vs.d_v_scale, ps.d_vs_snap, local_sn * sizeof(f32), cudaMemcpyDeviceToDevice, stream_));
            }
        }
        INK_CUDA_CHECK(cudaMemcpyAsync(vs.d_k_hist,    ps.d_kh_snap, 16ULL * cfg_.kv_dim() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_));
        INK_CUDA_CHECK(cudaMemcpyAsync(vs.d_v_hist,    ps.d_vh_snap, 16ULL * cfg_.kv_dim() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_));
        INK_CUDA_CHECK(cudaMemcpyAsync(vs.d_attn_hist, ps.d_ah_snap, 16ULL * H * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_));
        INK_CUDA_CHECK(cudaMemcpyAsync(vs.d_mlp_hist,  ps.d_mh_snap, 16ULL * H * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_));
    }
    out_start_pos = prefix_snap_len_;
    out_bt = prefix_bt_;
    return true;
}

namespace {
struct KvSnapshotHeader {
    u32 magic;        // 0x564B4C46 ("FLKV")
    u32 version;      // 1
    u32 kv_mode;      // KvQuantMode
    u32 num_visits;   // 48
    u32 num_kv_heads; // 8
    u32 head_dim;     // 128
    u32 swa_window;   // 1024
    u32 block_size;   // 16
    u32 seq_len;      // populated token count
    u32 num_blocks;   // populated global blocks
    u32 reserved[6];
};
} // namespace

bool Inkling2Engine::save_kv_file(const std::string& path, u32 seq_len, std::string& err) {
    Stopwatch sw;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        err = "failed to open KV snapshot for write: " + path;
        return false;
    }
    INK_CUDA_CHECK(cudaStreamSynchronize(stream_));

    const u32 num_blocks = (seq_len + kKvBlockSize - 1u) / kKvBlockSize;
    KvSnapshotHeader hdr{};
    hdr.magic        = 0x564B4C46u;
    hdr.version      = 1u;
    hdr.kv_mode      = static_cast<u32>(kv_mode_);
    hdr.num_visits   = static_cast<u32>(visits_.size());
    hdr.num_kv_heads = cfg_.num_kv_heads;
    hdr.head_dim     = cfg_.head_dim;
    hdr.swa_window   = cfg_.ink.sliding_window_size;
    hdr.block_size   = kKvBlockSize;
    hdr.seq_len      = seq_len;
    hdr.num_blocks   = num_blocks;
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    constexpr size_t kStageBytes = 32ULL * 1024 * 1024;
    u8* h_stage = nullptr;
    INK_CUDA_CHECK(cudaHostAlloc(&h_stage, kStageBytes, cudaHostAllocDefault));
    size_t total_written = sizeof(hdr);

    auto dump_dev = [&](const void* dev_ptr, size_t nbytes) -> bool {
        const auto* src = static_cast<const u8*>(dev_ptr);
        size_t off = 0;
        while (off < nbytes) {
            size_t chunk = std::min(kStageBytes, nbytes - off);
            INK_CUDA_CHECK(cudaMemcpy(h_stage, src + off, chunk, cudaMemcpyDeviceToHost));
            out.write(reinterpret_cast<const char*>(h_stage), static_cast<std::streamsize>(chunk));
            if (!out) return false;
            off += chunk;
            total_written += chunk;
        }
        return true;
    };

    const u32 H = cfg_.hidden_size;
    const bool global_wht = (kv_mode_ == KvQuantMode::WHT4 || kv_mode_ == KvQuantMode::WHT84);
    const bool global_k4  = (kv_mode_ == KvQuantMode::WHT4);
    const size_t local_kv_eb  = kv_int8_ ? sizeof(u8) : sizeof(__nv_bfloat16);
    const size_t local_sn     = static_cast<size_t>(cfg_.num_kv_heads) * cfg_.ink.sliding_window_size;
    const size_t global_sn    = static_cast<size_t>(num_blocks) * cfg_.num_kv_heads * kKvBlockSize;
    const size_t g_k_bph      = global_wht ? (global_k4 ? (cfg_.head_dim / 2) : cfg_.head_dim) : (cfg_.head_dim * local_kv_eb);
    const size_t g_v_bph      = global_wht ? (cfg_.head_dim / 2) : (cfg_.head_dim * local_kv_eb);
    const size_t g_sc_eb      = global_wht ? sizeof(__nv_bfloat16) : (kv_int8_ ? sizeof(f32) : 0ULL);

    bool ok = true;
    for (const auto& vs : visits_) {
        if (vs.is_local) {
            ok = ok && dump_dev(vs.d_k_cache, local_sn * cfg_.head_dim * local_kv_eb);
            ok = ok && dump_dev(vs.d_v_cache, local_sn * cfg_.head_dim * local_kv_eb);
            if (kv_int8_) {
                ok = ok && dump_dev(vs.d_k_scale, local_sn * sizeof(f32));
                ok = ok && dump_dev(vs.d_v_scale, local_sn * sizeof(f32));
            }
        } else {
            ok = ok && dump_dev(vs.d_k_cache, global_sn * g_k_bph);
            ok = ok && dump_dev(vs.d_v_cache, global_sn * g_v_bph);
            if (g_sc_eb > 0) {
                ok = ok && dump_dev(vs.d_k_scale, global_sn * g_sc_eb);
                ok = ok && dump_dev(vs.d_v_scale, global_sn * g_sc_eb);
            }
        }
        ok = ok && dump_dev(vs.d_k_hist,    16ULL * cfg_.kv_dim() * sizeof(__nv_bfloat16));
        ok = ok && dump_dev(vs.d_v_hist,    16ULL * cfg_.kv_dim() * sizeof(__nv_bfloat16));
        ok = ok && dump_dev(vs.d_attn_hist, 16ULL * H * sizeof(__nv_bfloat16));
        ok = ok && dump_dev(vs.d_mlp_hist,  16ULL * H * sizeof(__nv_bfloat16));
        if (!ok) break;
    }

    cudaFreeHost(h_stage);
    out.flush();
    if (!ok || !out) {
        err = "I/O error while writing KV snapshot: " + path;
        return false;
    }
    std::fprintf(stderr, "[flint kv] saved %u tokens (%.2f GiB) -> %s in %.2fs\n",
                 seq_len, static_cast<double>(total_written) / (1024.0 * 1024.0 * 1024.0),
                 path.c_str(), sw.elapsed_ms() / 1000.0);
    return true;
}

bool Inkling2Engine::load_kv_file(const std::string& path, u32& out_seq_len, std::string& err) {
    Stopwatch sw;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = "failed to open KV snapshot for read: " + path;
        return false;
    }
    KvSnapshotHeader hdr{};
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!in || hdr.magic != 0x564B4C46u || hdr.version != 1u) {
        err = "invalid KV snapshot header: " + path;
        return false;
    }
    if (hdr.kv_mode != static_cast<u32>(kv_mode_) ||
        hdr.num_visits != visits_.size() ||
        hdr.num_kv_heads != cfg_.num_kv_heads ||
        hdr.head_dim != cfg_.head_dim ||
        hdr.swa_window != cfg_.ink.sliding_window_size ||
        hdr.block_size != kKvBlockSize) {
        err = "KV snapshot incompatible with active model/kv-mode: " + path;
        return false;
    }
    if (hdr.seq_len > max_seq_) {
        err = "KV snapshot seq_len (" + std::to_string(hdr.seq_len) +
              ") exceeds --max-seq-len (" + std::to_string(max_seq_) + ")";
        return false;
    }

    constexpr size_t kStageBytes = 32ULL * 1024 * 1024;
    u8* h_stage[2] = {nullptr, nullptr};
    cudaEvent_t ev[2] = {nullptr, nullptr};
    INK_CUDA_CHECK(cudaHostAlloc(&h_stage[0], kStageBytes, cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaHostAlloc(&h_stage[1], kStageBytes, cudaHostAllocDefault));
    INK_CUDA_CHECK(cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming));
    INK_CUDA_CHECK(cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming));
    int sidx = 0;
    size_t total_read = sizeof(hdr);

    auto load_dev = [&](void* dev_ptr, size_t nbytes) -> bool {
        auto* dst = static_cast<u8*>(dev_ptr);
        size_t off = 0;
        while (off < nbytes) {
            size_t chunk = std::min(kStageBytes, nbytes - off);
            int cur = sidx;
            sidx ^= 1;
            INK_CUDA_CHECK(cudaEventSynchronize(ev[cur]));
            in.read(reinterpret_cast<char*>(h_stage[cur]), static_cast<std::streamsize>(chunk));
            if (!in) return false;
            INK_CUDA_CHECK(cudaMemcpyAsync(dst + off, h_stage[cur], chunk, cudaMemcpyHostToDevice, stream_));
            INK_CUDA_CHECK(cudaEventRecord(ev[cur], stream_));
            off += chunk;
            total_read += chunk;
        }
        return true;
    };

    const u32 H = cfg_.hidden_size;
    const bool global_wht = (kv_mode_ == KvQuantMode::WHT4 || kv_mode_ == KvQuantMode::WHT84);
    const bool global_k4  = (kv_mode_ == KvQuantMode::WHT4);
    const size_t local_kv_eb  = kv_int8_ ? sizeof(u8) : sizeof(__nv_bfloat16);
    const size_t local_sn     = static_cast<size_t>(cfg_.num_kv_heads) * cfg_.ink.sliding_window_size;
    const size_t global_sn    = static_cast<size_t>(hdr.num_blocks) * cfg_.num_kv_heads * kKvBlockSize;
    const size_t g_k_bph      = global_wht ? (global_k4 ? (cfg_.head_dim / 2) : cfg_.head_dim) : (cfg_.head_dim * local_kv_eb);
    const size_t g_v_bph      = global_wht ? (cfg_.head_dim / 2) : (cfg_.head_dim * local_kv_eb);
    const size_t g_sc_eb      = global_wht ? sizeof(__nv_bfloat16) : (kv_int8_ ? sizeof(f32) : 0ULL);

    bool ok = true;
    for (auto& vs : visits_) {
        if (vs.is_local) {
            ok = ok && load_dev(vs.d_k_cache, local_sn * cfg_.head_dim * local_kv_eb);
            ok = ok && load_dev(vs.d_v_cache, local_sn * cfg_.head_dim * local_kv_eb);
            if (kv_int8_) {
                ok = ok && load_dev(vs.d_k_scale, local_sn * sizeof(f32));
                ok = ok && load_dev(vs.d_v_scale, local_sn * sizeof(f32));
            }
        } else {
            ok = ok && load_dev(vs.d_k_cache, global_sn * g_k_bph);
            ok = ok && load_dev(vs.d_v_cache, global_sn * g_v_bph);
            if (g_sc_eb > 0) {
                ok = ok && load_dev(vs.d_k_scale, global_sn * g_sc_eb);
                ok = ok && load_dev(vs.d_v_scale, global_sn * g_sc_eb);
            }
        }
        ok = ok && load_dev(vs.d_k_hist,    16ULL * cfg_.kv_dim() * sizeof(__nv_bfloat16));
        ok = ok && load_dev(vs.d_v_hist,    16ULL * cfg_.kv_dim() * sizeof(__nv_bfloat16));
        ok = ok && load_dev(vs.d_attn_hist, 16ULL * H * sizeof(__nv_bfloat16));
        ok = ok && load_dev(vs.d_mlp_hist,  16ULL * H * sizeof(__nv_bfloat16));
        if (!ok) break;
    }

    INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
    cudaEventDestroy(ev[0]); cudaEventDestroy(ev[1]);
    cudaFreeHost(h_stage[0]); cudaFreeHost(h_stage[1]);
    if (!ok) {
        err = "I/O error while reading KV snapshot: " + path;
        return false;
    }
    out_seq_len = hdr.seq_len;
    std::fprintf(stderr, "[flint kv] loaded %u tokens (%.2f GiB) <- %s in %.2fs\n",
                 out_seq_len, static_cast<double>(total_read) / (1024.0 * 1024.0 * 1024.0),
                 path.c_str(), sw.elapsed_ms() / 1000.0);
    return true;
}

bool Inkling2Engine::eval_logits(std::span<const i32> token_ids, std::vector<f32>& logits_out) {
    if (token_ids.empty()) return false;
    clear_kv_cache();
    u32 n_blocks = (static_cast<u32>(token_ids.size()) + kKvBlockSize - 1) / kKvBlockSize;
    std::vector<i32> bt;
    for (u32 b = 0; b < n_blocks; ++b) {
        i32 blk = kv_mgr_.allocate_block();
        if (blk < 0) return false;
        bt.push_back(blk);
    }
    INK_CUDA_CHECK(cudaMemcpyAsync(d_block_table_, bt.data(), bt.size() * sizeof(i32), cudaMemcpyHostToDevice, stream_));

    u32 pos = 0;
    while (pos < token_ids.size()) {
        u32 chunk = std::min<u32>(max_batch_tokens_, static_cast<u32>(token_ids.size() - pos));
        INK_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, token_ids.data() + pos, chunk * sizeof(i32), cudaMemcpyHostToDevice, stream_));
        run_forward(chunk, nullptr, pos, true);
        pos += chunk;
    }

    u32 V = cfg_.ink.unpadded_vocab_size;
    std::vector<u16> h_bf16(V);
    INK_CUDA_CHECK(cudaMemcpyAsync(h_bf16.data(), d_logits_, V * sizeof(u16), cudaMemcpyDeviceToHost, stream_));
    INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
    logits_out.resize(V);
    for (u32 i = 0; i < V; ++i) logits_out[i] = bf16_to_f32(h_bf16[i]);
    kv_mgr_.release_blocks(bt, 0);
    return true;
}

GenerateResult Inkling2Engine::generate_ids(std::span<const i32> prompt_ids,
                                            const GenerateOptions& gopts,
                                            const std::function<void(std::string_view)>& on_token) {
    GenerateResult res;
    if (prompt_ids.empty() || gopts.max_tokens == 0) return res;

    Stopwatch ttft_sw;
    const bool use_mtp = (spec_mode_ == SpecMode::Mtp && mtp_loaded_ && gopts.temperature <= 0.0f);

    u32 start_pos = 0;
    u32 loaded_seq = 0;
    bool loaded_from_disk = false;
    if (!gopts.load_kv_path.empty()) {
        std::string kerr;
        if (!load_kv_file(gopts.load_kv_path, loaded_seq, kerr)) {
            res.error = kerr;
            return res;
        }
        start_pos = loaded_seq;
        loaded_from_disk = true;
    }

    const u32 total_prompt_len = loaded_seq + static_cast<u32>(prompt_ids.size());
    const u32 max_gen = (max_seq_ > total_prompt_len)
        ? std::min<u32>(gopts.max_tokens, max_seq_ - total_prompt_len)
        : 0u;
    if (max_gen == 0) {
        res.error = "prompt + loaded KV exceeds --max-seq-len";
        return res;
    }
    u32 total_needed = total_prompt_len + max_gen;
    u32 needed_blocks = (total_needed + kKvBlockSize - 1) / kKvBlockSize;

    // Exact 1-Slot Prefix Cache: if prompt matches previous prefix snapshot (aligned to kKvBlockSize=16),
    // restore local SWA rings + 48x4 conv rings in ~0.08ms and prefill ONLY the uncached tail
    std::vector<i32> bt;
    bool prefix_hit = (!loaded_from_disk && !use_mtp) && try_restore_prefix_snapshot(prompt_ids, start_pos, bt);
    if (loaded_from_disk || prefix_hit) {
        // Global blocks are allocated sequentially (0,1,2,...) after clear() so indices match 1:1
        kv_mgr_.clear();
        bt.reserve(needed_blocks);
        for (u32 b = 0; b < needed_blocks; ++b) {
            i32 blk = kv_mgr_.allocate_block();
            if (blk < 0) return res;
            bt.push_back(blk);
        }
    } else {
        clear_kv_cache();
        bt.reserve(needed_blocks);
        for (u32 b = 0; b < needed_blocks; ++b) {
            i32 blk = kv_mgr_.allocate_block();
            if (blk < 0) return res;
            bt.push_back(blk);
        }
    }
    INK_CUDA_CHECK(cudaMemcpyAsync(d_block_table_, bt.data(), bt.size() * sizeof(i32), cudaMemcpyHostToDevice, stream_));

    // Prefill prompt in chunks; skip 95 MiB prefix snapshot on >=256k context runs to protect 24 GB VRAM
    u32 snap_target = (!loaded_from_disk && !use_mtp && !prefix_hit && max_seq_ < 262144 && prompt_ids.size() >= 512)
        ? ((static_cast<u32>(prompt_ids.size()) - 64) & ~(kKvBlockSize - 1u))
        : 0;
    u32 pos = start_pos;
    u32 next_log_pos = (prompt_ids.size() >= 65536) ? (loaded_seq + 65536u) : 0u;
    while (pos < total_prompt_len) {
        u32 limit = (snap_target > pos) ? snap_target : total_prompt_len;
        u32 chunk = std::min<u32>(max_batch_tokens_, limit - pos);
        u32 src_off = pos - loaded_seq;
        INK_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, prompt_ids.data() + src_off, chunk * sizeof(i32), cudaMemcpyHostToDevice, stream_));
        run_forward(chunk, nullptr, pos, true, false, use_mtp);
        if (use_mtp) mtp_store_context(chunk, pos);
        pos += chunk;
        if (snap_target > 0 && pos == snap_target) {
            save_prefix_snapshot(prompt_ids.subspan(0, snap_target), bt);
            snap_target = 0;
        }
        if (next_log_pos > 0 && pos >= next_log_pos) {
            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            double sec = ttft_sw.elapsed_ms() / 1000.0;
            u32 done = pos - loaded_seq;
            std::fprintf(stderr, "[flint prefill] %u / %zu tok (%.1f%%) | %.1fs | %.0f tok/s\n",
                         done, prompt_ids.size(), 100.0 * done / prompt_ids.size(), sec, sec > 0 ? done / sec : 0.0);
            next_log_pos += 65536u;
        }
    }
    res.stats.prefix_cached_tokens = start_pos;

    if (!gopts.save_kv_path.empty()) {
        std::string kerr;
        if (!save_kv_file(gopts.save_kv_path, total_prompt_len, kerr)) {
            res.error = kerr;
            kv_mgr_.release_blocks(bt, 0);
            return res;
        }
    }

    cuda::launch_argmax_bf16(d_logits_, d_out_tokens_, 1, cfg_.vocab_size, stream_);
    INK_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_, sizeof(i32), cudaMemcpyDeviceToHost, stream_));
    INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
    i32 next_tok = h_pinned_token_[0];

    res.stats.ttft_ms = ttft_sw.elapsed_ms();
    res.stats.prompt_tokens = total_prompt_len;

    auto emit_token = [&](i32 tok) -> bool {
        res.output_ids.push_back(tok);
        if (!gopts.ignore_eos && cfg_.is_eos(tok)) return false;
        std::string piece = tok_->decode_token(tok, true);
        res.text += piece;
        if (on_token && !piece.empty()) on_token(piece);
        return res.output_ids.size() < max_gen;
    };

    bool keep_going = emit_token(next_tok);
    Stopwatch dec_sw;
    std::vector<double> itls_ms;
    u32 cur_pos = total_prompt_len;
    u32 target_forward_steps = 0;

    if (keep_going && use_mtp) {
        // CUDA-Graphed Built-In MTP (1 graph launch for 3-layer+Markov draft + 1 graph launch for M=8 verify)
        ensure_mtp_graphs();
        std::vector<i32> drafts, step_in;
        while (keep_going) {
            Stopwatch tsw;
            u32 rem = gopts.max_tokens - static_cast<u32>(res.output_ids.size());
            u32 K = std::min<u32>({cfg_.ink.mtp_block_size - 1, 7u, rem > 0 ? rem - 1 : 0});
            const bool use_full_graphs = (mtp_graphs_captured_ && K == 7 && cur_pos > 0);
            drafts.clear();
            if (use_full_graphs) {
                i32 cpos = static_cast<i32>(cur_pos);
                INK_CUDA_CHECK(cudaMemcpyAsync(d_decode_pos_, &cpos, sizeof(i32), cudaMemcpyHostToDevice, stream_));
                INK_CUDA_CHECK(cudaMemcpyAsync(d_out_tokens_, &next_tok, sizeof(i32), cudaMemcpyHostToDevice, stream_));
                INK_CUDA_CHECK(cudaGraphLaunch(mtp_draft_graph_exec_, stream_));
                INK_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, d_out_tokens_, 8 * sizeof(i32), cudaMemcpyDeviceToDevice, stream_));
                INK_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_ + 8, d_out_tokens_ + 1, 7 * sizeof(i32), cudaMemcpyDeviceToHost, stream_));
                INK_CUDA_CHECK(cudaGraphLaunch(mtp_verify_graph_exec_, stream_));
                target_forward_steps++;
                INK_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_, 8 * sizeof(i32), cudaMemcpyDeviceToHost, stream_));
                INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
                drafts.assign(h_pinned_token_ + 8, h_pinned_token_ + 15);
            } else {
                mtp_draft_block(next_tok, cur_pos, K, drafts);
                u32 n_d = static_cast<u32>(drafts.size());
                step_in.clear();
                step_in.push_back(next_tok);
                for (i32 d : drafts) step_in.push_back(d);
                u32 M = n_d + 1;
                INK_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, step_in.data(), M * sizeof(i32), cudaMemcpyHostToDevice, stream_));
                run_forward(M, nullptr, cur_pos, false, false, /*capture_mtp=*/true);
                target_forward_steps++;
                cuda::launch_argmax_bf16(d_logits_, d_out_tokens_, M, cfg_.vocab_size, stream_);
                INK_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_, M * sizeof(i32), cudaMemcpyDeviceToHost, stream_));
                INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            }
            u32 n_d = static_cast<u32>(drafts.size());
            u32 M   = n_d + 1;

            auto vr = verify_speculative_greedy(drafts, std::span<const i32>(h_pinned_token_, M));
            res.stats.spec_drafted += n_d;
            res.stats.spec_accepted += vr.accepted_drafts;
            double step_ms = tsw.elapsed_ms();
            for (size_t i = 0; i < vr.emitted.size(); ++i) itls_ms.push_back(step_ms / static_cast<double>(vr.emitted.size()));

            u32 committed = 1 + vr.accepted_drafts;
            mtp_store_context(committed, cur_pos);
            cur_pos += committed;
            for (i32 et : vr.emitted) {
                next_tok = et;
                keep_going = emit_token(et);
                if (!keep_going) break;
            }
        }
    } else if (keep_going && spec_mode_ == SpecMode::Loop1 && gopts.temperature <= 0.0f) {
        // Looped-Depth Self-Speculative Decoding (1-Pass Core drafts 3-Pass Core — 0 extra VRAM, 100% lossless)
        // Runs 24 visits (pass 0 only) to draft up to K tokens, then verifies M=K+1 tokens in 1 full 48-visit pass
        std::vector<i32> drafts, step_in;
        while (keep_going) {
            Stopwatch tsw;
            u32 rem = gopts.max_tokens - static_cast<u32>(res.output_ids.size());
            u32 K = std::min<u32>({gopts.spec_k, 4u, rem > 0 ? rem - 1 : 0});
            drafts.clear();
            i32 cur_d = next_tok;
            for (u32 s = 0; s < K; ++s) {
                INK_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, &cur_d, sizeof(i32), cudaMemcpyHostToDevice, stream_));
                run_forward(1, nullptr, cur_pos + s, false, /*draft_1pass=*/true);
                cuda::launch_argmax_bf16(d_logits_, d_out_tokens_, 1, cfg_.vocab_size, stream_);
                INK_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_, sizeof(i32), cudaMemcpyDeviceToHost, stream_));
                INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
                cur_d = h_pinned_token_[0];
                drafts.push_back(cur_d);
            }

            step_in.clear();
            step_in.push_back(next_tok);
            for (i32 d : drafts) step_in.push_back(d);
            u32 M = K + 1;

            INK_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, step_in.data(), M * sizeof(i32), cudaMemcpyHostToDevice, stream_));
            run_forward(M, nullptr, cur_pos, false, /*draft_1pass=*/false);
            target_forward_steps++;
            cuda::launch_argmax_bf16(d_logits_, d_out_tokens_, M, cfg_.vocab_size, stream_);
            INK_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_, M * sizeof(i32), cudaMemcpyDeviceToHost, stream_));
            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));

            auto vr = verify_speculative_greedy(drafts, std::span<const i32>(h_pinned_token_, M));
            res.stats.spec_drafted += K;
            res.stats.spec_accepted += vr.accepted_drafts;
            double step_ms = tsw.elapsed_ms();
            for (size_t i = 0; i < vr.emitted.size(); ++i) itls_ms.push_back(step_ms / static_cast<double>(vr.emitted.size()));

            cur_pos += (1 + vr.accepted_drafts);
            for (i32 et : vr.emitted) {
                next_tok = et;
                keep_going = emit_token(et);
                if (!keep_going) break;
            }
        }
    } else if (keep_going && spec_mode_ == SpecMode::Ngram && gopts.temperature <= 0.0f) {
        // Lossless N-Gram Speculative Decoding (16-slot conv history rings make M<=8 rollback zero-cost & exact)
        NgramDrafter drafter;
        drafter.ingest(prompt_ids);
        drafter.push(next_tok);
        std::vector<i32> step_in;
        while (keep_going) {
            Stopwatch tsw;
            u32 rem = gopts.max_tokens - static_cast<u32>(res.output_ids.size());
            u32 max_k = std::min<u32>({gopts.spec_k, 7u, rem > 0 ? rem - 1 : 0});
            auto drafts = (max_k > 0) ? drafter.draft(max_k) : std::vector<i32>{};
            u32 K = static_cast<u32>(drafts.size());

            step_in.clear();
            step_in.push_back(next_tok);
            for (i32 d : drafts) step_in.push_back(d);
            u32 M = K + 1;

            INK_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, step_in.data(), M * sizeof(i32), cudaMemcpyHostToDevice, stream_));
            run_forward(M, nullptr, cur_pos, false);
            target_forward_steps++;
            cuda::launch_argmax_bf16(d_logits_, d_out_tokens_, M, cfg_.vocab_size, stream_);
            INK_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_, M * sizeof(i32), cudaMemcpyDeviceToHost, stream_));
            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));

            auto vr = verify_speculative_greedy(drafts, std::span<const i32>(h_pinned_token_, M));
            res.stats.spec_drafted += K;
            res.stats.spec_accepted += vr.accepted_drafts;
            double step_ms = tsw.elapsed_ms();
            for (size_t i = 0; i < vr.emitted.size(); ++i) itls_ms.push_back(step_ms / static_cast<double>(vr.emitted.size()));

            cur_pos += (1 + vr.accepted_drafts);
            for (i32 et : vr.emitted) {
                drafter.push(et);
                next_tok = et;
                keep_going = emit_token(et);
                if (!keep_going) break;
            }
        }
    } else if (keep_going && use_cuda_graph_ && gopts.temperature <= 0.0f) {
        i32 start_pos_i = static_cast<i32>(cur_pos);
        INK_CUDA_CHECK(cudaMemcpyAsync(d_decode_pos_, &start_pos_i, sizeof(i32), cudaMemcpyHostToDevice, stream_));
        INK_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, &next_tok, sizeof(i32), cudaMemcpyHostToDevice, stream_));
        ensure_decode_graph();
        while (keep_going) {
            Stopwatch tsw;
            INK_CUDA_CHECK(cudaGraphLaunch(decode_graph_exec_, stream_));
            INK_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_, sizeof(i32), cudaMemcpyDeviceToHost, stream_));
            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            itls_ms.push_back(tsw.elapsed_ms());
            next_tok = h_pinned_token_[0];
            cur_pos++;
            keep_going = emit_token(next_tok);
        }
    } else {
        while (keep_going) {
            Stopwatch tsw;
            INK_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, &next_tok, sizeof(i32), cudaMemcpyHostToDevice, stream_));
            run_forward(1, nullptr, cur_pos, false);
            cuda::launch_argmax_bf16(d_logits_, d_out_tokens_, 1, cfg_.vocab_size, stream_);
            INK_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_, sizeof(i32), cudaMemcpyDeviceToHost, stream_));
            INK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            itls_ms.push_back(tsw.elapsed_ms());
            next_tok = h_pinned_token_[0];
            cur_pos++;
            keep_going = emit_token(next_tok);
        }
    }

    res.stats.decode_ms = dec_sw.elapsed_ms();
    res.stats.generated_tokens = static_cast<u32>(res.output_ids.size());
    u32 dec_steps = (res.stats.generated_tokens > 1) ? (res.stats.generated_tokens - 1) : 0;
    u32 phys_passes = (target_forward_steps > 0) ? target_forward_steps : dec_steps;
    if (dec_steps > 0 && res.stats.decode_ms > 0.0) {
        res.stats.decode_tok_s = static_cast<double>(dec_steps) / (res.stats.decode_ms / 1000.0);
        double avg_seq = static_cast<double>(total_prompt_len) + 0.5 * dec_steps;
        double local_bph  = kv_int8_ ? (static_cast<double>(cfg_.head_dim) + 4.0) : (2.0 * static_cast<double>(cfg_.head_dim));
        double global_pair_b = (kv_mode_ == KvQuantMode::WHT4)
            ? (static_cast<double>(cfg_.head_dim) + 4.0)             // 64B K + 64B V + 4B BF16 scales = 132 B/kv_head
            : ((kv_mode_ == KvQuantMode::WHT84)
                ? (static_cast<double>(cfg_.head_dim) * 1.5 + 4.0)   // 128B K + 64B V + 4B BF16 scales = 196 B/kv_head
                : (2.0 * local_bph));
        double local_kv  = 40.0 * 2.0 * cfg_.num_kv_heads * local_bph * std::min<double>(avg_seq, cfg_.ink.sliding_window_size);
        double global_kv =  8.0 * cfg_.num_kv_heads * global_pair_b * avg_seq;
        double embed_row = static_cast<double>(cfg_.hidden_size) * 2.0;
        double bytes_per_step = static_cast<double>(active_weight_bytes_) + embed_row + local_kv + global_kv;
        res.stats.achieved_hbm_gbs = (static_cast<double>(phys_passes) * bytes_per_step / 1e9) / (res.stats.decode_ms / 1000.0);
        if (gpu_peak_hbm_gbs_ > 0.0) res.stats.pct_hbm_peak = 100.0 * res.stats.achieved_hbm_gbs / gpu_peak_hbm_gbs_;
    }
    if (res.stats.spec_drafted > 0) {
        res.stats.spec_accept_rate = static_cast<double>(res.stats.spec_accepted) / static_cast<double>(res.stats.spec_drafted);
    }
    if (!itls_ms.empty()) {
        std::sort(itls_ms.begin(), itls_ms.end());
        res.stats.itl_p50_ms = itls_ms[itls_ms.size() / 2];
        res.stats.itl_p99_ms = itls_ms[std::min(itls_ms.size() - 1, static_cast<size_t>(itls_ms.size() * 0.99))];
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    res.stats.vram_used_mib = (total_b - free_b) / (1024 * 1024);
    res.stats.vram_total_mib = total_b / (1024 * 1024);
    res.stats.host_rss_mib = current_rss_kib() / 1024;
    kv_mgr_.release_blocks(bt, 0);
    return res;
}

#else

bool Inkling2Engine::load(const ModelConfig&, const BpeTokenizer&, const EngineOptions&, std::string& err) {
    err = "Inkling-2-Mini NVFP4 engine requires CUDA build (make CUDA=1)";
    return false;
}
GenerateResult Inkling2Engine::generate_ids(std::span<const i32>, const GenerateOptions&,
                                            const std::function<void(std::string_view)>&) { return {}; }
bool Inkling2Engine::eval_logits(std::span<const i32>, std::vector<f32>&) { return false; }

#endif

} // namespace flint
