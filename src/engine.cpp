#include "engine.hpp"
#include "kernels/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <thread>

namespace flint {

namespace {

i32 sample_logits(std::span<const f32> logits, f32 temp, f32 top_p, std::mt19937_64& rng) {
    if (temp <= 0.0f) {
        return CpuRefDecoder::argmax(logits);
    }
    const size_t V = logits.size();
    f32 max_v = -1e30f;
    for (size_t i = 0; i < V; ++i) {
        if (logits[i] > max_v) max_v = logits[i];
    }
    std::vector<std::pair<f32, i32>> probs;
    probs.reserve(V);
    f32 inv_t = 1.0f / temp;
    f64 sum = 0.0;
    for (size_t i = 0; i < V; ++i) {
        f32 p = std::exp((logits[i] - max_v) * inv_t);
        sum += p;
        probs.emplace_back(p, static_cast<i32>(i));
    }
    f32 inv_sum = static_cast<f32>(1.0 / sum);
    for (auto& kv : probs) kv.first *= inv_sum;

    if (top_p < 1.0f) {
        std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });
        f32 cum = 0.0f;
        size_t cut = 0;
        for (; cut < probs.size(); ++cut) {
            cum += probs[cut].first;
            if (cum >= top_p) { ++cut; break; }
        }
        probs.resize(std::max<size_t>(1, cut));
        f32 norm = 0.0f;
        for (const auto& kv : probs) norm += kv.first;
        std::uniform_real_distribution<f32> dist(0.0f, norm);
        f32 r = dist(rng);
        f32 acc = 0.0f;
        for (const auto& kv : probs) {
            acc += kv.first;
            if (acc >= r) return kv.second;
        }
        return probs.back().second;
    }

    std::uniform_real_distribution<f32> dist(0.0f, 1.0f);
    f32 r = dist(rng);
    f32 acc = 0.0f;
    for (const auto& kv : probs) {
        acc += kv.first;
        if (acc >= r) return kv.second;
    }
    return probs.back().second;
}

} // namespace

InferenceEngine::~InferenceEngine() {
    free_cuda_resources();
}

void InferenceEngine::clear_kv_cache() {
    kv_mgr_.clear();
    if (cpu_ref_) cpu_ref_->reset_kv();
    if (inkling2_) inkling2_->clear_kv_cache();
}

void InferenceEngine::free_cuda_resources() {
#if defined(FLINT_CUDA) && FLINT_CUDA == 1
    destroy_decode_graph();
    auto dfree = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
    if (d_weight_slab_) {
        // Single-slab allocation: free slab once and null out all interior view pointers
        dfree(d_weight_slab_);
        d_embed_ = nullptr;
        d_final_norm_ = nullptr;
        d_lm_head_ = nullptr;
        d_lm_head_fp8_ = nullptr;
        d_lm_head_scales_ = nullptr;
        d_layers_.clear();
    } else {
        dfree(d_embed_);
        dfree(d_final_norm_);
        dfree(d_lm_head_);
        dfree(d_lm_head_fp8_);
        dfree(d_lm_head_scales_);
        for (auto& l : d_layers_) {
            dfree(l.input_norm);
            dfree(l.post_attn_norm);
            dfree(l.q_norm);
            dfree(l.k_norm);
            dfree(l.qkv_bias);
            dfree(l.w_qkv);
            dfree(l.s_qkv);
            dfree(l.w_o);
            dfree(l.s_o);
            dfree(l.w_gu);
            dfree(l.s_gu);
            dfree(l.w_down);
            dfree(l.s_down);
        }
        d_layers_.clear();
    }
    dfree(d_inv_freq_);
    for (auto*& p : d_k_caches_) dfree(p);
    for (auto*& p : d_v_caches_) dfree(p);
    for (auto*& p : d_k_scales_) dfree(p);
    for (auto*& p : d_v_scales_) dfree(p);
    d_k_caches_.clear();
    d_v_caches_.clear();
    d_k_scales_.clear();
    d_v_scales_.clear();
    dfree(d_block_table_);
    dfree(d_x_);
    dfree(d_norm_);
    dfree(d_qkv_);
    dfree(d_q_);
    dfree(d_attn_out_);
    dfree(d_proj_out_);
    dfree(d_mlp_act_);
    dfree(d_mlp_gu_tmp_);
    dfree(d_logits_);
    dfree(d_dequant_tmp_);
    dfree(d_split_o_);
    dfree(d_split_ml_);
    dfree(d_token_ids_);
    dfree(d_out_tokens_);
    dfree(d_out_conf_);
    dfree(d_decode_pos_);
    dfree(d_cublaslt_ws_);
    dfree(d_x_fp8_);
    dfree(d_fp8_scale_);
    dfree(d_fp8_out_pad_);
    dfree(d_mk_layers_);
    dfree(d_mk_argmax_val_);
    dfree(d_mk_argmax_idx_);
    if (h_pinned_token_) { cudaFreeHost(h_pinned_token_); h_pinned_token_ = nullptr; }
    if (h_pinned_conf_)  { cudaFreeHost(h_pinned_conf_);  h_pinned_conf_ = nullptr; }
    if (cublaslt_) { cublasLtDestroy(cublaslt_); cublaslt_ = nullptr; }
    if (cublas_) { cublasDestroy(cublas_); cublas_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
#endif
}

bool InferenceEngine::load(const EngineOptions& opts, std::string& err) {
    Stopwatch sw;
    free_cuda_resources();
    opts_ = opts;
    quant_ = opts.quant;

    std::error_code ec;
    bool is_pack = std::filesystem::is_regular_file(opts.model_path, ec) &&
                   opts.model_path.extension() == ".pack";

    std::filesystem::path tok_dir = !opts.tokenizer_dir.empty()
        ? opts.tokenizer_dir
        : (is_pack ? opts.model_path.parent_path() : opts.model_path);

    // Fast path: load pre-parsed binary tokenizer embedded inside .pack (<25ms vs 450ms JSON parse)
    bool tok_loaded = false;
    if (is_pack) {
        PackedModelReader pr_tok;
        std::string perr;
        if (pr_tok.open(opts.model_path, perr)) {
            if (const auto* tv = pr_tok.get("flint.tokenizer")) {
                std::string berr;
                if (tokenizer_.load_binary(std::span<const u8>(tv->data, tv->nbytes), berr)) {
                    tok_loaded = true;
                }
            }
        }
    }
    if (!tok_loaded && !tokenizer_.load_from_dir(tok_dir, err)) {
        return false;
    }

#if defined(FLINT_CUDA) && FLINT_CUDA == 1
    int dev_count = 0;
    if (!opts.force_cpu && cudaGetDeviceCount(&dev_count) == cudaSuccess && dev_count > 0) {
        use_cuda_ = true;

        // Detect Inkling-2-Mini (inkling2_loop) from config.json or embedded .pack config
        ModelConfig probe_cfg;
        bool is_ink = false;
        if (is_pack) {
            PackedModelReader pr_cfg;
            std::string cerr;
            if (pr_cfg.open(opts.model_path, cerr)) {
                auto tmpd = std::filesystem::temp_directory_path() / ("flint_prb_" + std::to_string(::getpid()));
                std::filesystem::create_directories(tmpd, ec);
                { std::ofstream out(tmpd / "config.json"); out << pr_cfg.config_json(); }
                if (ModelConfig::load_from_dir(tmpd, probe_cfg, cerr) && probe_cfg.is_inkling2()) is_ink = true;
                std::filesystem::remove_all(tmpd, ec);
            }
        } else {
            std::string cerr;
            if (ModelConfig::load_from_dir(opts.model_path, probe_cfg, cerr) && probe_cfg.is_inkling2()) is_ink = true;
        }

        if (is_ink) {
            cfg_ = probe_cfg;
            quant_ = QuantMode::NVFP4;
            inkling2_ = std::make_unique<Inkling2Engine>();
            if (!inkling2_->load(cfg_, tokenizer_, opts, err)) {
                inkling2_.reset();
                return false;
            }
            active_weight_bytes_ = inkling2_->active_weight_bytes();
            gpu_name_ = inkling2_->gpu_name();
            gpu_peak_hbm_gbs_ = inkling2_->gpu_peak_hbm_gbs();
            cold_start_ms_ = sw.elapsed_ms();
            return true;
        }

        if (!load_cuda_weights(opts, err)) return false;

        // Optional draft model for speculative decoding
        if (opts.spec_mode == SpecMode::Draft && !opts.draft_model_path.empty()) {
            draft_engine_ = std::make_unique<InferenceEngine>();
            EngineOptions dopts;
            dopts.model_path = opts.draft_model_path;
            dopts.tokenizer_dir = tok_dir;
            dopts.quant = opts.quant;
            dopts.spec_mode = SpecMode::None;
            dopts.max_seq_len = opts.max_seq_len;
            dopts.use_cuda_graph = opts.use_cuda_graph;
            if (!draft_engine_->load(dopts, err)) {
                err = "Failed loading draft model: " + err;
                return false;
            }
        }

        cold_start_ms_ = sw.elapsed_ms();
        return true;
    }
#endif

    // CPU fallback (Milestone 0 verification & non-CUDA hosts)
    use_cuda_ = false;
    if (is_pack) {
        err = "CPU fallback requires raw HuggingFace safetensors directory";
        return false;
    }
    cpu_ref_ = std::make_unique<CpuRefDecoder>();
    if (!cpu_ref_->load(opts.model_path, opts.max_seq_len, err)) return false;
    cfg_ = cpu_ref_->config();
    active_weight_bytes_ = static_cast<u64>(cfg_.num_layers) *
        (static_cast<u64>(cfg_.q_dim() + 2 * cfg_.kv_dim()) * cfg_.hidden_size +
         static_cast<u64>(cfg_.hidden_size) * cfg_.q_dim() +
         static_cast<u64>(3) * cfg_.intermediate_size * cfg_.hidden_size) * 2ULL;
    cold_start_ms_ = sw.elapsed_ms();
    return true;
}

#if defined(FLINT_CUDA) && FLINT_CUDA == 1

bool InferenceEngine::load_cuda_weights(const EngineOptions& opts, std::string& err) {
    cudaDeviceProp prop{};
    FLINT_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    gpu_name_ = prop.name;
    // Theoretical HBM/GDDR bandwidth in GB/s: 2 (DDR) * memClock(kHz) * busWidth(bits) / 8 / 1e6
    int mem_clk_khz = 0, bus_bits = 0;
    cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    gpu_peak_hbm_gbs_ = 2.0 * static_cast<double>(mem_clk_khz) * static_cast<double>(bus_bits) / 8.0 / 1e6;

    FLINT_CUDA_CHECK(cudaStreamCreate(&stream_));
    if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) {
        err = "cublasCreate failed";
        return false;
    }
    cublasSetStream(cublas_, stream_);
    if (cublasLtCreate(&cublaslt_) != CUBLAS_STATUS_SUCCESS) {
        err = "cublasLtCreate failed";
        return false;
    }

    std::error_code ec;
    bool is_pack = std::filesystem::is_regular_file(opts.model_path, ec) &&
                   opts.model_path.extension() == ".pack";

    // Double-buffered 64 MiB pinned host staging pool + 4-thread parallel memcpy for PCIe Gen4 saturation
    constexpr size_t kStageBytes = 64ULL * 1024 * 1024;
    u8* h_stage[2] = {nullptr, nullptr};
    cudaEvent_t ev_stage[2] = {nullptr, nullptr};
    FLINT_CUDA_CHECK(cudaHostAlloc(&h_stage[0], kStageBytes, cudaHostAllocDefault));
    FLINT_CUDA_CHECK(cudaHostAlloc(&h_stage[1], kStageBytes, cudaHostAllocDefault));
    FLINT_CUDA_CHECK(cudaEventCreateWithFlags(&ev_stage[0], cudaEventDisableTiming));
    FLINT_CUDA_CHECK(cudaEventCreateWithFlags(&ev_stage[1], cudaEventDisableTiming));
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
            ts[t] = std::thread([=]() {
                std::memcpy(dst + t * part, src + t * part, part);
            });
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
            FLINT_CUDA_CHECK(cudaEventSynchronize(ev_stage[cur]));
            parallel_memcpy(h_stage[cur], s + off, chunk);
            FLINT_CUDA_CHECK(cudaMemcpyAsync(d + off, h_stage[cur], chunk, cudaMemcpyHostToDevice, stream_));
            FLINT_CUDA_CHECK(cudaEventRecord(ev_stage[cur], stream_));
            off += chunk;
        }
    };

    auto upload_raw = [&](const void* src, size_t nbytes, void** dst) {
        FLINT_CUDA_CHECK(cudaMalloc(dst, nbytes));
        stream_h2d_into(src, nbytes, *dst);
    };

    active_weight_bytes_ = 0;

    if (is_pack) {
        PackedModelReader pr;
        if (!pr.open(opts.model_path, err)) { cleanup_stage(); return false; }
        quant_ = pr.quant_mode();

        // Always parse the config.json embedded inside the .pack file itself (never a sibling dir's config)
        {
            auto tmpd = std::filesystem::temp_directory_path() / ("flint_cfg_" + std::to_string(::getpid()) + "_" + std::to_string(reinterpret_cast<uintptr_t>(this)));
            std::filesystem::create_directories(tmpd, ec);
            {
                std::ofstream out(tmpd / "config.json");
                out << pr.config_json();
            }
            bool ok = ModelConfig::load_from_dir(tmpd, cfg_, err);
            std::filesystem::remove_all(tmpd, ec);
            if (!ok) { cleanup_stage(); return false; }
        }

        // SINGLE-SLAB CONTIGUOUS VRAM UPLOAD: 1 cudaMalloc + 1 pipelined 4-thread DMA transfer
        const u8* host_slab = pr.base_ptr() + pr.data_offset();
        const u64 slab_bytes = pr.weight_slab_bytes();
        FLINT_CUDA_CHECK(cudaMalloc(&d_weight_slab_, slab_bytes));
        stream_h2d_into(host_slab, slab_bytes, d_weight_slab_);

        auto bind_t = [&](const std::string& name, void** dst, bool required = true, bool count_active = true) -> bool {
            const auto* tv = pr.get(name);
            if (!tv) {
                if (required) { err = "Missing tensor in pack: " + name; return false; }
                *dst = nullptr;
                return true;
            }
            size_t rel = static_cast<size_t>(tv->data - host_slab);
            *dst = d_weight_slab_ + rel;
            if (count_active) active_weight_bytes_ += tv->nbytes;
            return true;
        };

        // embed_tokens is ONLY read as 1 row (hidden_size * 2 B) per decode step unless tied to lm_head
        if (!bind_t("model.embed_tokens.weight", reinterpret_cast<void**>(&d_embed_), true, cfg_.tie_word_embeddings)) { cleanup_stage(); return false; }
        if (!bind_t("model.norm.weight", reinterpret_cast<void**>(&d_final_norm_))) { cleanup_stage(); return false; }
        if (!cfg_.tie_word_embeddings) {
            if (pr.get("lm_head.qweight") && pr.get("lm_head.scales")) {
                if (!bind_t("lm_head.qweight", reinterpret_cast<void**>(&d_lm_head_fp8_))) { cleanup_stage(); return false; }
                if (!bind_t("lm_head.scales",  reinterpret_cast<void**>(&d_lm_head_scales_))) { cleanup_stage(); return false; }
            } else {
                if (!bind_t("lm_head.weight", reinterpret_cast<void**>(&d_lm_head_), false)) { cleanup_stage(); return false; }
                if (!d_lm_head_) {
                    cfg_.tie_word_embeddings = true;
                    if (auto* etv = pr.get("model.embed_tokens.weight")) active_weight_bytes_ += etv->nbytes;
                }
            }
        }

        d_layers_.resize(cfg_.num_layers);
        for (u32 l = 0; l < cfg_.num_layers; ++l) {
            std::string pre = "model.layers." + std::to_string(l) + ".";
            auto& dl = d_layers_[l];
            if (!bind_t(pre + "input_layernorm.weight", reinterpret_cast<void**>(&dl.input_norm))) { cleanup_stage(); return false; }
            if (!bind_t(pre + "post_attention_layernorm.weight", reinterpret_cast<void**>(&dl.post_attn_norm))) { cleanup_stage(); return false; }
            if (cfg_.has_qk_norm) {
                if (!bind_t(pre + "self_attn.q_norm.weight", reinterpret_cast<void**>(&dl.q_norm))) { cleanup_stage(); return false; }
                if (!bind_t(pre + "self_attn.k_norm.weight", reinterpret_cast<void**>(&dl.k_norm))) { cleanup_stage(); return false; }
            }
            if (cfg_.has_qkv_bias) {
                if (!bind_t(pre + "self_attn.qkv_bias", reinterpret_cast<void**>(&dl.qkv_bias), false)) { cleanup_stage(); return false; }
            }
            if (quant_ == QuantMode::BF16) {
                if (!bind_t(pre + "self_attn.qkv_proj", &dl.w_qkv)) { cleanup_stage(); return false; }
                if (!bind_t(pre + "self_attn.o_proj", &dl.w_o)) { cleanup_stage(); return false; }
                if (!bind_t(pre + "mlp.gate_up_proj", &dl.w_gu)) { cleanup_stage(); return false; }
                if (!bind_t(pre + "mlp.down_proj", &dl.w_down)) { cleanup_stage(); return false; }
            } else {
                if (!bind_t(pre + "self_attn.qkv_proj.qweight", &dl.w_qkv)) { cleanup_stage(); return false; }
                if (!bind_t(pre + "self_attn.qkv_proj.scales",  &dl.s_qkv)) { cleanup_stage(); return false; }
                if (!bind_t(pre + "self_attn.o_proj.qweight",   &dl.w_o))   { cleanup_stage(); return false; }
                if (!bind_t(pre + "self_attn.o_proj.scales",    &dl.s_o))   { cleanup_stage(); return false; }
                if (!bind_t(pre + "mlp.gate_up_proj.qweight",   &dl.w_gu))  { cleanup_stage(); return false; }
                if (!bind_t(pre + "mlp.gate_up_proj.scales",    &dl.s_gu))  { cleanup_stage(); return false; }
                if (!bind_t(pre + "mlp.down_proj.qweight",      &dl.w_down)){ cleanup_stage(); return false; }
                if (!bind_t(pre + "mlp.down_proj.scales",       &dl.s_down)){ cleanup_stage(); return false; }
            }
        }
    } else {
        // Direct loading from HuggingFace .safetensors directory (with on-the-fly QKV/Gate+Up fusion & quantization)
        if (!ModelConfig::load_from_dir(opts.model_path, cfg_, err)) return false;
        SafetensorsLoader st;
        if (!st.open_dir(opts.model_path, err)) return false;

        auto upload_bf16_tv = [&](const std::string& name, __nv_bfloat16** dst, bool required = true, bool count_active = true) -> bool {
            const auto* tv = st.get(name);
            if (!tv) {
                if (required) { err = "Missing tensor: " + name; return false; }
                *dst = nullptr;
                return true;
            }
            if (tv->dtype == DType::BF16) {
                upload_raw(tv->data, tv->nbytes, reinterpret_cast<void**>(dst));
                if (count_active) active_weight_bytes_ += tv->nbytes;
            } else {
                std::vector<u16> b;
                SafetensorsLoader::read_as_bf16(*tv, b);
                upload_raw(b.data(), b.size() * 2, reinterpret_cast<void**>(dst));
                if (count_active) active_weight_bytes_ += b.size() * 2;
            }
            return true;
        };

        if (!upload_bf16_tv("model.embed_tokens.weight", &d_embed_, true, cfg_.tie_word_embeddings)) return false;
        if (!upload_bf16_tv("model.norm.weight", &d_final_norm_)) return false;
        if (!cfg_.tie_word_embeddings) {
            if (!upload_bf16_tv("lm_head.weight", &d_lm_head_, false)) return false;
            if (!d_lm_head_) {
                cfg_.tie_word_embeddings = true;
                if (auto* etv = st.get("model.embed_tokens.weight")) active_weight_bytes_ += etv->numel() * 2;
            }
        }

        auto upload_linear = [&](const std::vector<u16>& bf16_mat, u32 rows, u32 cols,
                                 void** dst_w, void** dst_s) {
            if (quant_ == QuantMode::BF16) {
                upload_raw(bf16_mat.data(), bf16_mat.size() * 2, dst_w);
                active_weight_bytes_ += bf16_mat.size() * 2;
            } else {
                std::vector<f32> fm(bf16_mat.size());
                for (size_t i = 0; i < bf16_mat.size(); ++i) fm[i] = bf16_to_f32(bf16_mat[i]);
                if (quant_ == QuantMode::W4A16) {
                    std::vector<u32> qw;
                    std::vector<u32> sc;
                    quantize_w4a16_g128(fm.data(), rows, cols, qw, sc);
                    upload_raw(qw.data(), qw.size() * 4, dst_w);
                    upload_raw(sc.data(), sc.size() * 4, dst_s);
                    active_weight_bytes_ += qw.size() * 4 + sc.size() * 4;
                } else {
                    std::vector<u8> qw;
                    std::vector<f32> sc;
                    quantize_fp8_e4m3_rowwise(fm.data(), rows, cols, qw, sc);
                    upload_raw(qw.data(), qw.size(), dst_w);
                    upload_raw(sc.data(), sc.size() * 4, dst_s);
                    active_weight_bytes_ += qw.size() + sc.size() * 4;
                }
            }
        };

        d_layers_.resize(cfg_.num_layers);
        std::vector<u16> bq, bk, bv, bo, bg, bu, bd, fused;
        for (u32 l = 0; l < cfg_.num_layers; ++l) {
            std::string pre = "model.layers." + std::to_string(l) + ".";
            auto& dl = d_layers_[l];
            if (!upload_bf16_tv(pre + "input_layernorm.weight", &dl.input_norm)) return false;
            if (!upload_bf16_tv(pre + "post_attention_layernorm.weight", &dl.post_attn_norm)) return false;
            if (cfg_.has_qk_norm) {
                if (!upload_bf16_tv(pre + "self_attn.q_norm.weight", &dl.q_norm)) return false;
                if (!upload_bf16_tv(pre + "self_attn.k_norm.weight", &dl.k_norm)) return false;
            }

            const auto* tq = st.get(pre + "self_attn.q_proj.weight");
            const auto* tk = st.get(pre + "self_attn.k_proj.weight");
            const auto* tv = st.get(pre + "self_attn.v_proj.weight");
            if (!tq || !tk || !tv) { err = "Missing QKV weight in layer " + std::to_string(l); return false; }
            SafetensorsLoader::read_as_bf16(*tq, bq);
            SafetensorsLoader::read_as_bf16(*tk, bk);
            SafetensorsLoader::read_as_bf16(*tv, bv);
            fused.clear();
            fused.reserve(bq.size() + bk.size() + bv.size());
            fused.insert(fused.end(), bq.begin(), bq.end());
            fused.insert(fused.end(), bk.begin(), bk.end());
            fused.insert(fused.end(), bv.begin(), bv.end());
            upload_linear(fused, cfg_.q_dim() + 2 * cfg_.kv_dim(), cfg_.hidden_size, &dl.w_qkv, &dl.s_qkv);

            if (cfg_.has_qkv_bias) {
                const auto* pbq = st.get(pre + "self_attn.q_proj.bias");
                const auto* pbk = st.get(pre + "self_attn.k_proj.bias");
                const auto* pbv = st.get(pre + "self_attn.v_proj.bias");
                if (pbq && pbk && pbv) {
                    SafetensorsLoader::read_as_bf16(*pbq, bq);
                    SafetensorsLoader::read_as_bf16(*pbk, bk);
                    SafetensorsLoader::read_as_bf16(*pbv, bv);
                    fused.clear();
                    fused.insert(fused.end(), bq.begin(), bq.end());
                    fused.insert(fused.end(), bk.begin(), bk.end());
                    fused.insert(fused.end(), bv.begin(), bv.end());
                    upload_raw(fused.data(), fused.size() * 2, reinterpret_cast<void**>(&dl.qkv_bias));
                    active_weight_bytes_ += fused.size() * 2;
                }
            }

            const auto* to = st.get(pre + "self_attn.o_proj.weight");
            if (!to) { err = "Missing o_proj in layer " + std::to_string(l); return false; }
            SafetensorsLoader::read_as_bf16(*to, bo);
            upload_linear(bo, cfg_.hidden_size, cfg_.q_dim(), &dl.w_o, &dl.s_o);

            const auto* tg = st.get(pre + "mlp.gate_proj.weight");
            const auto* tu = st.get(pre + "mlp.up_proj.weight");
            const auto* td = st.get(pre + "mlp.down_proj.weight");
            if (!tg || !tu || !td) { err = "Missing MLP weight in layer " + std::to_string(l); return false; }
            SafetensorsLoader::read_as_bf16(*tg, bg);
            SafetensorsLoader::read_as_bf16(*tu, bu);
            SafetensorsLoader::read_as_bf16(*td, bd);
            fused.clear();
            fused.reserve(bg.size() + bu.size());
            fused.insert(fused.end(), bg.begin(), bg.end());
            fused.insert(fused.end(), bu.begin(), bu.end());
            upload_linear(fused, 2 * cfg_.intermediate_size, cfg_.hidden_size, &dl.w_gu, &dl.s_gu);
            upload_linear(bd, cfg_.hidden_size, cfg_.intermediate_size, &dl.w_down, &dl.s_down);
        }
    }

    // Upload RoPE inverse frequencies and release pinned staging buffers
    auto inv_f = compute_rope_inv_freq(cfg_);
    upload_raw(inv_f.data(), inv_f.size() * sizeof(f32), reinterpret_cast<void**>(&d_inv_freq_));
    cleanup_stage();

    // Allocate Paged KV Cache VRAM pool (BF16, dynamic INT8, or Zero-Calibration Walsh-Hadamard 4-Bit WHT4)
    kv_mode_ = opts.kv_mode;
    if (opts.kv_fp8 && kv_mode_ == KvQuantMode::BF16) kv_mode_ = KvQuantMode::INT8;
    kv_fp8_ = (kv_mode_ == KvQuantMode::INT8);
    bool is_i4_kv = (kv_mode_ == KvQuantMode::WHT4 || kv_mode_ == KvQuantMode::INT4RAW);
    u32 max_seq = std::min(opts.max_seq_len, cfg_.max_position_embeddings);
    u32 num_blocks = opts.max_kv_blocks > 0
        ? opts.max_kv_blocks
        : ((max_seq + kKvBlockSize - 1) / kKvBlockSize + 64);
    kv_mgr_.init(num_blocks);

    size_t kv_scale_elems = static_cast<size_t>(num_blocks) * cfg_.num_kv_heads * kKvBlockSize;
    size_t k_layer_bytes = (kv_mode_ == KvQuantMode::INT4RAW)
        ? (kv_scale_elems * (cfg_.head_dim / 2))
        : (kv_scale_elems * cfg_.head_dim * ((kv_fp8_ || kv_mode_ == KvQuantMode::WHT4) ? sizeof(u8) : sizeof(__nv_bfloat16)));
    size_t v_layer_bytes = is_i4_kv
        ? (kv_scale_elems * (cfg_.head_dim / 2))
        : (kv_scale_elems * cfg_.head_dim * (kv_fp8_ ? sizeof(u8) : sizeof(__nv_bfloat16)));
    d_k_caches_.resize(cfg_.num_layers, nullptr);
    d_v_caches_.resize(cfg_.num_layers, nullptr);
    if (kv_fp8_ || is_i4_kv) {
        d_k_scales_.resize(cfg_.num_layers, nullptr);
        d_v_scales_.resize(cfg_.num_layers, nullptr);
    }
    for (u32 l = 0; l < cfg_.num_layers; ++l) {
        FLINT_CUDA_CHECK(cudaMalloc(&d_k_caches_[l], k_layer_bytes));
        FLINT_CUDA_CHECK(cudaMalloc(&d_v_caches_[l], v_layer_bytes));
        if (kv_fp8_ || is_i4_kv) {
            FLINT_CUDA_CHECK(cudaMalloc(&d_k_scales_[l], kv_scale_elems * sizeof(f32)));
            FLINT_CUDA_CHECK(cudaMalloc(&d_v_scales_[l], kv_scale_elems * sizeof(f32)));
        }
    }
    FLINT_CUDA_CHECK(cudaMalloc(&d_block_table_, (max_seq / kKvBlockSize + 64) * sizeof(i32)));

    // Allocate activation & scratch buffers (2048-token chunks cut W4A16 prefill dequant passes 4x)
    max_batch_tokens_ = 2048;
    const u32 M = max_batch_tokens_;
    const u32 H = cfg_.hidden_size;
    const u32 qkv_dim = cfg_.q_dim() + 2 * cfg_.kv_dim();
    FLINT_CUDA_CHECK(cudaMalloc(&d_x_,        static_cast<size_t>(M) * H * sizeof(__nv_bfloat16)));
    FLINT_CUDA_CHECK(cudaMalloc(&d_norm_,     static_cast<size_t>(M) * H * sizeof(__nv_bfloat16)));
    FLINT_CUDA_CHECK(cudaMalloc(&d_qkv_,      static_cast<size_t>(M) * qkv_dim * sizeof(__nv_bfloat16)));
    FLINT_CUDA_CHECK(cudaMalloc(&d_q_,        static_cast<size_t>(M) * cfg_.q_dim() * sizeof(__nv_bfloat16)));
    FLINT_CUDA_CHECK(cudaMalloc(&d_attn_out_, static_cast<size_t>(M) * cfg_.q_dim() * sizeof(__nv_bfloat16)));
    FLINT_CUDA_CHECK(cudaMalloc(&d_proj_out_, static_cast<size_t>(M) * H * sizeof(__nv_bfloat16)));
    FLINT_CUDA_CHECK(cudaMalloc(&d_mlp_act_,  static_cast<size_t>(M) * cfg_.intermediate_size * sizeof(__nv_bfloat16)));
    FLINT_CUDA_CHECK(cudaMalloc(&d_mlp_gu_tmp_, static_cast<size_t>(M) * 2 * cfg_.intermediate_size * sizeof(__nv_bfloat16)));
    FLINT_CUDA_CHECK(cudaMalloc(&d_logits_,   static_cast<size_t>(16) * cfg_.vocab_size * sizeof(__nv_bfloat16)));

    if (quant_ != QuantMode::BF16) {
        size_t max_mat_elems = std::max<size_t>(
            static_cast<size_t>(2) * cfg_.intermediate_size * H,
            static_cast<size_t>(qkv_dim) * H);
        FLINT_CUDA_CHECK(cudaMalloc(&d_dequant_tmp_, max_mat_elems * sizeof(__nv_bfloat16)));
    }
    if (quant_ == QuantMode::FP8 || quant_ == QuantMode::W4A16) {
        cublaslt_ws_bytes_ = 32ULL * 1024 * 1024;
        FLINT_CUDA_CHECK(cudaMalloc(&d_cublaslt_ws_, cublaslt_ws_bytes_));
        size_t max_k = std::max<size_t>(H, cfg_.intermediate_size);
        size_t max_n = std::max<size_t>(2 * cfg_.intermediate_size, qkv_dim);
        FLINT_CUDA_CHECK(cudaMalloc(&d_x_fp8_, static_cast<size_t>(M + 16) * max_k));
        FLINT_CUDA_CHECK(cudaMalloc(&d_fp8_out_pad_, static_cast<size_t>(M + 16) * max_n * sizeof(__nv_bfloat16)));
        FLINT_CUDA_CHECK(cudaMalloc(&d_fp8_scale_, 4 * sizeof(f32)));
        f32 one_val = 1.0f;
        FLINT_CUDA_CHECK(cudaMemcpyAsync(d_fp8_scale_ + 2, &one_val, sizeof(f32), cudaMemcpyHostToDevice, stream_));
    }

    // Split-K FlashDecoding scratch (128 splits * 4 KV heads = 512 blocks = 4 exact waves on RTX 4090)
    flash_num_splits_ = 128;
    FLINT_CUDA_CHECK(cudaMalloc(&d_split_o_,  static_cast<size_t>(16) * cfg_.num_q_heads * flash_num_splits_ * cfg_.head_dim * sizeof(f32)));
    FLINT_CUDA_CHECK(cudaMalloc(&d_split_ml_, static_cast<size_t>(16) * cfg_.num_q_heads * flash_num_splits_ * 2 * sizeof(f32)));

    FLINT_CUDA_CHECK(cudaMalloc(&d_token_ids_,  M * sizeof(i32)));
    FLINT_CUDA_CHECK(cudaMalloc(&d_out_tokens_, 16 * sizeof(i32)));
    FLINT_CUDA_CHECK(cudaMalloc(&d_out_conf_,   16 * sizeof(f32)));
    FLINT_CUDA_CHECK(cudaMalloc(&d_decode_pos_, sizeof(i32)));
    FLINT_CUDA_CHECK(cudaHostAlloc(&h_pinned_token_, 16 * sizeof(i32), cudaHostAllocDefault));
    FLINT_CUDA_CHECK(cudaHostAlloc(&h_pinned_conf_,  16 * sizeof(f32), cudaHostAllocDefault));

    // Build Persistent Cooperative Megakernel layer descriptor table (W4A16 + FP8 LM head, head_dim=128)
    if (quant_ == QuantMode::W4A16 && d_lm_head_fp8_ && !cfg_.has_qk_norm &&
        cfg_.head_dim == 128 && cfg_.num_kv_heads == 4 &&
        (kv_mode_ == KvQuantMode::BF16 || kv_mode_ == KvQuantMode::INT8)) {
        std::vector<cuda::MegakernelLayerDesc> h_mk(cfg_.num_layers);
        for (u32 l = 0; l < cfg_.num_layers; ++l) {
            const auto& dl = d_layers_[l];
            h_mk[l].input_norm     = dl.input_norm;
            h_mk[l].post_attn_norm = dl.post_attn_norm;
            h_mk[l].qkv_bias       = dl.qkv_bias;
            h_mk[l].w_qkv          = static_cast<const u32*>(dl.w_qkv);
            h_mk[l].s_qkv          = static_cast<const __nv_bfloat162*>(dl.s_qkv);
            h_mk[l].w_o            = static_cast<const u32*>(dl.w_o);
            h_mk[l].s_o            = static_cast<const __nv_bfloat162*>(dl.s_o);
            h_mk[l].w_gu           = static_cast<const u32*>(dl.w_gu);
            h_mk[l].s_gu           = static_cast<const __nv_bfloat162*>(dl.s_gu);
            h_mk[l].w_down         = static_cast<const u32*>(dl.w_down);
            h_mk[l].s_down         = static_cast<const __nv_bfloat162*>(dl.s_down);
            h_mk[l].k_cache        = d_k_caches_[l];
            h_mk[l].v_cache        = d_v_caches_[l];
            h_mk[l].k_scale        = kv_fp8_ ? d_k_scales_[l] : nullptr;
            h_mk[l].v_scale        = kv_fp8_ ? d_v_scales_[l] : nullptr;
        }
        FLINT_CUDA_CHECK(cudaMalloc(&d_mk_layers_, h_mk.size() * sizeof(cuda::MegakernelLayerDesc)));
        FLINT_CUDA_CHECK(cudaMemcpyAsync(d_mk_layers_, h_mk.data(),
                                         h_mk.size() * sizeof(cuda::MegakernelLayerDesc),
                                         cudaMemcpyHostToDevice, stream_));
        FLINT_CUDA_CHECK(cudaMalloc(&d_mk_argmax_val_, 512 * sizeof(f32)));
        FLINT_CUDA_CHECK(cudaMalloc(&d_mk_argmax_idx_, 512 * sizeof(i32)));
    }

    FLINT_CUDA_CHECK(cudaStreamSynchronize(stream_));
    return true;
}

bool InferenceEngine::run_fp8_cublaslt_gemm(const __nv_bfloat16* x, const u8* w_fp8, const f32* row_scales,
                                            const __nv_bfloat16* bias, __nv_bfloat16* out,
                                            u32 M, u32 N, u32 K) {
    if (!cublaslt_ || !d_x_fp8_ || !d_fp8_scale_ || (K % 16 != 0) || (N % 16 != 0)) return false;
    u32 M_pad = (M + 15u) & ~15u;
    if (M_pad > max_batch_tokens_ + 16) return false;

    cuda::launch_quant_bf16_to_fp8_pertensor(x, d_x_fp8_, d_fp8_scale_, M, M_pad, K, stream_);

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    bool ok = false;

    do {
        if (cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) break;
        cublasOperation_t trans_a = CUBLAS_OP_T, trans_b = CUBLAS_OP_N;
        int8_t fast_acc = 1;
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_acc, sizeof(fast_acc));
        const f32* a_scale_ptr = d_fp8_scale_ + 2; // 1.0f (per-row weight scales applied after matmul)
        const f32* b_scale_ptr = d_fp8_scale_ + 1; // dynamic x activation scale
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr));
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr));

        // Column-major equivalent: D^T[N, M_pad] = W^T_col[N, K] @ X^T[K, M_pad]
        // A is W_fp8 stored [N, K] row-major == [K, N] col-major (lda=K), op(A)=T -> [N, K]
        // B is X_fp8 stored [M_pad, K] row-major == [K, M_pad] col-major (ldb=K), op(B)=N -> [K, M_pad]
        if (cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, K, N, K) != CUBLAS_STATUS_SUCCESS) break;
        if (cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, K, M_pad, K) != CUBLAS_STATUS_SUCCESS) break;
        if (cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, N, M_pad, N) != CUBLAS_STATUS_SUCCESS) break;

        if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) break;
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &cublaslt_ws_bytes_, sizeof(cublaslt_ws_bytes_));
        cublasLtMatmulHeuristicResult_t heur{};
        int returned = 0;
        if (cublasLtMatmulAlgoGetHeuristic(cublaslt_, op_desc, a_desc, b_desc, c_desc, c_desc,
                                           pref, 1, &heur, &returned) != CUBLAS_STATUS_SUCCESS || returned == 0) {
            break;
        }
        __nv_bfloat16* dst_buf = (M_pad == M) ? out : d_fp8_out_pad_;
        f32 alpha = 1.0f, beta = 0.0f;
        if (cublasLtMatmul(cublaslt_, op_desc, &alpha,
                           w_fp8, a_desc, d_x_fp8_, b_desc,
                           &beta, dst_buf, c_desc, dst_buf, c_desc,
                           &heur.algo, d_cublaslt_ws_, cublaslt_ws_bytes_, stream_) != CUBLAS_STATUS_SUCCESS) {
            break;
        }
        if (M_pad != M) {
            FLINT_CUDA_CHECK(cudaMemcpyAsync(out, dst_buf, static_cast<size_t>(M) * N * sizeof(__nv_bfloat16),
                                             cudaMemcpyDeviceToDevice, stream_));
        }
        cuda::launch_apply_row_scales_bf16(out, row_scales, bias, M, N, stream_);
        ok = true;
    } while (false);

    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
    if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
    if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
    if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    return ok;
}

void InferenceEngine::run_linear_cuda(const __nv_bfloat16* x, void* w, void* s,
                                      const __nv_bfloat16* bias, __nv_bfloat16* out,
                                      u32 M, u32 N, u32 K) {
    // Use custom roofline GEMV for M <= 8 (BF16) and micro-batched GEMV for M <= 32 (W4A16) or M < 16 (FP8)
    if (M <= 8 || (quant_ == QuantMode::W4A16 && M <= 32) || (quant_ == QuantMode::FP8 && M < 16)) {
        u32 off = 0;
        while (off < M) {
            u32 mb = std::min<u32>(8u, M - off);
            const __nv_bfloat16* xm = x + static_cast<size_t>(off) * K;
            __nv_bfloat16* om = out + static_cast<size_t>(off) * N;
            if (quant_ == QuantMode::BF16) {
                cuda::launch_gemv_bf16(xm, static_cast<const __nv_bfloat16*>(w), bias, om, mb, N, K, stream_);
            } else if (quant_ == QuantMode::W4A16) {
                cuda::launch_gemv_w4a16(xm, static_cast<const u32*>(w), static_cast<const __nv_bfloat162*>(s),
                                        bias, om, mb, N, K, stream_);
            } else {
                cuda::launch_gemv_fp8(xm, static_cast<const u8*>(w), static_cast<const f32*>(s),
                                      bias, om, mb, N, K, stream_);
            }
            off += mb;
        }
        return;
    }

    // Native Ada/Hopper/Blackwell FP8 Tensor-Core GEMM (cublasLtMatmul E4M3 x E4M3 -> BF16, ~330+ TFLOPS)
    if (quant_ == QuantMode::FP8 &&
        run_fp8_cublaslt_gemm(x, static_cast<const u8*>(w), static_cast<const f32*>(s), bias, out, M, N, K)) {
        return;
    }
    // W4A16 prefill (M >= 32): dequantize W4A16 -> per-row FP8 inside d_dequant_tmp_ (half the bytes) -> 2x faster FP8 Tensor Cores
    if (quant_ == QuantMode::W4A16 && M >= 32 && d_x_fp8_ && d_dequant_tmp_) {
        u8*  w_fp8 = reinterpret_cast<u8*>(d_dequant_tmp_);
        f32* s_fp8 = reinterpret_cast<f32*>(w_fp8 + ((static_cast<size_t>(N) * K + 255u) & ~size_t(255u)));
        cuda::launch_dequant_w4a16_to_fp8(static_cast<const u32*>(w),
                                          static_cast<const __nv_bfloat162*>(s),
                                          w_fp8, s_fp8, N, K, stream_);
        if (run_fp8_cublaslt_gemm(x, w_fp8, s_fp8, bias, out, M, N, K)) {
            return;
        }
    }

    // Compute-bound prefill fallback: BF16 cuBLAS Tensor Core GEMM
    const __nv_bfloat16* w_bf16 = static_cast<const __nv_bfloat16*>(w);
    if (quant_ == QuantMode::W4A16) {
        cuda::launch_dequant_w4a16_to_bf16(static_cast<const u32*>(w),
                                           static_cast<const __nv_bfloat162*>(s),
                                           d_dequant_tmp_, N, K, stream_);
        w_bf16 = d_dequant_tmp_;
    } else if (quant_ == QuantMode::FP8) {
        cuda::launch_dequant_fp8_to_bf16(static_cast<const u8*>(w),
                                         static_cast<const f32*>(s),
                                         d_dequant_tmp_, N, K, stream_);
        w_bf16 = d_dequant_tmp_;
    }

    // Row-major out[M, N] = x[M, K] @ W[N, K]^T
    f32 alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(cublas_,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                 &alpha,
                 w_bf16, CUDA_R_16BF, static_cast<int>(K),
                 x,      CUDA_R_16BF, static_cast<int>(K),
                 &beta,
                 out,    CUDA_R_16BF, static_cast<int>(N),
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (bias) {
        cuda::launch_add_bias_bf16(out, bias, M, N, stream_);
    }
}

void InferenceEngine::run_gate_up_silu_cuda(const __nv_bfloat16* x, void* w, void* s,
                                            __nv_bfloat16* out, u32 M, u32 inter, u32 K) {
    if (M <= 8 || (quant_ == QuantMode::W4A16 && M <= 32) || (quant_ == QuantMode::FP8 && M < 16)) {
        u32 off = 0;
        while (off < M) {
            u32 mb = std::min<u32>(8u, M - off);
            const __nv_bfloat16* xm = x + static_cast<size_t>(off) * K;
            __nv_bfloat16* om = out + static_cast<size_t>(off) * inter;
            if (quant_ == QuantMode::BF16) {
                cuda::launch_fused_gate_up_silu_bf16(xm, static_cast<const __nv_bfloat16*>(w), om, mb, inter, K, stream_);
            } else if (quant_ == QuantMode::W4A16) {
                cuda::launch_fused_gate_up_silu_w4a16(xm, static_cast<const u32*>(w),
                                                      static_cast<const __nv_bfloat162*>(s), om, mb, inter, K, stream_);
            } else {
                cuda::launch_fused_gate_up_silu_fp8(xm, static_cast<const u8*>(w),
                                                    static_cast<const f32*>(s), om, mb, inter, K, stream_);
            }
            off += mb;
        }
        return;
    }
    run_linear_cuda(x, w, s, nullptr, d_mlp_gu_tmp_, M, 2 * inter, K);
    cuda::launch_silu_mul_prefill_bf16(d_mlp_gu_tmp_, out, M, inter, stream_);
}

void InferenceEngine::run_forward_cuda(u32 M, const i32* d_pos_ptr, u32 host_start_pos, bool is_prefill) {
    const u32 H = cfg_.hidden_size;
    const u32 qkv_dim = cfg_.q_dim() + 2 * cfg_.kv_dim();

    cuda::launch_embed_lookup_bf16(d_embed_, d_token_ids_, d_x_, M, H, stream_);

    for (u32 l = 0; l < cfg_.num_layers; ++l) {
        const auto& dl = d_layers_[l];

        // Layer 0 runs standalone RMSNorm; layers 1..L-1 already fused prev down_proj residual + input_norm
        if (l == 0) {
            cuda::launch_rmsnorm_bf16(d_x_, dl.input_norm, d_norm_, M, H, cfg_.rms_norm_eps, stream_);
        } else {
            cuda::launch_fused_residual_rmsnorm_bf16(d_x_, d_proj_out_, dl.input_norm, d_norm_,
                                                     M, H, cfg_.rms_norm_eps, stream_);
        }

        // Fused QKV projection
        run_linear_cuda(d_norm_, dl.w_qkv, dl.s_qkv, dl.qkv_bias, d_qkv_, M, qkv_dim, H);

        // Fused RoPE (+ optional Qwen3 QK-norm) + Paged KV Write (BF16, INT8, or Zero-Calib WHT4)
        const bool is_i4_kv = (kv_mode_ == KvQuantMode::WHT4 || kv_mode_ == KvQuantMode::INT4RAW);
        const bool use_wht  = (kv_mode_ == KvQuantMode::WHT4);
        if (is_i4_kv) {
            cuda::launch_fused_rope_kv_write_wht4(
                d_qkv_, d_q_, static_cast<u8*>(d_k_caches_[l]), static_cast<u8*>(d_v_caches_[l]),
                reinterpret_cast<__nv_bfloat162*>(d_k_scales_[l]),
                reinterpret_cast<__nv_bfloat162*>(d_v_scales_[l]),
                d_block_table_, d_inv_freq_, dl.q_norm, dl.k_norm,
                d_pos_ptr, host_start_pos, M,
                cfg_.num_q_heads, cfg_.num_kv_heads, cfg_.head_dim, kKvBlockSize, cfg_.rms_norm_eps, use_wht, stream_);
        } else if (kv_fp8_) {
            cuda::launch_fused_rope_kv_write_fp8(
                d_qkv_, d_q_, static_cast<u8*>(d_k_caches_[l]), static_cast<u8*>(d_v_caches_[l]),
                d_k_scales_[l], d_v_scales_[l],
                d_block_table_, d_inv_freq_, dl.q_norm, dl.k_norm,
                d_pos_ptr, host_start_pos, M,
                cfg_.num_q_heads, cfg_.num_kv_heads, cfg_.head_dim, kKvBlockSize, cfg_.rms_norm_eps, stream_);
        } else {
            cuda::launch_fused_rope_kv_write_bf16(
                d_qkv_, d_q_, static_cast<__nv_bfloat16*>(d_k_caches_[l]), static_cast<__nv_bfloat16*>(d_v_caches_[l]),
                d_block_table_, d_inv_freq_, dl.q_norm, dl.k_norm,
                d_pos_ptr, host_start_pos, M,
                cfg_.num_q_heads, cfg_.num_kv_heads, cfg_.head_dim, kKvBlockSize, cfg_.rms_norm_eps, stream_);
        }

        // Attention: Split-K FlashDecoding for decode/spec (M<=8) AND short prefix-cache tails
        // (M<=32 attending over >=1024 cached tokens — avoids 28-block SM underutilization in W5)
        if ((!is_prefill && M <= 8) || (M <= 32 && host_start_pos >= 1024 && !d_pos_ptr)) {
            u32 splits = flash_num_splits_;
            if (!d_pos_ptr) {
                splits = (host_start_pos < 1024) ? 16u : (host_start_pos < 4096 ? 32u : flash_num_splits_);
            } else {
                splits = active_graph_splits_;
            }
            for (u32 off = 0; off < M; off += 8) {
                u32 mb = std::min<u32>(8u, M - off);
                const __nv_bfloat16* qm = d_q_ + static_cast<size_t>(off) * cfg_.q_dim();
                __nv_bfloat16* om = d_attn_out_ + static_cast<size_t>(off) * cfg_.q_dim();
                if (is_i4_kv) {
                    cuda::launch_flash_decode_wht4(
                        qm, static_cast<const u8*>(d_k_caches_[l]), static_cast<const u8*>(d_v_caches_[l]),
                        reinterpret_cast<const __nv_bfloat162*>(d_k_scales_[l]),
                        reinterpret_cast<const __nv_bfloat162*>(d_v_scales_[l]),
                        d_block_table_, om, d_split_o_, d_split_ml_, d_pos_ptr, host_start_pos + off,
                        mb, cfg_.num_q_heads, cfg_.num_kv_heads, cfg_.head_dim,
                        kKvBlockSize, splits, use_wht, stream_);
                } else if (kv_fp8_) {
                    cuda::launch_flash_decode_fp8kv(
                        qm, static_cast<const u8*>(d_k_caches_[l]), static_cast<const u8*>(d_v_caches_[l]),
                        d_k_scales_[l], d_v_scales_[l],
                        d_block_table_, om, d_split_o_, d_split_ml_, d_pos_ptr, host_start_pos + off,
                        mb, cfg_.num_q_heads, cfg_.num_kv_heads, cfg_.head_dim,
                        kKvBlockSize, splits, stream_);
                } else {
                    cuda::launch_flash_decode_bf16(
                        qm, static_cast<const __nv_bfloat16*>(d_k_caches_[l]), static_cast<const __nv_bfloat16*>(d_v_caches_[l]),
                        d_block_table_, om, d_split_o_, d_split_ml_, d_pos_ptr, host_start_pos + off,
                        mb, cfg_.num_q_heads, cfg_.num_kv_heads, cfg_.head_dim,
                        kKvBlockSize, splits, stream_);
                }
            }
        } else {
            if (is_i4_kv) {
                cuda::launch_flash_prefill_wht4(
                    d_q_, static_cast<const u8*>(d_k_caches_[l]), static_cast<const u8*>(d_v_caches_[l]),
                    reinterpret_cast<const __nv_bfloat162*>(d_k_scales_[l]),
                    reinterpret_cast<const __nv_bfloat162*>(d_v_scales_[l]),
                    d_block_table_, d_attn_out_, host_start_pos, M,
                    cfg_.num_q_heads, cfg_.num_kv_heads, cfg_.head_dim, kKvBlockSize, use_wht, stream_);
            } else if (kv_fp8_) {
                cuda::launch_flash_prefill_fp8kv(
                    d_q_, static_cast<const u8*>(d_k_caches_[l]), static_cast<const u8*>(d_v_caches_[l]),
                    d_k_scales_[l], d_v_scales_[l],
                    d_block_table_, d_attn_out_, host_start_pos, M,
                    cfg_.num_q_heads, cfg_.num_kv_heads, cfg_.head_dim, kKvBlockSize, stream_);
            } else {
                cuda::launch_flash_prefill_bf16(
                    d_q_, static_cast<const __nv_bfloat16*>(d_k_caches_[l]), static_cast<const __nv_bfloat16*>(d_v_caches_[l]),
                    d_block_table_, d_attn_out_, host_start_pos, M,
                    cfg_.num_q_heads, cfg_.num_kv_heads, cfg_.head_dim, kKvBlockSize, stream_);
            }
        }

        // Output projection
        run_linear_cuda(d_attn_out_, dl.w_o, dl.s_o, nullptr, d_proj_out_, M, H, cfg_.q_dim());

        // Fused Residual Add + Post-Attention RMSNorm
        cuda::launch_fused_residual_rmsnorm_bf16(d_x_, d_proj_out_, dl.post_attn_norm, d_norm_,
                                                 M, H, cfg_.rms_norm_eps, stream_);

        // Fused Gate + Up + SiLU*Mul
        run_gate_up_silu_cuda(d_norm_, dl.w_gu, dl.s_gu, d_mlp_act_, M, cfg_.intermediate_size, H);

        // Down projection (residual add fused into next layer's input_norm, or final_norm after last layer)
        run_linear_cuda(d_mlp_act_, dl.w_down, dl.s_down, nullptr, d_proj_out_, M, H, cfg_.intermediate_size);
    }

    // Final Fused Residual Add + Final RMSNorm
    cuda::launch_fused_residual_rmsnorm_bf16(d_x_, d_proj_out_, d_final_norm_, d_norm_,
                                             M, H, cfg_.rms_norm_eps, stream_);

    // LM Head (BF16 for exact bf16 packs, FP8 E4M3 for w4a16/fp8 packs):
    // for prefill, compute logits ONLY for the final token `M-1`
    const __nv_bfloat16* lm_in = is_prefill ? (d_norm_ + static_cast<size_t>(M - 1) * H) : d_norm_;
    u32 lm_M = is_prefill ? 1 : M;
    if (d_lm_head_fp8_ && d_lm_head_scales_) {
        cuda::launch_gemv_fp8(lm_in, d_lm_head_fp8_, d_lm_head_scales_, nullptr,
                              d_logits_, lm_M, cfg_.vocab_size, H, stream_);
    } else {
        const __nv_bfloat16* head_w = cfg_.tie_word_embeddings ? d_embed_ : d_lm_head_;
        cuda::launch_gemv_bf16(lm_in, head_w, nullptr, d_logits_, lm_M, cfg_.vocab_size, H, stream_);
    }
}

bool InferenceEngine::draft_begin_sequence(std::span<const i32> prompt_ids, u32 max_gen_tokens) {
    kv_mgr_.clear();
    active_draft_blocks_.clear();
    active_draft_committed_ = 0;
    u32 needed = static_cast<u32>(prompt_ids.size()) + max_gen_tokens + 32;
    u32 n_blocks = (needed + kKvBlockSize - 1) / kKvBlockSize;
    active_draft_blocks_.reserve(n_blocks);
    for (u32 b = 0; b < n_blocks; ++b) {
        i32 blk = kv_mgr_.allocate_block();
        if (blk < 0) return false;
        active_draft_blocks_.push_back(blk);
    }
    FLINT_CUDA_CHECK(cudaMemcpyAsync(d_block_table_, active_draft_blocks_.data(),
                                     active_draft_blocks_.size() * sizeof(i32),
                                     cudaMemcpyHostToDevice, stream_));
    u32 pos = 0;
    while (pos < prompt_ids.size()) {
        u32 chunk = std::min<u32>(max_batch_tokens_, static_cast<u32>(prompt_ids.size() - pos));
        FLINT_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, prompt_ids.data() + pos,
                                         chunk * sizeof(i32), cudaMemcpyHostToDevice, stream_));
        run_forward_cuda(chunk, nullptr, pos, true);
        pos += chunk;
    }
    return true;
}

std::vector<i32> InferenceEngine::draft_propose(std::span<const i32> newly_accepted, u32 sync_pos, u32 K) {
    std::vector<i32> drafts;
    if (newly_accepted.empty() || K == 0) return drafts;
    drafts.reserve(K);

    constexpr f32 kConfThreshold = 0.65f; // Confidence-gated early exit prevents wasted low-confidence drafts

    u32 m_sync = static_cast<u32>(newly_accepted.size());
    u32 pos_off = 0;
    while (pos_off < m_sync) {
        u32 chunk = std::min<u32>(max_batch_tokens_, m_sync - pos_off);
        FLINT_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, newly_accepted.data() + pos_off, chunk * sizeof(i32),
                                         cudaMemcpyHostToDevice, stream_));
        // is_prefill=true computes LM head ONLY for the final token `chunk-1` into d_logits_[0..V-1]
        run_forward_cuda(chunk, nullptr, sync_pos + pos_off, chunk > 1);
        pos_off += chunk;
    }
    cuda::launch_argmax_conf_bf16(d_logits_, d_out_tokens_, d_out_conf_, 1, cfg_.vocab_size, stream_);
    FLINT_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_, sizeof(i32), cudaMemcpyDeviceToHost, stream_));
    FLINT_CUDA_CHECK(cudaMemcpyAsync(h_pinned_conf_,  d_out_conf_,   sizeof(f32), cudaMemcpyDeviceToHost, stream_));
    FLINT_CUDA_CHECK(cudaStreamSynchronize(stream_));

    i32 cur = h_pinned_token_[0];
    f32 conf = h_pinned_conf_[0];
    drafts.push_back(cur);
    if (conf < kConfThreshold || K == 1) return drafts;

    // Propose remaining K - 1 tokens with early confidence gating
    u32 dpos = sync_pos + m_sync;
    for (u32 step = 1; step < K; ++step) {
        FLINT_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, &cur, sizeof(i32), cudaMemcpyHostToDevice, stream_));
        run_forward_cuda(1, nullptr, dpos++, false);
        cuda::launch_argmax_conf_bf16(d_logits_, d_out_tokens_, d_out_conf_, 1, cfg_.vocab_size, stream_);
        FLINT_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_, sizeof(i32), cudaMemcpyDeviceToHost, stream_));
        FLINT_CUDA_CHECK(cudaMemcpyAsync(h_pinned_conf_,  d_out_conf_,   sizeof(f32), cudaMemcpyDeviceToHost, stream_));
        FLINT_CUDA_CHECK(cudaStreamSynchronize(stream_));
        cur = h_pinned_token_[0];
        conf = h_pinned_conf_[0];
        drafts.push_back(cur);
        if (conf < kConfThreshold) break;
    }
    return drafts;
}

void InferenceEngine::draft_end_sequence() {
    kv_mgr_.release_blocks(active_draft_blocks_, 0);
    active_draft_blocks_.clear();
}

void InferenceEngine::destroy_decode_graph() {
    if (decode_graph_exec_) { cudaGraphExecDestroy(decode_graph_exec_); decode_graph_exec_ = nullptr; }
    if (decode_graph_) { cudaGraphDestroy(decode_graph_); decode_graph_ = nullptr; }
    graph_captured_ = false;
}

void InferenceEngine::ensure_decode_graph() {
    if (graph_captured_) return;
    // Capture full M=1 decode step (all layers + LM head + argmax + state advance) into 1 CUDA Graph
    FLINT_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    run_forward_cuda(1, d_decode_pos_, 0, false);
    cuda::launch_argmax_bf16(d_logits_, d_out_tokens_, 1, cfg_.vocab_size, stream_);
    cuda::launch_advance_decode_state(d_decode_pos_, d_token_ids_, d_out_tokens_, stream_);
    FLINT_CUDA_CHECK(cudaStreamEndCapture(stream_, &decode_graph_));
    FLINT_CUDA_CHECK(cudaGraphInstantiate(&decode_graph_exec_, decode_graph_, nullptr, nullptr, 0));
    graph_captured_ = true;
}

#else
bool InferenceEngine::draft_begin_sequence(std::span<const i32>, u32) { return false; }
std::vector<i32> InferenceEngine::draft_propose(std::span<const i32>, u32, u32) { return {}; }
void InferenceEngine::draft_end_sequence() {}
#endif // FLINT_CUDA

bool InferenceEngine::eval_logits(std::span<const i32> prompt_ids,
                                  std::vector<f32>& logits_out,
                                  std::string& err) {
    if (prompt_ids.empty()) { err = "empty prompt"; return false; }
    if (inkling2_) {
        if (!inkling2_->eval_logits(prompt_ids, logits_out)) { err = "Inkling2 eval_logits failed"; return false; }
        return true;
    }

#if defined(FLINT_CUDA) && FLINT_CUDA == 1
    if (use_cuda_) {
        kv_mgr_.clear();
        u32 n_blocks = static_cast<u32>((prompt_ids.size() + kKvBlockSize - 1) / kKvBlockSize);
        std::vector<i32> btable;
        btable.reserve(n_blocks);
        for (u32 b = 0; b < n_blocks; ++b) btable.push_back(kv_mgr_.allocate_block());
        FLINT_CUDA_CHECK(cudaMemcpyAsync(d_block_table_, btable.data(), btable.size() * sizeof(i32),
                                         cudaMemcpyHostToDevice, stream_));

        // Chunked prefill in slices of max_batch_tokens_
        u32 pos = 0;
        while (pos < prompt_ids.size()) {
            u32 chunk = std::min<u32>(max_batch_tokens_, static_cast<u32>(prompt_ids.size() - pos));
            FLINT_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, prompt_ids.data() + pos, chunk * sizeof(i32),
                                             cudaMemcpyHostToDevice, stream_));
            run_forward_cuda(chunk, nullptr, pos, true);
            pos += chunk;
        }
        std::vector<u16> h_bf16(cfg_.vocab_size);
        FLINT_CUDA_CHECK(cudaMemcpyAsync(h_bf16.data(), d_logits_, cfg_.vocab_size * sizeof(u16),
                                         cudaMemcpyDeviceToHost, stream_));
        FLINT_CUDA_CHECK(cudaStreamSynchronize(stream_));
        logits_out.resize(cfg_.vocab_size);
        for (u32 i = 0; i < cfg_.vocab_size; ++i) logits_out[i] = bf16_to_f32(h_bf16[i]);
        kv_mgr_.release_blocks(btable, 0);
        return true;
    }
#endif

    cpu_ref_->forward_prefill(prompt_ids, logits_out);
    return true;
}

GenerateResult InferenceEngine::generate(
    std::string_view prompt,
    const GenerateOptions& gopts,
    const std::function<bool(i32, std::string_view)>& on_token) {

    std::string formatted;
    std::string_view input = prompt;
    if (gopts.use_chat_template) {
        formatted = tokenizer_.apply_chat_template(
            gopts.system_prompt, prompt, cfg_.arch == ModelArch::Llama);
        input = formatted;
    }
    auto ids = tokenizer_.encode(input, false);
    return generate_ids(ids, gopts, on_token);
}

GenerateResult InferenceEngine::generate_ids(
    std::span<const i32> prompt_ids,
    const GenerateOptions& gopts,
    const std::function<bool(i32, std::string_view)>& on_token) {

    GenerateResult res;
    if (prompt_ids.empty()) {
        res.error = "Empty prompt token sequence";
        return res;
    }
    if (inkling2_) {
        std::function<void(std::string_view)> cb;
        if (on_token) {
            cb = [&](std::string_view piece) { on_token(0, piece); };
        }
        auto r = inkling2_->generate_ids(prompt_ids, gopts, cb);
        r.stats.active_weight_bytes = active_weight_bytes_;
        return r;
    }
    res.stats.prompt_tokens = static_cast<u32>(prompt_ids.size());
    res.stats.active_weight_bytes = active_weight_bytes_;

    std::mt19937_64 rng(gopts.seed);
    std::vector<double> itls_ms;
    itls_ms.reserve(gopts.max_tokens);

#if defined(FLINT_CUDA) && FLINT_CUDA == 1
    if (use_cuda_) {
        Stopwatch ttft_sw;

        // 1. Radix Prefix KV Cache Lookup
        auto match = kv_mgr_.match_prefix(prompt_ids);
        res.stats.prefix_cached_tokens = match.matched_tokens;
        std::vector<i32> block_table = std::move(match.block_table);
        u32 committed_blocks = static_cast<u32>(block_table.size());
        u64 last_hash = match.last_block_hash;

        u32 total_needed_tokens = static_cast<u32>(prompt_ids.size()) + gopts.max_tokens + 16;
        u32 total_needed_blocks = (total_needed_tokens + kKvBlockSize - 1) / kKvBlockSize;
        while (block_table.size() < total_needed_blocks) {
            i32 b = kv_mgr_.allocate_block();
            if (b < 0) {
                kv_mgr_.release_blocks(block_table, committed_blocks);
                res.error = "Out of KV cache VRAM blocks";
                return res;
            }
            block_table.push_back(b);
        }
        FLINT_CUDA_CHECK(cudaMemcpyAsync(d_block_table_, block_table.data(),
                                         block_table.size() * sizeof(i32),
                                         cudaMemcpyHostToDevice, stream_));

        // 2. Prefill uncached suffix tokens in chunks of max_batch_tokens_
        u32 pos = match.matched_tokens;
        while (pos < prompt_ids.size()) {
            u32 chunk = std::min<u32>(max_batch_tokens_, static_cast<u32>(prompt_ids.size() - pos));
            FLINT_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, prompt_ids.data() + pos,
                                             chunk * sizeof(i32), cudaMemcpyHostToDevice, stream_));
            run_forward_cuda(chunk, nullptr, pos, true);
            pos += chunk;
        }

        // Commit newly prefilled full prompt blocks into Radix Prefix Cache
        u32 prompt_full_blocks = static_cast<u32>(prompt_ids.size()) / kKvBlockSize;
        for (u32 b = committed_blocks; b < prompt_full_blocks; ++b) {
            auto slice = prompt_ids.subspan(b * kKvBlockSize, kKvBlockSize);
            last_hash = kv_mgr_.insert_block(last_hash, slice, block_table[b]);
        }
        committed_blocks = std::max(committed_blocks, prompt_full_blocks);

        // Sample first output token
        i32 next_tok = 0;
        if (gopts.temperature <= 0.0f) {
            cuda::launch_argmax_bf16(d_logits_, d_out_tokens_, 1, cfg_.vocab_size, stream_);
            FLINT_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_, sizeof(i32),
                                             cudaMemcpyDeviceToHost, stream_));
            FLINT_CUDA_CHECK(cudaStreamSynchronize(stream_));
            next_tok = h_pinned_token_[0];
        } else {
            std::vector<u16> h_bf16(cfg_.vocab_size);
            FLINT_CUDA_CHECK(cudaMemcpyAsync(h_bf16.data(), d_logits_, cfg_.vocab_size * sizeof(u16),
                                             cudaMemcpyDeviceToHost, stream_));
            FLINT_CUDA_CHECK(cudaStreamSynchronize(stream_));
            std::vector<f32> h_f32(cfg_.vocab_size);
            for (u32 i = 0; i < cfg_.vocab_size; ++i) h_f32[i] = bf16_to_f32(h_bf16[i]);
            next_tok = sample_logits(h_f32, gopts.temperature, gopts.top_p, rng);
        }

        res.stats.ttft_ms = ttft_sw.elapsed_ms();
        u32 uncached_prompt = static_cast<u32>(prompt_ids.size()) - match.matched_tokens;
        res.stats.prefill_tok_s = (res.stats.ttft_ms > 0.0)
            ? (static_cast<double>(uncached_prompt) / (res.stats.ttft_ms / 1000.0)) : 0.0;

        auto emit_token = [&](i32 tok) -> bool {
            res.output_ids.push_back(tok);
            std::string piece = tokenizer_.decode_token(tok, false);
            res.text += piece;
            if (on_token && !on_token(tok, piece)) return false;
            if (!gopts.ignore_eos && cfg_.is_eos(tok)) return false;
            return res.output_ids.size() < gopts.max_tokens;
        };

        bool keep_going = emit_token(next_tok);

        // 3. Autoregressive Decode Loop (Full-Step CUDA Graph OR Lossless Speculative Decoding)
        Stopwatch dec_sw;
        u32 cur_pos = static_cast<u32>(prompt_ids.size());
        u32 target_forward_steps = 0;

        if (keep_going && opts_.spec_mode == SpecMode::Draft && draft_engine_ && gopts.temperature <= 0.0f) {
            // Hybrid Lossless Speculative Decoding:
            // 1. Zero-cost N-gram lookup first (hits ~300 tok/s on code edits & FIM)
            // 2. Confidence-gated tiny draft model (Qwen2.5-0.5B) on greenfield tokens
            draft_engine_->draft_begin_sequence(prompt_ids, gopts.max_tokens);
            NgramDrafter ngram;
            ngram.ingest(prompt_ids);
            ngram.push(next_tok);

            std::vector<i32> pending_for_draft = {next_tok};
            u32 draft_sync_pos = cur_pos;
            std::vector<i32> step_in;

            while (keep_going) {
                Stopwatch tok_sw;
                u32 rem = gopts.max_tokens - static_cast<u32>(res.output_ids.size());
                u32 max_k = std::min<u32>({gopts.spec_k, 7u, rem > 0 ? rem - 1 : 0});

                auto drafts = (max_k > 0) ? ngram.draft(max_k) : std::vector<i32>{};
                bool used_ngram = (drafts.size() >= 3);
                if (!used_ngram && max_k > 0) {
                    // Sync any skipped ngram tokens + propose confidence-gated draft tokens
                    drafts = draft_engine_->draft_propose(pending_for_draft, draft_sync_pos, max_k);
                }
                u32 K = static_cast<u32>(drafts.size());

                step_in.clear();
                step_in.push_back(next_tok);
                for (i32 d : drafts) step_in.push_back(d);
                u32 M = K + 1;

                FLINT_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, step_in.data(), M * sizeof(i32),
                                                 cudaMemcpyHostToDevice, stream_));
                run_forward_cuda(M, nullptr, cur_pos, false);
                target_forward_steps++;
                cuda::launch_argmax_bf16(d_logits_, d_out_tokens_, M, cfg_.vocab_size, stream_);
                FLINT_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_, M * sizeof(i32),
                                                 cudaMemcpyDeviceToHost, stream_));
                FLINT_CUDA_CHECK(cudaStreamSynchronize(stream_));

                auto vr = verify_speculative_greedy(drafts, std::span<const i32>(h_pinned_token_, M));
                res.stats.spec_drafted += K;
                res.stats.spec_accepted += vr.accepted_drafts;

                double step_ms = tok_sw.elapsed_ms();
                for (size_t i = 0; i < vr.emitted.size(); ++i) {
                    itls_ms.push_back(step_ms / static_cast<double>(vr.emitted.size()));
                }

                // Keep draft KV cache 100% synchronized whether ngram or draft model proposed this step
                if (used_ngram) {
                    // Draft model didn't run this step -> append all emitted tokens to pending_for_draft
                    for (i32 et : vr.emitted) pending_for_draft.push_back(et);
                } else {
                    pending_for_draft.clear();
                    if (K > 0 && vr.accepted_drafts == K) {
                        draft_sync_pos = cur_pos + K;
                        pending_for_draft.push_back(drafts[K - 1]);
                        pending_for_draft.push_back(vr.emitted.back());
                    } else {
                        draft_sync_pos = cur_pos + 1 + vr.accepted_drafts;
                        pending_for_draft.push_back(vr.emitted.back());
                    }
                }

                cur_pos += (1 + vr.accepted_drafts);
                for (i32 et : vr.emitted) {
                    ngram.push(et);
                    next_tok = et;
                    keep_going = emit_token(et);
                    if (!keep_going) break;
                }
            }
            draft_engine_->draft_end_sequence();
        } else if (keep_going && opts_.spec_mode == SpecMode::Ngram && gopts.temperature <= 0.0f) {
            // Lossless N-Gram Speculative Decoding (M = K + 1 verification in one target pass)
            NgramDrafter drafter;
            drafter.ingest(prompt_ids);
            drafter.push(next_tok);

            std::vector<i32> step_in;
            while (keep_going) {
                Stopwatch tok_sw;
                u32 rem = gopts.max_tokens - static_cast<u32>(res.output_ids.size());
                u32 max_k = std::min<u32>({gopts.spec_k, 7u, rem > 0 ? rem - 1 : 0});
                auto drafts = (max_k > 0) ? drafter.draft(max_k) : std::vector<i32>{};
                u32 K = static_cast<u32>(drafts.size());

                step_in.clear();
                step_in.push_back(next_tok);
                for (i32 d : drafts) step_in.push_back(d);
                u32 M = K + 1;

                FLINT_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, step_in.data(), M * sizeof(i32),
                                                 cudaMemcpyHostToDevice, stream_));
                run_forward_cuda(M, nullptr, cur_pos, false);
                target_forward_steps++;
                cuda::launch_argmax_bf16(d_logits_, d_out_tokens_, M, cfg_.vocab_size, stream_);
                FLINT_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_, M * sizeof(i32),
                                                 cudaMemcpyDeviceToHost, stream_));
                FLINT_CUDA_CHECK(cudaStreamSynchronize(stream_));

                auto vr = verify_speculative_greedy(drafts, std::span<const i32>(h_pinned_token_, M));
                res.stats.spec_drafted += K;
                res.stats.spec_accepted += vr.accepted_drafts;

                double step_ms = tok_sw.elapsed_ms();
                for (size_t i = 0; i < vr.emitted.size(); ++i) {
                    itls_ms.push_back(step_ms / static_cast<double>(vr.emitted.size()));
                }

                // Advance cur_pos by exact number of accepted tokens (1 base + accepted_drafts)
                cur_pos += (1 + vr.accepted_drafts);
                for (i32 et : vr.emitted) {
                    drafter.push(et);
                    next_tok = et;
                    keep_going = emit_token(et);
                    if (!keep_going) break;
                }
            }
        } else if (keep_going && opts_.use_megakernel && d_mk_layers_ && gopts.temperature <= 0.0f) {
            // Single-Launch Persistent Cooperative Megakernel (1 launch = all 28 layers + LM head + Argmax)
            i32 init_pos = static_cast<i32>(cur_pos);
            FLINT_CUDA_CHECK(cudaMemcpyAsync(d_decode_pos_, &init_pos, sizeof(i32), cudaMemcpyHostToDevice, stream_));
            FLINT_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, &next_tok, sizeof(i32), cudaMemcpyHostToDevice, stream_));
            while (keep_going) {
                Stopwatch tok_sw;
                bool ok = cuda::launch_megakernel_w4a16_decode(
                    static_cast<const cuda::MegakernelLayerDesc*>(d_mk_layers_),
                    d_embed_, d_final_norm_, d_lm_head_fp8_, d_lm_head_scales_,
                    d_inv_freq_, d_block_table_,
                    d_x_, d_norm_, d_qkv_, d_q_, d_attn_out_, d_mlp_act_,
                    d_split_o_, d_split_ml_, d_mk_argmax_val_, d_mk_argmax_idx_,
                    d_decode_pos_, d_token_ids_, d_out_tokens_,
                    cfg_.num_layers, cfg_.hidden_size, cfg_.intermediate_size, cfg_.vocab_size,
                    cfg_.num_q_heads, cfg_.num_kv_heads, kKvBlockSize, cfg_.rms_norm_eps,
                    kv_fp8_, stream_);
                if (!ok) {
                    res.error = "Persistent cooperative megakernel launch failed";
                    break;
                }
                FLINT_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_, sizeof(i32), cudaMemcpyDeviceToHost, stream_));
                FLINT_CUDA_CHECK(cudaStreamSynchronize(stream_));
                itls_ms.push_back(tok_sw.elapsed_ms());
                next_tok = h_pinned_token_[0];
                cur_pos++;
                keep_going = emit_token(next_tok);
            }
        } else if (keep_going && opts_.use_cuda_graph && gopts.temperature <= 0.0f) {
            // Ultra-low-overhead Full-Step CUDA Graph Decode Loop (GQA-grouped adaptive Split-K)
            u32 target_splits = (cur_pos < 1024) ? 16u : (cur_pos < 4096 ? 32u : flash_num_splits_);
            if (graph_captured_ && active_graph_splits_ != target_splits) {
                destroy_decode_graph();
            }
            active_graph_splits_ = target_splits;
            i32 init_pos = static_cast<i32>(cur_pos);
            FLINT_CUDA_CHECK(cudaMemcpyAsync(d_decode_pos_, &init_pos, sizeof(i32),
                                             cudaMemcpyHostToDevice, stream_));
            FLINT_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, &next_tok, sizeof(i32),
                                             cudaMemcpyHostToDevice, stream_));
            ensure_decode_graph();

            while (keep_going) {
                Stopwatch tok_sw;
                FLINT_CUDA_CHECK(cudaGraphLaunch(decode_graph_exec_, stream_));
                FLINT_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_, sizeof(i32),
                                                 cudaMemcpyDeviceToHost, stream_));
                FLINT_CUDA_CHECK(cudaStreamSynchronize(stream_));
                itls_ms.push_back(tok_sw.elapsed_ms());
                next_tok = h_pinned_token_[0];
                cur_pos++;
                keep_going = emit_token(next_tok);
            }
        } else {
            // Eager decode loop (used for non-greedy sampling or --no-graph)
            std::vector<u16> h_bf16(cfg_.vocab_size);
            std::vector<f32> h_f32(cfg_.vocab_size);
            while (keep_going) {
                Stopwatch tok_sw;
                FLINT_CUDA_CHECK(cudaMemcpyAsync(d_token_ids_, &next_tok, sizeof(i32),
                                                 cudaMemcpyHostToDevice, stream_));
                run_forward_cuda(1, nullptr, cur_pos, false);
                if (gopts.temperature <= 0.0f) {
                    cuda::launch_argmax_bf16(d_logits_, d_out_tokens_, 1, cfg_.vocab_size, stream_);
                    FLINT_CUDA_CHECK(cudaMemcpyAsync(h_pinned_token_, d_out_tokens_, sizeof(i32),
                                                     cudaMemcpyDeviceToHost, stream_));
                    FLINT_CUDA_CHECK(cudaStreamSynchronize(stream_));
                    next_tok = h_pinned_token_[0];
                } else {
                    FLINT_CUDA_CHECK(cudaMemcpyAsync(h_bf16.data(), d_logits_, cfg_.vocab_size * sizeof(u16),
                                                     cudaMemcpyDeviceToHost, stream_));
                    FLINT_CUDA_CHECK(cudaStreamSynchronize(stream_));
                    for (u32 i = 0; i < cfg_.vocab_size; ++i) h_f32[i] = bf16_to_f32(h_bf16[i]);
                    next_tok = sample_logits(h_f32, gopts.temperature, gopts.top_p, rng);
                }
                itls_ms.push_back(tok_sw.elapsed_ms());
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
            double avg_seq = static_cast<double>(prompt_ids.size()) + 0.5 * dec_steps;
            // Exact physical KV bytes per (K+V) head pair: BF16=512B, INT8=264B, WHT4(K8+V4)=200B, INT4RAW(K4+V4)=136B
            double kv_pair_bytes = (kv_mode_ == KvQuantMode::WHT4)
                ? (static_cast<double>(cfg_.head_dim) * 1.5 + 8.0)
                : ((kv_mode_ == KvQuantMode::INT4RAW)
                    ? (static_cast<double>(cfg_.head_dim) + 8.0)
                    : (kv_fp8_ ? (2.0 * static_cast<double>(cfg_.head_dim) + 8.0)
                               : (4.0 * static_cast<double>(cfg_.head_dim))));
            double kv_bytes_per_step = static_cast<double>(cfg_.num_layers) * cfg_.num_kv_heads * kv_pair_bytes * avg_seq;
            double embed_row_bytes = static_cast<double>(cfg_.hidden_size) * 2.0; // 1 row lookup, NOT full table
            double bytes_per_step = static_cast<double>(active_weight_bytes_) + embed_row_bytes + kv_bytes_per_step;
            double decode_s = res.stats.decode_ms / 1000.0;
            // True physical HBM bandwidth based on actual target model forward passes (never >100% from speculation)
            res.stats.achieved_hbm_gbs = (static_cast<double>(phys_passes) * bytes_per_step / 1e9) / decode_s;
            if (gpu_peak_hbm_gbs_ > 0.0) {
                res.stats.pct_hbm_peak = 100.0 * res.stats.achieved_hbm_gbs / gpu_peak_hbm_gbs_;
            }
        }
        if (!itls_ms.empty()) {
            std::sort(itls_ms.begin(), itls_ms.end());
            res.stats.itl_p50_ms = itls_ms[itls_ms.size() / 2];
            res.stats.itl_p99_ms = itls_ms[std::min(itls_ms.size() - 1, static_cast<size_t>(itls_ms.size() * 0.99))];
        }
        if (res.stats.spec_drafted > 0) {
            res.stats.spec_accept_rate = static_cast<double>(res.stats.spec_accepted) /
                                         static_cast<double>(res.stats.spec_drafted);
        }

        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        res.stats.vram_used_mib = (total_b - free_b) / (1024 * 1024);
        res.stats.vram_total_mib = total_b / (1024 * 1024);
        res.stats.host_rss_mib = current_rss_kib() / 1024;

        kv_mgr_.release_blocks(block_table, committed_blocks);
        return res;
    }
#endif

    // CPU reference generation path
    Stopwatch ttft_sw;
    std::vector<f32> logits;
    cpu_ref_->forward_prefill(prompt_ids, logits);
    i32 next_tok = sample_logits(logits, gopts.temperature, gopts.top_p, rng);
    res.stats.ttft_ms = ttft_sw.elapsed_ms();
    res.stats.prefill_tok_s = (res.stats.ttft_ms > 0.0)
        ? (static_cast<double>(prompt_ids.size()) / (res.stats.ttft_ms / 1000.0)) : 0.0;

    auto emit = [&](i32 tok) -> bool {
        res.output_ids.push_back(tok);
        std::string piece = tokenizer_.decode_token(tok, false);
        res.text += piece;
        if (on_token && !on_token(tok, piece)) return false;
        if (!gopts.ignore_eos && cfg_.is_eos(tok)) return false;
        return res.output_ids.size() < gopts.max_tokens;
    };

    bool cont = emit(next_tok);
    Stopwatch dec_sw;
    u32 pos = static_cast<u32>(prompt_ids.size());
    while (cont) {
        Stopwatch tsw;
        cpu_ref_->forward_token(next_tok, pos++, logits);
        next_tok = sample_logits(logits, gopts.temperature, gopts.top_p, rng);
        itls_ms.push_back(tsw.elapsed_ms());
        cont = emit(next_tok);
    }
    res.stats.decode_ms = dec_sw.elapsed_ms();
    res.stats.generated_tokens = static_cast<u32>(res.output_ids.size());
    u32 dec_steps = (res.stats.generated_tokens > 1) ? (res.stats.generated_tokens - 1) : 0;
    if (dec_steps > 0 && res.stats.decode_ms > 0.0) {
        res.stats.decode_tok_s = static_cast<double>(dec_steps) / (res.stats.decode_ms / 1000.0);
    }
    res.stats.host_rss_mib = current_rss_kib() / 1024;
    return res;
}

} // namespace flint
