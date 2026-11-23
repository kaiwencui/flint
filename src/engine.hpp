#pragma once

#include "common.hpp"
#include "cpu_ref.hpp"
#include "inkling2_engine.hpp"
#include "kv_cache.hpp"
#include "model_config.hpp"
#include "speculative.hpp"
#include "tokenizer.hpp"
#include "weights.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

#if defined(FLINT_CUDA) && FLINT_CUDA == 1
  #include <cublas_v2.h>
  #include <cublasLt.h>
#endif

namespace flint {

struct EngineOptions {
    std::filesystem::path model_path;      // HF dir or .pack file (with tokenizer in same dir or --tokenizer)
    std::filesystem::path tokenizer_dir;   // Optional override if model_path is a .pack file
    std::filesystem::path draft_model_path;// Optional draft model for --spec draft:<path>
    QuantMode quant = QuantMode::BF16;
    SpecMode spec_mode = SpecMode::None;
    u32 max_seq_len = 8192;
    u32 max_kv_blocks = 0;                 // 0 = auto from max_seq_len + headroom
    bool kv_fp8 = false;                   // Dynamic-scaled 1-byte INT8 KV cache (--kv-int8 / --kv-fp8)
    KvQuantMode kv_mode = KvQuantMode::BF16; // BF16 | INT8 | WHT4 (zero-calib Walsh-Hadamard 4-bit) | INT4RAW
    bool use_cuda_graph = true;
    bool use_megakernel = false;           // Single-Launch Persistent Cooperative Megakernel (--megakernel)
    bool force_cpu = false;
};

struct GenerateOptions {
    u32 max_tokens = 256;
    f32 temperature = 0.0f; // 0 = greedy
    f32 top_p = 1.0f;
    u64 seed = 42;
    bool ignore_eos = false;
    bool use_chat_template = false;
    std::string system_prompt;
    u32 spec_k = 5; // Draft tokens per step when speculative decoding enabled
    std::string save_kv_path; // Optional path to dump post-prefill KV snapshot (--save-kv)
    std::string load_kv_path; // Optional path to restore prefilled KV snapshot (--load-kv)
};

struct GenerateStats {
    u32 prompt_tokens = 0;
    u32 prefix_cached_tokens = 0;
    u32 generated_tokens = 0;
    double ttft_ms = 0.0;
    double decode_ms = 0.0;
    double prefill_tok_s = 0.0;
    double decode_tok_s = 0.0;
    double itl_p50_ms = 0.0;
    double itl_p99_ms = 0.0;
    double achieved_hbm_gbs = 0.0;
    double pct_hbm_peak = 0.0;
    u64 active_weight_bytes = 0;
    u64 vram_used_mib = 0;
    u64 vram_total_mib = 0;
    u64 host_rss_mib = 0;
    u32 spec_drafted = 0;
    u32 spec_accepted = 0;
    double spec_accept_rate = 0.0;
};

struct GenerateResult {
    std::vector<i32> output_ids;
    std::string text;
    GenerateStats stats;
    std::string error;
};

class InferenceEngine {
public:
    InferenceEngine() = default;
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    bool load(const EngineOptions& opts, std::string& err);

    const ModelConfig& config() const noexcept { return cfg_; }
    const BpeTokenizer& tokenizer() const noexcept { return tokenizer_; }
    QuantMode quant_mode() const noexcept { return quant_; }
    double cold_start_ms() const noexcept { return cold_start_ms_; }
    u64 active_weight_bytes() const noexcept { return active_weight_bytes_; }
    bool using_cuda() const noexcept { return use_cuda_; }
    std::string gpu_name() const { return gpu_name_; }
    double gpu_peak_hbm_gbs() const noexcept { return gpu_peak_hbm_gbs_; }

    // Generate from raw prompt text (streaming callback invoked per token if non-null)
    GenerateResult generate(std::string_view prompt,
                            const GenerateOptions& gopts,
                            const std::function<bool(i32 token_id, std::string_view piece)>& on_token = nullptr);

    // Generate from pre-encoded token IDs (used by parity tests & benchmarks)
    GenerateResult generate_ids(std::span<const i32> prompt_ids,
                                const GenerateOptions& gopts,
                                const std::function<bool(i32 token_id, std::string_view piece)>& on_token = nullptr);

    // Single prefill returning raw FP32 logits for the final token (used for Gate 2 logit parity)
    bool eval_logits(std::span<const i32> prompt_ids, std::vector<f32>& logits_out, std::string& err);

    void clear_kv_cache();

    // Draft-model helpers for lossless `--spec draft:<path>` speculative decoding
    bool draft_begin_sequence(std::span<const i32> prompt_ids, u32 max_gen_tokens);
    std::vector<i32> draft_propose(std::span<const i32> newly_accepted, u32 sync_pos, u32 K);
    void draft_end_sequence();

private:
    bool load_cuda_weights(const EngineOptions& opts, std::string& err);
    void free_cuda_resources();

#if defined(FLINT_CUDA) && FLINT_CUDA == 1
    struct CudaLayer {
        __nv_bfloat16* input_norm = nullptr;
        __nv_bfloat16* post_attn_norm = nullptr;
        __nv_bfloat16* q_norm = nullptr;
        __nv_bfloat16* k_norm = nullptr;
        __nv_bfloat16* qkv_bias = nullptr;

        // Quantized or BF16 linear weights (4 matrices per layer: qkv, o, gate_up, down)
        void* w_qkv = nullptr;
        void* s_qkv = nullptr;
        void* w_o = nullptr;
        void* s_o = nullptr;
        void* w_gu = nullptr;
        void* s_gu = nullptr;
        void* w_down = nullptr;
        void* s_down = nullptr;
    };

    void run_linear_cuda(const __nv_bfloat16* x, void* w, void* s,
                         const __nv_bfloat16* bias, __nv_bfloat16* out,
                         u32 M, u32 N, u32 K);
    bool run_fp8_cublaslt_gemm(const __nv_bfloat16* x, const u8* w_fp8, const f32* row_scales,
                               const __nv_bfloat16* bias, __nv_bfloat16* out,
                               u32 M, u32 N, u32 K);
    void run_gate_up_silu_cuda(const __nv_bfloat16* x, void* w, void* s,
                               __nv_bfloat16* out, u32 M, u32 inter, u32 K);
    void run_forward_cuda(u32 M, const i32* d_pos_ptr, u32 host_start_pos, bool is_prefill);
    void ensure_decode_graph();
    void destroy_decode_graph();

    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    cublasLtHandle_t cublaslt_ = nullptr;
    void* d_cublaslt_ws_ = nullptr;
    size_t cublaslt_ws_bytes_ = 0;
    u8* d_x_fp8_ = nullptr;
    f32* d_fp8_scale_ = nullptr; // [0]=amax, [1]=x_scale, [2]=1.0f
    __nv_bfloat16* d_fp8_out_pad_ = nullptr;

    // Single contiguous VRAM slab for .pack weights (1 cudaMalloc instead of 280)
    u8* d_weight_slab_ = nullptr;
    __nv_bfloat16* d_embed_ = nullptr;
    __nv_bfloat16* d_final_norm_ = nullptr;
    __nv_bfloat16* d_lm_head_ = nullptr;
    u8*  d_lm_head_fp8_ = nullptr;
    f32* d_lm_head_scales_ = nullptr;
    f32* d_inv_freq_ = nullptr;
    std::vector<CudaLayer> d_layers_;

    // Paged KV cache in VRAM: BF16, Dynamic-Scaled INT8, or Zero-Calibration Walsh-Hadamard 4-Bit (WHT4)
    bool kv_fp8_ = false;
    KvQuantMode kv_mode_ = KvQuantMode::BF16;
    std::vector<void*> d_k_caches_;
    std::vector<void*> d_v_caches_;
    std::vector<f32*>  d_k_scales_;
    std::vector<f32*>  d_v_scales_;
    i32* d_block_table_ = nullptr;
    std::vector<i32> active_draft_blocks_;
    u32 active_draft_committed_ = 0;

    // Activation & scratch buffers (sized for max_prefill_chunk = 2048)
    u32 max_batch_tokens_ = 2048;
    __nv_bfloat16* d_x_ = nullptr;
    __nv_bfloat16* d_norm_ = nullptr;
    __nv_bfloat16* d_qkv_ = nullptr;
    __nv_bfloat16* d_q_ = nullptr;
    __nv_bfloat16* d_attn_out_ = nullptr;
    __nv_bfloat16* d_proj_out_ = nullptr;
    __nv_bfloat16* d_mlp_act_ = nullptr;
    __nv_bfloat16* d_mlp_gu_tmp_ = nullptr; // Prefill only
    __nv_bfloat16* d_logits_ = nullptr;
    __nv_bfloat16* d_dequant_tmp_ = nullptr; // Prefill dequant staging for W4A16/FP8

    // Split-K FlashDecoding scratch (adaptive 16 / 32 / 96 splits by context length)
    u32 flash_num_splits_ = 96;
    u32 active_graph_splits_ = 16;
    f32* d_split_o_ = nullptr;
    f32* d_split_ml_ = nullptr;

    // GPU-side decode state + pinned host token readback for full-step CUDA Graph
    i32* d_token_ids_ = nullptr;
    i32* d_out_tokens_ = nullptr;
    f32* d_out_conf_ = nullptr;
    i32* d_decode_pos_ = nullptr;
    i32* h_pinned_token_ = nullptr;
    f32* h_pinned_conf_ = nullptr;

    cudaGraph_t decode_graph_ = nullptr;
    cudaGraphExec_t decode_graph_exec_ = nullptr;
    bool graph_captured_ = false;
    void* d_mk_layers_ = nullptr; // cuda::MegakernelLayerDesc[num_layers]
    f32*  d_mk_argmax_val_ = nullptr;
    i32*  d_mk_argmax_idx_ = nullptr;
#endif

    ModelConfig cfg_;
    BpeTokenizer tokenizer_;
    EngineOptions opts_;
    QuantMode quant_ = QuantMode::BF16;
    bool use_cuda_ = false;
    double cold_start_ms_ = 0.0;
    u64 active_weight_bytes_ = 0;
    std::string gpu_name_;
    double gpu_peak_hbm_gbs_ = 0.0;

    RadixKvManager kv_mgr_;
    std::unique_ptr<CpuRefDecoder> cpu_ref_;
    std::unique_ptr<InferenceEngine> draft_engine_;
    std::unique_ptr<Inkling2Engine> inkling2_;
};

} // namespace flint
