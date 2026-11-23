#pragma once

#include "common.hpp"
#include "kv_cache.hpp"
#include "model_config.hpp"
#include "speculative.hpp"
#include "tokenizer.hpp"
#include "weights.hpp"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(FLINT_CUDA) && FLINT_CUDA == 1
  #include <cuda_runtime.h>
  #include <cuda_bf16.h>
  #include <cublas_v2.h>
  #include <cublasLt.h>
#endif

namespace flint {

struct EngineOptions;
struct GenerateOptions;
struct GenerateResult;

// Dedicated zero-dependency CUDA engine for thinkingmachines/Inkling-2-Mini:
// - 24 physical layers, looped core [6..18) x 3 passes = 48 logical layer visits
// - Block AttnRes (every 6 layers) + Rank-64 Cross-Pass Message Controllers
// - Hybrid Attention: 5:1 Local SWA (1024-tok ring buffer) + Global Full (paged KV)
// with Relative Position Logits (d_rel=16, extent=1024), ShortConv (W=4),
// L2-clipped Q/K/R, Value RMSNorm (no gamma), Log-Scaling tau, Output Gate + Sink
// - Sparse NVFP4 MoE (128 experts, Top-8 routed, block-16 FP8 scales, SiTU softcaps)
// - Full-Step CUDA Graph Capture across all 48 visits
class Inkling2Engine {
public:
    Inkling2Engine() = default;
    ~Inkling2Engine();

    bool load(const ModelConfig& cfg,
              const BpeTokenizer& tok,
              const EngineOptions& opts,
              std::string& err);

    GenerateResult generate_ids(std::span<const i32> prompt_ids,
                                const GenerateOptions& gopts,
                                const std::function<void(std::string_view)>& on_token = nullptr);

    bool eval_logits(std::span<const i32> token_ids, std::vector<f32>& logits_out);
    void clear_kv_cache();

    u64 active_weight_bytes() const noexcept { return active_weight_bytes_; }
    const std::string& gpu_name() const noexcept { return gpu_name_; }
    double gpu_peak_hbm_gbs() const noexcept { return gpu_peak_hbm_gbs_; }

private:
    u64 active_weight_bytes_ = 0;
    std::string gpu_name_;
    double gpu_peak_hbm_gbs_ = 0.0;

#if defined(FLINT_CUDA) && FLINT_CUDA == 1
    ModelConfig cfg_;
    const BpeTokenizer* tok_ = nullptr;
    bool kv_int8_ = false;
    KvQuantMode kv_mode_ = KvQuantMode::BF16;
    u32 max_seq_ = 8192;
    bool use_cuda_graph_ = true;
    SpecMode spec_mode_ = SpecMode::None;
    struct ControllerWeights {
        __nv_bfloat16* down = nullptr;   // [64, 4608]
        __nv_bfloat16* gain = nullptr;   // [2304, 64]
        __nv_bfloat16* forget = nullptr; // [2304, 64]
    };

    struct PhysicalLayer {
        bool is_local = true;
        // Norms: local has attn_norm [2304]; global has stacked [3, 2304] (q, k, v)
        __nv_bfloat16* attn_norm = nullptr;
        __nv_bfloat16* mlp_norm = nullptr;
        // Attention projections (BF16 or per-row FP8 when packed with --quant fp8)
        __nv_bfloat16* wq = nullptr;     // [3072, 2304]
        __nv_bfloat16* wr = nullptr;     // [384, 2304]
        __nv_bfloat16* w_gate = nullptr; // [3072, 2304]
        __nv_bfloat16* w_sink = nullptr; // [24, 2304] (always BF16)
        __nv_bfloat16* wk = nullptr;     // [1024, 2304]
        __nv_bfloat16* wv = nullptr;     // [1024, 2304]
        __nv_bfloat16* wo = nullptr;     // [2304, 3072]
        u8*  wq_fp8 = nullptr; f32* sq_fp8 = nullptr;
        u8*  wr_fp8 = nullptr; f32* sr_fp8 = nullptr;
        u8*  wg_fp8 = nullptr; f32* sg_fp8 = nullptr;
        u8*  wk_fp8 = nullptr; f32* sk_fp8 = nullptr;
        u8*  wv_fp8 = nullptr; f32* sv_fp8 = nullptr;
        u8*  wo_fp8 = nullptr; f32* so_fp8 = nullptr;
        // Scales & relative projection (transposed [1024, 16])
        __nv_bfloat16* q_scale = nullptr;   // [24, 128]
        __nv_bfloat16* r_scale = nullptr;   // [24, 16]
        __nv_bfloat16* rel_proj_T = nullptr;// [1024, 16]
        // Short-Conv weights
        __nv_bfloat16* k_sconv = nullptr;     // local: [1024, 4]; global: [4, 1024]
        __nv_bfloat16* v_sconv = nullptr;
        __nv_bfloat16* k_sconv_proj = nullptr;// global only: [32, 2304]
        __nv_bfloat16* v_sconv_proj = nullptr;// global only: [32, 2304]
        __nv_bfloat16* attn_sconv = nullptr;  // [2304, 4]
        __nv_bfloat16* mlp_sconv = nullptr;   // [2304, 4]
        // MoE Router & NVFP4 / MXFP8 Experts
        bool is_mxfp8 = false;
        __nv_bfloat16* moe_gate_w = nullptr;  // [128, 2304] (BF16 mode)
        u8*  moe_gate_fp8 = nullptr;          // [128, 2304] (FP8 mode)
        f32* moe_gate_scale = nullptr;        // [128]
        f32* moe_gate_bias = nullptr;         // [128]
        u8*  w13_weight = nullptr;            // [128, 2304, 1152] U8
        u8*  w13_scale = nullptr;             // [128, 2304, 144] FP8
        f32* w13_amax = nullptr;              // [128]
        f32* w13_outer_denom = nullptr;       // [1]
        u8*  w2_weight = nullptr;             // [128, 2304, 576] U8
        u8*  w2_scale = nullptr;              // [128, 2304, 72] FP8
        f32* w2_amax = nullptr;               // [128]
        f32* w2_outer_denom = nullptr;        // [1]
        // Looped Core Controllers (layers 6..17 only: 0=attn, 1=k, 2=v, 3=mlp)
        bool has_controllers = false;
        ControllerWeights ctrl[4];
    };

    struct VisitState {
        u32 phys_layer = 0;
        u32 pass_idx = 0;
        bool is_local = true;
        // KV cache: ring buffer [8, 1024, 128] for local; paged [num_blocks, 8, 16, 128] for global
        void* d_k_cache = nullptr;
        void* d_v_cache = nullptr;
        f32*  d_k_scale = nullptr;
        f32*  d_v_scale = nullptr;
        // 4-slot causal short-conv history rings (K, V, attn_out, mlp_out)
        __nv_bfloat16* d_k_hist = nullptr;    // [4, 1024]
        __nv_bfloat16* d_v_hist = nullptr;    // [4, 1024]
        __nv_bfloat16* d_attn_hist = nullptr; // [4, 2304]
        __nv_bfloat16* d_mlp_hist = nullptr;  // [4, 2304]
    };

    // Built-in MTP (DSpark 3-layer + Rank-256 Markov Head from mtp.safetensors)
    struct MtpLayer {
        bool is_local = false;
        __nv_bfloat16* dflare_norm = nullptr;
        __nv_bfloat16* input_norm = nullptr;
        __nv_bfloat16* post_attn_norm = nullptr;
        __nv_bfloat16* q_norm = nullptr;
        __nv_bfloat16* k_norm = nullptr;
        __nv_bfloat16* k_norm_ctx = nullptr;
        __nv_bfloat16* wq = nullptr;
        __nv_bfloat16* wk = nullptr;
        __nv_bfloat16* wv = nullptr;
        __nv_bfloat16* wk_ctx = nullptr;
        __nv_bfloat16* wv_ctx = nullptr;
        __nv_bfloat16* wo = nullptr;
        __nv_bfloat16* w_gate = nullptr;
        __nv_bfloat16* w_up = nullptr;
        __nv_bfloat16* w_down = nullptr;
        // FP8 MTP weights (loaded when packed with --quant fp8 -> 2x faster MTP draft layers)
        u8*  wq_fp8 = nullptr; f32* sq_fp8 = nullptr;
        u8*  wk_fp8 = nullptr; f32* sk_fp8 = nullptr;
        u8*  wv_fp8 = nullptr; f32* sv_fp8 = nullptr;
        u8*  wk_ctx_fp8 = nullptr; f32* sk_ctx_fp8 = nullptr;
        u8*  wv_ctx_fp8 = nullptr; f32* sv_ctx_fp8 = nullptr;
        u8*  wo_fp8 = nullptr; f32* so_fp8 = nullptr;
        u8*  w_gate_fp8 = nullptr; f32* s_gate_fp8 = nullptr;
        u8*  w_up_fp8 = nullptr; f32* s_up_fp8 = nullptr;
        u8*  w_down_fp8 = nullptr; f32* s_down_fp8 = nullptr;
        // Contiguous context KV cache [max_mtp_seq, 8, 128]
        __nv_bfloat16* d_k_ctx = nullptr;
        __nv_bfloat16* d_v_ctx = nullptr;
    };

    void free_resources();
    void run_bf16_linear(const __nv_bfloat16* x, const __nv_bfloat16* w,
                         __nv_bfloat16* out, u32 M, u32 N, u32 K, bool add_to_out = false);
    bool run_fp8_cublaslt_gemm(const __nv_bfloat16* x, const u8* w_fp8, const f32* row_scales,
                               __nv_bfloat16* out, u32 M, u32 N, u32 K);
    void run_fp8_linear(const __nv_bfloat16* x, const u8* w_fp8, const f32* s_fp8,
                        __nv_bfloat16* out, u32 M, u32 N, u32 K, bool add_to_out = false);
    void run_forward(u32 M, const i32* d_pos_ptr, u32 host_start_pos, bool is_prefill,
                     bool draft_1pass = false, bool capture_mtp = false);
    void mtp_store_context(u32 M, u32 start_pos);
    void mtp_draft_block(i32 anchor_tok, u32 cur_pos, u32 K, std::vector<i32>& out_drafts);
    void ensure_decode_graph();
    void destroy_decode_graph();
    void ensure_mtp_graphs();
    void destroy_mtp_graphs();
    void save_prefix_snapshot(std::span<const i32> prefix_ids, const std::vector<i32>& bt);
    bool try_restore_prefix_snapshot(std::span<const i32> prompt_ids, u32& out_start_pos, std::vector<i32>& out_bt);
    bool save_kv_file(const std::string& path, u32 seq_len, std::string& err);
    bool load_kv_file(const std::string& path, u32& out_seq_len, std::string& err);

    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    cublasLtHandle_t cublaslt_ = nullptr;
    u8* d_fp8_x_ = nullptr;
    f32* d_fp8_scale_ = nullptr;         // [3]: [0]=amax, [1]=x_scale, [2]=1.0f
    __nv_bfloat16* d_fp8_out_pad_ = nullptr;
    f32* d_moe_fp8_scales_ = nullptr;    // [2 * inter = 2304] per-row scales for NVFP4->FP8 MoE prefill
    void* d_cublaslt_ws_ = nullptr;
    size_t cublaslt_ws_bytes_ = 0;
    std::unordered_map<u64, cublasLtMatmulAlgo_t> lt_algo_cache_; // O(1) algo cache (kills 12,288 CPU heuristic calls/chunk)
    i32* h_moe_pinned_ids_ = nullptr;    // [M * topk] pinned D2H/H2D -> eliminates 12,288 pageable H2D syncs/chunk
    f32* h_moe_pinned_wts_ = nullptr;
    i32* h_moe_pack_idx_ = nullptr;      // [M * topk] pinned contiguous sorted indices
    f32* h_moe_pack_gam_ = nullptr;      // [M * topk] pinned contiguous sorted gammas
    i32* d_moe_pack_idx_ = nullptr;      // [M * topk] GPU contiguous sorted indices
    f32* d_moe_pack_gam_ = nullptr;      // [M * topk] GPU contiguous sorted gammas
    // Blackwell sm_120a Hardware FP4 Tensor-Core Grouped MoE scratch (zero host sync)
    u32* d_tile_meta_ = nullptr;         // [128 + M*topk/16]
    u32* d_tile_cnt_ = nullptr;
    u32* d_num_tiles_ = nullptr;
    u8*  d_x_fp4_ = nullptr;             // [M, H/2]
    u8*  d_x_fp4_sc_ = nullptr;          // [M, H/16]
    f32* d_x_fp4_gs_ = nullptr;          // [M]
    u8*  d_act_fp4_ = nullptr;           // [M*topk, inter/2]
    u8*  d_act_fp4_sc_ = nullptr;        // [M*topk, inter/16]
    f32* d_act_fp4_gs_ = nullptr;        // [M*topk]

    u8* d_weight_slab_ = nullptr;
    __nv_bfloat16* d_embed_ = nullptr;
    u8*  d_embed_fp8_ = nullptr;
    f32* d_embed_scale_ = nullptr;
    u32* d_embed_qw_ = nullptr;
    u32* d_embed_sc_ = nullptr;
    __nv_bfloat16* d_embed_norm_ = nullptr;
    __nv_bfloat16* d_final_norm_ = nullptr;
    __nv_bfloat16* d_unembed_ = nullptr;
    u8*  d_unembed_fp8_ = nullptr;
    f32* d_unembed_scale_ = nullptr;
    bool fp8_attn_ = false;
    bool fp8_embed_ = false;
    bool w4_embed_ = false;

    // Pre-scaled F32 AttnRes queries (multiplied by H**-0.5)
    std::vector<f32*> d_attn_queries_; // index i-1 for layer i=1..23 (1*H for local, 3*H for global)
    std::vector<f32*> d_mlp_queries_;  // index i for layer i=0..23 (1*H)
    f32* d_output_query_ = nullptr;    // [1, H]
    std::vector<f32*> d_msg_queries_;  // index c=0..11 for core layer 6+c ([4, H]: attn, k, v, mlp)

    std::vector<PhysicalLayer> layers_;
    std::vector<VisitState> visits_; // 48 visits

    // Global paged KV manager
    RadixKvManager kv_mgr_;
    i32* d_block_table_ = nullptr;
    u32 max_batch_tokens_ = 256;

    // Residual stack & looped bank & scratch buffers
    __nv_bfloat16* d_stack_[5] = {};       // [5][M, H]
    __nv_bfloat16* d_prelude_partial_ = nullptr;
    __nv_bfloat16* d_bank_[2] = {};        // [2][M, H]
    const __nv_bfloat16** d_ptr_table_ = nullptr; // device array of stack ptrs for attn_res
    f32* d_mix_f32_ = nullptr;             // [3, M, H]
    f32* d_prelude_dests_f32_ = nullptr;   // [3, M, H]
    f32* d_msg_f32_ = nullptr;             // [4, M, H]
    __nv_bfloat16* d_pair_bf16_ = nullptr; // [3, M, 2*H]
    __nv_bfloat16* d_ctrl_h_ = nullptr;    // [3, M, 64]
    __nv_bfloat16* d_ctrl_gf_ = nullptr;   // [3, M, 2*H] (gain, forget)
    __nv_bfloat16* d_attn_in_ = nullptr;   // [3, M, H]
    __nv_bfloat16* d_mlp_in_ = nullptr;    // [M, H]
    __nv_bfloat16* d_q_raw_ = nullptr;     // [M, 3072]
    __nv_bfloat16* d_q_clip_ = nullptr;    // [M, 3072]
    __nv_bfloat16* d_r_raw_ = nullptr;     // [M, 384]
    __nv_bfloat16* d_r_clip_ = nullptr;    // [M, 384]
    __nv_bfloat16* d_gate_ = nullptr;      // [M, 3072]
    __nv_bfloat16* d_sink_ = nullptr;      // [M, 24]
    __nv_bfloat16* d_k_raw_ = nullptr;     // [M, 1024]
    __nv_bfloat16* d_v_raw_ = nullptr;     // [M, 1024]
    __nv_bfloat16* d_k_dyn_ = nullptr;     // [M, 32]
    __nv_bfloat16* d_v_dyn_ = nullptr;     // [M, 32]
    __nv_bfloat16* d_attn_out_ = nullptr;  // [M, 3072]
    __nv_bfloat16* d_proj_out_ = nullptr;  // [M, 2304]
    __nv_bfloat16* d_attn_conv_ = nullptr; // [M, 2304]
    __nv_bfloat16* d_moe_out_ = nullptr;   // [M, 2304]
    __nv_bfloat16* d_moe_act_ = nullptr;   // [M, 8, 1152]
    __nv_bfloat16* d_router_logits_ = nullptr; // [M, 128] for 32-SM parallel router
    __nv_bfloat16* d_moe_dequant_ = nullptr;   // [2304, 2304] single-expert BF16 staging for batched prefill
    __nv_bfloat16* d_moe_gather_x_ = nullptr;  // [M, 2304]
    __nv_bfloat16* d_moe_w13_out_ = nullptr;   // [M, 2304]
    __nv_bfloat16* d_moe_act_b_ = nullptr;     // [M, 1152]
    __nv_bfloat16* d_moe_y_b_ = nullptr;       // [M, 2304]
    f32* d_moe_out_f32_ = nullptr;             // [M, 2304]
    i32* d_moe_tok_idx_ = nullptr;             // [M]
    f32* d_moe_gammas_ = nullptr;              // [M]
    i32* d_expert_ids_ = nullptr;          // [M, 8]
    f32* d_expert_weights_ = nullptr;      // [M, 8]
    u32* d_uniq_e_ = nullptr;              // [64] compact active expert IDs
    u64* d_uniq_mask_ = nullptr;           // [64] bitmask of routed (m, k) slots
    u32* d_uniq_count_ = nullptr;          // [1] count of unique active experts
    __nv_bfloat16* d_logits_ = nullptr;    // [16, vocab_size]
    f32* d_split_o_ = nullptr;             // [16, 24, 64, 128] Split-K relative attention scratch
    f32* d_split_ml_ = nullptr;            // [16, 24, 64, 2]
    u32 splitk_max_ = 64;

    i32* d_token_ids_ = nullptr;
    i32* d_out_tokens_ = nullptr;
    i32* d_decode_pos_ = nullptr;
    i32* h_pinned_token_ = nullptr;

    // Conv history backup for lossless speculative rollback
    std::vector<__nv_bfloat16*> d_hist_backup_;

    // Built-in MTP weights & scratch (allocated only when spec_mode_ == SpecMode::Mtp)
    bool mtp_loaded_ = false;
    bool mtp_fp8_ = false;
    u32 mtp_max_seq_ = 0;
    MtpLayer mtp_layers_[3];
    f32* d_mtp_alpha_ = nullptr;           // [4, 4] softmax(fuse_weight)
    f32* d_mtp_inv_freq_ = nullptr;        // [64]
    __nv_bfloat16* d_mtp_mask_embed_ = nullptr; // [2304]
    __nv_bfloat16* d_mtp_norm_ = nullptr;       // [2304]
    __nv_bfloat16* d_mtp_mh_norm_ = nullptr;    // [2304]
    __nv_bfloat16* d_mtp_markov_w1_ = nullptr;  // [201024, 256]
    __nv_bfloat16* d_mtp_rnn_w_ = nullptr;      // [768, 5120]
    __nv_bfloat16* d_mtp_rnn_b_ = nullptr;      // [768]
    __nv_bfloat16* d_mtp_hout_w_ = nullptr;     // [2304, 256]
    __nv_bfloat16* d_mtp_hout_b_ = nullptr;     // [2304]
    __nv_bfloat16* d_mtp_fold_w_ = nullptr;     // [201024, 256] = unembed @ hout_w (BF16 mode)
    u8*  d_mtp_fold_fp8_ = nullptr;             // [201024, 256] FP8 (-51 MiB + 2x faster Markov head)
    f32* d_mtp_fold_scale_ = nullptr;           // [201024]
    __nv_bfloat16* d_mtp_fold_b_ = nullptr;     // [201024] = unembed @ hout_b
    __nv_bfloat16* d_mtp_feats_[4] = {};        // [4][M, 2304] captured target features
    __nv_bfloat16* d_mtp_last_feats_[4] = {};   // [4][1, 2304] last committed token features
    const __nv_bfloat16** d_mtp_feat_ptrs_ = nullptr; // [8]: [0..3]=feats, [4..7]=last_feats
    __nv_bfloat16* d_mtp_x_ = nullptr;          // [8, 2304]
    __nv_bfloat16* d_mtp_norm_x_ = nullptr;     // [M, 2304]
    __nv_bfloat16* d_mtp_q_ = nullptr;          // [8, 3072]
    __nv_bfloat16* d_mtp_k_ = nullptr;          // [8, 1024]
    __nv_bfloat16* d_mtp_v_ = nullptr;          // [8, 1024]
    __nv_bfloat16* d_mtp_attn_ = nullptr;       // [8, 3072]
    __nv_bfloat16* d_mtp_mlp_g_ = nullptr;      // [8, 9216]
    __nv_bfloat16* d_mtp_mlp_u_ = nullptr;      // [8, 9216]
    __nv_bfloat16* d_mtp_f_rnn_ = nullptr;      // [1, 2304]
    __nv_bfloat16* d_mtp_state_ = nullptr;      // [256]
    __nv_bfloat16* d_mtp_latent_ = nullptr;     // [256]
    __nv_bfloat16* d_mtp_rnn_in_ = nullptr;     // [5120]
    __nv_bfloat16* d_mtp_raw3r_ = nullptr;      // [768]
    f32* d_mtp_rnn_ctx_f32_ = nullptr;          // [7, 768] precomputed Markov RNN context
    __nv_bfloat16* d_mtp_bias_ = nullptr;       // [201024]
    __nv_bfloat16* d_mtp_logits_tmp_ = nullptr; // [201024]

    cudaGraph_t decode_graph_ = nullptr;
    cudaGraphExec_t decode_graph_exec_ = nullptr;
    bool graph_captured_ = false;

    // Full CUDA Graphs for Built-In MTP (1 graph launch for draft + 1 graph launch for M=8 verify)
    cudaGraph_t mtp_draft_graph_ = nullptr;
    cudaGraphExec_t mtp_draft_graph_exec_ = nullptr;
    cudaGraph_t mtp_verify_graph_ = nullptr;
    cudaGraphExec_t mtp_verify_graph_exec_ = nullptr;
    bool mtp_graphs_captured_ = false;

    // Exact 1-Slot Inkling-2 Prefix Cache (snapshots 40 local SWA rings + 48x4 conv rings)
    struct PrefixSnapVisit {
        void* d_k_snap = nullptr;
        void* d_v_snap = nullptr;
        f32*  d_ks_snap = nullptr;
        f32*  d_vs_snap = nullptr;
        __nv_bfloat16* d_kh_snap = nullptr;
        __nv_bfloat16* d_vh_snap = nullptr;
        __nv_bfloat16* d_ah_snap = nullptr;
        __nv_bfloat16* d_mh_snap = nullptr;
    };
    std::vector<i32> prefix_tokens_;
    std::vector<i32> prefix_bt_;
    std::vector<PrefixSnapVisit> prefix_snap_;
    bool prefix_snap_alloc_ = false;
    u32 prefix_snap_len_ = 0;
#endif
};

} // namespace flint
