#pragma once

#include "common.hpp"
#include "model_config.hpp"
#include "weights.hpp"

#include <string>
#include <vector>

namespace flint {

// Exact FP32-accumulate / BF16-weight CPU reference decoder for Qwen2.5, Qwen3, and Llama-3.
// Used for Milestone 0 Gate 2 verification ($0 GPU) and unit-testing custom CUDA kernels.
class CpuRefDecoder {
public:
    bool load(const std::filesystem::path& model_dir, u32 max_seq_len, std::string& err);

    const ModelConfig& config() const noexcept { return cfg_; }
    void reset_kv() noexcept { cur_pos_ = 0; }

    // Run one token at position `pos` and write vocab_size logits into `logits_out`.
    void forward_token(i32 token_id, u32 pos, std::vector<f32>& logits_out);

    // Run a sequence of prompt tokens and return final token's logits.
    void forward_prefill(std::span<const i32> tokens, std::vector<f32>& logits_out);

    // Greedy argmax helper
    static i32 argmax(std::span<const f32> logits) noexcept;

private:
    struct LayerWeights {
        std::vector<f32> input_norm;
        std::vector<f32> post_attn_norm;
        std::vector<f32> q_norm; // Qwen3 optional
        std::vector<f32> k_norm;
        // Keep matrices in compact BF16 (u16) so 0.5B fits in ~1 GB RAM, accumulate in FP32
        std::vector<u16> w_q, w_k, w_v, w_o;
        std::vector<f32> b_q, b_k, b_v; // Qwen2.5 QKV bias
        std::vector<u16> w_gate, w_up, w_down;
    };

    ModelConfig cfg_;
    u32 max_seq_len_ = 2048;
    u32 cur_pos_ = 0;

    std::vector<u16> embed_tokens_; // [vocab, hidden] BF16
    std::vector<f32> final_norm_;   // [hidden]
    std::vector<u16> lm_head_;      // [vocab, hidden] BF16 (empty if tied)
    std::vector<LayerWeights> layers_;
    std::vector<f32> rope_inv_freq_;

    // KV Cache: [num_layers][max_seq_len * kv_dim] FP32
    std::vector<std::vector<f32>> k_cache_;
    std::vector<std::vector<f32>> v_cache_;

    // Scratch buffers
    std::vector<f32> x_, xb_, xb2_, q_, k_, v_, attn_out_, gate_, up_, scores_;
};

} // namespace flint
