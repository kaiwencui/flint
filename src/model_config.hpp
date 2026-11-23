#pragma once

#include "common.hpp"
#include "json.hpp"

#include <fstream>
#include <sstream>

namespace flint {

enum class ModelArch : u8 {
    Qwen2 = 0,        // Qwen2.5 (has QKV bias)
    Qwen3 = 1,        // Qwen3 (no QKV bias, has QK RMSNorm)
    Llama = 2,        // Llama-3.1 / 3.2 (no QKV bias, optional Llama-3 RoPE scaling)
    Inkling2Loop = 3  // thinkingmachines/Inkling-2-Mini (Looped MoE + NVFP4 + Rel-Bias Hybrid Attn + AttnRes + MTP)
};

struct Inkling2Config {
    u32 unpadded_vocab_size = 200058;
    u32 d_rel = 16;
    u32 rel_extent = 1024;
    u32 sliding_window_size = 1024;
    u32 sconv_kernel_size = 4;
    u32 n_layers_per_attn_res_block = 6;
    bool attn_res_normalize_values = true;
    bool use_value_norm_no_gamma = true;
    f32 qk_l2_clip_threshold = 22.627416997969522f;
    f32 rel_l2_clip_threshold = 8.0f;
    u32 log_scaling_n_floor = 8192;
    f32 log_scaling_alpha = 0.1f;
    f32 final_logit_softcapping = 30.0f;
    // MoE
    u32 n_routed_experts = 128;
    u32 num_experts_per_tok = 8;
    f32 mlp_gate_softcap = 4.0f;
    f32 mlp_up_softcap = 25.0f;
    f32 route_scale = 8.0f;
    // Looped Core
    u32 core_start = 6;
    u32 core_end = 18;
    u32 num_passes = 3;
    u32 gate_rank = 64;
    // Local SWA layer flags (length == num_layers)
    std::vector<u8> is_local_layer;
    // Built-in MTP head config
    bool has_mtp = false;
    u32 mtp_num_layers = 3;
    u32 mtp_intermediate_size = 9216;
    u32 mtp_block_size = 8;
    u32 mtp_markov_rank = 256;
    f64 mtp_rope_theta = 1000000.0;
    std::vector<u32> mtp_target_layer_ids; // [5, 23, 41, 48]

    bool is_local(u32 layer) const noexcept {
        return layer < is_local_layer.size() && is_local_layer[layer] != 0;
    }
    u32 total_visits(u32 num_layers) const noexcept {
        return core_start + (core_end - core_start) * num_passes + (num_layers - core_end);
    }
};

struct RopeScaling {
    bool enabled = false;
    std::string type; // "llama3", "yarn", "linear"
    f32 factor = 1.0f;
    f32 low_freq_factor = 1.0f;
    f32 high_freq_factor = 4.0f;
    u32 original_max_pos = 8192;
};

struct ModelConfig {
    ModelArch arch = ModelArch::Qwen2;
    std::string model_type;
    u32 vocab_size = 0;
    u32 hidden_size = 0;
    u32 intermediate_size = 0;
    u32 num_layers = 0;
    u32 num_q_heads = 0;
    u32 num_kv_heads = 0;
    u32 head_dim = 0;
    u32 max_position_embeddings = 32768;
    f32 rms_norm_eps = 1e-6f;
    f64 rope_theta = 1000000.0;
    bool tie_word_embeddings = false;
    bool has_qkv_bias = false;
    bool has_qk_norm = false;
    i32 bos_token_id = -1;
    std::vector<i32> eos_token_ids;
    RopeScaling rope_scaling;
    Inkling2Config ink;

    bool is_inkling2() const noexcept { return arch == ModelArch::Inkling2Loop; }

    u32 q_dim()  const noexcept { return num_q_heads * head_dim; }
    u32 kv_dim() const noexcept { return num_kv_heads * head_dim; }
    u32 gqa_group_size() const noexcept { return num_q_heads / std::max<u32>(1, num_kv_heads); }

    bool is_eos(i32 id) const noexcept {
        for (i32 e : eos_token_ids) {
            if (e == id) return true;
        }
        return false;
    }

    static bool load_from_dir(const std::filesystem::path& model_dir,
                              ModelConfig& out,
                              std::string& err) {
        auto cfg_path = model_dir / "config.json";
        std::ifstream in(cfg_path);
        if (!in) {
            err = "Cannot open " + cfg_path.string();
            return false;
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        JsonValue root;
        if (!parse_json(ss.str(), root, err)) return false;

        out = {};
        out.model_type = root.get_string("model_type");
        if (out.model_type == "inkling2_loop" || out.model_type == "inkling2_loop_text") {
            out.arch = ModelArch::Inkling2Loop;
            const JsonValue* tc = root.get("text_config");
            if (!tc || !tc->is_object()) tc = &root;
            out.vocab_size = static_cast<u32>(tc->get_int("vocab_size", 201024));
            out.hidden_size = static_cast<u32>(tc->get_int("hidden_size", 2304));
            out.intermediate_size = static_cast<u32>(tc->get_int("intermediate_size", 1152));
            out.num_layers = static_cast<u32>(tc->get_int("num_hidden_layers", 24));
            out.num_q_heads = static_cast<u32>(tc->get_int("num_attention_heads", 24));
            out.num_kv_heads = static_cast<u32>(tc->get_int("num_key_value_heads", 8));
            out.head_dim = static_cast<u32>(tc->get_int("head_dim", 128));
            out.max_position_embeddings = static_cast<u32>(tc->get_int("model_max_length", 1048576));
            out.rms_norm_eps = static_cast<f32>(tc->get_double("rms_norm_eps", 1e-6));
            out.tie_word_embeddings = false;
            out.has_qkv_bias = false;
            out.has_qk_norm = false;

            auto& ik = out.ink;
            ik.unpadded_vocab_size = static_cast<u32>(tc->get_int("unpadded_vocab_size", out.vocab_size));
            ik.d_rel = static_cast<u32>(tc->get_int("d_rel", 16));
            ik.rel_extent = static_cast<u32>(tc->get_int("rel_extent", 1024));
            ik.sliding_window_size = static_cast<u32>(tc->get_int("sliding_window_size", 1024));
            ik.sconv_kernel_size = static_cast<u32>(tc->get_int("sconv_kernel_size", 4));
            ik.n_layers_per_attn_res_block = static_cast<u32>(tc->get_int("n_layers_per_attn_res_block", 6));
            ik.attn_res_normalize_values = tc->get_bool("attn_res_normalize_values", true);
            ik.use_value_norm_no_gamma = tc->get_bool("use_value_norm_no_gamma", true);
            ik.qk_l2_clip_threshold = static_cast<f32>(tc->get_double("qk_l2_clip_threshold", 22.627416997969522));
            ik.rel_l2_clip_threshold = static_cast<f32>(tc->get_double("rel_l2_clip_threshold", 8.0));
            ik.log_scaling_n_floor = static_cast<u32>(tc->get_int("log_scaling_n_floor", 8192));
            ik.log_scaling_alpha = static_cast<f32>(tc->get_double("log_scaling_alpha", 0.1));
            ik.final_logit_softcapping = static_cast<f32>(tc->get_double("final_logit_softcapping", 30.0));
            ik.n_routed_experts = static_cast<u32>(tc->get_int("n_routed_experts", 128));
            ik.num_experts_per_tok = static_cast<u32>(tc->get_int("num_experts_per_tok", 8));
            ik.mlp_gate_softcap = static_cast<f32>(tc->get_double("mlp_gate_softcap", 4.0));
            ik.mlp_up_softcap = static_cast<f32>(tc->get_double("mlp_up_softcap", 25.0));
            ik.route_scale = static_cast<f32>(tc->get_double("route_scale", 8.0));
            ik.is_local_layer.assign(out.num_layers, 0);
            if (const auto* ll = tc->get("local_layer_ids"); ll && ll->is_array()) {
                for (const auto& x : ll->as_array()) {
                    u32 lid = static_cast<u32>(x.as_int());
                    if (lid < out.num_layers) ik.is_local_layer[lid] = 1;
                }
            }
            if (const auto* lc = tc->get("loop_config"); lc && lc->is_object()) {
                ik.core_start = static_cast<u32>(lc->get_int("core_start", 6));
                ik.core_end   = static_cast<u32>(lc->get_int("core_end", 18));
                ik.num_passes = static_cast<u32>(lc->get_int("num_passes", 3));
                ik.gate_rank  = static_cast<u32>(lc->get_int("gate_rank", 64));
            }
            if (const auto* mc = root.get("mtp_config"); mc && mc->is_object()) {
                ik.has_mtp = true;
                ik.mtp_num_layers = static_cast<u32>(mc->get_int("num_hidden_layers", 3));
                ik.mtp_intermediate_size = static_cast<u32>(mc->get_int("intermediate_size", 9216));
                ik.mtp_block_size = static_cast<u32>(mc->get_int("block_size", 8));
                ik.mtp_markov_rank = static_cast<u32>(mc->get_int("markov_rank", 256));
                ik.mtp_rope_theta = mc->get_double("rope_theta", 1000000.0);
                if (const auto* tl = mc->get("target_layer_ids"); tl && tl->is_array()) {
                    for (const auto& x : tl->as_array()) ik.mtp_target_layer_ids.push_back(static_cast<u32>(x.as_int()));
                }
            }
        } else if (out.model_type == "qwen3") {
            out.arch = ModelArch::Qwen3;
            out.has_qkv_bias = root.get_bool("attention_bias", false);
            out.has_qk_norm = true;
        } else if (out.model_type == "llama") {
            out.arch = ModelArch::Llama;
            out.has_qkv_bias = root.get_bool("attention_bias", false);
            out.has_qk_norm = false;
        } else {
            // Qwen2 / Qwen2.5
            out.arch = ModelArch::Qwen2;
            out.has_qkv_bias = true;
            out.has_qk_norm = false;
        }

        if (!out.is_inkling2()) {
            out.vocab_size = static_cast<u32>(root.get_int("vocab_size"));
            out.hidden_size = static_cast<u32>(root.get_int("hidden_size"));
            out.intermediate_size = static_cast<u32>(root.get_int("intermediate_size"));
            out.num_layers = static_cast<u32>(root.get_int("num_hidden_layers"));
            out.num_q_heads = static_cast<u32>(root.get_int("num_attention_heads"));
            out.num_kv_heads = static_cast<u32>(root.get_int("num_key_value_heads", out.num_q_heads));
            out.head_dim = static_cast<u32>(root.get_int("head_dim",
                out.num_q_heads > 0 ? (out.hidden_size / out.num_q_heads) : 0));
            out.max_position_embeddings = static_cast<u32>(root.get_int("max_position_embeddings", 32768));
            out.rms_norm_eps = static_cast<f32>(root.get_double("rms_norm_eps", 1e-6));
            out.rope_theta = root.get_double("rope_theta", 1000000.0);
            out.tie_word_embeddings = root.get_bool("tie_word_embeddings", false);
        }
        out.bos_token_id = static_cast<i32>(root.get_int("bos_token_id", -1));

        if (const auto* eos = root.get("eos_token_id")) {
            if (eos->is_int()) {
                out.eos_token_ids.push_back(static_cast<i32>(eos->as_int()));
            } else if (eos->is_array()) {
                for (const auto& x : eos->as_array()) {
                    if (x.is_int()) out.eos_token_ids.push_back(static_cast<i32>(x.as_int()));
                }
            }
        }

        // Also check generation_config.json for additional eos_token_id entries (e.g. <|im_end|>)
        auto gen_path = model_dir / "generation_config.json";
        if (std::ifstream gin(gen_path); gin) {
            std::ostringstream gss;
            gss << gin.rdbuf();
            JsonValue groot;
            std::string gerr;
            if (parse_json(gss.str(), groot, gerr)) {
                if (const auto* eos = groot.get("eos_token_id")) {
                    auto add_eos = [&](i32 id) {
                        if (std::find(out.eos_token_ids.begin(), out.eos_token_ids.end(), id) == out.eos_token_ids.end()) {
                            out.eos_token_ids.push_back(id);
                        }
                    };
                    if (eos->is_int()) add_eos(static_cast<i32>(eos->as_int()));
                    else if (eos->is_array()) {
                        for (const auto& x : eos->as_array()) {
                            if (x.is_int()) add_eos(static_cast<i32>(x.as_int()));
                        }
                    }
                }
            }
        }

        if (const auto* rs = root.get("rope_scaling"); rs && rs->is_object()) {
            out.rope_scaling.enabled = true;
            out.rope_scaling.type = rs->get_string("rope_type", rs->get_string("type"));
            out.rope_scaling.factor = static_cast<f32>(rs->get_double("factor", 1.0));
            out.rope_scaling.low_freq_factor = static_cast<f32>(rs->get_double("low_freq_factor", 1.0));
            out.rope_scaling.high_freq_factor = static_cast<f32>(rs->get_double("high_freq_factor", 4.0));
            out.rope_scaling.original_max_pos = static_cast<u32>(
                rs->get_int("original_max_position_embeddings", 8192));
        }

        if (out.vocab_size == 0 || out.hidden_size == 0 || out.num_layers == 0 || out.num_q_heads == 0) {
            err = "Invalid model config in " + cfg_path.string();
            return false;
        }
        return true;
    }
};

// Compute exact RoPE inverse frequencies (supports standard + Llama-3 smooth scaling)
inline std::vector<f32> compute_rope_inv_freq(const ModelConfig& cfg) {
    const u32 half = cfg.head_dim / 2;
    std::vector<f32> inv_freq(half);
    for (u32 i = 0; i < half; ++i) {
        f64 freq = 1.0 / std::pow(cfg.rope_theta, static_cast<f64>(2 * i) / static_cast<f64>(cfg.head_dim));
        if (cfg.rope_scaling.enabled && cfg.rope_scaling.type == "llama3") {
            f64 wavelen = 2.0 * M_PI / freq;
            f64 low_wl  = static_cast<f64>(cfg.rope_scaling.original_max_pos) / cfg.rope_scaling.low_freq_factor;
            f64 high_wl = static_cast<f64>(cfg.rope_scaling.original_max_pos) / cfg.rope_scaling.high_freq_factor;
            if (wavelen > low_wl) {
                freq /= cfg.rope_scaling.factor;
            } else if (wavelen >= high_wl) {
                f64 smooth = (static_cast<f64>(cfg.rope_scaling.original_max_pos) / wavelen -
                              cfg.rope_scaling.low_freq_factor) /
                             (cfg.rope_scaling.high_freq_factor - cfg.rope_scaling.low_freq_factor);
                freq = (1.0 - smooth) * freq / cfg.rope_scaling.factor + smooth * freq;
            }
        }
        inv_freq[i] = static_cast<f32>(freq);
    }
    return inv_freq;
}

} // namespace flint
