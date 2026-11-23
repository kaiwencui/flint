#include "cpu_ref.hpp"

#include <atomic>
#include <cmath>
#include <thread>

namespace flint {

namespace {

inline void rmsnorm_f32(f32* out, const f32* x, const f32* w, u32 n, f32 eps) noexcept {
    f32 ss = 0.0f;
    for (u32 i = 0; i < n; ++i) ss += x[i] * x[i];
    f32 inv = 1.0f / std::sqrt(ss / static_cast<f32>(n) + eps);
    for (u32 i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
}

// Parallel BF16-weight x FP32-vector GEMV: y[rows] = W[rows, cols] @ x[cols] (+ optional bias)
void matvec_bf16(f32* y, const u16* w, const f32* x, const f32* bias, u32 rows, u32 cols) {
    const u32 nthreads = std::min<u32>(16, std::max<u32>(1, std::thread::hardware_concurrency()));
    if (rows < 256 || nthreads == 1) {
        for (u32 r = 0; r < rows; ++r) {
            const u16* row = w + static_cast<size_t>(r) * cols;
            f32 acc = bias ? bias[r] : 0.0f;
            for (u32 c = 0; c < cols; ++c) {
                acc += bf16_to_f32(row[c]) * x[c];
            }
            y[r] = acc;
        }
        return;
    }

    std::atomic<u32> next_row{0};
    constexpr u32 kBlock = 64;
    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    for (u32 t = 0; t < nthreads; ++t) {
        workers.emplace_back([&]() {
            while (true) {
                u32 r0 = next_row.fetch_add(kBlock, std::memory_order_relaxed);
                if (r0 >= rows) break;
                u32 r1 = std::min(r0 + kBlock, rows);
                for (u32 r = r0; r < r1; ++r) {
                    const u16* row = w + static_cast<size_t>(r) * cols;
                    f32 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                    u32 c = 0;
                    for (; c + 4 <= cols; c += 4) {
                        acc0 += bf16_to_f32(row[c + 0]) * x[c + 0];
                        acc1 += bf16_to_f32(row[c + 1]) * x[c + 1];
                        acc2 += bf16_to_f32(row[c + 2]) * x[c + 2];
                        acc3 += bf16_to_f32(row[c + 3]) * x[c + 3];
                    }
                    f32 acc = (bias ? bias[r] : 0.0f) + (acc0 + acc1) + (acc2 + acc3);
                    for (; c < cols; ++c) acc += bf16_to_f32(row[c]) * x[c];
                    y[r] = acc;
                }
            }
        });
    }
    for (auto& th : workers) th.join();
}

// NeoX / HF half-split RoPE: (x[i], x[i + half]) rotated by cos/sin(pos * inv_freq[i])
inline void apply_rope_neox(f32* vec, u32 num_heads, u32 head_dim, u32 pos, const f32* inv_freq) noexcept {
    const u32 half = head_dim / 2;
    for (u32 h = 0; h < num_heads; ++h) {
        f32* hv = vec + static_cast<size_t>(h) * head_dim;
        for (u32 i = 0; i < half; ++i) {
            f32 theta = static_cast<f32>(pos) * inv_freq[i];
            f32 c = std::cos(theta);
            f32 s = std::sin(theta);
            f32 x0 = hv[i];
            f32 x1 = hv[i + half];
            hv[i]        = x0 * c - x1 * s;
            hv[i + half] = x1 * c + x0 * s;
        }
    }
}

inline f32 silu(f32 x) noexcept {
    return x / (1.0f + std::exp(-x));
}

} // namespace

bool CpuRefDecoder::load(const std::filesystem::path& model_dir, u32 max_seq_len, std::string& err) {
    if (!ModelConfig::load_from_dir(model_dir, cfg_, err)) return false;
    SafetensorsLoader st;
    if (!st.open_dir(model_dir, err)) return false;

    max_seq_len_ = std::min(max_seq_len, cfg_.max_position_embeddings);
    rope_inv_freq_ = compute_rope_inv_freq(cfg_);

    auto load_bf16 = [&](const std::string& name, std::vector<u16>& dst) -> bool {
        const auto* tv = st.get(name);
        if (!tv) { err = "Missing tensor: " + name; return false; }
        return SafetensorsLoader::read_as_bf16(*tv, dst);
    };
    auto load_f32 = [&](const std::string& name, std::vector<f32>& dst) -> bool {
        const auto* tv = st.get(name);
        if (!tv) { err = "Missing tensor: " + name; return false; }
        return SafetensorsLoader::read_as_f32(*tv, dst);
    };

    if (!load_bf16("model.embed_tokens.weight", embed_tokens_)) return false;
    if (!load_f32("model.norm.weight", final_norm_)) return false;
    if (!cfg_.tie_word_embeddings) {
        if (st.get("lm_head.weight")) {
            if (!load_bf16("lm_head.weight", lm_head_)) return false;
        } else {
            cfg_.tie_word_embeddings = true;
        }
    }

    layers_.resize(cfg_.num_layers);
    for (u32 l = 0; l < cfg_.num_layers; ++l) {
        std::string pre = "model.layers." + std::to_string(l) + ".";
        auto& lw = layers_[l];
        if (!load_f32(pre + "input_layernorm.weight", lw.input_norm)) return false;
        if (!load_f32(pre + "post_attention_layernorm.weight", lw.post_attn_norm)) return false;
        if (cfg_.has_qk_norm) {
            if (!load_f32(pre + "self_attn.q_norm.weight", lw.q_norm)) return false;
            if (!load_f32(pre + "self_attn.k_norm.weight", lw.k_norm)) return false;
        }
        if (!load_bf16(pre + "self_attn.q_proj.weight", lw.w_q)) return false;
        if (!load_bf16(pre + "self_attn.k_proj.weight", lw.w_k)) return false;
        if (!load_bf16(pre + "self_attn.v_proj.weight", lw.w_v)) return false;
        if (!load_bf16(pre + "self_attn.o_proj.weight", lw.w_o)) return false;
        if (cfg_.has_qkv_bias && st.get(pre + "self_attn.q_proj.bias")) {
            if (!load_f32(pre + "self_attn.q_proj.bias", lw.b_q)) return false;
            if (!load_f32(pre + "self_attn.k_proj.bias", lw.b_k)) return false;
            if (!load_f32(pre + "self_attn.v_proj.bias", lw.b_v)) return false;
        }
        if (!load_bf16(pre + "mlp.gate_proj.weight", lw.w_gate)) return false;
        if (!load_bf16(pre + "mlp.up_proj.weight", lw.w_up)) return false;
        if (!load_bf16(pre + "mlp.down_proj.weight", lw.w_down)) return false;
    }

    k_cache_.assign(cfg_.num_layers, std::vector<f32>(static_cast<size_t>(max_seq_len_) * cfg_.kv_dim(), 0.0f));
    v_cache_.assign(cfg_.num_layers, std::vector<f32>(static_cast<size_t>(max_seq_len_) * cfg_.kv_dim(), 0.0f));

    x_.resize(cfg_.hidden_size);
    xb_.resize(cfg_.hidden_size);
    xb2_.resize(cfg_.hidden_size);
    q_.resize(cfg_.q_dim());
    k_.resize(cfg_.kv_dim());
    v_.resize(cfg_.kv_dim());
    attn_out_.resize(cfg_.q_dim());
    gate_.resize(cfg_.intermediate_size);
    up_.resize(cfg_.intermediate_size);
    scores_.resize(max_seq_len_);
    cur_pos_ = 0;
    return true;
}

void CpuRefDecoder::forward_token(i32 token_id, u32 pos, std::vector<f32>& logits_out) {
    const u32 H = cfg_.hidden_size;
    const u32 D = cfg_.head_dim;
    const u32 nq = cfg_.num_q_heads;
    const u32 nkv = cfg_.num_kv_heads;
    const u32 gqa = cfg_.gqa_group_size();
    const f32 attn_scale = 1.0f / std::sqrt(static_cast<f32>(D));

    // Embedding lookup
    const u16* emb = embed_tokens_.data() + static_cast<size_t>(token_id) * H;
    for (u32 i = 0; i < H; ++i) x_[i] = bf16_to_f32(emb[i]);

    for (u32 l = 0; l < cfg_.num_layers; ++l) {
        const auto& lw = layers_[l];

        // 1. Pre-attention RMSNorm
        rmsnorm_f32(xb_.data(), x_.data(), lw.input_norm.data(), H, cfg_.rms_norm_eps);

        // 2. Q, K, V projections
        matvec_bf16(q_.data(), lw.w_q.data(), xb_.data(), lw.b_q.empty() ? nullptr : lw.b_q.data(), cfg_.q_dim(), H);
        matvec_bf16(k_.data(), lw.w_k.data(), xb_.data(), lw.b_k.empty() ? nullptr : lw.b_k.data(), cfg_.kv_dim(), H);
        matvec_bf16(v_.data(), lw.w_v.data(), xb_.data(), lw.b_v.empty() ? nullptr : lw.b_v.data(), cfg_.kv_dim(), H);

        // Optional Qwen3 per-head QK RMSNorm (before RoPE)
        if (cfg_.has_qk_norm) {
            for (u32 h = 0; h < nq; ++h) {
                rmsnorm_f32(q_.data() + h * D, q_.data() + h * D, lw.q_norm.data(), D, cfg_.rms_norm_eps);
            }
            for (u32 h = 0; h < nkv; ++h) {
                rmsnorm_f32(k_.data() + h * D, k_.data() + h * D, lw.k_norm.data(), D, cfg_.rms_norm_eps);
            }
        }

        // 3. RoPE
        apply_rope_neox(q_.data(), nq, D, pos, rope_inv_freq_.data());
        apply_rope_neox(k_.data(), nkv, D, pos, rope_inv_freq_.data());

        // 4. Write K, V into cache at `pos`
        f32* kc = k_cache_[l].data();
        f32* vc = v_cache_[l].data();
        std::memcpy(kc + static_cast<size_t>(pos) * cfg_.kv_dim(), k_.data(), cfg_.kv_dim() * sizeof(f32));
        std::memcpy(vc + static_cast<size_t>(pos) * cfg_.kv_dim(), v_.data(), cfg_.kv_dim() * sizeof(f32));

        // 5. GQA causal attention over [0..pos]
        const u32 seq_len = pos + 1;
        for (u32 h = 0; h < nq; ++h) {
            const u32 kvh = h / gqa;
            const f32* qh = q_.data() + h * D;
            f32 max_s = -1e30f;
            for (u32 t = 0; t < seq_len; ++t) {
                const f32* kh = kc + static_cast<size_t>(t) * cfg_.kv_dim() + kvh * D;
                f32 dot = 0.0f;
                for (u32 d = 0; d < D; ++d) dot += qh[d] * kh[d];
                f32 s = dot * attn_scale;
                scores_[t] = s;
                if (s > max_s) max_s = s;
            }
            f32 sum_exp = 0.0f;
            for (u32 t = 0; t < seq_len; ++t) {
                f32 e = std::exp(scores_[t] - max_s);
                scores_[t] = e;
                sum_exp += e;
            }
            f32 inv_sum = 1.0f / sum_exp;
            f32* oh = attn_out_.data() + h * D;
            std::memset(oh, 0, D * sizeof(f32));
            for (u32 t = 0; t < seq_len; ++t) {
                f32 w = scores_[t] * inv_sum;
                const f32* vh = vc + static_cast<size_t>(t) * cfg_.kv_dim() + kvh * D;
                for (u32 d = 0; d < D; ++d) oh[d] += w * vh[d];
            }
        }

        // 6. Output projection + residual add
        matvec_bf16(xb2_.data(), lw.w_o.data(), attn_out_.data(), nullptr, H, cfg_.q_dim());
        for (u32 i = 0; i < H; ++i) x_[i] += xb2_[i];

        // 7. Post-attention RMSNorm + SwiGLU MLP + residual add
        rmsnorm_f32(xb_.data(), x_.data(), lw.post_attn_norm.data(), H, cfg_.rms_norm_eps);
        matvec_bf16(gate_.data(), lw.w_gate.data(), xb_.data(), nullptr, cfg_.intermediate_size, H);
        matvec_bf16(up_.data(),   lw.w_up.data(),   xb_.data(), nullptr, cfg_.intermediate_size, H);
        for (u32 i = 0; i < cfg_.intermediate_size; ++i) {
            gate_[i] = silu(gate_[i]) * up_[i];
        }
        matvec_bf16(xb2_.data(), lw.w_down.data(), gate_.data(), nullptr, H, cfg_.intermediate_size);
        for (u32 i = 0; i < H; ++i) x_[i] += xb2_[i];
    }

    // Final RMSNorm + LM Head
    rmsnorm_f32(xb_.data(), x_.data(), final_norm_.data(), H, cfg_.rms_norm_eps);
    logits_out.resize(cfg_.vocab_size);
    const u16* head_w = cfg_.tie_word_embeddings ? embed_tokens_.data() : lm_head_.data();
    matvec_bf16(logits_out.data(), head_w, xb_.data(), nullptr, cfg_.vocab_size, H);
    cur_pos_ = pos + 1;
}

void CpuRefDecoder::forward_prefill(std::span<const i32> tokens, std::vector<f32>& logits_out) {
    reset_kv();
    for (size_t i = 0; i < tokens.size(); ++i) {
        // Only compute full LM head logits on the final prompt token
        if (i + 1 < tokens.size()) {
            // Still need to populate KV cache for token i; forward_token computes logits too,
            // which is fast enough for 0.5B CPU verification, or optimize by skipping LM head.
            std::vector<f32> dummy;
            // Fast path: duplicate loop without LM head? Keep clean: 0.5B LM head takes ~15ms.
            forward_token(tokens[i], static_cast<u32>(i), dummy);
        } else {
            forward_token(tokens[i], static_cast<u32>(i), logits_out);
        }
    }
}

i32 CpuRefDecoder::argmax(std::span<const f32> logits) noexcept {
    i32 best = 0;
    f32 best_v = -1e30f;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = static_cast<i32>(i);
        }
    }
    return best;
}

} // namespace flint
