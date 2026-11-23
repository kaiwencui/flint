#include "weights.hpp"
#include "json.hpp"
#include "tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>

namespace flint {

namespace {

DType parse_st_dtype(std::string_view s) noexcept {
    if (s == "BF16") return DType::BF16;
    if (s == "F16")  return DType::F16;
    if (s == "F32")  return DType::F32;
    if (s == "F8_E4M3") return DType::FP8_E4M3;
    if (s == "U8")   return DType::U8;
    return DType::BF16;
}

bool write_all(int fd, const void* buf, size_t len) {
    const auto* p = static_cast<const u8*>(buf);
    size_t off = 0;
    while (off < len) {
        ssize_t n = ::write(fd, p + off, len - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        off += static_cast<size_t>(n);
    }
    return true;
}

bool write_zeros(int fd, size_t n) {
    static const u8 z[128] = {};
    while (n > 0) {
        size_t chunk = std::min<size_t>(n, sizeof(z));
        if (!write_all(fd, z, chunk)) return false;
        n -= chunk;
    }
    return true;
}

} // namespace

bool SafetensorsLoader::add_shard(const std::filesystem::path& shard_path, std::string& err) {
    MmapFile mf;
    if (!mf.open(shard_path, true)) {
        err = "Failed to mmap safetensors shard: " + shard_path.string();
        return false;
    }
    if (mf.size() < 8) {
        err = "Truncated safetensors file: " + shard_path.string();
        return false;
    }
    u64 header_len = 0;
    std::memcpy(&header_len, mf.data(), 8);
    if (8 + header_len > mf.size()) {
        err = "Invalid safetensors header length in " + shard_path.string();
        return false;
    }

    std::string_view hdr_sv(reinterpret_cast<const char*>(mf.data() + 8), header_len);
    JsonValue hdr;
    if (!parse_json(hdr_sv, hdr, err)) return false;
    if (!hdr.is_object()) {
        err = "safetensors header is not a JSON object";
        return false;
    }

    const u8* data_base = mf.data() + 8 + header_len;
    const u64 data_size = mf.size() - (8 + header_len);

    for (const auto& [name, val] : hdr.as_object()) {
        if (name == "__metadata__" || !val.is_object()) continue;
        TensorView tv;
        tv.name = name;
        tv.dtype = parse_st_dtype(val.get_string("dtype"));
        if (const auto* sh = val.get("shape"); sh && sh->is_array()) {
            for (const auto& d : sh->as_array()) tv.shape.push_back(static_cast<u64>(d.as_int()));
        }
        const auto* offs = val.get("data_offsets");
        if (!offs || !offs->is_array() || offs->as_array().size() != 2) continue;
        u64 b0 = static_cast<u64>(offs->as_array()[0].as_int());
        u64 b1 = static_cast<u64>(offs->as_array()[1].as_int());
        if (b1 < b0 || b1 > data_size) {
            err = "Out-of-bounds tensor offset for " + name;
            return false;
        }
        tv.data = data_base + b0;
        tv.nbytes = b1 - b0;
        total_bytes_ += tv.nbytes;
        tensors_[name] = std::move(tv);
    }

    shards_.push_back(std::move(mf));
    return true;
}

bool SafetensorsLoader::open_dir(const std::filesystem::path& model_dir, std::string& err) {
    shards_.clear();
    tensors_.clear();
    total_bytes_ = 0;

    // Single file mode if path is directly a .safetensors file
    std::error_code ec;
    if (std::filesystem::is_regular_file(model_dir, ec)) {
        return add_shard(model_dir, err);
    }

    // Check model.safetensors.index.json first (multi-shard HF models like 7B/8B/14B)
    auto idx_json = model_dir / "model.safetensors.index.json";
    if (std::filesystem::exists(idx_json, ec)) {
        std::ifstream in(idx_json);
        std::ostringstream ss;
        ss << in.rdbuf();
        JsonValue root;
        if (!parse_json(ss.str(), root, err)) return false;
        std::set<std::string> shard_files;
        if (const auto* wm = root.get("weight_map"); wm && wm->is_object()) {
            for (const auto& [_, fval] : wm->as_object()) {
                if (fval.is_string()) shard_files.insert(fval.as_string());
            }
        }
        if (shard_files.empty()) {
            err = "No shards listed in " + idx_json.string();
            return false;
        }
        for (const auto& sf : shard_files) {
            if (!add_shard(model_dir / sf, err)) return false;
        }
        return true;
    }

    // Single model.safetensors
    auto single = model_dir / "model.safetensors";
    if (std::filesystem::exists(single, ec)) {
        return add_shard(single, err);
    }

    // Fallback: scan for any *.safetensors
    std::vector<std::filesystem::path> found;
    for (const auto& ent : std::filesystem::directory_iterator(model_dir, ec)) {
        if (ent.path().extension() == ".safetensors") found.push_back(ent.path());
    }
    std::sort(found.begin(), found.end());
    if (found.empty()) {
        err = "No .safetensors files found in " + model_dir.string();
        return false;
    }
    for (const auto& p : found) {
        if (!add_shard(p, err)) return false;
    }
    return true;
}

const TensorView* SafetensorsLoader::get(const std::string& name) const noexcept {
    auto it = tensors_.find(name);
    return it != tensors_.end() ? &it->second : nullptr;
}

bool SafetensorsLoader::read_as_f32(const TensorView& tv, std::vector<f32>& out) {
    u64 n = tv.numel();
    out.resize(n);
    if (tv.dtype == DType::F32) {
        std::memcpy(out.data(), tv.data, n * sizeof(f32));
        return true;
    }
    if (tv.dtype == DType::BF16) {
        const auto* p = reinterpret_cast<const u16*>(tv.data);
        for (u64 i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
        return true;
    }
    if (tv.dtype == DType::F16) {
        const auto* p = reinterpret_cast<const u16*>(tv.data);
        for (u64 i = 0; i < n; ++i) out[i] = fp16_to_f32(p[i]);
        return true;
    }
    return false;
}

bool SafetensorsLoader::read_as_bf16(const TensorView& tv, std::vector<u16>& out) {
    u64 n = tv.numel();
    out.resize(n);
    if (tv.dtype == DType::BF16) {
        std::memcpy(out.data(), tv.data, n * sizeof(u16));
        return true;
    }
    if (tv.dtype == DType::F32) {
        const auto* p = reinterpret_cast<const f32*>(tv.data);
        for (u64 i = 0; i < n; ++i) out[i] = f32_to_bf16(p[i]);
        return true;
    }
    if (tv.dtype == DType::F16) {
        const auto* p = reinterpret_cast<const u16*>(tv.data);
        for (u64 i = 0; i < n; ++i) out[i] = f32_to_bf16(fp16_to_f32(p[i]));
        return true;
    }
    return false;
}

void quantize_w4a16_g128(const f32* src, u32 rows, u32 cols,
                         std::vector<u32>& qweight_out,
                         std::vector<u32>& scales_out) {
    const u32 groups_per_row = (cols + kW4GroupSize - 1) / kW4GroupSize;
    const u32 words_per_row = cols / 8;
    qweight_out.assign(static_cast<size_t>(rows) * words_per_row, 0);
    scales_out.assign(static_cast<size_t>(rows) * groups_per_row, 0);

    for (u32 r = 0; r < rows; ++r) {
        const f32* row = src + static_cast<size_t>(r) * cols;
        u32* qrow = qweight_out.data() + static_cast<size_t>(r) * words_per_row;
        u32* srow = scales_out.data() + static_cast<size_t>(r) * groups_per_row;

        for (u32 g = 0; g < groups_per_row; ++g) {
            u32 c0 = g * kW4GroupSize;
            u32 c1 = std::min(c0 + kW4GroupSize, cols);
            f32 wmin = row[c0];
            f32 wmax = row[c0];
            for (u32 c = c0 + 1; c < c1; ++c) {
                wmin = std::min(wmin, row[c]);
                wmax = std::max(wmax, row[c]);
            }
            // Asymmetric INT4 in [0..15]: w ~= q * scale + wmin
            f32 range = wmax - wmin;
            f32 scale = (range > 0.0f) ? (range / 15.0f) : 1.0f;
            f32 inv_scale = (range > 0.0f) ? (1.0f / scale) : 0.0f;
            u16 s_bf = f32_to_bf16(scale);
            u16 m_bf = f32_to_bf16(wmin);
            srow[g] = static_cast<u32>(s_bf) | (static_cast<u32>(m_bf) << 16);

            // Use rounded BF16 scale & min during quantization so rounding errors don't accumulate
            f32 s_q = bf16_to_f32(s_bf);
            f32 m_q = bf16_to_f32(m_bf);
            f32 inv_sq = (range > 0.0f && s_q > 0.0f) ? (1.0f / s_q) : inv_scale;

            for (u32 c = c0; c < c1; c += 8) {
                u32 qv[8] = {};
                for (u32 k = 0; k < 8 && (c + k) < c1; ++k) {
                    int q = static_cast<int>(std::nearbyint((row[c + k] - m_q) * inv_sq));
                    qv[k] = static_cast<u32>(std::clamp(q, 0, 15));
                }
                // Marlin half2-interleaved nibble order for 2-op PTX lop3.b32 dequant:
                // bits 0..3=q0, 4..7=q2, 8..11=q4, 12..15=q6, 16..19=q1, 20..23=q3, 24..27=q5, 28..31=q7
                u32 packed = (qv[0] << 0)  | (qv[2] << 4)  | (qv[4] << 8)  | (qv[6] << 12) |
                             (qv[1] << 16) | (qv[3] << 20) | (qv[5] << 24) | (qv[7] << 28);
                qrow[c / 8] = packed;
            }
        }
    }
}

void quantize_fp8_e4m3_rowwise(const f32* src, u32 rows, u32 cols,
                               std::vector<u8>& qweight_out,
                               std::vector<f32>& scales_out) {
    qweight_out.resize(static_cast<size_t>(rows) * cols);
    scales_out.resize(rows);

    for (u32 r = 0; r < rows; ++r) {
        const f32* row = src + static_cast<size_t>(r) * cols;
        u8* qrow = qweight_out.data() + static_cast<size_t>(r) * cols;
        f32 amax = 0.0f;
        for (u32 c = 0; c < cols; ++c) amax = std::max(amax, std::fabs(row[c]));
        f32 scale = (amax > 0.0f) ? (amax / 448.0f) : 1.0f;
        f32 inv_scale = (amax > 0.0f) ? (1.0f / scale) : 0.0f;
        scales_out[r] = scale;
        for (u32 c = 0; c < cols; ++c) {
            qrow[c] = f32_to_fp8_e4m3(row[c] * inv_scale);
        }
    }
}

bool pack_model(const std::filesystem::path& hf_model_dir,
                const std::filesystem::path& out_pack_path,
                QuantMode quant,
                std::string& err) {
    ModelConfig cfg;
    if (!ModelConfig::load_from_dir(hf_model_dir, cfg, err)) return false;

    std::ifstream cfg_in(hf_model_dir / "config.json");
    std::ostringstream cfg_ss;
    cfg_ss << cfg_in.rdbuf();
    std::string cfg_str = cfg_ss.str();

    // Streaming zero-RAM-copy text-only packer for Inkling-2-Mini (skips ~637 MB vision weights,
    // preserves native NVFP4/BF16 tensors verbatim, embeds mtp.safetensors + binary tokenizer).
    if (cfg.is_inkling2()) {
        SafetensorsLoader st;
        if (!st.open_dir(hf_model_dir, err)) return false;
        SafetensorsLoader st_mtp;
        bool has_mtp_file = false;
        std::error_code ec;
        if (std::filesystem::exists(hf_model_dir / "mtp.safetensors", ec)) {
            std::string merr;
            if (st_mtp.add_shard(hf_model_dir / "mtp.safetensors", merr)) {
                has_mtp_file = true;
            }
        }

        struct RefTensor {
            std::string name;
            DType dtype;
            std::vector<u64> shape;
            const u8* ptr;
            u64 nbytes;
            std::vector<u8> owned; // Non-empty for per-row FP8 quantized attention/unembed tensors
        };
        std::vector<RefTensor> refs;
        auto is_ink_fp8_target = [&](const std::string& n) -> bool {
            if (quant != QuantMode::FP8 && quant != QuantMode::W4A16) return false;
            if (n == "model.llm.unembed.weight") return true;
            return n.find(".attn.wq_du.weight") != std::string::npos ||
                   n.find(".attn.wr_du.weight") != std::string::npos ||
                   n.find(".attn.gate_attn.weight") != std::string::npos ||
                   n.find(".attn.wk_dv.weight") != std::string::npos ||
                   n.find(".attn.wv_dv.weight") != std::string::npos ||
                   n.find(".attn.wo_ud.weight") != std::string::npos;
        };
        auto is_mtp_fp8_target = [&](const std::string& n) -> bool {
            if (quant != QuantMode::FP8 && quant != QuantMode::W4A16) return false;
            return n.find(".self_attn.q_proj.weight") != std::string::npos ||
                   n.find(".self_attn.k_proj.weight") != std::string::npos ||
                   n.find(".self_attn.v_proj.weight") != std::string::npos ||
                   n.find(".self_attn.k_proj_ctx.weight") != std::string::npos ||
                   n.find(".self_attn.v_proj_ctx.weight") != std::string::npos ||
                   n.find(".self_attn.o_proj.weight") != std::string::npos ||
                   n.find(".mlp.gate_proj.weight") != std::string::npos ||
                   n.find(".mlp.up_proj.weight") != std::string::npos ||
                   n.find(".mlp.down_proj.weight") != std::string::npos;
        };
        // Helper: exact FP8 E4M3 -> F32 + F32 -> NVFP4 E2M1 nibble for 16GB mode (L20..23 MXFP8 -> NVFP4, -2.0 GiB)
        auto fp8_e4m3_to_f32_local = [](u8 b) -> f32 {
            u32 s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u;
            if (e == 15u && m == 7u) return 0.0f; // NaN
            f32 v = (e == 0u) ? std::ldexp(static_cast<f32>(m), -9)
                              : std::ldexp(1.0f + static_cast<f32>(m) * 0.125f, static_cast<int>(e) - 7);
            return s ? -v : v;
        };
        auto f32_to_fp4_nibble = [](f32 x) -> u8 {
            u8 sign = (x < 0.0f) ? 8u : 0u;
            f32 ax = std::fabs(x);
            // Midpoints between NVFP4 E2M1 magnitudes {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
            u8 mag = (ax < 0.25f) ? 0u : (ax < 0.75f) ? 1u : (ax < 1.25f) ? 2u :
                     (ax < 1.75f) ? 3u : (ax < 2.50f) ? 4u : (ax < 3.50f) ? 5u :
                     (ax < 5.00f) ? 6u : 7u;
            return static_cast<u8>(sign | mag);
        };
        for (const auto& [name, tv] : st.tensors()) {
            if (name.find("vision") != std::string::npos || name.find("visual") != std::string::npos) {
                continue; // Text-only pack: skip vision encoder weights (~637 MB)
            }
            // In 16GB mode (--quant w4a16), skip original MXFP8 *_weight_scale regardless of map iteration order
            if (quant == QuantMode::W4A16 && tv.shape.size() == 3 &&
                (name.find(".mlp.experts.w13_weight_scale") != std::string::npos ||
                 name.find(".mlp.experts.w2_weight_scale")  != std::string::npos)) {
                std::string w_name = name.substr(0, name.size() - 6); // strip "_scale"
                const auto* tv_w = st.get(w_name);
                if (tv_w && tv_w->dtype == DType::FP8_E4M3) continue;
            }
            if (quant == QuantMode::W4A16 && tv.dtype == DType::FP8_E4M3 && tv.shape.size() == 3 &&
                (name.find(".mlp.experts.w13_weight") != std::string::npos ||
                 name.find(".mlp.experts.w2_weight")  != std::string::npos)) {
                // Requantize MXFP8 (L20..23) -> native NVFP4 E2M1 (-2.0 GiB VRAM -> fits 16 GB consumer GPUs)
                std::string sc_name = name + "_scale";
                const auto* tv_sc = st.get(sc_name);
                if (tv_sc) {
                    u32 E = static_cast<u32>(tv.shape[0]);
                    u32 N = static_cast<u32>(tv.shape[1]);
                    u32 K = static_cast<u32>(tv.shape[2]);
                    u32 K2 = K >> 1, K16 = K >> 4, K32 = K >> 5;
                    constexpr f32 kOuterDenom = 448.0f * 6.0f;
                    std::vector<u8> nv_w(static_cast<size_t>(E) * N * K2);
                    std::vector<u8> nv_s(static_cast<size_t>(E) * N * K16);
                    std::vector<f32> nv_amax(E, 0.0f);
                    std::vector<f32> row_buf(K);
                    for (u32 e = 0; e < E; ++e) {
                        // Pass 1: find expert amax
                        f32 e_amax = 0.0f;
                        for (u32 n = 0; n < N; ++n) {
                            const u8* w_row = tv.data + (static_cast<size_t>(e) * N + n) * K;
                            const u8* s_row = tv_sc->data + (static_cast<size_t>(e) * N + n) * K32;
                            for (u32 b32 = 0; b32 < K32; ++b32) {
                                f32 sc32 = std::ldexp(1.0f, static_cast<int>(s_row[b32]) - 127);
                                for (u32 k = 0; k < 32; ++k) {
                                    f32 v = std::fabs(fp8_e4m3_to_f32_local(w_row[b32 * 32 + k]) * sc32);
                                    if (v > e_amax) e_amax = v;
                                }
                            }
                        }
                        if (e_amax < 1e-8f) e_amax = 1e-8f;
                        nv_amax[e] = e_amax;
                        f32 g_scale = e_amax / kOuterDenom;
                        // Pass 2: block-16 FP8 scale + NVFP4 E2M1 nibble packing
                        for (u32 n = 0; n < N; ++n) {
                            const u8* w_row = tv.data + (static_cast<size_t>(e) * N + n) * K;
                            const u8* s_row = tv_sc->data + (static_cast<size_t>(e) * N + n) * K32;
                            for (u32 b32 = 0; b32 < K32; ++b32) {
                                f32 sc32 = std::ldexp(1.0f, static_cast<int>(s_row[b32]) - 127);
                                for (u32 k = 0; k < 32; ++k) {
                                    row_buf[b32 * 32 + k] = fp8_e4m3_to_f32_local(w_row[b32 * 32 + k]) * sc32;
                                }
                            }
                            u8* dst_w = nv_w.data() + (static_cast<size_t>(e) * N + n) * K2;
                            u8* dst_s = nv_s.data() + (static_cast<size_t>(e) * N + n) * K16;
                            for (u32 b16 = 0; b16 < K16; ++b16) {
                                f32 bmax = 0.0f;
                                for (u32 k = 0; k < 16; ++k) {
                                    f32 av = std::fabs(row_buf[b16 * 16 + k]);
                                    if (av > bmax) bmax = av;
                                }
                                u8 bs_fp8 = f32_to_fp8_e4m3(bmax / (6.0f * g_scale));
                                dst_s[b16] = bs_fp8;
                                f32 eff_sc = fp8_e4m3_to_f32_local(bs_fp8) * g_scale;
                                f32 inv_sc = (eff_sc > 0.0f) ? (1.0f / eff_sc) : 0.0f;
                                for (u32 k = 0; k < 8; ++k) {
                                    u8 lo = f32_to_fp4_nibble(row_buf[b16 * 16 + 2 * k + 0] * inv_sc);
                                    u8 hi = f32_to_fp4_nibble(row_buf[b16 * 16 + 2 * k + 1] * inv_sc);
                                    dst_w[b16 * 8 + k] = static_cast<u8>((lo & 0xFu) | ((hi & 0xFu) << 4));
                                }
                            }
                        }
                    }
                    RefTensor rw{name, DType::U8, {E, N, K2}, nullptr, nv_w.size(), std::move(nv_w)};
                    rw.ptr = rw.owned.data(); refs.push_back(std::move(rw));
                    RefTensor rs{sc_name, DType::FP8_E4M3, {E, N, K16}, nullptr, nv_s.size(), std::move(nv_s)};
                    rs.ptr = rs.owned.data(); refs.push_back(std::move(rs));
                    std::vector<u8> am_b(nv_amax.size() * sizeof(f32));
                    std::memcpy(am_b.data(), nv_amax.data(), am_b.size());
                    RefTensor ra{name + "_amax", DType::F32, {E}, nullptr, am_b.size(), std::move(am_b)};
                    ra.ptr = ra.owned.data(); refs.push_back(std::move(ra));
                    std::vector<u8> od_b(sizeof(f32));
                    std::memcpy(od_b.data(), &kOuterDenom, sizeof(f32));
                    RefTensor ro{name + "_outer_scale_denominator", DType::F32, {1}, nullptr, od_b.size(), std::move(od_b)};
                    ro.ptr = ro.owned.data(); refs.push_back(std::move(ro));
                    continue;
                }
            }
            if (quant == QuantMode::W4A16 && name == "model.llm.embed.weight" && tv.shape.size() == 2) {
                // 16GB mode: W4A16 group-128 embed.weight (884 MiB BF16 -> 233 MiB W4A16 = -651 MiB VRAM)
                u32 rows = static_cast<u32>(tv.shape[0]);
                u32 cols = static_cast<u32>(tv.shape[1]);
                std::vector<f32> wf;
                SafetensorsLoader::read_as_f32(tv, wf);
                std::vector<u32> qw, sc;
                quantize_w4a16_g128(wf.data(), rows, cols, qw, sc);
                std::vector<u8> qw_b(qw.size() * sizeof(u32)); std::memcpy(qw_b.data(), qw.data(), qw_b.size());
                std::vector<u8> sc_b(sc.size() * sizeof(u32)); std::memcpy(sc_b.data(), sc.data(), sc_b.size());
                RefTensor rw{name + ".qweight", DType::U8, {rows, cols / 2}, nullptr, qw_b.size(), std::move(qw_b)};
                rw.ptr = rw.owned.data(); refs.push_back(std::move(rw));
                RefTensor rs{name + ".scales", DType::U8, {rows, (cols / 128) * 4}, nullptr, sc_b.size(), std::move(sc_b)};
                rs.ptr = rs.owned.data(); refs.push_back(std::move(rs));
                continue;
            }
            if (is_ink_fp8_target(name) && tv.shape.size() == 2) {
                u32 rows = static_cast<u32>(tv.shape[0]);
                u32 cols = static_cast<u32>(tv.shape[1]);
                std::vector<f32> wf;
                SafetensorsLoader::read_as_f32(tv, wf);
                std::vector<u8> qw;
                std::vector<f32> sc;
                quantize_fp8_e4m3_rowwise(wf.data(), rows, cols, qw, sc);
                RefTensor rw{name + ".fp8", DType::FP8_E4M3, {rows, cols}, nullptr, qw.size(), std::move(qw)};
                rw.ptr = rw.owned.data();
                refs.push_back(std::move(rw));
                std::vector<u8> sc_b(sc.size() * sizeof(f32));
                std::memcpy(sc_b.data(), sc.data(), sc_b.size());
                RefTensor rs{name + ".scale", DType::F32, {rows}, nullptr, sc_b.size(), std::move(sc_b)};
                rs.ptr = rs.owned.data();
                refs.push_back(std::move(rs));
            } else {
                refs.push_back({name, tv.dtype, tv.shape, tv.data, tv.nbytes, {}});
            }
        }
        if (has_mtp_file) {
            for (const auto& [name, tv] : st_mtp.tensors()) {
                if (is_mtp_fp8_target(name) && tv.shape.size() == 2) {
                    u32 rows = static_cast<u32>(tv.shape[0]);
                    u32 cols = static_cast<u32>(tv.shape[1]);
                    std::vector<f32> wf;
                    SafetensorsLoader::read_as_f32(tv, wf);
                    std::vector<u8> qw;
                    std::vector<f32> sc;
                    quantize_fp8_e4m3_rowwise(wf.data(), rows, cols, qw, sc);
                    RefTensor rw{"mtp." + name + ".fp8", DType::FP8_E4M3, {rows, cols}, nullptr, qw.size(), std::move(qw)};
                    rw.ptr = rw.owned.data();
                    refs.push_back(std::move(rw));
                    std::vector<u8> sc_b(sc.size() * sizeof(f32));
                    std::memcpy(sc_b.data(), sc.data(), sc_b.size());
                    RefTensor rs{"mtp." + name + ".scale", DType::F32, {rows}, nullptr, sc_b.size(), std::move(sc_b)};
                    rs.ptr = rs.owned.data();
                    refs.push_back(std::move(rs));
                } else {
                    refs.push_back({"mtp." + name, tv.dtype, tv.shape, tv.data, tv.nbytes, {}});
                }
            }
        }
        std::sort(refs.begin(), refs.end(), [](const RefTensor& a, const RefTensor& b) {
            return a.name < b.name;
        });

        std::vector<u8> tok_blob;
        {
            BpeTokenizer tok;
            std::string terr;
            if (tok.load_from_dir(hf_model_dir, terr)) {
                tok.serialize_binary(tok_blob);
            }
        }
        if (!tok_blob.empty()) {
            refs.push_back({"flint.tokenizer", DType::FP8_E4M3, {tok_blob.size()}, tok_blob.data(), tok_blob.size(), {}});
        }
        // Fix raw pointers for owned vectors after sort/reallocation
        for (auto& rt : refs) {
            if (!rt.owned.empty()) rt.ptr = rt.owned.data();
        }

        // Build binary PackHeader + PackTensorEntry directory + name arena (matches PackedModelReader)
        std::string name_arena;
        std::vector<PackTensorEntry> dir(refs.size());
        for (size_t i = 0; i < refs.size(); ++i) {
            dir[i] = {};
            dir[i].name_off = static_cast<u32>(name_arena.size());
            dir[i].name_len = static_cast<u16>(refs[i].name.size());
            name_arena += refs[i].name;
            dir[i].dtype = static_cast<u8>(refs[i].dtype);
            dir[i].ndim = static_cast<u8>(std::min<size_t>(4, refs[i].shape.size()));
            for (u8 d = 0; d < dir[i].ndim; ++d) dir[i].shape[d] = refs[i].shape[d];
            dir[i].nbytes = refs[i].nbytes;
        }

        PackHeader hdr{};
        std::memcpy(hdr.magic, kPackMagic, 8);
        hdr.version = kPackVersion;
        hdr.quant_mode = static_cast<u32>(QuantMode::NVFP4);
        hdr.num_tensors = static_cast<u32>(refs.size());
        hdr.config_json_len = static_cast<u32>(cfg_str.size());
        hdr.config_json_off = sizeof(PackHeader);
        hdr.tensor_dir_off  = align_up(hdr.config_json_off + cfg_str.size(), 64);
        hdr.name_arena_off  = align_up(hdr.tensor_dir_off + dir.size() * sizeof(PackTensorEntry), 64);
        hdr.data_off        = align_up(hdr.name_arena_off + name_arena.size(), 128);

        u64 cur_off = hdr.data_off;
        for (size_t i = 0; i < refs.size(); ++i) {
            cur_off = align_up(cur_off, 128);
            dir[i].data_off = cur_off;
            cur_off += refs[i].nbytes;
        }
        hdr.total_file_bytes = cur_off;

        int fd = ::open(out_pack_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            err = "Cannot create output pack file: " + out_pack_path.string();
            return false;
        }
        bool ok = write_all(fd, &hdr, sizeof(hdr)) &&
                  write_all(fd, cfg_str.data(), cfg_str.size()) &&
                  write_zeros(fd, hdr.tensor_dir_off - (hdr.config_json_off + cfg_str.size())) &&
                  write_all(fd, dir.data(), dir.size() * sizeof(PackTensorEntry)) &&
                  write_zeros(fd, hdr.name_arena_off - (hdr.tensor_dir_off + dir.size() * sizeof(PackTensorEntry))) &&
                  write_all(fd, name_arena.data(), name_arena.size()) &&
                  write_zeros(fd, hdr.data_off - (hdr.name_arena_off + name_arena.size()));
        u64 file_pos = hdr.data_off;
        for (size_t i = 0; i < refs.size() && ok; ++i) {
            if (dir[i].data_off > file_pos) {
                ok = write_zeros(fd, dir[i].data_off - file_pos);
                file_pos = dir[i].data_off;
            }
            if (ok) {
                ok = write_all(fd, refs[i].ptr, refs[i].nbytes);
                file_pos += refs[i].nbytes;
            }
        }
        ::close(fd);
        if (!ok) {
            err = "Write failed for " + out_pack_path.string();
            return false;
        }
        return true;
    }

    SafetensorsLoader st;
    if (!st.open_dir(hf_model_dir, err)) return false;

    struct PendingTensor {
        std::string name;
        DType dtype;
        std::vector<u64> shape;
        std::vector<u8> bytes;
    };
    std::vector<PendingTensor> items;

    auto add_bf16_raw = [&](const std::string& name, const TensorView* tv) -> bool {
        if (!tv) { err = "Missing tensor: " + name; return false; }
        std::vector<u16> b;
        if (!SafetensorsLoader::read_as_bf16(*tv, b)) {
            err = "Cannot convert tensor to BF16: " + name;
            return false;
        }
        PendingTensor pt;
        pt.name = name;
        pt.dtype = DType::BF16;
        pt.shape = tv->shape;
        pt.bytes.resize(b.size() * 2);
        std::memcpy(pt.bytes.data(), b.data(), pt.bytes.size());
        items.push_back(std::move(pt));
        return true;
    };

    auto add_linear = [&](const std::string& name, const f32* data, u32 rows, u32 cols) {
        if (quant == QuantMode::BF16) {
            PendingTensor pt;
            pt.name = name;
            pt.dtype = DType::BF16;
            pt.shape = {rows, cols};
            pt.bytes.resize(static_cast<size_t>(rows) * cols * 2);
            auto* dst = reinterpret_cast<u16*>(pt.bytes.data());
            for (size_t i = 0; i < static_cast<size_t>(rows) * cols; ++i) {
                dst[i] = f32_to_bf16(data[i]);
            }
            items.push_back(std::move(pt));
        } else if (quant == QuantMode::W4A16) {
            std::vector<u32> qw;
            std::vector<u32> sc;
            quantize_w4a16_g128(data, rows, cols, qw, sc);
            PendingTensor pt_w;
            pt_w.name = name + ".qweight";
            pt_w.dtype = DType::INT4_G128;
            pt_w.shape = {rows, cols / 8};
            pt_w.bytes.resize(qw.size() * 4);
            std::memcpy(pt_w.bytes.data(), qw.data(), pt_w.bytes.size());
            items.push_back(std::move(pt_w));

            PendingTensor pt_s;
            pt_s.name = name + ".scales";
            pt_s.dtype = DType::F32; // packed bf162(scale, min_val)
            pt_s.shape = {rows, (cols + kW4GroupSize - 1) / kW4GroupSize};
            pt_s.bytes.resize(sc.size() * 4);
            std::memcpy(pt_s.bytes.data(), sc.data(), pt_s.bytes.size());
            items.push_back(std::move(pt_s));
        } else {
            std::vector<u8> qw;
            std::vector<f32> sc;
            quantize_fp8_e4m3_rowwise(data, rows, cols, qw, sc);
            PendingTensor pt_w;
            pt_w.name = name + ".qweight";
            pt_w.dtype = DType::FP8_E4M3;
            pt_w.shape = {rows, cols};
            pt_w.bytes = std::move(qw);
            items.push_back(std::move(pt_w));

            PendingTensor pt_s;
            pt_s.name = name + ".scales";
            pt_s.dtype = DType::F32;
            pt_s.shape = {rows};
            pt_s.bytes.resize(sc.size() * 4);
            std::memcpy(pt_s.bytes.data(), sc.data(), pt_s.bytes.size());
            items.push_back(std::move(pt_s));
        }
    };

    // Embeddings & final norm (BF16); LM head is BF16 for bf16 packs, FP8 E4M3 for w4a16/fp8 packs
    if (!add_bf16_raw("model.embed_tokens.weight", st.get("model.embed_tokens.weight"))) return false;
    if (!add_bf16_raw("model.norm.weight", st.get("model.norm.weight"))) return false;
    if (!cfg.tie_word_embeddings) {
        const auto* t_lm = st.get("lm_head.weight");
        if (!t_lm) { err = "Missing tensor: lm_head.weight"; return false; }
        if (quant == QuantMode::BF16) {
            if (!add_bf16_raw("lm_head.weight", t_lm)) return false;
        } else {
            std::vector<f32> flm;
            SafetensorsLoader::read_as_f32(*t_lm, flm);
            std::vector<u8> qw;
            std::vector<f32> sc;
            quantize_fp8_e4m3_rowwise(flm.data(), cfg.vocab_size, cfg.hidden_size, qw, sc);
            PendingTensor pt_w;
            pt_w.name = "lm_head.qweight";
            pt_w.dtype = DType::FP8_E4M3;
            pt_w.shape = {cfg.vocab_size, cfg.hidden_size};
            pt_w.bytes = std::move(qw);
            items.push_back(std::move(pt_w));

            PendingTensor pt_s;
            pt_s.name = "lm_head.scales";
            pt_s.dtype = DType::F32;
            pt_s.shape = {cfg.vocab_size};
            pt_s.bytes.resize(sc.size() * 4);
            std::memcpy(pt_s.bytes.data(), sc.data(), pt_s.bytes.size());
            items.push_back(std::move(pt_s));
        }
    }

    std::vector<f32> fq, fk, fv, fo, fg, fu, fd;
    for (u32 l = 0; l < cfg.num_layers; ++l) {
        std::string pre = "model.layers." + std::to_string(l) + ".";
        if (!add_bf16_raw(pre + "input_layernorm.weight", st.get(pre + "input_layernorm.weight"))) return false;
        if (!add_bf16_raw(pre + "post_attention_layernorm.weight", st.get(pre + "post_attention_layernorm.weight"))) return false;

        if (cfg.has_qk_norm) {
            if (!add_bf16_raw(pre + "self_attn.q_norm.weight", st.get(pre + "self_attn.q_norm.weight"))) return false;
            if (!add_bf16_raw(pre + "self_attn.k_norm.weight", st.get(pre + "self_attn.k_norm.weight"))) return false;
        }

        // Fuse Q + K + V weights into [q_dim + 2*kv_dim, hidden]
        const auto* tq = st.get(pre + "self_attn.q_proj.weight");
        const auto* tk = st.get(pre + "self_attn.k_proj.weight");
        const auto* tv = st.get(pre + "self_attn.v_proj.weight");
        if (!tq || !tk || !tv) { err = "Missing QKV weights for layer " + std::to_string(l); return false; }
        SafetensorsLoader::read_as_f32(*tq, fq);
        SafetensorsLoader::read_as_f32(*tk, fk);
        SafetensorsLoader::read_as_f32(*tv, fv);

        std::vector<f32> qkv;
        qkv.reserve(fq.size() + fk.size() + fv.size());
        qkv.insert(qkv.end(), fq.begin(), fq.end());
        qkv.insert(qkv.end(), fk.begin(), fk.end());
        qkv.insert(qkv.end(), fv.begin(), fv.end());
        add_linear(pre + "self_attn.qkv_proj", qkv.data(), cfg.q_dim() + 2 * cfg.kv_dim(), cfg.hidden_size);

        if (cfg.has_qkv_bias) {
            const auto* bq = st.get(pre + "self_attn.q_proj.bias");
            const auto* bk = st.get(pre + "self_attn.k_proj.bias");
            const auto* bv = st.get(pre + "self_attn.v_proj.bias");
            if (bq && bk && bv) {
                std::vector<u16> uq, uk, uv, ub;
                SafetensorsLoader::read_as_bf16(*bq, uq);
                SafetensorsLoader::read_as_bf16(*bk, uk);
                SafetensorsLoader::read_as_bf16(*bv, uv);
                ub.insert(ub.end(), uq.begin(), uq.end());
                ub.insert(ub.end(), uk.begin(), uk.end());
                ub.insert(ub.end(), uv.begin(), uv.end());
                PendingTensor pt;
                pt.name = pre + "self_attn.qkv_bias";
                pt.dtype = DType::BF16;
                pt.shape = {ub.size()};
                pt.bytes.resize(ub.size() * 2);
                std::memcpy(pt.bytes.data(), ub.data(), pt.bytes.size());
                items.push_back(std::move(pt));
            }
        }

        const auto* to = st.get(pre + "self_attn.o_proj.weight");
        if (!to) { err = "Missing o_proj for layer " + std::to_string(l); return false; }
        SafetensorsLoader::read_as_f32(*to, fo);
        add_linear(pre + "self_attn.o_proj", fo.data(), cfg.hidden_size, cfg.q_dim());

        // Fuse Gate + Up weights into [2 * intermediate, hidden]
        const auto* tg = st.get(pre + "mlp.gate_proj.weight");
        const auto* tu = st.get(pre + "mlp.up_proj.weight");
        const auto* td = st.get(pre + "mlp.down_proj.weight");
        if (!tg || !tu || !td) { err = "Missing MLP weights for layer " + std::to_string(l); return false; }
        SafetensorsLoader::read_as_f32(*tg, fg);
        SafetensorsLoader::read_as_f32(*tu, fu);
        SafetensorsLoader::read_as_f32(*td, fd);

        std::vector<f32> gu;
        gu.reserve(fg.size() + fu.size());
        gu.insert(gu.end(), fg.begin(), fg.end());
        gu.insert(gu.end(), fu.begin(), fu.end());
        add_linear(pre + "mlp.gate_up_proj", gu.data(), 2 * cfg.intermediate_size, cfg.hidden_size);
        add_linear(pre + "mlp.down_proj", fd.data(), cfg.hidden_size, cfg.intermediate_size);
    }

    // Append pre-parsed binary tokenizer AT THE VERY END of the pack (so all GPU weights stay one contiguous slab)
    {
        BpeTokenizer tok;
        std::string terr;
        if (tok.load_from_dir(hf_model_dir, terr)) {
            PendingTensor pt_tok;
            pt_tok.name = "flint.tokenizer";
            pt_tok.dtype = DType::FP8_E4M3; // raw byte blob
            tok.serialize_binary(pt_tok.bytes);
            pt_tok.shape = {pt_tok.bytes.size()};
            items.push_back(std::move(pt_tok));
        }
    }

    // Layout & write 128-byte-aligned binary pack file
    std::string name_arena;
    std::vector<PackTensorEntry> entries(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        entries[i].name_off = static_cast<u32>(name_arena.size());
        entries[i].name_len = static_cast<u16>(items[i].name.size());
        entries[i].dtype = static_cast<u8>(items[i].dtype);
        entries[i].ndim = static_cast<u8>(std::min<size_t>(items[i].shape.size(), 4));
        for (u8 d = 0; d < entries[i].ndim; ++d) entries[i].shape[d] = items[i].shape[d];
        entries[i].nbytes = items[i].bytes.size();
        name_arena.append(items[i].name);
        name_arena.push_back('\0');
    }

    PackHeader hdr{};
    std::memcpy(hdr.magic, kPackMagic, 8);
    hdr.version = kPackVersion;
    hdr.quant_mode = static_cast<u32>(quant);
    hdr.num_tensors = static_cast<u32>(entries.size());

    u64 off = align_up(sizeof(PackHeader), 64);
    hdr.config_json_off = off;
    hdr.config_json_len = static_cast<u32>(cfg_str.size());
    off = align_up(off + cfg_str.size(), 64);

    hdr.tensor_dir_off = off;
    off = align_up(off + entries.size() * sizeof(PackTensorEntry), 64);

    hdr.name_arena_off = off;
    off = align_up(off + name_arena.size(), 128);

    hdr.data_off = off;
    for (size_t i = 0; i < entries.size(); ++i) {
        entries[i].data_off = off;
        off = align_up(off + entries[i].nbytes, 128);
    }
    hdr.total_file_bytes = off;

    auto tmp_path = out_pack_path;
    tmp_path += ".tmp." + std::to_string(::getpid());
    int out_fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (out_fd < 0) {
        err = "Cannot create pack output: " + tmp_path.string();
        return false;
    }

    u64 cur = 0;
    auto write_at = [&](u64 target_off, const void* ptr, size_t sz) -> bool {
        if (target_off > cur && !write_zeros(out_fd, target_off - cur)) return false;
        cur = target_off;
        if (sz > 0 && !write_all(out_fd, ptr, sz)) return false;
        cur += sz;
        return true;
    };

    bool ok = write_at(0, &hdr, sizeof(hdr)) &&
              write_at(hdr.config_json_off, cfg_str.data(), cfg_str.size()) &&
              write_at(hdr.tensor_dir_off, entries.data(), entries.size() * sizeof(PackTensorEntry)) &&
              write_at(hdr.name_arena_off, name_arena.data(), name_arena.size());

    for (size_t i = 0; ok && i < items.size(); ++i) {
        ok = write_at(entries[i].data_off, items[i].bytes.data(), items[i].bytes.size());
    }
    if (ok && hdr.total_file_bytes > cur) {
        ok = write_zeros(out_fd, hdr.total_file_bytes - cur);
    }
    ::close(out_fd);

    if (!ok || ::rename(tmp_path.c_str(), out_pack_path.c_str()) != 0) {
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        err = "Failed writing packed model file";
        return false;
    }
    return true;
}

bool PackedModelReader::open(const std::filesystem::path& pack_path, std::string& err) {
    if (!mmap_.open(pack_path, true)) {
        err = "Failed to mmap pack file: " + pack_path.string();
        return false;
    }
    if (mmap_.size() < sizeof(PackHeader)) {
        err = "Truncated pack file";
        return false;
    }
    hdr_ = reinterpret_cast<const PackHeader*>(mmap_.data());
    if (std::memcmp(hdr_->magic, kPackMagic, 8) != 0 || hdr_->version != kPackVersion) {
        err = "Invalid flint.pack header/version";
        return false;
    }
    config_json_.assign(reinterpret_cast<const char*>(mmap_.data() + hdr_->config_json_off),
                        hdr_->config_json_len);

    const auto* entries = reinterpret_cast<const PackTensorEntry*>(mmap_.data() + hdr_->tensor_dir_off);
    const char* names = reinterpret_cast<const char*>(mmap_.data() + hdr_->name_arena_off);
    tensors_.clear();
    tensors_.reserve(hdr_->num_tensors);
    for (u32 i = 0; i < hdr_->num_tensors; ++i) {
        const auto& e = entries[i];
        TensorView tv;
        tv.name.assign(names + e.name_off, e.name_len);
        tv.dtype = static_cast<DType>(e.dtype);
        for (u8 d = 0; d < e.ndim; ++d) tv.shape.push_back(e.shape[d]);
        tv.data = mmap_.data() + e.data_off;
        tv.nbytes = e.nbytes;
        tensors_[tv.name] = std::move(tv);
    }
    return true;
}

const TensorView* PackedModelReader::get(const std::string& name) const noexcept {
    auto it = tensors_.find(name);
    return it != tensors_.end() ? &it->second : nullptr;
}

} // namespace flint
