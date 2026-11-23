#pragma once

#include "common.hpp"
#include "model_config.hpp"

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace flint {

enum class DType : u8 {
    F32     = 0,
    BF16    = 1,
    F16     = 2,
    INT4_G128 = 3, // Packed W4A16 group-128 (8 x u4 per u32)
    FP8_E4M3  = 4,
    U8        = 5  // Packed NVFP4 (2 x E2M1 per u8)
};

struct TensorView {
    std::string name;
    DType dtype = DType::BF16;
    std::vector<u64> shape;
    const u8* data = nullptr;
    u64 nbytes = 0;

    u64 numel() const noexcept {
        u64 n = 1;
        for (u64 d : shape) n *= d;
        return n;
    }
};

// Zero-copy mmap loader for HuggingFace .safetensors (single or multi-shard)
class SafetensorsLoader {
public:
    bool open_dir(const std::filesystem::path& model_dir, std::string& err);
    const TensorView* get(const std::string& name) const noexcept;
    const std::unordered_map<std::string, TensorView>& tensors() const noexcept { return tensors_; }
    u64 total_tensor_bytes() const noexcept { return total_bytes_; }

    // Helper: read any F32/BF16/F16 tensor into host F32 vector (for CPU ref & quantization)
    static bool read_as_f32(const TensorView& tv, std::vector<f32>& out);
    // Helper: read any F32/BF16/F16 tensor into host BF16 vector (u16)
    static bool read_as_bf16(const TensorView& tv, std::vector<u16>& out);
    bool add_shard(const std::filesystem::path& shard_path, std::string& err);

private:
    std::vector<MmapFile> shards_;
    std::unordered_map<std::string, TensorView> tensors_;
    u64 total_bytes_ = 0;
};

// ============================================================================
// Offline 64-byte-aligned Zero-Copy Packed Format (`flint.pack`)
// ============================================================================
// Enables instant mmap + async H2D loading of BF16, W4A16 (INT4 group-128),
// or FP8 E4M3 weights with pre-fused QKV and Gate+Up matrices
constexpr char kPackMagic[8] = {'F', 'L', 'N', 'T', 'P', 'A', 'K', '\0'};
constexpr u32 kPackVersion = 2; // v2: asymmetric INT4-g128 (scale,min) + Marlin half2 nibble order + binary tokenizer
constexpr u32 kW4GroupSize = 128;

struct alignas(64) PackHeader {
    char magic[8];
    u32 version;
    u32 quant_mode; // QuantMode
    u32 num_tensors;
    u32 config_json_len;
    u64 config_json_off;
    u64 tensor_dir_off;
    u64 name_arena_off;
    u64 data_off;
    u64 total_file_bytes;
    u8 pad[16];
};
static_assert(sizeof(PackHeader) == 128);

struct PackTensorEntry {
    u32 name_off;
    u16 name_len;
    u8 dtype; // DType
    u8 ndim;
    u64 shape[4];
    u64 data_off; // Offset from file start (always 128-byte aligned)
    u64 nbytes;
};
static_assert(sizeof(PackTensorEntry) == 56);

// Offline packer: converts HF safetensors dir -> single 128B-aligned flint.pack
// Pre-fuses Q+K+V and Gate+Up weights and quantizes linear projections to BF16 / W4A16 / FP8
bool pack_model(const std::filesystem::path& hf_model_dir,
                const std::filesystem::path& out_pack_path,
                QuantMode quant,
                std::string& err);

// Zero-copy mmap reader for flint.pack
class PackedModelReader {
public:
    bool open(const std::filesystem::path& pack_path, std::string& err);
    QuantMode quant_mode() const noexcept { return static_cast<QuantMode>(hdr_->quant_mode); }
    const TensorView* get(const std::string& name) const noexcept;
    const std::unordered_map<std::string, TensorView>& tensors() const noexcept { return tensors_; }
    const std::string& config_json() const noexcept { return config_json_; }
    size_t file_bytes() const noexcept { return mmap_.size(); }
    const u8* base_ptr() const noexcept { return mmap_.data(); }
    u64 data_offset() const noexcept { return hdr_->data_off; }
    u64 weight_slab_bytes() const noexcept {
        // Everything from data_off up to tokenizer blob (or end of file) is contiguous GPU weight slab
        if (auto* tv = get("flint.tokenizer")) {
            return static_cast<u64>(tv->data - (mmap_.data() + hdr_->data_off));
        }
        return hdr_->total_file_bytes - hdr_->data_off;
    }

private:
    MmapFile mmap_;
    const PackHeader* hdr_ = nullptr;
    std::string config_json_;
    std::unordered_map<std::string, TensorView> tensors_;
};

// Quantize a [rows, cols] weight matrix into asymmetric W4A16 INT4 group-128:
// - qweight: [rows, cols / 8] u32 (8 unsigned 4-bit ints in [0..15], Marlin half2-interleaved nibble order:
// nibbles 0..3 = (q0,q2,q4,q6), nibbles 4..7 = (q1,q3,q5,q7) for PTX lop3.b32 FP16-magic dequant)
// - scales: [rows, cols / 128] u32 (packed bf162: low16 = bf16(scale), high16 = bf16(min_val))
// reconstruction: w = q * scale + min_val
void quantize_w4a16_g128(const f32* src, u32 rows, u32 cols,
                         std::vector<u32>& qweight_out,
                         std::vector<u32>& scales_out);

// Quantize a [rows, cols] weight matrix into FP8 E4M3 with per-row FP32 scales:
// - qweight: [rows, cols] u8 (FP8 E4M3)
// - scales: [rows] f32
void quantize_fp8_e4m3_rowwise(const f32* src, u32 rows, u32 cols,
                               std::vector<u8>& qweight_out,
                               std::vector<f32>& scales_out);

} // namespace flint
