#pragma once

#include "common.hpp"
#include "json.hpp"

#include <array>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace flint {

struct AddedToken {
    i32 id = -1;
    std::string content;
    bool special = false;
    bool lstrip = false;
    bool rstrip = false;
};

// Pure C++23 Byte-Level BPE tokenizer parsing HuggingFace tokenizer.json directly.
// Supports Qwen2.5, Qwen3, and Llama-3.1 / 3.2 with 100% parity against transformers.AutoTokenizer.
class BpeTokenizer {
public:
    BpeTokenizer();

    bool load_from_dir(const std::filesystem::path& model_dir, std::string& err);
    bool load_file(const std::filesystem::path& tokenizer_json, std::string& err);

    // Zero-JSON-parse binary serialization for instant (<25ms) loading from flint.pack
    void serialize_binary(std::vector<u8>& out) const;
    bool load_binary(std::span<const u8> data, std::string& err);

    std::vector<i32> encode(std::string_view text, bool add_bos = false) const;
    std::string decode(std::span<const i32> ids, bool skip_special_tokens = false) const;
    std::string decode_token(i32 id, bool skip_special_tokens = false) const;

    // Chat template formatting (Qwen ChatML / Llama-3 Instruct)
    std::string apply_chat_template(std::string_view system_prompt,
                                    std::string_view user_prompt,
                                    bool is_llama3) const;

    u32 vocab_size() const noexcept { return static_cast<u32>(id_to_token_.size()); }
    i32 token_to_id(std::string_view tok) const noexcept;
    bool is_special(i32 id) const noexcept;

private:
    void init_byte_tables();
    void pretokenize(std::string_view text, std::vector<std::string_view>& out) const;
    void bpe_encode_piece(std::string_view piece, std::vector<i32>& out) const;

    u32 max_digit_run_ = 1; // 1 for Qwen2.5/Qwen3, 3 for Llama-3 / o200k
    bool o200k_mode_ = false;
    bool ignore_merges_ = false;
    i32 bos_id_ = -1;
    bool add_bos_by_default_ = false;

    std::unordered_map<std::string, i32> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::vector<u8> is_special_id_;

    // Merge rank keyed by "left\x01right" -> rank (smaller = higher priority)
    std::unordered_map<std::string, u32> merge_ranks_;

    // Sorted longest-first for greedy special/added token matching
    std::vector<AddedToken> added_tokens_;

    // GPT-2 byte <-> UTF-8 string tables
    std::array<std::string, 256> byte_to_utf8_;
    std::unordered_map<u32, u8> cp_to_byte_;
};

} // namespace flint
