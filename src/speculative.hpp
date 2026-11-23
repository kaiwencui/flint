#pragma once

#include "common.hpp"

#include <span>
#include <unordered_map>
#include <vector>

namespace flint {

enum class SpecMode : u8 {
    None  = 0,
    Ngram = 1, // Zero-VRAM prompt/context n-gram suffix lookup drafter (huge for code/FIM)
    Draft = 2, // Tiny draft model (e.g. Qwen2.5-0.5B drafting for 7B/14B)
    Mtp   = 3, // Built-in Multi-Token Prediction head (Inkling-2-Mini mtp.safetensors)
    Loop1 = 4  // Looped-Depth Self-Speculative Decoding (1-pass core drafts 3-pass core, 0 extra VRAM)
};

// Zero-VRAM Prompt / Context N-Gram Suffix Drafter.
// Indexes n-grams (n = 4, 3, 2) of the prompt + generated token history and drafts
// up to K continuation tokens in microseconds. Ideal for code editing & FIM
class NgramDrafter {
public:
    void reset() { history_.clear(); }
    void ingest(std::span<const i32> tokens) {
        history_.insert(history_.end(), tokens.begin(), tokens.end());
    }
    void push(i32 tok) { history_.push_back(tok); }

    // Draft up to `max_draft` tokens matching the longest recent suffix in history_.
    // Searches backward from most recent position for freshest local code patterns
    std::vector<i32> draft(u32 max_draft = 6) const;

private:
    std::vector<i32> history_;
};

// Exact mathematically-lossless greedy speculative verifier:
// Given `draft_tokens[0..K-1]` and target argmax predictions `target_next[0..K]`
// (where `target_next[i]` is the target model's greedy prediction after `draft_tokens[0..i-1]`):
// Returns the accepted prefix + 1 bonus/corrected token (`1..K+1` tokens guaranteed bit-identical to non-speculative greedy).
struct SpecVerifyResult {
    std::vector<i32> emitted;
    u32 accepted_drafts = 0; // Number of draft tokens accepted (0..K)
};

SpecVerifyResult verify_speculative_greedy(std::span<const i32> draft_tokens,
                                           std::span<const i32> target_next);

} // namespace flint
