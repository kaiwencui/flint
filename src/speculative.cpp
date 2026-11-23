#include "speculative.hpp"

#include <algorithm>

namespace flint {

std::vector<i32> NgramDrafter::draft(u32 max_draft) const {
    std::vector<i32> out;
    const size_t n = history_.size();
    if (n < 4 || max_draft == 0) return out;

    // Try longest n-gram match first (n = 5, 4, 3, 2) searching most-recent first
    constexpr u32 kMaxN = 5;
    constexpr u32 kMinN = 2;
    const size_t max_window = std::min<size_t>(n, 16384);
    const size_t start_bound = n - max_window;

    for (u32 ng = std::min<u32>(kMaxN, static_cast<u32>(n - 1)); ng >= kMinN; --ng) {
        const i32* pat = history_.data() + (n - ng);
        // Search backward from n - ng - 1 down to start_bound
        for (size_t pos = n - ng; pos > start_bound; --pos) {
            size_t idx = pos - 1;
            if (history_[idx + ng - 1] != pat[ng - 1]) continue;
            bool match = true;
            for (u32 k = 0; k + 1 < ng; ++k) {
                if (history_[idx + k] != pat[k]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                size_t avail = (n - 1) - (idx + ng);
                // Even if right next to end, grab up to max_draft continuation tokens
                avail = std::min<size_t>(max_draft, n - (idx + ng));
                if (avail > 0) {
                    out.reserve(avail);
                    for (size_t j = 0; j < avail; ++j) {
                        out.push_back(history_[idx + ng + j]);
                    }
                    return out;
                }
            }
        }
    }
    return out;
}

SpecVerifyResult verify_speculative_greedy(std::span<const i32> draft_tokens,
                                           std::span<const i32> target_next) {
    SpecVerifyResult res;
    const size_t K = draft_tokens.size();
    res.emitted.reserve(K + 1);

    for (size_t i = 0; i < K; ++i) {
        i32 t_tok = target_next[i];
        res.emitted.push_back(t_tok);
        if (t_tok == draft_tokens[i]) {
            res.accepted_drafts++;
        } else {
            // Divergence at position i: emit target's exact token and stop
            return res;
        }
    }
    // All K drafts accepted -> emit the (K+1)-th bonus token
    if (target_next.size() > K) {
        res.emitted.push_back(target_next[K]);
    }
    return res;
}

} // namespace flint
