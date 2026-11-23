#include "tokenizer.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace flint {

namespace {

// Decode one UTF-8 codepoint at s[pos], advancing pos
inline u32 decode_utf8(std::string_view s, size_t& pos) noexcept {
    if (pos >= s.size()) return 0;
    u8 b0 = static_cast<u8>(s[pos++]);
    if (b0 < 0x80u) return b0;
    if ((b0 & 0xE0u) == 0xC0u && pos < s.size()) {
        u8 b1 = static_cast<u8>(s[pos++]);
        return ((b0 & 0x1Fu) << 6) | (b1 & 0x3Fu);
    }
    if ((b0 & 0xF0u) == 0xE0u && pos + 1 < s.size()) {
        u8 b1 = static_cast<u8>(s[pos++]);
        u8 b2 = static_cast<u8>(s[pos++]);
        return ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
    }
    if ((b0 & 0xF8u) == 0xF0u && pos + 2 < s.size()) {
        u8 b1 = static_cast<u8>(s[pos++]);
        u8 b2 = static_cast<u8>(s[pos++]);
        u8 b3 = static_cast<u8>(s[pos++]);
        return ((b0 & 0x07u) << 18) | ((b1 & 0x3Fu) << 12) | ((b2 & 0x3Fu) << 6) | (b3 & 0x3Fu);
    }
    return 0xFFFDu;
}

inline void encode_utf8(u32 cp, std::string& out) {
    if (cp <= 0x7Fu) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFu) {
        out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
        out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}

// Accurate Unicode general category helpers for Qwen2.5/Qwen3/Llama-3 pre-tokenization
inline bool is_unicode_ws(u32 cp) noexcept {
    if (cp <= 0x20u) {
        return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v';
    }
    if (cp == 0x0085u || cp == 0x00A0u || cp == 0x1680u ||
        (cp >= 0x2000u && cp <= 0x200Au) || cp == 0x2028u || cp == 0x2029u ||
        cp == 0x202Fu || cp == 0x205Fu || cp == 0x3000u) {
        return true;
    }
    return false;
}

inline bool is_unicode_number(u32 cp) noexcept {
    if (cp >= '0' && cp <= '9') return true;
    if (cp == 0x00B2u || cp == 0x00B3u || cp == 0x00B9u ||
        (cp >= 0x00BCu && cp <= 0x00BEu)) return true;
    if ((cp >= 0x0660u && cp <= 0x0669u) || (cp >= 0x06F0u && cp <= 0x06F9u) ||
        (cp >= 0x0966u && cp <= 0x096Fu) || (cp >= 0xFF10u && cp <= 0xFF19u) ||
        (cp >= 0x2150u && cp <= 0x2189u)) return true;
    return false;
}

inline bool is_unicode_letter(u32 cp) noexcept {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp < 0x80u) return false;
    // Latin-1 Supplement letters
    if ((cp >= 0x00C0u && cp <= 0x00D6u) || (cp >= 0x00D8u && cp <= 0x00F6u) ||
        (cp >= 0x00F8u && cp <= 0x02FFu)) return true;
    // Greek & Coptic, Cyrillic, Armenian, Hebrew, Arabic, Devanagari, etc.
    if ((cp >= 0x0370u && cp <= 0x0377u) || (cp >= 0x037Au && cp <= 0x037Du) ||
        cp == 0x037Fu || (cp >= 0x0386u && cp <= 0x03FFu && cp != 0x0387u && cp != 0x038Bu && cp != 0x038Du && cp != 0x03A2u)) return true;
    if (cp >= 0x0400u && cp <= 0x052Fu) return true; // Cyrillic
    if (cp >= 0x0531u && cp <= 0x0587u) return true; // Armenian
    if (cp >= 0x05D0u && cp <= 0x05EAu) return true; // Hebrew
    if (cp >= 0x0620u && cp <= 0x064Au) return true; // Arabic
    if (cp >= 0x066Eu && cp <= 0x06D5u) return true;
    if (cp >= 0x0904u && cp <= 0x0939u) return true; // Devanagari
    if (cp >= 0x0E01u && cp <= 0x0E30u) return true; // Thai
    if (cp >= 0x1100u && cp <= 0x11FFu) return true; // Hangul Jamo
    if (cp >= 0x1E00u && cp <= 0x1EFFu) return true; // Latin Extended Additional
    if (cp >= 0x3041u && cp <= 0x3096u) return true; // Hiragana
    if (cp >= 0x30A1u && cp <= 0x30FAu) return true; // Katakana
    if (cp >= 0x3131u && cp <= 0x318Eu) return true; // Hangul Compat Jamo
    if (cp >= 0x3400u && cp <= 0x4DBFu) return true; // CJK Ext A
    if (cp >= 0x4E00u && cp <= 0x9FFFu) return true; // CJK Unified Ideographs
    if (cp >= 0xAC00u && cp <= 0xD7A3u) return true; // Hangul Syllables
    if (cp >= 0xF900u && cp <= 0xFAFFu) return true; // CJK Compat
    if ((cp >= 0xFF21u && cp <= 0xFF3Au) || (cp >= 0xFF41u && cp <= 0xFF5Au) ||
        (cp >= 0xFF66u && cp <= 0xFFDCu)) return true;
    if (cp >= 0x20000u && cp <= 0x2FA1Fu) return true; // CJK Ext B..F
    return false;
}

inline char ascii_lower_ch(char c) noexcept {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c;
}

// Match English contractions (?i:'s|'t|'re|'ve|'m|'ll|'d) at s[pos]
size_t match_contraction(std::string_view s, size_t pos) noexcept {
    if (pos + 2 > s.size() || s[pos] != '\'') return 0;
    char c1 = ascii_lower_ch(s[pos + 1]);
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
    if (pos + 3 <= s.size()) {
        char c2 = ascii_lower_ch(s[pos + 2]);
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
            return 3;
        }
    }
    return 0;
}

} // namespace

BpeTokenizer::BpeTokenizer() {
    init_byte_tables();
}

void BpeTokenizer::init_byte_tables() {
    std::array<bool, 256> direct{};
    for (int b = 33; b <= 126; ++b) direct[b] = true;
    for (int b = 161; b <= 172; ++b) direct[b] = true;
    for (int b = 174; b <= 255; ++b) direct[b] = true;

    u32 extra = 0;
    for (int b = 0; b < 256; ++b) {
        u32 cp = direct[b] ? static_cast<u32>(b) : (256u + extra++);
        std::string u8s;
        encode_utf8(cp, u8s);
        byte_to_utf8_[b] = std::move(u8s);
        cp_to_byte_[cp] = static_cast<u8>(b);
    }
}

bool BpeTokenizer::load_from_dir(const std::filesystem::path& model_dir, std::string& err) {
    return load_file(model_dir / "tokenizer.json", err);
}

bool BpeTokenizer::load_file(const std::filesystem::path& tokenizer_json, std::string& err) {
    MmapFile mf;
    if (!mf.open(tokenizer_json)) {
        err = "Cannot open " + tokenizer_json.string();
        return false;
    }
    JsonValue root;
    std::string_view sv(reinterpret_cast<const char*>(mf.data()), mf.size());
    if (!parse_json(sv, root, err)) return false;

    // Detect pre-tokenizer mode: o200k_base (Inkling-2), Llama-3 ({1,3}), or Qwen2.5/Qwen3 (1)
    max_digit_run_ = 1;
    o200k_mode_ = false;
    ignore_merges_ = false;
    if (const auto* pt = root.get("pre_tokenizer")) {
        auto check_pattern = [&](const JsonValue& node) {
            if (const auto* pat = node.get("pattern")) {
                std::string rx = pat->get_string("Regex");
                if (rx.find("p{Lu}") != std::string::npos || rx.find("[\\r\\n/]*") != std::string::npos) {
                    o200k_mode_ = true;
                    max_digit_run_ = 3;
                } else if (rx.find("{1,3}") != std::string::npos) {
                    max_digit_run_ = 3;
                }
            }
        };
        check_pattern(*pt);
        if (const auto* list = pt->get("pretokenizers"); list && list->is_array()) {
            for (const auto& sub : list->as_array()) check_pattern(sub);
        }
    }

    // Load model.vocab
    const auto* model = root.get("model");
    if (!model || !model->is_object()) {
        err = "tokenizer.json missing 'model' object";
        return false;
    }
    ignore_merges_ = model->get_bool("ignore_merges", false);
    const auto* vocab = model->get("vocab");
    if (!vocab || !vocab->is_object()) {
        err = "tokenizer.json missing 'model.vocab'";
        return false;
    }

    token_to_id_.clear();
    i32 max_id = -1;
    for (const auto& [tok, v] : vocab->as_object()) {
        i32 id = static_cast<i32>(v.as_int(-1));
        if (id > max_id) max_id = id;
    }

    // Also inspect added_tokens to get true max_id
    added_tokens_.clear();
    if (const auto* added = root.get("added_tokens"); added && added->is_array()) {
        for (const auto& item : added->as_array()) {
            if (!item.is_object()) continue;
            AddedToken at;
            at.id = static_cast<i32>(item.get_int("id", -1));
            at.content = item.get_string("content");
            at.special = item.get_bool("special", false);
            at.lstrip = item.get_bool("lstrip", false);
            at.rstrip = item.get_bool("rstrip", false);
            if (at.id >= 0 && !at.content.empty()) {
                if (at.id > max_id) max_id = at.id;
                added_tokens_.push_back(std::move(at));
            }
        }
    }

    id_to_token_.assign(static_cast<size_t>(max_id + 1), "");
    is_special_id_.assign(static_cast<size_t>(max_id + 1), 0);
    token_to_id_.reserve(vocab->as_object().size() + added_tokens_.size());

    for (const auto& [tok, v] : vocab->as_object()) {
        i32 id = static_cast<i32>(v.as_int(-1));
        if (id >= 0) {
            token_to_id_[tok] = id;
            id_to_token_[static_cast<size_t>(id)] = tok;
        }
    }
    for (const auto& at : added_tokens_) {
        token_to_id_[at.content] = at.id;
        id_to_token_[static_cast<size_t>(at.id)] = at.content;
        if (at.special) is_special_id_[static_cast<size_t>(at.id)] = 1;
        if (at.content == "<|begin_of_text|>") {
            bos_id_ = at.id;
            add_bos_by_default_ = !o200k_mode_; // Llama-3 prepends BOS; Inkling-2 (o200k) does not
        }
    }

    // Sort added_tokens longest-first for greedy matching
    std::sort(added_tokens_.begin(), added_tokens_.end(),
              [](const AddedToken& a, const AddedToken& b) {
                  return a.content.size() > b.content.size();
              });

    // Load model.merges (handles both "a b" strings and ["a", "b"] 2-element arrays)
    merge_ranks_.clear();
    if (const auto* merges = model->get("merges"); merges && merges->is_array()) {
        const auto& arr = merges->as_array();
        merge_ranks_.reserve(arr.size());
        std::string key;
        for (size_t r = 0; r < arr.size(); ++r) {
            const auto& m = arr[r];
            if (m.is_string()) {
                const auto& ms = m.as_string();
                size_t sp = ms.find(' ');
                if (sp != std::string::npos) {
                    key.clear();
                    key.append(ms.data(), sp);
                    key.push_back('\x01');
                    key.append(ms.data() + sp + 1, ms.size() - sp - 1);
                    merge_ranks_.emplace(key, static_cast<u32>(r));
                }
            } else if (m.is_array() && m.as_array().size() == 2 &&
                       m.as_array()[0].is_string() && m.as_array()[1].is_string()) {
                key.clear();
                key.append(m.as_array()[0].as_string());
                key.push_back('\x01');
                key.append(m.as_array()[1].as_string());
                merge_ranks_.emplace(key, static_cast<u32>(r));
            }
        }
    }

    return true;
}

i32 BpeTokenizer::token_to_id(std::string_view tok) const noexcept {
    auto it = token_to_id_.find(std::string(tok));
    return it != token_to_id_.end() ? it->second : -1;
}

bool BpeTokenizer::is_special(i32 id) const noexcept {
    if (id < 0 || static_cast<size_t>(id) >= is_special_id_.size()) return false;
    return is_special_id_[static_cast<size_t>(id)] != 0;
}

// Exact Qwen2.5 / Qwen3 / Llama-3 pre-tokenizer:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)
// | [^\r\n\p{L}\p{N}]?\p{L}+
// | \p{N}{1,max_digit_run_}
// | ?[^\s\p{L}\p{N}]+[\r\n]*
// | \s*[\r\n]+
// | \s+(?\S)
// | \s+
void BpeTokenizer::pretokenize(std::string_view s, std::vector<std::string_view>& out) const {
    out.clear();
    size_t pos = 0;
    const size_t n = s.size();

    auto is_upper_or_title = [](u32 c) noexcept {
        if (c >= 'A' && c <= 'Z') return true;
        if ((c >= 0xC0 && c <= 0xD6) || (c >= 0xD8 && c <= 0xDE)) return true;
        return false;
    };
    auto is_combining_mark = [](u32 c) noexcept {
        return (c >= 0x0300 && c <= 0x036F) || // Combining Diacritical Marks
               (c >= 0x0900 && c <= 0x0903) || (c >= 0x093A && c <= 0x094F) || (c >= 0x0951 && c <= 0x0957) || (c >= 0x0962 && c <= 0x0963) || // Devanagari \p{M}
               (c >= 0x064B && c <= 0x065F) || (c == 0x0670) || // Arabic \p{M}
               (c >= 0x0E31 && c <= 0x0E3A) || (c >= 0x0E47 && c <= 0x0E4E); // Thai \p{M}
    };
    auto is_lower_or_other_letter = [&](u32 c) noexcept {
        return (is_unicode_letter(c) || is_combining_mark(c)) && !is_upper_or_title(c);
    };

    while (pos < n) {
        if (!o200k_mode_) {
            // Branch 1 (Qwen2/Llama-3): Contractions (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if (size_t clen = match_contraction(s, pos); clen > 0) {
                out.push_back(s.substr(pos, clen));
                pos += clen;
                continue;
            }
        }

        size_t p1 = pos;
        u32 cp0 = decode_utf8(s, p1);

        if (o200k_mode_) {
            // o200k_base Branches 1 & 2:
            // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|...)?
            // | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|...)?
            auto try_o200k_word = [&](size_t letter_start) -> size_t {
                size_t cur = letter_start;
                size_t upper_end = cur;
                while (cur < n) {
                    size_t nxt = cur;
                    u32 c = decode_utf8(s, nxt);
                    if (!is_upper_or_title(c)) break;
                    cur = nxt;
                    upper_end = cur;
                }
                size_t lower_start = cur;
                while (cur < n) {
                    size_t nxt = cur;
                    u32 c = decode_utf8(s, nxt);
                    if (!is_lower_or_other_letter(c)) break;
                    cur = nxt;
                }
                if (cur == letter_start) return 0;
                // Optional trailing contraction (?i:'s|'t|'re|'ve|'m|'ll|'d)
                if (size_t cl = match_contraction(s, cur); cl > 0) cur += cl;
                (void)upper_end;
                (void)lower_start;
                return cur;
            };
            if (is_unicode_letter(cp0)) {
                size_t end_w = try_o200k_word(pos);
                if (end_w > pos) {
                    out.push_back(s.substr(pos, end_w - pos));
                    pos = end_w;
                    continue;
                }
            } else if (cp0 != '\r' && cp0 != '\n' && !is_unicode_number(cp0) && p1 < n) {
                size_t p2 = p1;
                u32 cp1 = decode_utf8(s, p2);
                if (is_unicode_letter(cp1)) {
                    size_t end_w = try_o200k_word(p1);
                    if (end_w > p1) {
                        out.push_back(s.substr(pos, end_w - pos));
                        pos = end_w;
                        continue;
                    }
                }
            }
        } else {
            // Branch 2 (Qwen2/Llama-3): [^\r\n\p{L}\p{N}]?\p{L}+
            if (is_unicode_letter(cp0)) {
                size_t cur = p1;
                while (cur < n) {
                    size_t nxt = cur;
                    u32 c = decode_utf8(s, nxt);
                    if (!is_unicode_letter(c)) break;
                    cur = nxt;
                }
                out.push_back(s.substr(pos, cur - pos));
                pos = cur;
                continue;
            }
            if (cp0 != '\r' && cp0 != '\n' && !is_unicode_number(cp0) && p1 < n) {
                size_t p2 = p1;
                u32 cp1 = decode_utf8(s, p2);
                if (is_unicode_letter(cp1)) {
                    size_t cur = p2;
                    while (cur < n) {
                        size_t nxt = cur;
                        u32 c = decode_utf8(s, nxt);
                        if (!is_unicode_letter(c)) break;
                        cur = nxt;
                    }
                    out.push_back(s.substr(pos, cur - pos));
                    pos = cur;
                    continue;
                }
            }
        }

        // Branch 3: \p{N}{1,max_digit_run_}
        if (is_unicode_number(cp0)) {
            size_t cur = p1;
            u32 cnt = 1;
            while (cur < n && cnt < max_digit_run_) {
                size_t nxt = cur;
                u32 c = decode_utf8(s, nxt);
                if (!is_unicode_number(c)) break;
                cur = nxt;
                ++cnt;
            }
            out.push_back(s.substr(pos, cur - pos));
            pos = cur;
            continue;
        }

        // Branch 4: ?[^\s\p{L}\p{N}]+[\r\n]*
        auto is_punct_or_sym = [](u32 c) noexcept {
            return !is_unicode_ws(c) && !is_unicode_letter(c) && !is_unicode_number(c);
        };
        if (is_punct_or_sym(cp0) || (cp0 == ' ' && p1 < n)) {
            size_t start_p = pos;
            size_t cur = p1;
            bool ok = false;
            if (is_punct_or_sym(cp0)) {
                ok = true;
            } else { // cp0 == ' '
                size_t p2 = p1;
                u32 cp1 = decode_utf8(s, p2);
                if (is_punct_or_sym(cp1)) {
                    ok = true;
                    cur = p2;
                }
            }
            if (ok) {
                while (cur < n) {
                    size_t nxt = cur;
                    u32 c = decode_utf8(s, nxt);
                    if (!is_punct_or_sym(c)) break;
                    cur = nxt;
                }
                while (cur < n && (s[cur] == '\r' || s[cur] == '\n' || (o200k_mode_ && s[cur] == '/'))) ++cur;
                out.push_back(s.substr(start_p, cur - start_p));
                pos = cur;
                continue;
            }
        }

        // Branch 5, 6, 7: whitespace rules: \s*[\r\n]+ | \s+(?\S) | \s+
        if (is_unicode_ws(cp0)) {
            // Scan maximal whitespace run
            size_t ws_end = p1;
            size_t last_nl_end = (cp0 == '\r' || cp0 == '\n') ? p1 : 0;
            while (ws_end < n) {
                size_t nxt = ws_end;
                u32 c = decode_utf8(s, nxt);
                if (!is_unicode_ws(c)) break;
                ws_end = nxt;
                if (c == '\r' || c == '\n') last_nl_end = ws_end;
            }
            if (last_nl_end > 0) {
                // \s*[\r\n]+
                out.push_back(s.substr(pos, last_nl_end - pos));
                pos = last_nl_end;
                continue;
            }
            // No newline in whitespace run.
            // If ws_end < n (followed by non-whitespace), \s+(?\S) stops 1 whitespace char before \S
            // UNLESS run has only 1 char (then falls through to \s+).
            if (ws_end < n) {
                // Find start of last whitespace codepoint before ws_end
                size_t prev_ws = pos;
                size_t scan = pos;
                while (scan < ws_end) {
                    size_t nxt = scan;
                    decode_utf8(s, nxt);
                    if (nxt == ws_end) break;
                    scan = nxt;
                    prev_ws = scan;
                }
                if (prev_ws > pos) {
                    out.push_back(s.substr(pos, prev_ws - pos));
                    pos = prev_ws;
                    continue;
                }
            }
            out.push_back(s.substr(pos, ws_end - pos));
            pos = ws_end;
            continue;
        }

        // Fallback: single codepoint
        out.push_back(s.substr(pos, p1 - pos));
        pos = p1;
    }
}

void BpeTokenizer::bpe_encode_piece(std::string_view piece, std::vector<i32>& out) const {
    if (piece.empty()) return;

    // Fast path for ignore_merges=true (o200k / Inkling-2): direct whole-piece vocab lookup
    if (ignore_merges_) {
        std::string whole;
        whole.reserve(piece.size() * 2);
        for (unsigned char b : piece) whole += byte_to_utf8_[b];
        if (auto it = token_to_id_.find(whole); it != token_to_id_.end()) {
            out.push_back(it->second);
            return;
        }
    }

    // Convert raw bytes into GPT-2 unicode symbol list
    std::vector<std::string> syms;
    syms.reserve(piece.size());
    for (unsigned char b : piece) {
        syms.push_back(byte_to_utf8_[b]);
    }

    if (syms.size() == 1) {
        auto it = token_to_id_.find(syms[0]);
        if (it != token_to_id_.end()) out.push_back(it->second);
        return;
    }

    // Fast path: check if entire piece is already in vocab?
    // NOTE: BPE is greedy by merge rank, NOT longest vocab token Always run BPE merges
    std::string key;
    while (syms.size() >= 2) {
        u32 best_rank = UINT32_MAX;
        size_t best_idx = SIZE_MAX;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            key.clear();
            key.append(syms[i]);
            key.push_back('\x01');
            key.append(syms[i + 1]);
            auto it = merge_ranks_.find(key);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_idx == SIZE_MAX) break;

        // Merge all non-overlapping occurrences of the winning pair (standard BPE)
        std::string left = syms[best_idx];
        std::string right = syms[best_idx + 1];
        std::string merged = left + right;
        std::vector<std::string> next;
        next.reserve(syms.size());
        size_t i = 0;
        while (i < syms.size()) {
            if (i + 1 < syms.size() && syms[i] == left && syms[i + 1] == right) {
                next.push_back(merged);
                i += 2;
            } else {
                next.push_back(std::move(syms[i]));
                ++i;
            }
        }
        syms.swap(next);
    }

    for (const auto& s : syms) {
        auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
        }
    }
}

std::vector<i32> BpeTokenizer::encode(std::string_view text, bool add_bos) const {
    std::vector<i32> ids;
    ids.reserve(text.size() / 3 + 8);
    if ((add_bos || add_bos_by_default_) && bos_id_ >= 0) {
        // Don't double-prepend BOS if text already starts with <|begin_of_text|>
        if (text.rfind("<|begin_of_text|>", 0) != 0) {
            ids.push_back(bos_id_);
        }
    }

    std::vector<std::string_view> pieces;
    auto encode_plain = [&](std::string_view seg) {
        if (seg.empty()) return;
        pretokenize(seg, pieces);
        for (auto p : pieces) bpe_encode_piece(p, ids);
    };

    // Split text on added/special tokens (longest match at earliest occurrence)
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best_pos = std::string_view::npos;
        const AddedToken* best_tok = nullptr;

        for (const auto& at : added_tokens_) {
            size_t found = text.find(at.content, pos);
            if (found != std::string_view::npos) {
                if (found < best_pos ||
                    (found == best_pos && best_tok && at.content.size() > best_tok->content.size())) {
                    best_pos = found;
                    best_tok = &at;
                }
            }
        }

        if (!best_tok) {
            encode_plain(text.substr(pos));
            break;
        }
        if (best_pos > pos) {
            encode_plain(text.substr(pos, best_pos - pos));
        }
        ids.push_back(best_tok->id);
        pos = best_pos + best_tok->content.size();
    }

    return ids;
}

std::string BpeTokenizer::decode_token(i32 id, bool skip_special_tokens) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return "";
    if (skip_special_tokens && is_special_id_[static_cast<size_t>(id)]) return "";
    const auto& tok = id_to_token_[static_cast<size_t>(id)];
    if (is_special_id_[static_cast<size_t>(id)]) return tok;

    // Reverse GPT-2 byte-level unicode encoding to raw bytes
    std::string bytes;
    bytes.reserve(tok.size());
    size_t pos = 0;
    while (pos < tok.size()) {
        u32 cp = decode_utf8(tok, pos);
        auto it = cp_to_byte_.find(cp);
        if (it != cp_to_byte_.end()) {
            bytes.push_back(static_cast<char>(it->second));
        } else {
            encode_utf8(cp, bytes);
        }
    }
    return bytes;
}

std::string BpeTokenizer::decode(std::span<const i32> ids, bool skip_special_tokens) const {
    std::string out;
    out.reserve(ids.size() * 4);
    for (i32 id : ids) {
        out += decode_token(id, skip_special_tokens);
    }
    return out;
}

std::string BpeTokenizer::apply_chat_template(std::string_view system_prompt,
                                              std::string_view user_prompt,
                                              bool is_llama3) const {
    std::string out;
    if (is_llama3) {
        out += "<|begin_of_text|>";
        if (!system_prompt.empty()) {
            out += "<|start_header_id|>system<|end_header_id|>\n\n";
            out += system_prompt;
            out += "<|eot_id|>";
        }
        out += "<|start_header_id|>user<|end_header_id|>\n\n";
        out += user_prompt;
        out += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    } else {
        // Qwen2.5 / Qwen3 ChatML
        if (!system_prompt.empty()) {
            out += "<|im_start|>system\n";
            out += system_prompt;
            out += "<|im_end|>\n";
        }
        out += "<|im_start|>user\n";
        out += user_prompt;
        out += "<|im_end|>\n<|im_start|>assistant\n";
    }
    return out;
}

void BpeTokenizer::serialize_binary(std::vector<u8>& out) const {
    out.clear();
    auto put_u32 = [&](u32 v) {
        size_t o = out.size();
        out.resize(o + 4);
        std::memcpy(out.data() + o, &v, 4);
    };
    auto put_str = [&](const std::string& s) {
        put_u32(static_cast<u32>(s.size()));
        out.insert(out.end(), s.begin(), s.end());
    };

    // Magic "FLNTTOK1"
    const char magic[8] = {'F','L','N','T','T','O','K','1'};
    out.insert(out.end(), magic, magic + 8);
    put_u32(max_digit_run_);
    put_u32(static_cast<u32>(bos_id_));
    u32 mode_flags = (add_bos_by_default_ ? 1u : 0u) | (o200k_mode_ ? 2u : 0u) | (ignore_merges_ ? 4u : 0u);
    put_u32(mode_flags);

    put_u32(static_cast<u32>(id_to_token_.size()));
    for (size_t i = 0; i < id_to_token_.size(); ++i) {
        out.push_back(is_special_id_[i]);
        put_str(id_to_token_[i]);
    }

    put_u32(static_cast<u32>(added_tokens_.size()));
    for (const auto& at : added_tokens_) {
        put_u32(static_cast<u32>(at.id));
        out.push_back(static_cast<u8>((at.special ? 1 : 0) | (at.lstrip ? 2 : 0) | (at.rstrip ? 4 : 0)));
        put_str(at.content);
    }

    put_u32(static_cast<u32>(merge_ranks_.size()));
    for (const auto& [k, r] : merge_ranks_) {
        put_u32(r);
        put_str(k);
    }
}

bool BpeTokenizer::load_binary(std::span<const u8> data, std::string& err) {
    size_t pos = 0;
    auto get_u32 = [&](u32& v) -> bool {
        if (pos + 4 > data.size()) return false;
        std::memcpy(&v, data.data() + pos, 4);
        pos += 4;
        return true;
    };
    auto get_str = [&](std::string& s) -> bool {
        u32 len = 0;
        if (!get_u32(len) || pos + len > data.size()) return false;
        s.assign(reinterpret_cast<const char*>(data.data() + pos), len);
        pos += len;
        return true;
    };

    if (data.size() < 20 || std::memcmp(data.data(), "FLNTTOK1", 8) != 0) {
        err = "Invalid binary tokenizer magic";
        return false;
    }
    pos = 8;
    u32 bos_u = 0, add_bos_u = 0, vocab_n = 0, added_n = 0, merges_n = 0;
    if (!get_u32(max_digit_run_) || !get_u32(bos_u) || !get_u32(add_bos_u) || !get_u32(vocab_n)) {
        err = "Truncated binary tokenizer header";
        return false;
    }
    bos_id_ = static_cast<i32>(bos_u);
    add_bos_by_default_ = (add_bos_u & 1u) != 0;
    o200k_mode_         = (add_bos_u & 2u) != 0;
    ignore_merges_      = (add_bos_u & 4u) != 0;

    id_to_token_.resize(vocab_n);
    is_special_id_.resize(vocab_n);
    token_to_id_.clear();
    token_to_id_.reserve(vocab_n + 64);
    for (u32 i = 0; i < vocab_n; ++i) {
        if (pos >= data.size()) { err = "Truncated vocab"; return false; }
        is_special_id_[i] = data[pos++];
        if (!get_str(id_to_token_[i])) { err = "Truncated vocab string"; return false; }
        if (!id_to_token_[i].empty()) token_to_id_.emplace(id_to_token_[i], static_cast<i32>(i));
    }

    if (!get_u32(added_n)) { err = "Truncated added_tokens"; return false; }
    added_tokens_.resize(added_n);
    for (u32 i = 0; i < added_n; ++i) {
        u32 id_u = 0;
        if (!get_u32(id_u) || pos >= data.size()) { err = "Truncated added token"; return false; }
        u8 flags = data[pos++];
        added_tokens_[i].id = static_cast<i32>(id_u);
        added_tokens_[i].special = (flags & 1) != 0;
        added_tokens_[i].lstrip  = (flags & 2) != 0;
        added_tokens_[i].rstrip  = (flags & 4) != 0;
        if (!get_str(added_tokens_[i].content)) { err = "Truncated added token string"; return false; }
    }

    if (!get_u32(merges_n)) { err = "Truncated merges"; return false; }
    merge_ranks_.clear();
    merge_ranks_.reserve(merges_n);
    std::string k;
    for (u32 i = 0; i < merges_n; ++i) {
        u32 r = 0;
        if (!get_u32(r) || !get_str(k)) { err = "Truncated merge entry"; return false; }
        merge_ranks_.emplace(std::move(k), r);
    }
    return true;
}

} // namespace flint
