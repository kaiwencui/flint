#pragma once

#include "common.hpp"

#include <charconv>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace flint {

// Zero-dependency RFC 8259 JSON parser (handles UTF-16 surrogate pairs, escapes, large tokenizer.json)
struct JsonValue {
    using Object = std::unordered_map<std::string, JsonValue>;
    using Array  = std::vector<JsonValue>;
    using ObjectPtr = std::shared_ptr<Object>;
    using ArrayPtr  = std::shared_ptr<Array>;
    using Storage = std::variant<std::nullptr_t, bool, i64, f64, std::string, ArrayPtr, ObjectPtr>;

    Storage val = nullptr;

    bool is_null()   const noexcept { return std::holds_alternative<std::nullptr_t>(val); }
    bool is_bool()   const noexcept { return std::holds_alternative<bool>(val); }
    bool is_int()    const noexcept { return std::holds_alternative<i64>(val); }
    bool is_double() const noexcept { return std::holds_alternative<f64>(val); }
    bool is_number() const noexcept { return is_int() || is_double(); }
    bool is_string() const noexcept { return std::holds_alternative<std::string>(val); }
    bool is_array()  const noexcept { return std::holds_alternative<ArrayPtr>(val); }
    bool is_object() const noexcept { return std::holds_alternative<ObjectPtr>(val); }

    bool as_bool(bool def = false) const noexcept {
        return is_bool() ? std::get<bool>(val) : def;
    }
    i64 as_int(i64 def = 0) const noexcept {
        if (is_int()) return std::get<i64>(val);
        if (is_double()) return static_cast<i64>(std::get<f64>(val));
        return def;
    }
    f64 as_double(f64 def = 0.0) const noexcept {
        if (is_double()) return std::get<f64>(val);
        if (is_int()) return static_cast<f64>(std::get<i64>(val));
        return def;
    }
    const std::string& as_string() const { return std::get<std::string>(val); }
    const Array& as_array() const { return *std::get<ArrayPtr>(val); }
    const Object& as_object() const { return *std::get<ObjectPtr>(val); }

    const JsonValue* get(const std::string& key) const noexcept {
        if (!is_object()) return nullptr;
        const auto& obj = *std::get<ObjectPtr>(val);
        auto it = obj.find(key);
        return it != obj.end() ? &it->second : nullptr;
    }

    i64 get_int(const std::string& key, i64 def = 0) const noexcept {
        auto* p = get(key);
        return p ? p->as_int(def) : def;
    }
    f64 get_double(const std::string& key, f64 def = 0.0) const noexcept {
        auto* p = get(key);
        return p ? p->as_double(def) : def;
    }
    bool get_bool(const std::string& key, bool def = false) const noexcept {
        auto* p = get(key);
        return p ? p->as_bool(def) : def;
    }
    std::string get_string(const std::string& key, std::string def = "") const {
        auto* p = get(key);
        return (p && p->is_string()) ? p->as_string() : def;
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string_view src) : s_(src) {}

    bool parse(JsonValue& out, std::string& err) {
        skip_ws();
        if (!parse_value(out, err)) return false;
        skip_ws();
        if (pos_ != s_.size()) {
            err = "trailing characters after JSON root";
            return false;
        }
        return true;
    }

private:
    void skip_ws() noexcept {
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') ++pos_;
            else break;
        }
    }

    static void append_utf8(std::string& out, u32 cp) {
        if (cp <= 0x7Fu) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FFu) {
            out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else if (cp <= 0xFFFFu) {
            out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else if (cp <= 0x10FFFFu) {
            out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        }
    }

    bool parse_hex4(u32& out) noexcept {
        if (pos_ + 4 > s_.size()) return false;
        u32 v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = s_[pos_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<u32>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<u32>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<u32>(c - 'A' + 10);
            else return false;
        }
        out = v;
        return true;
    }

    bool parse_string(std::string& out, std::string& err) {
        if (pos_ >= s_.size() || s_[pos_] != '"') {
            err = "expected '\"'";
            return false;
        }
        ++pos_;
        out.clear();
        while (pos_ < s_.size()) {
            char c = s_[pos_++];
            if (c == '"') return true;
            if (c == '\\') {
                if (pos_ >= s_.size()) { err = "unterminated escape"; return false; }
                char e = s_[pos_++];
                switch (e) {
                    case '"':  out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/'); break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'u': {
                        u32 cp = 0;
                        if (!parse_hex4(cp)) { err = "invalid \\uXXXX"; return false; }
                        if (cp >= 0xD800u && cp <= 0xDBFFu &&
                            pos_ + 6 <= s_.size() && s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
                            pos_ += 2;
                            u32 lo = 0;
                            if (!parse_hex4(lo)) { err = "invalid surrogate"; return false; }
                            if (lo >= 0xDC00u && lo <= 0xDFFFu) {
                                cp = 0x10000u + (((cp - 0xD800u) << 10) | (lo - 0xDC00u));
                            }
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default:
                        err = "invalid JSON escape";
                        return false;
                }
            } else {
                out.push_back(c);
            }
        }
        err = "unterminated JSON string";
        return false;
    }

    bool parse_number(JsonValue& out, std::string& err) {
        size_t start = pos_;
        if (pos_ < s_.size() && s_[pos_] == '-') ++pos_;
        bool is_float = false;
        while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
        if (pos_ < s_.size() && s_[pos_] == '.') {
            is_float = true;
            ++pos_;
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
        }
        if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
            is_float = true;
            ++pos_;
            if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) ++pos_;
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
        }
        std::string_view tok = s_.substr(start, pos_ - start);
        if (!is_float) {
            i64 iv = 0;
            auto [ptr, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), iv);
            if (ec == std::errc{} && ptr == tok.data() + tok.size()) {
                out.val = iv;
                return true;
            }
        }
        std::string tmp(tok);
        char* endp = nullptr;
        f64 dv = std::strtod(tmp.c_str(), &endp);
        if (endp != tmp.c_str() + tmp.size()) {
            err = "invalid JSON number";
            return false;
        }
        out.val = dv;
        return true;
    }

    bool parse_value(JsonValue& out, std::string& err) {
        skip_ws();
        if (pos_ >= s_.size()) { err = "unexpected EOF"; return false; }
        char c = s_[pos_];
        if (c == '"') {
            std::string str;
            if (!parse_string(str, err)) return false;
            out.val = std::move(str);
            return true;
        }
        if (c == '{') {
            ++pos_;
            auto obj = std::make_shared<JsonValue::Object>();
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == '}') {
                ++pos_;
                out.val = std::move(obj);
                return true;
            }
            while (true) {
                skip_ws();
                std::string key;
                if (!parse_string(key, err)) return false;
                skip_ws();
                if (pos_ >= s_.size() || s_[pos_] != ':') { err = "expected ':'"; return false; }
                ++pos_;
                JsonValue val;
                if (!parse_value(val, err)) return false;
                obj->emplace(std::move(key), std::move(val));
                skip_ws();
                if (pos_ < s_.size() && s_[pos_] == ',') { ++pos_; continue; }
                if (pos_ < s_.size() && s_[pos_] == '}') { ++pos_; break; }
                err = "expected ',' or '}'";
                return false;
            }
            out.val = std::move(obj);
            return true;
        }
        if (c == '[') {
            ++pos_;
            auto arr = std::make_shared<JsonValue::Array>();
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == ']') {
                ++pos_;
                out.val = std::move(arr);
                return true;
            }
            while (true) {
                JsonValue elem;
                if (!parse_value(elem, err)) return false;
                arr->push_back(std::move(elem));
                skip_ws();
                if (pos_ < s_.size() && s_[pos_] == ',') { ++pos_; continue; }
                if (pos_ < s_.size() && s_[pos_] == ']') { ++pos_; break; }
                err = "expected ',' or ']'";
                return false;
            }
            out.val = std::move(arr);
            return true;
        }
        if (c == 't' && s_.substr(pos_, 4) == "true")  { pos_ += 4; out.val = true; return true; }
        if (c == 'f' && s_.substr(pos_, 5) == "false") { pos_ += 5; out.val = false; return true; }
        if (c == 'n' && s_.substr(pos_, 4) == "null")  { pos_ += 4; out.val = nullptr; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number(out, err);
        err = std::string("unexpected char in JSON: ") + c;
        return false;
    }

    std::string_view s_;
    size_t pos_ = 0;
};

inline bool parse_json(std::string_view src, JsonValue& out, std::string& err) {
    JsonParser p(src);
    return p.parse(out, err);
}

} // namespace flint
