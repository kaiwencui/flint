#include "common.hpp"
#include "engine.hpp"
#include "json.hpp"
#include "kv_cache.hpp"
#include "speculative.hpp"
#include "tokenizer.hpp"
#include "weights.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace flint;

namespace {

int g_pass = 0;
int g_fail = 0;

#define EXPECT(cond)                                                                           \
    do {                                                                                       \
        if (!(cond)) {                                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);               \
            ++g_fail;                                                                          \
            return;                                                                            \
        }                                                                                      \
    } while (0)

void test_json_and_dtypes() {
    std::string err;
    JsonValue v;
    EXPECT(parse_json(R"({"a":42,"b":"hello\n\u4e2d\u6587\ud83d\ude80","c":[true,null,3.5]})", v, err));
    EXPECT(v.get_int("a") == 42);
    EXPECT(v.get_string("b") == "hello\n中文🚀");
    EXPECT(v.get("c")->as_array().size() == 3);

    // BF16 & FP8 roundtrips
    f32 vals[] = {0.0f, 1.0f, -1.0f, 3.140625f, -0.0625f, 128.0f};
    for (f32 f : vals) {
        EXPECT(std::fabs(bf16_to_f32(f32_to_bf16(f)) - f) < 0.02f);
        EXPECT(std::fabs(fp8_e4m3_to_f32(f32_to_fp8_e4m3(f)) - f) < 0.25f);
    }

    // Asymmetric W4A16 group-128 quantization + Marlin half2-interleaved nibble order roundtrip
    std::vector<f32> w(2 * 256);
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = std::sin(static_cast<f32>(i) * 0.1f) * 0.5f + 0.15f; // deliberately asymmetric!
    }
    std::vector<u32> qw;
    std::vector<u32> sc;
    quantize_w4a16_g128(w.data(), 2, 256, qw, sc);
    EXPECT(qw.size() == 2 * 32);
    EXPECT(sc.size() == 2 * 2);
    f32 max_err = 0.0f;
    for (u32 r = 0; r < 2; ++r) {
        for (u32 c = 0; c < 256; ++c) {
            u32 word = qw[r * 32 + c / 8];
            u32 k = c % 8;
            // Marlin nibble map: even k -> bits (k/2)*4; odd k -> bits 16 + (k/2)*4
            u32 shift = ((k & 1u) ? 16u : 0u) + (k >> 1) * 4u;
            u32 q = (word >> shift) & 0xFu;
            u32 sm = sc[r * 2 + c / 128];
            f32 scale = bf16_to_f32(static_cast<u16>(sm & 0xFFFFu));
            f32 wmin  = bf16_to_f32(static_cast<u16>(sm >> 16));
            f32 recon = static_cast<f32>(q) * scale + wmin;
            max_err = std::max(max_err, std::fabs(recon - w[r * 256 + c]));
        }
    }
    EXPECT(max_err < 0.04f);
    ++g_pass;
}

void test_binary_tokenizer_roundtrip() {
    // Test binary tokenizer serialization/deserialization on local Qwen2.5-0.5B if present,
    // plus synthetic empty-safe error check
    BpeTokenizer t_bad;
    std::string err;
    std::vector<u8> junk = {1, 2, 3, 4};
    EXPECT(!t_bad.load_binary(junk, err));

    std::filesystem::path local_m = "/var/tmp/flint-models/Qwen2.5-0.5B-Instruct";
    std::error_code ec;
    if (std::filesystem::exists(local_m / "tokenizer.json", ec)) {
        BpeTokenizer t1, t2;
        EXPECT(t1.load_from_dir(local_m, err));
        std::vector<u8> blob;
        t1.serialize_binary(blob);
        EXPECT(blob.size() > 100000);
        EXPECT(t2.load_binary(blob, err));
        const char* sample = "<|im_start|>user\nWrite C++23 AVX-512 code: 你好🚀!<|im_end|>\n";
        EXPECT(t1.encode(sample) == t2.encode(sample));
        EXPECT(t1.decode(t1.encode(sample)) == t2.decode(t2.encode(sample)));
    }
    ++g_pass;
}

void test_radix_kv_cache() {
    RadixKvManager kv(8);
    EXPECT(kv.free_blocks() == 8);

    std::vector<i32> p1(40);
    for (int i = 0; i < 40; ++i) p1[i] = 100 + i;

    auto m1 = kv.match_prefix(p1);
    EXPECT(m1.matched_tokens == 0);

    i32 b0 = kv.allocate_block();
    i32 b1 = kv.allocate_block();
    i32 b2 = kv.allocate_block();
    u64 h0 = kv.insert_block(0, std::span<const i32>(p1).subspan(0, 16), b0);
    kv.insert_block(h0, std::span<const i32>(p1).subspan(16, 16), b1);
    kv.release_blocks(std::vector<i32>{b0, b1, b2}, 2);
    EXPECT(kv.cached_blocks() == 2);
    EXPECT(kv.free_blocks() == 6);

    // Second request sharing 32 prefix tokens -> instant Radix KV hit!
    std::vector<i32> p2 = p1;
    p2[35] = 9999;
    auto m2 = kv.match_prefix(p2);
    EXPECT(m2.matched_tokens == 32);
    EXPECT(m2.block_table.size() == 2);
    EXPECT(m2.block_table[0] == b0 && m2.block_table[1] == b1);
    kv.release_blocks(m2.block_table, 2);

    // LRU eviction works when pool fills
    std::vector<i32> allocs;
    for (int i = 0; i < 8; ++i) {
        i32 b = kv.allocate_block();
        EXPECT(b >= 0);
        allocs.push_back(b);
    }
    EXPECT(kv.free_blocks() == 0);
    EXPECT(kv.cached_blocks() == 0);
    ++g_pass;
}

void test_speculative_ngram_and_verify() {
    NgramDrafter d;
    std::vector<i32> hist = {10, 20, 30, 40, 50, 60, 70, 80, 10, 20, 30};
    d.ingest(hist);
    auto drafts = d.draft(4);
    EXPECT(drafts.size() == 4);
    EXPECT(drafts[0] == 40 && drafts[1] == 50 && drafts[2] == 60 && drafts[3] == 70);

    // Full accept -> K + 1 tokens emitted
    auto r1 = verify_speculative_greedy(std::vector<i32>{40, 50, 60},
                                        std::vector<i32>{40, 50, 60, 70});
    EXPECT(r1.accepted_drafts == 3);
    EXPECT(r1.emitted == (std::vector<i32>{40, 50, 60, 70}));

    // Divergence at index 1 -> accepts 40, emits target's 99, drops rest (lossless!)
    auto r2 = verify_speculative_greedy(std::vector<i32>{40, 50, 60},
                                        std::vector<i32>{40, 99, 100, 101});
    EXPECT(r2.accepted_drafts == 1);
    EXPECT(r2.emitted == (std::vector<i32>{40, 99}));
    ++g_pass;
}

} // namespace

int main() {
    std::printf("Running flint unit tests...\n");
    test_json_and_dtypes();
    test_binary_tokenizer_roundtrip();
    test_radix_kv_cache();
    test_speculative_ngram_and_verify();
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
