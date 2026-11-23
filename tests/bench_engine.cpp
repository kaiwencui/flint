#include "common.hpp"
#include "engine.hpp"
#include "kernels/kernels.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace flint;

int main(int argc, char** argv) {
    std::filesystem::path model_path;
    QuantMode quant = QuantMode::BF16;
    u32 seq_len = 4096;
    u32 gen_tokens = 128;
    u32 warmup_runs = 2;
    u32 timed_runs = 7;
    bool decode_step_only = false;
    bool use_megakernel = false;
    bool kv_fp8 = false;
    KvQuantMode kv_mode = KvQuantMode::BF16;

    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--seq-len" && i + 1 < argc) seq_len = static_cast<u32>(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--gen" && i + 1 < argc) gen_tokens = static_cast<u32>(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--warmup" && i + 1 < argc) warmup_runs = static_cast<u32>(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--runs" && i + 1 < argc) timed_runs = static_cast<u32>(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--decode-step") decode_step_only = true;
        else if (a == "--megakernel") use_megakernel = true;
        else if (a == "--kv-int8" || a == "--kv-fp8") { kv_fp8 = true; kv_mode = KvQuantMode::INT8; }
        else if (a == "--kv-wht4") { kv_mode = KvQuantMode::WHT4; }
        else if (a == "--kv-wht84") { kv_mode = KvQuantMode::WHT84; }
        else if (a == "--kv-int4-raw") { kv_mode = KvQuantMode::INT4RAW; }
        else if (a == "--no-warmup") warmup_runs = 0;
        else if (a == "--quant" && i + 1 < argc) {
            std::string_view q = argv[++i];
            if (q == "w4a16") quant = QuantMode::W4A16;
            else if (q == "fp8") quant = QuantMode::FP8;
            else if (q == "nvfp4") quant = QuantMode::NVFP4;
            else quant = QuantMode::BF16;
        }
    }

    if (model_path.empty()) {
        std::printf("Usage: flint-bench --model <path> [--quant bf16|w4a16|fp8] [--kv-fp8] [--seq-len 4096] [--gen 256] [--warmup 2] [--runs 7]\n");
        return 0;
    }

    EngineOptions eopts;
    eopts.model_path = model_path;
    eopts.quant = quant;
    eopts.kv_fp8 = kv_fp8;
    eopts.kv_mode = kv_mode;
    eopts.max_seq_len = (seq_len >= 262144) ? std::max<u32>(seq_len + gen_tokens, 262144) : std::max<u32>(seq_len + gen_tokens + 64, 4096);
    eopts.use_cuda_graph = !decode_step_only;
    eopts.use_megakernel = use_megakernel;

    InferenceEngine eng;
    std::string err;
    if (!eng.load(eopts, err)) {
        std::fprintf(stderr, "Failed to load model: %s\n", err.c_str());
        return 1;
    }

    std::vector<i32> prompt(seq_len, 1000);
    GenerateOptions gopts;
    gopts.max_tokens = decode_step_only ? 2 : gen_tokens;
    gopts.temperature = 0.0f;
    gopts.ignore_eos = true;

    // Warmup runs (discarded from timing; clear KV cache between runs so prefill is cold!)
    for (u32 w = 0; w < (decode_step_only ? 0u : warmup_runs); ++w) {
        eng.clear_kv_cache();
        eng.generate_ids(prompt, gopts);
    }

    u32 n_timed = decode_step_only ? 1u : std::max<u32>(1u, timed_runs);
    std::vector<double> ttfts, tps_list, hbm_list, pct_list, itl50_list, itl99_list;
    GenerateResult last_r;
    for (u32 r_i = 0; r_i < n_timed; ++r_i) {
        eng.clear_kv_cache();
        last_r = eng.generate_ids(prompt, gopts);
        ttfts.push_back(last_r.stats.ttft_ms);
        tps_list.push_back(last_r.stats.decode_tok_s);
        hbm_list.push_back(last_r.stats.achieved_hbm_gbs);
        pct_list.push_back(last_r.stats.pct_hbm_peak);
        itl50_list.push_back(last_r.stats.itl_p50_ms);
        itl99_list.push_back(last_r.stats.itl_p99_ms);
    }

    auto med = [](std::vector<double>& v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    double ttft_p50 = med(ttfts);
    double ttft_p99 = ttfts.back();
    double tps_p50  = med(tps_list);
    double tps_p99  = tps_list.front(); // slowest run
    double hbm_p50  = med(hbm_list);
    double pct_p50  = med(pct_list);
    double itl50    = med(itl50_list);
    double itl99    = med(itl99_list);

    std::printf(
        "model=%s backend=%s quant=%s seq=%u gen=%u runs=%u | cold=%.1fms ttft=%.2fms (p99=%.2fms) "
        "decode=%.1f tok/s (p99=%.1f itl_p50=%.2fms itl_p99=%.2fms) "
        "hbm=%.1f GB/s (%.1f%% peak) active_w=%.3f GB vram=%llu MiB rss=%llu MiB\n",
        model_path.filename().c_str(),
        eng.using_cuda() ? eng.gpu_name().c_str() : "CPU",
        quant_name(eng.quant_mode()),
        seq_len, last_r.stats.generated_tokens, n_timed,
        eng.cold_start_ms(), ttft_p50, ttft_p99,
        tps_p50, tps_p99, itl50, itl99,
        hbm_p50, pct_p50,
        static_cast<double>(eng.active_weight_bytes()) / 1e9,
        static_cast<unsigned long long>(last_r.stats.vram_used_mib),
        static_cast<unsigned long long>(last_r.stats.host_rss_mib));
    return 0;
}
