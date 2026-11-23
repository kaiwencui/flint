#include "cli.hpp"
#include "engine.hpp"
#include "server.hpp"
#include "weights.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    auto cli = flint::parse_cli(argc, argv);
    if (!cli.error.empty()) {
        std::cerr << "flint: " << cli.error << "\n";
        return 2;
    }

    switch (cli.cmd) {
        case flint::CliCmd::Help:
            flint::print_help(std::cout);
            return 0;
        case flint::CliCmd::Version:
            flint::print_version(std::cout);
            return 0;
        case flint::CliCmd::Tokenize: {
            flint::BpeTokenizer tok;
            std::string err;
            if (!tok.load_from_dir(cli.engine_opts.model_path, err)) {
                std::cerr << "flint tokenize: " << err << "\n";
                return 2;
            }
            auto ids = tok.encode(cli.prompt, false);
            std::cout << "[";
            for (size_t i = 0; i < ids.size(); ++i) {
                if (i > 0) std::cout << ",";
                std::cout << ids[i];
            }
            std::cout << "]\n";
            return 0;
        }
        case flint::CliCmd::Pack: {
            if (cli.out_path.empty()) {
                std::cerr << "flint pack: missing -o <out.pack>\n";
                return 2;
            }
            flint::Stopwatch sw;
            std::string err;
            if (!flint::pack_model(cli.engine_opts.model_path, cli.out_path,
                                   cli.engine_opts.quant, err)) {
                std::cerr << "flint pack: " << err << "\n";
                return 2;
            }
            std::error_code ec;
            auto sz = std::filesystem::file_size(cli.out_path, ec);
            std::printf("Packed %s -> %s (quant=%s, %.2f MiB) in %.2f s\n",
                        cli.engine_opts.model_path.c_str(), cli.out_path.c_str(),
                        flint::quant_name(cli.engine_opts.quant),
                        static_cast<double>(sz) / (1024.0 * 1024.0), sw.elapsed_s());
            return 0;
        }
        case flint::CliCmd::EvalLogits: {
            flint::InferenceEngine eng;
            std::string err;
            if (!eng.load(cli.engine_opts, err)) {
                std::cerr << "flint eval-logits: " << err << "\n";
                return 2;
            }
            auto ids = eng.tokenizer().encode(cli.prompt, false);
            std::vector<flint::f32> logits;
            if (!eng.eval_logits(ids, logits, err)) {
                std::cerr << "flint eval-logits: " << err << "\n";
                return 2;
            }
            if (!cli.out_path.empty()) {
                std::ofstream out(cli.out_path, std::ios::binary | std::ios::trunc);
                out.write(reinterpret_cast<const char*>(logits.data()),
                          static_cast<std::streamsize>(logits.size() * sizeof(flint::f32)));
            }
            flint::i32 top = flint::CpuRefDecoder::argmax(logits);
            std::printf("argmax=%d logit=%.6f vocab=%zu\n", top, logits[top], logits.size());
            return 0;
        }
        case flint::CliCmd::Run: {
            flint::InferenceEngine eng;
            std::string err;
            if (!eng.load(cli.engine_opts, err)) {
                std::cerr << "flint run: " << err << "\n";
                return 2;
            }
            auto r = eng.generate(cli.prompt, cli.gen_opts,
                cli.stream_output
                    ? std::function<bool(flint::i32, std::string_view)>(
                          [](flint::i32, std::string_view piece) {
                              std::cout << piece << std::flush;
                              return true;
                          })
                    : nullptr);

            if (!r.error.empty()) {
                std::cerr << "flint run: " << r.error << "\n";
                return 2;
            }
            if (cli.json_output) {
                std::cout << "{\"output_ids\":[";
                for (size_t i = 0; i < r.output_ids.size(); ++i) {
                    if (i > 0) std::cout << ",";
                    std::cout << r.output_ids[i];
                }
                std::cout << "],\"cold_start_ms\":" << eng.cold_start_ms()
                          << ",\"ttft_ms\":" << r.stats.ttft_ms
                          << ",\"decode_tok_s\":" << r.stats.decode_tok_s
                          << ",\"achieved_hbm_gbs\":" << r.stats.achieved_hbm_gbs
                          << ",\"pct_hbm_peak\":" << r.stats.pct_hbm_peak
                          << ",\"vram_used_mib\":" << r.stats.vram_used_mib
                          << ",\"host_rss_mib\":" << r.stats.host_rss_mib << "}\n";
            } else {
                std::cout << "\n";
            }

            if (cli.show_stats) {
                std::fprintf(stderr,
                    "[flint stats] backend=%s quant=%s cold_start=%.1fms ttft=%.2fms "
                    "prompt=%u (cached=%u) gen=%u decode=%.1f tok/s (itl p50=%.2fms p99=%.2fms) "
                    "hbm=%.1f GB/s (%.1f%% peak) vram=%llu/%llu MiB rss=%llu MiB",
                    eng.using_cuda() ? eng.gpu_name().c_str() : "CPU",
                    flint::quant_name(eng.quant_mode()),
                    eng.cold_start_ms(), r.stats.ttft_ms,
                    r.stats.prompt_tokens, r.stats.prefix_cached_tokens, r.stats.generated_tokens,
                    r.stats.decode_tok_s, r.stats.itl_p50_ms, r.stats.itl_p99_ms,
                    r.stats.achieved_hbm_gbs, r.stats.pct_hbm_peak,
                    static_cast<unsigned long long>(r.stats.vram_used_mib),
                    static_cast<unsigned long long>(r.stats.vram_total_mib),
                    static_cast<unsigned long long>(r.stats.host_rss_mib));
                if (r.stats.spec_drafted > 0) {
                    std::fprintf(stderr, " spec_accept=%.1f%% (%u/%u)",
                                 100.0 * r.stats.spec_accept_rate,
                                 r.stats.spec_accepted, r.stats.spec_drafted);
                }
                std::fprintf(stderr, "\n");
            }
            return 0;
        }
        case flint::CliCmd::Serve: {
            flint::InferenceEngine eng;
            std::string err;
            if (!eng.load(cli.engine_opts, err)) {
                std::cerr << "flint serve: " << err << "\n";
                return 2;
            }
            flint::FlintServer srv(eng, cli.server_opts);
            return srv.run();
        }
    }
    return 0;
}
