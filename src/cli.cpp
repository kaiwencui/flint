#include "cli.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

namespace flint {

void print_version(std::ostream& out) {
#if defined(FLINT_CUDA) && FLINT_CUDA == 1
    out << "flint 2.2.0 (C++23 + CUDA)\n";
#else
    out << "flint 2.2.0 (C++23, CPU-reference)\n";
#endif
}

void print_help(std::ostream& out) {
    out <<
R"(flint 2.2.0 — Zero-dependency single-user (batch=1) C++23 + CUDA LLM engine for consumer NVIDIA GPUs

USAGE:
    flint run <model_dir_or_pack> -p "prompt" [OPTIONS]
    flint pack <hf_model_dir> -o <out.pack> --quant bf16|w4a16|fp8|nvfp4
    flint serve <model_dir_or_pack> [--port 8080] [--unix /tmp/flint.sock] [OPTIONS]
    flint tokenize <model_dir> -p "text"
    flint eval-logits <model_dir_or_pack> -p "prompt" -o <logits.f32>

OPTIONS:
    -p, --prompt <TEXT>         Prompt text
    -f, --prompt-file <PATH>    Read prompt from file
    -o, --output <PATH>         Output file (for pack / eval-logits)
        --tokenizer <DIR>       Tokenizer directory (when loading standalone .pack)
        --max-tokens <N>        Maximum tokens to generate (default: 256)
        --max-seq-len <N>       Maximum KV context length (default: 8192)
        --temp <FLOAT>          Sampling temperature (0 = greedy, default: 0)
        --top-p <FLOAT>         Nucleus top-p sampling (default: 1.0)
        --seed <N>              RNG seed (default: 42)
        --greedy                Force greedy decoding (temp = 0)
        --ignore-eos            Generate exact max-tokens ignoring EOS
        --chat                  Wrap prompt with model chat template
        --system <TEXT>         System prompt when --chat is set
        --quant <MODE>          bf16 | w4a16 | fp8 | nvfp4 (default: bf16)
        --kv-int8               Per-token/per-head INT8 KV cache (2x smaller KV; recommended for Inkling-2-Mini <= 256k)
        --kv-wht4               128-pt Walsh-Hadamard 4-bit KV (Dense: K8V4; Inkling-2: K4V4, 1M ctx in 24 GB)
        --kv-wht84              Inkling-2 WHT-K8V4 global KV (INT8 K + 128-pt Hadamard 4-bit V -> 1M ctx in 32 GB)
        --save-kv <PATH>        Save post-prefill compressed KV cache snapshot to disk
        --load-kv <PATH>        Restore prefilled KV cache snapshot from disk and append prompt
        --spec <MODE>           none | ngram | mtp | draft:<path> (default: none)
        --no-graph              Disable full-step CUDA Graph capture
        --cpu                   Force CPU reference decoder
        --port <PORT>           HTTP server port (default: 8080)
        --host <ADDR>           HTTP bind address (default: 127.0.0.1)
        --unix <PATH>           AF_UNIX RPC socket path (default: /tmp/flint.sock)
        --stats                 Print latency, tok/s, HBM GB/s, VRAM & RSS to stderr
        --json                  Output structured JSON result
    -V, --version               Print version
        --help                  Print help

EXPERIMENTAL / LABS FLAGS (benchmark ablations):
        --megakernel            Single-launch persistent cooperative megakernel (dense W4A16)
        --spec loop1            1-pass looped-core self-speculation (Inkling-2-Mini)
        --kv-int4-raw           Unrotated INT4 KV negative control
)";
}

ParsedCli parse_cli(int argc, char** argv) {
    ParsedCli res;
    if (argc <= 1) {
        res.cmd = CliCmd::Help;
        return res;
    }

    std::string_view sub = argv[1];
    int i = 2;
    if (sub == "run") res.cmd = CliCmd::Run;
    else if (sub == "pack") res.cmd = CliCmd::Pack;
    else if (sub == "serve") res.cmd = CliCmd::Serve;
    else if (sub == "tokenize") res.cmd = CliCmd::Tokenize;
    else if (sub == "eval-logits") res.cmd = CliCmd::EvalLogits;
    else if (sub == "--help" || sub == "-h") { res.cmd = CliCmd::Help; return res; }
    else if (sub == "--version" || sub == "-V") { res.cmd = CliCmd::Version; return res; }
    else {
        res.error = "unknown subcommand: " + std::string(sub);
        return res;
    }

    auto need = [&](const char* opt) -> const char* {
        if (i + 1 >= argc) {
            res.error = std::string("missing value for ") + opt;
            return nullptr;
        }
        return argv[++i];
    };

    std::vector<std::string> pos;
    for (; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--help" || a == "-h") { res.cmd = CliCmd::Help; return res; }
        if (a == "--version" || a == "-V") { res.cmd = CliCmd::Version; return res; }
        if (a == "-p" || a == "--prompt") { const char* v = need(argv[i]); if (!v) return res; res.prompt = v; continue; }
        if (a == "-f" || a == "--prompt-file") { const char* v = need(argv[i]); if (!v) return res; res.prompt_file = v; continue; }
        if (a == "-o" || a == "--output") { const char* v = need(argv[i]); if (!v) return res; res.out_path = v; continue; }
        if (a == "--tokenizer") { const char* v = need(argv[i]); if (!v) return res; res.engine_opts.tokenizer_dir = v; continue; }
        if (a == "--max-tokens") { const char* v = need(argv[i]); if (!v) return res; res.gen_opts.max_tokens = static_cast<u32>(std::strtoul(v, nullptr, 10)); continue; }
        if (a == "--max-seq-len") { const char* v = need(argv[i]); if (!v) return res; res.engine_opts.max_seq_len = static_cast<u32>(std::strtoul(v, nullptr, 10)); continue; }
        if (a == "--temp") { const char* v = need(argv[i]); if (!v) return res; res.gen_opts.temperature = std::strtof(v, nullptr); continue; }
        if (a == "--top-p") { const char* v = need(argv[i]); if (!v) return res; res.gen_opts.top_p = std::strtof(v, nullptr); continue; }
        if (a == "--seed") { const char* v = need(argv[i]); if (!v) return res; res.gen_opts.seed = std::strtoull(v, nullptr, 10); continue; }
        if (a == "--greedy") { res.gen_opts.temperature = 0.0f; continue; }
        if (a == "--ignore-eos") { res.gen_opts.ignore_eos = true; continue; }
        if (a == "--chat") { res.gen_opts.use_chat_template = true; continue; }
        if (a == "--system") { const char* v = need(argv[i]); if (!v) return res; res.gen_opts.system_prompt = v; continue; }
        if (a == "--no-graph") { res.engine_opts.use_cuda_graph = false; continue; }
        if (a == "--megakernel") { res.engine_opts.use_megakernel = true; continue; }
        if (a == "--kv-int8" || a == "--kv-fp8") { res.engine_opts.kv_fp8 = true; res.engine_opts.kv_mode = KvQuantMode::INT8; continue; }
        if (a == "--kv-wht4") { res.engine_opts.kv_mode = KvQuantMode::WHT4; continue; }
        if (a == "--kv-wht84") { res.engine_opts.kv_mode = KvQuantMode::WHT84; continue; }
        if (a == "--kv-int4-raw") { res.engine_opts.kv_mode = KvQuantMode::INT4RAW; continue; }
        if (a == "--save-kv") { const char* v = need(argv[i]); if (!v) return res; res.gen_opts.save_kv_path = v; continue; }
        if (a == "--load-kv") { const char* v = need(argv[i]); if (!v) return res; res.gen_opts.load_kv_path = v; continue; }
        if (a == "--cpu") { res.engine_opts.force_cpu = true; continue; }
        if (a == "--stats") { res.show_stats = true; continue; }
        if (a == "--json") { res.json_output = true; res.stream_output = false; continue; }
        if (a == "--port") { const char* v = need(argv[i]); if (!v) return res; res.server_opts.port = static_cast<u16>(std::strtoul(v, nullptr, 10)); continue; }
        if (a == "--host") { const char* v = need(argv[i]); if (!v) return res; res.server_opts.host = v; continue; }
        if (a == "--unix") { const char* v = need(argv[i]); if (!v) return res; res.server_opts.unix_socket = v; continue; }
        if (a == "--quant") {
            const char* v = need(argv[i]);
            if (!v) return res;
            std::string_view qv(v);
            if (qv == "bf16") res.engine_opts.quant = QuantMode::BF16;
            else if (qv == "w4a16") res.engine_opts.quant = QuantMode::W4A16;
            else if (qv == "fp8") res.engine_opts.quant = QuantMode::FP8;
            else if (qv == "nvfp4") res.engine_opts.quant = QuantMode::NVFP4;
            else { res.error = "unknown --quant mode: " + std::string(qv); return res; }
            continue;
        }
        if (a == "--spec") {
            const char* v = need(argv[i]);
            if (!v) return res;
            std::string_view sv(v);
            if (sv == "none") res.engine_opts.spec_mode = SpecMode::None;
            else if (sv == "ngram") res.engine_opts.spec_mode = SpecMode::Ngram;
            else if (sv == "mtp") res.engine_opts.spec_mode = SpecMode::Mtp;
            else if (sv == "loop1") res.engine_opts.spec_mode = SpecMode::Loop1;
            else if (sv.rfind("draft:", 0) == 0) {
                res.engine_opts.spec_mode = SpecMode::Draft;
                res.engine_opts.draft_model_path = std::string(sv.substr(6));
            } else {
                res.error = "unknown --spec mode: " + std::string(sv);
                return res;
            }
            continue;
        }
        if (!a.empty() && a[0] == '-') {
            res.error = "unknown option: " + std::string(a);
            return res;
        }
        pos.emplace_back(a);
    }

    if (pos.empty()) {
        res.error = "missing <model_dir_or_pack> argument";
        return res;
    }
    res.engine_opts.model_path = pos[0];
    res.server_opts.model_name = res.engine_opts.model_path.filename().string();

    if (!res.prompt_file.empty()) {
        std::ifstream in(res.prompt_file, std::ios::binary);
        if (!in) {
            res.error = "cannot open prompt file: " + res.prompt_file.string();
            return res;
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        res.prompt = ss.str();
    }
    return res;
}

} // namespace flint
