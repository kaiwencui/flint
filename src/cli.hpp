#pragma once

#include "engine.hpp"
#include "server.hpp"

#include <string>

namespace flint {

enum class CliCmd : u8 {
    Run,
    Pack,
    Serve,
    Tokenize,    // Helper for Gate 1 parity test (encodes stdin/prompt -> JSON token IDs)
    EvalLogits,  // Helper for Gate 2 parity test (dumps top logits & raw binary logits)
    Help,
    Version
};

struct ParsedCli {
    CliCmd cmd = CliCmd::Help;
    EngineOptions engine_opts;
    GenerateOptions gen_opts;
    ServerOptions server_opts;
    std::string prompt;
    std::filesystem::path prompt_file;
    std::filesystem::path out_path;
    bool show_stats = false;
    bool json_output = false;
    bool stream_output = true;
    std::string error;
};

ParsedCli parse_cli(int argc, char** argv);
void print_help(std::ostream& out);
void print_version(std::ostream& out);

} // namespace flint
