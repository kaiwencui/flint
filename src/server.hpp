#pragma once

#include "engine.hpp"

#include <atomic>
#include <filesystem>
#include <string>

namespace flint {

struct ServerOptions {
    std::string host = "127.0.0.1";
    u16 port = 8080;
    std::filesystem::path unix_socket = "/tmp/flint.sock";
    std::string model_name = "flint-model";
};

// Minimal zero-dependency C++23 HTTP/1.1 + SSE OpenAI API server + AF_UNIX binary RPC daemon.
// Endpoints:
// POST /v1/chat/completions (streaming SSE + non-streaming)
// POST /v1/completions (supports FIM prefix/suffix & ignore_eos)
// GET /v1/models
// GET /health
class FlintServer {
public:
    FlintServer(InferenceEngine& engine, ServerOptions opts);
    ~FlintServer();

    int run();
    void stop() noexcept { stop_.store(true, std::memory_order_release); }

private:
    void handle_http_client(int fd);
    void handle_unix_client(int fd);

    InferenceEngine& engine_;
    ServerOptions opts_;
    int http_fd_ = -1;
    int unix_fd_ = -1;
    std::atomic<bool> stop_{false};
};

} // namespace flint
