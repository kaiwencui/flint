#include "server.hpp"
#include "json.hpp"

#include <arpa/inet.h>
#include <csignal>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sstream>
#include <sys/socket.h>
#include <sys/un.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace flint {

namespace {

int open_stream_socket(int domain) {
#if defined(SOCK_CLOEXEC)
    return ::socket(domain, SOCK_STREAM | SOCK_CLOEXEC, 0);
#else
    int fd = ::socket(domain, SOCK_STREAM, 0);
    if (fd >= 0) ::fcntl(fd, F_SETFD, FD_CLOEXEC);
    return fd;
#endif
}

int accept_cloexec(int listen_fd) {
#if defined(__linux__) && defined(SOCK_CLOEXEC)
    return ::accept4(listen_fd, nullptr, nullptr, SOCK_CLOEXEC);
#else
    int cfd = ::accept(listen_fd, nullptr, nullptr);
    if (cfd >= 0) {
        ::fcntl(cfd, F_SETFD, FD_CLOEXEC);
#if defined(SO_NOSIGPIPE)
        int one = 1;
        ::setsockopt(cfd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    }
    return cfd;
#endif
}

std::atomic<FlintServer*> g_srv{nullptr};
void srv_sig_handler(int) {
    if (auto* s = g_srv.load(std::memory_order_relaxed)) s->stop();
}

bool send_all(int fd, const void* buf, size_t len) {
    const auto* p = static_cast<const u8*>(buf);
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

std::string json_escape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

void send_http_json(int fd, int status, const char* reason, const std::string& body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << " " << reason << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    std::string resp = oss.str();
    send_all(fd, resp.data(), resp.size());
}

} // namespace

FlintServer::FlintServer(InferenceEngine& engine, ServerOptions opts)
    : engine_(engine), opts_(std::move(opts)) {}

FlintServer::~FlintServer() {
    stop();
    if (http_fd_ >= 0) ::close(http_fd_);
    if (unix_fd_ >= 0) ::close(unix_fd_);
    std::error_code ec;
    if (!opts_.unix_socket.empty()) std::filesystem::remove(opts_.unix_socket, ec);
}

void FlintServer::handle_http_client(int fd) {
    // Read HTTP request header + body
    std::string raw;
    raw.reserve(4096);
    char buf[4096];
    size_t hdr_end = std::string::npos;
    size_t content_len = 0;

    while (true) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        raw.append(buf, static_cast<size_t>(n));
        if (hdr_end == std::string::npos) {
            hdr_end = raw.find("\r\n\r\n");
            if (hdr_end != std::string::npos) {
                std::string_view hdrs(raw.data(), hdr_end);
                size_t cl = hdrs.find("Content-Length:");
                if (cl == std::string_view::npos) cl = hdrs.find("content-length:");
                if (cl != std::string_view::npos) {
                    content_len = std::strtoull(raw.c_str() + cl + 15, nullptr, 10);
                }
            }
        }
        if (hdr_end != std::string::npos && raw.size() >= hdr_end + 4 + content_len) break;
    }

    if (hdr_end == std::string::npos) {
        ::close(fd);
        return;
    }

    size_t sp1 = raw.find(' ');
    size_t sp2 = (sp1 != std::string::npos) ? raw.find(' ', sp1 + 1) : std::string::npos;
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
        ::close(fd);
        return;
    }
    std::string method = raw.substr(0, sp1);
    std::string path = raw.substr(sp1 + 1, sp2 - sp1 - 1);
    std::string body = raw.substr(hdr_end + 4, content_len);

    if (method == "GET" && path == "/health") {
        send_http_json(fd, 200, "OK", "{\"status\":\"ok\"}");
        ::close(fd);
        return;
    }
    if (method == "GET" && path == "/v1/models") {
        std::string j = "{\"object\":\"list\",\"data\":[{\"id\":\"" +
                        json_escape(opts_.model_name) + "\",\"object\":\"model\"}]}";
        send_http_json(fd, 200, "OK", j);
        ::close(fd);
        return;
    }

    if (method == "POST" && (path == "/v1/completions" || path == "/v1/chat/completions")) {
        JsonValue req;
        std::string jerr;
        if (!parse_json(body, req, jerr) || !req.is_object()) {
            send_http_json(fd, 400, "Bad Request", "{\"error\":\"invalid JSON\"}");
            ::close(fd);
            return;
        }

        GenerateOptions gopts;
        gopts.max_tokens = static_cast<u32>(req.get_int("max_tokens", 256));
        gopts.temperature = static_cast<f32>(req.get_double("temperature", 0.0));
        gopts.top_p = static_cast<f32>(req.get_double("top_p", 1.0));
        gopts.ignore_eos = req.get_bool("ignore_eos", false);
        bool stream = req.get_bool("stream", false);

        std::string prompt;
        if (path == "/v1/completions") {
            prompt = req.get_string("prompt");
            std::string suffix = req.get_string("suffix");
            if (!suffix.empty()) {
                // Qwen2.5-Coder FIM format
                prompt = "<|fim_prefix|>" + prompt + "<|fim_suffix|>" + suffix + "<|fim_middle|>";
            }
        } else {
            // /v1/chat/completions
            std::string sys_msg, user_msg;
            if (const auto* msgs = req.get("messages"); msgs && msgs->is_array()) {
                for (const auto& m : msgs->as_array()) {
                    std::string role = m.get_string("role");
                    std::string content = m.get_string("content");
                    if (role == "system") sys_msg = content;
                    else if (role == "user") user_msg = content;
                }
            }
            prompt = engine_.tokenizer().apply_chat_template(
                sys_msg, user_msg, engine_.config().arch == ModelArch::Llama);
        }

        if (stream) {
            const char* sse_hdr =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Connection: close\r\n\r\n";
            send_all(fd, sse_hdr, std::strlen(sse_hdr));
            bool is_chat = (path == "/v1/chat/completions");
            engine_.generate(prompt, gopts, [&](i32, std::string_view piece) -> bool {
                std::string chunk;
                if (is_chat) {
                    chunk = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" +
                            json_escape(piece) + "\"}}]}\n\n";
                } else {
                    chunk = "data: {\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"" +
                            json_escape(piece) + "\"}]}\n\n";
                }
                return send_all(fd, chunk.data(), chunk.size());
            });
            const char* done = "data: [DONE]\n\n";
            send_all(fd, done, std::strlen(done));
        } else {
            auto r = engine_.generate(prompt, gopts);
            std::ostringstream out;
            if (path == "/v1/chat/completions") {
                out << "{\"object\":\"chat.completion\",\"choices\":[{\"index\":0,"
                    << "\"message\":{\"role\":\"assistant\",\"content\":\"" << json_escape(r.text)
                    << "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":"
                    << r.stats.prompt_tokens << ",\"completion_tokens\":" << r.stats.generated_tokens
                    << ",\"ttft_ms\":" << r.stats.ttft_ms << ",\"decode_tok_s\":" << r.stats.decode_tok_s
                    << "}}";
            } else {
                out << "{\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\""
                    << json_escape(r.text) << "\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":"
                    << r.stats.prompt_tokens << ",\"completion_tokens\":" << r.stats.generated_tokens
                    << ",\"ttft_ms\":" << r.stats.ttft_ms << ",\"decode_tok_s\":" << r.stats.decode_tok_s
                    << "}}";
            }
            send_http_json(fd, 200, "OK", out.str());
        }
        ::close(fd);
        return;
    }

    send_http_json(fd, 404, "Not Found", "{\"error\":\"not found\"}");
    ::close(fd);
}

void FlintServer::handle_unix_client(int fd) {
    // Framed binary RPC over AF_UNIX: [u32 len][json request] -> [u32 len][json response]
    u32 sz = 0;
    if (::recv(fd, &sz, 4, MSG_WAITALL) != 4 || sz > 16u * 1024u * 1024u) {
        ::close(fd);
        return;
    }
    std::string req_str(sz, '\0');
    if (sz > 0 && ::recv(fd, req_str.data(), sz, MSG_WAITALL) != static_cast<ssize_t>(sz)) {
        ::close(fd);
        return;
    }
    JsonValue req;
    std::string jerr;
    std::string resp_str = "{\"error\":\"bad request\"}";
    if (parse_json(req_str, req, jerr) && req.is_object()) {
        GenerateOptions gopts;
        gopts.max_tokens = static_cast<u32>(req.get_int("max_tokens", 256));
        gopts.temperature = static_cast<f32>(req.get_double("temperature", 0.0));
        gopts.ignore_eos = req.get_bool("ignore_eos", false);
        auto r = engine_.generate(req.get_string("prompt"), gopts);
        std::ostringstream oss;
        oss << "{\"text\":\"" << json_escape(r.text)
            << "\",\"prompt_tokens\":" << r.stats.prompt_tokens
            << ",\"prefix_cached_tokens\":" << r.stats.prefix_cached_tokens
            << ",\"generated_tokens\":" << r.stats.generated_tokens
            << ",\"ttft_ms\":" << r.stats.ttft_ms
            << ",\"decode_tok_s\":" << r.stats.decode_tok_s
            << ",\"achieved_hbm_gbs\":" << r.stats.achieved_hbm_gbs
            << ",\"pct_hbm_peak\":" << r.stats.pct_hbm_peak << "}";
        resp_str = oss.str();
    }
    u32 rsz = static_cast<u32>(resp_str.size());
    send_all(fd, &rsz, 4);
    send_all(fd, resp_str.data(), rsz);
    ::close(fd);
}

int FlintServer::run() {
    http_fd_ = open_stream_socket(AF_INET);
    if (http_fd_ < 0) return 1;
    int yes = 1;
    ::setsockopt(http_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(opts_.port);
    ::inet_pton(AF_INET, opts_.host.c_str(), &addr.sin_addr);
    if (::bind(http_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0 ||
        ::listen(http_fd_, 32) != 0) {
        std::fprintf(stderr, "flint serve: failed to bind %s:%u\n", opts_.host.c_str(), opts_.port);
        return 1;
    }

    if (!opts_.unix_socket.empty()) {
        std::error_code ec;
        std::filesystem::remove(opts_.unix_socket, ec);
        unix_fd_ = open_stream_socket(AF_UNIX);
        if (unix_fd_ >= 0) {
            struct sockaddr_un uaddr {};
            uaddr.sun_family = AF_UNIX;
            std::string sp = opts_.unix_socket.string();
            if (sp.size() < sizeof(uaddr.sun_path)) {
                std::memcpy(uaddr.sun_path, sp.c_str(), sp.size() + 1);
                if (::bind(unix_fd_, reinterpret_cast<struct sockaddr*>(&uaddr), sizeof(uaddr)) != 0 ||
                    ::listen(unix_fd_, 32) != 0) {
                    ::close(unix_fd_);
                    unix_fd_ = -1;
                }
            }
        }
    }

    g_srv.store(this, std::memory_order_release);
    std::signal(SIGINT, srv_sig_handler);
    std::signal(SIGTERM, srv_sig_handler);

    std::fprintf(stderr, "flint serve listening on http://%s:%u (unix=%s, quant=%s, %s)\n",
                 opts_.host.c_str(), opts_.port,
                 unix_fd_ >= 0 ? opts_.unix_socket.c_str() : "disabled",
                 quant_name(engine_.quant_mode()),
                 engine_.using_cuda() ? engine_.gpu_name().c_str() : "CPU");

    while (!stop_.load(std::memory_order_acquire)) {
        struct pollfd pfds[2];
        int nfds = 0;
        pfds[nfds++] = {http_fd_, POLLIN, 0};
        if (unix_fd_ >= 0) pfds[nfds++] = {unix_fd_, POLLIN, 0};

        int rc = ::poll(pfds, nfds, 100);
        if (rc <= 0) continue;

        if (pfds[0].revents & POLLIN) {
            int cfd = accept_cloexec(http_fd_);
            if (cfd >= 0) {
                int nodelay = 1;
                ::setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
                handle_http_client(cfd);
            }
        }
        if (nfds > 1 && (pfds[1].revents & POLLIN)) {
            int cfd = accept_cloexec(unix_fd_);
            if (cfd >= 0) handle_unix_client(cfd);
        }
    }
    return 0;
}

} // namespace flint
