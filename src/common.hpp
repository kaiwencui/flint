#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(FLINT_CUDA) && FLINT_CUDA == 1
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
  #include <cuda_fp8.h>
  #include <cuda_runtime.h>
#endif

namespace flint {

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;

enum class QuantMode : u8 {
    BF16  = 0,
    W4A16 = 1,
    FP8   = 2,
    NVFP4 = 3
};

inline const char* quant_name(QuantMode q) noexcept {
    switch (q) {
        case QuantMode::BF16:  return "bf16";
        case QuantMode::W4A16: return "w4a16";
        case QuantMode::FP8:   return "fp8";
        case QuantMode::NVFP4: return "nvfp4";
    }
    return "unknown";
}

enum class KvQuantMode : u8 {
    BF16    = 0,
    INT8    = 1,
    WHT4    = 2, // Dense: WHT-K8V4 | Inkling2: WHT-K4V4 (128-pt Hadamard 4-bit K + 4-bit V + BF16 scales -> 8.25 GiB @ 1M)
    INT4RAW = 3, // Unrotated asymmetric INT4 KV (ablation baseline to isolate WHT gain)
    WHT84   = 4  // Inkling2 32 GB 1M mode: WHT-K8V4 (INT8 K + 128-pt Hadamard 4-bit V + BF16 scales -> 12.25 GiB @ 1M)
};

inline const char* kv_quant_name(KvQuantMode m) noexcept {
    switch (m) {
        case KvQuantMode::BF16:    return "bf16";
        case KvQuantMode::INT8:    return "int8";
        case KvQuantMode::WHT4:    return "wht4";
        case KvQuantMode::INT4RAW: return "int4raw";
        case KvQuantMode::WHT84:   return "wht84";
    }
    return "bf16";
}

inline constexpr u64 align_up(u64 v, u64 a) noexcept {
    return (v + a - 1) & ~(a - 1);
}

// Host BF16 / FP16 / FP8_E4M3 bit conversions (exact IEEE-754 RNE)
inline f32 bf16_to_f32(u16 b) noexcept {
    u32 u = static_cast<u32>(b) << 16;
    f32 f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline u16 f32_to_bf16(f32 f) noexcept {
    u32 u;
    std::memcpy(&u, &f, sizeof(u));
    // Round-to-nearest-even (handles NaN cleanly)
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
        return static_cast<u16>((u >> 16) | 0x0040u);
    }
    u32 rounding_bias = 0x00007FFFu + ((u >> 16) & 1u);
    return static_cast<u16>((u + rounding_bias) >> 16);
}

inline f32 fp16_to_f32(u16 h) noexcept {
    u32 sign = static_cast<u32>(h & 0x8000u) << 16;
    u32 exp  = (h >> 10) & 0x1Fu;
    u32 mant = h & 0x03FFu;
    u32 out = 0;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            // Subnormal
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFu;
            out = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    f32 f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

// IEEE / OCP FP8 E4M3FN conversion (bias = 7, max = 448.0, no Inf, 0x7F/0xFF = NaN)
inline u8 f32_to_fp8_e4m3(f32 f) noexcept {
    if (std::isnan(f)) return 0x7Fu;
    u8 sign = (std::signbit(f) ? 0x80u : 0x00u);
    f32 af = std::fabs(f);
    if (af >= 448.0f) return static_cast<u8>(sign | 0x7Eu);
    if (af == 0.0f) return sign;

    u32 bits;
    std::memcpy(&bits, &af, sizeof(bits));
    int exp = static_cast<int>((bits >> 23) & 0xFFu) - 127;
    u32 mant = bits & 0x7FFFFFu;

    int e4 = exp + 7;
    if (e4 <= 0) {
        // Subnormal in E4M3: value = m / 2^9
        f32 scaled = af * 512.0f;
        int m = static_cast<int>(std::nearbyint(scaled));
        if (m <= 0) return sign;
        if (m >= 8) return static_cast<u8>(sign | 0x08u); // smallest normal 2^-6
        return static_cast<u8>(sign | static_cast<u8>(m));
    }
    // Round 23-bit mantissa to 3 bits (RNE)
    u32 m3 = mant >> 20;
    u32 rem = mant & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (m3 & 1u))) {
        m3++;
        if (m3 == 8) {
            m3 = 0;
            e4++;
        }
    }
    if (e4 > 15 || (e4 == 15 && m3 >= 7)) {
        return static_cast<u8>(sign | 0x7Eu); // clamp to 448.0
    }
    return static_cast<u8>(sign | (static_cast<u8>(e4) << 3) | static_cast<u8>(m3));
}

inline f32 fp8_e4m3_to_f32(u8 v) noexcept {
    f32 sign = (v & 0x80u) ? -1.0f : 1.0f;
    u32 e = (v >> 3) & 0xFu;
    u32 m = v & 0x7u;
    if (e == 0) {
        return sign * (static_cast<f32>(m) / 512.0f);
    }
    if (e == 15 && m == 7) {
        return std::numeric_limits<f32>::quiet_NaN();
    }
    f32 val = (1.0f + static_cast<f32>(m) / 8.0f) * std::ldexp(1.0f, static_cast<int>(e) - 7);
    return sign * val;
}

// Fast 64-bit hash for Radix KV prefix blocks
inline u64 hash_bytes64(const void* data, size_t len, u64 seed = 0x9E3779B97F4A7C15ULL) noexcept {
    const auto* p = static_cast<const u8*>(data);
    u64 h = seed ^ (len * 0xBF58476D1CE4E5B9ULL);
    size_t i = 0;
    while (i + 8 <= len) {
        u64 k;
        std::memcpy(&k, p + i, 8);
        k *= 0x94D049BB133111EBULL;
        k ^= k >> 31;
        k *= 0xBF58476D1CE4E5B9ULL;
        h ^= k;
        h = (h << 27) | (h >> 37);
        h = h * 5 + 0x52DCE729ULL;
        i += 8;
    }
    u64 tail = 0;
    size_t rem = len - i;
    if (rem > 0) {
        std::memcpy(&tail, p + i, rem);
        tail *= 0x94D049BB133111EBULL;
        tail ^= tail >> 31;
        h ^= tail;
    }
    h ^= h >> 30;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 27;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 31;
    return h;
}

struct Stopwatch {
    using Clock = std::chrono::steady_clock;
    Clock::time_point tp = Clock::now();
    void reset() noexcept { tp = Clock::now(); }
    double elapsed_us() const noexcept {
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - tp).count()) / 1e3;
    }
    double elapsed_ms() const noexcept { return elapsed_us() / 1e3; }
    double elapsed_s() const noexcept { return elapsed_us() / 1e6; }
};

class MmapFile {
public:
    MmapFile() = default;
    ~MmapFile() { close(); }
    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;
    MmapFile(MmapFile&& o) noexcept : data_(o.data_), size_(o.size_) {
        o.data_ = nullptr;
        o.size_ = 0;
    }
    MmapFile& operator=(MmapFile&& o) noexcept {
        if (this != &o) {
            close();
            data_ = o.data_;
            size_ = o.size_;
            o.data_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    bool open(const std::filesystem::path& path, bool sequential = true) {
        close();
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) return false;
        struct stat st {};
        if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
            ::close(fd);
            return false;
        }
        size_ = static_cast<size_t>(st.st_size);
        if (size_ == 0) {
            ::close(fd);
            return true;
        }
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) {
            size_ = 0;
            return false;
        }
        data_ = static_cast<const u8*>(p);
        ::madvise(p, size_, sequential ? MADV_SEQUENTIAL : MADV_RANDOM);
        return true;
    }

    void close() noexcept {
        if (data_ && size_ > 0) ::munmap(const_cast<u8*>(data_), size_);
        data_ = nullptr;
        size_ = 0;
    }

    const u8* data() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }
    bool valid() const noexcept { return data_ != nullptr; }

private:
    const u8* data_ = nullptr;
    size_t size_ = 0;
};

inline u64 current_rss_kib() {
    FILE* f = std::fopen("/proc/self/smaps_rollup", "r");
    if (!f) return 0;
    char line[256];
    u64 kb = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "Rss:", 4) == 0) {
            unsigned long long v = 0;
            if (std::sscanf(line + 4, "%llu", &v) == 1) kb = v;
            break;
        }
    }
    std::fclose(f);
    return kb;
}

inline u64 current_private_dirty_kib() {
    FILE* f = std::fopen("/proc/self/smaps_rollup", "r");
    if (!f) return 0;
    char line[256];
    u64 kb = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "Private_Dirty:", 14) == 0) {
            unsigned long long v = 0;
            if (std::sscanf(line + 14, "%llu", &v) == 1) kb = v;
            break;
        }
    }
    std::fclose(f);
    return kb;
}

#if defined(FLINT_CUDA) && FLINT_CUDA == 1
#define FLINT_CUDA_CHECK(call)                                                               \
    do {                                                                                     \
        cudaError_t _err = (call);                                                           \
        if (_err != cudaSuccess) {                                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s (%d)\n",                              \
                         __FILE__, __LINE__, cudaGetErrorString(_err), static_cast<int>(_err)); \
            std::abort();                                                                    \
        }                                                                                    \
    } while (0)
#endif

} // namespace flint
