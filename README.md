# Flint

**Zero-dependency C++23 + raw CUDA/PTX single-user LLM inference engine for consumer NVIDIA GPUs.**

Flint is a single-binary inference engine (~17.6k lines across C++23 and 13 `.cu` kernel units) built from scratch for `batch = 1` local coding on 16 GB, 24 GB, and 32 GB consumer NVIDIA GPUs (`sm_89` Ada and `sm_120a` Blackwell) with zero external runtime dependencies (no Python, PyTorch, Triton, cuDNN, CUTLASS, Boost, Protobuf, Rust tokenizers, or `libcurl`). Every `.cu` unit compiles with `0` register spills (`ptxas -v`), and passes `compute-sanitizer` (`memcheck`, `racecheck`, `initcheck`).

| Metric (`batch = 1`, greedy, `ctx = 17,408`) | Flint | Competitor Baseline | Advantage |
| --- | ---: | ---: | --- |
| **24 GB GPU (`RTX 4090`): `Inkling-2-Mini` 1,048,576-tok (1M) ctx** | **39.8 tok/s** (`23.36 GiB` VRAM, `3/3` NIAH pass) | `vLLM` caps 80 GB H100 at `262k` (needs 192 GB GB200 for `1M`) | Full native `1M` context 100% in 24 GB VRAM (`--kv-wht4`) |
| **32 GB GPU (`RTX 5090`): `Inkling-2-Mini` 1,048,576-tok (1M) ctx** | **51.4–52.1 tok/s** (`23.7–27.7 GiB` VRAM, `3/3` NIAH) | `llama.cpp` caps at `~45k` in 24 GB VRAM without offload | Exact dense attention across `1,048,576` tokens (`--kv-wht4`/`--kv-wht84`) |
| **16 GB GPU (`RTX 4060 Ti`): `Inkling-2-Mini` (26.4B MoE)** | **60.3 tok/s** (`15.04 GiB` VRAM, **119.8** MTP) | `vLLM` / `SGLang` / `llama.cpp` `IQ4_NL` all OOM | Full-VRAM 4-bit where `vLLM`/`SGLang`/`llama.cpp` OOM (`76.5%` peak GDDR6) |
| **24 GB GPU (`RTX 4090`): `Inkling-2-Mini` base / MTP** | **151.3** / **349.6 tok/s** (`15.3 GiB` VRAM) | 73.6 tok/s (`llama.cpp`); `vLLM`/`SGLang` OOM | **2.06×** base, **4.75×** lossless MTP |
| **32 GB GPU (`RTX 5090`): `Inkling-2-Mini` base (`128 / 4k / 16k`)** | **183.8** (`185.9` short) / **167.1** / **157.3 tok/s** | `vLLM` 175.2 / 170.4 / 163.8 · `SGLang` 161.2 / 156.6 / 150.9 | **+6.0%** vs `vLLM` short; beats `SGLang` at all contexts |
| **32 GB GPU (`RTX 5090`): `Inkling-2-Mini` prefill & MTP** | **220.1 / 892.3 ms** (`pp1024`/`4096`), **386.2 tok/s** MTP | `llama.cpp` 522.4 / 1,940 ms, 88.3 tok/s | **2.17×–2.37×** prefill, **4.37×** lossless MTP vs `llama.cpp` |
| **24 GB GPU (`RTX 4090`): `Qwen2.5-Coder-7B` FP8 (`128 / 4k / 16k`)** | **121.3 / 110.1 / 100.0 tok/s**, **34.2 ms** `7k` prefix hit | `SGLang` 112.1 / 104.8 / 87.7 (64.7 ms) · `vLLM` 90.7 / 88.9 / 83.7 (85.5 ms) | Beats `SGLang` & `vLLM` at all contexts; **1.9×–2.5×** prefix hit |
| **24 GB GPU (`RTX 4090`): `Qwen2.5-Coder-7B` W4A16 (`128` / `16k`)** | **198.9** / **156.2 tok/s** (`--kv-wht4`), **366.7** spec | `llama.cpp Q4_0` 191.0 / 154.7 · `Q4_K_M` 180.5 / 148.2 | Beats both `Q4_0` and `Q4_K_M` at `128` and `16k` |
| **Cold start to first token (`Inkling-2-Mini`)** | **0.72–2.2 s** | 26.8 s (`llama-server`) · 108.7 s (`SGLang`) · 195–451 s (`vLLM`) | **12×–625× faster** |
| **Host RAM & binary footprint** | **252–626 MiB** RSS, **~1.1 MB** stripped binary | 3.8–9.0 GiB RSS (`vLLM` / `SGLang` / `llama-server`) | **7.5×–20× lower host RAM** |

---

## Quick Start

Requires a C++23 compiler (`g++ >= 13` or `clang++ >= 17`) and CUDA Toolkit `>= 12.4` (`>= 12.8` for Blackwell `sm_120a` PTX 8.7 FP4 Tensor Cores). CPU-only builds (`CUDA=0`) work on Linux and macOS for running unit tests and the FP32 CPU reference verifier.

```sh
# Build (RTX 40-series: sm_89 | RTX 50-series: sm_120 -> auto sm_120a PTX FP4 TC | CPU: CUDA=0)
make CUDA=1 CUDA_ARCH=sm_89 -j$(nproc) all flint-test && ./flint-test

# 1. Pack a Hugging Face checkpoint once into a self-contained 128 B-aligned .pack (embeds binary BPE)
./flint pack /path/to/Qwen2.5-Coder-7B-Instruct -o qwen7b-w4a16.pack --quant w4a16
./flint pack /path/to/Inkling-2-Mini            -o inkling2-16gb.pack --quant w4a16  # 14.51 GiB -> fits 16 GB GPUs
./flint pack /path/to/Inkling-2-Mini            -o inkling2-fp8.pack  --quant fp8    # 16.92 GiB -> 24–32 GB GPUs

# 2. Interactive CLI (including full 1,048,576-token context & persistent KV snapshots on 24 GB)
./flint run qwen7b-w4a16.pack  --kv-wht4 --spec ngram -p "Write a C++23 radix sort:" --max-tokens 256 --greedy --stats
./flint run inkling2-16gb.pack --kv-int8 --spec mtp   -p "int fibonacci(int n) {"    --max-tokens 256 --greedy --stats
./flint run inkling2-16gb.pack --kv-wht4 --max-seq-len 1048576 -f repo_1m.txt --save-kv repo_1m.flkv --max-tokens 1 --greedy --stats
./flint run inkling2-16gb.pack --kv-wht4 --max-seq-len 1048576 --load-kv repo_1m.flkv -p "Question: ..." --max-tokens 76 --greedy --stats

# 3. Local OpenAI HTTP/SSE + AF_UNIX daemon (with prefix KV cache)
./flint serve inkling2-16gb.pack --kv-int8 --spec mtp --port 8080 --unix /tmp/flint.sock --max-seq-len 17408
```

---

## Benchmarks

All runs use `batch = 1`, greedy decoding (`temperature = 0.0`, `ignore_eos = true`), `ctx = 17,408` across HTTP/SSE servers, and `p50` medians (`2` warmup + `7` timed runs on W1/W4). Physical HBM bandwidth counts strictly **active bytes read per target forward pass** (the single `7 KB` embedding row lookup, active weights, LM head, and active KV cache — never the unreferenced `1.090 GB` embedding table, and counting only actual target passes during speculative decoding).

### Track 2A — `thinkingmachines/Inkling-2-Mini` on RTX 5090 (32 GB GDDR7, `1,792 GB/s`)

Four engines measured on the same RTX 5090: `flint` (`sm_120a` PTX 8.7 FP4 Tensor Cores, `benchmarks/bench_results_5090_v20.json`), `vLLM` (`0.34.0rc1.dev1130+g8c17e4692`, `--language-model-only`), `SGLang` (`0.5.6.post3.dev15390+g9c64454bb`, PR `#51926`), and `llama.cpp` (`b052049`, `IQ4_NL`, `-ngl 99 -fa on`) (`benchmarks/bench_results_inkling2_4engine.json`). `flint` 16 GB mode requantizes `L20..23` MXFP8 experts to NVFP4 (`cos = 0.9974`, `top1 = exact`); even with `L20..23` left in native MXFP8 (`18.2 GiB`), `flint` reaches `180.2 tok/s` base.

#### Decode, prefill, and speculative code editing

| Engine & Mode (RTX 5090) | W1 (`128`) | W2 (`4k`) | W3 (`16k`) | W4 `pp1024` / `pp4096` | W7 Code Edit (`gen=256`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **`flint` 16 GB (PTX FP4 TC)** | **183.8** (`185.9` short) | 167.1 | 157.3 | 220.1 / 892.3 ms | **386.2** (`256/256` MTP) |
| `vLLM` nightly (Base) | 175.2 | **170.4** | **163.8** | **66.5 / 226.0 ms** | 171.7 |
| `vLLM` nightly (DSpark MTP, `k=7`) | 210.0 | — | — | — | **657.4** |
| `SGLang` PR `#51926` (Base) | 161.2 | 156.6 | 150.9 | 83.9 / 255.8 ms | 157.6 |
| `SGLang` PR `#51926` (DSpark MTP, `k=8`) | **286.7** | — | — | — | 592.2 |
| `llama-server` `b052049` (`IQ4_NL`) | 90.4 (`93.0` direct) | 88.1 | 84.7 | 522.4 / 1,940 ms | 88.3 |

#### Prefix cache, cold start, and memory footprint

| Engine & Mode (RTX 5090) | W5 Prefix (`7k` Cold → Hit) | W6 Cold Start | Peak VRAM | Host RSS |
| --- | ---: | ---: | ---: | ---: |
| **`flint` 16 GB (PTX FP4 TC)** | 4,035 → 147.6 ms (`27.3×`) | **0.72–2.2 s** | **15.9–16.2 GiB** *(fits 16 GB)* | **571–626 MiB** |
| `vLLM` nightly (Base) | 3,719 → 823.0 ms | 450.58 s *(cold JIT)* | 27.3 GiB *(OOMs 24 GB)* | 4,577 MiB |
| `vLLM` nightly (DSpark MTP, `k=7`) | — | 194.99 s *(warm JIT)* | 27.2 GiB *(OOMs 24 GB)* | 5,225 MiB |
| `SGLang` PR `#51926` (Base) | 444 → **53.1 ms** | 108.65 s *(warm Triton)* | 28.3 GiB *(OOMs 24 GB)* | 7,578 MiB |
| `SGLang` PR `#51926` (DSpark MTP, `k=8`) | — | 178.20 s *(warm Triton)* | 27.7 GiB *(OOMs 24 GB)* | 8,222 MiB |
| `llama-server` `b052049` (`IQ4_NL`) | 3,429 → 515.8 ms | 26.84 s | 19.7 GiB *(OOMs 16 GB)* | 8,972 MiB |

### Track 2B — `Inkling-2-Mini` on Consumer 16 GB & 24 GB GPUs

Neither `vLLM` nor `SGLang` fits in 24 GB VRAM (`27.2–28.3 GiB`); on 16 GB, `llama.cpp IQ4_NL` (`19.7 GiB`) also OOMs. Receipts: [`benchmarks/bench_results_4060ti_16gb_v12.json`](benchmarks/bench_results_4060ti_16gb_v12.json) and [`benchmarks/bench_results_4090_v14.json`](benchmarks/bench_results_4090_v14.json).

| GPU & Engine | W1 (`128`) | W2 (`4k`) | W3 (`16k`) | W4 `pp1024` / `pp4096` | W7 Code Edit (`--spec mtp`, `256/256`) | Live VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **RTX 4090 24 GB — `flint` (16 GB pack)** | **151.3 tok/s** | **140.9 tok/s** | **134.1 tok/s** | **347.0 / 1,182.9 ms** | **349.6 tok/s** (`4.75×` `llama.cpp`) | **15.3–15.9 GiB** |
| RTX 4090 24 GB — `llama.cpp` (`1fdda89`, `IQ4_NL`) | 73.6 tok/s | 71.2 tok/s | 67.4 tok/s | 377.7 / 1,989.7 ms | — *(no MTP)* | 19.7 GiB |
| **RTX 4060 Ti 16 GB — `flint` (16 GB pack)** | **60.3 tok/s** (`76.3%` peak) | **57.8 tok/s** | **55.3 tok/s** (`76.5%` peak) | **829.7 / 3,354.7 ms** | **119.8 tok/s** | **15.04–15.32 GiB** |

### Track 2C — Full 1,048,576-Token (1M) Context 100% in VRAM

Because 40 of `Inkling-2-Mini`'s 48 visits use a fixed `1,024`-token sliding window (`85 MiB` total in INT8), only 8 global visits grow with context. Rotating global Keys and Values with a 128-point orthonormal Walsh-Hadamard transform (`H^T = H, H^2 = I`) into symmetric 4-bit nibbles with BF16 per-head scales (`--kv-wht4`, `WHT-K4V4`) compresses `1,048,576` tokens of global KV cache to `8.25 GiB` (`12.25 GiB` with `--kv-wht84` INT8 K + WHT-4-bit V on 32 GB GPUs) — fitting the entire `26.4B` model and its native `1,048,576`-token dense attention window 100% in VRAM on a single 24 GB RTX 4090 with zero CPU offload and zero token eviction.

Receipts: [`benchmarks/bench_results_4090_1m_v21.json`](benchmarks/bench_results_4090_1m_v21.json), [`benchmarks/bench_results_4090_v22.json`](benchmarks/bench_results_4090_v22.json), and [`benchmarks/bench_results_5090_1m_v21.json`](benchmarks/bench_results_5090_1m_v21.json).

| GPU & KV Mode (`inkling2-16gb.pack`, `batch = 1`) | Context Length | Decode `tok/s` (`itl p50`) | Prefill / Snapshot Time | Live Peak VRAM (`nvidia-smi` / CUDA) | Host RSS | Multi-Depth NIAH Retrieval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **RTX 4090 24 GB (`--kv-wht4`, `WHT-K4V4`)** | `128` | **150.0 tok/s** (`6.67 ms`) | `0.09 s` | `15,777 MiB` (`15.41 GiB`) | `535 MiB` | — |
| RTX 4090 24 GB (`--kv-wht4`, `WHT-K4V4`) | `16,384` | **134.3 tok/s** (`7.44 ms`) | `3.83 s` | `15,967 MiB` (`15.59 GiB`) | `536 MiB` | `16k` `cos = 0.9878`, `top1 = exact` |
| RTX 4090 24 GB (`--kv-wht4`, `WHT-K4V4`) | `65,536` | **118.5 tok/s** (`8.43 ms`) | `21.93 s` | `16,363 MiB` (`15.98 GiB`) | `536 MiB` | `1/1` pass (`66.6k` @ `20%` depth) |
| RTX 4090 24 GB (`--kv-wht4`, `WHT-K4V4`) | `131,072` | **106.5 tok/s** (`9.30 ms`) | `60.97 s` | `16,923 MiB` (`16.53 GiB`) | `537 MiB` | — |
| RTX 4090 24 GB (`--kv-wht4`, `WHT-K4V4`) | `262,144` (`256k`) | **87.1 tok/s** (`11.38 ms`) | `188.0–252.8 s` | `17,887 MiB` (`17.47 GiB`) | `539 MiB` | **`2/2` pass** (`15%`, `55%` depths) |
| RTX 4090 24 GB (`--kv-wht4`, `WHT-K4V4`) | `521,788` (`512k`) | **67.1 tok/s** (`14.77 ms`) | `750.4 s` *(or **`0.73 s`** `--load-kv`)* | `19,935 MiB` (`19.47 GiB`) | `534 MiB` | **`3/3` pass** (`10%`, `50%`, `90%` depths) |
| **RTX 4090 24 GB (`--kv-wht4`, `WHT-K4V4`)** | **`1,048,107` (+76 gen)** | **39.8 tok/s** (`25.04 ms`) | `4,598.2 s` | **`23,927 MiB`** (`23.37 / 24.0 GiB`) | **`547 MiB`** | **`3/3` pass** (`10%`, `50%`, `95%` depths) |
| **RTX 5090 32 GB (`--kv-wht4`, `WHT-K4V4`)** | **`1,048,107` (+76 gen)** | **51.4 tok/s** (`19.42 ms`) | `2,445.5 s` | **`24,269 MiB`** (`23.70 GiB`) | `664 MiB` | Identical greedy path |
| **RTX 5090 32 GB (`--kv-wht84`, `WHT-K8V4`)** | **`1,048,107` (+76 gen)** | **52.1 tok/s** (`19.11 ms`) | `2,492.0 s` | **`28,365 MiB`** (`27.70 GiB`) | `640 MiB` | **`3/3` pass** (`10%`, `50%`, `95%` depths) |

- **Zero-recompute persistent KV snapshots (`--save-kv` / `--load-kv`):** Dumping a `521,724`-token `WHT-K4V4` KV snapshot (`4.19 GiB`) to NVMe takes `1.30 s`; reloading via double-buffered pinned DMA takes `0.73–1.62 s` (`1.97–2.91 s` total TTFT including query prefill, retrieving all `3/3` needles verbatim at `67.1 tok/s`).

### Track 1 — `Qwen/Qwen2.5-Coder-7B-Instruct` on RTX 4090 & RTX 5090

Four-engine HTTP/SSE comparison on RTX 4090 (`ctx = 17,408`, [`benchmarks/bench_results_4090_v6.json`](benchmarks/bench_results_4090_v6.json), [`benchmarks/bench_results_4090_v10.json`](benchmarks/bench_results_4090_v10.json)):

| Server & Precision (RTX 4090) | W1 (`128`) | W2 (`4k`) | W3 (`16k`) | W4 `pp1024` / `pp4096` | W5 Prefix Hit (`7,168 / 7,196`) | W6 Cold Start | VRAM / Host RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`flint serve` W4A16** | **197.7** (`198.9` direct) | **169.6** | **146.8** (`156.2` `WHT-K8V4`) | 125.5 / 576.2 ms | 1,199 → 49.5 ms | **2.84 s** (`1.51 s` CLI) | **6.9 GiB** / **441 MiB** |
| **`flint serve` FP8** | **121.3** (`121.8` direct) | **110.1** | **100.0** | **77.1** / 403.5 ms | 911 → **34.2 ms** | **6.07 s** | **10.0 GiB** / **252 MiB** |
| `SGLang 0.5.23` FP8 | 112.1 | 104.8 | 87.7 | 84.4 / **284.2 ms** | 533 → 64.7 ms | 63.78 s | 21.9 GiB / 5,105 MiB |
| `vLLM 0.34.0` FP8 | 90.7 | 88.9 | 83.7 | 84.8 / 303.0 ms | 536 → 85.5 ms | 104.22 s | 21.5 GiB / 4,092 MiB |
| **`flint serve` BF16** | 64.0 (`90.1%` peak) | 60.7 | 57.5 | **110.8** / 540.1 ms | 1,136 → **41.7 ms** | **6.82 s** | **16.3 GiB** / **441 MiB** |
| `SGLang 0.5.23` BF16 | **65.1** | **62.6** | 56.7 | 130.3 / 439.5 ms | 801 → 59.6 ms | 116.84 s | 21.8 GiB / 5,111 MiB |
| `vLLM 0.34.0` BF16 | 63.5 | 62.5 | **59.9** | 119.5 / **434.0 ms** | 756 → 82.8 ms | 116.29 s | 21.5 GiB / 3,784 MiB |

- **RTX 4090 direct vs. `llama.cpp` (`1fdda89`, `sm_89`):** W4A16 `198.9 tok/s` (`156.2` @ `16k` with `--kv-wht4`, `133.3` @ `32k` in `6.7 GiB`) beats `Q4_0` (`191.0` / `154.7`) and `Q4_K_M` (`180.5` / `148.2`); lossless speculative decoding reaches **`366.7 tok/s`** (`--spec draft:0.5B`) and **`361.1 tok/s`** (`--spec ngram`), `256/256` identical.
- **RTX 5090 (`benchmarks/bench_results_5090_v08.json`):** `Qwen2.5-Coder-7B` W4A16 runs at **`301.4 tok/s`** (`3.31 ms` ITL, `0.96 s` cold start, `309 MiB` RSS), `pp1024 = 56.18 ms`, `pp4096 = 316.98 ms`, and `198.1 tok/s` at `16k` (`--kv-wht4`).

<details>
<summary>Full archive of machine-readable benchmark JSON receipts</summary>

- RTX 4090 24 GB `Inkling-2-Mini` `LDG.E.128` 128k–512k decode & `--save-kv`/`--load-kv` snapshots: [`benchmarks/bench_results_4090_v22.json`](benchmarks/bench_results_4090_v22.json)
- RTX 4090 24 GB `Inkling-2-Mini` 1,048,576-token (1M) NIAH & scaling: [`benchmarks/bench_results_4090_1m_v21.json`](benchmarks/bench_results_4090_1m_v21.json)
- RTX 5090 32 GB `Inkling-2-Mini` 1,048,576-token (1M) `WHT-K4V4` & `WHT-K8V4`: [`benchmarks/bench_results_5090_1m_v21.json`](benchmarks/bench_results_5090_1m_v21.json)
- RTX 5090 `Inkling-2-Mini` (`sm_120a` PTX FP4 TC): [`benchmarks/bench_results_5090_v20.json`](benchmarks/bench_results_5090_v20.json)
- RTX 5090 4-Engine Shootout (`flint` / `vLLM` / `SGLang` / `llama.cpp`): [`benchmarks/bench_results_inkling2_4engine.json`](benchmarks/bench_results_inkling2_4engine.json)
- RTX 4090 `Inkling-2-Mini` (16 GB pack): [`benchmarks/bench_results_4090_v14.json`](benchmarks/bench_results_4090_v14.json)
- Physical 16 GB RTX 4060 Ti live receipt: [`benchmarks/bench_results_4060ti_16gb_v12.json`](benchmarks/bench_results_4060ti_16gb_v12.json)
- RTX 4090 `Qwen2.5-Coder-7B` 4-engine matrix & `WHT-K8V4` / Megakernel ablations: [`v6`](benchmarks/bench_results_4090_v6.json), [`v7`](benchmarks/bench_results_4090_v7.json), [`v8`](benchmarks/bench_results_4090_v8.json), [`v9`](benchmarks/bench_results_4090_v9.json), [`v10`](benchmarks/bench_results_4090_v10.json)
- Earlier RTX 5090 sweep receipts: [`v08`](benchmarks/bench_results_5090_v08.json), [`v09`](benchmarks/bench_results_5090_v09.json), [`v10`](benchmarks/bench_results_5090_v10.json), [`v11`](benchmarks/bench_results_5090_v11.json), [`v12`](benchmarks/bench_results_5090_v12.json), [`v13`](benchmarks/bench_results_5090_v13.json), [`v15`](benchmarks/bench_results_5090_v15.json)

</details>

---

## Architecture

1. **Single-slab zero-copy `.pack` loader (`src/weights.cpp`, `src/tokenizer.cpp`):**
   - `flint pack` converts Hugging Face `.safetensors` into a 128-byte-aligned contiguous binary slab with fused QKV / Gate+Up weights and a pre-parsed binary BPE table (`< 25 ms` load). At startup, one `cudaMalloc` plus 4-thread double-buffered pinned DMA transfers the whole model in `0.72–2.2 s`.
   - For `Inkling-2-Mini`'s 16 GB consumer pack (`--quant w4a16`, `14.51 GiB` on disk, `15.04 GiB` live VRAM), `flint` strips the `637 MB` vision tower, requantizes `L20..23` MXFP8 experts to native NVFP4 (`-2.0 GiB`), packs `embed` as W4A16 and attention projections + `unembed` as per-row FP8, and skips uploading the `mtp.*` slab when `--spec mtp` is off. Out of `14.51 GiB`, `13.09 GiB` is raw 4.5-bit NVFP4 MoE experts alone — the physical floor for full-VRAM 4-bit execution.

2. **Roofline `M = 1..8` GEMV kernels & Amdahl decomposition (`src/kernels/gemv_*.cu`):**
   - Fitting BF16 (`15.60 ms` / `14.16 GB`), FP8 (`8.23 ms` / `7.09 GB`), and W4A16 (`5.07 ms` / `4.03 GB`) on RTX 4090 gives `ITL(W) ≈ 0.78 ms + W / 955 GB/s` — all three custom GEMVs sustain **`~955 GB/s` (`94.7%` of GDDR6X peak)** above a `0.78 ms` (`27.8 µs/layer`) fixed non-GEMV floor.
   - Asymmetric W4A16 uses native BF16x2 PTX `lop3.b32` with IEEE-754 BF16 magic bias (`0x43004300` = `bf162(128.0)` where `ULP = 1.0`, `39–40` regs, `0` spills), eliminating ALU stalls on `M = 2..8` speculative verification (`366.7 tok/s` lossless).

3. **Zero-calibration INT8 & Walsh-Hadamard `WHT-K8V4` KV cache (`src/kernels/rope_kv.cu`, `flash_decode.cu`):**
   - `Qwen2.5-Coder-7B` Layer 27 has `max|k_proj.bias| = 442.0` vs. median `0.34` (1,300:1 outlier ratio), so even per-token/per-head dynamic-scaled FP8 E4M3 KV only reached `cos = 0.914` (`+1.06` PPL).
   - Dynamic-scaled 7-bit linear `--kv-int8` reaches `cos = 0.999538` (`PPL = 2.0972` vs. BF16 `2.0715`) with PTX `__dp4a` attention. Pairing INT8 Keys with 128-point Walsh-Hadamard rotated 4-bit Values (`--kv-wht4`) plus a 16×16×16 WMMA prefill kernel (`69` regs, `0` spills) achieves `cos = 0.998838` (`PPL = 2.0993` vs. unrotated `INT4-RAW` `14,343.67`), hitting `156.2 tok/s` at `16k` and `133.3 tok/s` at `32k`.

4. **Looped MoE + Raw Blackwell PTX 8.7 FP4 Tensor Cores (`src/inkling2_engine.cpp`, `inkling2_forward.cpp`, `gemv_nvfp4_moe.cu`, `inkling2_{core,attn,wht}.cu`):**
   - Implements `Inkling-2-Mini`'s full 48-visit looped core (`[6..18) × 3`), Block AttnRes, rank-64 cross-pass controllers, `W = 4` short convs with 16-slot speculative rollback rings, 5:1 local SWA (`1,024`-token ring) + global relative-bias attention, 128-expert Top-8 MoE (NVFP4 + MXFP8), and CUDA-graphed zero-host-sync 7-step MTP.
   - On Blackwell (`sm_120a`), `inkling_moe_build_tiles_kernel` (~5 µs, zero host sync) feeds `.pack` NVFP4 weights directly into `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3` (`48` regs, `0` spills) for `M >= 16` prefill (`pp128 = 29.2 ms`, `pp4096 = 892.3 ms`), while keeping `M = 2..8` MTP verify on exact `BF16 × NVFP4` for speed (`386.2 tok/s`) and `256/256` parity.

5. **Empirical negative ablations (reverted cleanly):**
   - **Cooperative Persistent Megakernel (`--megakernel`, `1` launch for all 28 layers + LM head + argmax):** `256/256` identical and sanitizer-clean, but measured `178.1–184.7 tok/s` (`170.3` at 64 regs) vs. `197.4–197.9 tok/s` for the CUDA Graph — monolithic `120`-reg pressure drops SM occupancy (`8–16` warps/SM) more than inter-kernel graph gaps cost. Kept as an opt-in ablation flag.
   - Four other launch-reduction fusions also lost to SM parallelism and were reverted: 1-block 1,024-thread fused router (`-12.7 tok/s`), 24-block 1-launch local SWA (`-52 tok/s` @ `4k`), `splits = 1` short-seq FlashDecode (`198.9 → 184.6 tok/s`, losing GQA 7× reuse), and 128-expert brute-force grid dedup (`18.9 → 28.4 ms`, `atomicAdd` contention).

### Codebase Map (~17.6k lines in `src/`)

| File | Lines | Responsibility |
| --- | ---: | --- |
| `src/engine.hpp/.cpp` | 1,760 | Dense Transformer engine (`Qwen2.5`, `Qwen3`, `Llama-3`), CUDA Graph decode, Radix KV, speculative loop |
| `src/inkling2_engine.hpp/.cpp` | 1,629 | `Inkling-2-Mini` lifecycle, `.pack` weight slab loader, VRAM-adaptive chunking, prefix snapshots, `generate_ids` / `eval_logits` |
| `src/inkling2_forward.cpp` | 913 | `Inkling-2-Mini` 48-visit forward pass, BF16/FP8 `cublasLt` linears, MTP context/drafting, CUDA Graph capture |
| `src/kernels/inkling2_core.cu`, `inkling2_common.cuh` | 1,426 | AttnRes blends, 3-stream controllers/norms, hidden & KV short-conv, Q/R clip, fused BF16/FP8 attention GEMVs |
| `src/kernels/inkling2_attn.cu` | 636 | Local SWA (`1,024`-token ring) and global BF16/INT8 `__dp4a` Split-K decode + WMMA relative-bias prefill |
| `src/kernels/inkling2_wht.cu` | 679 | 128-point WHT 4-bit/8-bit global KV write, 256-split global decode, inverse-WHT epilogue, 64-Q WMMA prefill |
| `src/kernels/inkling2_mtp.cu` | 684 | Built-in 3-layer MTP transformer + rank-256 Markov RNN drafting kernels |
| `src/kernels/gemv_nvfp4_moe.cu` | 1,453 | NVFP4/MXFP8 MoE GEMVs, fused `down_sconv`, Blackwell `sm_120a` PTX 8.7 `mxf4nvf4` FP4 Tensor-Core grouped MoE |
| `src/kernels/flash_decode.cu` | 1,220 | GQA-grouped Split-K FlashDecoding (BF16 / INT8 `__dp4a` / `WHT-K8V4`) + WMMA Tensor-Core prefill |
| `src/kernels/gemv_{bf16,fp8,w4a16}.cu` | 839 | Roofline `M = 1..8` dense GEMVs, PTX `lop3.b32` W4A16, native FP8 `cublasLt` activation quant |
| `src/kernels/megakernel_w4a16.cu` | 557 | Opt-in 1-launch cooperative persistent megakernel ablation (`--megakernel`) |
| `src/kernels/{rmsnorm,rope_kv,sampling}.cu`, `kernels.hpp` | 1,557 | Fused Residual+RMSNorm, RoPE + paged BF16/INT8/WHT4 KV write, GPU argmax + confidence, kernel declarations |
| `src/weights.hpp/.cpp`, `tokenizer.hpp/.cpp` | 1,904 | Zero-copy `.safetensors` & `.pack` loader, streaming offline packer, pure C++23 BPE (`Qwen` + `o200k`) |
| `src/server.hpp/.cpp`, `kv_cache.*`, `speculative.*`, `cli.*`, `main.cpp`, `common.hpp`, `json.hpp`, `model_config.hpp`, `cpu_ref.*` | 2,386 | OpenAI HTTP/SSE + `AF_UNIX` daemon, paged Radix KV manager, n-gram drafter, RFC 8259 JSON, CPU ref |

---

## Verification Gates

| Gate | `Qwen/Qwen2.5-Coder-7B-Instruct` | `thinkingmachines/Inkling-2-Mini` |
| --- | --- | --- |
| **Gate 1 — Tokenizer (`AutoTokenizer`)** | `100/100` exact | `100/100` exact (`200,058`-vocab `o200k` + `ignore_merges`) |
| **Gate 2 — Numerical & Greedy Parity** | BF16 `cos = 0.999999` vs. HF; FP8 `cublasLt` `cos = 0.996787, top1 = exact` | `32/32` exact greedy vs. `llama.cpp`; FP8-Attn `cos = 0.999499`; 16 GB pack `cos = 0.997381, top1 = exact` |
| **Gate 3 — KV Fidelity & PPL** | `--kv-int8`: `cos = 0.999538, PPL = 2.0972`; `--kv-wht4`: `cos = 0.998838, PPL = 2.0993` | `--kv-int8`: `cos = 0.993870` (`16k`); `--kv-wht84`: `cos = 0.992638`; `--kv-wht4`: `cos = 0.987805`, all `top1 = exact`; `3/3` NIAH pass at `1,048,107` tokens |
| **Gate 4 — Lossless Speculative** | `256/256` bit-identical (`ngram`, `draft:0.5B`) | `256/256` bit-identical on `code_edit` (16 / 24 / 32 GB) + BF16-Attn greenfield & edit |
| **Gate 5 — Sanitizers & Spills** | `0` spills across all 13 `.cu` units; `memcheck`/`racecheck`/`initcheck` clean; host ASan+UBSan `4/4` | `0` spills (`sm_89` & `sm_120a`); `memcheck`/`racecheck`/`initcheck` clean; host ASan+UBSan `4/4` |

### Supported Models & Deliberate Non-Goals

- **GPU-verified models:** `thinkingmachines/Inkling-2-Mini` (16 GB RTX 4060 Ti, 24 GB RTX 4090, 32 GB RTX 5090) and `Qwen/Qwen2.5-Coder-7B-Instruct` (+ `0.5B` draft on RTX 4090 & RTX 5090). The dense engine also implements `Qwen2`/`Qwen2.5`, `Qwen3` (QK-RMSNorm), and `Llama-3.1`/`3.2` (Llama-3/YaRN RoPE scaling).
- **Deliberate non-goals:** multi-user continuous batching (`batch > 1`), multi-GPU parallelism, multimodal vision towers (text-only), LoRA hot-swapping, and non-CUDA GPU backends. On 32 GB+ Blackwell GPUs once booted, `vLLM` and `SGLang` still lead steady-state `Inkling-2-Mini` long prefill (`66–84 ms` vs. `220 ms`) and DSpark MTP (`592–657` vs. `386 tok/s`) via CUTLASS 4 / FlashAttention-4.

## License

[MIT](LICENSE)
