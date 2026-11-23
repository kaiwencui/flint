#!/usr/bin/env python3
"""
Unified Single-Session 4-Engine RTX 4090 Benchmark Matrix for flint v5 (W1-W7):
  Engines: flint (CLI/bench, AF_UNIX, HTTP/SSE), llama.cpp (sm_89), vLLM (BF16/FP8), SGLang (BF16/FP8).
  Methodology: 2 warmup + 7 timed runs (p50 median & p99 tail), greedy (temp=0), ignore_eos=true.
"""
import argparse
import http.client
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path


def wait_tcp(host, port, timeout=180.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection((host, port), timeout=0.3):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def wait_http_ready(host, port, path="/v1/models", timeout=240.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2.0)
            conn.request("GET", path)
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def proc_tree_rss_mib(root_pid):
    """Sum Rss and Private_Dirty across root_pid and its child processes."""
    pids = {root_pid}
    try:
        out = subprocess.check_output(["pgrep", "-P", str(root_pid)], text=True)
        for line in out.splitlines():
            if line.strip().isdigit():
                pids.add(int(line.strip()))
    except Exception:
        pass
    rss = pd = 0.0
    for pid in pids:
        try:
            for line in open(f"/proc/{pid}/smaps_rollup"):
                if line.startswith("Rss:"):
                    rss += int(line.split()[1]) / 1024.0
                elif line.startswith("Private_Dirty:"):
                    pd += int(line.split()[1]) / 1024.0
        except Exception:
            pass
    return rss, pd


def gpu_vram_used_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True
    ).strip()
    return int(out.splitlines()[0])


def http_sse_completion(host, port, model_id, prompt, max_tokens=256, ignore_eos=True):
    """Measure exact HTTP/SSE TTFT, ITL p50/p99, and decode tok/s against any OpenAI /v1/completions server."""
    body = json.dumps({
        "model": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": ignore_eos,
        "stream": True,
    }).encode()
    conn = http.client.HTTPConnection(host, port, timeout=180.0)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/completions", body=body,
                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    resp = conn.getresponse()
    if resp.status != 200:
        err = resp.read().decode(errors="replace")
        conn.close()
        raise RuntimeError(f"HTTP {resp.status}: {err[:300]}")

    tok_times = []
    chunks = 0
    while True:
        raw_line = resp.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
            ch = obj.get("choices", [])
            if ch and (ch[0].get("text") is not None and ch[0].get("text") != "" or chunks > 0):
                tok_times.append(time.perf_counter())
                chunks += 1
        except Exception:
            pass
    conn.close()
    t_end = time.perf_counter()

    if not tok_times:
        return {"ttft_ms": (t_end - t0) * 1000.0, "decode_tok_s": 0.0, "itl_p50_ms": 0.0, "itl_p99_ms": 0.0, "chunks": 0}
    ttft_ms = (tok_times[0] - t0) * 1000.0
    if len(tok_times) >= 2:
        dec_s = tok_times[-1] - tok_times[0]
        decode_tok_s = (len(tok_times) - 1) / max(dec_s, 1e-6)
        itls = sorted((tok_times[i] - tok_times[i - 1]) * 1000.0 for i in range(1, len(tok_times)))
        itl_p50 = itls[len(itls) // 2]
        itl_p99 = itls[min(len(itls) - 1, int(len(itls) * 0.99))]
    else:
        decode_tok_s = 0.0
        itl_p50 = itl_p99 = 0.0
    return {
        "ttft_ms": round(ttft_ms, 2),
        "decode_tok_s": round(decode_tok_s, 2),
        "itl_p50_ms": round(itl_p50, 2),
        "itl_p99_ms": round(itl_p99, 2),
        "chunks": len(tok_times),
    }


def make_token_prompt(tok, target_tokens, seed_text="The quick brown fox jumps over the lazy dog in C++23 CUDA kernels. "):
    ids = tok.encode(seed_text, add_special_tokens=False)
    rep = (target_tokens // max(1, len(ids))) + 2
    full_ids = (ids * rep)[:target_tokens]
    return tok.decode(full_ids)


def bench_http_server(name, cmd, env, port, model_id, tok, warmup=2, runs=7, seq_list=(128, 4096, 16384)):
    """Start an OpenAI HTTP/SSE server, measure W6 cold start, W1-W5, and VRAM/RSS footprint, then shut down."""
    print(f"\n=== Benchmarking HTTP/SSE Server: {name} (port={port}) ===", flush=True)
    res = {"status": "ok"}
    t_start = time.perf_counter()
    logf = open(f"/tmp/srv_{name}.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    try:
        if not wait_http_ready("127.0.0.1", port, "/v1/models", timeout=240.0):
            raise RuntimeError(f"{name} failed to start within 240s; see /tmp/srv_{name}.log")

        # W6: Cold start to first token
        short_p = make_token_prompt(tok, 64)
        first = http_sse_completion("127.0.0.1", port, model_id, short_p, max_tokens=1, ignore_eos=True)
        res["w6_cold_start_s"] = round(time.perf_counter() - t_start, 3)
        print(f"  [{name}] W6 cold start -> 1st token: {res['w6_cold_start_s']:.3f} s", flush=True)

        # W1 / W2 / W3: Decode at seq=128, 4096, 16384 (2 warmup + 7 timed)
        for wid, slen in [("w1", 128), ("w2", 4096), ("w3", 16384)]:
            if slen not in seq_list:
                continue
            p_txt = make_token_prompt(tok, slen, seed_text=f"/* {wid} {slen} */ int f(int x) {{ return x * 3 + 1; }}\n")
            n_r = runs if slen <= 4096 else min(runs, 5)
            # Use unique prefix tag per run so prefix cache does NOT contaminate cold W1-W4 timings!
            for w_i in range(warmup):
                http_sse_completion("127.0.0.1", port, model_id, f"// warm{w_i}\n" + p_txt, max_tokens=32, ignore_eos=True)
            samples = []
            for r_i in range(n_r):
                s = http_sse_completion("127.0.0.1", port, model_id, f"// run{r_i}_{time.time_ns()}\n" + p_txt, max_tokens=256, ignore_eos=True)
                samples.append(s)
            samples.sort(key=lambda x: x["decode_tok_s"])
            med = samples[len(samples) // 2]
            res[wid] = med
            print(f"  [{name}] {wid.upper()} (seq={slen}): decode={med['decode_tok_s']:6.1f} tok/s | itl_p50={med['itl_p50_ms']:5.2f} ms | ttft={med['ttft_ms']:7.2f} ms", flush=True)

        # W4: Warm prefill TTFT (pp1024, pp4096 -> 1)
        res["w4"] = {}
        for plen in [1024, 4096]:
            p_txt = make_token_prompt(tok, plen, seed_text="uint64_t hash_block(const uint8_t* p, size_t n) { return n ^ 0x9e3779b9ULL; }\n")
            for w_i in range(warmup):
                http_sse_completion("127.0.0.1", port, model_id, f"// pw{w_i}\n" + p_txt, max_tokens=1, ignore_eos=True)
            ttfts = []
            for r_i in range(runs):
                s = http_sse_completion("127.0.0.1", port, model_id, f"// pr{r_i}_{time.time_ns()}\n" + p_txt, max_tokens=1, ignore_eos=True)
                ttfts.append(s["ttft_ms"])
            ttfts.sort()
            p50 = ttfts[len(ttfts) // 2]
            res["w4"][f"pp{plen}"] = {"ttft_ms": round(p50, 2), "pp_tok_s": round(plen / (p50 / 1000.0), 1)}
            print(f"  [{name}] W4 pp{plen}: ttft={p50:7.2f} ms ({plen/(p50/1000.0):7.1f} tok/s)", flush=True)

        # W5: Prefix-cache hit TTFT (7,168 shared + 23 new -> 64 gen)
        shared_code = ("int compute_bucket(uint32_t x, uint32_t y) { return (x * 2654435761u) ^ (y >> 3); }\n") * 175
        p_cold = shared_code + "\n// Query A: implement a fast binary search over sorted uint32_t:\n"
        r_cold = http_sse_completion("127.0.0.1", port, model_id, p_cold, max_tokens=64, ignore_eos=True)
        hit_ttfts = []
        r_hit = None
        for k in range(5):
            pk = shared_code + f"\n// Query {k+10}: implement a fast SIMD prefix sum variant {k} over uint32_t:\n"
            r_hit = http_sse_completion("127.0.0.1", port, model_id, pk, max_tokens=64, ignore_eos=True)
            hit_ttfts.append(r_hit["ttft_ms"])
        hit_ttfts.sort()
        r_hit["ttft_ms_p50"] = round(hit_ttfts[len(hit_ttfts) // 2], 2)
        res["w5"] = {"cold": r_cold, "hit": r_hit}
        print(f"  [{name}] W5 prefix cache: cold={r_cold['ttft_ms']:.2f} ms -> HIT_p50={r_hit['ttft_ms_p50']:.2f} ms (decode={r_hit['decode_tok_s']:.1f} tok/s)", flush=True)

        rss, pd = proc_tree_rss_mib(proc.pid)
        vram = gpu_vram_used_mib()
        res["footprint"] = {"rss_mib": round(rss, 1), "private_dirty_mib": round(pd, 1), "vram_mib": vram}
        print(f"  [{name}] Footprint: VRAM={vram} MiB | Host RSS={rss:.1f} MiB (Private_Dirty={pd:.1f} MiB)", flush=True)
    except Exception as e:
        res["status"] = f"error: {e}"
        print(f"  [{name}] ERROR: {e}", flush=True)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=15)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        logf.close()
        time.sleep(2.0)
    return res


def run_flint_bench(flint_bench, model_path, quant, seq_len, gen_len, warmup=2, runs=7, kv_fp8=False):
    cmd = [flint_bench, "--model", model_path, "--quant", quant,
           "--seq-len", str(seq_len), "--gen", str(gen_len),
           "--warmup", str(warmup), "--runs", str(runs)]
    if kv_fp8:
        cmd.append("--kv-fp8")
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    line = p.stdout.strip()
    rec = {"raw": line}
    for tok in line.replace("(", " ").replace(")", " ").split():
        if tok.startswith("ttft="): rec["ttft_ms"] = float(tok.split("=")[1].replace("ms", ""))
        elif tok.startswith("decode="): rec["decode_tok_s"] = float(tok.split("=")[1])
        elif tok.startswith("hbm="): rec["hbm_gbs"] = float(tok.split("=")[1])
        elif tok.startswith("cold="): rec["cold_ms"] = float(tok.split("=")[1].replace("ms", ""))
        elif tok.startswith("vram="): rec["vram_mib"] = int(tok.split("=")[1])
        elif tok.startswith("rss="): rec["rss_mib"] = int(tok.split("=")[1])
        elif tok.startswith("active_w="): rec["active_weight_gb"] = float(tok.split("=")[1])
        elif tok.startswith("itl_p50="): rec["itl_p50_ms"] = float(tok.split("=")[1].replace("ms", ""))
        elif tok.startswith("itl_p99="): rec["itl_p99_ms"] = float(tok.split("=")[1].replace("ms", ""))
    return rec


def run_llama_bench(llama_bench, gguf_path, pp, tg, d=0, reps=7):
    cmd = [llama_bench, "-m", gguf_path, "-p", str(pp), "-n", str(tg), "-d", str(d),
           "-r", str(reps), "-fa", "1", "-o", "json"]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    arr = json.loads(p.stdout)
    out = {}
    for row in arr:
        if row.get("n_prompt", 0) > 0 and row.get("n_gen", 0) == 0:
            out["pp_tok_s"] = row["avg_ts"]
            out["ttft_ms"] = 1000.0 * row["n_prompt"] / max(row["avg_ts"], 1e-6)
        elif row.get("n_gen", 0) > 0:
            out["tg_tok_s"] = row["avg_ts"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flint", default="./flint")
    ap.add_argument("--flint-bench", default="./flint-bench")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--packs-dir", required=True)
    ap.add_argument("--llama-bin-dir", default="")
    ap.add_argument("--vllm-python", default="")
    ap.add_argument("--sglang-python", default="")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--json-out", default="bench_results_4090_v5.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    packs = Path(args.packs_dir)
    llama_bench = str(Path(args.llama_bin_dir) / "llama-bench") if args.llama_bin_dir else ""
    llama_cli   = str(Path(args.llama_bin_dir) / "llama-cli")   if args.llama_bin_dir else ""
    has_llama = bool(llama_bench) and os.path.exists(llama_bench)

    report = {
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True
        ).strip(),
        "methodology": f"{args.warmup} warmup + {args.runs} timed runs (p50 median), greedy, ignore_eos=true",
        "w1": {}, "w2": {}, "w3": {}, "w4": {}, "w5": {}, "w6": {}, "w7": {},
        "http_servers": {},
    }

    print("=" * 94)
    print(f"  flint v5 Unified 4-Engine Benchmark Matrix | {report['gpu']}")
    print(f"  Methodology: {report['methodology']}")
    print("=" * 94, flush=True)

    # 1. Direct Engine Benchmarks (flint-bench & llama-bench)
    print("\n--- W1: Short Decode (seq=128, gen=256) ---", flush=True)
    for q in ["bf16", "fp8", "w4a16"]:
        pk = str(packs / f"qwen7b-{q}.pack")
        r = run_flint_bench(args.flint_bench, pk, q, 128, 256, warmup=args.warmup, runs=args.runs)
        report["w1"][f"flint_{q}"] = r
        print(f"  flint ({q:<5}):  decode={r['decode_tok_s']:6.1f} tok/s | itl_p50={r.get('itl_p50_ms',0):5.2f} ms | hbm={r['hbm_gbs']:6.1f} GB/s ({r['hbm_gbs']/10.08:4.1f}% peak) | active_w={r.get('active_weight_gb',0):.3f} GB | ttft={r['ttft_ms']:5.2f} ms | vram={r['vram_mib']} MiB", flush=True)
    if has_llama:
        for tag, fname in [("bf16", "qwen7b-bf16.gguf"), ("q4_0", "qwen7b-q4_0.gguf"), ("q4_k_m", "qwen7b-q4_k_m.gguf")]:
            gp = str(packs / fname)
            if os.path.exists(gp):
                lr = run_llama_bench(llama_bench, gp, pp=128, tg=256, d=0, reps=args.runs)
                report["w1"][f"llama_{tag}"] = lr
                print(f"  llama ({tag:<6}): decode={lr.get('tg_tok_s',0):6.1f} tok/s | pp128={lr.get('pp_tok_s',0):7.1f} tok/s", flush=True)

    for wid, seq in [("w2", 4096), ("w3", 16384)]:
        print(f"\n--- {wid.upper()}: Context Decode (seq={seq}, gen=256) ---", flush=True)
        n_r = args.runs if seq <= 4096 else min(args.runs, 5)
        for q in ["bf16", "fp8", "w4a16"]:
            pk = str(packs / f"qwen7b-{q}.pack")
            r = run_flint_bench(args.flint_bench, pk, q, seq, 256, warmup=1, runs=n_r, kv_fp8=False)
            report[wid][f"flint_{q}"] = r
            print(f"  flint ({q:<5} bf16kv): decode={r['decode_tok_s']:6.1f} tok/s | hbm={r['hbm_gbs']:6.1f} GB/s ({r['hbm_gbs']/10.08:4.1f}% peak) | ttft={r['ttft_ms']:7.2f} ms", flush=True)
        # Dynamic-scaled FP8 KV cache rows (BF16 weights + W4A16 weights)
        for q in ["bf16", "w4a16"]:
            pk = str(packs / f"qwen7b-{q}.pack")
            r8 = run_flint_bench(args.flint_bench, pk, q, seq, 256, warmup=1, runs=n_r, kv_fp8=True)
            report[wid][f"flint_{q}_fp8kv"] = r8
            print(f"  flint ({q:<5} fp8kv ): decode={r8['decode_tok_s']:6.1f} tok/s | hbm={r8['hbm_gbs']:6.1f} GB/s ({r8['hbm_gbs']/10.08:4.1f}% peak) | ttft={r8['ttft_ms']:7.2f} ms | vram={r8['vram_mib']} MiB", flush=True)
        if has_llama:
            for tag, fname in [("bf16", "qwen7b-bf16.gguf"), ("q4_0", "qwen7b-q4_0.gguf"), ("q4_k_m", "qwen7b-q4_k_m.gguf")]:
                gp = str(packs / fname)
                if os.path.exists(gp):
                    lr = run_llama_bench(llama_bench, gp, pp=0, tg=256, d=seq, reps=n_r)
                    report[wid][f"llama_{tag}"] = lr
                    print(f"  llama ({tag:<6}): decode={lr.get('tg_tok_s',0):6.1f} tok/s", flush=True)

    print("\n--- W4: Warm Prefill TTFT (1024 / 4096 -> 1) ---", flush=True)
    for plen in [1024, 4096]:
        for q in ["bf16", "w4a16"]:
            r = run_flint_bench(args.flint_bench, str(packs / f"qwen7b-{q}.pack"), q, plen, 1, warmup=args.warmup, runs=args.runs)
            report["w4"][f"flint_{q}_{plen}"] = r
            print(f"  flint {q:<5} pp{plen}: ttft={r['ttft_ms']:7.2f} ms ({plen/(r['ttft_ms']/1000.0):7.1f} tok/s)", flush=True)
        if has_llama:
            for tag, fname in [("bf16", "qwen7b-bf16.gguf"), ("q4_k_m", "qwen7b-q4_k_m.gguf")]:
                gp = str(packs / fname)
                if os.path.exists(gp):
                    lr = run_llama_bench(llama_bench, gp, pp=plen, tg=0, reps=args.runs)
                    report["w4"][f"llama_{tag}_{plen}"] = lr
                    print(f"  llama {tag:<6} pp{plen}: ttft={lr.get('ttft_ms',0):7.2f} ms ({lr.get('pp_tok_s',0):7.1f} tok/s)", flush=True)

    # W5 AF_UNIX (flint serve)
    print("\n--- W5: Radix Prefix-Cache Hit (flint serve AF_UNIX, 7,168 shared + 28 new) ---", flush=True)
    shared_code = ("int compute_bucket(uint32_t x, uint32_t y) { return (x * 2654435761u) ^ (y >> 3); }\n") * 175
    prompt_a = shared_code + "\n// Query A: implement a fast binary search over sorted uint32_t:\n"
    sock_path = "/tmp/flint_w5_v5.sock"
    srv = subprocess.Popen(
        [args.flint, "serve", str(packs / "qwen7b-w4a16.pack"),
         "--port", "18081", "--unix", sock_path, "--max-seq-len", "8192"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        if wait_tcp("127.0.0.1", 18081, timeout=30.0):
            def unix_rpc(pr):
                payload = json.dumps({"prompt": pr, "max_tokens": 64, "temperature": 0.0, "ignore_eos": True}).encode()
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(sock_path)
                s.sendall(struct.pack("<I", len(payload)) + payload)
                hdr = s.recv(4, socket.MSG_WAITALL)
                sz = struct.unpack("<I", hdr)[0]
                data = s.recv(sz, socket.MSG_WAITALL)
                s.close()
                return json.loads(data)

            unix_rpc("Hello warmup")
            r_cold = unix_rpc(prompt_a)
            hit_ttfts = []
            r_hit = None
            for k in range(5):
                pk = shared_code + f"\n// Query {k+10}: implement a fast SIMD prefix sum variant {k} over uint32_t:\n"
                r_hit = unix_rpc(pk)
                hit_ttfts.append(r_hit["ttft_ms"])
            hit_ttfts.sort()
            r_hit["ttft_ms_p50"] = round(hit_ttfts[len(hit_ttfts) // 2], 2)
            rss, pd = proc_tree_rss_mib(srv.pid)
            report["w5"]["flint_unix_cold"] = r_cold
            report["w5"]["flint_unix_hit"] = r_hit
            report["w5"]["flint_serve_rss_mib"] = round(rss, 1)
            report["w5"]["flint_serve_pd_mib"] = round(pd, 1)
            print(f"  flint AF_UNIX cold: {r_cold['ttft_ms']:.2f} ms -> HIT_p50: {r_hit['ttft_ms_p50']:.2f} ms (cached={r_hit['prefix_cached_tokens']}/{r_hit['prompt_tokens']}, decode={r_hit['decode_tok_s']:.1f} tok/s)", flush=True)
    finally:
        srv.terminate()
        srv.wait(timeout=5)

    # W6 Cold CLI Start -> First Token
    print("\n--- W6: Cold CLI Process Start -> First Token (64 prompt -> 1 gen) ---", flush=True)
    short_p = "Write a high-performance C++23 function to intersect two sorted uint32_t arrays using AVX-512 intrinsics."
    w6_cmds = [
        ("flint w4a16.pack", [args.flint, "run", str(packs / "qwen7b-w4a16.pack"),
                              "-p", short_p, "--max-tokens", "1", "--greedy", "--ignore-eos"]),
    ]
    if has_llama and os.path.exists(str(packs / "qwen7b-q4_k_m.gguf")):
        w6_cmds.append(("llama-cli q4_k_m", [llama_cli, "-m", str(packs / "qwen7b-q4_k_m.gguf"),
                                             "-p", short_p, "-n", "1", "--temp", "0", "-ngl", "99", "--no-display-prompt"]))
    for label, cmd in w6_cmds:
        for _ in range(2):
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        report["w6"][label] = {"p50_s": round(ts[len(ts) // 2], 3), "p99_s": round(ts[-1], 3)}
        print(f"  {label:<20}: p50={ts[len(ts)//2]:.3f} s (p99={ts[-1]:.3f} s)", flush=True)

    # W7: Lossless Speculative Decoding
    print("\n--- W7: Lossless Speculative Decoding (W4A16, 256 gen) ---", flush=True)
    src_code = Path("src/kv_cache.cpp").read_text()
    edit_prompt = (
        "<|im_start|>system\nYou are a C++23 expert. Rewrite the file cleanly preserving all method signatures and logic.<|im_end|>\n"
        f"<|im_start|>user\nAdd a brief doc comment above every method in this file and output the complete file:\n```cpp\n{src_code}\n```<|im_end|>\n"
        "<|im_start|>assistant\n```cpp\n"
    )
    pf = Path("/tmp/flint_w7_prompt.txt")
    pf.write_text(edit_prompt)
    draft_pack = str(packs / "qwen05b-w4a16.pack")
    modes = ["none", "ngram"] + ([f"draft:{draft_pack}"] if os.path.exists(draft_pack) else [])
    ids_map = {}
    for mode in modes:
        tag = mode.split(":")[0]
        p = subprocess.run(
            [args.flint, "run", str(packs / "qwen7b-w4a16.pack"),
             "-f", str(pf), "--max-tokens", "256", "--greedy", "--ignore-eos",
             "--spec", mode, "--stats", "--json"],
            capture_output=True, text=True, check=True
        )
        j = json.loads(p.stdout.strip())
        ids_map[tag] = j["output_ids"]
        report["w7"][f"edit_spec_{tag}"] = {"decode_tok_s": j["decode_tok_s"], "stats_line": p.stderr.strip()}
        print(f"  edit  spec={tag:<5}: {p.stderr.strip()}", flush=True)
    for tag in ids_map:
        if tag != "none":
            ok = (ids_map[tag] == ids_map["none"])
            report["w7"][f"edit_lossless_{tag}"] = ok
            print(f"  W7 Edit Lossless (--spec {tag} == --spec none): {'PASS (256/256)' if ok else 'FAIL'}", flush=True)

    if os.path.exists(draft_pack):
        gf_prompt = (
            "<|im_start|>system\nYou are a C++23 systems engineer.<|im_end|>\n"
            "<|im_start|>user\nImplement a complete lock-free SPSC ring buffer in C++23 with cache-line padding and atomic acquire/release memory ordering.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        gf_ids = {}
        for mode in ["none", f"draft:{draft_pack}"]:
            tag = mode.split(":")[0]
            p = subprocess.run(
                [args.flint, "run", str(packs / "qwen7b-w4a16.pack"),
                 "-p", gf_prompt, "--max-tokens", "256", "--greedy", "--ignore-eos",
                 "--spec", mode, "--stats", "--json"],
                capture_output=True, text=True, check=True
            )
            j = json.loads(p.stdout.strip())
            gf_ids[tag] = j["output_ids"]
            report["w7"][f"greenfield_spec_{tag}"] = {"decode_tok_s": j["decode_tok_s"], "stats_line": p.stderr.strip()}
            print(f"  green spec={tag:<5}: {p.stderr.strip()}", flush=True)
        gok = (gf_ids["none"] == gf_ids["draft"])
        report["w7"]["greenfield_draft_lossless"] = gok
        print(f"  W7 Greenfield Lossless (--spec draft == --spec none): {'PASS (256/256)' if gok else 'FAIL'}", flush=True)

    # 2. Apples-to-Apples OpenAI HTTP/SSE Server Matrix (flint serve vs vLLM vs SGLang)
    # Use identical 17408 max context across all servers so seq=16384 + 256 gen fits cleanly!
    base_env = os.environ.copy()
    base_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    if os.path.exists("/workspace/bin/nvcc"):
        base_env["PATH"] = "/workspace/bin:/usr/local/cuda/bin:" + base_env.get("PATH", "")

    report["http_servers"]["flint_bf16"] = bench_http_server(
        "flint_bf16",
        [args.flint, "serve", str(packs / "qwen7b-bf16.pack"), "--port", "18090", "--unix", "/tmp/fl_bf16.sock", "--max-seq-len", "17408"],
        base_env, 18090, "qwen7b-bf16.pack", tok, warmup=args.warmup, runs=args.runs
    )
    report["http_servers"]["flint_fp8"] = bench_http_server(
        "flint_fp8",
        [args.flint, "serve", str(packs / "qwen7b-fp8.pack"), "--port", "18091", "--unix", "/tmp/fl_fp8.sock", "--max-seq-len", "17408"],
        base_env, 18091, "qwen7b-fp8.pack", tok, warmup=args.warmup, runs=args.runs
    )
    report["http_servers"]["flint_w4a16"] = bench_http_server(
        "flint_w4a16",
        [args.flint, "serve", str(packs / "qwen7b-w4a16.pack"), "--port", "18092", "--unix", "/tmp/fl_w4.sock", "--max-seq-len", "17408"],
        base_env, 18092, "qwen7b-w4a16.pack", tok, warmup=args.warmup, runs=args.runs
    )

    if args.vllm_python and os.path.exists(args.vllm_python):
        report["http_servers"]["vllm_bf16"] = bench_http_server(
            "vllm_bf16",
            [args.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
             "--model", args.model_dir, "--dtype", "bfloat16", "--max-model-len", "17408",
             "--enable-prefix-caching", "--gpu-memory-utilization", "0.90",
             "--host", "127.0.0.1", "--port", "18093"],
            base_env, 18093, args.model_dir, tok, warmup=args.warmup, runs=args.runs
        )
        report["http_servers"]["vllm_fp8"] = bench_http_server(
            "vllm_fp8",
            [args.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
             "--model", args.model_dir, "--dtype", "bfloat16", "--quantization", "fp8",
             "--max-model-len", "17408", "--enable-prefix-caching", "--gpu-memory-utilization", "0.90",
             "--host", "127.0.0.1", "--port", "18094"],
            base_env, 18094, args.model_dir, tok, warmup=args.warmup, runs=args.runs
        )

    if args.sglang_python and os.path.exists(args.sglang_python):
        report["http_servers"]["sglang_bf16"] = bench_http_server(
            "sglang_bf16",
            [args.sglang_python, "-m", "sglang.launch_server",
             "--model-path", args.model_dir, "--dtype", "bfloat16", "--context-length", "17408",
             "--attention-backend", "triton", "--mem-fraction-static", "0.85",
             "--host", "127.0.0.1", "--port", "18095"],
            base_env, 18095, args.model_dir, tok, warmup=args.warmup, runs=args.runs
        )
        report["http_servers"]["sglang_fp8"] = bench_http_server(
            "sglang_fp8",
            [args.sglang_python, "-m", "sglang.launch_server",
             "--model-path", args.model_dir, "--dtype", "bfloat16", "--quantization", "fp8",
             "--context-length", "17408", "--attention-backend", "triton", "--mem-fraction-static", "0.85",
             "--host", "127.0.0.1", "--port", "18096"],
            base_env, 18096, args.model_dir, tok, warmup=args.warmup, runs=args.runs
        )

    Path(args.json_out).write_text(json.dumps(report, indent=2))
    print(f"\nSaved unified v5 4-engine matrix JSON to {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
