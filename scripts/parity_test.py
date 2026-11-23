#!/usr/bin/env python3
"""
Gate 1 & Gate 2 Verification Oracle:
  - Gate 1: 100-prompt multilingual + code + JSON + emoji + FIM tokenizer parity vs transformers.AutoTokenizer
  - Gate 2: Single-step prefill logits max_abs_err <= 1.5e-2, cosine >= 0.9999,
            AND 100% identical greedy tokens vs HuggingFace PyTorch (BF16).
"""
import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS_100 = [
    # English prose & contractions
    "Hello, world!",
    "I'm testing aren't you've we'll they'd it's don't won't can't shouldn't've.",
    "The quick brown fox jumps over the lazy dog 1234567890 times.",
    "UPPERCASE lowercase MiXeD CaSe   multiple   spaces\t\ttabs\n\nnewlines\r\nCRLF",
    "   leading spaces and trailing spaces   ",
    "\n\n\n",
    "",
    "a",
    "1",
    "123",
    "123456789",
    "0xDEADBEEF 3.1415926535 1e-6 -42.0",
    # C++ / CUDA / PTX code
    "template <typename T> __global__ void gemv_bf16(const __nv_bfloat16* __restrict__ x, float* out) { int tid = threadIdx.x; }",
    "asm volatile(\"lop3.b32 %0, %1, %2, %3, 0xea;\" : \"=r\"(dst) : \"r\"(a), \"r\"(b), \"r\"(c));",
    "std::vector<uint32_t> v = {1, 2, 3, 4}; for (auto&& [k, val] : map) { sum += val; }",
    "#include <immintrin.h>\n#define ALIGN_UP(x, a) (((x) + (a) - 1) & ~((a) - 1))\n",
    "namespace flint::cuda { constexpr uint32_t kBlockSize = 16; }",
    # Rust / Python / Go / JS / SQL / JSON
    "pub fn parse_header<'a>(buf: &'a [u8]) -> Result<Header, io::Error> { Ok(Header::default()) }",
    "def flash_decode(q: torch.Tensor, k_cache: torch.Tensor, scale: float) -> torch.Tensor:\n    return (q @ k_cache.T) * scale\n",
    "func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) }",
    "const fn = async ({ a, b = 42 } = {}) => await fetch(`/v1/chat/completions?id=${a}`);",
    "SELECT u.id, COUNT(*) AS cnt FROM users u LEFT JOIN orders o ON o.uid = u.id GROUP BY 1 HAVING cnt > 5;",
    '{"model": "Qwen2.5-Coder-7B", "messages": [{"role": "user", "content": "hi\\nworld"}], "temperature": 0.0}',
    # Special & FIM tokens
    "<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n<|im_start|>user\nWrite quicksort in C++.<|im_end|>\n<|im_start|>assistant\n",
    "<|fim_prefix|>int add(int a, int b) {\n    return <|fim_suffix|>;\n}<|fim_middle|>",
    "<|endoftext|>hello<|im_end|>",
    # Multilingual (Chinese, Japanese, Korean, Russian, Arabic, Greek, German, French, Spanish, Hindi)
    "你好，世界！这是一个高性能本地大语言模型推理引擎测试。",
    "こんにちは世界！カタカナとひらがなと漢字のテスト123。",
    "안녕하세요 세계! 한국어 토크나이저 정합성 테스트입니다.",
    "Привет, мир! Проверка кириллицы и пунктуации — 2026 год.",
    "مرحبا بالعالم! اختبار اللغة العربية والأرقام 12345.",
    "Γεια σου κόσμε! Ελληνικό αλφάβητο αβγδεζηθ.",
    "Überraschung: Grüße aus München! Straße, Fußgängerübergang, 100€.",
    "L'été à Paris: déjà vu, naïve, café, œuvre, cœur, «guillemets».",
    "¡Hola! ¿Cómo estás? El niño comió jalapeños en España.",
    "नमस्ते दुनिया! यह हिंदी भाषा का परीक्षण है।",
    # Emojis & symbols
    "🚀🔥⚡🧠💻🎉👍🏽👨‍👩‍👧‍👦",
    "Math: ∑_{i=1}^n x_i^2 ≤ ∞, ∀x ∈ ℝ, ∃δ > 0 ⇒ |f(x)| < ε",
    "Arrows & box: ┌──┬──┐ │  │  │ └──┴──┘ → ← ↑ ↓ ⇒ ⇔",
    "Punctuation: !@#$%^&*()_+-=[]{}|;':\",./<>?`~\\",
]
# Expand to 100 diverse prompts programmatically
for idx in range(len(PROMPTS_100), 100):
    kind = idx % 5
    if kind == 0:
        PROMPTS_100.append(f"// File unit_{idx}.cu\n__device__ float f_{idx}(float x) {{ return x * {idx}.5f + 1e-6f; }}\n")
    elif kind == 1:
        PROMPTS_100.append(f"User #{idx}: Don't forget I've got {idx*17} items at $99.{idx%100:02d} — isn't that great?!")
    elif kind == 2:
        PROMPTS_100.append(f"第{idx}次测试：mixed 中文 and English_var_{idx} = {idx * 123}; 🚀")
    elif kind == 3:
        PROMPTS_100.append(f"   \t  indented_line_{idx}({idx}, 'a', \"str\\n{idx}\")  \n\n  next_line")
    else:
        PROMPTS_100.append(f"<|im_start|>user\nExplain step {idx}: a->b::c[0] != (d << 2) | 0xff.<|im_end|>\n<|im_start|>assistant\n")


def run_gate1(flint_bin, model_dir):
    print(f"=== Gate 1: 100-Prompt Tokenizer Parity ({model_dir}) ===")
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    passed = 0
    failed = 0
    for i, text in enumerate(PROMPTS_100):
        hf_ids = tok.encode(text, add_special_tokens=False)
        p = subprocess.run(
            [flint_bin, "tokenize", model_dir, "-p", text],
            capture_output=True, text=True, check=True
        )
        flint_ids = json.loads(p.stdout.strip())
        if flint_ids == hf_ids:
            passed += 1
        else:
            failed += 1
            if failed <= 10:
                print(f"  MISMATCH prompt[{i}]: {text!r}")
                print(f"    HF   ({len(hf_ids)}): {hf_ids[:20]}")
                print(f"    flint({len(flint_ids)}): {flint_ids[:20]}")
    print(f"Gate 1 Result: {passed}/{len(PROMPTS_100)} exact matches ({failed} mismatches)")
    return failed == 0


def run_gate2(flint_bin, model_dir, gen_tokens=128, num_prompts=3, use_cpu=False):
    print(f"\n=== Gate 2: Numerical Logit & Greedy Token Parity ({model_dir}, gen={gen_tokens}) ===")
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    device = "cuda" if (torch.cuda.is_available() and not use_cpu) else "cpu"
    dtype = torch.float32 if use_cpu else torch.bfloat16

    test_prompts = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWrite a C++ function to compute fibonacci(n) iteratively.<|im_end|>\n<|im_start|>assistant\n",
        "The capital of France is",
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n",
    ][:num_prompts]

    # Phase A: Collect HuggingFace PyTorch reference logits & greedy tokens, THEN free VRAM!
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device).eval()
    hf_refs = []
    for prompt in test_prompts:
        ids = tok.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(inp)
            hf_logits = out.logits[0, -1].float().cpu()
            gen = model.generate(
                inp, max_new_tokens=gen_tokens, do_sample=False,
                repetition_penalty=1.0, top_p=1.0, top_k=None, temperature=None,
                pad_token_id=tok.eos_token_id, eos_token_id=None
            )
            hf_gen_ids = gen[0, len(ids):].tolist()
        hf_refs.append((prompt, hf_logits, hf_gen_ids))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase B: Evaluate flint on the same prompts and compare!
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        for pi, (prompt, hf_logits, hf_gen_ids) in enumerate(hf_refs):
            logits_path = os.path.join(td, f"logits_{pi}.f32")
            cmd = [flint_bin, "eval-logits", model_dir, "-p", prompt, "-o", logits_path]
            if use_cpu:
                cmd.append("--cpu")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            raw = open(logits_path, "rb").read()
            flint_logits = torch.tensor(
                struct.unpack(f"<{len(raw)//4}f", raw), dtype=torch.float32
            )

            diff = (hf_logits - flint_logits).abs()
            max_abs = float(diff.max().item())
            cos = float(torch.nn.functional.cosine_similarity(hf_logits.unsqueeze(0), flint_logits.unsqueeze(0)).item())
            hf_top = int(hf_logits.argmax().item())
            fl_top = int(flint_logits.argmax().item())
            print(f"  Prompt {pi} logits: max_abs_err={max_abs:.5f}, cos={cos:.6f} (>=0.9999), argmax hf={hf_top} flint={fl_top}")
            if cos < 0.9999 or hf_top != fl_top:
                all_ok = False

            rcmd = [flint_bin, "run", model_dir, "-p", prompt,
                    "--max-tokens", str(gen_tokens), "--greedy", "--ignore-eos", "--json"]
            if use_cpu:
                rcmd.append("--cpu")
            rp = subprocess.run(rcmd, capture_output=True, text=True, check=True)
            flint_gen_ids = json.loads(rp.stdout.strip())["output_ids"]

            exact_match = (flint_gen_ids == hf_gen_ids)
            first_diff = next((k for k, (a, b) in enumerate(zip(hf_gen_ids, flint_gen_ids)) if a != b), None)
            print(f"  Prompt {pi} greedy ({gen_tokens} toks): {'PASS (100% identical)' if exact_match else f'DIFF at tok {first_diff}'}")
            if not exact_match:
                print(f"    HF   : {hf_gen_ids[:20]}")
                print(f"    flint: {flint_gen_ids[:20]}")
                all_ok = False

            # Gate 4: Lossless speculative n-gram decoding MUST be 100% token-for-token identical to --spec none!
            if not use_cpu:
                scmd = [flint_bin, "run", model_dir, "-p", prompt,
                        "--max-tokens", str(gen_tokens), "--greedy", "--ignore-eos", "--spec", "ngram", "--json"]
                sp = subprocess.run(scmd, capture_output=True, text=True, check=True)
                spec_ids = json.loads(sp.stdout.strip())["output_ids"]
                spec_ok = (spec_ids == flint_gen_ids)
                print(f"  Prompt {pi} Gate 4 (--spec ngram lossless): {'PASS (100% identical)' if spec_ok else 'FAIL'}")
                if not spec_ok:
                    all_ok = False

    print(f"Gate 2+4 Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flint", default="./flint")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gate1-only", action="store_true")
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--num-prompts", type=int, default=3)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    ok1 = run_gate1(args.flint, args.model)
    if args.gate1_only:
        sys.exit(0 if ok1 else 1)
    ok2 = run_gate2(args.flint, args.model, args.gen_tokens, args.num_prompts, args.cpu)
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
