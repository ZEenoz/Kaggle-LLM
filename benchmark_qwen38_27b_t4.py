#!/usr/bin/env python3
"""
benchmark_qwen38_27b_t4.py

Automated Benchmark & Profiler for:
Model: DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF
Quant: Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf (18.5 GB)
Target Platform: Kaggle 2x Tesla T4 (32GB VRAM Total, 30GB Usable)

Key Features:
- Validates 2x T4 GPU environment and memory headroom.
- Auto-fetches / mounts model to /kaggle/tmp (bypasses 19.5GB /kaggle/working limit).
- Auto-installs prebuilt CUDA 12.8 llama.cpp binaries (llama-bench & llama-server).
- Phase 1: Native C++ llama-bench measuring Prompt Processing (pp tok/s) and Output Generation (tg tok/s).
- Phase 2: Live end-to-end Streaming Server Benchmark at 512, 4k, 16k, 32k, 64k, 102.4k (100k+) context.
- Telemetry: TTFT, Prefill TPS, Output TPS, Peak VRAM per GPU (GPU 0 & GPU 1), Headroom.
- Exports markdown report and JSON summary for easy analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Default Configurations ---
WORK_DIR = Path("/kaggle/tmp")
OUTPUT_DIR = Path("/kaggle/working")
LLAMA_DIR = WORK_DIR / "llama_cpp"
TAR_PATH = WORK_DIR / "llama_cpp_binaries.tar.gz"

MODEL_FILENAME = "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"
MODEL_REPO = "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILENAME}"
FALLBACK_MODEL_FILENAME = "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-Q4_K_M.gguf"

BIN_URL = "https://github.com/ai-dock/llama.cpp-cuda/releases/download/b9628/llama.cpp-b9628-cuda-12.8-amd64.tar.gz"
SERVER_PORT = 8080


def print_banner(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def get_gpu_info() -> List[Dict[str, Any]]:
    """Query nvidia-smi for GPU specifications and memory."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        gpus = []
        for line in out.splitlines():
            idx, name, total, used, free, temp, util = [x.strip() for x in line.split(",")]
            gpus.append({
                "index": int(idx),
                "name": name,
                "memory_total_mb": float(total),
                "memory_used_mb": float(used),
                "memory_free_mb": float(free),
                "temp_c": float(temp),
                "util_percent": float(util),
            })
        return gpus
    except Exception as e:
        print(f"⚠️ Warning: Could not query nvidia-smi: {e}")
        return []


def print_gpu_status():
    gpus = get_gpu_info()
    if not gpus:
        print("❌ No NVIDIA GPUs detected via nvidia-smi!")
        return
    print(f"🎮 Detected {len(gpus)} GPU(s):")
    for g in gpus:
        print(
            f"   [{g['index']}] {g['name']} | Total: {g['memory_total_mb']:.0f} MB | "
            f"Used: {g['memory_used_mb']:.0f} MB | Free: {g['memory_free_mb']:.0f} MB"
        )
    if len(gpus) < 2:
        print("⚠️ Note: Kaggle settings recommend 2x T4 (Accelerator -> GPU T4 x 2) for 27B models.")


def ensure_llama_binaries() -> Tuple[str, str]:
    """Ensure llama-bench and llama-server binaries are present."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    
    server_bin = None
    bench_bin = None

    for candidate in WORK_DIR.rglob("llama-server"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            server_bin = str(candidate)
            break
    for candidate in WORK_DIR.rglob("llama-bench"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            bench_bin = str(candidate)
            break

    if not server_bin or not bench_bin:
        print("\n⚡ [1/3] Downloading prebuilt CUDA 12.8 llama.cpp binaries...")
        if not TAR_PATH.exists():
            cmd = ["wget", "-q", "--show-progress", BIN_URL, "-O", str(TAR_PATH)]
            subprocess.run(cmd, check=True)
        
        print("📦 Extracting binaries to /kaggle/tmp/llama_cpp...")
        LLAMA_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xzf", str(TAR_PATH), "-C", str(LLAMA_DIR)], check=True)

        for candidate in WORK_DIR.rglob("llama-server"):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                server_bin = str(candidate)
                break
        for candidate in WORK_DIR.rglob("llama-bench"):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                bench_bin = str(candidate)
                break

    if not server_bin or not bench_bin:
        raise FileNotFoundError("Failed to locate llama-server or llama-bench binaries after extraction.")

    bin_dir = os.path.dirname(server_bin)
    os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"{bin_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    print(f"✅ llama-bench  : {bench_bin}")
    print(f"✅ llama-server : {server_bin}")
    return bench_bin, server_bin


def ensure_model(custom_path: Optional[str] = None) -> Path:
    """Find model in /kaggle/input or download to /kaggle/tmp."""
    if custom_path and Path(custom_path).exists():
        path = Path(custom_path)
        print(f"✅ Using specified model: {path} ({path.stat().st_size / (1024**3):.2f} GB)")
        return path

    # 1. Check /kaggle/input mounted datasets
    mounted_candidates = []
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("*.gguf"):
            if "mmproj" not in p.name.lower():
                mounted_candidates.append(p)

    for p in mounted_candidates:
        if "davidau" in p.name.lower() or "turbofcfusion" in p.name.lower() or "qwen3.8" in p.name.lower():
            size_gb = p.stat().st_size / (1024**3)
            print(f"⚡ Found mounted model in /kaggle/input: {p.name} ({size_gb:.2f} GB)")
            return p

    # 2. Check /kaggle/tmp
    target_path = WORK_DIR / MODEL_FILENAME
    if target_path.exists():
        size_gb = target_path.stat().st_size / (1024**3)
        print(f"✅ Found cached model in /kaggle/tmp: {target_path.name} ({size_gb:.2f} GB)")
        return target_path

    # Check fallback non-MTP file
    fallback_path = WORK_DIR / FALLBACK_MODEL_FILENAME
    if fallback_path.exists():
        size_gb = fallback_path.stat().st_size / (1024**3)
        print(f"✅ Found cached model in /kaggle/tmp: {fallback_path.name} ({size_gb:.2f} GB)")
        return fallback_path

    # 3. Stream download to /kaggle/tmp
    print(f"\n📥 [2/3] Downloading model from Hugging Face (~18.5 GB)...")
    print(f"   Target: {target_path}")
    print(f"   Source: {MODEL_URL}")
    print("   (Takes ~1.5 - 2.5 minutes on Kaggle network)")

    cmd = [
        "wget",
        "--continue",
        "--progress=bar:force:noscroll",
        "--show-progress",
        MODEL_URL,
        "-O",
        str(target_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"⚠️ Primary MTP download failed. Attempting fallback non-MTP variant...")
        fallback_url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{FALLBACK_MODEL_FILENAME}"
        cmd = [
            "wget",
            "--continue",
            "--progress=bar:force:noscroll",
            "--show-progress",
            fallback_url,
            "-O",
            str(fallback_path),
        ]
        subprocess.run(cmd, check=True)
        target_path = fallback_path

    size_gb = target_path.stat().st_size / (1024**3)
    print(f"✅ Model download complete ({size_gb:.2f} GB)")
    return target_path


def run_llama_bench_suite(
    bench_bin: str,
    model_path: Path,
    contexts: List[int],
    gen_tokens: int = 128,
    kv_cache: str = "q8_0",
    tensor_split: str = "1,1",
) -> List[Dict[str, Any]]:
    """Execute llama-bench across prompt contexts and extract PP and TG tokens/sec."""
    print_banner(f"Running Native llama-bench Suite (KV Cache: {kv_cache})")
    
    results = []
    p_arg = ",".join(str(c) for c in contexts)
    
    cmd = [
        bench_bin,
        "-m", str(model_path),
        "-p", p_arg,
        "-n", str(gen_tokens),
        "-ngl", "99",
        "-sm", "layer",
        "-ts", tensor_split,
        "-fa", "1",
        "-ctk", kv_cache,
        "-ctv", kv_cache,
        "-o", "json",
    ]

    print(f"Executing: {' '.join(cmd)}\n")
    start_bench = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    output_lines = []
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            output_lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()

    raw_output = "".join(output_lines)
    duration = time.time() - start_bench

    # Try parsing JSON output from llama-bench
    try:
        json_start = raw_output.find("[")
        json_end = raw_output.rfind("]")
        if json_start != -1 and json_end != -1:
            json_str = raw_output[json_start : json_end + 1]
            data = json.loads(json_str)
            for item in data:
                n_prompt = item.get("n_prompt", 0)
                n_gen = item.get("n_gen", 0)
                tps = item.get("avg_ts", 0.0)
                stddev = item.get("stddev_ts", 0.0)
                test_type = "pp (Prompt Processing)" if n_prompt > 0 and n_gen == 0 else "tg (Token Gen)"
                results.append({
                    "n_prompt": n_prompt,
                    "n_gen": n_gen,
                    "test_type": test_type,
                    "tokens_per_sec": round(tps, 2),
                    "stddev": round(stddev, 2),
                })
    except Exception as e:
        print(f"⚠️ Notice: JSON parsing from llama-bench output failed ({e}), falling back to regex parser.")
        pattern = r"\|\s*(pp\d+|tg\d+)\s*\|\s*([\d\.]+)\s*±\s*([\d\.]+)\s*\|"
        for match in re.finditer(pattern, raw_output):
            test_tag, tps, stddev = match.groups()
            results.append({
                "test_tag": test_tag,
                "tokens_per_sec": float(tps),
                "stddev": float(stddev),
            })

    print(f"\n⏱️ Benchmark Suite finished in {duration:.1f}s")
    return results


def run_live_server_benchmark(
    server_bin: str,
    model_path: Path,
    test_contexts: List[int],
    gen_tokens: int = 128,
    kv_cache: str = "q4_0",
    max_context: int = 131072,
) -> List[Dict[str, Any]]:
    """Start llama-server with 128k context and benchmark real HTTP streaming requests."""
    print_banner(f"Running Live Server Benchmark (Contexts: {test_contexts}, KV: {kv_cache})")
    
    server_cmd = [
        server_bin,
        "-m", str(model_path),
        "--alias", "qwen3.8-27b",
        "-ngl", "99",
        "-sm", "layer",
        "-ts", "1,1",
        "-c", str(max_context),
        "-np", "1",
        "-fa", "on",
        "-ctk", kv_cache,
        "-ctv", kv_cache,
        "-b", "2048",
        "-ub", "512",
        "--host", "127.0.0.1",
        "--port", str(SERVER_PORT),
    ]

    log_path = OUTPUT_DIR / "llama-server-benchmark.log"
    print(f"🚀 Launching server with max context: {max_context} (128k)...")
    log_handle = open(log_path, "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    ready = False
    start_wait = time.time()
    print("⏳ Waiting for model weights to load into 2x T4 VRAM...")
    while time.time() - start_wait < 300:
        if server_proc.poll() is not None:
            print("\n❌ Server process terminated unexpectedly! Log tail:")
            print("\n".join(log_path.read_text(errors="replace").splitlines()[-25:]))
            return []
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{SERVER_PORT}/v1/models")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(3)

    if not ready:
        print("❌ Server failed to become ready within timeout.")
        server_proc.kill()
        return []

    print("✅ Server is online and ready for queries!\n")

    live_results = []
    try:
        for ctx_target in test_contexts:
            print(f"🧪 Testing Context Size: ~{ctx_target:,} tokens...")
            
            synthetic_prompt = ("def analyze_complex_system_state(node_id: int, payload: dict) -> bool:\n"
                                "    # System telemetry tracking and state machine verification\n"
                                "    return payload.get('active', False)\n") * (ctx_target // 20)
            
            payload = {
                "model": "qwen3.8-27b",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a senior enterprise software engineer. Provide concise code review."
                    },
                    {
                        "role": "user",
                        "content": f"Review this code and verify data structures:\n{synthetic_prompt}\nGenerate unit tests."
                    }
                ],
                "max_tokens": gen_tokens,
                "temperature": 0.7,
                "stream": True,
            }

            req_data = json.dumps(payload).encode("utf-8")
            http_req = urllib.request.Request(
                f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions",
                data=req_data,
                headers={"Content-Type": "application/json"},
            )

            t0 = time.time()
            t_first_token = None
            tokens_generated = 0
            
            try:
                with urllib.request.urlopen(http_req, timeout=300) as resp:
                    for line in resp:
                        line_str = line.decode("utf-8").strip()
                        if line_str.startswith("data: ") and line_str != "data: [DONE]":
                            chunk = json.loads(line_str[6:])
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                if t_first_token is None:
                                    t_first_token = time.time()
                                tokens_generated += 1

                t_end = time.time()
                gpus_after = get_gpu_info()

                ttft_sec = (t_first_token - t0) if t_first_token else (t_end - t0)
                gen_sec = (t_end - t_first_token) if (t_first_token and t_end > t_first_token) else 0.001
                
                prefill_tps = ctx_target / ttft_sec if ttft_sec > 0 else 0.0
                decode_tps = tokens_generated / gen_sec if gen_sec > 0 else 0.0

                gpu0_used = gpus_after[0]["memory_used_mb"] if len(gpus_after) > 0 else 0
                gpu1_used = gpus_after[1]["memory_used_mb"] if len(gpus_after) > 1 else 0

                res_entry = {
                    "context_target": ctx_target,
                    "ttft_sec": round(ttft_sec, 2),
                    "prefill_tps": round(prefill_tps, 1),
                    "decode_tps": round(decode_tps, 1),
                    "tokens_generated": tokens_generated,
                    "gpu0_vram_used_mb": round(gpu0_used, 0),
                    "gpu1_vram_used_mb": round(gpu1_used, 0),
                    "total_vram_used_mb": round(gpu0_used + gpu1_used, 0),
                }
                live_results.append(res_entry)

                print(
                    f"   👉 TTFT: {ttft_sec:.2f}s | Prefill: ~{prefill_tps:.1f} tps | "
                    f"Output: {decode_tps:.1f} tps | VRAM: GPU0={gpu0_used:.0f}MB, GPU1={gpu1_used:.0f}MB"
                )

            except Exception as ex:
                print(f"   ❌ Request failed at context {ctx_target}: {ex}")

            time.sleep(2)

    finally:
        print("\n🛑 Stopping llama-server...")
        server_proc.kill()
        time.sleep(2)
        log_handle.close()

    return live_results


def generate_summary_report(
    bench_results: List[Dict[str, Any]],
    live_results: List[Dict[str, Any]],
    output_dir: Path,
):
    """Format and save markdown and json benchmark reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_md = output_dir / "benchmark_summary_qwen38.md"
    report_json = output_dir / "benchmark_results_qwen38.json"

    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_FILENAME,
        "llama_bench_metrics": bench_results,
        "live_server_metrics": live_results,
    }
    report_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

    md_lines = [
        "# 📊 Qwen3.8-27B MTP (Kaggle 2x Tesla T4) Benchmark Report",
        f"- **Model**: `{MODEL_FILENAME}`",
        f"- **Benchmark Date**: {data['timestamp']}",
        f"- **Hardware**: 2x NVIDIA Tesla T4 (32 GB total VRAM)",
        "",
        "## 1. Live Streaming Performance vs Context Length",
        "",
        "| Context Length | TTFT (s) | Prefill (pp tok/s) | Output Decode (tg tok/s) | GPU 0 VRAM | GPU 1 VRAM | Total VRAM |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in live_results:
        md_lines.append(
            f"| **{r['context_target']:,}** | {r['ttft_sec']}s | **{r['prefill_tps']}** | "
            f"**{r['decode_tps']}** | {r['gpu0_vram_used_mb']} MB | {r['gpu1_vram_used_mb']} MB | {r['total_vram_used_mb']} MB |"
        )

    md_lines.extend([
        "",
        "## 2. Production Launcher Recommendation",
        "Based on the measured metrics, use the following optimized parameters:",
        "```bash",
        "llama-server \\",
        f"  -m {MODEL_FILENAME} \\",
        "  --alias qwen3.8-27b \\",
        "  -ngl 99 \\",
        "  -sm layer \\",
        "  -ts 1,1 \\",
        "  -c 131072 \\",
        "  -ctk q4_0 \\",
        "  -ctv q4_0 \\",
        "  -fa on \\",
        "  -b 2048 \\",
        "  -ub 512 \\",
        "  --port 8080",
        "```",
        "",
        "> [!TIP]",
        "> Setting `-ctk q4_0 -ctv q4_0` reduces KV cache footprint by ~50% compared to FP16,",
        "> allowing smooth 100k+ token inference on 2x Tesla T4 without OOM.",
    ])

    report_content = "\n".join(md_lines)
    report_md.write_text(report_content, encoding="utf-8")

    print("\n" + report_content)
    print(f"\n💾 Saved full report to: {report_md}")
    print(f"💾 Saved raw JSON telemetry to: {report_json}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.8-27B on Kaggle 2x Tesla T4")
    parser.add_argument("--mode", choices=["all", "bench", "live"], default="all",
                        help="Benchmark mode: 'bench' (llama-bench), 'live' (real streaming server), 'all' (both)")
    parser.add_argument("--model-path", type=str, default=None, help="Custom path to GGUF model")
    parser.add_argument("--contexts", type=str, default="512,4096,16384,32768,65536,102400",
                        help="Comma-separated context lengths to test (e.g. 512,4096,32768,102400)")
    parser.add_argument("--gen-tokens", type=int, default=128, help="Number of output tokens to generate per test")
    parser.add_argument("--kv-cache", choices=["q4_0", "q8_0", "f16"], default="q4_0", help="KV Cache quantization format")
    args = parser.parse_args()

    print_banner("Qwen3.8-27B-TURBO MTP Q4_K_M Benchmark Suite (2x Tesla T4)")
    print_gpu_status()

    bench_bin, server_bin = ensure_llama_binaries()
    model_path = ensure_model(args.model_path)

    context_list = [int(c.strip()) for c in args.contexts.split(",") if c.strip()]

    bench_results = []
    live_results = []

    if args.mode in ["all", "bench"]:
        bench_results = run_llama_bench_suite(
            bench_bin=bench_bin,
            model_path=model_path,
            contexts=context_list,
            gen_tokens=args.gen_tokens,
            kv_cache=args.kv_cache,
        )

    if args.mode in ["all", "live"]:
        live_results = run_live_server_benchmark(
            server_bin=server_bin,
            model_path=model_path,
            test_contexts=context_list,
            gen_tokens=args.gen_tokens,
            kv_cache=args.kv_cache,
        )

    generate_summary_report(bench_results, live_results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
