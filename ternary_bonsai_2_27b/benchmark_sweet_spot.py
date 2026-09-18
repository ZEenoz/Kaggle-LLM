#!/usr/bin/env python3
"""Sweet-spot discovery and benchmark harness for Ternary-Bonsai-2-27B (PQ2_0) on Kaggle 2x Tesla T4.

Sweeps context window from 64k to 262k with KV cache quantized to q8_0,
batch 1024, ubatch 512, row split mode (-sm row) across 2x T4 GPUs, and measures:
- Memory footprint (VRAM per GPU, Total VRAM, System RAM)
- Prompt processing / Prefill speed (Prompt tok/s) and Time-To-First-Token (TTFT)
- Generation / Output decode speed (Output tok/s)
- Maximum stable context window (Sweet Spot)

Outputs:
    results.json      Full benchmark dataset
    results.csv       Tidy long-format CSV
    sweet_spot.json   Recommended configuration & rationale
    summary.md        Human-readable summary report

Usage:
    python benchmark_sweet_spot.py
    python benchmark_sweet_spot.py --context-sizes 65536,131072,196608,262144
    python benchmark_sweet_spot.py --batch-size 1024 --ubatch-size 512 --kv-cache q8_0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# ============================================================================
# Defaults & Constants
# ============================================================================

DEFAULT_MODEL_REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf"
DEFAULT_MODEL_FILE = "Ternary-Bonsai-2-27B-PQ2_0.gguf"
DEFAULT_MODEL_URL = (
    "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/"
    "Ternary-Bonsai-2-27B-PQ2_0.gguf"
)
DEFAULT_MODEL_ALIAS = "ternary-bonsai-2-27b"
DEFAULT_API_KEY = "bonsai-bench-key"
DEFAULT_PORT = 8080

DEFAULT_KV_CACHE = "q8_0"
DEFAULT_BATCH_SIZE = 1024
DEFAULT_UBATCH_SIZE = 512

# Context sweep from 64k to 262k
DEFAULT_CONTEXT_SIZES = (65536, 131072, 196608, 262144)
DEFAULT_NGL = 99

# Sampling defaults
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_MIN_P = 0.0

DEFAULT_SERVER_READY_TIMEOUT_S = 900
DEFAULT_REQUEST_TIMEOUT_S = 1800
MEMORY_SAMPLE_INTERVAL_S = 0.5

# PrismML custom llama.cpp release URLs
PRISM_LLAMA_PRIMARY_URL = (
    "https://github.com/PrismML-Eng/llama.cpp/releases/download/"
    "prism-b10687-5d80cff/llama-prism-b10687-5d80cff-bin-linux-cuda-12.8-x64.tar.gz"
)
PRISM_LLAMA_FALLBACK_URL = (
    "https://github.com/PrismML-Eng/llama.cpp/releases/download/"
    "prism-b10687-5d80cff/llama-prism-b10687-5d80cff-bin-linux-cuda-12.4-x64.tar.gz"
)

USER_AGENT = "ternary-bonsai-benchmark/1.0"
FILLER_SENTENCE = (
    "The quick brown fox jumps over the lazy dog near the ancient stone tower on a misty autumn morning. "
)


@dataclass
class CompletionResult:
    ok: bool
    text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft_s: float | None = None
    elapsed_s: float = 0.0
    prompt_tps: float | None = None
    gen_tps: float | None = None
    error: str = ""

    @property
    def metrics(self) -> dict:
        return {
            "ok": self.ok,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_s": self.ttft_s,
            "elapsed_s": self.elapsed_s,
            "prompt_tps": self.prompt_tps,
            "gen_tps": self.gen_tps,
            "error": self.error,
        }


def log(msg: str) -> None:
    now = time.strftime("[%H:%M:%S]")
    print(f"{now} {msg}", flush=True)


def is_kaggle() -> bool:
    return Path("/kaggle").exists()


def default_workdir() -> Path:
    return Path("/kaggle/tmp/bonsai_bench" if is_kaggle() else Path.cwd() / "scratch" / "bonsai_bench")


def default_outdir() -> Path:
    return Path("/kaggle/working" if is_kaggle() else Path.cwd() / "bench_results" / "bonsai")


def stop_process(proc: subprocess.Popen | None, patterns: tuple[str, ...]) -> None:
    if shutil.which("pkill"):
        for pattern in patterns:
            subprocess.run(["pkill", "-9", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif os.name != "nt":
        for pattern in patterns:
            os.system(f"pkill -9 -f '{pattern}' >/dev/null 2>&1")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ============================================================================
# GPU & Environment Detection
# ============================================================================


def detect_gpus() -> list[dict]:
    """Detect NVIDIA GPUs using nvidia-smi, logging device names and memory."""
    gpus: list[dict] = []
    if not shutil.which("nvidia-smi"):
        log("nvidia-smi not found in PATH; GPU detection skipped.")
        return gpus
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                idx = int(parts[0])
                name = parts[1]
                total_mb = int(parts[2])
                gpus.append({"index": idx, "name": name, "total_mb": total_mb})
        log(f"Detected {len(gpus)} GPU(s):")
        for g in gpus:
            log(f"  [GPU {g['index']}] {g['name']} - {g['total_mb']} MiB ({g['total_mb'] / 1024:.2f} GiB)")
    except Exception as exc:
        log(f"Warning: GPU detection failed: {exc}")
    return gpus


# ============================================================================
# File Downloader with Resume & Progress
# ============================================================================


def download_file_with_resume(url: str, dest_path: Path) -> Path:
    """Download a file with aria2c, curl, wget, or urllib with resume support."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("aria2c"):
        log(f"Downloading with aria2c: {url} -> {dest_path}")
        subprocess.run(
            [
                "aria2c",
                "-c",
                "-x", "16",
                "-s", "16",
                "-k", "1M",
                "-d", str(dest_path.parent),
                "-o", dest_path.name,
                url,
            ],
            check=True,
        )
        return dest_path

    if shutil.which("curl"):
        log(f"Downloading with curl: {url} -> {dest_path}")
        subprocess.run(
            ["curl", "-L", "-C", "-", "-o", str(dest_path), url],
            check=True,
        )
        return dest_path

    if shutil.which("wget"):
        log(f"Downloading with wget: {url} -> {dest_path}")
        subprocess.run(
            ["wget", "-c", "-O", str(dest_path), url],
            check=True,
        )
        return dest_path

    # Fallback: urllib with chunked streaming
    log(f"Downloading with urllib: {url} -> {dest_path}")
    existing_bytes = dest_path.stat().st_size if dest_path.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            mode = "ab" if existing_bytes > 0 and resp.status == 206 else "wb"
            total = int(resp.headers.get("content-length", 0)) + (existing_bytes if mode == "ab" else 0)
            downloaded = existing_bytes if mode == "ab" else 0
            chunk_size = 1024 * 1024 * 8  # 8MB chunk
            t_last = time.time()
            with open(dest_path, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if time.time() - t_last > 5.0:
                        pct = (downloaded / total * 100) if total else 0
                        log(f"  Downloaded: {downloaded / 1024**3:.2f} GB ({pct:.1f}%)")
                        t_last = time.time()
    except urllib.error.HTTPError as err:
        if err.code == 416:
            # Requested Range Not Satisfiable -> file already complete
            log(f"File already completely downloaded ({existing_bytes / 1024**3:.2f} GB)")
            return dest_path
        raise

    return dest_path


# ============================================================================
# Model & PrismML llama.cpp Installation
# ============================================================================


def get_or_download_model(repo: str, filename: str, model_dir: Path) -> Path:
    """Download GGUF or find it mounted on Kaggle dataset."""
    model_dir.mkdir(parents=True, exist_ok=True)

    if Path("/kaggle/input").exists():
        mounted = [p for p in Path("/kaggle/input").rglob("*.gguf") if "mmproj" not in p.name.lower()]
        for m in mounted:
            name_lower = m.name.lower()
            if "bonsai" in name_lower or "ternary" in name_lower:
                log(f"Using mounted dataset model: {m.name} ({m.stat().st_size / 1024**3:.2f} GB)")
                return m

    target = model_dir / filename
    if target.exists() and target.stat().st_size > 1024**3:
        log(f"Using cached model: {target} ({target.stat().st_size / 1024**3:.2f} GB)")
        return target

    log(f"Downloading {repo}/{filename} (~7.21 GB)...")
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(model_dir))
        return Path(downloaded)
    except Exception as exc:
        log(f"hf_hub_download failed or not installed ({exc}); using direct download with resume")
        download_file_with_resume(url, target)

    log(f"Model ready: {target} ({target.stat().st_size / 1024**3:.2f} GB)")
    return target


def install_prism_llamacpp(workdir: Path) -> Path:
    """Download prebuilt PrismML llama.cpp (sm_75 CUDA 12.8 / 12.4) into workdir."""
    bin_dir = workdir / "bin"
    server_bin = bin_dir / "llama-server"
    if server_bin.exists() and os.access(server_bin, os.X_OK):
        log(f"PrismML llama.cpp already installed at {bin_dir}")
        return bin_dir

    found = [p for p in workdir.rglob("llama-server") if p.is_file() and os.access(p, os.X_OK)]
    if found:
        bin_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found[0], server_bin)
        return bin_dir

    log("Downloading prebuilt PrismML llama.cpp (CUDA 12.8 / fallback 12.4)...")
    urls_to_try = [
        (PRISM_LLAMA_PRIMARY_URL, "llama-prism-cuda-12.8.tar.gz"),
        (PRISM_LLAMA_FALLBACK_URL, "llama-prism-cuda-12.4.tar.gz"),
    ]

    archive_path = None
    for url, filename in urls_to_try:
        dest = workdir / filename
        try:
            log(f"Trying download from {url}...")
            download_file_with_resume(url, dest)
            archive_path = dest
            break
        except Exception as exc:
            log(f"Download failed from {url}: {exc}")

    if not archive_path or not archive_path.exists():
        raise RuntimeError("Failed to download PrismML llama.cpp binaries from primary and fallback URLs.")

    log(f"Extracting {archive_path.name}...")
    try:
        with tarfile.open(archive_path, "r:gz") as tf:
            try:
                tf.extractall(workdir, filter="data")
            except TypeError:
                tf.extractall(workdir)
    except Exception as exc:
        log(f"tarfile extraction notice: {exc}; using system tar CLI...")
        subprocess.run(["tar", "-xzf", str(archive_path), "-C", str(workdir)], check=True)

    found = [p for p in workdir.rglob("llama-server") if p.is_file()]
    if not found:
        raise RuntimeError("llama-server executable not found after extraction")

    real = found[0]
    try:
        real.chmod(real.stat().st_mode | 0o755)
    except Exception:
        pass

    bin_dir.mkdir(parents=True, exist_ok=True)
    if not server_bin.exists():
        try:
            server_bin.symlink_to(real)
        except Exception:
            shutil.copy2(real, server_bin)

    log(f"PrismML llama.cpp ready: {bin_dir}")
    return bin_dir


def so_dirs_from(workdir: Path) -> str:
    dirs = {str(p.parent.resolve()) for p in workdir.rglob("*.so*") if p.is_file()}
    return ":".join(dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")


# ============================================================================
# Memory Telemetry
# ============================================================================


class MemorySampler(threading.Thread):
    """Samples per-GPU VRAM and system RAM in the background."""

    def __init__(self, interval_s: float = MEMORY_SAMPLE_INTERVAL_S) -> None:
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.gpu_used_mb: list[int] = []
        self.gpu_total_mb: list[int] = []
        self.peak_gpu_mb: int = 0
        self.peak_per_gpu_mb: list[int] = []
        self.peak_ram_gb: float = 0.0
        self.total_gpu_mb: int = 0
        self._stop_event = threading.Event()

    def snapshot(self) -> tuple[int, float]:
        gpu_mb, ram_gb = 0, 0.0
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                text=True,
                timeout=10,
            )
            current_gpus: list[int] = []
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    used = int(parts[1])
                    total = int(parts[2])
                    current_gpus.append(used)
                    gpu_mb += used
                    if not self.gpu_total_mb:
                        self.total_gpu_mb += total

            if current_gpus:
                if not self.peak_per_gpu_mb:
                    self.peak_per_gpu_mb = [0] * len(current_gpus)
                for i, u in enumerate(current_gpus):
                    self.peak_per_gpu_mb[i] = max(self.peak_per_gpu_mb[i], u)

            self.gpu_used_mb = current_gpus
        except Exception:
            pass

        try:
            meminfo_path = Path("/proc/meminfo")
            if meminfo_path.exists():
                info = {}
                for line in meminfo_path.read_text().splitlines():
                    if ":" in line:
                        key, val = line.split(":", 1)
                        info[key.strip()] = int(val.strip().split()[0])
                ram_gb = (info.get("MemTotal", 0) - info.get("MemAvailable", 0)) / 1024**2
        except Exception:
            pass

        self.peak_gpu_mb = max(self.peak_gpu_mb, gpu_mb)
        self.peak_ram_gb = max(self.peak_ram_gb, ram_gb)
        return gpu_mb, ram_gb

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.snapshot()
            self._stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=5)


# ============================================================================
# llama-server Lifecycle & Client
# ============================================================================


def build_server_cmd(
    server_bin: Path,
    model_path: Path,
    args: argparse.Namespace,
    context: int,
    ngl: int,
) -> list[str]:
    """Constructs llama-server arguments with -sm row and -ts 1,1 for Hadamard/Ternary distribution."""
    cmd = [
        str(server_bin),
        "-m", str(model_path),
        "-ngl", str(ngl),
        "-sm", "row",
        "-ts", "1,1",
        "-c", str(context),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "-np", "1",
        "--cache-type-k", args.kv_cache,
        "--cache-type-v", args.kv_cache,
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--alias", args.model_alias,
        "--api-key", args.api_key,
    ]
    return cmd


def start_server(
    bin_dir: Path,
    model_path: Path,
    args: argparse.Namespace,
    context: int,
    ngl: int,
    log_path: Path,
) -> subprocess.Popen | None:
    stop_process(None, ("[l]lama-server",))
    server_binary = bin_dir / "llama-server"
    cmd = build_server_cmd(server_binary, model_path, args, context, ngl)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = so_dirs_from(bin_dir.parent)
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["GGML_CUDA_NO_VMM"] = "1"

    try:
        help_text = subprocess.run(
            [str(server_binary), "--help"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        ).stdout or ""
    except Exception:
        help_text = ""

    if "--flash-attn" in help_text:
        cmd.extend(["--flash-attn", "on"])
    else:
        cmd.extend(["-fa", "on"])

    if "-fit" in help_text:
        cmd.extend(["-fit", "off"])
    elif "--fit" in help_text:
        cmd.extend(["--fit", "off"])

    log(
        f"Starting llama-server: ctx={context:,}, ngl={ngl}, "
        f"batch={args.batch_size}/{args.ubatch_size}, kv={args.kv_cache}, split=row (1,1)..."
    )
    server_log = open(log_path, "w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT, env=env, text=True)
    return proc


def wait_for_server(args: argparse.Namespace, proc: subprocess.Popen | None, timeout: int) -> bool:
    """Polls /health and /v1/models until llama-server is completely ready."""
    if proc is None:
        return False
    health_url = f"http://127.0.0.1:{args.port}/health"
    models_url = f"http://127.0.0.1:{args.port}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return False

        # Check /health endpoint
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    try:
                        data = json.loads(resp.read().decode())
                        if data.get("status") in ("ok", "ready"):
                            return True
                    except Exception:
                        return True
        except Exception:
            pass

        # Check /v1/models endpoint as well
        try:
            req = urllib.request.Request(models_url, headers={"Authorization": f"Bearer {args.api_key}"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass

        time.sleep(2)
    return False


def server_log_tail(log_path: Path, lines: int = 30) -> str:
    try:
        return "\n".join(log_path.read_text(errors="replace").splitlines()[-lines:])
    except Exception:
        return ""


def stream_completion(
    args: argparse.Namespace,
    prompt: str,
    max_tokens: int = 128,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
) -> CompletionResult:
    """Executes a streaming chat completion request and records TTFT and token generation speed."""
    temp = args.temperature if temperature is None else temperature
    tp = args.top_p if top_p is None else top_p
    tk = args.top_k if top_k is None else top_k
    mp = args.min_p if min_p is None else min_p

    payload = {
        "model": args.model_alias,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temp,
        "top_p": tp,
        "top_k": tk,
        "min_p": mp,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
            "Accept": "text/event-stream",
        },
    )
    result = CompletionResult(ok=False)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=args.request_timeout_s) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[len("data:"):].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        if result.ttft_s is None:
                            result.ttft_s = time.time() - t0
                        result.text += piece
                usage = chunk.get("usage")
                if usage:
                    result.prompt_tokens = usage.get("prompt_tokens")
                    result.completion_tokens = usage.get("completion_tokens")
        result.elapsed_s = time.time() - t0
        result.ok = True
    except Exception as exc:
        result.elapsed_s = time.time() - t0
        result.error = str(exc)

    if result.ok:
        if not result.completion_tokens and result.text:
            result.completion_tokens = max(len(result.text.split()), len(result.text) // 4)

        if result.ttft_s is not None:
            t_gen = result.elapsed_s - result.ttft_s
            if t_gen > 0 and result.completion_tokens:
                result.gen_tps = result.completion_tokens / t_gen
            if result.prompt_tokens and result.ttft_s > 0:
                result.prompt_tps = result.prompt_tokens / result.ttft_s
    return result


def build_fill_prompt(target_tokens: int, suffix: str) -> str:
    """Generates synthetic prompt to push KV context depth deterministically."""
    target_chars = max(target_tokens * 4, 256)
    filler = (FILLER_SENTENCE * (target_chars // len(FILLER_SENTENCE) + 1))[:target_chars]
    return filler + "\n" + suffix


def fmt(value: float | None, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


# ============================================================================
# Sweet-Spot Benchmark Runner
# ============================================================================


def run_sweet_spot_sweep(
    args: argparse.Namespace,
    model_path: Path,
    bin_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    log("=" * 70)
    log("Starting Ternary-Bonsai-2-27B Context Sweet-Spot Sweep (64k-262k)")
    log(
        f"Config: KV={args.kv_cache}, Batch={args.batch_size}, UBatch={args.ubatch_size}, "
        f"NGL={args.ngl}, Split=row (1,1)"
    )
    log("=" * 70)

    for context in args.context_sizes:
        log_path = args.workdir / f"server_ctx{context}_ngl{args.ngl}.log"
        row: dict = {
            "phase": "context_sweep",
            "context": context,
            "ngl": args.ngl,
            "kv_cache": args.kv_cache,
            "batch": args.batch_size,
            "ubatch": args.ubatch_size,
            "fit": False,
            "error": "",
            "idle_vram_mb": 0,
            "peak_vram_mb": 0,
            "peak_vram_gb": 0.0,
            "vram_per_gpu_mb": [],
            "idle_ram_gb": 0.0,
            "peak_ram_gb": 0.0,
            "short_request": {},
            "long_request": {},
        }

        sampler = MemorySampler()
        proc = start_server(bin_dir, model_path, args, context, args.ngl, log_path)

        if proc is None or not wait_for_server(args, proc, args.server_ready_timeout_s):
            row["error"] = "Server failed to initialize (CUDA OOM?)\n" + server_log_tail(log_path, 15)
            log(f"  ❌ ctx={context:,}: Failed to initialize server.")
            stop_process(proc, ("[l]lama-server",))
            rows.append(row)
            continue

        sampler.start()
        idle_gpu_mb, idle_ram_gb = sampler.snapshot()
        row["idle_vram_mb"] = idle_gpu_mb
        row["idle_ram_gb"] = idle_ram_gb

        gpu0_mb = sampler.gpu_used_mb[0] if sampler.gpu_used_mb else 0
        gpu1_mb = sampler.gpu_used_mb[1] if len(sampler.gpu_used_mb) > 1 else 0
        log(
            f"  🟢 ctx={context:,}: Server live. Idle VRAM: {idle_gpu_mb / 1024:.2f} GB "
            f"(GPU 0: {gpu0_mb} MB, GPU 1: {gpu1_mb} MB)"
        )

        # 1. Warmup request
        warmup = stream_completion(args, "Respond with 'Ready'.", max_tokens=8, temperature=0.0)
        if not warmup.ok:
            row["error"] = "Warmup failed: " + warmup.error
            log(f"  ⚠️ ctx={context:,}: Warmup error: {warmup.error}")
            sampler.stop()
            stop_process(proc, ("[l]lama-server",))
            rows.append(row)
            continue
        time.sleep(1.0)

        # 2. Short request (baseline decode speed)
        short_prompt = (
            "Explain in 3 bullet points why ternary quantization (1.58-bit / PQ2_0) "
            "enables ultra-long context reasoning."
        )
        short = stream_completion(args, short_prompt, max_tokens=128)
        row["short_request"] = short.metrics

        # 3. Long context probe request (push context depth to ~80%)
        long_target_tokens = max(1024, min(int(context * 0.80), context - 512))
        long_prompt = build_fill_prompt(
            long_target_tokens,
            "Based on the text above, who jumps over the lazy dog? Answer in one short sentence.",
        )
        log(f"  ⏳ ctx={context:,}: Running long prompt probe (~{long_target_tokens:,} tokens, ~80% ctx)...")
        long = stream_completion(args, long_prompt, max_tokens=128)
        row["long_request"] = long.metrics

        sampler.stop()
        stop_process(proc, ("[l]lama-server",))

        row.update({
            "fit": bool(short.ok and long.ok),
            "peak_vram_mb": sampler.peak_gpu_mb,
            "peak_vram_gb": round(sampler.peak_gpu_mb / 1024, 2),
            "vram_per_gpu_mb": sampler.peak_per_gpu_mb,
            "peak_ram_gb": round(sampler.peak_ram_gb, 2),
        })
        if not short.ok or not long.ok:
            row["error"] = f"Short err: {short.error} | Long err: {long.error}".strip(" |")

        log(
            f"  📊 ctx={context:>7,d} | Fit={row['fit']} | "
            f"Short Gen: {fmt(short.gen_tps)} tok/s | "
            f"Long Prefill: {fmt(long.prompt_tps)} tok/s | "
            f"Long TTFT: {fmt(long.ttft_s)} s | "
            f"Long Gen: {fmt(long.gen_tps)} tok/s | "
            f"Peak VRAM: {row['peak_vram_gb']} GB"
        )

        rows.append(row)

    return rows


def synthesize_sweet_spot(rows: list[dict], total_vram_limit_gb: float = 30.0) -> dict:
    """Finds optimal context configuration balancing stability, context depth, and generation TPS."""
    fit_rows = [r for r in rows if r.get("fit")]
    if not fit_rows:
        return {
            "recommended": None,
            "reason": "No context window successfully completed both short and long inference without OOM.",
            "candidates": rows,
        }

    # Safety headroom: peak VRAM must be <= 29.5 GB on 2x T4
    safe_rows = [r for r in fit_rows if r.get("peak_vram_gb", 99.0) <= 29.5]
    pool = safe_rows or fit_rows

    def score(r: dict) -> float:
        ctx = r["context"]
        gen_tps = (r.get("long_request") or {}).get("gen_tps") or (r.get("short_request") or {}).get("gen_tps") or 1.0
        vram_headroom_gb = total_vram_limit_gb - r.get("peak_vram_gb", 28.0)
        return (ctx / 1000.0) * 1.5 + gen_tps * 2.0 + min(vram_headroom_gb, 2.0) * 3.0

    ranked = sorted(pool, key=score, reverse=True)
    best = ranked[0]

    return {
        "recommended": {
            "context": best["context"],
            "ngl": best["ngl"],
            "kv_cache": best["kv_cache"],
            "batch_size": best["batch"],
            "ubatch_size": best["ubatch"],
            "short_gen_tps": (best.get("short_request") or {}).get("gen_tps"),
            "long_prefill_tps": (best.get("long_request") or {}).get("prompt_tps"),
            "long_ttft_s": (best.get("long_request") or {}).get("ttft_s"),
            "long_gen_tps": (best.get("long_request") or {}).get("gen_tps"),
            "peak_vram_gb": best.get("peak_vram_gb"),
            "score": round(score(best), 2),
        },
        "reason": (
            f"Context {best['context']:,} tokens achieves peak VRAM {best.get('peak_vram_gb')} GB "
            f"with generation speed {fmt((best.get('long_request') or {}).get('gen_tps'))} tok/s "
            f"and long prefill {fmt((best.get('long_request') or {}).get('prompt_tps'))} tok/s "
            f"safely within Kaggle 2x T4 limits."
        ),
        "all_ranked": [
            {
                "context": r["context"],
                "peak_vram_gb": r.get("peak_vram_gb"),
                "short_tps": (r.get("short_request") or {}).get("gen_tps"),
                "long_tps": (r.get("long_request") or {}).get("gen_tps"),
                "score": round(score(r), 2),
            }
            for r in ranked
        ],
    }


def write_benchmark_outputs(out_dir: Path, results: dict, sweet_spot: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. results.json
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # 2. sweet_spot.json
    (out_dir / "sweet_spot.json").write_text(json.dumps(sweet_spot, indent=2), encoding="utf-8")

    # 3. results.csv
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "context", "fit", "peak_vram_gb", "peak_ram_gb",
            "short_gen_tps", "long_prompt_tps", "long_ttft_s", "long_gen_tps", "error",
        ])
        for r in results.get("context_sweep", []):
            short = r.get("short_request") or {}
            long = r.get("long_request") or {}
            writer.writerow([
                r.get("context"),
                r.get("fit"),
                r.get("peak_vram_gb"),
                r.get("peak_ram_gb"),
                fmt(short.get("gen_tps")),
                fmt(long.get("prompt_tps")),
                fmt(long.get("ttft_s")),
                fmt(long.get("gen_tps")),
                r.get("error", "").replace("\n", " ")[:60],
            ])

    # 4. summary.md
    md = [
        "# Ternary-Bonsai-2-27B (PQ2_0) 2x Tesla T4 Sweet-Spot Benchmark",
        "",
        "**Hardware**: Kaggle 2x Nvidia Tesla T4 (32GB Total VRAM)",
        f"**Model**: `{DEFAULT_MODEL_REPO}` (`{DEFAULT_MODEL_FILE}`)",
        f"**KV Cache**: `{DEFAULT_KV_CACHE}` | **Batch**: `{DEFAULT_BATCH_SIZE}` | **UBatch**: `{DEFAULT_UBATCH_SIZE}`",
        "**Split Mode**: `-sm row` | **Tensor Split**: `-ts 1,1` | **Flash Attention**: `-fa on`",
        "",
        "## 📊 Context Scaling & Performance Results",
        "",
        "| Context Window | Fit | Peak VRAM (GB) | Short Gen tok/s | Long Prefill tok/s | Long TTFT (s) | Long Gen tok/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results.get("context_sweep", []):
        short = r.get("short_request") or {}
        long = r.get("long_request") or {}
        fit_symbol = "✅ Pass" if r.get("fit") else "❌ Fail"
        md.append(
            f"| {r.get('context'):,d} | {fit_symbol} | {r.get('peak_vram_gb')} | "
            f"{fmt(short.get('gen_tps'))} | {fmt(long.get('prompt_tps'))} | "
            f"{fmt(long.get('ttft_s'))} | {fmt(long.get('gen_tps'))} |"
        )

    rec = sweet_spot.get("recommended")
    md += ["", "## 🎯 Recommended Sweet Spot", ""]
    if rec:
        md.append(f"- **Optimal Context Size**: `{rec['context']:,}` tokens")
        md.append(f"- **KV Cache Quantization**: `{rec['kv_cache']}`")
        md.append(f"- **Batch / UBatch**: `{rec['batch_size']}` / `{rec['ubatch_size']}`")
        md.append(f"- **Peak VRAM**: `{rec['peak_vram_gb']} GB` (fits comfortably within 2x T4 budget)")
        md.append(f"- **Decode Generation Speed**: `{fmt(rec['long_gen_tps'])} tok/s`")
        md.append(
            f"- **Prefill Processing Speed**: `{fmt(rec['long_prefill_tps'])} tok/s` "
            f"(TTFT: `{fmt(rec['long_ttft_s'])} s`)"
        )
        md.append("")
        md.append(f"**Rationale**: {sweet_spot.get('reason')}")
    else:
        md.append("No viable configuration found without OOM. See `results.json` for details.")

    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    log(f"All benchmark reports successfully generated in {out_dir}")


# ============================================================================
# Main Entry Point
# ============================================================================


def strip_kernel_args(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "-f":
            skip_next = True
            continue
        cleaned.append(arg)
    return cleaned


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    argv = strip_kernel_args(raw_argv)
    parser = argparse.ArgumentParser(
        description="Benchmark Ternary-Bonsai-2-27B sweet-spot (64k-262k context) on Kaggle 2x Tesla T4"
    )
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO, help="Hugging Face repository ID")
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE, help="GGUF weight filename")
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS, help="Model alias for API requests")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key for server access")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for llama-server")
    parser.add_argument(
        "--kv-cache",
        default=DEFAULT_KV_CACHE,
        help="KV cache quantization format (default: q8_0)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Logical batch size")
    parser.add_argument("--ubatch-size", type=int, default=DEFAULT_UBATCH_SIZE, help="Physical micro-batch size")
    parser.add_argument("--ngl", type=int, default=DEFAULT_NGL, help="GPU layers offloaded (default: 99)")
    parser.add_argument(
        "--context-sizes",
        default=",".join(str(c) for c in DEFAULT_CONTEXT_SIZES),
        help="Comma-separated context sizes to sweep (default: 65536,131072,196608,262144)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Sampling top-p (default: 0.95)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Sampling top-k (default: 20)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=DEFAULT_MIN_P,
        help="Sampling min-p (default: 0.0)",
    )
    parser.add_argument(
        "--server-ready-timeout-s",
        type=int,
        default=DEFAULT_SERVER_READY_TIMEOUT_S,
        help="Timeout waiting for server startup in seconds",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="Inference request timeout in seconds",
    )
    parser.add_argument("--workdir", default=None, help="Working/scratch directory")
    parser.add_argument("--out-dir", default=None, help="Output directory for reports")
    args = parser.parse_args(argv)

    args.context_sizes = [int(v.strip()) for v in args.context_sizes.split(",") if v.strip()]
    args.workdir = Path(args.workdir) if args.workdir else default_workdir()
    args.out_dir = Path(args.out_dir) if args.out_dir else default_outdir()
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    stop_process(None, ("[l]lama-server",))
    args.workdir.mkdir(parents=True, exist_ok=True)

    log("Detecting GPU environment...")
    detect_gpus()

    log("Provisioning model weights and PrismML llama.cpp CUDA binaries...")
    model_path = get_or_download_model(args.model_repo, args.model_file, args.workdir / "models")
    bin_dir = install_prism_llamacpp(args.workdir)

    results: dict = {
        "meta": {
            "model_repo": args.model_repo,
            "model_file": args.model_file,
            "context_sizes": args.context_sizes,
            "ngl": args.ngl,
            "kv_cache": args.kv_cache,
            "batch": args.batch_size,
            "ubatch": args.ubatch_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "split_mode": "row",
            "tensor_split": "1,1",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "context_sweep": run_sweet_spot_sweep(args, model_path, bin_dir),
    }

    stop_process(None, ("[l]lama-server",))

    sweet_spot = synthesize_sweet_spot(results["context_sweep"])
    results["sweet_spot"] = sweet_spot

    write_benchmark_outputs(args.out_dir, results, sweet_spot)

    rec = sweet_spot.get("recommended")
    if rec:
        log("=" * 70)
        log(
            f"🏆 SWEET SPOT IDENTIFIED -> Context={rec['context']:,} tokens | "
            f"VRAM={rec['peak_vram_gb']} GB | Long Gen={fmt(rec['long_gen_tps'])} tok/s | "
            f"Prefill={fmt(rec['long_prefill_tps'])} tok/s"
        )
        log("=" * 70)
    else:
        log("No optimal configuration identified.")

    return results


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (0, None):
            stop_process(None, ("[l]lama-server",))
            sys.exit(e.code)
    except KeyboardInterrupt:
        stop_process(None, ("[l]lama-server",))
        sys.exit(130)
    finally:
        stop_process(None, ("[l]lama-server",))
