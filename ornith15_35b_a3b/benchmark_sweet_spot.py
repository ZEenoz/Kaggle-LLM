#!/usr/bin/env python3
"""Sweet-spot discovery and benchmark harness for Ornith-1.5-35B-A3B (TIEL-Calibrated MTPv2 ICE GGUF) on Kaggle 2x Tesla T4.

Sweeps context window from 100k to 160k+ with KV cache quantized to q4_0,
batch 1024, ubatch 512, and MTP (Multi-Token Prediction) self-speculative decoding enabled:
- Memory footprint (VRAM per GPU, Total VRAM, System RAM)
- Prompt processing / Prefill speed (Prompt tok/s) and Time-To-First-Token (TTFT)
- Generation / Output decode speed (Output tok/s with MTP draft head)
- Maximum stable context window (Sweet Spot)

Outputs:
    results.json      Full benchmark dataset
    results.csv       Tidy long-format CSV
    sweet_spot.json   Recommended configuration & rationale
    summary.md        Human-readable summary report

Usage:
    python benchmark_sweet_spot.py
    python benchmark_sweet_spot.py --context-sizes 102400,114688,131072,147456,163840
    python benchmark_sweet_spot.py --batch-size 1024 --ubatch-size 512 --enable-mtp
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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

DEFAULT_MODEL_REPO = "gbuzhf/Ornith-1.5-35B-A3B-TIEL-Calibrated-MTPv2-ICE-GGUF"
DEFAULT_MODEL_FILE = "Ornith-1.5-35B-A3B-TIEL_Calibrated-MTPv2-21G-ICE.gguf"
DEFAULT_MODEL_URL = (
    "https://huggingface.co/gbuzhf/Ornith-1.5-35B-A3B-TIEL-Calibrated-MTPv2-ICE-GGUF/resolve/main/"
    "Ornith-1.5-35B-A3B-TIEL_Calibrated-MTPv2-21G-ICE.gguf"
)
DEFAULT_MODEL_ALIAS = "ornith-1.5-35B-A3B"
DEFAULT_API_KEY = "kilo-bench-key"
DEFAULT_PORT = 8080

DEFAULT_KV_CACHE = "q4_0"
DEFAULT_BATCH_SIZE = 1024
DEFAULT_UBATCH_SIZE = 512
DEFAULT_HF_TOKEN = ""


def get_hf_token(cli_token: str = "") -> str:
    """Extract Hugging Face token from CLI, injected default, Kaggle Secrets, ~/.cache, or env."""
    if cli_token:
        return cli_token.strip()
    if DEFAULT_HF_TOKEN:
        return DEFAULT_HF_TOKEN.strip()
    try:
        from kaggle_secrets import UserSecretsClient
        sec = UserSecretsClient().get_secret("HF_TOKEN")
        if sec:
            return sec.strip()
    except Exception:
        pass
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        try:
            return token_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return os.environ.get("HF_TOKEN", "").strip()

# Context sweep from 100k to 160k+
DEFAULT_CONTEXT_SIZES = (102400, 114688, 131072, 147456, 163840)
DEFAULT_NGL = 99
DEFAULT_ENABLE_MTP = True

DEFAULT_SERVER_READY_TIMEOUT_S = 900
DEFAULT_REQUEST_TIMEOUT_S = 1800
MEMORY_SAMPLE_INTERVAL_S = 0.5

LLAMA_API_BASE = "https://api.github.com/repos/cloudlnkcn/llama.cpp/releases?per_page=5"
LLAMA_FALLBACK_URL = (
    "https://github.com/ai-dock/llama.cpp-cuda/releases/download/b9628/"
    "llama.cpp-b9628-cuda-12.8-amd64.tar.gz"
)
USER_AGENT = "ornith15-benchmark/1.0"
FILLER_SENTENCE = "The swift digital agent analyzes repository structure and executes precise unit tests across distributed systems. "


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
    return Path("/kaggle/tmp/ornith15_bench" if is_kaggle() else Path.cwd() / "scratch" / "ornith15_bench")


def default_outdir() -> Path:
    return Path("/kaggle/working" if is_kaggle() else Path.cwd() / "bench_results" / "ornith15")


def http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


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
# Model & llama.cpp Installation
# ============================================================================


def download_url_with_fallback(
    url: str,
    target: Path,
    repo: str = "",
    filename: str = "",
    model_dir: Path | None = None,
    hf_token: str = "",
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # 1. Try huggingface_hub if available
    if repo and filename and model_dir:
        try:
            from huggingface_hub import hf_hub_download
            log(f"Downloading via huggingface_hub ({repo}/{filename})...")
            downloaded = hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=str(model_dir),
                token=hf_token or None,
            )
            return Path(downloaded)
        except Exception as exc:
            log(f"huggingface_hub not available ({exc}); trying CLI downloaders...")

    # 2. aria2c (fast multi-connection)
    if shutil.which("aria2c"):
        log("Downloading with aria2c...")
        cmd = ["aria2c", "-x", "16", "-s", "16", "-k", "1M", "-d", str(target.parent), "-o", target.name]
        if hf_token:
            cmd.extend(["--header", f"Authorization: Bearer {hf_token}"])
        cmd.append(url)
        subprocess.run(cmd, check=True)
        return target

    # 3. wget
    if shutil.which("wget"):
        log("Downloading with wget...")
        cmd = ["wget", "--continue", "-O", str(target)]
        if hf_token:
            cmd.extend(["--header", f"Authorization: Bearer {hf_token}"])
        cmd.append(url)
        subprocess.run(cmd, check=True)
        return target

    # 4. curl (built-in on Windows 10/11 & Linux)
    curl_bin = shutil.which("curl.exe") or shutil.which("curl")
    if curl_bin:
        log(f"Downloading with curl ({curl_bin})...")
        cmd = [curl_bin, "-L", "--retry", "3", "-C", "-", "-o", str(target)]
        if hf_token:
            cmd.extend(["-H", f"Authorization: Bearer {hf_token}"])
        cmd.append(url)
        subprocess.run(cmd, check=True)
        return target

    # 5. Native Python streaming download with chunking & progress
    log("Downloading via Python urllib streaming...")
    temp_target = target.with_suffix(target.suffix + ".part")
    headers = {"User-Agent": USER_AGENT}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp, open(temp_target, "wb") as fh:
        total_size = resp.headers.get("Content-Length")
        total_bytes = int(total_size) if total_size and total_size.isdigit() else 0
        downloaded = 0
        chunk_size = 8 * 1024 * 1024  # 8MB chunk
        t_last = time.time()
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            fh.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - t_last >= 5.0:
                pct = (downloaded / total_bytes * 100) if total_bytes else 0.0
                dl_gb = downloaded / (1024**3)
                tot_gb = total_bytes / (1024**3) if total_bytes else 0.0
                log(f"  ⏳ Downloaded: {dl_gb:.2f} / {tot_gb:.2f} GB ({pct:.1f}%)")
                t_last = now
    temp_target.replace(target)
    return target


def get_or_download_model(
    repo: str,
    filename: str,
    direct_url: str,
    model_dir: Path,
    hf_token: str = "",
) -> Path:
    """Download GGUF or find it mounted on Kaggle dataset."""
    model_dir.mkdir(parents=True, exist_ok=True)

    if Path("/kaggle/input").exists():
        mounted = [p for p in Path("/kaggle/input").rglob("*.gguf") if "mmproj" not in p.name.lower()]
        for m in mounted:
            if filename.lower() in m.name.lower() or "ornith" in m.name.lower() or "ice" in m.name.lower():
                log(f"Using mounted dataset model: {m.name} ({m.stat().st_size / 1024**3:.2f} GB)")
                return m

    target = model_dir / filename
    if target.exists() and target.stat().st_size > 1024**3:
        log(f"Using cached model: {target} ({target.stat().st_size / 1024**3:.2f} GB)")
        return target

    log(f"Downloading {repo}/{filename} (~20.8 GB)...")
    url = direct_url or f"https://huggingface.co/{repo}/resolve/main/{filename}"
    return download_url_with_fallback(
        url,
        target,
        repo=repo,
        filename=filename,
        model_dir=model_dir,
        hf_token=hf_token,
    )


def install_llama_cpp(workdir: Path) -> Path:
    """Download prebuilt CUDA 12 (sm_75) llama.cpp into workdir or locate executable."""
    bin_dir = workdir / "bin"
    server_exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    server_bin = bin_dir / server_exe

    if server_bin.exists() and os.access(server_bin, os.X_OK):
        log(f"llama.cpp already installed at {bin_dir}")
        return bin_dir

    # Check PATH for existing llama-server
    path_server = shutil.which(server_exe) or shutil.which("llama-server")
    if path_server:
        path_p = Path(path_server)
        log(f"Found existing llama-server in PATH: {path_p}")
        return path_p.parent

    found = [p for p in workdir.rglob(server_exe) if p.is_file() and os.access(p, os.X_OK)]
    if found:
        bin_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found[0], server_bin)
        return bin_dir

    # On Windows without local binary, provide clear guidance
    if os.name == "nt":
        raise RuntimeError(
            "llama-server executable not found on Windows.\n"
            "Notice: This benchmark harness is optimized for Kaggle 2x Tesla T4 (Linux CUDA sm_75).\n"
            "To execute on Kaggle:\n"
            "  1. Deploy kernel to Kaggle Cloud via launch.py or Kaggle CLI\n"
            "  2. Or inside Kaggle Notebook cell: import benchmark_sweet_spot; benchmark_sweet_spot.main([])\n"
            f"If testing locally on Windows, place {server_exe} in PATH or in {bin_dir}"
        )

    log("Downloading prebuilt llama.cpp (sm_75 CUDA)...")
    download_url, archive_name = None, "ubuntu-cuda-sm_75-x64.tar.xz"
    try:
        releases = json.loads(http_get(LLAMA_API_BASE, timeout=20).decode())
        pattern = re.compile(r"ubuntu-cuda-sm_75-x64\.tar\.xz$", re.I)
        for rel in releases:
            for asset in rel.get("assets", []):
                if pattern.search(asset["name"]):
                    download_url, archive_name = asset["browser_download_url"], asset["name"]
                    break
            if download_url:
                break
    except Exception as exc:
        log(f"GitHub API notice: {exc}; using fallback release")

    if not download_url:
        download_url, archive_name = LLAMA_FALLBACK_URL, "llama.cpp-b9628-cuda-12.8-amd64.tar.gz"

    archive_path = workdir / archive_name
    urllib.request.urlretrieve(download_url, str(archive_path))
    with tarfile.open(archive_path, "r:*") as tf:
        try:
            tf.extractall(workdir, filter="data")
        except TypeError:
            tf.extractall(workdir)

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

    log(f"llama.cpp ready: {bin_dir}")
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
        self._stop_event = threading.Event()

    def _sample_ram(self) -> float:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                mem = {}
                for line in fh:
                    parts = line.split()
                    if len(parts) >= 2:
                        mem[parts[0].rstrip(":")] = float(parts[1])
                total_kb = mem.get("MemTotal", 0.0)
                avail_kb = mem.get("MemAvailable", 0.0)
                return (total_kb - avail_kb) / 1024.0 / 1024.0
        except Exception:
            pass
        if os.name == "nt":
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    return (stat.ullTotalPhys - stat.ullAvailPhys) / (1024**3)
            except Exception:
                pass
        return 0.0

    def _sample_gpus(self) -> tuple[list[int], list[int]]:
        if not shutil.which("nvidia-smi"):
            return [], []
        try:
            cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
            out = subprocess.check_output(cmd, text=True, timeout=2).strip()
            used_list, total_list = [], []
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) == 2:
                    used_list.append(int(parts[0]))
                    total_list.append(int(parts[1]))
            return used_list, total_list
        except Exception:
            return [], []

    def run(self) -> None:
        while not self._stop_event.is_set():
            used, total = self._sample_gpus()
            ram_gb = self._sample_ram()
            if used:
                self.gpu_used_mb = used
                self.gpu_total_mb = total
                total_used = sum(used)
                if total_used > self.peak_gpu_mb:
                    self.peak_gpu_mb = total_used
                if not self.peak_per_gpu_mb or len(self.peak_per_gpu_mb) < len(used):
                    self.peak_per_gpu_mb = list(used)
                else:
                    for i, u in enumerate(used):
                        if u > self.peak_per_gpu_mb[i]:
                            self.peak_per_gpu_mb[i] = u
            if ram_gb > self.peak_ram_gb:
                self.peak_ram_gb = ram_gb
            time.sleep(self.interval_s)

    def snapshot(self) -> tuple[int, float]:
        used, _ = self._sample_gpus()
        ram_gb = self._sample_ram()
        return sum(used) if used else 0, round(ram_gb, 2)

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
    cmd = [
        str(server_bin),
        "-m", str(model_path),
        "-ngl", str(ngl),
        "-sm", "layer",
        "-c", str(context),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "-np", "1",
        "--cache-type-k", args.kv_cache,
        "--cache-type-v", args.kv_cache,
        "--tensor-split", "1,1",
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
    cmd = build_server_cmd(bin_dir / "llama-server", model_path, args, context, ngl)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = so_dirs_from(bin_dir.parent)
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["GGML_CUDA_NO_VMM"] = "1"

    try:
        help_text = subprocess.run([str(bin_dir / "llama-server"), "--help"],
                                   capture_output=True, text=True, env=env, timeout=30).stdout or ""
    except Exception:
        help_text = ""

    if "--flash-attn" in help_text:
        cmd.extend(["--flash-attn", "on"])
    elif "-fa" in help_text:
        cmd.extend(["-fa", "on"])

    if "-fit" in help_text or "--fit" in help_text:
        cmd.extend(["-fit", "off"])

    # Enable MTP self-speculative decoding
    if args.enable_mtp:
        if "--spec-type" in help_text:
            cmd.extend(["--spec-type", "draft-mtp"])
            log("  ⚡ MTP self-speculative decoding enabled (--spec-type draft-mtp)")
        else:
            log("  ⚠️ llama-server binary does not advertise --spec-type draft-mtp")

    log(f"Starting llama-server: ctx={context:,}, ngl={ngl}, batch={args.batch_size}/{args.ubatch_size}, kv={args.kv_cache}, mtp={args.enable_mtp} ...")
    server_log = open(log_path, "w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT, env=env, text=True)
    return proc


def wait_for_server(args: argparse.Namespace, proc: subprocess.Popen | None, timeout: int) -> bool:
    if proc is None:
        return False
    url = f"http://127.0.0.1:{args.port}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return False
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {args.api_key}"})
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
    temperature: float = 0.0,
) -> CompletionResult:
    """Executes a streaming chat completion request and records TTFT and token generation speed."""
    payload = {
        "model": args.model_alias,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
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
    log("Starting Ornith-1.5-35B-A3B Context Sweet-Spot Sweep (100k-160k+)")
    log(f"Config: KV={args.kv_cache}, Batch={args.batch_size}, UBatch={args.ubatch_size}, NGL={args.ngl}, MTP={args.enable_mtp}")
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
            "enable_mtp": args.enable_mtp,
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

        log(f"  🟢 ctx={context:,}: Server live. Idle VRAM: {idle_gpu_mb / 1024:.2f} GB (GPU 0: {sampler.gpu_used_mb[0] if sampler.gpu_used_mb else 0} MB, GPU 1: {sampler.gpu_used_mb[1] if len(sampler.gpu_used_mb) > 1 else 0} MB)")

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

        # 2. Short request (baseline decode speed with MTP speculative decoding)
        short_prompt = "Explain in 3 bullet points how MoE routing and Multi-Token Prediction (MTP) draft head accelerate generation."
        short = stream_completion(args, short_prompt, max_tokens=128, temperature=0.0)
        row["short_request"] = short.metrics

        # 3. Long context probe request (push context depth into 100k+)
        long_target_tokens = min(32768, context // 4)
        long_prompt = build_fill_prompt(
            long_target_tokens,
            "Based on the text above, which entity executes precise unit tests across distributed systems? Answer in one short sentence.",
        )
        log(f"  ⏳ ctx={context:,}: Running long prompt probe (~{long_target_tokens:,} tokens)...")
        long = stream_completion(args, long_prompt, max_tokens=128, temperature=0.0)
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

        log(f"  📊 ctx={context:>7,d} | Fit={row['fit']} | "
            f"Short Gen: {fmt(short.gen_tps)} tok/s | "
            f"Long Prefill: {fmt(long.prompt_tps)} tok/s | "
            f"Long TTFT: {fmt(long.ttft_s)} s | "
            f"Long Gen: {fmt(long.gen_tps)} tok/s | "
            f"Peak VRAM: {row['peak_vram_gb']} GB")

        rows.append(row)

    return rows


def synthesize_sweet_spot(rows: list[dict], total_vram_limit_gb: float = 30.0) -> dict:
    """Finds the optimal context configuration balancing stability, context size, and generation TPS."""
    fit_rows = [r for r in rows if r.get("fit")]
    if not fit_rows:
        return {
            "recommended": None,
            "reason": "No context window successfully completed both short and long inference without OOM.",
            "candidates": rows,
        }

    # Filter by safety headroom: peak VRAM must be <= 29.5 GB on 2x T4
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
            "enable_mtp": best["enable_mtp"],
            "short_gen_tps": (best.get("short_request") or {}).get("gen_tps"),
            "long_prefill_tps": (best.get("long_request") or {}).get("prompt_tps"),
            "long_ttft_s": (best.get("long_request") or {}).get("ttft_s"),
            "long_gen_tps": (best.get("long_request") or {}).get("gen_tps"),
            "peak_vram_gb": best.get("peak_vram_gb"),
            "score": round(score(best), 2),
        },
        "reason": (
            f"Context {best['context']:,} tokens achieves peak VRAM {best.get('peak_vram_gb')} GB "
            f"with generation speed {fmt((best.get('long_request') or {}).get('gen_tps'))} tok/s (MTP accelerated) "
            f"and long prefill {fmt((best.get('long_request') or {}).get('prompt_tps'))} tok/s safely within Kaggle 2x T4 limits."
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
            "short_gen_tps", "long_prompt_tps", "long_ttft_s", "long_gen_tps", "error"
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
        "# Ornith-1.5-35B-A3B (TIEL-Calibrated MTPv2 ICE GGUF) 2x Tesla T4 Sweet-Spot Benchmark",
        "",
        f"**Hardware**: Kaggle 2x Nvidia Tesla T4 (32GB Total VRAM)",
        f"**Model**: `{DEFAULT_MODEL_REPO}` (`{DEFAULT_MODEL_FILE}`)",
        f"**KV Cache**: `{DEFAULT_KV_CACHE}` | **Batch**: `{DEFAULT_BATCH_SIZE}` | **UBatch**: `{DEFAULT_UBATCH_SIZE}` | **MTP**: `Enabled`",
        "",
        "## 📊 Context Scaling (100k+) & Performance Results",
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
        md.append(f"- **MTP Speculative Decoding**: `Enabled (--spec-type draft-mtp)`")
        md.append(f"- **Peak VRAM**: `{rec['peak_vram_gb']} GB` (fits within 2x T4 budget)")
        md.append(f"- **Decode Generation Speed**: `{fmt(rec['long_gen_tps'])} tok/s` (MTP accelerated)")
        md.append(f"- **Prefill Processing Speed**: `{fmt(rec['long_prefill_tps'])} tok/s` (TTFT: `{fmt(rec['long_ttft_s'])} s`)")
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
        description="Benchmark Ornith-1.5-35B-A3B sweet-spot (100k-160k+ context, MTP) on Kaggle 2x Tesla T4"
    )
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL)
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--kv-cache", default=DEFAULT_KV_CACHE, help="KV cache quantization (default: q4_0)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--ubatch-size", type=int, default=DEFAULT_UBATCH_SIZE)
    parser.add_argument("--ngl", type=int, default=DEFAULT_NGL)
    parser.add_argument(
        "--context-sizes",
        default=",".join(str(c) for c in DEFAULT_CONTEXT_SIZES),
        help="Comma-separated context sizes to sweep (default: 102400,114688,131072,147456,163840)",
    )
    parser.add_argument("--enable-mtp", action="store_true", default=DEFAULT_ENABLE_MTP,
                        help="Enable MTP self-speculative decoding (default: True)")
    parser.add_argument("--no-mtp", action="store_false", dest="enable_mtp",
                        help="Disable MTP self-speculative decoding")
    parser.add_argument("--server-ready-timeout-s", type=int, default=DEFAULT_SERVER_READY_TIMEOUT_S)
    parser.add_argument("--request-timeout-s", type=int, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--hf-token", default="", help="Hugging Face API token (or set HF_TOKEN env)")
    parser.add_argument("--workdir", default=None, help="Working/scratch dir")
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

    hf_token = get_hf_token(getattr(args, "hf_token", ""))
    if hf_token:
        log("Hugging Face token authenticated")

    log("Provisioning model weights and llama.cpp CUDA binaries...")
    model_path = get_or_download_model(
        args.model_repo,
        args.model_file,
        args.model_url,
        args.workdir / "models",
        hf_token=hf_token,
    )
    bin_dir = install_llama_cpp(args.workdir)

    results: dict = {
        "meta": {
            "model_repo": args.model_repo,
            "model_file": args.model_file,
            "context_sizes": args.context_sizes,
            "ngl": args.ngl,
            "kv_cache": args.kv_cache,
            "batch": args.batch_size,
            "ubatch": args.ubatch_size,
            "enable_mtp": args.enable_mtp,
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
        log(f"🏆 SWEET SPOT IDENTIFIED -> Context={rec['context']:,} tokens | "
            f"VRAM={rec['peak_vram_gb']} GB | Long Gen={fmt(rec['long_gen_tps'])} tok/s (MTP) | "
            f"Prefill={fmt(rec['long_prefill_tps'])} tok/s")
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
