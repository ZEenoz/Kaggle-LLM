#!/usr/bin/env python3
"""v2 throughput & context benchmark for Ornith-1.5-35B-A3B (Q4_K_M) on 2x Tesla T4.

Optimized for Kaggle 2x Tesla T4 GPUs (32GB VRAM total, ~30.2GB usable, sm_75):

    B -- KV cache matrix: targeted configs (q4_0 at 32k & 128k, q8_0 at 32k,
         fa on/off) + perplexity quality check on GSM8K (40 problems).
         Pre-filters and skips OOM-class combos (e.g. q8_0 at 128k).
    D -- Context scaling limits: 4-step progressive probe (32k -> 64k -> 96k -> 128k)
         with full-context probe requests, then -ngl offload boundary sweep (99 vs 80).
    C -- Batching & concurrency: 5 targeted llama.cpp cells (np {1,4,8} x batch/ubatch pairs)
         under closed-loop concurrency (1, 4, 8) and open-loop Poisson arrivals (1.0, 2.0, 3.0 req/s);
         reports TTFT_p95, ITL_p95, system tok/s, and sustainable arrival-rate ceiling.

Instrumentation & Resilience:
    - GpuSampler: 0.5s sampling of VRAM, GPU util%, power draw, SM clocks,
      temperature + system RAM; time series written to samples_v2.csv.
    - Per-request JSONL (requests_v2.jsonl): submit/first-token/chunk
      timestamps, prompt & completion token counts, ITL distribution.
    - Checkpointing & Resume: saves state atomically after every run to
      checkpoint_v2.json; automatically resumes across Kaggle disconnects (--resume).
    - Safeguards: passes -fit off to prevent silent context reduction; guards against
      high idle VRAM (>29GB) on 2x T4.

Outputs (in --out-dir, default /kaggle/working):
    results_v2.json       full nested results
    checkpoint_v2.json    checkpoint state for resuming
    sweet_spot_v2.json    SLO-filtered ranking, Pareto frontier, recommendation
    results_v2.csv        tidy long-format rows (phase, run_id, profile, metric)
    samples_v2.csv        GPU/RAM time series per run
    requests_v2.jsonl     per-request records
    summary_v2.md         human-readable tables & copy-paste launcher configs

Run on Kaggle (GPU T4 x 2, Internet ON):
    python benchmark_q4_v2.py                 # full targeted plan, ~1.5-2h
    python benchmark_q4_v2.py --quick         # ~15-20 min smoke of every phase
    python benchmark_q4_v2.py --skip c        # only KV + context phases

Inside a Jupyter/Kaggle notebook cell:
    import benchmark_q4_v2
    benchmark_q4_v2.main([])
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zlib
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================================
# Configuration (single source of truth; all overridable via CLI arguments)
# ============================================================================

DEFAULT_MODEL_REPO = "ornith-ai/Ornith-1.5-35B-A3B-GGUF"
DEFAULT_MODEL_FILE = "Ornith-1.5-35B-Q4_K_M.gguf"
DEFAULT_MODEL_ALIAS = "ornith-bench"
DEFAULT_API_KEY = "kilo-bench-key"
DEFAULT_PORT = 8080
DEFAULT_VLLM_PORT = 8000

# Phase B -- KV matrix
DEFAULT_B_CTX_SIZES = (32768, 131072)
DEFAULT_KV_TYPES = ("q4_0", "q8_0")
DEFAULT_B_TARGETED_CONFIGS = (
    # (ctx, kv, fa, reuse)
    (32768, "q4_0", "on", "off"),
    (131072, "q4_0", "on", "off"),
    (32768, "q8_0", "on", "off"),
    (32768, "q4_0", "off", "off"),
)
FA_MODES = ("on", "off")
REUSE_MODES = ("off",)
B_SKIP_KV_CTX: set[tuple[str, int]] = {("q8_0", 131072)}  # known OOM-class combos: recorded, not booted
B_LONG_PROMPT_TOKENS = 8192
PPL_KV_TYPES = ("q4_0", "q8_0")

# Phase C -- batching & concurrency
DEFAULT_C_CONTEXT = 32768
DEFAULT_C_TARGETED_CELLS = (
    # (np, batch, ubatch)
    (1, 1024, 512),
    (4, 1024, 512),
    (4, 2048, 512),
    (8, 1024, 512),
    (8, 512, 512),
)
DEFAULT_NP_VALUES = (1, 4, 8)
DEFAULT_BATCH_PAIRS = (
    (1024, 512),
    (2048, 512),
    (512, 512),
)
DEFAULT_CLOSED_CONCURRENCIES = (1, 4, 8)
DEFAULT_POISSON_RATES = (1.0, 2.0, 3.0)
DEFAULT_POISSON_N = 16
DEFAULT_VLLM_MAX_NUM_SEQS = (1, 16)
DEFAULT_VLLM_PREFIX_CACHE = ("off",)
MIX_SHORT_TOKENS = 1024
MIX_LONG_TOKENS = 8192
MIX_LONG_FRACTION = 0.4
COMPLETION_TOKENS = 128
CHARS_PER_TOKEN = 4

# Phase D -- context limits
DEFAULT_D_STEPS = (32768, 65536, 98304, 131072)
DEFAULT_D_LO = 32768
DEFAULT_D_HI = 131072
DEFAULT_D_KV_TYPES = ("q4_0",)
DEFAULT_NGL_SWEEP = (99, 80)
CTX_ALIGN = 4096
FULL_CTX_GEN_TOKENS = 64


def is_oom_risk(kv: str, ctx: int) -> bool:
    """Predict whether a KV quant and context size will exceed 2x T4 free VRAM (~8.5GB)."""
    return kv == "q8_0" and ctx > 32768

# SLOs used for sweet-spot synthesis
DEFAULT_SLO_TTFT_S = 5.0
DEFAULT_SLO_ITL_S = 0.15

# Infrastructure
SERVER_READY_TIMEOUT_S = 900
REQUEST_TIMEOUT_S = 1800
FULL_CTX_REQUEST_TIMEOUT_S = 1500
SHORT_REQUEST_TIMEOUT_S = 600
MEMORY_SAMPLE_INTERVAL_S = 0.5
SAMPLES_MAX = 8000
PPL_CONTEXT = 4096
PPL_QUESTIONS = 40

LLAMA_API_BASE = "https://api.github.com/repos/cloudlnkcn/llama.cpp/releases?per_page=5"
LLAMA_FALLBACK_URL = (
    "https://github.com/ai-dock/llama.cpp-cuda/releases/download/b9628/"
    "llama.cpp-b9628-cuda-12.8-amd64.tar.gz"
)
USER_AGENT = "ornith15-benchmark-v2/1.0"
GSM8K_RAW_URL = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test.jsonl"
GOLD_ANSWER_RE = re.compile(r"Answer: ([\d,]+(?:\.\d+)?)")
FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog near the old stone mill. "

# Fallback corpus if the GSM8K download fails (only used for perplexity).
FALLBACK_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning models trade accuracy for efficiency. "
    "The inference server processes prompts in batches and generates tokens one at a time. "
) * 200


# ============================================================================
# Small helpers
# ============================================================================


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def strip_kernel_args(argv: list[str]) -> list[str]:
    """Drop the ``-f <connection file>`` pair Jupyter kernels leave in sys.argv."""
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


def default_workdir() -> Path:
    return Path("/kaggle/tmp/ornith_bench" if Path("/kaggle").exists() else "/tmp/ornith_bench")


def default_outdir() -> Path:
    return Path("/kaggle/working" if Path("/kaggle").exists() else Path.cwd() / "bench_results_v2")


def build_fill_prompt(target_tokens: int, suffix: str) -> str:
    """Deterministic filler prompt of roughly target_tokens tokens."""
    target_chars = max(target_tokens * CHARS_PER_TOKEN, 512)
    filler = (FILLER_SENTENCE * (target_chars // len(FILLER_SENTENCE) + 1))[:target_chars]
    return filler + suffix


def fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def so_dirs_from(workdir: Path) -> str:
    dirs = {str(p.parent.resolve()) for p in workdir.rglob("*.so*") if p.is_file()}
    return ":".join(dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")


def stop_all_servers() -> None:
    """Best-effort cleanup of any llama-server / vLLM processes."""
    if shutil.which("pkill"):
        for pattern in ("[l]lama-server", "[v]llm.entrypoints", "[uvicorn]"):
            try:
                subprocess.run(
                    ["pkill", "-9", "-f", pattern],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
    elif os.name == "nt":
        for pattern in ("llama-server.exe", "llama-server"):
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", pattern],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass


def kill_process_tree(proc: "subprocess.Popen | None") -> None:
    if proc is None:
        return
    if proc.poll() is None:
        if hasattr(os, "killpg") and hasattr(os, "getpgid"):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
        else:
            proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
            else:
                proc.kill()


# ============================================================================
# GPU / RAM sampling
# ============================================================================


class GpuSampler(threading.Thread):
    """Background sampler: VRAM, GPU util/power/clocks/temp + system RAM.

    Keeps per-run peaks and a capped time series (samples_v2.csv) for
    throttle and power-cap detection.
    """

    def __init__(self, interval_s: float = MEMORY_SAMPLE_INTERVAL_S) -> None:
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.peak_vram_mb: int = 0
        self.peak_ram_gb: float = 0.0
        self.total_vram_mb: int = 0
        self.series: deque[tuple] = deque(maxlen=SAMPLES_MAX)
        self._stop_event = threading.Event()

    def snapshot(self) -> tuple[int, float]:
        """Sample once and update peaks; returns (used VRAM MB, used RAM GB)."""
        vram_mb, ram_gb = 0, 0.0
        gpu_fields: tuple = (-1, 0.0, 0.0, -1, -1)
        row_total = 0
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total,utilization.gpu,"
                    "power.draw,clocks.sm,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            )
            util = 0.0
            power = 0.0
            clocks = None
            temp = 0
            for line in out.strip().splitlines():
                _, mem, tot, util_v, pw, clk, tmp = (p.strip() for p in line.split(","))
                vram_mb += int(mem)
                row_total += int(tot)
                util = max(util, float(util_v))
                power = max(power, float(pw))
                clocks = int(clk) if clocks is None else min(clocks, int(clk))
                temp = max(temp, int(tmp))
            gpu_fields = (util, power, clocks if clocks is not None else -1, temp, vram_mb)
        except Exception:  # noqa: BLE001 - sampling must never crash the run
            pass
        try:
            info: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, val = line.split(":", 1)
                info[key.strip()] = int(val.strip().split()[0])  # kB
            ram_gb = (info["MemTotal"] - info["MemAvailable"]) / 1024**2
        except Exception:  # noqa: BLE001
            pass
        util, power, clocks, temp, sampled_vram = gpu_fields
        self.series.append((time.time(), sampled_vram, util, power, clocks, temp, ram_gb))
        self.peak_vram_mb = max(self.peak_vram_mb, vram_mb)
        self.peak_ram_gb = max(self.peak_ram_gb, ram_gb)
        self.total_vram_mb = max(self.total_vram_mb, row_total)
        return vram_mb, ram_gb

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.snapshot()
            self._stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=5)


# ============================================================================
# Model + llama.cpp provisioning (same strategy as v1)
# ============================================================================


def get_or_download_model(repo: str, filename: str, model_dir: Path) -> Path:
    """Locate the GGUF on a mounted dataset, on disk, or download it."""
    model_dir.mkdir(parents=True, exist_ok=True)
    if Path("/kaggle/input").exists():
        mounted = [p for p in Path("/kaggle/input").rglob("*.gguf") if "mmproj" not in p.name.lower()]
        for m in mounted:
            if "ornith" in m.name.lower():
                log(f"Using mounted dataset model: {m.name} ({m.stat().st_size / 1024**3:.2f} GB)")
                return m
    target = model_dir / filename
    if target.exists() and target.stat().st_size > 1024**3:
        log(f"Using cached model: {target} ({target.stat().st_size / 1024**3:.2f} GB)")
        return target
    log(f"Downloading {repo}/{filename} (~21.7 GB)...")
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=repo, filename=filename, local_dir=str(model_dir)))
    except Exception as exc:  # noqa: BLE001
        log(f"hf_hub_download failed ({exc}); direct download fallback")
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        if shutil.which("aria2c"):
            subprocess.run(["aria2c", "-x", "16", "-s", "16", "-k", "1M",
                            "-d", str(model_dir), "-o", filename, url], check=True)
        else:
            subprocess.run(["wget", "--continue", "-O", str(target), url], check=True)
    log(f"Model ready: {target} ({target.stat().st_size / 1024**3:.2f} GB)")
    return target


def install_llama_cpp(workdir: Path) -> Path:
    """Fetch a prebuilt CUDA llama.cpp; return the bin directory."""
    bin_dir = workdir / "bin"
    server_bin = bin_dir / "llama-server"
    if server_bin.exists() and os.access(server_bin, os.X_OK):
        log(f"llama.cpp already installed at {bin_dir}")
        return bin_dir
    found = [p for p in workdir.rglob("llama-server") if p.is_file() and os.access(p, os.X_OK)]
    if found:
        bin_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found[0], server_bin)
        return bin_dir
    log("Downloading prebuilt llama.cpp (sm_75) ...")
    download_url, archive_name = None, "ubuntu-cuda-sm_75-x64.tar.xz"
    try:
        req = urllib.request.Request(LLAMA_API_BASE, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp:
            releases = json.loads(resp.read().decode())
        pattern = re.compile(r"ubuntu-cuda-sm_75-x64\.tar\.xz$", re.I)
        for rel in releases:
            for asset in rel.get("assets", []):
                if pattern.search(asset["name"]):
                    download_url, archive_name = asset["browser_download_url"], asset["name"]
                    break
            if download_url:
                break
    except Exception as exc:  # noqa: BLE001
        log(f"GitHub API notice: {exc}; using fallback release URL: {LLAMA_FALLBACK_URL}")
    if not download_url:
        download_url, archive_name = LLAMA_FALLBACK_URL, "llama.cpp-b9628-cuda-12.8-amd64.tar.gz"
    archive_path = workdir / archive_name
    urllib.request.urlretrieve(download_url, str(archive_path))
    with tarfile.open(archive_path, "r:*") as tf:
        try:
            tf.extractall(workdir, filter="data")
        except TypeError:  # Python < 3.12: no filter kwarg
            tf.extractall(workdir)
    found = [p for p in workdir.rglob("llama-server") if p.is_file()]
    if not found:
        raise RuntimeError("llama-server executable not found after extraction")
    real = found[0]
    real.chmod(real.stat().st_mode | 0o755)
    bin_dir.mkdir(parents=True, exist_ok=True)
    if not server_bin.exists():
        try:
            server_bin.symlink_to(real)
        except Exception:
            shutil.copy2(real, server_bin)
    for tool_name in ("llama-perplexity", "llama-perplex"):
        found_tool = list(workdir.rglob(tool_name))
        if found_tool:
            tool_real = found_tool[0]
            tool_real.chmod(tool_real.stat().st_mode | 0o755)
            for alias in ("llama-perplexity", "llama-perplex"):
                dst = bin_dir / alias
                if not dst.exists():
                    try:
                        dst.symlink_to(tool_real)
                    except Exception:
                        shutil.copy2(tool_real, dst)
            break
    log(f"llama.cpp ready: {bin_dir}")
    return bin_dir


def ensure_vllm() -> bool:
    """Best-effort vLLM availability check + install; never fatal."""
    if importlib.util.find_spec("vllm") is not None:
        log("vLLM already installed")
        return True
    log("pip-installing vLLM (one-time, ~10-20 min, large wheels) ...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "vllm"],
            timeout=3600,
        )
        return result.returncode == 0
    except Exception as exc:  # noqa: BLE001
        log(f"vLLM install failed: {exc}")
        return False


# ============================================================================
# Completion client (streaming SSE, per-request instrumentation)
# ============================================================================


@dataclass
class CompletionResult:
    """Instrumented chat completion result."""

    ok: bool
    ts_submit: float = field(default_factory=time.time)
    ts_first: float | None = None
    ts_done: float | None = None
    text: str = ""
    chunk_ts: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str = ""

    @property
    def ttft_s(self) -> float | None:
        return self.ts_first - self.ts_submit if self.ts_first else None

    @property
    def gen_tps(self) -> float | None:
        last = self.chunk_ts[-1] if self.chunk_ts else self.ts_done
        if not self.ts_first or last is None or not self.completion_tokens:
            return None
        span = last - self.ts_first
        return self.completion_tokens / span if span > 0 else None

    @property
    def pp_tps_incl_queue(self) -> float | None:
        if self.ts_first and self.prompt_tokens:
            delta = self.ts_first - self.ts_submit
            return self.prompt_tokens / delta if delta > 0 else None
        return None

    @property
    def itl_p95_s(self) -> float | None:
        diffs = [b - a for a, b in zip(self.chunk_ts, self.chunk_ts[1:])]
        return percentile(diffs, 0.95)


def stream_completion(
    port: int,
    api_key: str,
    model_alias: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> CompletionResult:
    """One streaming chat completion with chunk-level timestamps."""
    payload = {
        "model": model_alias,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "none",
            "Accept": "text/event-stream",
        },
    )
    result = CompletionResult(ok=False, ts_submit=time.time())
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
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
                    piece = (choices[0].get("delta") or {}).get("content")
                    if piece:
                        now = time.time()
                        if result.ts_first is None:
                            result.ts_first = now
                        if len(result.chunk_ts) < 512:
                            result.chunk_ts.append(now)
                        result.text += piece
                usage = chunk.get("usage")
                if usage:
                    result.prompt_tokens = usage.get("prompt_tokens")
                    result.completion_tokens = usage.get("completion_tokens")
        result.ts_done = time.time()
        result.ok = True
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
    return result


def record_request(jsonl_path: Path | None, record: dict) -> None:
    if jsonl_path is None:
        return
    line = json.dumps(record, ensure_ascii=False)
    with open(jsonl_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def record_to_dict(cfg_run_id: str, profile: str, index: int | str,
                   comp: CompletionResult) -> dict:
    return {
        "run_id": cfg_run_id,
        "profile": profile,
        "index": index,
        "ts_submit": comp.ts_submit,
        "ts_first": comp.ts_first,
        "ttft_s": comp.ttft_s,
        "itl_p95_s": comp.itl_p95_s,
        "gen_tps": comp.gen_tps,
        "pp_tps_incl_queue": comp.pp_tps_incl_queue,
        "prompt_tokens": comp.prompt_tokens,
        "completion_tokens": comp.completion_tokens,
        "ok": comp.ok,
        "error": comp.error,
    }


# ============================================================================
# Server lifecycle (llama.cpp and vLLM behind one config)
# ============================================================================


@dataclass
class RunCfg:
    """One server configuration = one benchmark run."""

    run_id: str
    phase: str
    engine: str = "llama"
    context: int = DEFAULT_C_CONTEXT
    ngl: int = 99
    kv: str = "q4_0"
    fa: str = "on"
    reuse: str = "off"
    np: int = 1
    batch: int = 1024
    ubatch: int = 512
    max_num_seqs: int | None = None
    prefix_cache: str | None = None
    port: int = DEFAULT_PORT


def build_llama_cmd(cfg: RunCfg, server_bin: Path, model_path: Path, help_text: str,
                    model_alias: str, api_key: str) -> list[str]:
    cmd = [
        str(server_bin),
        "-m", str(model_path),
        "-ngl", str(cfg.ngl),
        "-sm", "layer",
        "-c", str(cfg.context),
        "-b", str(cfg.batch),
        "-ub", str(cfg.ubatch),
        "-np", str(cfg.np),
        "--cache-type-k", cfg.kv,
        "--cache-type-v", cfg.kv,
        "--tensor-split", "1,1",
        "--host", "127.0.0.1",
        "--port", str(cfg.port),
        "--alias", model_alias,
        "--api-key", api_key,
    ]
    if cfg.fa == "on":
        if "--flash-attn" in help_text:
            cmd.extend(["--flash-attn", "on"])
        elif "-fa" in help_text:
            cmd.extend(["-fa", "on"])
    if "-fit" in help_text or "--fit" in help_text:
        cmd.extend(["-fit", "off"])
    if cfg.reuse == "on":
        if "cache-reuse" in help_text:
            cmd.extend(["--cache-reuse", "1024"])
        else:
            log(f"note: --cache-reuse not in this build ({cfg.run_id}); treated as off")
    return cmd


def start_llama(cfg: RunCfg, model_path: Path, bin_dir: Path, args: argparse.Namespace,
                log_path: Path) -> "subprocess.Popen | None":
    server_bin = bin_dir / "llama-server"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = so_dirs_from(bin_dir.parent)
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["GGML_CUDA_NO_VMM"] = "1"
    try:
        help_text = subprocess.run([str(server_bin), "--help"], capture_output=True,
                                   text=True, env=env, timeout=60).stdout or ""
    except Exception:  # noqa: BLE001
        help_text = ""
    cmd = build_llama_cmd(cfg, server_bin, model_path, help_text, args.model_alias, args.api_key)
    log(f"starting {cfg.run_id} ({cfg.engine}, ctx={cfg.context}, ngl={cfg.ngl}, "
        f"kv={cfg.kv}, fa={cfg.fa}, np={cfg.np}, b/ub={cfg.batch}/{cfg.ubatch})")
    server_log = open(log_path, "w", encoding="utf-8", buffering=1)
    return subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT, env=env,
                            text=True, start_new_session=True)


def start_vllm(cfg: RunCfg, model_path: Path, args: argparse.Namespace,
               log_path: Path) -> "subprocess.Popen | None":
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        str(model_path),
        "--tensor-parallel-size", "2",
        "--max-model-len", str(cfg.context),
        "--max-num-seqs", str(cfg.max_num_seqs or 16),
        "--gpu-memory-utilization", "0.92",
        "--port", str(cfg.port),
        "--served-model-name", args.model_alias,
    ]
    if cfg.prefix_cache == "on":
        cmd.append("--enable-prefix-caching")
    log(f"starting {cfg.run_id} (vLLM, ctx={cfg.context}, max_num_seqs={cfg.max_num_seqs}, "
        f"prefix_cache={cfg.prefix_cache})")
    server_log = open(log_path, "w", encoding="utf-8", buffering=1)
    return subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)


def server_log_tail(log_path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(log_path.read_text(errors="replace").splitlines()[-lines:])
    except Exception:  # noqa: BLE001
        return ""


def wait_for_server(port: int, api_key: str, proc: "subprocess.Popen | None",
                    timeout: int) -> bool:
    if proc is None:
        return False
    url = f"http://127.0.0.1:{port}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return False
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    return False


# ============================================================================
# Workloads
# ============================================================================

SHORT_PROMPT = "Explain what a hash map is in one short paragraph."


def one_request(args: argparse.Namespace, cfg: RunCfg, prompt: str,
                max_tokens: int, temperature: float = 0.0,
                timeout_s: int = REQUEST_TIMEOUT_S) -> CompletionResult:
    return stream_completion(cfg.port, args.api_key, args.model_alias, prompt,
                             max_tokens, temperature, timeout_s)


def workload_l1(args: argparse.Namespace, cfg: RunCfg, jsonl_path: Path | None) -> dict:
    """Single-stream ceiling: short generation + long (8k) prompt."""
    results = {
        "profile": "l1",
        "short_tps": None,
        "short_itl_p95_s": None,
        "long_ttft_s": None,
        "long_pp_tps": None,
        "long_gen_tps": None,
        "long_prompt_tokens": None,
    }
    for i in range(3):
        comp = one_request(args, cfg, SHORT_PROMPT, COMPLETION_TOKENS, 0.0, SHORT_REQUEST_TIMEOUT_S)
        record_request(jsonl_path, record_to_dict(cfg.run_id, f"l1.short.{i}", i, comp))
        if comp.ok and comp.gen_tps:
            results["short_tps"] = median([results["short_tps"], comp.gen_tps])
            results["short_itl_p95_s"] = comp.itl_p95_s
        time.sleep(0.5)
    long_prompt = build_fill_prompt(B_LONG_PROMPT_TOKENS,
                                    "Question: what is the capital of France? Answer in one word.")
    comp = one_request(args, cfg, long_prompt, COMPLETION_TOKENS, 0.0, REQUEST_TIMEOUT_S)
    record_request(jsonl_path, record_to_dict(cfg.run_id, "l1.long", "long", comp))
    if comp.ok:
        results["long_ttft_s"] = comp.ttft_s
        results["long_pp_tps"] = comp.pp_tps_incl_queue
        results["long_gen_tps"] = comp.gen_tps
        results["long_prompt_tokens"] = comp.prompt_tokens
    return results


def workload_l2(args: argparse.Namespace, cfg: RunCfg, concurrency: int,
                jsonl_path: Path | None) -> dict:
    """Closed-loop: `concurrency` requests submitted simultaneously, all 128-token."""
    barrier = threading.Barrier(concurrency)
    results: list[CompletionResult] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        barrier.wait()
        prompt = f"{SHORT_PROMPT} (variation {i:02d})"
        comp = one_request(args, cfg, prompt, COMPLETION_TOKENS, 0.0, SHORT_REQUEST_TIMEOUT_S)
        record_request(jsonl_path, record_to_dict(cfg.run_id, f"l2.c{concurrency}", i, comp))
        with lock:
            results.append(comp)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=args.request_timeout_s + 30)
    wall = time.time() - t0

    ok = [r for r in results if r.ok]
    out = {
        "profile": f"l2.c{concurrency}",
        "concurrency": concurrency,
        "wall_s": wall,
        "n_ok": len(ok),
        "n_failed": len(results) - len(ok),
        "system_tok_s": (sum(r.completion_tokens or 0 for r in ok) / wall) if wall > 0 else None,
        "ttft_p50_s": percentile([r.ttft_s for r in ok if r.ttft_s], 0.5),
        "ttft_p95_s": percentile([r.ttft_s for r in ok if r.ttft_s], 0.95),
        "itl_p95_s": percentile([r.itl_p95_s for r in ok if r.itl_p95_s], 0.95),
        "per_stream_tps_p50": percentile([r.gen_tps for r in ok if r.gen_tps], 0.5),
    }
    return out


def workload_l3(args: argparse.Namespace, cfg: RunCfg, rate: float, n: int,
                jsonl_path: Path | None) -> dict:
    """Open-loop Poisson arrivals: mixed 1k/8k prompts, 128-token completions."""
    random.seed(zlib.crc32(cfg.run_id.encode("utf-8")))
    prompts = [
        build_fill_prompt(MIX_LONG_TOKENS, "Summarize the text above in one sentence.")
        if random.random() < MIX_LONG_FRACTION
        else build_fill_prompt(MIX_SHORT_TOKENS, "Answer the question at the end in one word.")
        for _ in range(n)
    ]
    results: list[CompletionResult] = []
    lock = threading.Lock()
    t_open = time.time()
    t_next = t_open
    next_idx = 0

    def worker(i: int) -> None:
        comp = one_request(args, cfg, prompts[i], COMPLETION_TOKENS, 0.0, REQUEST_TIMEOUT_S)
        record_request(jsonl_path, record_to_dict(cfg.run_id, f"l3.r{rate}", i, comp))
        with lock:
            results.append(comp)

    while next_idx < n:
        delay = max(0.0, t_next - time.time())
        time.sleep(delay)
        threading.Thread(target=worker, args=(next_idx,), daemon=True).start()
        next_idx += 1
        t_next += random.expovariate(rate)

    deadline = time.time() + args.request_timeout_s
    while (len(results) < n) and time.time() < deadline:
        time.sleep(2)
    t_close = time.time()
    span = t_close - t_open

    ok = [r for r in results if r.ok]
    out = {
        "profile": f"l3.r{rate}",
        "arrival_rate_per_s": rate,
        "n_requested": n,
        "n_ok": len(ok),
        "timeouts_or_failed": n - len(ok),
        "span_s": span,
        "system_tok_s": (sum(r.completion_tokens or 0 for r in ok) / span) if span > 0 else None,
        "ttft_p50_s": percentile([r.ttft_s for r in ok if r.ttft_s], 0.5),
        "ttft_p95_s": percentile([r.ttft_s for r in ok if r.ttft_s], 0.95),
        "itl_p95_s": percentile([r.itl_p95_s for r in ok if r.itl_p95_s], 0.95),
        "per_stream_tps_p50": percentile([r.gen_tps for r in ok if r.gen_tps], 0.5),
    }
    return out


# ============================================================================
# Checkpointing & State Persistence
# ============================================================================


def load_checkpoint(out_dir: Path) -> dict:
    cp_path = out_dir / "checkpoint_v2.json"
    if cp_path.exists():
        try:
            data = json.loads(cp_path.read_text(encoding="utf-8"))
            log(f"Loaded checkpoint from {cp_path} ({len(data.get('completed_runs', {}))} runs recorded)")
            return data
        except Exception as exc:  # noqa: BLE001
            log(f"Warning: could not read checkpoint {cp_path}: {exc}")
    return {"completed_runs": {}, "samples": []}


def save_checkpoint(out_dir: Path, state: dict, results: dict, synthesis: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        cp_tmp = out_dir / "checkpoint_v2.tmp"
        cp_path = out_dir / "checkpoint_v2.json"
        cp_tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        cp_tmp.replace(cp_path)
    except Exception as exc:  # noqa: BLE001
        log(f"Warning: could not save checkpoint: {exc}")

    try:
        res_tmp = out_dir / "results_v2.tmp"
        res_path = out_dir / "results_v2.json"
        res_tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        res_tmp.replace(res_path)
    except Exception as exc:  # noqa: BLE001
        log(f"Warning: could not save intermediate results: {exc}")

    if synthesis is not None:
        try:
            sw_tmp = out_dir / "sweet_spot_v2.tmp"
            sw_path = out_dir / "sweet_spot_v2.json"
            sw_tmp.write_text(json.dumps(synthesis, indent=2, ensure_ascii=False), encoding="utf-8")
            sw_tmp.replace(sw_path)
        except Exception:  # noqa: BLE001
            pass


def run_or_resume(
    args: argparse.Namespace,
    cfg: RunCfg,
    model_path: Path,
    bin_dir: Path,
    workloads: list[tuple[str, Callable[[], dict]]],
    samples_out: list[dict],
    jsonl_path: Path,
    state: dict | None,
    results_holder: dict | None = None,
) -> dict:
    if state and getattr(args, "resume", True) and cfg.run_id in state.get("completed_runs", {}):
        log(f"  [checkpoint] Resuming completed run {cfg.run_id}")
        return state["completed_runs"][cfg.run_id]

    rec = run_server(args, cfg, model_path, bin_dir, workloads, samples_out, jsonl_path)
    if state is not None:
        state.setdefault("completed_runs", {})[cfg.run_id] = rec
        save_checkpoint(args.out_dir, state, results_holder or {})
    return rec


# ============================================================================
# Run harness: boot server, sample, run workloads, record, kill
# ============================================================================


def run_server(
    args: argparse.Namespace,
    cfg: RunCfg,
    model_path: Path,
    bin_dir: Path,
    workloads: list[tuple[str, Callable[[], dict]]],
    samples_out: list[dict],
    jsonl_path: Path,
) -> dict:
    """Boot one server config, run the given workloads, return the run record."""
    log_path = Path(args.workdir) / f"server_{cfg.run_id}.log"
    stop_all_servers()
    if cfg.engine == "llama":
        proc = start_llama(cfg, model_path, bin_dir, args, log_path)
    else:
        proc = start_vllm(cfg, model_path, args, log_path)
    record: dict = {
        "phase": cfg.phase,
        "run_id": cfg.run_id,
        "config": cfg_to_dict(cfg),
        "fit": False,
        "error": "",
        "workloads": {},
    }
    ready = wait_for_server(cfg.port, args.api_key, proc, args.server_ready_timeout_s)
    if not ready:
        record["error"] = "server failed to become ready (OOM?)\n" + server_log_tail(log_path)
        log(f"  !! {cfg.run_id}: not ready")
        kill_process_tree(proc)
        return record

    sampler = GpuSampler()
    sampler.start()
    idle_vram_mb, _ = sampler.snapshot()
    record["idle_vram_mb"] = idle_vram_mb

    if idle_vram_mb > 29000 and cfg.context > 32768:
        log(f"  !! {cfg.run_id}: idle VRAM dangerously high ({idle_vram_mb} MB); aborting run to prevent OOM crash")
        record["fit"] = False
        record["error"] = f"idle VRAM dangerously high ({idle_vram_mb} MB) for context {cfg.context}"
        sampler.stop()
        kill_process_tree(proc)
        return record

    if proc.poll() is None:
        record["fit"] = True
    for name, work in workloads:
        if proc.poll() is not None:
            record["fit"] = False
            record["error"] = "server died before workload " + name
            break
        t0 = time.time()
        try:
            record["workloads"][name] = work()
        except Exception as exc:  # noqa: BLE001 - keep going across workloads
            record["workloads"][name] = {"error": str(exc)}
        log(f"  {cfg.run_id} {name}: {summarize_workload(record['workloads'][name])} "
            f"({time.time() - t0:.0f}s)")

    sampler.stop()
    record["peak_vram_mb"] = sampler.peak_vram_mb
    record["peak_ram_gb"] = sampler.peak_ram_gb
    record["total_vram_mb"] = sampler.total_vram_mb
    if record["fit"] and proc.poll() is not None:
        record["fit"] = False
        record["error"] = "server died during run\n" + server_log_tail(log_path)
    for sample in sampler.series:
        samples_out.append(dict(zip(
            ["ts", "vram_mb", "util_pct", "power_w", "clock_mhz", "temp_c", "ram_gb"],
            sample,
        ), run_id=cfg.run_id))
    kill_process_tree(proc)
    return record


def cfg_to_dict(cfg: RunCfg) -> dict:
    return {
        "engine": cfg.engine, "context": cfg.context, "ngl": cfg.ngl, "kv": cfg.kv,
        "fa": cfg.fa, "reuse": cfg.reuse, "np": cfg.np, "batch": cfg.batch,
        "ubatch": cfg.ubatch, "max_num_seqs": cfg.max_num_seqs,
        "prefix_cache": cfg.prefix_cache,
    }


def summarize_workload(work: dict) -> str:
    if not work:
        return "n/a"
    bits = []
    for key in ("short_tps", "long_gen_tps", "system_tok_s", "ttft_p95_s", "itl_p95_s"):
        if work.get(key) is not None:
            bits.append(f"{key}={work[key]:.1f}")
    if "n_ok" in work:
        bits.append(f"ok={work['n_ok']}/{work.get('n_requested', work.get('concurrency', '?'))}")
    if "error" in work:
        bits.append("error")
    return " ".join(bits[:4]) if bits else "done"


# ============================================================================
# Perplexity (llama-perplex on fixed GSM8K slice; per KV quant)
# ============================================================================


def load_ppl_corpus(items: int) -> str:
    """GSM8K questions as a stable perplexity corpus; fallback text if offline."""
    try:
        req = urllib.request.Request(GSM8K_RAW_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
        lines = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("question"):
                lines.append(obj["question"])
            if len(lines) >= items:
                break
        if lines:
            log(f"PPL corpus: {len(lines)} GSM8K questions")
            return "\n".join(lines)
        raise RuntimeError("no parsable problems")
    except Exception as exc:  # noqa: BLE001
        log(f"PPL corpus: GSM8K fetch failed ({exc}); using fallback text")
        return FALLBACK_TEXT


def run_perplexity(args: argparse.Namespace, bin_dir: Path, model_path: Path,
                   corpus: str, kv: str) -> dict:
    perplex_bin = bin_dir / "llama-perplexity"
    if not perplex_bin.exists():
        perplex_bin = bin_dir / "llama-perplex"
    if not perplex_bin.exists():
        return {"kv": kv, "error": f"neither llama-perplexity nor llama-perplex found in {bin_dir}"}
    corpus_path = Path(args.workdir) / "ppl_corpus_v2.txt"
    corpus_path.write_text(corpus, encoding="utf-8")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = so_dirs_from(bin_dir.parent)
    try:
        help_text = subprocess.run([str(perplex_bin), "--help"], capture_output=True,
                                   text=True, env=env, timeout=60).stdout or ""
    except Exception:  # noqa: BLE001
        help_text = ""
    cmd = [str(perplex_bin), "-m", str(model_path), "-f", str(corpus_path),
           "-c", str(PPL_CONTEXT), "-b", "512", "-ub", "512", "--tensor-split", "1,1"]
    if "--cache-type-k" in help_text:
        cmd.extend(["--cache-type-k", kv, "--cache-type-v", kv])
    if "--flash-attn" in help_text:
        cmd.extend(["--flash-attn", "on"])
    elif "-fa" in help_text:
        cmd.extend(["-fa", "on"])
    if "-fit" in help_text or "--fit" in help_text:
        cmd.extend(["-fit", "off"])
    log(f"{perplex_bin.name} (kv={kv}) ...")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=env,
                             timeout=args.request_timeout_s)
    except subprocess.TimeoutExpired:
        return {"kv": kv, "error": f"{perplex_bin.name} timed out"}
    text = (out.stdout or "") + "\n" + (out.stderr or "")
    match = re.search(r"pp=([\d.]+)", text) or re.search(r"[Pp]erplexity = ([\d.]+)", text)
    if not match:
        return {"kv": kv, "error": text[-600:]}
    return {"kv": kv, "perplexity": float(match.group(1))}


# ============================================================================
# Phase B -- KV cache matrix
# ============================================================================


def phase_b(args: argparse.Namespace, model_path: Path, bin_dir: Path,
            samples_out: list[dict], jsonl_path: Path,
            state: dict | None = None, results_holder: dict | None = None) -> list[dict]:
    records: list[dict] = []
    runs = getattr(args, "b_runs", None)
    if runs is None:
        runs = list(DEFAULT_B_TARGETED_CONFIGS)

    count = 0
    for ctx, kv, fa, reuse in runs:
        if is_oom_risk(kv, ctx) or (kv, ctx) in B_SKIP_KV_CTX:
            log(f"B: skipping kv={kv} ctx={ctx} (known OOM-class on 2x T4)")
            records.append({"phase": "b", "run_id": f"B.{kv}.c{ctx}", "config":
                            {"context": ctx, "kv": kv, "fa": fa, "reuse": reuse},
                            "fit": False, "skipped": True,
                            "error": "skipped: predicted OOM (large KV exceeds free 2x T4 VRAM)"})
            continue
        count += 1
        run_id = f"B.{kv}.fa{fa[0]}.ru{reuse[0]}.c{ctx}"
        cfg = RunCfg(run_id=run_id, phase="b", engine="llama",
                     context=ctx, kv=kv, fa=fa, reuse=reuse,
                     ngl=99, np=1, batch=1024, ubatch=512,
                     port=args.port)

        workloads = [("l1", lambda c=cfg: workload_l1(args, c, jsonl_path))]
        rec = run_or_resume(args, cfg, model_path, bin_dir,
                            workloads, samples_out, jsonl_path, state, results_holder)
        records.append(rec)
        if results_holder is not None:
            results_holder["b"] = records
            save_checkpoint(args.out_dir, state or {}, results_holder)

    # Quality: PPL per KV quant (fa on, reuse off) at context 4096.
    log("B: perplexity per KV quant ...")
    corpus = load_ppl_corpus(PPL_QUESTIONS * 2)
    for kv in PPL_KV_TYPES:
        if kv not in args.kv_types:
            continue
        ppl_run_id = f"B.ppl.{kv}"
        if state and getattr(args, "resume", True) and ppl_run_id in state.get("completed_runs", {}):
            log(f"  [checkpoint] Resuming completed run {ppl_run_id}")
            records.append(state["completed_runs"][ppl_run_id])
        else:
            rec = {"phase": "b", "run_id": ppl_run_id, "config": {"kv": kv},
                   **run_perplexity(args, bin_dir, model_path, corpus, kv)}
            records.append(rec)
            if state is not None:
                state.setdefault("completed_runs", {})[ppl_run_id] = rec
            if results_holder is not None:
                results_holder["b"] = records
                save_checkpoint(args.out_dir, state or {}, results_holder)
    log(f"B: done ({count} server runs + PPL)")
    return records


# ============================================================================
# Phase D -- context limits (progressive step probe)
# ============================================================================


def d_ctx_test(args: argparse.Namespace, cfg: RunCfg, model_path: Path, bin_dir: Path,
               samples_out: list[dict], jsonl_path: Path,
               state: dict | None = None, results_holder: dict | None = None) -> dict:
    """Boot at cfg.context and probe the full window (fill to ctx-512, generate 64)."""
    prompt = build_fill_prompt(cfg.context - 512, "Answer in one word.")

    def probe() -> dict:
        comp = one_request(args, cfg, prompt, FULL_CTX_GEN_TOKENS, 0.0,
                           FULL_CTX_REQUEST_TIMEOUT_S)
        record_request(jsonl_path, record_to_dict(cfg.run_id, "d.full_ctx", cfg.context, comp))
        return {"ok": comp.ok, "gen_tps": comp.gen_tps, "ttft_s": comp.ttft_s,
                "error": comp.error}

    return run_or_resume(args, cfg, model_path, bin_dir, [("full_ctx", probe)],
                         samples_out, jsonl_path, state, results_holder)


def d_test_fits(rec: dict) -> bool:
    works = rec.get("workloads", {}) or {}
    probe = works.get("full_ctx")
    return bool(rec.get("fit") and isinstance(probe, dict) and probe.get("ok"))


def phase_d(args: argparse.Namespace, model_path: Path, bin_dir: Path,
            samples_out: list[dict], jsonl_path: Path,
            state: dict | None = None, results_holder: dict | None = None) -> dict:
    records: list[dict] = []
    max_ctx_by_kv: dict[str, int] = {}
    steps = getattr(args, "d_steps", DEFAULT_D_STEPS)

    for kv in args.d_kv_types:
        log(f"D: progressive step probe for kv={kv} across {steps}")
        max_ctx = 0
        for ctx in steps:
            if is_oom_risk(kv, ctx):
                log(f"D: skipping kv={kv} ctx={ctx} (OOM predicted on 2x T4)")
                records.append({"phase": "d", "run_id": f"D.{kv}.c{ctx}",
                                "config": {"context": ctx, "kv": kv},
                                "fit": False, "skipped": True, "error": "predicted OOM"})
                break

            rec = d_ctx_test(args, RunCfg(run_id=f"D.{kv}.c{ctx}", phase="d", kv=kv,
                                          context=ctx, ngl=99, np=1, port=args.port),
                             model_path, bin_dir, samples_out, jsonl_path, state, results_holder)
            records.append(rec)
            if results_holder is not None:
                results_holder["d"] = {"records": records, "max_ctx_by_kv": max_ctx_by_kv}
                save_checkpoint(args.out_dir, state or {}, results_holder)

            if d_test_fits(rec):
                max_ctx = ctx
            else:
                log(f"D: context {ctx} failed; halting higher step probes for kv={kv}")
                break
        max_ctx_by_kv[kv] = max_ctx
        log(f"D: max verified ctx for kv={kv} = {max_ctx or 'none'}")

    # Offload boundary at the best q4_0 context (never below the known-good lo).
    anchor_ctx = min(max(steps), max_ctx_by_kv.get("q4_0", max(steps)))
    if anchor_ctx <= 0:
        anchor_ctx = args.d_lo
    log(f"D: offload boundary test at ctx={anchor_ctx}")
    for ngl in args.ngl_sweep:
        cfg = RunCfg(run_id=f"D.ngl{ngl}.c{anchor_ctx}", phase="d",
                     context=anchor_ctx, ngl=ngl, kv="q4_0", fa="on",
                     np=1, port=args.port)
        rec = run_or_resume(args, cfg, model_path, bin_dir,
                            [("l1", lambda c=cfg: workload_l1(args, c, jsonl_path))],
                            samples_out, jsonl_path, state, results_holder)
        rec["kind"] = "ngl"
        records.append(rec)
        if results_holder is not None:
            results_holder["d"] = {"records": records, "max_ctx_by_kv": max_ctx_by_kv, "anchor_ctx": anchor_ctx}
            save_checkpoint(args.out_dir, state or {}, results_holder)

    return {"phase": "d", "records": records, "max_ctx_by_kv": max_ctx_by_kv,
            "anchor_ctx": anchor_ctx}


# ============================================================================
# Phase C -- batching & concurrency
# ============================================================================


def phase_c(args: argparse.Namespace, model_path: Path, bin_dir: Path,
            samples_out: list[dict], jsonl_path: Path,
            state: dict | None = None, results_holder: dict | None = None) -> list[dict]:
    records: list[dict] = []
    cells = getattr(args, "c_cells", None)
    if cells is None:
        cells = list(DEFAULT_C_TARGETED_CELLS)

    for np_, b, ub in cells:
        run_id = f"C.llama.np{np_}.b{b}x{ub}"
        cfg = RunCfg(run_id=run_id, phase="c", engine="llama", context=args.c_context,
                     ngl=99, kv="q4_0", fa="on", reuse="off", np=np_,
                     batch=b, ubatch=ub, port=args.port)
        rec = run_cell(args, cfg, model_path, bin_dir, samples_out, jsonl_path, state, results_holder)
        records.append(rec)
        if results_holder is not None:
            results_holder["c"] = records
            save_checkpoint(args.out_dir, state or {}, results_holder)

    if args.include_vllm:
        if not ensure_vllm():
            records.append({"phase": "c", "run_id": "C.vllm", "fit": False,
                            "error": "vLLM unavailable (install failed); phase C2 skipped"})
        else:
            for nseq in args.vllm_max_num_seqs:
                for pc in args.vllm_prefix_cache:
                    run_id = f"C.vllm.ns{nseq}.pc{pc[0]}"
                    cfg = RunCfg(run_id=run_id, phase="c", engine="vllm",
                                 context=args.c_context,
                                 max_num_seqs=nseq, prefix_cache=pc,
                                 port=args.vllm_port)
                    rec = run_cell(args, cfg, model_path, bin_dir, samples_out, jsonl_path, state, results_holder)
                    records.append(rec)
                    if results_holder is not None:
                        results_holder["c"] = records
                        save_checkpoint(args.out_dir, state or {}, results_holder)
    return records


def run_cell(args: argparse.Namespace, cfg: RunCfg, model_path: Path, bin_dir: Path,
             samples_out: list[dict], jsonl_path: Path,
             state: dict | None = None, results_holder: dict | None = None) -> dict:
    def l1():
        return "warm", workload_l1(args, cfg, jsonl_path)

    def make_l2(concurrency: int):
        return f"l2.c{concurrency}", lambda: workload_l2(args, cfg, concurrency, jsonl_path)

    workloads: list[tuple[str, Callable[[], dict]]] = [("warm", lambda: workload_l1(args, cfg, jsonl_path))]
    for conc in args.closed_concurrencies:
        workloads.append((f"l2.c{conc}", lambda c=conc: workload_l2(args, cfg, c, jsonl_path)))
    for rate in args.poisson_rates:
        workloads.append(("l3.r" + str(rate).replace(".", "_") + "x" + str(args.poisson_n),
                          lambda r=rate: workload_l3(args, cfg, r, args.poisson_n, jsonl_path)))
    return run_or_resume(args, cfg, model_path, bin_dir, workloads, samples_out, jsonl_path, state, results_holder)


# ============================================================================
# Synthesis: SLO filter + Pareto + recommendation
# ============================================================================


def synthesize(c_records: list[dict], d_result: dict, args: argparse.Namespace) -> dict:
    def cell_summary(rec: dict) -> dict | None:
        if not rec.get("fit"):
            return None
        works = rec.get("workloads", {})
        l3 = {k: v for k, v in works.items() if k.startswith("l3") and "arrival_rate_per_s" in v}
        if not l3:
            return None
        cfg = rec.get("config", {})
        worst_itl = max((v.get("itl_p95_s") or 0.0) for v in l3.values())
        worst_ttft = max((v.get("ttft_p95_s") or 0.0) for v in l3.values())
        failures = sum(v.get("timeouts_or_failed", 0) for v in l3.values())
        ceiling = 0.0
        per_rate = []
        for v in sorted(l3.values(), key=lambda item: item["arrival_rate_per_s"]):
            per_rate.append({
                "rate_per_s": v["arrival_rate_per_s"],
                "system_tok_s": v.get("system_tok_s"),
                "ttft_p95_s": v.get("ttft_p95_s"),
                "itl_p95_s": v.get("itl_p95_s"),
                "timeouts": v.get("timeouts_or_failed"),
            })
            if (v.get("itl_p95_s") or 9e9) <= args.slo_itl_s and not v.get("timeouts_or_failed"):
                ceiling = max(ceiling, v["arrival_rate_per_s"])
        return {
            "run_id": rec["run_id"],
            "engine": cfg.get("engine"),
            "batching": cfg.get("max_num_seqs", cfg.get("np")),
            "prefix_cache": cfg.get("prefix_cache"),
            "batch_ubatch": f"{cfg.get('batch')}/{cfg.get('ubatch')}",
            "peak_vram_gb": round((rec.get("peak_vram_mb") or 0) / 1024, 2),
            "worst_itl_p95_s": round(worst_itl, 4),
            "worst_ttft_p95_s": round(worst_ttft, 3),
            "failures": failures,
            "slo_pass": bool(worst_itl <= args.slo_itl_s and worst_ttft <= args.slo_ttft_s
                             and failures == 0),
            "arrival_ceiling_rate": ceiling,
            "per_rate": per_rate,
        }

    summaries = [s for s in (cell_summary(r) for r in c_records) if s is not None]
    for s in summaries:
        s["score"] = round(
            (s["per_rate"][-1]["system_tok_s"] or 0.0) if s["per_rate"] else 0.0, 3)
    passing = [s for s in summaries if s["slo_pass"]]
    candidates = passing or summaries
    recommended = None
    if candidates:
        recommended = max(
            candidates,
            key=lambda s: (s["score"], s["peak_vram_gb"] * -1),
        )
    if recommended:
        batch_val, ubatch_val = (recommended.get("batch_ubatch") or "1024/512").split("/")
        recommended["suggested_launcher_config"] = {
            "context_size": d_result.get("anchor_ctx", 131072),
            "kv_cache_k": "q4_0",
            "kv_cache_v": "q4_0",
            "batch_size": int(batch_val),
            "ubatch_size": int(ubatch_val),
            "parallel_slots_np": recommended.get("batching", 1),
            "flash_attn": "on",
            "tensor_split": "1,1",
        }

    frontier = sorted(
        ((s["run_id"], s["score"], s["peak_vram_gb"], s["arrival_ceiling_rate"])
         for s in summaries),
        key=lambda t: (-t[1], t[2]))[:12]
    return {
        "slo": {"ttft_p95_s": args.slo_ttft_s, "itl_p95_s": args.slo_itl_s},
        "llama_cells": [s for s in summaries if s["engine"] == "llama"],
        "vllm_cells": [s for s in summaries if s["engine"] == "vllm"],
        "max_ctx_by_kv": d_result.get("max_ctx_by_kv", {}),
        "nvl_boundary_context": d_result.get("anchor_ctx"),
        "recommended": recommended,
        "frontier": frontier,
        "rationale": (
            "SLO: worst-case ITL_p95 <= slo_itl and TTFT_p95 <= slo_ttft across all L3 rates, "
            "no timeouts. Recommendation = highest system tok/s at the top arrival rate "
            "among SLO-passing cells (tie: lower peak VRAM). arrival_ceiling_rate = highest "
            "Poisson rate still meeting the ITL SLO."
        ),
    }


# ============================================================================
# Output writers
# ============================================================================


def write_outputs(out_dir: Path, results: dict, synthesis: dict,
                  samples: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results_v2.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "sweet_spot_v2.json").write_text(
        json.dumps(synthesis, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_rows: list[dict] = []

    def add(phase: str, run_id: str, profile: str, metric: str, value) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            csv_rows.append({"phase": phase, "run_id": run_id, "profile": profile,
                             "metric": metric, "value": value})

    for rec in results.get("b", []) + results.get("c", []) + (
            results.get("d", {}).get("records", []) if isinstance(results.get("d"), dict) else []):
        phase = rec.get("phase", "?")
        run_id = rec.get("run_id", "?")
        add(phase, run_id, "", "idle_vram_mb", rec.get("idle_vram_mb"))
        add(phase, run_id, "", "peak_vram_mb", rec.get("peak_vram_mb"))
        add(phase, run_id, "", "peak_ram_gb", rec.get("peak_ram_gb"))
        if "perplexity" in rec:
            add(phase, run_id, "", "perplexity", rec.get("perplexity"))
        for name, work in (rec.get("workloads") or {}).items():
            if not isinstance(work, dict):
                continue
            for key, value in work.items():
                add(phase, run_id, name, key, value)

    with open(out_dir / "results_v2.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["phase", "run_id", "profile", "metric", "value"])
        writer.writeheader()
        writer.writerows(csv_rows)

    if samples:
        with open(out_dir / "samples_v2.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["ts", "run_id", "vram_mb", "util_pct", "power_w",
                                "clock_mhz", "temp_c", "ram_gb"])
            writer.writeheader()
            writer.writerows(samples)

    md = ["# Ornith-1.5-35B-A3B Q4_K_M -- v2 Throughput & Context Benchmark", ""]
    md += ["## Phase B: KV cache matrix", "",
           "| run | ctx | kv | fa | reuse | fit | short tok/s | long tok/s | long TTFT (s) | peak VRAM (GB) | KV (MB/1k tok) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for rec in results.get("b", []):
        cfg = rec.get("config", {})
        works = rec.get("workloads", {}) or {}
        l1 = works.get("l1", {}) or {}
        ctx = cfg.get("context", "")
        kv = cfg.get("kv", "")
        kv_growth = None
        if rec.get("peak_vram_mb") and rec.get("idle_vram_mb") and ctx:
            kv_growth = (rec["peak_vram_mb"] - rec["idle_vram_mb"]) / (int(ctx) / 1024)
        md.append(
            f"| {rec.get('run_id')} | {ctx} | {kv} | {cfg.get('fa', '-')} "
            f"| {cfg.get('reuse', '-')} | {rec.get('fit')} "
            f"| {fmt(l1.get('short_tps')) if isinstance(l1, dict) else 'n/a'} "
            f"| {fmt(l1.get('long_gen_tps')) if isinstance(l1, dict) else 'n/a'} "
            f"| {fmt(l1.get('long_ttft_s')) if isinstance(l1, dict) else 'n/a'} "
            f"| {round((rec.get('peak_vram_mb') or 0) / 1024, 1)} "
            f"| {f'{kv_growth:.0f}' if kv_growth else 'n/a'} |")
    ppl_rows = [r for r in results.get("b", []) if "perplexity" in r]
    if ppl_rows:
        md += ["", "## Perplexity by KV quant", "",
               "| kv | PPL |", "|---|---|"] + [
            f"| {r['kv']} | {r['perplexity']:.3f} |" for r in ppl_rows]
    d = results.get("d") or {}
    if isinstance(d, dict):
        md += ["", "## Phase D: context limits", "",
               f"max_ctx_by_kv: {d.get('max_ctx_by_kv')}", "",
               "| run | tested ctx | fit | peak VRAM (GB) |", "|---|---|---|---|"]
        for rec in d.get("records", []):
            md.append(f"| {rec.get('run_id')} | {rec.get('config', {}).get('context')} "
                      f"| {rec.get('fit')} | {round((rec.get('peak_vram_mb') or 0) / 1024, 1)} |")
    md += ["", "## Phase C: batching & concurrency (per-rate detail in results_v2.json)", ""]
    for cell in synthesis.get("llama_cells", []) + synthesis.get("vllm_cells", []):
        md.append(f"- **{cell['run_id']}** engine={cell['engine']} batching={cell['batching']} "
                  f"peak_vram={cell['peak_vram_gb']}GB slo_pass={cell['slo_pass']} "
                  f"worst_itl_p95={cell['worst_itl_p95_s']} worst_ttft_p95={cell['worst_ttft_p95_s']} "
                  f"ceiling={cell['arrival_ceiling_rate']} req/s")
    rec = synthesis.get("recommended")
    md += ["", "## Recommended deployment config",
           f"```json\n{json.dumps(rec, indent=2)}\n```" if rec else "none",
           "", "Criteria: " + synthesis["rationale"]]
    if rec and "suggested_launcher_config" in rec:
        sc = rec["suggested_launcher_config"]
        md += [
            "", "### Copy-Paste for Kaggle Launcher / serve_ornith.py",
            "```python",
            f"CONTEXT_SIZE = {sc['context_size']}",
            f"BATCH_SIZE = {sc['batch_size']}",
            f"UBATCH_SIZE = {sc['ubatch_size']}",
            f"KV_CACHE_TYPE = \"{sc['kv_cache_k']}\"",
            "```"
        ]
    (out_dir / "summary_v2.md").write_text("\n".join(md), encoding="utf-8")
    log(f"outputs written to {out_dir}")


# ============================================================================
# CLI
# ============================================================================


def apply_quick(args: argparse.Namespace) -> None:
    """Shrink every matrix to a ~15-20 min minimal smoke run."""
    args.b_runs = [(32768, "q4_0", "on", "off")]
    args.b_ctx_sizes = [32768]
    args.kv_types = ["q4_0"]
    args.c_context = 16384
    args.c_cells = [(1, 512, 512), (4, 512, 512)]
    args.closed_concurrencies = [1, 4]
    args.poisson_rates = [2.0]
    args.poisson_n = 8
    args.d_steps = [32768, 65536]
    args.d_hi = 65536
    args.d_kv_types = ["q4_0"]
    args.ngl_sweep = [99]
    log("--quick active: reduced matrices for a fast smoke run (~15-20 min)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    argv = strip_kernel_args(raw_argv)
    parser = argparse.ArgumentParser(
        description="v2 throughput & context benchmark (KV matrix, context bisection, "
                    "batching/concurrency) for Ornith-1.5-35B-A3B Q4_K_M on 2xT4.")
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument("--include-vllm", action="store_true",
                        help="Phase C2: install + benchmark vLLM (PagedAttention)")
    parser.add_argument("--skip", action="append", choices=["b", "d", "c"],
                        default=[], help="Phases to skip (repeatable)")
    parser.add_argument("--quick", action="store_true", help="Reduced smoke matrices (~15-20 min)")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="Resume from existing checkpoint if available (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Ignore existing checkpoint and run all benchmarks from scratch")
    parser.add_argument("--b-ctx", default=",".join(str(c) for c in DEFAULT_B_CTX_SIZES))
    parser.add_argument("--kv-types", default=",".join(DEFAULT_KV_TYPES))
    parser.add_argument("--c-context", type=int, default=DEFAULT_C_CONTEXT)
    parser.add_argument("--np-values", default=",".join(str(n) for n in DEFAULT_NP_VALUES))
    parser.add_argument("--batch-pairs", default="1024x512,2048x512,512x512",
                        help="Comma list of <batch>x<ubatch>")
    parser.add_argument("--closed-conc", default=",".join(str(n) for n in DEFAULT_CLOSED_CONCURRENCIES))
    parser.add_argument("--poisson-rates", default=",".join(str(r) for r in DEFAULT_POISSON_RATES))
    parser.add_argument("--poisson-n", type=int, default=DEFAULT_POISSON_N,
                        help="Requests per Poisson arrival-rate cell")
    parser.add_argument("--vllm-max-num-seqs", default=",".join(str(n) for n in DEFAULT_VLLM_MAX_NUM_SEQS))
    parser.add_argument("--vllm-prefix-cache", default=",".join(DEFAULT_VLLM_PREFIX_CACHE))
    parser.add_argument("--d-steps", default=",".join(str(s) for s in DEFAULT_D_STEPS),
                        help="Comma-separated context sizes for Phase D progressive probe")
    parser.add_argument("--d-lo", type=int, default=DEFAULT_D_LO)
    parser.add_argument("--d-hi", type=int, default=DEFAULT_D_HI)
    parser.add_argument("--d-kv", default=",".join(DEFAULT_D_KV_TYPES))
    parser.add_argument("--ngl-sweep", default=",".join(str(n) for n in DEFAULT_NGL_SWEEP))
    parser.add_argument("--slo-ttft-s", type=float, default=DEFAULT_SLO_TTFT_S)
    parser.add_argument("--slo-itl-s", type=float, default=DEFAULT_SLO_ITL_S)
    parser.add_argument("--server-ready-timeout-s", type=int, default=SERVER_READY_TIMEOUT_S)
    parser.add_argument("--request-timeout-s", type=int, default=REQUEST_TIMEOUT_S)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    args.b_ctx_sizes = [int(v) for v in args.b_ctx.split(",") if v.strip()]
    args.kv_types = [v.strip() for v in args.kv_types.split(",") if v.strip()]
    args.np_values = [int(v) for v in args.np_values.split(",") if v.strip()]
    args.batch_pairs = [
        tuple(int(x) for x in pair.split("x")) for pair in args.batch_pairs.split(",") if pair.strip()
    ]
    args.closed_concurrencies = [int(v) for v in args.closed_conc.split(",") if v.strip()]
    args.poisson_rates = [float(v) for v in args.poisson_rates.split(",") if v.strip()]
    args.vllm_max_num_seqs = [int(v) for v in args.vllm_max_num_seqs.split(",") if v.strip()]
    args.vllm_prefix_cache = [v.strip() for v in args.vllm_prefix_cache.split(",") if v.strip()]
    args.d_steps = [int(v) for v in args.d_steps.split(",") if v.strip()]
    args.d_kv_types = [v.strip() for v in args.d_kv.split(",") if v.strip()]
    args.ngl_sweep = [int(v) for v in args.ngl_sweep.split(",") if v.strip()]
    args.skip = list(set(args.skip))
    args.workdir = Path(args.workdir) if args.workdir else default_workdir()
    args.out_dir = Path(args.out_dir) if args.out_dir else default_outdir()

    # Determine cell and run matrices
    if any(arg in raw_argv for arg in ("--batch-pairs", "--np-values")):
        args.c_cells = [
            (np_, b, ub)
            for np_ in args.np_values
            for (b, ub) in args.batch_pairs
        ]
    else:
        args.c_cells = list(DEFAULT_C_TARGETED_CELLS)

    if any(arg in raw_argv for arg in ("--b-ctx", "--kv-types")):
        args.b_runs = [
            (ctx, kv, fa, "off")
            for ctx in args.b_ctx_sizes
            for kv in args.kv_types
            for fa in ("on", "off")
        ]
    else:
        args.b_runs = list(DEFAULT_B_TARGETED_CONFIGS)

    if args.quick:
        apply_quick(args)
    return args


def main(argv: list[str] | None = None) -> None:
    """Run the v2 benchmark plan (B, D, C) and write synthesis outputs."""
    args = parse_args(argv)
    stop_all_servers()
    args.workdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "requests_v2.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    state = load_checkpoint(args.out_dir) if args.resume else {"completed_runs": {}, "samples": []}
    samples_out: list[dict] = state.get("samples", [])

    results: dict = {"meta": {
        "model_repo": args.model_repo, "model_file": args.model_file,
        "phases": {"b": "b" not in args.skip, "d": "d" not in args.skip,
                   "c": "c" not in args.skip, "vllm": args.include_vllm},
        "slo": {"ttft_p95_s": args.slo_ttft_s, "itl_p95_s": args.slo_itl_s},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }}

    log("Provisioning model and llama.cpp ...")
    model_path = get_or_download_model(args.model_repo, args.model_file, args.workdir / "models")
    bin_dir = install_llama_cpp(args.workdir)

    if "b" not in args.skip:
        log("Phase B: KV cache matrix")
        results["b"] = phase_b(args, model_path, bin_dir, samples_out, jsonl_path, state, results)
        save_checkpoint(args.out_dir, state, results)

    if "d" not in args.skip:
        log("Phase D: context scaling limits (progressive step probe)")
        results["d"] = phase_d(args, model_path, bin_dir, samples_out, jsonl_path, state, results)
        save_checkpoint(args.out_dir, state, results)

    if "c" not in args.skip:
        log("Phase C: batching & concurrency")
        results["c"] = phase_c(args, model_path, bin_dir, samples_out, jsonl_path, state, results)
        save_checkpoint(args.out_dir, state, results)
    else:
        results["c"] = []

    stop_all_servers()
    d_result = results.get("d", {})
    synthesis = synthesize(results.get("c", []), d_result, args)
    results["sweet_spot"] = synthesis
    state["samples"] = samples_out
    save_checkpoint(args.out_dir, state, results, synthesis)
    write_outputs(args.out_dir, results, synthesis, samples_out)

    rec = synthesis.get("recommended")
    if rec:
        log(f"RECOMMENDED -> {rec['run_id']} engine={rec['engine']} "
            f"batching={rec['batching']} peak_vram={rec['peak_vram_gb']}GB "
            f"ceiling={rec['arrival_ceiling_rate']} req/s")
    else:
        log("No SLO-passing configuration found; see sweet_spot_v2.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_all_servers()
        sys.exit(130)
    finally:
        stop_all_servers()
