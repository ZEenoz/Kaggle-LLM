#!/usr/bin/env python3
"""End-to-end benchmarking harness for Nex-N2.5-mini (Q4_K_M) on 2x Tesla T4.

Phases (each independently skippable via --skip):
    speed -- llama-server context/offload sweep; measures TTFT, prompt tok/s and
             generation tok/s while a background thread samples VRAM (nvidia-smi)
             and system RAM (/proc/meminfo).
    ppl   -- perplexity over a fixed GSM8K text slice via llama-perplex.
    eval  -- reasoning accuracy on a GSM8K subset at the reference context.
    temp  -- temperature sweep of reasoning accuracy + determinism check at 0.0.

Outputs (written to --out-dir, default /kaggle/working on Kaggle):
    results.json    full nested results
    results.csv     tidy long-format rows (phase, config, metric, value, unit)
    sweet_spot.json recommended deployment config + rationale + ranking
    summary.md      human-readable tables

Run on Kaggle (GPU T4 x 2, Internet ON):
    python benchmark_q4.py
    python benchmark_q4.py --context-sizes 8192,32768,131072
    python benchmark_q4.py --skip speed --num-questions 24

Inside a Jupyter/Kaggle notebook cell (the kernel's injected ``-f <kernel.json>``
argv is stripped automatically):
    import benchmark_q4
    benchmark_q4.main([])                       # defaults
    benchmark_q4.main(["--skip", "speed"])      # quality phases only

The script is self-contained: it fetches the model (HF / mounted Kaggle
dataset), fetches a prebuilt CUDA llama.cpp, and orchestrates everything.
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
# Configuration (single source of truth; all overridable via CLI arguments)
# ============================================================================

DEFAULT_MODEL_REPO = "abenzerps/Nex-N2.5-mini-GGUF"
DEFAULT_MODEL_FILE = "Nex-N2.5-Mini-Q4_K_M.gguf"
DEFAULT_MODEL_ALIAS = "nex-bench"
DEFAULT_API_KEY = "kilo-bench-key"
DEFAULT_PORT = 8080
DEFAULT_KV_CACHE_TYPE = "q4_0"
DEFAULT_BATCH_SIZE = 1024
DEFAULT_UBATCH_SIZE = 512
DEFAULT_CONTEXT_SIZES = (8192, 16384, 32768, 65536, 98304, 131072)
DEFAULT_NGL_VALUES = (99,)
DEFAULT_TEMPERATURES = (0.0, 0.2, 0.5, 0.7, 1.0)
DEFAULT_MIN_GEN_TPS = 15.0
DEFAULT_NUM_QUESTIONS = 16
DEFAULT_EVAL_MAX_TOKENS = 300
DEFAULT_PPL_CONTEXT = 4096
DEFAULT_SERVER_READY_TIMEOUT_S = 900
DEFAULT_REQUEST_TIMEOUT_S = 1800
MEMORY_SAMPLE_INTERVAL_S = 0.5

LLAMA_API_BASE = "https://api.github.com/repos/cloudlnkcn/llama.cpp/releases?per_page=5"
LLAMA_FALLBACK_URL = (
    "https://github.com/ai-dock/llama.cpp-cuda/releases/download/b9628/"
    "llama.cpp-b9628-cuda-12.8-amd64.tar.gz"
)
USER_AGENT = "nex25-benchmark/1.0"
GSM8K_RAW_URL = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test.jsonl"
GOLD_ANSWER_RE = re.compile(r"Answer: ([\d,]+(?:\.\d+)?)")
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog near the old stone mill. "

# Deterministic fallback used only if the GSM8K download fails.
FALLBACK_PROBLEMS: list[dict[str, str]] = [
    {"question": "A farmer has 42 chickens. He sells 12 and buys 8 more. How many chickens does he have now?", "answer": "38"},
    {"question": "Maria buys 3 notebooks at $4 each and a pen for $2. She pays with a $20 bill. How much change does she get?", "answer": "6"},
    {"question": "A train travels 60 km in 45 minutes. At the same speed, how many km does it travel in 3 hours?", "answer": "240"},
    {"question": "Tom reads 24 pages a day. How many pages does he read in 2 weeks?", "answer": "336"},
    {"question": "A shop has 15 blue pens and 23 green pens. It sells 11 blue pens. How many pens are left in total?", "answer": "27"},
    {"question": "A recipe needs 3 cups of flour for 12 muffins. How many cups of flour are needed for 36 muffins?", "answer": "9"},
    {"question": "Jane earns $18 per hour. She works 5 hours on Monday and 3 hours on Tuesday. How much does she earn in total?", "answer": "144"},
    {"question": "A tank holds 200 liters of water and 45 liters are used each day. How many full days can the tank last?", "answer": "4"},
    {"question": "Lucy is 6 years old and her sister is three times as old. How old will Lucy's sister be in 4 years?", "answer": "22"},
    {"question": "A box contains 24 red balls and half as many blue balls. How many balls are in the box in total?", "answer": "36"},
    {"question": "A movie is 135 minutes long and the show includes a 15 minute break. What is the total running time in hours?", "answer": "2.5"},
    {"question": "Peter has 5 shelves, each holding 12 books. If 7 books are missing, how many books are on the shelves?", "answer": "53"},
    {"question": "A pencil costs $0.50 and an eraser costs $0.25. How much do 4 pencils and 2 erasers cost together?", "answer": "2.5"},
    {"question": "A car uses 8 liters of fuel per 100 km. How many liters are needed for a 350 km trip?", "answer": "28"},
    {"question": "In a class of 30 students, 18 are girls. How many more boys are needed so boys outnumber girls by 2?", "answer": "8"},
    {"question": "A bakery sold 48 cupcakes in the morning and twice as many in the afternoon. How many cupcakes were sold in total?", "answer": "144"},
]


@dataclass
class CompletionResult:
    """Result of one (streaming or not) chat completion request."""

    ok: bool
    text: str = ""
    ttft_s: float | None = None
    elapsed_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    gen_tps: float | None = None
    prompt_tps: float | None = None
    error: str = ""

    @property
    def metrics(self) -> dict[str, float | int | None]:
        return {
            "ttft_s": self.ttft_s,
            "elapsed_s": self.elapsed_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "gen_tps": self.gen_tps,
            "prompt_tps": self.prompt_tps,
        }


# ============================================================================
# Small helpers
# ============================================================================


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_kaggle() -> bool:
    return Path("/kaggle").exists()


def default_workdir() -> Path:
    return Path("/kaggle/tmp/nex25_bench" if is_kaggle() else "/tmp/nex25_bench")


def default_outdir() -> Path:
    return Path("/kaggle/working" if is_kaggle() else Path.cwd() / "bench_results")


def http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def stop_process(proc: "subprocess.Popen | None", patterns: tuple[str, ...]) -> None:
    """Terminate a launched process (optionally via pkill patterns)."""
    for pattern in patterns:
        subprocess.run(["pkill", "-9", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


# ============================================================================
# Model + llama.cpp provisioning (same strategy as kernel/serve_nex.py)
# ============================================================================


def get_or_download_model(repo: str, filename: str, model_dir: Path) -> Path:
    """Locate the GGUF on a mounted dataset, on disk, or download it."""
    model_dir.mkdir(parents=True, exist_ok=True)

    if Path("/kaggle/input").exists():
        mounted = [p for p in Path("/kaggle/input").rglob("*.gguf") if "mmproj" not in p.name.lower()]
        for m in mounted:
            if "nex" in m.name.lower() or "q4_k_m" in m.name.lower() or filename.lower() in m.name.lower():
                log(f"Using mounted dataset model: {m.name} ({m.stat().st_size / 1024**3:.2f} GB)")
                return m

    target = model_dir / filename
    if target.exists() and target.stat().st_size > 1024**3:
        log(f"Using cached model: {target} ({target.stat().st_size / 1024**3:.2f} GB)")
        return target

    log(f"Downloading {repo}/{filename} (~21.7 GB)...")
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(model_dir))
        return Path(downloaded)
    except Exception as exc:  # noqa: BLE001 - any download backend failure falls back
        log(f"hf_hub_download failed ({exc}); falling back to direct download")
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        if shutil.which("aria2c"):
            subprocess.run(
                ["aria2c", "-x", "16", "-s", "16", "-k", "1M", "-d", str(model_dir), "-o", filename, url],
                check=True,
            )
        else:
            subprocess.run(["wget", "--continue", "-O", str(target), url], check=True)
    log(f"Model ready: {target} ({target.stat().st_size / 1024**3:.2f} GB)")
    return target


def install_llama_cpp(workdir: Path) -> Path:
    """Fetch a prebuilt CUDA llama.cpp into workdir; return the bin directory."""
    bin_dir = workdir / "bin"
    server_bin = bin_dir / "llama-server"
    if server_bin.exists() and os.access(server_bin, os.X_OK):
        log(f"llama.cpp already installed at {bin_dir}")
        return bin_dir

    found = [p for p in workdir.rglob("llama-server") if p.is_file() and os.access(p, os.X_OK)]
    if found:
        shutil.copy2(found[0], server_bin)
        return server_bin.parent

    log("Downloading prebuilt llama.cpp (sm_75) ...")
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
    except Exception as exc:  # noqa: BLE001
        log(f"GitHub API warning: {exc}; using fallback release")

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
        server_bin.symlink_to(real)
    for tool in ("llama-perplex", "llama-bench"):
        src = real.parent / tool
        dst = bin_dir / tool
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
    log(f"llama.cpp ready: {bin_dir}")
    return bin_dir


def so_dirs_from(workdir: Path) -> str:
    dirs = {str(p.parent.resolve()) for p in workdir.rglob("*.so*") if p.is_file()}
    return ":".join(dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")


# ============================================================================
# Memory sampling
# ============================================================================


class MemorySampler(threading.Thread):
    """Background thread sampling GPU VRAM (nvidia-smi) and system RAM (/proc/meminfo)."""

    def __init__(self, interval_s: float = MEMORY_SAMPLE_INTERVAL_S) -> None:
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.peak_gpu_mb: int = 0
        self.peak_ram_gb: float = 0.0
        self.total_gpu_mb: int = 0
        self._stop_event = threading.Event()

    @property
    def available(self) -> bool:
        return shutil.which("nvidia-smi") is not None

    def snapshot(self) -> tuple[int, float]:
        """Return (used vram across all GPUs in MB, used system RAM in GB)."""
        gpu_mb, ram_gb = 0, 0.0
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                text=True,
                timeout=10,
            )
            for line in out.strip().splitlines():
                used, total = (int(v) for v in line.split(","))
                gpu_mb += used
                self.total_gpu_mb += total
        except Exception:  # noqa: BLE001 - sampling must never crash the run
            pass
        try:
            info = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, val = line.split(":", 1)
                info[key.strip()] = int(val.strip().split()[0])  # kB
            ram_gb = (info["MemTotal"] - info["MemAvailable"]) / 1024**2
        except Exception:  # noqa: BLE001 - non-Linux or transient
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
# llama-server lifecycle + OpenAI-compatible client (stdlib only)
# ============================================================================


def build_server_cmd(
    server_bin: Path,
    model_path: Path,
    args: argparse.Namespace,
    context: int,
    ngl: int,
    log_path: Path,
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
) -> "subprocess.Popen | None":
    """Launch llama-server with the given context/offload; return the Popen or None."""
    stop_process(None, ("[l]lama-server",))
    cmd = build_server_cmd(bin_dir / "llama-server", model_path, args, context, ngl, log_path)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = so_dirs_from(bin_dir.parent)
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["GGML_CUDA_NO_VMM"] = "1"
    try:
        help_text = subprocess.run([str(bin_dir / "llama-server"), "--help"],
                                   capture_output=True, text=True, env=env, timeout=60).stdout or ""
    except Exception:  # noqa: BLE001
        help_text = ""
    if "--flash-attn" in help_text:
        cmd.extend(["--flash-attn", "on"])
    elif "-fa" in help_text:
        cmd.extend(["-fa", "on"])
    log(f"Starting llama-server (ctx={context}, ngl={ngl}) ...")
    server_log = open(log_path, "w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT, env=env, text=True)
    return proc


def wait_for_server(args: argparse.Namespace, proc: "subprocess.Popen | None", timeout: int) -> bool:
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
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    return False


def server_log_tail(log_path: Path, lines: int = 25) -> str:
    try:
        return "\n".join(log_path.read_text(errors="replace").splitlines()[-lines:])
    except Exception:  # noqa: BLE001
        return ""


def stream_completion(
    args: argparse.Namespace,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> CompletionResult:
    """One chat completion with streaming SSE parsing (stdlib urllib only)."""
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
    except Exception as exc:  # noqa: BLE001
        result.elapsed_s = time.time() - t0
        result.error = str(exc)

    if result.ok and result.ttft_s is not None:
        t_gen = result.elapsed_s - result.ttft_s
        if t_gen > 0:
            if result.completion_tokens:
                result.gen_tps = result.completion_tokens / t_gen
            if result.prompt_tokens and result.ttft_s > 0:
                result.prompt_tps = result.prompt_tokens / result.ttft_s
    return result


def build_fill_prompt(target_tokens: int, suffix: str) -> str:
    """Build a deterministic filler prompt of roughly target_tokens tokens."""
    target_chars = max(target_tokens * 4, 512)
    filler = (FILLER_SENTENCE * (target_chars // len(FILLER_SENTENCE) + 1))[:target_chars]
    return filler + suffix


def fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


# ============================================================================
# Phase 1: speed + resource sweep
# ============================================================================


def run_speed_sweep(args: argparse.Namespace, model_path: Path, bin_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for context in args.context_sizes:
        for ngl in args.ngl_values:
            log_path = Path(args.workdir) / f"server_ctx{context}_ngl{ngl}.log"
            row: dict = {
                "phase": "speed",
                "config": {"context": context, "ngl": ngl, "kv_cache": args.kv_cache,
                           "batch": args.batch_size, "ubatch": args.ubatch_size},
                "fit": False,
                "error": "",
            }
            sampler = MemorySampler()
            proc = start_server(bin_dir, model_path, args, context, ngl, log_path)
            if proc is None or not wait_for_server(args, proc, args.server_ready_timeout_s):
                row["error"] = "server failed to start (OOM?)\n" + server_log_tail(log_path)
                log(f"  !! ctx={context} ngl={ngl}: server did not become ready")
                stop_process(proc, ("[l]lama-server",))
                rows.append(row)
                continue

            sampler.start()
            idle_gpu_mb, idle_ram_gb = sampler.snapshot()
            row["idle_vram_mb"], row["idle_ram_gb"] = idle_gpu_mb, idle_ram_gb

            # Warmup (loads prompt pipeline, avoids cold-cache distortion)
            warmup = stream_completion(args, "Say ok.", max_tokens=8, temperature=0.0)
            if not warmup.ok:
                row["error"] = "warmup request failed: " + warmup.error
                sampler.stop()
                stop_process(proc, ("[l]lama-server",))
                rows.append(row)
                continue
            time.sleep(1.0)

            short = stream_completion(args, "Explain what a hash map is in one short paragraph.",
                                      max_tokens=128, temperature=0.0)
            long_prompt_tokens = min(context // 2, 20000)
            long_prompt = build_fill_prompt(long_prompt_tokens,
                                             "Question: what is the capital of France? Answer in one word.")
            long = stream_completion(args, long_prompt, max_tokens=128, temperature=0.0)

            sampler.stop()
            stop_process(proc, ("[l]lama-server",))

            row.update({
                "fit": bool(short.ok),
                "peak_vram_mb": sampler.peak_gpu_mb,
                "peak_ram_gb": sampler.peak_ram_gb,
                "total_vram_mb": sampler.total_gpu_mb,
                "short": {"prompt_tokens_approx": long_prompt_tokens, **short.metrics},
                "long": {"prompt_tokens_approx": long_prompt_tokens, **long.metrics},
            })
            if not short.ok:
                row["error"] = short.error
            log(f"  ctx={context:>7d} ngl={ngl:>2d} fit={row['fit']} "
                f"short_tps={fmt(short.gen_tps)} long_tps={fmt(long.gen_tps)} "
                f"long_ttft={fmt(long.ttft_s)}s peak_vram={sampler.peak_gpu_mb / 1024:.1f}GB")
            rows.append(row)
    return rows


# ============================================================================
# Phase 2: perplexity (llama-perplex, fixed corpus slice -> relative signal)
# ============================================================================


def load_gsm8k(items: int) -> list[dict[str, str]]:
    """Fetch the first `items` GSM8K test rows; fall back to the embedded set."""
    try:
        raw = http_get(GSM8K_RAW_URL, timeout=60).decode("utf-8")
        problems = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            gold = GOLD_ANSWER_RE.search(obj.get("answer", ""))
            if obj.get("question") and gold:
                problems.append({"question": obj["question"], "answer": gold.group(1).replace(",", "")})
            if len(problems) >= items:
                break
        if problems:
            log(f"GSM8K: fetched {len(problems)} problems from HuggingFace")
            return problems
        raise RuntimeError("GSM8K file downloaded but no parsable problems found")
    except Exception as exc:  # noqa: BLE001
        log(f"GSM8K fetch failed ({exc}); using embedded fallback set")
        return FALLBACK_PROBLEMS[: max(items, len(FALLBACK_PROBLEMS))]


def run_perplexity(args: argparse.Namespace, bin_dir: Path, model_path: Path,
                   problems: list[dict[str, str]]) -> dict:
    perplex_bin = bin_dir / "llama-perplex"
    if not perplex_bin.exists():
        return {"phase": "ppl", "error": f"{perplex_bin} not available"}

    # Use a slice AFTER the eval slice to avoid overlap between quality signals.
    slice_start = min(args.num_questions, max(len(problems) - 40, 0))
    corpus_lines = [p["question"] for p in problems[slice_start: slice_start + 40]] or \
                   [p["question"] for p in problems]
    corpus_path = Path(args.workdir) / "ppl_corpus.txt"
    corpus_path.write_text("\n".join(corpus_lines) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = so_dirs_from(bin_dir.parent)
    cmd = [
        str(perplex_bin),
        "-m", str(model_path),
        "-f", str(corpus_path),
        "-c", str(args.ppl_context),
        "-b", "512",
        "-ub", "512",
        "--tensor-split", "1,1",
    ]
    log(f"llama-perplex on {len(corpus_lines)} problems (ctx={args.ppl_context}) ...")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=env,
                             timeout=args.request_timeout_s)
    except subprocess.TimeoutExpired:
        return {"phase": "ppl", "error": "llama-perplex timed out"}
    text = (out.stdout or "") + "\n" + (out.stderr or "")
    match = re.search(r"pp=([\d.]+)", text) or re.search(r"[Pp]erplexity = ([\d.]+)", text)
    if not match:
        return {"phase": "ppl", "error": text[-800:]}
    return {"phase": "ppl", "perplexity": float(match.group(1)),
            "corpus_problems": len(corpus_lines)}


# ============================================================================
# Phase 3+4: reasoning accuracy + temperature sweep
# ============================================================================


def extract_prediction(text: str) -> float | None:
    """Last numeric token in the response (after the last 'Answer:' if present)."""
    tail = text.rsplit("Answer:", 1)[-1] if "Answer:" in text else text
    nums = [n.replace(",", "") for n in NUMBER_RE.findall(tail)]
    if not nums:
        nums = [n.replace(",", "") for n in NUMBER_RE.findall(text)]
    try:
        return float(nums[-1]) if nums else None
    except ValueError:
        return None


def run_eval_once(
    args: argparse.Namespace,
    problems: list[dict[str, str]],
    temperature: float,
    gen_twice: bool = False,
) -> dict:
    correct = 0
    latencies: list[float] = []
    first_two: list[str] = []
    for i, problem in enumerate(problems):
        prompt = problem["question"] + "\nLet's work it out step by step.\nAnswer:"
        comp = stream_completion(args, prompt, args.eval_max_tokens, temperature)
        if not comp.ok:
            continue
        latencies.append(comp.elapsed_s)
        gold = float(problem["answer"])
        pred = extract_prediction(comp.text)
        if pred is not None and abs(pred - gold) < 0.011:
            correct += 1
        if gen_twice and i < 2:
            first_two.append(comp.text)
            second = stream_completion(args, prompt, args.eval_max_tokens, temperature)
            first_two.append(second.text)
    total = len(problems)
    result = {
        "temperature": temperature,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "failed_requests": total - len(latencies),
    }
    if gen_twice and len(first_two) == 2:
        result["deterministic_at_zero"] = first_two[0] == first_two[1]
    return result


def run_evaluation(args: argparse.Namespace, ref_context: int, bin_dir: Path,
                   model_path: Path, problems: list[dict[str, str]]) -> dict:
    subset = problems[: args.num_questions]
    log_path = Path(args.workdir) / "server_eval.log"
    proc = start_server(bin_dir, model_path, args, ref_context, args.ngl_values[0], log_path)
    ready = proc is not None and wait_for_server(args, proc, args.server_ready_timeout_s)
    if not ready:
        stop_process(proc, ("[l]lama-server",))
        return {"phase": "eval", "error": "eval server not ready\n" + server_log_tail(log_path)}

    temp_rows = [run_eval_once(args, subset, args.temperatures[0],
                               gen_twice=args.temperatures[0] == 0.0)]
    log(f"  temp={args.temperatures[0]}: accuracy={temp_rows[0]['correct']}/{temp_rows[0]['total']}")
    for temp in args.temperatures[1:]:
        if "temp" in args.skip:
            break
        row = run_eval_once(args, subset, temp)
        temp_rows.append(row)
        log(f"  temp={temp}: accuracy={row['correct']}/{row['total']}")

    stop_process(proc, ("[l]lama-server",))
    return {"phase": "eval", "context": ref_context,
            "num_questions": len(subset), "temperatures": temp_rows}


# ============================================================================
# Synthesis: sweet spot
# ============================================================================


def synthesize_speed(rows: list[dict], total_vram_mb: int) -> dict:
    """Rank fitting speed configs; pick the best speed/VRAM balance."""
    candidates = [r for r in rows if r.get("fit")]
    ranked = []
    for r in candidates:
        gen = (r.get("short") or {}).get("gen_tps") or 0.0
        vram = r.get("peak_vram_mb") or total_vram_mb
        # Normalized composite: 70% throughput (vs best), 30% VRAM headroom.
        best_tps = max((rr.get("short", {}).get("gen_tps") or 0.0) for rr in candidates)
        tps_norm = gen / best_tps if best_tps > 0 else 0.0
        headroom = max(0.0, 1.0 - vram / total_vram_mb) if total_vram_mb else 0.0
        score = 0.7 * tps_norm + 0.3 * headroom
        cfg = r["config"]
        ranked.append({
            "context": cfg["context"],
            "ngl": cfg["ngl"],
            "gen_tps": gen,
            "long_gen_tps": (r.get("long") or {}).get("gen_tps"),
            "long_ttft_s": (r.get("long") or {}).get("ttft_s"),
            "peak_vram_gb": round(vram / 1024, 2),
            "peak_ram_gb": round(r.get("peak_ram_gb") or 0.0, 2),
            "score": round(score, 4),
        })
    eligible = [r for r in ranked if (r["gen_tps"] or 0.0) >= args_min_gen_tps()]
    if not eligible:
        eligible = ranked  # fall back to everything that fit; caller explains
    ranked.sort(key=lambda r: r["score"], reverse=True)
    recommended = eligible[0] if eligible else None
    rationale = (
        "score = 0.7 * (gen_tps / best_gen_tps) + 0.3 * vram_headroom; "
        f"min_gen_tps floor = {args_min_gen_tps()} tok/s"
    )
    return {
        "recommended": recommended,
        "ranking": ranked[: min(len(ranked), 12)],
        "criteria": {"min_gen_tps": args_min_gen_tps(), "formula": rationale},
        "fit_summary": {
            "fit": sum(1 for r in candidates),
            "oom_or_failed": len(rows) - len(candidates),
            "failed_configs": [
                {"context": r["config"]["context"], "ngl": r["config"]["ngl"],
                 "error_tail": (r.get("error") or "")[-300:]}
                for r in rows if not r.get("fit")
            ],
        },
    }


_MIN_GEN_TPS = DEFAULT_MIN_GEN_TPS


def args_min_gen_tps() -> float:
    return _MIN_GEN_TPS


# ============================================================================
# Output writers
# ============================================================================


def _flatten(phase: str, config: dict, extra: dict) -> list[dict]:
    """Flatten a phase result into tidy CSV rows."""
    rows = []
    for key, value in extra.items():
        if isinstance(value, dict):
            for k2, v2 in value.items():
                if isinstance(v2, (int, float)) and not isinstance(v2, bool):
                    rows.append({"phase": phase, "config": json.dumps(config, sort_keys=True),
                                 "metric": f"{key}.{k2}", "value": v2, "unit": ""})
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append({"phase": phase, "config": json.dumps(config, sort_keys=True),
                         "metric": key, "value": value, "unit": ""})
    return rows


def write_outputs(out_dir: Path, results: dict, speed_spot: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    (out_dir / "sweet_spot.json").write_text(json.dumps(speed_spot, indent=2, ensure_ascii=False),
                                             encoding="utf-8")

    csv_rows: list[dict] = []
    for row in results.get("speed", []):
        csv_rows += _flatten("speed", row.get("config", {}), row)
    for item in (results.get("eval") or {}).get("temperatures", []):
        cfg = {"context": (results.get("eval") or {}).get("context"),
               "num_questions": (results.get("eval") or {}).get("num_questions")}
        csv_rows += _flatten("eval", cfg, item)
    if isinstance(results.get("ppl"), dict):
        csv_rows += _flatten("ppl", {}, results["ppl"])
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["phase", "config", "metric", "value", "unit"])
        writer.writeheader()
        writer.writerows(csv_rows)

    md = ["# Nex-N2.5-mini Q4_K_M Benchmark Summary", ""]
    md.append("## Speed / resource sweep")
    md.append("| context | ngl | fit | short tok/s | long tok/s | long TTFT (s) | peak VRAM (GB) | peak RAM (GB) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in results.get("speed", []):
        md.append(
            f"| {r['config']['context']} | {r['config']['ngl']} | {r.get('fit')} "
            f"| {fmt((r.get('short') or {}).get('gen_tps'))} "
            f"| {fmt((r.get('long') or {}).get('gen_tps'))} "
            f"| {fmt((r.get('long') or {}).get('ttft_s'))} "
            f"| {r.get('peak_vram_mb', 0) / 1024:.1f} | {r.get('peak_ram_gb', 0.0):.1f} |")
    md += ["", "## Quality (GSM8K subset by temperature)"]
    md.append("| temp | accuracy | avg latency (s) |")
    md.append("|---|---|---|")
    for item in (results.get("eval") or {}).get("temperatures", []):
        md.append(f"| {item.get('temperature')} | {item.get('accuracy', 0) * 100:.1f}% "
                  f"| {fmt(item.get('avg_latency_s'))} |")
    ppl = results.get("ppl") or {}
    if "perplexity" in ppl:
        md += ["", f"Perplexity (GSM8K slice): {ppl['perplexity']:.3f}", ""]
    md += ["", "## Recommended sweet spot", json.dumps(speed_spot.get("recommended"), indent=2),
           "", f"Criteria: {speed_spot['criteria']['formula']}"]
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    log(f"Outputs written to {out_dir}")


# ============================================================================
# Main
# ============================================================================


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    argv = strip_kernel_args(raw_argv)
    parser = argparse.ArgumentParser(
        description="Benchmarks Nex-N2.5-mini Q4_K_M: speed/VRAM/RAM, PPL, "
                    "GSM8K accuracy, temperature sweep. Outputs CSV/JSON for sweet-spot analysis.")
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--kv-cache", default=DEFAULT_KV_CACHE_TYPE,
                        help="KV cache quant type (default q4_0 to fit 128k on 2xT4)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--ubatch-size", type=int, default=DEFAULT_UBATCH_SIZE)
    parser.add_argument("--context-sizes", default=",".join(str(c) for c in DEFAULT_CONTEXT_SIZES),
                        help="Comma-separated context window sizes to sweep")
    parser.add_argument("--ngl-values", default=",".join(str(n) for n in DEFAULT_NGL_VALUES),
                        help="Comma-separated -ngl layer offload values (99 = all layers)")
    parser.add_argument("--temperatures", default=",".join(str(t) for t in DEFAULT_TEMPERATURES))
    parser.add_argument("--ref-context", type=int, default=8192,
                        help="Context window for the quality/temperature phase")
    parser.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS)
    parser.add_argument("--eval-max-tokens", type=int, default=DEFAULT_EVAL_MAX_TOKENS)
    parser.add_argument("--ppl-context", type=int, default=DEFAULT_PPL_CONTEXT)
    parser.add_argument("--min-gen-tps", type=float, default=DEFAULT_MIN_GEN_TPS,
                        help="Minimum generation tok/s a config must reach to be eligible")
    parser.add_argument("--server-ready-timeout-s", type=int, default=DEFAULT_SERVER_READY_TIMEOUT_S)
    parser.add_argument("--request-timeout-s", type=int, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--workdir", default=None, help="Scratch dir (default /kaggle/tmp/nex25_bench)")
    parser.add_argument("--out-dir", default=None, help="Results dir (default /kaggle/working)")
    parser.add_argument("--skip", action="append", choices=["speed", "ppl", "eval", "temp"],
                        default=[], help="Phases to skip (repeatable)")
    args = parser.parse_args(argv)
    args.context_sizes = [int(v) for v in args.context_sizes.split(",") if v.strip()]
    args.ngl_values = [int(v) for v in args.ngl_values.split(",") if v.strip()]
    args.temperatures = [float(v) for v in args.temperatures.split(",") if v.strip()]
    args.workdir = Path(args.workdir) if args.workdir else default_workdir()
    args.out_dir = Path(args.out_dir) if args.out_dir else default_outdir()
    args.skip = list(set(args.skip))
    return args


def main(argv: list[str] | None = None) -> None:
    """Run all bench phases. Pass an explicit argv when calling from a notebook."""
    args = parse_args(argv)
    global _MIN_GEN_TPS
    _MIN_GEN_TPS = args.min_gen_tps

    stop_process(None, ("[l]lama-server",))
    args.workdir.mkdir(parents=True, exist_ok=True)

    log("Provisioning model and llama.cpp ...")
    model_path = get_or_download_model(args.model_repo, args.model_file, args.workdir / "models")
    bin_dir = install_llama_cpp(args.workdir)

    results: dict = {"meta": {
        "model_repo": args.model_repo, "model_file": args.model_file,
        "context_sizes": args.context_sizes, "ngl_values": args.ngl_values,
        "temperatures": args.temperatures, "kv_cache": args.kv_cache,
        "batch": args.batch_size, "ubatch": args.ubatch_size,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }}

    if "speed" not in args.skip:
        log("Phase 1/4: speed + resource sweep")
        results["speed"] = run_speed_sweep(args, model_path, bin_dir)

    problems: list[dict[str, str]] = []
    if not {"ppl", "eval"} <= set(args.skip):
        problems = load_gsm8k(max(args.num_questions + 40, 80))

    if "ppl" not in args.skip:
        log("Phase 2/4: perplexity")
        results["ppl"] = run_perplexity(args, bin_dir, model_path, problems)

    if "eval" not in args.skip:
        log("Phase 3/4 + 4/4: GSM8K accuracy + temperature sweep")
        if not problems:
            problems = load_gsm8k(max(args.num_questions + 40, 80))
        results["eval"] = run_evaluation(args, args.ref_context, bin_dir, model_path, problems)

    stop_process(None, ("[l]lama-server",))

    total_vram_mb = max((r.get("total_vram_mb") or 0) for r in results.get("speed", [])) or 30720
    results["sweet_spot"] = synthesize_speed(results.get("speed", []), total_vram_mb)
    write_outputs(args.out_dir, results, results["sweet_spot"])

    rec = results["sweet_spot"].get("recommended")
    if rec:
        log(f"SWEET SPOT -> context={rec['context']} ngl={rec['ngl']} "
            f"gen={rec['gen_tps']:.1f} tok/s peak_vram={rec['peak_vram_gb']} GB "
            f"score={rec['score']}")
    else:
        log("No fitting configuration found; see results.json for failure details.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_process(None, ("[l]lama-server",))
        sys.exit(130)
    finally:
        stop_process(None, ("[l]lama-server",))
