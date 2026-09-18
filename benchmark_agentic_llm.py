#!/usr/bin/env python3
"""
benchmark_agentic_llm.py

End-to-End Benchmark & Sweetspot Finder:
Ornith 1.5 35B-A3B (Q4_K_M) vs Qwen 3.6 35B-A3B MTP (Q4_K_M)
Optimized for Kaggle / Colab 2x NVIDIA Tesla T4 (32GB VRAM total).

Dataset references (Kaggle):
- Ornith: https://www.kaggle.com/datasets/zeenoz/ornith-1-5-35b-q4-k-m
  Expected file: Ornith-1.5-35B-Q4_K_M.gguf
- Qwen:   https://www.kaggle.com/datasets/zeenoz/qwen-3-6-a3b-mtp-q4-k-m/data
  Expected file: Qwen-3.6-A3B-MTP-Q4_K_M.gguf

Features:
- Automatic Jupyter/Colab kernel arg stripping (-f kernel-*.json)
- Zero-config auto-detection of Kaggle datasets in /kaggle/input
- Auto-locates or downloads prebuilt CUDA llama.cpp for sm_75 (T4)
- Benchmarks:
    * Prompt / Prefill tok/s (pp)
    * Token generation / Decode tok/s (tg)
    * Time to First Token (TTFT) via streaming
    * Peak VRAM per GPU, headroom, util, power via nvidia-smi
    * Autonomous agentic coding pass rate on Python coding tasks
- Generates:
    * sweet_spot.json  (Winning profile, per-model sweet spot, launcher config)
    * summary.md       (Markdown table & copy-paste config for Kaggle launcher)
    * summary.csv      (Tidy data table)
    * results.json     (Full benchmark telemetry)
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None  # Handled gracefully if missing, but required for client calls


# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

LLAMA_API_BASE = "https://api.github.com/repos/cloudlnkcn/llama.cpp/releases?per_page=5"
LLAMA_FALLBACK_URL = (
    "https://github.com/ai-dock/llama.cpp-cuda/releases/download/b9628/"
    "llama.cpp-b9628-cuda-12.8-amd64.tar.gz"
)

# Standard Kaggle mount paths
KAGGLE_DATASET_PATHS = {
    "ornith": [
        Path("/kaggle/input/ornith-1-5-35b-q4-k-m/Ornith-1.5-35B-Q4_K_M.gguf"),
        Path("/kaggle/input/ornith-1-5-35b-q4-k-m/ornith-1.5-35b-q4_k_m.gguf"),
    ],
    "qwen": [
        Path("/kaggle/input/qwen-3-6-a3b-mtp-q4-k-m/Qwen-3.6-A3B-MTP-Q4_K_M.gguf"),
        Path("/kaggle/input/qwen-3-6-a3b-mtp-q4-k-m/qwen-3-6-a3b-mtp-q4_k_m.gguf"),
    ],
}

HF_FALLBACK_MODELS = {
    "ornith": ("ornith-ai/Ornith-1.5-35B-A3B-GGUF", "Ornith-1.5-35B-Q4_K_M.gguf"),
    "qwen": ("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
}


# ---------------------------------------------------------------------------
# Jupyter / Colab Kernel Argument Cleaning
# ---------------------------------------------------------------------------

def strip_kernel_args(raw_argv: List[str]) -> List[str]:
    """
    Strips injected Jupyter / Google Colab / IPyKernel arguments such as:
    -f /root/.local/share/jupyter/runtime/kernel-uuid.json
    or standalone kernel-*.json paths.
    """
    cleaned: List[str] = []
    skip_next = False
    for arg in raw_argv:
        if skip_next:
            skip_next = False
            continue
        if arg in ("-f", "--f"):
            skip_next = True
            continue
        if arg.startswith(("-f=", "--f=")):
            continue
        # Check for kernel json files injected as positional arguments
        if arg.endswith(".json") and any(k in arg for k in ("kernel-", "jupyter", "ipykernel", "colab")):
            continue
        # Drop ipykernel / colab launcher flags
        if any(arg.startswith(prefix) for prefix in (
            "--ip=", "--stdin=", "--control=", "--hb=",
            "--Session.", "--IPKernelApp.", "--parent=",
            "--transport=", "--profile-dir=", "--history-path="
        )):
            continue
        cleaned.append(arg)
    return cleaned


# ---------------------------------------------------------------------------
# Tuning Profiles for 2x Tesla T4 (32GB VRAM total)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Profile:
    name: str
    ctx: int
    batch: int
    ubatch: int
    cache_k: str
    cache_v: str
    split_mode: str = "layer"
    tensor_split: Optional[str] = "1,1"
    flash_attn: str = "on"
    note: str = ""


PROFILES: Dict[str, Profile] = {
    "speed32_q8": Profile(
        "speed32_q8", 32768, 2048, 512, "q8_0", "q8_0", "layer", "1,1", "on",
        "32K baseline, higher-precision Q8 KV cache",
    ),
    "speed32_q4": Profile(
        "speed32_q4", 32768, 2048, 512, "q4_0", "q4_0", "layer", "1,1", "on",
        "32K with Q4 KV for maximum VRAM headroom",
    ),
    "balanced64": Profile(
        "balanced64", 65536, 2048, 512, "q4_0", "q4_0", "layer", "1,1", "on",
        "Recommended starting point for coding & agentic workflows",
    ),
    "balanced64_ub256": Profile(
        "balanced64_ub256", 65536, 2048, 256, "q4_0", "q4_0", "layer", "1,1", "on",
        "64K with smaller physical micro-batch (lower compute spikes)",
    ),
    "row64": Profile(
        "row64", 65536, 2048, 512, "q4_0", "q4_0", "row", "1,1", "on",
        "Compare row split against layer split across 2x T4",
    ),
    "tensor64": Profile(
        "tensor64", 65536, 2048, 512, "q4_0", "q4_0", "tensor", "1,1", "on",
        "Tensor-parallel split across 2x T4",
    ),
    "long96": Profile(
        "long96", 98304, 1024, 256, "q4_0", "q4_0", "layer", "1,1", "on",
        "96K long-context coding profile",
    ),
    "tensor128": Profile(
        "tensor128", 131072, 1024, 256, "q4_0", "q4_0", "tensor", "1,1", "on",
        "128K stress profile with tensor split on 2x T4",
    ),
}

PRESETS = {
    "quick": ["speed32_q4", "balanced64", "long96"],
    "balanced": ["speed32_q8", "speed32_q4", "balanced64", "long96"],
    "full": list(PROFILES.keys()),
}


# ---------------------------------------------------------------------------
# Coding Tasks for Agentic Quality Evaluation
# ---------------------------------------------------------------------------

@dataclass
class CodingTask:
    name: str
    instruction: str
    files: Dict[str, str]
    tests: Dict[str, str]


CODING_TASKS: List[CodingTask] = [
    CodingTask(
        name="merge_intervals",
        instruction=(
            "Fix src/ranges.py. merge_intervals(intervals) must accept unsorted pairs, "
            "normalize reversed endpoints, merge overlapping OR touching intervals, "
            "not mutate the input, and return [] for empty input."
        ),
        files={
            "src/ranges.py": """def merge_intervals(intervals):
    \"\"\"Return merged intervals.\"\"\"
    if not intervals:
        return []
    intervals.sort()
    out = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = out[-1]
        if start < last_end:
            out[-1] = (last_start, max(last_end, end))
        else:
            out.append((start, end))
    return out
""",
            "src/__init__.py": "",
        },
        tests={
            "tests/test_ranges.py": """import unittest
from src.ranges import merge_intervals

class TestRanges(unittest.TestCase):
    def test_unsorted_touching(self):
        self.assertEqual(
            merge_intervals([(8, 10), (1, 3), (3, 6), (15, 12)]),
            [(1, 6), (8, 10), (12, 15)],
        )

    def test_reversed_overlap(self):
        self.assertEqual(
            merge_intervals([(5, 1), (2, 7), (10, 10)]),
            [(1, 7), (10, 10)],
        )

    def test_does_not_mutate(self):
        data = [(5, 1), (2, 4)]
        snapshot = list(data)
        merge_intervals(data)
        self.assertEqual(data, snapshot)

    def test_empty(self):
        self.assertEqual(merge_intervals([]), [])

if __name__ == \"__main__\":
    unittest.main()
""",
        },
    ),
    CodingTask(
        name="lru_cache",
        instruction=(
            "Repair src/cache.py. Implement deterministic LRUCache.get and put. get returns -1 "
            "when missing. Successful get and update refresh recency. Inserting beyond capacity "
            "evicts LRU. capacity <= 0 stores nothing."
        ),
        files={
            "src/cache.py": """class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = {}

    def get(self, key):
        return self.data.get(key, -1)

    def put(self, key, value):
        self.data[key] = value
        if len(self.data) > self.capacity:
            self.data.pop(next(iter(self.data)))
""",
            "src/__init__.py": "",
        },
        tests={
            "tests/test_cache.py": """import unittest
from src.cache import LRUCache

class TestLRU(unittest.TestCase):
    def test_eviction_after_get(self):
        c = LRUCache(2)
        c.put(\"a\", 1)
        c.put(\"b\", 2)
        self.assertEqual(c.get(\"a\"), 1)
        c.put(\"c\", 3)
        self.assertEqual(c.get(\"b\"), -1)
        self.assertEqual(c.get(\"a\"), 1)
        self.assertEqual(c.get(\"c\"), 3)

    def test_update_refreshes_recency(self):
        c = LRUCache(2)
        c.put(\"a\", 1)
        c.put(\"b\", 2)
        c.put(\"a\", 10)
        c.put(\"c\", 3)
        self.assertEqual(c.get(\"a\"), 10)
        self.assertEqual(c.get(\"b\"), -1)

    def test_zero_capacity(self):
        c = LRUCache(0)
        c.put(\"x\", 1)
        self.assertEqual(c.get(\"x\"), -1)

if __name__ == \"__main__\":
    unittest.main()
""",
        },
    ),
    CodingTask(
        name="config_parser",
        instruction=(
            "Fix src/config.py parse_config(text). Format is KEY=VALUE. Ignore blank lines and "
            "lines whose first non-space char is #. Split only on first '='. Trim key/value. "
            "Empty key or malformed non-comment line must raise ValueError. Last duplicate wins."
        ),
        files={
            "src/config.py": """def parse_config(text):
    result = {}
    for line in text.splitlines():
        if not line or line.startswith(\"#\"):
            continue
        key, value = line.split(\"=\")
        result[key] = value
    return result
""",
            "src/__init__.py": "",
        },
        tests={
            "tests/test_config.py": """import unittest
from src.config import parse_config

class TestConfig(unittest.TestCase):
    def test_normal(self):
        text = \"\"\"
          # comment
        HOST = localhost
        TOKEN = a=b=c
        HOST = example.org

        \"\"\"
        self.assertEqual(parse_config(text), {
            \"HOST\": \"example.org\",
            \"TOKEN\": \"a=b=c\",
        })

    def test_bad_line(self):
        with self.assertRaises(ValueError):
            parse_config(\"OK=1\\nBADLINE\")

    def test_empty_key(self):
        with self.assertRaises(ValueError):
            parse_config(\"   = value\")

if __name__ == \"__main__\":
    unittest.main()
""",
        },
    ),
    CodingTask(
        name="bank_transfer_multifile",
        instruction=(
            "Repair account transfer across src/store.py and src/service.py. transfer(store, src, "
            "dst, amount) must reject amount <= 0, missing accounts, and insufficient funds. "
            "Successful transfer moves money atomically. If destination credit raises, restore "
            "the source balance. Keep public API unchanged. You may modify either source file."
        ),
        files={
            "src/store.py": """class AccountStore:
    def __init__(self, balances=None):
        self.balances = dict(balances or {})

    def exists(self, account):
        return account in self.balances

    def balance(self, account):
        return self.balances[account]

    def debit(self, account, amount):
        self.balances[account] -= amount

    def credit(self, account, amount):
        self.balances[account] += amount
""",
            "src/service.py": """def transfer(store, src, dst, amount):
    store.debit(src, amount)
    store.credit(dst, amount)
    return True
""",
            "src/__init__.py": "",
        },
        tests={
            "tests/test_transfer.py": """import unittest
from src.store import AccountStore
from src.service import transfer

class FailingCreditStore(AccountStore):
    def credit(self, account, amount):
        raise RuntimeError(\"simulated write failure\")

class TestTransfer(unittest.TestCase):
    def test_success(self):
        s = AccountStore({\"a\": 100, \"b\": 25})
        self.assertTrue(transfer(s, \"a\", \"b\", 30))
        self.assertEqual(s.balance(\"a\"), 70)
        self.assertEqual(s.balance(\"b\"), 55)

    def test_reject_bad_amount(self):
        for amount in (0, -1):
            s = AccountStore({\"a\": 100, \"b\": 25})
            with self.assertRaises(ValueError):
                transfer(s, \"a\", \"b\", amount)
            self.assertEqual(s.balances, {\"a\": 100, \"b\": 25})

    def test_missing_account(self):
        s = AccountStore({\"a\": 100})
        with self.assertRaises(KeyError):
            transfer(s, \"a\", \"missing\", 10)
        self.assertEqual(s.balances, {\"a\": 100})

    def test_insufficient(self):
        s = AccountStore({\"a\": 5, \"b\": 20})
        with self.assertRaises(ValueError):
            transfer(s, \"a\", \"b\", 10)
        self.assertEqual(s.balances, {\"a\": 5, \"b\": 20})

    def test_rollback(self):
        s = FailingCreditStore({\"a\": 100, \"b\": 20})
        with self.assertRaises(RuntimeError):
            transfer(s, \"a\", \"b\", 10)
        self.assertEqual(s.balances, {\"a\": 100, \"b\": 20})

if __name__ == \"__main__\":
    unittest.main()
""",
        },
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def mean_or_none(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.mean(vals) if vals else None


def median_or_none(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.median(vals) if vals else None


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def run_capture(
    cmd: List[str], timeout: int = 30, env: Optional[Dict[str, str]] = None
) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        return p.returncode, p.stdout
    except Exception as exc:
        return 999, str(exc)


def clean_args_dict(args: Any) -> Dict[str, Any]:
    """Strip functions and non-serializable objects from args for clean JSON dumps."""
    if isinstance(args, argparse.Namespace):
        items = vars(args).items()
    elif isinstance(args, dict):
        items = args.items()
    else:
        return {}
    clean: Dict[str, Any] = {}
    for k, v in items:
        if callable(v):
            clean[k] = getattr(v, "__name__", str(v))
        elif isinstance(v, Path):
            clean[k] = v.as_posix()
        elif isinstance(v, dict):
            clean[k] = clean_args_dict(v)
        else:
            clean[k] = v
    return clean


def system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp": now_iso(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    rc, out = run_capture([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ], timeout=10)
    info["nvidia_smi"] = out.strip() if rc == 0 else None
    rc, out = run_capture([
        "nvidia-smi",
        "--query-gpu=compute_cap",
        "--format=csv,noheader",
    ], timeout=10)
    info["gpu_compute_capability"] = out.strip() if rc == 0 else None
    return info


def write_repo(root: Path, files: Dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def read_repo_sources(root: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith("tests/") or rel.endswith(".pyc") or "__pycache__" in rel:
            continue
        result[rel] = path.read_text(encoding="utf-8", errors="replace")
    return result


def format_repo(files: Dict[str, str]) -> str:
    chunks = []
    for name, content in sorted(files.items()):
        chunks.append(f"\n===== {name} =====\n{content}")
    return "".join(chunks)


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise ValueError("No valid JSON object found in model output")


def apply_patch_json(repo: Path, response_text: str) -> List[str]:
    obj = extract_json_object(response_text)
    files = obj.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Expected non-empty JSON 'files' mapping")

    root_resolved = repo.resolve()
    changed: List[str] = []
    for rel, content in files.items():
        if not isinstance(rel, str) or not isinstance(content, str):
            raise ValueError("Every files entry must be path:string -> full-content:string")
        rel = rel.replace("\\", "/").lstrip("/")
        if rel.startswith("tests/"):
            raise ValueError("Model is not allowed to modify hidden tests")
        dest = (repo / rel).resolve()
        try:
            dest.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"Unsafe path rejected: {rel}") from exc
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        changed.append(rel)
    return changed


def run_tests(repo: Path, timeout: int = 20) -> Tuple[bool, str, float]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    start = time.perf_counter()
    try:
        p = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return p.returncode == 0, p.stdout[-12000:], time.perf_counter() - start
    except subprocess.TimeoutExpired as exc:
        return False, f"TEST TIMEOUT\n{exc}", time.perf_counter() - start
    except Exception as exc:
        return False, f"TEST ERROR: {exc}", time.perf_counter() - start


def float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Model & llama.cpp Auto-Discovery / Provisioning
# ---------------------------------------------------------------------------

def find_kaggle_model(model_name: str, explicit_path: Optional[str] = None) -> Optional[Path]:
    """
    Search order:
    1. Explicit path (if provided and exists)
    2. Kaggle dataset mounts (/kaggle/input/...)
    3. Dynamic search in /kaggle/input for any matching .gguf
    4. Local working directory / models folder
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p

    # Check known Kaggle dataset mounts
    candidates = KAGGLE_DATASET_PATHS.get(model_name, [])
    for p in candidates:
        if p.exists():
            return p

    # Dynamic search in /kaggle/input
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        kw = "ornith" if model_name == "ornith" else "qwen"
        # First priority: matching keyword and Q4_K_M
        for f in kaggle_input.rglob("*.gguf"):
            name_lower = f.name.lower()
            if kw in name_lower and "q4_k_m" in name_lower:
                return f
        # Second priority: any matching GGUF
        for f in kaggle_input.rglob("*.gguf"):
            if kw in f.name.lower():
                return f

    # Search local folders
    for base in [Path("."), Path("./models"), Path("/kaggle/tmp/models"), Path("/tmp/models")]:
        if base.exists():
            kw = "ornith" if model_name == "ornith" else "qwen"
            for f in base.rglob("*.gguf"):
                if kw in f.name.lower() and "q4_k_m" in f.name.lower():
                    return f

    return None


def resolve_llama_server(explicit_path: Optional[str] = None, workdir: Optional[Path] = None) -> Optional[Path]:
    """Locate llama-server executable from explicit path, PATH, or standard Kaggle directories."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists() and os.access(p, os.X_OK):
            return p

    which = shutil.which("llama-server")
    if which:
        return Path(which)

    search_dirs = [
        Path("/kaggle/working/llama.cpp/build/bin"),
        Path("/kaggle/tmp/llm_server/bin"),
        Path("/tmp/llm_server/bin"),
        Path("./bin"),
        Path("."),
    ]
    if workdir:
        search_dirs.insert(0, workdir / "bin")

    for directory in search_dirs:
        candidate = directory / "llama-server"
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate

    return None


def install_llama_cpp(workdir: Path) -> Path:
    """Fetch prebuilt CUDA llama.cpp binary (sm_75 for T4) into workdir / 'bin'."""
    bin_dir = workdir / "bin"
    server_bin = bin_dir / "llama-server"
    if server_bin.exists() and os.access(server_bin, os.X_OK):
        return server_bin

    print("Fetching prebuilt CUDA llama.cpp for sm_75 (Tesla T4)...", flush=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    # 1. Try github release API for specific sm_75 ubuntu binary
    download_url = None
    archive_name = "ubuntu-cuda-sm_75-x64.tar.xz"
    try:
        req = urllib.request.Request(
            LLAMA_API_BASE,
            headers={"User-Agent": "Mozilla/5.0 (compatible; KaggleBenchmark/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            releases = json.loads(resp.read().decode())
            pattern = re.compile(r"ubuntu-cuda-sm_75-x64\.tar\.xz$", re.I)
            for rel in releases:
                for asset in rel.get("assets", []):
                    if pattern.search(asset["name"]):
                        download_url = asset["browser_download_url"]
                        archive_name = asset["name"]
                        break
                if download_url:
                    break
    except Exception as exc:
        print(f"Warning: GitHub API release query warning ({exc}); using fallback release...", file=sys.stderr)

    if not download_url:
        download_url = LLAMA_FALLBACK_URL
        archive_name = "llama.cpp-b9628-cuda-12.8-amd64.tar.gz"

    archive_path = workdir / archive_name
    print(f"Downloading llama.cpp from {download_url} ...")
    urllib.request.urlretrieve(download_url, str(archive_path))

    print(f"Extracting {archive_name} ...")
    with tarfile.open(archive_path, "r:*") as tf:
        try:
            tf.extractall(workdir, filter="data")
        except TypeError:
            tf.extractall(workdir)

    found = [p for p in workdir.rglob("llama-server") if p.is_file()]
    if not found:
        raise RuntimeError("llama-server executable not found after extraction.")

    real_server = found[0]
    real_server.chmod(real_server.stat().st_mode | 0o755)

    # Link or copy real_server into bin_dir
    if server_bin != real_server:
        try:
            if server_bin.exists() or server_bin.is_symlink():
                server_bin.unlink()
            server_bin.symlink_to(real_server)
        except Exception:
            shutil.copy2(real_server, server_bin)
            server_bin.chmod(0o755)

    # Copy / symlink all *.so* libraries found next to real_server or in workdir to bin_dir
    for so_file in real_server.parent.glob("*.so*"):
        target_so = bin_dir / so_file.name
        if not target_so.exists():
            try:
                target_so.symlink_to(so_file)
            except Exception:
                shutil.copy2(so_file, target_so)

    print(f"llama-server ready: {server_bin}")
    return server_bin


def build_server_env(
    workdir: Optional[Path] = None, server_bin: Optional[Path] = None
) -> Dict[str, str]:
    """Gather all shared library (.so) directories for LD_LIBRARY_PATH to avoid error code 127."""
    env = os.environ.copy()
    so_dirs: set[str] = set()

    if server_bin:
        resolved_server = Path(server_bin).resolve()
        so_dirs.add(str(resolved_server.parent))
        if resolved_server.parent.name == "bin":
            so_dirs.add(str(resolved_server.parent.parent / "lib"))

    if workdir and Path(workdir).exists():
        for p in Path(workdir).rglob("*.so*"):
            if p.is_file():
                so_dirs.add(str(p.parent.resolve()))

    # Common CUDA library paths on Linux/Kaggle
    for cand in [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-12.8/lib64",
        "/usr/local/cuda-12.6/lib64",
        "/usr/local/cuda-12.4/lib64",
        "/usr/local/cuda-12.2/lib64",
        "/usr/local/cuda-12.1/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib",
    ]:
        p_cand = Path(cand)
        if p_cand.exists():
            so_dirs.add(str(p_cand.resolve()))

    existing_ld = env.get("LD_LIBRARY_PATH", "")
    new_dirs = [d for d in sorted(so_dirs) if d]
    ld_str = ":".join(new_dirs)
    if existing_ld:
        ld_str = f"{ld_str}:{existing_ld}"
    env["LD_LIBRARY_PATH"] = ld_str
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1")
    env["GGML_CUDA_NO_VMM"] = "1"
    return env


# ---------------------------------------------------------------------------
# GPU Monitor
# ---------------------------------------------------------------------------

class GPUMonitor:
    def __init__(self, interval: float = 0.4):
        self.interval = interval
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        rc, out = run_capture([
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], timeout=5)
        if rc != 0:
            return
        ts = time.time()
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            try:
                self.samples.append({
                    "ts": ts,
                    "gpu": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": float(parts[2]),
                    "memory_used_mb": float(parts[3]),
                    "util_pct": float(parts[4]),
                    "temp_c": float(parts[5]),
                    "power_w": float(parts[6]),
                })
            except ValueError:
                pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._stop.clear()
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._sample()

    def summary(self) -> Dict[str, Any]:
        by_gpu: Dict[int, List[Dict[str, Any]]] = {}
        for sample in self.samples:
            by_gpu.setdefault(sample["gpu"], []).append(sample)
        result: Dict[str, Any] = {}
        for gpu, rows in sorted(by_gpu.items()):
            result[str(gpu)] = {
                "name": rows[0]["name"],
                "memory_total_mb": max(r["memory_total_mb"] for r in rows),
                "peak_memory_used_mb": max(r["memory_used_mb"] for r in rows),
                "avg_util_pct": mean_or_none(r["util_pct"] for r in rows),
                "max_util_pct": max(r["util_pct"] for r in rows),
                "max_temp_c": max(r["temp_c"] for r in rows),
                "avg_power_w": mean_or_none(r["power_w"] for r in rows),
            }
        return result


# ---------------------------------------------------------------------------
# OpenAI-Compatible llama.cpp Client
# ---------------------------------------------------------------------------

class LlamaClient:
    def __init__(
        self,
        base_url: str,
        model: Optional[str] = None,
        timeout: float = 600.0,
        extra_json: Optional[Dict[str, Any]] = None,
    ):
        if requests is None:
            raise RuntimeError("Missing required 'requests' package. Install with: pip install requests")
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.model = model
        self.timeout = timeout
        self.extra_json = extra_json or {}
        self.session = requests.Session()

    @property
    def chat_url(self) -> str:
        return self.base_url + "/chat/completions"

    def _payload(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "seed": 42,
            "timings_per_token": True,
        }
        if self.model:
            payload["model"] = self.model
        payload.update(copy.deepcopy(self.extra_json))
        return payload

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        payload = self._payload(messages, max_tokens, temperature, False)
        start = time.perf_counter()
        response = self.session.post(self.chat_url, json=payload, timeout=self.timeout)
        latency = time.perf_counter() - start
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1500]}")
        data = response.json()
        choices = data.get("choices") or []
        content = ""
        finish_reason = None
        if choices:
            choice = choices[0]
            finish_reason = choice.get("finish_reason")
            message = choice.get("message") or {}
            content = message.get("content") or ""
            if not content and isinstance(message.get("reasoning_content"), str):
                content = message["reasoning_content"]
        return {
            "content": content,
            "latency_s": latency,
            "usage": data.get("usage") or {},
            "timings": data.get("timings") or {},
            "finish_reason": finish_reason,
            "raw_model": data.get("model"),
        }

    def stream_ttft(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        payload = self._payload(messages, max_tokens, temperature, True)
        start = time.perf_counter()
        first_token_time: Optional[float] = None
        content: List[str] = []
        timings: Dict[str, Any] = {}
        usage: Dict[str, Any] = {}

        with self.session.post(
            self.chat_url,
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1500]}")
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data_s = line[5:].strip()
                if data_s == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_s)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk.get("timings"), dict):
                    timings.update(chunk["timings"])
                if isinstance(chunk.get("usage"), dict):
                    usage.update(chunk["usage"])
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    token_text = delta.get("content")
                    if token_text:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        content.append(token_text)

        end = time.perf_counter()
        return {
            "ttft_s": first_token_time - start if first_token_time else None,
            "latency_s": end - start,
            "content": "".join(content),
            "timings": timings,
            "usage": usage,
        }


# ---------------------------------------------------------------------------
# Server Process Lifecycle
# ---------------------------------------------------------------------------

class ServerProcess:
    def __init__(
        self,
        llama_server: str,
        model_path: str,
        profile: Profile,
        port: int,
        log_path: Path,
        extra_args: Optional[List[str]] = None,
        startup_timeout: int = 300,
        env: Optional[Dict[str, str]] = None,
    ):
        self.llama_server = llama_server
        self.model_path = model_path
        self.profile = profile
        self.port = port
        self.log_path = log_path
        self.extra_args = extra_args or []
        self.startup_timeout = startup_timeout
        self.env = env or os.environ.copy()
        self.proc: Optional[subprocess.Popen] = None
        self._log_file = None

    @property
    def root_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def api_url(self) -> str:
        return self.root_url + "/v1"

    def command(self) -> List[str]:
        p = self.profile
        cmd = [
            self.llama_server,
            "--model", self.model_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--ctx-size", str(p.ctx),
            "--batch-size", str(p.batch),
            "--ubatch-size", str(p.ubatch),
            "--cache-type-k", p.cache_k,
            "--cache-type-v", p.cache_v,
            "--split-mode", p.split_mode,
            "--flash-attn", p.flash_attn,
            "--n-gpu-layers", "all",
            "--parallel", "1",
            "--fit", "off",
            "--metrics",
            "--jinja",
        ]
        if p.tensor_split:
            cmd += ["--tensor-split", p.tensor_split]
        cmd += self.extra_args
        return cmd

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("w", encoding="utf-8")
        cmd = self.command()
        self._log_file.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        self._log_file.flush()

        kwargs: Dict[str, Any] = {
            "stdout": self._log_file,
            "stderr": subprocess.STDOUT,
            "text": True,
            "env": self.env,
        }
        if os.name != "nt":
            kwargs["preexec_fn"] = os.setsid
        self.proc = subprocess.Popen(cmd, **kwargs)

        deadline = time.time() + self.startup_timeout
        last_error = ""
        while time.time() < deadline:
            if self.proc.poll() is not None:
                if self._log_file:
                    self._log_file.flush()
                log_tail = ""
                if self.log_path.exists():
                    try:
                        lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        log_tail = "\n".join(lines[-25:])
                    except Exception:
                        pass
                raise RuntimeError(
                    f"llama-server exited during startup (code {self.proc.returncode}).\n"
                    f"--- Log excerpt ({self.log_path}) ---\n{log_tail}\n"
                    f"-----------------------------------------"
                )
            for endpoint in (self.root_url + "/health", self.api_url + "/models"):
                try:
                    r = requests.get(endpoint, timeout=3)
                    if r.status_code == 200:
                        return
                    last_error = f"{endpoint}: HTTP {r.status_code}"
                except Exception as exc:
                    last_error = str(exc)
            time.sleep(1)

        raise TimeoutError(
            f"Server not ready after {self.startup_timeout}s: {last_error}. See {self.log_path}"
        )

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                else:
                    self.proc.terminate()
                self.proc.wait(timeout=12)
            except Exception:
                try:
                    if os.name != "nt":
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    else:
                        self.proc.kill()
                except Exception:
                    pass
        if self._log_file:
            self._log_file.close()


# ---------------------------------------------------------------------------
# Benchmarks Implementation
# ---------------------------------------------------------------------------

AGENT_SYSTEM = """You are a careful autonomous coding agent.
You are editing a small Python repository.

Rules:
1. Diagnose the task and current repository.
2. Return ONLY one JSON object, no Markdown:
   {"files":{"relative/path.py":"FULL NEW FILE CONTENT"},"notes":"short reason"}
3. Include only files that need changing.
4. Never modify tests.
5. Each file value must be COMPLETE final content, not a diff.
6. Preserve public APIs unless the task says otherwise.
"""


def agentic_task(
    client: LlamaClient,
    task: CodingTask,
    max_turns: int,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "task": task.name,
        "passed": False,
        "turns": 0,
        "attempts": [],
        "total_latency_s": 0.0,
        "total_completion_tokens": 0,
    }

    with tempfile.TemporaryDirectory(prefix=f"agentbench_{task.name}_") as td:
        repo = Path(td)
        write_repo(repo, task.files)
        write_repo(repo, task.tests)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": AGENT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"TASK:\n{task.instruction}\n\nCURRENT REPOSITORY:\n"
                    + format_repo(read_repo_sources(repo))
                ),
            },
        ]

        for turn in range(1, max_turns + 1):
            attempt: Dict[str, Any] = {"turn": turn}
            try:
                response = client.complete(messages, max_tokens=max_tokens, temperature=temperature)
                attempt.update({
                    "latency_s": response["latency_s"],
                    "usage": response["usage"],
                    "timings": response["timings"],
                    "finish_reason": response["finish_reason"],
                })
                result["total_latency_s"] += response["latency_s"]
                result["total_completion_tokens"] += int(
                    response["usage"].get("completion_tokens")
                    or response["timings"].get("predicted_n")
                    or 0
                )

                model_text = response["content"]
                attempt["response_excerpt"] = model_text[-3000:]
                attempt["changed_files"] = apply_patch_json(repo, model_text)

                passed, test_output, test_time = run_tests(repo)
                attempt["tests_passed"] = passed
                attempt["test_time_s"] = test_time
                attempt["test_output"] = test_output[-6000:]
                result["attempts"].append(attempt)
                result["turns"] = turn

                if passed:
                    result["passed"] = True
                    break

                messages.append({"role": "assistant", "content": model_text})
                messages.append({
                    "role": "user",
                    "content": (
                        "Tests failed. Diagnose and repair the repository.\n\n"
                        f"TEST OUTPUT:\n{test_output[-6000:]}\n\n"
                        "CURRENT REPOSITORY:\n"
                        + format_repo(read_repo_sources(repo))
                        + "\n\nReturn ONLY the required JSON object."
                    ),
                })

            except Exception as exc:
                attempt["error"] = str(exc)
                result["attempts"].append(attempt)
                result["turns"] = turn
                if turn < max_turns:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Previous attempt could not be applied: {exc}\n"
                            "Return ONLY valid JSON with complete file contents."
                        ),
                    })
                else:
                    break

    return result


def make_perf_context(approx_tokens: int) -> str:
    target_chars = max(1000, approx_tokens * 4)
    parts = [
        "You are reviewing a synthetic Python codebase. Read the context, then answer the final request.\n\n"
    ]
    current_len = len(parts[0])
    i = 0
    while current_len < target_chars:
        block = (
            f"def helper_{i:06d}(x, y):\n"
            f"    # Synthetic helper {i}.\n"
            f"    total = x * {((i % 97) + 1)} + y\n"
            f"    if total % {((i % 13) + 2)} == 0:\n"
            f"        return total // {((i % 7) + 1)}\n"
            f"    return total + {i % 31}\n\n"
        )
        parts.append(block)
        current_len += len(block)
        i += 1
    parts.append(
        "\nFINAL REQUEST: In at most 8 bullets, list general code-review principles. "
        "Do not quote the synthetic context."
    )
    return "".join(parts)


def perf_one_prompt(
    client: LlamaClient,
    approx_prompt_tokens: int,
    output_tokens: int,
) -> Dict[str, Any]:
    messages = [{"role": "user", "content": make_perf_context(approx_prompt_tokens)}]

    stream = client.stream_ttft(messages, max_tokens=min(64, output_tokens), temperature=0.0)
    normal = client.complete(messages, max_tokens=output_tokens, temperature=0.0)
    timings = normal["timings"]
    usage = normal["usage"]

    return {
        "target_prompt_tokens_approx": approx_prompt_tokens,
        "actual_prompt_tokens": int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0),
        "completion_tokens": int(
            usage.get("completion_tokens") or timings.get("predicted_n") or 0
        ),
        "prompt_tps": float_or_none(timings.get("prompt_per_second")),
        "decode_tps": float_or_none(timings.get("predicted_per_second")),
        "prompt_ms": float_or_none(timings.get("prompt_ms")),
        "predicted_ms": float_or_none(timings.get("predicted_ms")),
        "ttft_s": stream.get("ttft_s"),
        "stream_latency_s": stream.get("latency_s"),
        "normal_latency_s": normal.get("latency_s"),
        "finish_reason": normal.get("finish_reason"),
    }


def benchmark_endpoint(
    client: LlamaClient,
    profile: Optional[Profile],
    perf_prompt_sizes: List[int],
    perf_repeats: int,
    quality_repeats: int,
    max_agent_turns: int,
    agent_max_tokens: int,
    temperature: float,
    run_perf: bool,
    run_quality: bool,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {"perf": [], "agentic": [], "started_at": now_iso()}

    if run_perf:
        for approx_tokens in perf_prompt_sizes:
            if profile and approx_tokens > int(profile.ctx * 0.70):
                continue
            for rep in range(perf_repeats):
                try:
                    row = perf_one_prompt(client, approx_tokens, output_tokens=512)
                    row["repeat"] = rep + 1
                    output["perf"].append(row)
                    print(
                        f"    PERF ~{approx_tokens:>6} tok | "
                        f"pp={str(row['prompt_tps']):>9} t/s | "
                        f"tg={str(row['decode_tps']):>8} t/s | "
                        f"TTFT={str(row['ttft_s']):>7}s"
                    )
                except Exception as exc:
                    output["perf"].append({
                        "target_prompt_tokens_approx": approx_tokens,
                        "repeat": rep + 1,
                        "error": str(exc),
                    })
                    print(f"    PERF ~{approx_tokens} failed: {exc}")

    if run_quality:
        for rep in range(quality_repeats):
            for task in CODING_TASKS:
                print(f"    AGENT {task.name} (rep {rep + 1}) ... ", end="", flush=True)
                row = agentic_task(
                    client,
                    task,
                    max_agent_turns,
                    agent_max_tokens,
                    temperature,
                )
                row["repeat"] = rep + 1
                output["agentic"].append(row)
                status = "PASS" if row["passed"] else "FAIL"
                print(f"{status} in {row['turns']} turn(s)")

    output["finished_at"] = now_iso()
    return output


# ---------------------------------------------------------------------------
# Scoring & Sweet Spot Analysis
# ---------------------------------------------------------------------------

def summarize_run(run: Dict[str, Any]) -> Dict[str, Any]:
    perf = [x for x in run.get("bench", {}).get("perf", []) if not x.get("error")]
    agent = run.get("bench", {}).get("agentic", [])
    passed = [x for x in agent if x.get("passed")]
    solved_turns = [x.get("turns") for x in passed if x.get("turns") is not None]

    gpu_summary = run.get("gpu_summary") or {}
    peak_per_gpu = None
    min_headroom = None
    if gpu_summary:
        peaks: List[float] = []
        headrooms: List[float] = []
        for gpu in gpu_summary.values():
            peak = gpu.get("peak_memory_used_mb")
            total = gpu.get("memory_total_mb")
            if peak is not None:
                peaks.append(float(peak))
            if peak is not None and total is not None:
                headrooms.append(float(total) - float(peak))
        if peaks:
            peak_per_gpu = max(peaks)
        if headrooms:
            min_headroom = min(headrooms)

    profile = run.get("profile") or {}
    total_agent = len(agent)
    return {
        "model": run.get("model_name"),
        "model_path": run.get("model_path"),
        "profile": profile.get("name", "endpoint"),
        "status": run.get("status"),
        "ctx": profile.get("ctx"),
        "batch": profile.get("batch"),
        "ubatch": profile.get("ubatch"),
        "cache_k": profile.get("cache_k"),
        "cache_v": profile.get("cache_v"),
        "split_mode": profile.get("split_mode"),
        "median_prompt_tps": median_or_none(x.get("prompt_tps") for x in perf),
        "median_decode_tps": median_or_none(x.get("decode_tps") for x in perf),
        "median_ttft_s": median_or_none(x.get("ttft_s") for x in perf),
        "median_latency_s": median_or_none(x.get("normal_latency_s") for x in perf),
        "agent_pass_rate": len(passed) / total_agent if total_agent else None,
        "avg_turns_to_solve": mean_or_none(solved_turns),
        "peak_vram_per_gpu_mb": peak_per_gpu,
        "min_vram_headroom_mb": min_headroom,
        "error": run.get("error"),
    }


def minmax(values: List[Optional[float]], higher_is_better: bool = True) -> List[float]:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not valid:
        return [0.0] * len(values)
    lo, hi = min(valid), max(valid)
    if math.isclose(lo, hi):
        return [1.0 if v is not None else 0.0 for v in values]
    result: List[float] = []
    for value in values:
        if value is None:
            result.append(0.0)
            continue
        norm = (float(value) - lo) / (hi - lo)
        result.append(norm if higher_is_better else 1.0 - norm)
    return result


def rank_summaries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    good = [r for r in rows if r.get("status") == "ok"]
    if good:
        quality = minmax([r.get("agent_pass_rate") for r in good], True)
        decode = minmax([r.get("median_decode_tps") for r in good], True)
        prompt = minmax([r.get("median_prompt_tps") for r in good], True)
        ttft = minmax([r.get("median_ttft_s") for r in good], False)
        headroom = minmax([r.get("min_vram_headroom_mb") for r in good], True)

        for i, row in enumerate(good):
            row["agentic_score_100"] = round(
                100.0 * (
                    0.50 * quality[i]
                    + 0.20 * decode[i]
                    + 0.15 * prompt[i]
                    + 0.075 * ttft[i]
                    + 0.075 * headroom[i]
                ),
                2,
            )

    ranked = sorted(
        rows,
        key=lambda r: (
            r.get("status") == "ok",
            r.get("agentic_score_100") if r.get("agentic_score_100") is not None else -1,
        ),
        reverse=True,
    )
    rank = 0
    for row in ranked:
        if row.get("status") == "ok":
            rank += 1
            row["rank"] = rank
        else:
            row["rank"] = None
    return ranked


def synthesize_sweet_spot(ranked_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Identify overall best and per-model best configurations."""
    overall_best = None
    best_by_model: Dict[str, Any] = {}

    for row in ranked_summaries:
        if row.get("status") != "ok":
            continue
        if overall_best is None:
            overall_best = row
        m = row.get("model")
        if m and m not in best_by_model:
            best_by_model[m] = row

    synthesis = {
        "timestamp": now_iso(),
        "overall_winner": overall_best,
        "per_model_sweet_spot": best_by_model,
        "suggested_launcher_configs": {},
    }

    for model_name, best_row in best_by_model.items():
        synthesis["suggested_launcher_configs"][model_name] = {
            "model_name": model_name,
            "profile": best_row.get("profile"),
            "context_size": best_row.get("ctx"),
            "batch_size": best_row.get("batch"),
            "ubatch_size": best_row.get("ubatch"),
            "kv_cache_k": best_row.get("cache_k"),
            "kv_cache_v": best_row.get("cache_v"),
            "split_mode": best_row.get("split_mode"),
            "tensor_split": "1,1",
            "flash_attn": "on",
            "score": best_row.get("agentic_score_100"),
            "decode_tok_s": best_row.get("median_decode_tps"),
            "prompt_tok_s": best_row.get("median_prompt_tps"),
            "pass_rate": best_row.get("agent_pass_rate"),
        }

    return synthesis


def save_results(
    output_dir: Path,
    raw: Dict[str, Any],
    summaries: List[Dict[str, Any]],
    quiet: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    fields = [
        "rank", "model", "profile", "status", "agentic_score_100",
        "agent_pass_rate", "avg_turns_to_solve",
        "median_prompt_tps", "median_decode_tps", "median_ttft_s", "median_latency_s",
        "ctx", "batch", "ubatch", "cache_k", "cache_v", "split_mode",
        "peak_vram_per_gpu_mb", "min_vram_headroom_mb", "model_path", "error",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    # Sweetspot synthesis
    sweet_spot = synthesize_sweet_spot(summaries)
    (output_dir / "sweet_spot.json").write_text(
        json.dumps(sweet_spot, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    def fmt(value: Optional[float], digits: int = 1) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    header = (
        f"{'RANK':>4}  {'MODEL':<8} {'PROFILE':<20} {'PASS':>6} "
        f"{'PP t/s':>10} {'TG t/s':>9} {'TTFT':>8} {'VRAM/GPU':>10} {'SCORE':>7}"
    )
    lines = [header, "-" * len(header)]
    for row in summaries:
        pass_rate = row.get("agent_pass_rate")
        pass_text = f"{pass_rate * 100:.0f}%" if pass_rate is not None else "-"
        lines.append(
            f"{str(row.get('rank') or '-'):>4}  "
            f"{str(row.get('model') or '-'):<8.8} "
            f"{str(row.get('profile') or '-'):<20.20} "
            f"{pass_text:>6} "
            f"{fmt(row.get('median_prompt_tps')):>10} "
            f"{fmt(row.get('median_decode_tps')):>9} "
            f"{fmt(row.get('median_ttft_s'), 2):>8} "
            f"{fmt(row.get('peak_vram_per_gpu_mb'), 0):>10} "
            f"{fmt(row.get('agentic_score_100')):>7}"
        )

    summary_text = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    # Generate rich Markdown report summary.md
    md: List[str] = [
        "# Ornith 1.5 35B vs Qwen 3.6 35B A3B -- 2x T4 Sweet Spot Benchmark",
        "",
        f"- **Generated:** {now_iso()}",
        f"- **GPUs:** 2x Tesla T4 (32 GB total VRAM)",
        "",
        "## Leaderboard & Profiles",
        "",
        "| Rank | Model | Profile | Context | KV | Pass Rate | PP tok/s | TG tok/s | TTFT (s) | Peak VRAM/GPU | Score |",
        "|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for row in summaries:
        pass_rate = row.get("agent_pass_rate")
        pass_text = f"{pass_rate * 100:.0f}%" if pass_rate is not None else "-"
        md.append(
            f"| {row.get('rank') or '-'} "
            f"| **{row.get('model') or '-'}** "
            f"| {row.get('profile') or '-'} "
            f"| {row.get('ctx') or '-'} "
            f"| {row.get('cache_k') or '-'}/{row.get('cache_v') or '-'} "
            f"| {pass_text} "
            f"| {fmt(row.get('median_prompt_tps'))} "
            f"| {fmt(row.get('median_decode_tps'))} "
            f"| {fmt(row.get('median_ttft_s'), 2)} "
            f"| {fmt(row.get('peak_vram_per_gpu_mb'), 0)} MB "
            f"| **{fmt(row.get('agentic_score_100'))}** |"
        )

    md.extend(["", "## Recommended Sweet Spot Configurations", ""])
    for m_name, cfg in sweet_spot.get("suggested_launcher_configs", {}).items():
        md.extend([
            f"### Winner for {m_name.upper()} ({cfg['profile']})",
            f"- **Agentic Score:** {cfg['score']} / 100",
            f"- **Generation Speed:** {fmt(cfg['decode_tok_s'])} tok/s",
            f"- **Prompt Speed:** {fmt(cfg['prompt_tok_s'])} tok/s",
            f"- **Context:** {cfg['context_size']} tokens",
            "",
            "Copy-paste configuration for `serve_ornith.py` / `serve_qwen.py` / `launch.py`:",
            "```python",
            f"CONTEXT_SIZE = {cfg['context_size']}",
            f"BATCH_SIZE = {cfg['batch_size']}",
            f"UBATCH_SIZE = {cfg['ubatch_size']}",
            f"KV_CACHE_TYPE = \"{cfg['kv_cache_k']}\"",
            f"SPLIT_MODE = \"{cfg['split_mode']}\"",
            f"TENSOR_SPLIT = \"{cfg['tensor_split']}\"",
            "FLASH_ATTN = \"on\"",
            "```",
            "",
        ])

    (output_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    if not quiet:
        print("\n" + summary_text)
        print("Outputs saved:")
        print(f"  {output_dir / 'summary.md'}")
        print(f"  {output_dir / 'sweet_spot.json'}")
        print(f"  {output_dir / 'summary.csv'}")
        print(f"  {output_dir / 'summary.txt'}")
        print(f"  {output_dir / 'results.json'}")


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_extra_json(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("--extra-json must decode to a JSON object")
    return value


def cmd_sweep(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir) if args.workdir else Path("/kaggle/tmp/bench" if Path("/kaggle").exists() else "./tmp_bench")
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve llama-server
    server_bin = resolve_llama_server(args.llama_server, workdir)
    if not server_bin:
        if platform.system() == "Linux":
            server_bin = install_llama_cpp(workdir)
        if not server_bin or not server_bin.exists():
            print(
                "ERROR: llama-server executable could not be found.\n"
                "Please specify --llama-server /path/to/llama-server or ensure llama.cpp is built.",
                file=sys.stderr,
            )
            return 1
    print(f"Using llama-server: {server_bin}")

    # 2. Profiles
    profile_names = (
        [x.strip() for x in args.profiles.split(",") if x.strip()]
        if args.profiles else PRESETS[args.preset]
    )
    unknown = [name for name in profile_names if name not in PROFILES]
    if unknown:
        raise SystemExit(
            f"Unknown profiles: {unknown}. Available: {', '.join(PROFILES)}"
        )

    # 3. Models
    models: List[Tuple[str, str, List[str]]] = []
    
    # Ornith resolution
    ornith_path = find_kaggle_model("ornith", args.ornith)
    if ornith_path:
        models.append((
            "ornith",
            str(ornith_path),
            shlex.split(args.ornith_extra_server_args or ""),
        ))
    elif args.ornith:
        print(f"Warning: specified Ornith path not found: {args.ornith}")

    # Qwen resolution
    qwen_path = find_kaggle_model("qwen", args.qwen)
    if qwen_path:
        qwen_extra = shlex.split(args.qwen_extra_server_args or "")
        # Multi-Token Prediction (MTP) flags if requested and supported
        if args.qwen_mtp and "--spec-type" not in args.qwen_extra_server_args:
            qwen_extra += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
        models.append(("qwen", str(qwen_path), qwen_extra))
    elif args.qwen:
        print(f"Warning: specified Qwen path not found: {args.qwen}")

    if not models:
        print(
            "ERROR: Neither Ornith nor Qwen models were detected or provided.\n"
            "On Kaggle, attach datasets:\n"
            "  - zeenoz/ornith-1-5-35b-q4-k-m\n"
            "  - zeenoz/qwen-3-6-a3b-mtp-q4-k-m\n"
            "Or specify --ornith /path/to/model.gguf and/or --qwen /path/to/model.gguf",
            file=sys.stderr,
        )
        return 1

    extra_json = parse_extra_json(args.extra_json)
    prompt_sizes = parse_int_list(args.prompt_tokens)
    server_env = build_server_env(workdir=workdir, server_bin=server_bin)

    # Pre-flight check on llama-server
    rc, out = run_capture([str(server_bin), "--version"], env=server_env, timeout=10)
    if rc != 0:
        rc, out = run_capture([str(server_bin), "--help"], env=server_env, timeout=10)
    if rc != 0:
        print(
            f"Warning: llama-server preflight test exited with code {rc}:\n{out[:400]}\n"
            "Will attempt execution with LD_LIBRARY_PATH...",
            file=sys.stderr,
        )

    raw: Dict[str, Any] = {
        "system": system_info(),
        "args": clean_args_dict(args),
        "profiles": {name: asdict(PROFILES[name]) for name in profile_names},
        "runs": [],
    }

    total_runs = len(models) * len(profile_names)
    run_index = 0

    for model_name, model_path, extra_server_args in models:
        for profile_name in profile_names:
            run_index += 1
            profile = PROFILES[profile_name]
            print(
                f"\n=== [{run_index}/{total_runs}] {model_name.upper()} | {profile.name} | "
                f"ctx={profile.ctx} b={profile.batch} ub={profile.ubatch} "
                f"KV={profile.cache_k}/{profile.cache_v} split={profile.split_mode} ==="
            )
            print(f"    Model path: {model_path}")

            run: Dict[str, Any] = {
                "model_name": model_name,
                "model_path": model_path,
                "profile": asdict(profile),
                "status": "failed",
                "server_extra_args": extra_server_args,
            }
            log_path = output_dir / "logs" / f"{safe_name(model_name)}__{safe_name(profile.name)}.log"
            server = ServerProcess(
                llama_server=str(server_bin),
                model_path=model_path,
                profile=profile,
                port=args.port,
                log_path=log_path,
                extra_args=extra_server_args,
                startup_timeout=args.startup_timeout,
                env=server_env,
            )
            monitor = GPUMonitor(args.gpu_sample_interval)

            try:
                print("  Starting llama-server...")
                server.start()
                print("  Server ready.")
                monitor.start()

                client = LlamaClient(
                    server.api_url,
                    model=args.api_model,
                    timeout=args.request_timeout,
                    extra_json=extra_json,
                )

                try:
                    client.complete(
                        [{"role": "user", "content": "Reply with exactly: READY"}],
                        max_tokens=8,
                        temperature=0.0,
                    )
                except Exception as exc:
                    print(f"  Warmup warning: {exc}")

                run["bench"] = benchmark_endpoint(
                    client=client,
                    profile=profile,
                    perf_prompt_sizes=prompt_sizes,
                    perf_repeats=args.perf_repeats,
                    quality_repeats=args.quality_repeats,
                    max_agent_turns=args.max_agent_turns,
                    agent_max_tokens=args.agent_max_tokens,
                    temperature=args.temperature,
                    run_perf=not args.no_perf,
                    run_quality=not args.no_quality,
                )
                run["status"] = "ok"

            except Exception as exc:
                run["error"] = str(exc)
                print(f"  FAILED: {exc}")
            finally:
                monitor.stop()
                run["gpu_summary"] = monitor.summary()
                server.stop()
                time.sleep(args.cooldown)

            raw["runs"].append(run)
            progressive = rank_summaries([summarize_run(r) for r in raw["runs"]])
            save_results(output_dir, raw, progressive, quiet=True)

    summaries = rank_summaries([summarize_run(r) for r in raw["runs"]])
    save_results(output_dir, raw, summaries, quiet=False)
    return 0


def cmd_endpoint(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_json = parse_extra_json(args.extra_json)

    profile = None
    if args.ctx:
        profile = Profile(
            name=args.name,
            ctx=args.ctx,
            batch=0,
            ubatch=0,
            cache_k="unknown",
            cache_v="unknown",
            split_mode="unknown",
            tensor_split=None,
            note="External endpoint",
        )

    run: Dict[str, Any] = {
        "model_name": args.name,
        "model_path": args.url,
        "profile": asdict(profile) if profile else None,
        "status": "failed",
    }

    monitor = GPUMonitor(args.gpu_sample_interval)
    try:
        monitor.start()
        client = LlamaClient(
            args.url,
            model=args.api_model,
            timeout=args.request_timeout,
            extra_json=extra_json,
        )
        run["bench"] = benchmark_endpoint(
            client=client,
            profile=profile,
            perf_prompt_sizes=parse_int_list(args.prompt_tokens),
            perf_repeats=args.perf_repeats,
            quality_repeats=args.quality_repeats,
            max_agent_turns=args.max_agent_turns,
            agent_max_tokens=args.agent_max_tokens,
            temperature=args.temperature,
            run_perf=not args.no_perf,
            run_quality=not args.no_quality,
        )
        run["status"] = "ok"
    except Exception as exc:
        run["error"] = str(exc)
        print(f"FAILED: {exc}")
    finally:
        monitor.stop()
        run["gpu_summary"] = monitor.summary()

    raw = {"system": system_info(), "args": clean_args_dict(args), "runs": [run]}
    summaries = rank_summaries([summarize_run(run)])
    save_results(output_dir, raw, summaries)
    return 0


# ---------------------------------------------------------------------------
# Argument Parser Construction
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default="bench_results")
    parser.add_argument("--workdir", default=None, help="Working directory for binary downloads / temporary files")
    parser.add_argument(
        "--prompt-tokens",
        default="1024,8192,16384",
        help="Approximate prompt sizes; actual token count is read from llama.cpp timings.",
    )
    parser.add_argument("--perf-repeats", type=int, default=2)
    parser.add_argument("--quality-repeats", type=int, default=1)
    parser.add_argument("--max-agent-turns", type=int, default=3)
    parser.add_argument("--agent-max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.4)
    parser.add_argument("--api-model", default=None)
    parser.add_argument(
        "--extra-json",
        default=None,
        help=(
            "Extra JSON merged into each request. Example: "
            "--extra-json '{\"chat_template_kwargs\":{\"enable_thinking\":true}}'"
        ),
    )
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument("--no-quality", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark/tune Ornith and Qwen for 2x T4 agentic coding.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=False)

    sweep = sub.add_parser("sweep", help="Launch llama-server for each tuning profile.")
    sweep.add_argument("--llama-server", default=None, help="Path to llama-server (auto-detected if omitted)")
    sweep.add_argument("--ornith", default=None, help="Path to Ornith GGUF (auto-detected from Kaggle input if omitted)")
    sweep.add_argument("--qwen", default=None, help="Path to Qwen GGUF (auto-detected from Kaggle input if omitted)")
    sweep.add_argument("--qwen-mtp", action="store_true", default=True, help="Enable MTP speculative decoding flags for Qwen")
    sweep.add_argument("--no-qwen-mtp", dest="qwen_mtp", action="store_false", help="Disable MTP flags for Qwen")
    sweep.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    sweep.add_argument(
        "--profiles",
        default=None,
        help="Comma-separated profile names. Available: " + ", ".join(PROFILES),
    )
    sweep.add_argument("--port", type=int, default=8080)
    sweep.add_argument("--startup-timeout", type=int, default=300)
    sweep.add_argument("--cooldown", type=float, default=2.0)
    sweep.add_argument("--ornith-extra-server-args", default="")
    sweep.add_argument("--qwen-extra-server-args", default="")
    add_common_args(sweep)
    sweep.set_defaults(func=cmd_sweep)

    endpoint = sub.add_parser("endpoint", help="Benchmark an already-running endpoint.")
    endpoint.add_argument("--url", required=True, help="e.g. http://127.0.0.1:8080/v1")
    endpoint.add_argument("--name", required=True)
    endpoint.add_argument("--ctx", type=int, default=None)
    add_common_args(endpoint)
    endpoint.set_defaults(func=cmd_endpoint)

    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments with automatic kernel arg stripping and 'sweep' default."""
    if argv is None:
        argv = strip_kernel_args(sys.argv[1:])
    else:
        argv = strip_kernel_args(list(argv))

    # If no subcommand is given (e.g. running in notebook or without args), default to 'sweep'
    if not argv or argv[0] not in ("sweep", "endpoint", "-h", "--help"):
        argv = ["sweep"] + argv

    parser = build_parser()
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point. Can be invoked directly from command line or inside a notebook."""
    try:
        args = parse_args(argv)
        return args.func(args)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        return 130


if __name__ == "__main__":
    ret = main()
    if ret != 0:
        raise SystemExit(ret)
