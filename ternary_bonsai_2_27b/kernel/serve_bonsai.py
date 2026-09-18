#!/usr/bin/env python3
"""
Serve Ternary-Bonsai-2-27B (PQ2_0) on Kaggle 2x Tesla T4 GPUs with PrismML llama.cpp and Cloudflare Named / Quick Tunnel.
Pushed as a script kernel to Kaggle Cloud by launch.py.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

CFG = None  # __LAUNCHER_CONFIG__ (launch.py replaces this line with dictionary)

DEFAULTS = {
    "model_repo": "prism-ml/Ternary-Bonsai-2-27B-gguf",
    "model_file": "Ternary-Bonsai-2-27B-PQ2_0.gguf",
    "model_url": "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PQ2_0.gguf",
    "model_name": "ternary-bonsai-2-27b",
    "context_size": 196608,
    "batch_size": 1024,
    "ubatch_size": 512,
    "kv_cache_k": "q8_0",
    "kv_cache_v": "q8_0",
    "port": 8080,
    "keepalive_min": 480,
    "api_key": "kilo-secret-key",
    "static_domain": "https://api.zexnoz.dev",
    "cf_token": "",
    "ntfy_topic": "",
}

CFG = {**DEFAULTS, **(CFG or {})}

WORKDIR = Path("/kaggle/tmp/llm_server" if Path("/kaggle").exists() else "/tmp/llm_server")
BIN_DIR = WORKDIR / "bin"
MODEL_DIR = Path("/kaggle/tmp/models" if Path("/kaggle").exists() else "/tmp/models")
LOG_DIR = Path("/kaggle/working" if Path("/kaggle").exists() else WORKDIR / "logs")
CLOUDFLARED_BIN = WORKDIR / "cloudflared"
LLAMA_LOG = LOG_DIR / "llama-server.log"
CF_LOG = LOG_DIR / "cloudflared.log"
USER_AGENT = "kaggle-bonsai-installer/1.0"

PRISM_LLAMA_PRIMARY_URL = (
    "https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b10687-5d80cff/"
    "llama-prism-b10687-5d80cff-bin-linux-cuda-12.8-x64.tar.gz"
)
PRISM_LLAMA_FALLBACK_URL = (
    "https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b10687-5d80cff/"
    "llama-prism-b10687-5d80cff-bin-linux-cuda-12.4-x64.tar.gz"
)

server_proc = None
cf_proc = None


def notify(phase, message="", extra=None):
    payload = {
        "phase": phase,
        "message": message,
        "extra": extra or {},
        "timestamp": time.time(),
    }
    print(f"[{time.strftime('%H:%M:%S')}] [{phase}] {message}", flush=True)
    if CFG.get("ntfy_topic"):
        try:
            url = f"https://ntfy.sh/{CFG['ntfy_topic']}"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Title": f"Kaggle Ternary-Bonsai: {phase}", "Tags": "robot"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass


def cleanup(sig=None, frame=None):
    global server_proc, cf_proc
    notify("stopping", "Cleaning up background services...")
    if server_proc and server_proc.poll() is None:
        server_proc.terminate()
    if cf_proc and cf_proc.poll() is None:
        cf_proc.terminate()
    os.system("pkill -9 -f '[l]lama-server' >/dev/null 2>&1")
    os.system("pkill -9 -f '[c]loudflared' >/dev/null 2>&1")
    notify("stopped", "Services stopped.")
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def get_cf_token():
    """Extract token from launcher CFG, Kaggle Secrets, or environment variable."""
    if CFG.get("cf_token"):
        return CFG["cf_token"].strip()
    try:
        from kaggle_secrets import UserSecretsClient

        sec = UserSecretsClient().get_secret("CF_TUNNEL_TOKEN")
        if sec:
            return sec.strip()
    except Exception:
        pass
    return os.environ.get("CF_TUNNEL_TOKEN", "").strip()


def setup_directories():
    WORKDIR.mkdir(parents=True, exist_ok=True)
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def check_gpu():
    notify("starting", "Kernel initialized. Checking GPU environment...")
    if not shutil.which("nvidia-smi"):
        notify("failed", "nvidia-smi not found. Enable GPU accelerator.")
        raise RuntimeError("nvidia-smi not found. Enable GPU accelerator.")
    try:
        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                text=True,
            )
            .strip()
            .splitlines()
        )
        notify("gpu_info", f"Detected {len(out)} GPU(s): {', '.join(out)}")
        return out
    except Exception as e:
        notify("gpu_warn", f"Could not query GPUs: {e}")
        return []


def download_file(url: str, dest_path: Path):
    """Download file using aria2c, curl, wget, or urllib fallback."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("aria2c"):
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
        subprocess.run(["curl", "-L", "-C", "-", "-o", str(dest_path), url], check=True)
        return dest_path
    if shutil.which("wget"):
        subprocess.run(["wget", "-c", "-O", str(dest_path), url], check=True)
        return dest_path

    headers = {"User-Agent": USER_AGENT}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    return dest_path


def install_prebuilt_llamacpp():
    """Ensure PrismML llama.cpp (CUDA 12.8 / fallback 12.4) binaries are ready."""
    server_bin = BIN_DIR / "llama-server"
    if server_bin.exists() and os.access(server_bin, os.X_OK):
        notify("llama_ready", f"Found existing llama-server at {server_bin}")
        return server_bin

    found = [p for p in WORKDIR.rglob("llama-server") if p.is_file() and os.access(p, os.X_OK)]
    if found:
        try:
            if server_bin.exists():
                server_bin.unlink()
            server_bin.symlink_to(found[0])
        except Exception:
            shutil.copy2(found[0], server_bin)
        notify("llama_ready", f"Found existing llama-server: {found[0]}")
        return server_bin

    notify("setup_llama", "Downloading prebuilt PrismML llama.cpp CUDA binaries...")
    urls_to_try = [
        (PRISM_LLAMA_PRIMARY_URL, "llama-prism-cuda-12.8.tar.gz"),
        (PRISM_LLAMA_FALLBACK_URL, "llama-prism-cuda-12.4.tar.gz"),
    ]

    archive_path = None
    for url, filename in urls_to_try:
        dest = WORKDIR / filename
        try:
            print(f"Attempting download: {url}")
            download_file(url, dest)
            archive_path = dest
            break
        except Exception as e:
            print(f"⚠️ Binary download failed from {url}: {e}")

    if not archive_path or not archive_path.exists():
        notify("failed", "Failed to download PrismML llama.cpp binaries from primary and fallback URLs.")
        raise RuntimeError("Failed to download PrismML llama.cpp binaries.")

    print(f"Extracting {archive_path.name}...")
    try:
        with tarfile.open(archive_path, "r:gz") as tf:
            try:
                tf.extractall(WORKDIR, filter="data")
            except TypeError:
                tf.extractall(WORKDIR)
    except Exception as e:
        print(f"tarfile extraction notice: {e}; falling back to tar command...")
        subprocess.run(["tar", "-xzf", str(archive_path), "-C", str(WORKDIR)], check=True)

    found = [p for p in WORKDIR.rglob("llama-server") if p.is_file()]
    if not found:
        notify("failed", "llama-server executable not found after extraction.")
        raise RuntimeError("llama-server executable not found.")

    real_server = found[0]
    try:
        real_server.chmod(real_server.stat().st_mode | 0o755)
    except Exception:
        pass

    if server_bin.exists():
        try:
            server_bin.unlink()
        except Exception:
            pass
    try:
        server_bin.symlink_to(real_server)
    except Exception:
        shutil.copy2(real_server, server_bin)

    notify("llama_ready", f"PrismML llama-server ready: {server_bin}")
    return server_bin


def install_cloudflared():
    """Ensure cloudflared binary is installed and executable."""
    if CLOUDFLARED_BIN.exists() and os.access(CLOUDFLARED_BIN, os.X_OK):
        notify("cloudflared_ready", f"cloudflared ready: {CLOUDFLARED_BIN}")
        return CLOUDFLARED_BIN

    notify("setup_cloudflared", "Downloading cloudflared tunnel client...")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    try:
        download_file(url, CLOUDFLARED_BIN)
    except Exception:
        urllib.request.urlretrieve(url, CLOUDFLARED_BIN)
    CLOUDFLARED_BIN.chmod(0o755)
    notify("cloudflared_ready", f"cloudflared ready: {CLOUDFLARED_BIN}")
    return CLOUDFLARED_BIN


def get_or_download_model(repo_id: str, filename: str, direct_url: str) -> Path:
    """Find model in Kaggle input, local disk cache, or download with resume support."""
    # 1. Check mounted Kaggle dataset
    if Path("/kaggle/input").exists():
        mounted = [p for p in Path("/kaggle/input").rglob("*.gguf") if "mmproj" not in p.name.lower()]
        for m in mounted:
            name_lower = m.name.lower()
            if filename.lower() in name_lower or "bonsai" in name_lower or "ternary" in name_lower:
                size_gb = m.stat().st_size / (1024**3)
                notify("model_ready", f"Mounted from Dataset: {m.name} ({size_gb:.2f} GB) [0s Mount ⚡]")
                return m
        if mounted:
            size_gb = mounted[0].stat().st_size / (1024**3)
            notify("model_ready", f"Mounted from Dataset: {mounted[0].name} ({size_gb:.2f} GB) [0s Mount ⚡]")
            return mounted[0]

    # 2. Check local disk cache
    target = MODEL_DIR / filename
    if target.exists() and target.stat().st_size > 1024 * 1024 * 1024:
        size_gb = target.stat().st_size / (1024**3)
        notify("model_ready", f"Cached on disk: {target.name} ({size_gb:.2f} GB)")
        return target

    # 3. Download model
    notify("model_download", f"Downloading {filename} (~7.21 GB) to scratch space...")
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODEL_DIR),
        )
        target = Path(downloaded)
    except Exception:
        url = direct_url or f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        download_file(url, target)

    size_gb = target.stat().st_size / (1024**3)
    notify("model_ready", f"Model verified: {target.name} ({size_gb:.2f} GB)")
    return target


def wait_for_server_ready(port: int, api_key: str, timeout_sec: int = 360):
    """Poll /health and /v1/models endpoints until llama-server is live and healthy."""
    notify("loading_vram", f"Loading model weights into 2x Tesla T4 VRAM (Context: {CFG['context_size']:,})...")
    start_t = time.time()
    while time.time() - start_t < timeout_sec:
        global server_proc
        if server_proc and server_proc.poll() is not None:
            last_log = ""
            if LLAMA_LOG.exists():
                last_log = "\n".join(LLAMA_LOG.read_text(errors="replace").splitlines()[-30:])
            notify("failed", f"llama-server exited unexpectedly:\n{last_log}")
            raise RuntimeError("llama-server crashed")

        # 1. Primary check: /health endpoint
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/health",
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    notify("server_ready", "Local llama-server initialized and healthy (/health).")
                    return True
        except Exception:
            pass

        # 2. Secondary check: /v1/models endpoint
        try:
            req_v1 = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/models",
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            )
            with urllib.request.urlopen(req_v1, timeout=3) as resp_v1:
                if resp_v1.status == 200:
                    notify("server_ready", "Local llama-server initialized and healthy (/v1/models).")
                    return True
        except Exception:
            pass

        time.sleep(2)
    return False


def main():
    global server_proc, cf_proc

    setup_directories()

    # Clean old processes
    os.system("pkill -9 -f '[l]lama-server' >/dev/null 2>&1")
    os.system("pkill -9 -f '[c]loudflared' >/dev/null 2>&1")

    gpu_list = check_gpu()
    server_bin = install_prebuilt_llamacpp()
    install_cloudflared()

    model_path = get_or_download_model(
        CFG["model_repo"],
        CFG["model_file"],
        CFG["model_url"],
    )

    # Prepare environment with extracted libraries
    so_dirs = {str(p.parent.resolve()) for p in WORKDIR.rglob("*.so*") if p.is_file()}
    ld_path = ":".join(so_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ld_path
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["GGML_CUDA_NO_VMM"] = "1"

    help_text = subprocess.run([str(server_bin), "--help"], capture_output=True, text=True, env=env).stdout or ""

    notify("server_launch", f"Starting llama-server (2xT4 Split, Context: {CFG['context_size']:,} tokens)...")
    server_cmd = [
        str(server_bin),
        "-m", str(model_path),
        "-ngl", "99",
        "-sm", "row",
        "-ts", "1,1",
        "-c", str(CFG["context_size"]),
        "-b", str(CFG["batch_size"]),
        "-ub", str(CFG["ubatch_size"]),
        "-np", "1",
        "--host", "127.0.0.1",
        "--port", str(CFG["port"]),
        "--alias", CFG["model_name"],
    ]

    # Flash attention flag
    if "-fa" in help_text:
        server_cmd.extend(["-fa", "on"])
    elif "--flash-attn" in help_text:
        server_cmd.extend(["--flash-attn", "on"])
    else:
        server_cmd.extend(["-fa", "on"])

    # KV cache flags: --ctk / --cache-type-k
    ctk_flag = "--ctk" if "--ctk" in help_text or "--cache-type-k" not in help_text else "--cache-type-k"
    ctv_flag = "--ctv" if "--ctv" in help_text or "--cache-type-v" not in help_text else "--cache-type-v"
    server_cmd.extend([ctk_flag, CFG["kv_cache_k"], ctv_flag, CFG["kv_cache_v"]])

    if "-fit" in help_text or "--fit" in help_text:
        server_cmd.extend(["-fit", "off"])

    if CFG.get("api_key"):
        server_cmd.extend(["--api-key", CFG["api_key"]])

    server_log = open(LLAMA_LOG, "w", encoding="utf-8", buffering=1)
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )

    if not wait_for_server_ready(CFG["port"], CFG.get("api_key", ""), timeout_sec=360):
        notify("failed", "Timeout waiting for llama-server initialization.")
        cleanup()
        return

    # Start Cloudflare Tunnel
    notify("tunnel_launch", "Starting Cloudflare Tunnel...")
    cf_token = get_cf_token()
    cf_log = open(CF_LOG, "w", encoding="utf-8", buffering=1)

    public_url = None
    if cf_token:
        print("🔐 Using Cloudflare Named Tunnel token...")
        cf_proc = subprocess.Popen(
            [str(CLOUDFLARED_BIN), "tunnel", "run", "--token", cf_token],
            stdout=cf_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        public_url = CFG.get("static_domain", "https://api.zexnoz.dev").rstrip("/")
    else:
        print("⚠️ No CF_TUNNEL_TOKEN found. Launching free Quick Tunnel fallback...")
        cf_proc = subprocess.Popen(
            [str(CLOUDFLARED_BIN), "tunnel", "--url", f"http://127.0.0.1:{CFG['port']}"],
            stdout=cf_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        start_t = time.time()
        while time.time() - start_t < 45:
            if cf_proc.poll() is not None:
                notify("failed", "cloudflared quick tunnel crashed.")
                break
            if CF_LOG.exists():
                match = re.search(r"https://[-0-9a-z]+\.trycloudflare\.com", CF_LOG.read_text(errors="replace"))
                if match:
                    public_url = match.group(0)
                    break
            time.sleep(2)

    if not public_url:
        notify("failed", "Could not establish Cloudflare Tunnel.")
        cleanup()
        return

    notify("live", f"Server is live at {public_url}/v1", {
        "public_url": f"{public_url}/v1",
        "model": CFG["model_name"],
        "context": CFG["context_size"],
        "batch_size": CFG["batch_size"],
        "ubatch_size": CFG["ubatch_size"],
        "api_key": CFG.get("api_key", ""),
    })

    max_runtime_sec = CFG["keepalive_min"] * 60
    started_serving = time.time()
    try:
        while time.time() - started_serving < max_runtime_sec:
            time.sleep(10)
            if server_proc.poll() is not None:
                notify("failed", "llama-server stopped unexpectedly.")
                break
            if cf_proc.poll() is not None:
                notify("failed", "cloudflared stopped unexpectedly.")
                break
    except KeyboardInterrupt:
        pass

    notify("auto_shutdown", "Session ended. Shutting down cleanly.")
    cleanup()


if __name__ == "__main__":
    main()
