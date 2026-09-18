#!/usr/bin/env python3
"""
Serve Qwen3.8-27B on Kaggle 2x Tesla T4 GPUs with llama.cpp and Cloudflare Quick Tunnel.
Pushed as a script kernel to Kaggle Cloud by launch.py.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CFG = None  # __LAUNCHER_CONFIG__ (launch.py replaces this line with dictionary)

DEFAULTS = {
    "model_url": "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf",
    "model_name": "qwen3.8-27b",
    "context_size": 131072,
    "batch_size": 2048,
    "ubatch_size": 512,
    "kv_cache_k": "q4_0",
    "kv_cache_v": "q4_0",
    "port": 8080,
    "keepalive_min": 480,
    "api_key": "",
    "ntfy_topic": "",
}

CFG = {**DEFAULTS, **(CFG or {})}

WORK_DIR = Path("/kaggle/tmp")
LOG_DIR = Path("/kaggle/working")
MODEL_PATH = WORK_DIR / "Qwen3.8-27B-UD-Q4_K_M.gguf"
LLAMA_DIR = WORK_DIR / "llama_cpp"
CF_BIN = WORK_DIR / "cloudflared"
LLAMA_LOG = LOG_DIR / "llama-server.log"
CF_LOG = LOG_DIR / "cloudflared.log"

server_proc = None
cf_proc = None

def notify(phase, message="", extra=None):
    payload = {
        "phase": phase,
        "message": message,
        "extra": extra or {},
        "timestamp": time.time()
    }
    print(f"[{time.strftime('%H:%M:%S')}] [{phase}] {message}", flush=True)
    if CFG.get("ntfy_topic"):
        try:
            url = f"https://ntfy.sh/{CFG['ntfy_topic']}"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Title": f"Kaggle Qwen: {phase}", "Tags": "robot"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            pass

def cleanup(sig=None, frame=None):
    global server_proc, cf_proc
    notify("stopping", "Cleaning up services...")
    if server_proc and server_proc.poll() is None:
        server_proc.terminate()
    if cf_proc and cf_proc.poll() is None:
        cf_proc.terminate()
    os.system("pkill -9 -f '[l]lama-server' >/dev/null 2>&1")
    os.system("pkill -9 -f '[c]loudflared' >/dev/null 2>&1")
    notify("stopped", "Server stopped.")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

WORK_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Kill old processes
os.system("pkill -9 -f '[l]lama-server' >/dev/null 2>&1")
os.system("pkill -9 -f '[c]loudflared' >/dev/null 2>&1")

notify("starting", "Kernel initialized. Checking GPU environment...")

# 1. Check GPU
try:
    gpu_out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True
    ).strip().splitlines()
    notify("gpu_info", f"Detected {len(gpu_out)} GPU(s): {', '.join(gpu_out)}")
except Exception as e:
    notify("gpu_warn", f"Could not query GPUs: {e}")

# 2. Setup llama.cpp CUDA binaries
notify("setup_llama", "Setting up prebuilt llama.cpp CUDA 12 binaries...")
server_bin = None
for candidate in WORK_DIR.rglob("llama-server"):
    if candidate.is_file() and os.access(candidate, os.X_OK):
        server_bin = str(candidate)
        break

if not server_bin:
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = WORK_DIR / "llama_cpp_binaries.tar.gz"
    bin_url = "https://github.com/ai-dock/llama.cpp-cuda/releases/download/b9628/llama.cpp-b9628-cuda-12.8-amd64.tar.gz"
    if not tar_path.exists():
        subprocess.run(["wget", "-q", bin_url, "-O", str(tar_path)], check=True)
    subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(LLAMA_DIR)], check=True)
    for candidate in WORK_DIR.rglob("llama-server"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            server_bin = str(candidate)
            break

extract_dir = os.path.dirname(server_bin)
os.environ["PATH"] = f"{extract_dir}:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"{extract_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
notify("llama_ready", f"llama.cpp CUDA ready at {extract_dir}")

# 3. Setup cloudflared
notify("setup_cloudflared", "Downloading cloudflared...")
if not CF_BIN.exists():
    cf_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    subprocess.run(["wget", "-q", cf_url, "-O", str(CF_BIN)], check=True)
    CF_BIN.chmod(0o755)
notify("cloudflared_ready", "cloudflared ready.")

# 4. Check if Model is mounted from a Kaggle Dataset in /kaggle/input, else download
mounted_models = [
    p for p in Path("/kaggle/input").rglob("*.gguf")
    if "mmproj" not in p.name.lower()
] if Path("/kaggle/input").exists() else []

if mounted_models:
    MODEL_PATH = mounted_models[0]
    size_gb = MODEL_PATH.stat().st_size / (1024**3)
    notify("model_ready", f"Mounted from Dataset: {MODEL_PATH.name} ({size_gb:.2f} GB) [Instant Mount ⚡]")
else:
    if not MODEL_PATH.exists():
        notify("model_download", "Downloading Qwen3.8-27B GGUF (~17 GB) to scratch storage...")
        subprocess.run([
            "wget", "--continue", "--progress=bar:force:noscroll",
            CFG["model_url"], "-O", str(MODEL_PATH)
        ], check=True)
    size_gb = MODEL_PATH.stat().st_size / (1024**3)
    notify("model_ready", f"Model verified on disk ({size_gb:.2f} GB)")

# 5. Launch llama-server
notify("server_launch", f"Starting llama-server (2xT4 Split, Context: {CFG['context_size']} tokens)...")

help_text = subprocess.run([server_bin, "--help"], capture_output=True, text=True, env=os.environ).stdout or ""
gpu_count = len(gpu_out) if 'gpu_out' in locals() and gpu_out else 1
server_cmd = [
    server_bin,
    "-m", str(MODEL_PATH),
    "--alias", CFG["model_name"],
    "-ngl", "99",
    "-c", str(CFG["context_size"]),
    "-np", "1",
    "-fa", "on",
    "-ctk", CFG["kv_cache_k"],
    "-ctv", CFG["kv_cache_v"],
    "-b", str(CFG["batch_size"]),
    "-ub", str(CFG["ubatch_size"]),
    "--host", "127.0.0.1",
    "--port", str(CFG["port"]),
]

if gpu_count >= 2:
    server_cmd.extend(["-sm", "layer", "-ts", "1,1"])

if "-fit" in help_text or "--fit" in help_text:
    server_cmd.extend(["-fit", "off"])

if CFG.get("api_key"):
    server_cmd.extend(["--api-key", CFG["api_key"]])

llama_log_file = open(LLAMA_LOG, "w", encoding="utf-8", buffering=1)
server_proc = subprocess.Popen(
    server_cmd,
    stdout=llama_log_file,
    stderr=subprocess.STDOUT,
    text=True,
    env=os.environ
)

# Wait for local health check
notify("loading_vram", "Loading model weights into 2x Tesla T4 VRAM...")
ready = False
start_t = time.time()
while time.time() - start_t < 300:
    if server_proc.poll() is not None:
        last_log = "\n".join(Path(LLAMA_LOG).read_text(errors="replace").splitlines()[-25:])
        notify("failed", f"llama-server crashed:\n{last_log}")
        sys.exit(1)
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{CFG['port']}/v1/models")
        if CFG.get("api_key"):
            req.add_header("Authorization", f"Bearer {CFG['api_key']}")
        with urllib.request.urlopen(req, timeout=3) as r:
            if r.status == 200:
                ready = True
                break
    except Exception:
        pass
    time.sleep(2)

if not ready:
    tail_log = ""
    if Path(LLAMA_LOG).exists():
        tail_log = "\n".join(Path(LLAMA_LOG).read_text(errors="replace").splitlines()[-25:])
    notify("failed", f"Timeout waiting for llama-server initialization.\nLast logs:\n{tail_log}")
    sys.exit(1)

notify("server_ready", "Local llama-server is healthy and ready.")

# 6. Launch Cloudflare Quick Tunnel
notify("tunnel_launch", "Creating free Cloudflare Quick Tunnel...")
cf_log_file = open(CF_LOG, "w", encoding="utf-8", buffering=1)
cf_proc = subprocess.Popen(
    [str(CF_BIN), "tunnel", "--url", f"http://127.0.0.1:{CFG['port']}"],
    stdout=cf_log_file,
    stderr=subprocess.STDOUT,
    text=True
)

public_url = None
start_t = time.time()
while time.time() - start_t < 45:
    if cf_proc.poll() is not None:
        notify("failed", "cloudflared tunnel failed to start.")
        break
    if CF_LOG.exists():
        match = re.search(r"https://[-0-9a-z]+\.trycloudflare\.com", CF_LOG.read_text(errors="replace"))
        if match:
            public_url = match.group(0)
            break
    time.sleep(2)

if not public_url:
    notify("failed", "Could not obtain Cloudflare public URL.")
    sys.exit(1)

# Broadcast LIVE status
notify("live", f"Server is live at {public_url}/v1", {
    "public_url": f"{public_url}/v1",
    "model": CFG["model_name"],
    "context": CFG["context_size"],
    "api_key": CFG.get("api_key", "")
})

# 7. Keepalive loop
max_runtime_sec = CFG["keepalive_min"] * 60
started_serving = time.time()

while time.time() - started_serving < max_runtime_sec:
    if server_proc.poll() is not None or cf_proc.poll() is not None:
        notify("failed", "One of the background services crashed unexpectedly.")
        break
    time.sleep(30)

notify("auto_shutdown", "Keepalive window ended. Shutting down cleanly.")
cleanup()
