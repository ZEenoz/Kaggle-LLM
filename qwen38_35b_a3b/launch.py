#!/usr/bin/env python3
"""
kaggle-gpu-lab launcher — Serve & Benchmark Qwen3.8-35B-A3B on Kaggle 2x Tesla T4 GPUs.

Usage:
    python launch.py benchmark      # Run sweet-spot context sweep (130k-196k, KV q8_0, batch 1024/512)
    python launch.py serve          # Launch high-performance 24/7 inference server with Cloudflare Tunnel
    python launch.py status         # Check current Kaggle kernel status
    python launch.py stop           # Stop running session on Kaggle Cloud

Supports:
    - ~/.kaggle/access_token (Kaggle Access Token: KGAT_...)
    - ~/.kaggle/kaggle.json  (Legacy API key)
    - Cloudflare Named Tunnel (api.zexnoz.dev) via --cf-token or CF_TUNNEL_TOKEN
    - Automatic Fallback to Free Cloudflare Quick Tunnel if no token provided
"""

import argparse
import json
import os
import re
import secrets
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

# Inject system/enterprise truststore to handle SSL certificates smoothly
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
KERNEL_SRC = HERE / "kernel" / "serve_qwen.py"
BENCHMARK_SRC = HERE / "benchmark_sweet_spot.py"
STATE_FILE = Path.home() / ".kaggle-qwen38-lab.json"

PHASE_MESSAGES = {
    "starting": "Kaggle kernel initialized. Starting GPU environment...",
    "gpu_info": "GPU status: {msg}",
    "gpu_warn": "GPU warning: {msg}",
    "setup_llama": "Fetching prebuilt llama.cpp CUDA 12 binaries (sm_75)...",
    "llama_ready": "llama.cpp CUDA is ready: {msg}",
    "setup_cloudflared": "Fetching cloudflared tunnel client...",
    "cloudflared_ready": "cloudflared ready: {msg}",
    "model_download": "Downloading Qwen3.8-35B-A3B GGUF (~20.2 GB) to scratch space...",
    "model_ready": "Model verified: {msg}",
    "server_launch": "Starting llama-server (2x Tesla T4 split, 196k context, batch 1024/512)...",
    "loading_vram": "Loading model weights into 2x Tesla T4 VRAM...",
    "server_ready": "Local server initialized and healthy.",
    "tunnel_launch": "Establishing Cloudflare Tunnel...",
    "live": "SERVER IS LIVE!",
    "failed": "ERROR: {msg}",
    "auto_shutdown": "Keepalive window expired. Shutting down cleanly.",
    "stopping": "Stopping services...",
    "stopped": "Services stopped."
}

def say(msg):
    now = time.strftime("[%H:%M:%S]")
    print(f"{now}  {msg}", flush=True)

def get_kaggle_api():
    """Authenticate KaggleApi supporting both ~/.kaggle/access_token and kaggle.json."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("❌ Package 'kaggle' is not installed. Run: pip install kaggle")

    kaggle_dir = Path.home() / ".kaggle"
    access_token_file = kaggle_dir / "access_token"
    kaggle_json_file = kaggle_dir / "kaggle.json"

    has_token = access_token_file.exists() and access_token_file.stat().st_size > 0
    has_json = kaggle_json_file.exists() and kaggle_json_file.stat().st_size > 0
    has_env = bool(os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")))

    if not (has_token or has_json or has_env):
        print("=" * 70)
        print("🔑 ไม่พบไฟล์ยืนยันตัวตน Kaggle API (~/.kaggle/access_token หรือ kaggle.json)")
        print("=" * 70)
        print("วิธีรับ Token จาก Kaggle:")
        print("1. เข้าเว็บ https://www.kaggle.com/settings")
        print("2. เลื่อนไปที่หัวข้อ 'API' แล้วคลิก 'Create New Token'")
        print(f"3. บันทึก Token ลงที่: {access_token_file}")
        print("-" * 70)
        
        choice = input("ต้องการป้อน Access Token ตอนนี้เลยหรือไม่? [y/N]: ").strip().lower()
        if choice in ("y", "yes"):
            token_val = input("Kaggle Access Token (KGAT_...): ").strip()
            if token_val:
                kaggle_dir.mkdir(parents=True, exist_ok=True)
                access_token_file.write_text(token_val.strip(), encoding="utf-8")
                print("✅ บันทึก ~/.kaggle/access_token เรียบร้อยแล้ว")
            else:
                sys.exit(1)
        else:
            sys.exit("❌ กรุณาตั้งค่า Kaggle credentials ก่อนใช้งาน")

    api = KaggleApi()
    api.authenticate()
    return api

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_state(data):
    STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def poll_events(topic):
    url = f"https://ntfy.sh/{topic}/json"
    req = urllib.request.Request(url, headers={"User-Agent": "qwen38-launcher/1.0"})
    seen_ids = set()

    say("Listening for server telemetry from Kaggle...")
    while True:
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                for line in resp:
                    if not line:
                        continue
                    try:
                        evt = json.loads(line.decode("utf-8", errors="replace"))
                        eid = evt.get("id")
                        if eid and eid in seen_ids:
                            continue
                        if eid:
                            seen_ids.add(eid)

                        event_type = evt.get("event")
                        if event_type == "message":
                            raw_msg = evt.get("message", "")
                            data = json.loads(raw_msg) if raw_msg.startswith("{") else {}
                            phase = data.get("phase", "status")
                            msg = data.get("message", raw_msg)
                            extra = data.get("extra", {})

                            pattern = PHASE_MESSAGES.get(phase, "[{phase}] {msg}")
                            formatted = pattern.format(phase=phase, msg=msg)
                            say(formatted)

                            if phase == "live":
                                render_live_banner(extra)
                                return
                            elif phase == "failed":
                                say("❌ Execution failed on Kaggle Cloud.")
                                return
                    except Exception:
                        pass
        except urllib.error.URLError:
            time.sleep(2)
        except Exception:
            time.sleep(2)

def render_live_banner(extra):
    public_url = extra.get("public_url", "https://api.zexnoz.dev/v1")
    model = extra.get("model", "qwen3.8-35B-A3B")
    context = extra.get("context", 196608)
    batch_size = extra.get("batch_size", 1024)
    ubatch_size = extra.get("ubatch_size", 512)
    api_key = extra.get("api_key", "kilo-secret-key")
    
    print("\n" + "=" * 74)
    print("🚀 QWEN3.8-35B-A3B SERVER ONLINE (2x TESLA T4)")
    print("=" * 74)
    print(f"📡 Base URL       : {public_url}")
    print(f"🤖 Model ID       : {model}")
    print(f"🧠 Context Length : {context:,} tokens (196k)")
    print(f"⚡ Batch / UBatch : {batch_size} / {ubatch_size} (KV: q8_0 | Flash-Attn: ON)")
    if api_key:
        print(f"🔑 API Key        : {api_key}")
    else:
        print("🔑 API Key        : Not required (open access)")
    print("=" * 74)

    print("\n📌 Quick test via curl:")
    auth_header = f' -H "Authorization: Bearer {api_key}"' if api_key else ""
    print(f"""curl {public_url}/chat/completions \\{auth_header} \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "{model}", "messages": [{{"role": "user", "content": "สวัสดีครับ Qwen 3.8"}}]}}'""")

    print("\n📌 Use with Claude Code / Codex / Cursor / OpenAI SDK:")
    print(f"export OPENAI_BASE_URL=\"{public_url}\"")
    print(f"export OPENAI_API_KEY=\"{api_key or 'none'}\"")
    print(f"export OPENAI_MODEL=\"{model}\"")
    print("-" * 74)
    print("🔔 Session running in cloud. Press Ctrl+C to detach (server stays online).")

def cmd_serve(args):
    api = get_kaggle_api()
    username = api.get_config_value("username") or api.config_values.get("username")
    if not username:
        username = "zeenoz"
    
    slug = args.slug
    topic = "kqwen38-" + uuid.uuid4().hex[:16]
    cf_token = args.cf_token or os.environ.get("CF_TUNNEL_TOKEN", "")

    cfg = {
        "ntfy_topic": topic,
        "api_key": args.api_key,
        "context_size": args.context,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "keepalive_min": args.keepalive,
        "model_name": args.model_name,
        "static_domain": args.static_domain,
        "cf_token": cf_token,
    }

    if not KERNEL_SRC.exists():
        sys.exit(f"❌ Missing kernel script: {KERNEL_SRC}")

    src = KERNEL_SRC.read_text(encoding="utf-8")
    src, n = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$", f"CFG = {cfg!r}", src, count=1, flags=re.M)
    if n != 1:
        sys.exit("❌ Could not inject configuration into kernel/serve_qwen.py")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "serve_qwen.py").write_text(src, encoding="utf-8")
        meta = {
            "id": f"{username}/{slug}",
            "title": slug,
            "code_file": "serve_qwen.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [args.dataset] if args.dataset else [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": []
        }
        (td_path / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        say(f"Pushing kernel '{username}/{slug}' to Kaggle Cloud (2x Tesla T4)...")
        try:
            resp = api.kernels_push(str(td_path), acc="NvidiaTeslaT4")
            if hasattr(resp, "error") and resp.error:
                sys.exit(f"❌ Failed to push kernel: {resp.error}")
        except TypeError:
            api.kernels_push(str(td_path))

        save_state({
            "kernel_id": f"{username}/{slug}",
            "topic": topic,
            "started_at": time.time(),
            "type": "serve",
            "config": cfg,
        })
        say(f"Kernel '{slug}' successfully pushed to Kaggle!")

    try:
        poll_events(topic)
    except KeyboardInterrupt:
        say("Detached from stream. Kernel continues running in background on Kaggle.")

def cmd_benchmark(args):
    api = get_kaggle_api()
    username = api.get_config_value("username") or api.config_values.get("username")
    if not username:
        username = "zeenoz"

    slug = args.slug or "qwen38-35b-sweet-spot-bench"
    if not BENCHMARK_SRC.exists():
        sys.exit(f"❌ Missing benchmark script: {BENCHMARK_SRC}")

    src = BENCHMARK_SRC.read_text(encoding="utf-8")

    # Override defaults if specified in CLI
    if args.context_sizes:
        src = re.sub(r"DEFAULT_CONTEXT_SIZES = \(.*?\)", f"DEFAULT_CONTEXT_SIZES = tuple({[int(c.strip()) for c in args.context_sizes.split(',')]})", src)
    if args.batch_size:
        src = re.sub(r"DEFAULT_BATCH_SIZE = \d+", f"DEFAULT_BATCH_SIZE = {args.batch_size}", src)
    if args.ubatch_size:
        src = re.sub(r"DEFAULT_UBATCH_SIZE = \d+", f"DEFAULT_UBATCH_SIZE = {args.ubatch_size}", src)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "benchmark_sweet_spot.py").write_text(src, encoding="utf-8")
        meta = {
            "id": f"{username}/{slug}",
            "title": slug,
            "code_file": "benchmark_sweet_spot.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [args.dataset] if args.dataset else [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": []
        }
        (td_path / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        say(f"Pushing benchmark kernel '{username}/{slug}' to Kaggle Cloud (2x Tesla T4)...")
        try:
            resp = api.kernels_push(str(td_path), acc="NvidiaTeslaT4")
            if hasattr(resp, "error") and resp.error:
                sys.exit(f"❌ Failed to push benchmark kernel: {resp.error}")
        except TypeError:
            api.kernels_push(str(td_path))

        save_state({
            "kernel_id": f"{username}/{slug}",
            "started_at": time.time(),
            "type": "benchmark",
        })
        say(f"Benchmark kernel pushed! You can monitor live logs via:")
        print(f"  kaggle kernels status {username}/{slug}")
        print(f"  kaggle kernels output {username}/{slug}")

def cmd_status(args):
    api = get_kaggle_api()
    state = load_state()
    kernel_id = state.get("kernel_id", "zeenoz/qwen38-35b-2x-t4-server")
    say(f"Checking status for {kernel_id}...")
    try:
        status = api.kernels_status(kernel_id)
        print(f"📊 Status: {status.get('status', 'unknown')}")
        if "failureMessage" in status and status["failureMessage"]:
            print(f"⚠️ Error: {status['failureMessage']}")
    except Exception as e:
        print(f"❌ Could not query kernel status: {e}")

def cmd_stop(args):
    api = get_kaggle_api()
    state = load_state()
    kernel_id = state.get("kernel_id", "zeenoz/qwen38-35b-2x-t4-server")
    say(f"Stopping kernel {kernel_id}...")
    try:
        api.kernels_cancel(kernel_id)
        say(f"✅ Kernel {kernel_id} has been stopped.")
    except Exception as e:
        print(f"⚠️ Could not cancel kernel: {e}")

def main():
    parser = argparse.ArgumentParser(description="Kaggle 2x Tesla T4 Runner for Qwen3.8-35B-A3B")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve command
    p_serve = subparsers.add_parser("serve", help="Deploy inference server on Kaggle")
    p_serve.add_argument("--slug", default="qwen38-35b-2x-t4-server", help="Kernel slug on Kaggle")
    p_serve.add_argument("--cf-token", help="Cloudflare Tunnel Token (api.zexnoz.dev)")
    p_serve.add_argument("--static-domain", default="https://api.zexnoz.dev", help="Custom domain")
    p_serve.add_argument("--context", type=int, default=196608, help="Context length (default: 196608)")
    p_serve.add_argument("--batch-size", type=int, default=1024, help="Batch size (default: 1024)")
    p_serve.add_argument("--ubatch-size", type=int, default=512, help="UBatch size (default: 512)")
    p_serve.add_argument("--api-key", default="kilo-secret-key", help="Server API key")
    p_serve.add_argument("--keepalive", type=int, default=480, help="Max run duration in minutes")
    p_serve.add_argument("--model-name", default="qwen3.8-35B-A3B", help="Model alias")
    p_serve.add_argument("--dataset", help="Kaggle dataset to mount for instant boot")

    # benchmark command
    p_bench = subparsers.add_parser("benchmark", help="Run sweet-spot context sweep on Kaggle")
    p_bench.add_argument("--slug", default="qwen38-35b-sweet-spot-bench", help="Kernel slug for benchmark")
    p_bench.add_argument("--context-sizes", default="131072,147456,163840,180224,196608", help="Context sizes to sweep")
    p_bench.add_argument("--batch-size", type=int, default=1024, help="Batch size (default: 1024)")
    p_bench.add_argument("--ubatch-size", type=int, default=512, help="UBatch size (default: 512)")
    p_bench.add_argument("--dataset", help="Kaggle dataset to mount for instant boot")

    # status command
    subparsers.add_parser("status", help="Check server status")

    # stop command
    subparsers.add_parser("stop", help="Stop server on Kaggle")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "stop":
        cmd_stop(args)

if __name__ == "__main__":
    main()
