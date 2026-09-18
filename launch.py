#!/usr/bin/env python3
"""
kaggle-gpu-lab launcher — Serve Qwen3.8-27B on free Kaggle 2x Tesla T4 GPUs from your local terminal.

Usage:
    python launch.py serve          # Push kernel to Kaggle and stream live progress
    python launch.py status         # Check current server status
    python launch.py stop           # Stop running session on Kaggle

Supports both:
    - ~/.kaggle/access_token (Kaggle Access Token: KGAT_...)
    - ~/.kaggle/kaggle.json  (Legacy API key)
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
STATE_FILE = Path.home() / ".kaggle-gpu-lab.json"

PHASE_MESSAGES = {
    "starting": "Kaggle kernel initialized. Starting GPU environment...",
    "gpu_info": "GPU status: {msg}",
    "setup_llama": "Fetching prebuilt llama.cpp CUDA 12 binaries...",
    "llama_ready": "llama.cpp CUDA is ready.",
    "setup_cloudflared": "Fetching cloudflared tunnel client...",
    "cloudflared_ready": "cloudflared ready.",
    "model_download": "Downloading Qwen3.8-27B GGUF (~17 GB) to scratch space (takes ~1.5 - 2 min)...",
    "model_ready": "Model verified: {msg}",
    "server_launch": "Starting llama-server (2x Tesla T4 split, 128k context)...",
    "loading_vram": "Loading model weights into GPU VRAM...",
    "server_ready": "Local server initialized and healthy.",
    "tunnel_launch": "Establishing free Cloudflare Quick Tunnel...",
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
        sys.exit("❌ Package 'kaggle' is not installed. Run: pip install -r requirements.txt")

    kaggle_dir = Path.home() / ".kaggle"
    access_token_file = kaggle_dir / "access_token"
    kaggle_json_file = kaggle_dir / "kaggle.json"

    # Check if credentials exist
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
                access_token_file.write_text(token_val, encoding="utf-8")
                print("✅ บันทึก ~/.kaggle/access_token เรียบร้อยแล้ว!\n")
            else:
                sys.exit("❌ ข้อมูลไม่ครบถ้วน ยกเลิกการทำงาน")
        else:
            sys.exit("❌ กรุณาใส่ ~/.kaggle/access_token แล้วลองใหม่อีกครั้ง")

    try:
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        sys.exit(f"❌ Kaggle API authentication failed:\n{e}")

def stream_events(ntfy_topic):
    """Listen to ntfy.sh SSE stream and display live progress."""
    url = f"https://ntfy.sh/{ntfy_topic}/json"
    req = urllib.request.Request(url, headers={"User-Agent": "kaggle-gpu-lab-client"})
    
    say("Waiting for Kaggle to provision GPU VM and start script...")
    
    while True:
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("event") == "message" and "message" in data:
                            msg_content = data.get("message", "")
                            try:
                                event_obj = json.loads(msg_content)
                                phase = event_obj.get("phase", "")
                                msg = event_obj.get("message", "")
                                extra = event_obj.get("extra", {})
                            except Exception:
                                phase = "info"
                                msg = msg_content
                                extra = {}

                            tmpl = PHASE_MESSAGES.get(phase, "{msg}")
                            formatted = tmpl.format(msg=msg)
                            say(formatted)

                            if phase == "live":
                                render_live_banner(extra)
                                return
                            elif phase == "failed":
                                say("❌ Deployment failed on Kaggle Cloud.")
                                return
                    except Exception:
                        pass
        except urllib.error.URLError:
            time.sleep(2)
        except Exception:
            time.sleep(2)

def render_live_banner(extra):
    public_url = extra.get("public_url", "")
    model = extra.get("model", "qwen3.8-27b")
    context = extra.get("context", 131072)
    api_key = extra.get("api_key", "")
    
    print("\n" + "=" * 70)
    print("🎉 YOUR QWEN 3.8 (27B) ENDPOINT IS LIVE!")
    print("=" * 70)
    print(f"🌐 Public API Base   : {public_url}")
    print(f"🧠 Model ID          : {model}")
    print(f"📏 Context Length    : {context:,} tokens (128k)")
    if api_key:
        print(f"🔑 API Key           : {api_key}")
    else:
        print(f"🔑 API Key           : Not required (open access)")
    print("=" * 70)

    print("\n📌 Quick test via curl:")
    print(f"""curl {public_url}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "{model}", "messages": [{{"role": "user", "content": "สวัสดีครับ Qwen"}}]}}'""")

    print("\n📌 Use with Claude Code / Codex / OpenAI:")
    print(f"export OPENAI_BASE_URL=\"{public_url}\"")
    print(f"export OPENAI_API_KEY=\"{api_key or 'none'}\"")
    print(f"export OPENAI_MODEL=\"{model}\"")
    print("-" * 70)
    print("🔔 Session running in cloud. Press Ctrl+C to detach (server stays online).")

def cmd_serve(args):
    api = get_kaggle_api()
    username = api.get_config_value("username") or api.config_values.get("username")
    if not username:
        username = "zeenoz"
    
    slug = args.slug
    topic = "kqwen-" + uuid.uuid4().hex[:16]
    api_key = ("sk-" + secrets.token_hex(16)) if args.require_key else ""

    cfg = {
        "ntfy_topic": topic,
        "api_key": api_key,
        "context_size": args.context,
        "keepalive_min": args.keepalive,
        "model_name": args.model_name
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
            say(f"Kernel pushed successfully! URL: https://www.kaggle.com/code/{username}/{slug}")
        except Exception as e:
            sys.exit(f"❌ Failed to push kernel:\n{e}")

    # Save state
    state = {
        "kernel": f"{username}/{slug}",
        "topic": topic,
        "api_key": api_key,
        "started_at": time.time(),
        "slug": slug
    }
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    try:
        stream_events(topic)
    except KeyboardInterrupt:
        print("\n👋 Detached. The server is still running on Kaggle Cloud!")
        print("To check status: python launch.py status")
        print("To stop server : python launch.py stop")

def cmd_status(args):
    api = get_kaggle_api()
    if not STATE_FILE.exists():
        sys.exit("No active deployment found. Run `python launch.py serve` first.")
    
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    kernel_id = state["kernel"]
    say(f"Checking Kaggle status for {kernel_id}...")
    try:
        res = api.kernels_status(kernel_id)
        print(json.dumps(res, indent=2, ensure_ascii=False) if isinstance(res, dict) else str(res))
    except Exception as e:
        print(f"Error querying status: {e}")

def cmd_stop(args):
    api = get_kaggle_api()
    if not STATE_FILE.exists():
        sys.exit("❌ ไม่พบเซสชันที่กำลังทำงานอยู่ (No active deployment found)")
    
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    kernel_id = state["kernel"]
    say(f"Stopping and terminating GPU session for '{kernel_id}' on Kaggle...")
    try:
        api.kernels_delete(kernel_id, no_confirm=True)
        say("✅ เซิร์ฟเวอร์และเซสชัน GPU บน Kaggle ถูกสั่งหยุดการทำงานเรียบร้อยแล้ว (GPU Quota released)")
    except Exception as e:
        say(f"⚠️ Note: {e}")
        print(f"คุณสามารถตรวจสอบหรือกด Stop ผ่านหน้าเว็บได้ที่: https://www.kaggle.com/code/{kernel_id}")
    
    try:
        STATE_FILE.unlink(missing_ok=True)
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Kaggle GPU Lab: Serve Qwen 3.8 27B on free 2x Tesla T4 GPUs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve command
    p_serve = subparsers.add_parser("serve", help="Deploy Qwen to Kaggle 2xT4 and stream output")
    p_serve.add_argument("--slug", default="qwen38-2x-t4-server", help="Kaggle kernel slug name")
    p_serve.add_argument("--context", type=int, default=131072, help="Context window size (default: 131072)")
    p_serve.add_argument("--keepalive", type=int, default=480, help="Max session minutes (default: 480)")
    p_serve.add_argument("--require-key", action="store_true", help="Generate and enforce an API key")
    p_serve.add_argument("--model-name", default="qwen3.8-27b", help="Model alias name for API")
    p_serve.add_argument("--dataset", default="", help="Attach existing Kaggle dataset (e.g. drvivektrivedi34/qwen38-27b-q4km)")

    # status command
    subparsers.add_parser("status", help="Check deployment status")

    # stop command
    subparsers.add_parser("stop", help="Stop running session")

    args = parser.parse_args()
    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "stop":
        cmd_stop(args)

if __name__ == "__main__":
    main()
