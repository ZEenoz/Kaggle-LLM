#!/usr/bin/env python3
"""
kaggle-gpu-lab launcher — Serve Ornith-1.5-35B-A3B on free Kaggle 2x Tesla T4 GPUs from your local terminal.

Usage:
    python launch.py serve          # Push kernel to Kaggle and stream live progress
    python launch.py status         # Check current server status
    python launch.py stop           # Stop running session on Kaggle

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
import shutil
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
KERNEL_SRC = HERE / "kernel" / "serve_ornith.py"
STATE_FILE = Path.home() / ".kaggle-ornith15-lab.json"

PHASE_MESSAGES = {
    "starting": "Kaggle kernel initialized. Starting GPU environment...",
    "gpu_info": "GPU status: {msg}",
    "gpu_warn": "GPU warning: {msg}",
    "setup_llama": "Fetching prebuilt llama.cpp CUDA 12 binaries (sm_75)...",
    "llama_ready": "llama.cpp CUDA is ready: {msg}",
    "setup_cloudflared": "Fetching cloudflared tunnel client...",
    "cloudflared_ready": "cloudflared ready: {msg}",
    "model_download": "Downloading Ornith-1.5-35B-A3B TIEL-ICE GGUF (~20.8 GB) to scratch space...",
    "model_ready": "Model verified: {msg}",
    "server_launch": "Starting llama-server (2x Tesla T4 split, 192k context, batch 1024/512)...",
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

    # Check if credentials exist
    has_token = access_token_file.exists() and access_token_file.stat().st_size > 0
    has_json = kaggle_json_file.exists() and kaggle_json_file.stat().st_size > 0
    has_env = bool(os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")))

    if not (has_token or has_json or has_env):
        print("=" * 70)
        print("🔑 Kaggle API credentials not found (~/.kaggle/access_token or kaggle.json)")
        print("=" * 70)
        print("How to get a Kaggle token:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll to the 'API' section and click 'Create New Token'")
        print(f"3. Save the token to: {access_token_file}")
        print("-" * 70)
        
        choice = input("Enter Access Token now? [y/N]: ").strip().lower()
        if choice in ("y", "yes"):
            token_val = input("Kaggle Access Token (KGAT_...): ").strip()
            if token_val:
                kaggle_dir.mkdir(parents=True, exist_ok=True)
                access_token_file.write_text(token_val, encoding="utf-8")
                print("✅ ~/.kaggle/access_token saved!\n")
            else:
                sys.exit("❌ Incomplete input, aborting")
        else:
            sys.exit("❌ Please place ~/.kaggle/access_token and try again")

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
    public_url = extra.get("public_url", "https://api.zexnoz.dev/v1")
    model = extra.get("model", "ornith-1.5-35B-A3B")
    context = extra.get("context", 131072)
    batch_size = extra.get("batch_size", 1024)
    ubatch_size = extra.get("ubatch_size", 512)
    api_key = extra.get("api_key", "kilo-secret-key")
    
    print("\n" + "=" * 74)
    print("🚀 ORNITH-1.5-35B-A3B SERVER ONLINE (2x TESLA T4)")
    print("=" * 74)
    print(f"📡 Base URL       : {public_url}")
    print(f"🤖 Model ID       : {model}")
    print(f"🧠 Context Length : {context:,} tokens (128k)")
    print(f"⚡ Batch / UBatch : {batch_size} / {ubatch_size} (KV: q4_0)")
    if api_key:
        print(f"🔑 API Key        : {api_key}")
    else:
        print(f"🔑 API Key        : Not required (open access)")
    print("=" * 74)

    print("\n📌 Quick test via curl:")
    auth_header = f' -H "Authorization: Bearer {api_key}"' if api_key else ""
    print(f"""curl {public_url}/chat/completions \\{auth_header} \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "{model}", "messages": [{{"role": "user", "content": "Hello Ornith 1.5"}}]}}'""")

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
    topic = "kornith15-" + uuid.uuid4().hex[:16]

    cf_token = args.cf_token or os.environ.get("CF_TUNNEL_TOKEN", "")
    hf_token = getattr(args, "hf_token", "") or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        local_token_file = Path.home() / ".cache" / "huggingface" / "token"
        if local_token_file.exists():
            try:
                hf_token = local_token_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass

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
        "hf_token": hf_token,
    }

    if not KERNEL_SRC.exists():
        sys.exit(f"❌ Missing kernel script: {KERNEL_SRC}")

    src = KERNEL_SRC.read_text(encoding="utf-8")
    src, n = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$", f"CFG = {cfg!r}", src, count=1, flags=re.M)
    if n != 1:
        sys.exit("❌ Could not inject configuration into kernel/serve_ornith.py")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "serve_ornith.py").write_text(src, encoding="utf-8")
        meta = {
            "id": f"{username}/{slug}",
            "title": slug,
            "code_file": "serve_ornith.py",
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
        "api_key": args.api_key,
        "started_at": time.time(),
        "slug": slug,
        "static_domain": args.static_domain
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
        sys.exit("❌ No running session found. Run `python launch.py serve` first.")
    
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    kernel_id = state["kernel"]
    say(f"Stopping and terminating GPU session for '{kernel_id}' on Kaggle...")
    try:
        api.kernels_delete(kernel_id, no_confirm=True)
        say("✅ Server and GPU session on Kaggle stopped (GPU quota released)")
    except Exception as e:
        say(f"⚠️ Note: {e}")
        print(f"You can also check or Stop via the web: https://www.kaggle.com/code/{kernel_id}")
    
    try:
        STATE_FILE.unlink(missing_ok=True)
    except Exception:
        pass

def cmd_benchmark(args):
    api = get_kaggle_api()
    username = api.get_config_value("username") or api.config_values.get("username")
    if not username:
        username = "zeenoz"

    slug = args.slug
    bench_src = HERE / "benchmark_sweet_spot.py"
    if not bench_src.exists():
        sys.exit(f"❌ Missing benchmark script: {bench_src}")

    hf_token = getattr(args, "hf_token", "") or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        local_token_file = Path.home() / ".cache" / "huggingface" / "token"
        if local_token_file.exists():
            try:
                hf_token = local_token_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass

    src = bench_src.read_text(encoding="utf-8")
    if hf_token:
        src = re.sub(
            r'^DEFAULT_HF_TOKEN\s*=\s*".*?"',
            f'DEFAULT_HF_TOKEN = "{hf_token}"',
            src,
            flags=re.M,
        )

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
                sys.exit(f"❌ Failed to push benchmark: {resp.error}")
            say(f"Benchmark kernel pushed successfully! URL: https://www.kaggle.com/code/{username}/{slug}")
            say("Check execution status via: python launch.py status")
        except Exception as e:
            sys.exit(f"❌ Failed to push benchmark:\n{e}")

    state = {
        "kernel": f"{username}/{slug}",
        "topic": "benchmark",
        "api_key": "",
        "started_at": time.time(),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Kaggle GPU Lab: Serve Ornith 1.5 35B-A3B on free 2x Tesla T4 GPUs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve command
    p_serve = subparsers.add_parser("serve", help="Deploy Ornith-1.5-35B to Kaggle 2xT4 and stream output")
    p_serve.add_argument("--slug", default="ornith15-35b-2x-t4-server", help="Kaggle kernel slug name")
    p_serve.add_argument("--context", type=int, default=196608, help="Context window size (default: 196608)")
    p_serve.add_argument("--batch-size", type=int, default=1024, help="Batch size (default: 1024)")
    p_serve.add_argument("--ubatch-size", type=int, default=512, help="Micro-batch size (default: 512)")
    p_serve.add_argument("--keepalive", type=int, default=480, help="Max session minutes (default: 480)")
    p_serve.add_argument("--api-key", default="kilo-secret-key", help="API key (default: kilo-secret-key)")
    p_serve.add_argument("--model-name", default="ornith-1.5-35B-A3B", help="Model alias name for API")
    p_serve.add_argument("--static-domain", default="https://api.zexnoz.dev", help="Static domain for Cloudflare Named Tunnel")
    p_serve.add_argument("--cf-token", default="", help="Cloudflare Tunnel token (or set CF_TUNNEL_TOKEN env)")
    p_serve.add_argument("--hf-token", default="", help="Hugging Face API token (or set HF_TOKEN env)")
    p_serve.add_argument("--dataset", default="", help="Attach existing Kaggle dataset (e.g. user/dataset-slug)")

    # benchmark command
    p_bench = subparsers.add_parser("benchmark", help="Deploy and run Sweet-Spot Benchmark on Kaggle 2xT4")
    p_bench.add_argument("--slug", default="ornith15-35b-sweet-spot-benchmark", help="Kaggle kernel slug name")
    p_bench.add_argument("--hf-token", default="", help="Hugging Face API token (or set HF_TOKEN env)")
    p_bench.add_argument("--dataset", default="", help="Attach existing Kaggle dataset (e.g. user/dataset-slug)")

    # status command
    subparsers.add_parser("status", help="Check deployment status")

    # stop command
    subparsers.add_parser("stop", help="Stop running session")

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
