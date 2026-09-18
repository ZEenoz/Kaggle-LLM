# Task 2 Brief: kernel/serve_bonsai.py

**Target File:** `c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\kernel\serve_bonsai.py`
**Reference Pattern:** `c:\Users\nxg22304\Desktop\kaggle\qwen38_35b_a3b\kernel\serve_qwen.py`

## Requirements:
1. Configuration & Injection:
   - Include `CFG = None` at the top (to be replaced by `launch.py` with custom parameters).
   - `DEFAULTS` dictionary:
     - `model_repo`: "prism-ml/Ternary-Bonsai-2-27B-gguf"
     - `model_file`: "Ternary-Bonsai-2-27B-PQ2_0.gguf"
     - `model_url`: "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PQ2_0.gguf"
     - `model_name`: "ternary-bonsai-2-27b"
     - `context_size`: 196608 (default production context)
     - `batch_size`: 1024
     - `ubatch_size`: 512
     - `kv_cache_k`: "q8_0"
     - `kv_cache_v`: "q8_0"
     - `port`: 8080
     - `keepalive_min`: 480
     - `api_key`: "kilo-secret-key"
     - `static_domain`: "https://api.zexnoz.dev"
     - `cf_token`: ""
     - `ntfy_topic`: ""

2. Dual Tesla T4 Setup & GPU Validation:
   - Check GPU using `nvidia-smi`, report count and memory to ntfy / console.

3. Binary Installation:
   - Download PrismML llama.cpp Linux CUDA 12.8 binaries:
     `https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b10687-5d80cff/llama-prism-b10687-5d80cff-bin-linux-cuda-12.8-x64.tar.gz`
     Fallback: `https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b10687-5d80cff/llama-prism-b10687-5d80cff-bin-linux-cuda-12.4-x64.tar.gz`
   - Extract and locate `llama-server`.
   - Download `cloudflared` from official GitHub release.

4. Server Execution:
   - Run `llama-server` with:
     `-m <model_path> -ngl 99 -fa on -ts 1,1 -sm row -c <context_size> -b <batch_size> -ub <ubatch_size> --ctk <kv_cache_k> --ctv <kv_cache_v> --port <port> --api-key <api_key> --alias <model_name>`
   - Poll `/health` endpoint until live.

5. Cloudflare Tunnel Integration:
   - Support Named Tunnel via `cf_token` (routes to `static_domain` e.g. `https://api.zexnoz.dev`).
   - Support Quick Tunnel (free trycloudflare.com ephemeral URL) if no token is provided. Extract URL from log output.

6. Telemetry & Keepalive:
   - Notify `ntfy.sh` on lifecycle phases (`starting`, `gpu_info`, `setup_llama`, `model_download`, `loading_vram`, `live`, `stopping`, `stopped`, `failed`).
   - Run keepalive loop respecting `keepalive_min`. Clean up background processes on SIGINT/SIGTERM.

7. Verification:
   - Must pass `python -m py_compile c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\kernel\serve_bonsai.py`.
