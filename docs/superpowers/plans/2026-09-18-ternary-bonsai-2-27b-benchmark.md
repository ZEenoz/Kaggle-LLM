# Ternary-Bonsai-2-27B Kaggle 2x Tesla T4 Benchmark & Inference Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete sweet-spot benchmark harness, production API server, CLI launcher, and Jupyter notebook for running Ternary-Bonsai-2-27B (PQ2_0 GGUF) on Kaggle 2x Tesla T4 GPUs across 64k to 262k contexts.

**Architecture:** Custom PrismML llama.cpp Linux CUDA 12.8 binaries with Hadamard activation transform and PQ2_0 support. Tensor split 1:1 across 2x T4 GPUs, Flash Attention, and quantized KV cache (q8_0) for extreme context depth with 0 OOM risk.

**Tech Stack:** Python 3.10+, llama.cpp (PrismML fork), CUDA 12.8 / sm_75, Kaggle API, Cloudflare Tunnel (cloudflared), ntfy.sh telemetry, Jupyter Notebook (.ipynb).

## Global Constraints

- Target root directory: `c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b`
- Model file: `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7.21 GB)
- Hugging Face Model URL: `https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PQ2_0.gguf`
- PrismML Release Binary: `https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b10687-5d80cff/llama-prism-b10687-5d80cff-bin-linux-cuda-12.8-x64.tar.gz` (Fallback: `...-cuda-12.4-x64.tar.gz`)
- Hardware Target: Kaggle 2x Nvidia Tesla T4 GPUs (32GB VRAM total, 30.2GB usable)
- Context sweep list: `(65536, 131072, 196608, 262144)`
- Runtime flags: `-ts 1,1 -sm row -ngl 99 -fa on -b 1024 -ub 512 --ctk q8_0 --ctv q8_0`
- Sampling defaults: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0`

---

### Task 1: Scaffolding and `benchmark_sweet_spot.py`

**Files:**
- Create: `c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\benchmark_sweet_spot.py`

**Interfaces:**
- Consumes: Kaggle Linux environment, NVIDIA CUDA drivers, Hugging Face CDN, GitHub PrismML releases
- Produces: CLI script with `--context-sizes`, `--batch-size`, `--ubatch-size`, `--kv-cache`; outputs `results.json`, `results.csv`, `sweet_spot.json`, `summary.md`

- [ ] **Step 1: Write `benchmark_sweet_spot.py`**
  - Implement environment detection (`nvidia-smi` parser for 2 GPUs).
  - Implement `install_prism_llamacpp()` downloading and unpacking `llama-prism-b10687-5d80cff-bin-linux-cuda-12.8-x64.tar.gz` with fallback.
  - Implement `download_model()` downloading `Ternary-Bonsai-2-27B-PQ2_0.gguf` with progress logging and resume support.
  - Implement `run_server_context()` with health check polling (`http://127.0.0.1:8080/health`).
  - Implement memory monitoring thread sampling GPU VRAM (via NVML/nvidia-smi) and system RAM every 0.5s.
  - Implement benchmark test battery:
    - Warmup request.
    - Short generation decode throughput test (128 output tokens, measures tokens/s).
    - Long context prefill test (generates prompt matching target context, measures prompt tok/s and TTFT).
  - Implement report generation saving `results.json`, `results.csv`, `sweet_spot.json`, and markdown `summary.md`.

- [ ] **Step 2: Verify `benchmark_sweet_spot.py` syntax and CLI argument parsing**
  - Run: `python -m py_compile c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\benchmark_sweet_spot.py`
  - Run: `python c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\benchmark_sweet_spot.py --help`
  - Expected: Zero syntax errors, returns full help text with default context sizes `(65536, 131072, 196608, 262144)`.

---

### Task 2: Cloud Inference Server `kernel/serve_bonsai.py`

**Files:**
- Create: `c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\kernel\serve_bonsai.py`

**Interfaces:**
- Consumes: PrismML `llama-server`, Cloudflare `cloudflared` tunnel client, `ntfy.sh` webhook
- Produces: Persistent OpenAI-compatible server on Kaggle with public tunnel endpoint (`/v1/chat/completions`)

- [ ] **Step 1: Write `kernel/serve_bonsai.py`**
  - Set up configurable `CFG` dict supporting injection by `launch.py`.
  - Implement `notify()` broadcasting structured JSON heartbeats to `ntfy.sh/{topic}`.
  - Implement `check_gpu()` validating dual T4 setup.
  - Implement `install_prism_llamacpp()` and `install_cloudflared()`.
  - Implement model download and integrity check.
  - Launch `llama-server` on port 8080 with `-ts 1,1 -sm row -ngl 99 -fa on -c 196608 -b 1024 -ub 512 --ctk q8_0 --ctv q8_0 --api-key <KEY>`.
  - Launch `cloudflared` (Named Tunnel if token present, or ephemeral Quick Tunnel parsing `https://*.trycloudflare.com`).
  - Monitor health in loop and notify live endpoint URL.

- [ ] **Step 2: Verify `kernel/serve_bonsai.py` syntax**
  - Run: `python -m py_compile c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\kernel\serve_bonsai.py`
  - Expected: Clean compilation with 0 errors.

---

### Task 3: Local CLI Orchestrator `launch.py`

**Files:**
- Create: `c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\launch.py`

**Interfaces:**
- Consumes: Local Kaggle API credentials (`~/.kaggle/access_token` or `kaggle.json`), `kernel/serve_bonsai.py`, `benchmark_sweet_spot.py`
- Produces: CLI commands: `python launch.py benchmark`, `serve`, `status`, `stop`

- [ ] **Step 1: Write `launch.py`**
  - Implement `get_kaggle_api()` with dual-credential support.
  - Implement `cmd_benchmark()`:
    - Generates temporary kernel package with `benchmark_sweet_spot.py`.
    - Generates `kernel-metadata.json` (`machine_shape: "NvidiaTeslaT4"`, `enable_gpu: true`, `enable_internet: true`).
    - Pushes kernel via `api.kernels_push()`.
    - Saves state to `~/.kaggle-bonsai2-lab.json`.
  - Implement `cmd_serve()`:
    - Generates temporary kernel package with injected `CFG` config into `serve_bonsai.py`.
    - Pushes kernel to Kaggle and starts `poll_events(topic)` streaming live status and tunnel URL.
  - Implement `cmd_status()` and `cmd_stop()` via `api.kernels_status()` and `api.kernels_cancel()`.

- [ ] **Step 2: Verify `launch.py` CLI interface**
  - Run: `python -m py_compile c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\launch.py`
  - Run: `python c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\launch.py --help`
  - Expected: Clean compilation, subcommands `benchmark`, `serve`, `status`, `stop` listed properly.

---

### Task 4: Interactive Kaggle Notebook `notebook/ternary_bonsai_2_27b_t4.ipynb`

**Files:**
- Create: `c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\notebook\ternary_bonsai_2_27b_t4.ipynb`

**Interfaces:**
- Consumes: Kaggle Web UI environment (Jupyter / nteract)
- Produces: Complete 4-cell notebook for self-contained execution directly inside Kaggle browser UI

- [ ] **Step 1: Write Jupyter Notebook**
  - Cell 1: Environment & GPU verification (`nvidia-smi`, Python version, CUDA check).
  - Cell 2: Setup PrismML `llama.cpp` CUDA 12.8 binaries and download `Ternary-Bonsai-2-27B-PQ2_0.gguf`.
  - Cell 3: Interactive Sweet-Spot Benchmark Runner (executing sweep across 64k, 128k, 192k, 262k and displaying summary tables & charts inline).
  - Cell 4: Production 24/7 Server with Cloudflare Quick Tunnel and interactive cURL / OpenAI Python SDK query demo.

- [ ] **Step 2: Validate Notebook JSON structure**
  - Run validation script in Python checking `nbformat.validate` or valid JSON schema.
  - Expected: Valid JSON, nbformat 4.x.

---

### Task 5: Technical Documentation & VRAM Analysis `README.md`

**Files:**
- Create: `c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\README.md`

**Interfaces:**
- Consumes: Sizing calculations, command syntax, architecture analysis
- Produces: Markdown guide for users

- [ ] **Step 1: Write `README.md`**
  - Document model background: Ternary g128, blockwise Hadamard transform, 98.2% FP16 intelligence.
  - VRAM and architecture breakdown for 2x Tesla T4 (table showing 64k, 128k, 192k, 262k).
  - Quickstart CLI instructions (`launch.py benchmark`, `launch.py serve`).
  - Kaggle Notebook usage instructions.
  - Optimal inference parameters and OpenAI client configuration.

---

### Task 6: End-to-End Verification

**Files:**
- Review all files in `c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\`

- [ ] **Step 1: Syntax compilation of all python files**
- [ ] **Step 2: Asset URL health verification (HTTP 200/302 for model and PrismML binary URLs)**
- [ ] **Step 3: CLI execution dry-run of launcher**
