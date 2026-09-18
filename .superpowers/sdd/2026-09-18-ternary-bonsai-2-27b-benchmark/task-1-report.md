# Task 1 Report: Implement `benchmark_sweet_spot.py` for Ternary-Bonsai-2-27B

- **Task Status:** Complete
- **Target File:** `c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\benchmark_sweet_spot.py`
- **Date:** 2026-09-18

---

## 1. Summary of Work

Implemented the complete sweet-spot benchmarking harness for **Ternary-Bonsai-2-27B (PQ2_0)** targeting Kaggle 2x Nvidia Tesla T4 GPUs (32 GB total VRAM).

The harness sweeps context windows from 64k up to 262k tokens with quantized `q8_0` KV cache, measuring VRAM consumption across both GPUs, prefill throughput (prompt tok/s), time-to-first-token (TTFT), and decode generation throughput (output tok/s).

---

## 2. Specification & Implementation Details

- **Model Configuration:**
  - Repo: `prism-ml/Ternary-Bonsai-2-27B-gguf`
  - File: `Ternary-Bonsai-2-27B-PQ2_0.gguf` (~7.21 GB)
  - Download URL: `https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PQ2_0.gguf`
  - Download Strategy: Supports mounted dataset `/kaggle/input`, local cache check, and resume downloading via `aria2c` / `curl` / `wget` / `urllib` / `hf_hub_download`.

- **PrismML llama.cpp Binaries:**
  - Primary URL: `https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b10687-5d80cff/llama-prism-b10687-5d80cff-bin-linux-cuda-12.8-x64.tar.gz`
  - Fallback URL: `https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b10687-5d80cff/llama-prism-b10687-5d80cff-bin-linux-cuda-12.4-x64.tar.gz`
  - Dynamic extraction, permission enforcement (`chmod 755`), and symlinking to workdir `bin/llama-server`.

- **Execution & Server Flags:**
  - Offload: `-ngl 99`
  - Split Mode: `-sm row` (essential for blockwise Hadamard activation & PQ2_0 distribution across dual GPUs)
  - Tensor Split: `-ts 1,1` (even 50/50 split across GPU 0 & GPU 1)
  - Flash Attention: `-fa on`
  - KV Cache Quantization: `--cache-type-k q8_0 --cache-type-v q8_0`
  - Batch / UBatch: `-b 1024 -ub 512`
  - Context Sizes: `(65536, 131072, 196608, 262144)`
  - Sampling Defaults: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0`

- **Dual-GPU & System Telemetry:**
  - `detect_gpus()` logs device names and individual VRAM totals using `nvidia-smi`.
  - `MemorySampler` background thread polls every 0.5s recording per-GPU VRAM (GPU 0 & GPU 1), total GPU VRAM peak, and system RAM from `/proc/meminfo`.

- **Benchmark Battery per Context:**
  1. Health check polling (`/health` and `/v1/models` up to 900s).
  2. Warmup query (8 max tokens).
  3. Short generation test (128 max tokens) -> measures baseline decode throughput (gen tok/s).
  4. Long context prefill test (scaled to ~80% context window) -> measures prefill speed (prompt tok/s) and TTFT.
  5. Clean server shutdown.

- **Reporting:**
  - `results.json`: Full machine-readable dataset.
  - `results.csv`: Long-format tabular metrics.
  - `sweet_spot.json`: Recommended context configuration, rationale, and ranked candidate scores.
  - `summary.md`: Human-readable markdown report with context scaling tables.

---

## 3. Verification Results

### 3.1 Python Syntax Compilation (`py_compile`)
- **Command:**
  ```powershell
  python -m py_compile c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\benchmark_sweet_spot.py
  ```
- **Result:** Exit code `0` (clean compilation, zero errors or warnings).

### 3.2 CLI `--help` Verification
- **Command:**
  ```powershell
  python c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\benchmark_sweet_spot.py --help
  ```
- **Result:** Exit code `0`. Full help output matches all specified defaults:
  - Context sizes: `65536,131072,196608,262144`
  - Batch size: `1024`, ubatch size: `512`
  - KV cache: `q8_0`
  - Offload layers: `99`
  - Sampling parameters: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0`
  - Model: `prism-ml/Ternary-Bonsai-2-27B-gguf` / `Ternary-Bonsai-2-27B-PQ2_0.gguf`

---

## 4. Conclusion
Task 1 is fully complete and verified. The codebase is ready for Task 2.
