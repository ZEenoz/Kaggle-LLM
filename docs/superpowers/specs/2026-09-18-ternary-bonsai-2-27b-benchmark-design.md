# Design Specification: Ternary-Bonsai-2-27B Benchmark & Inference on Kaggle 2x Tesla T4

- **Date**: 2026-09-18
- **Target Hardware**: Kaggle 2x Nvidia Tesla T4 GPUs (32 GB total VRAM / ~30.2 GB usable)
- **Model**: `prism-ml/Ternary-Bonsai-2-27B-gguf` -> `Ternary-Bonsai-2-27B-PQ2_0.gguf` (~7.21 GB)
- **Architecture**: Hybrid Attention (~75% Linear / ~25% Full), SwiGLU, RMSNorm, 27.36B parameters
- **Weight Format**: Ternary g128 in 2-bit slots (PQ2_0, true 2.13 bpw) with blockwise Hadamard transform

---

## 1. System Architecture & Objectives

1. **Sweet-Spot Context Benchmark**:
   - Sweep context sizes: `65,536` (64k), `131,072` (128k), `196,608` (192k), and `262,144` (262k Native).
   - Evaluate Prefill throughput (Prompt tok/s) and Time-To-First-Token (TTFT) across long contexts.
   - Evaluate Decode throughput (Gen tok/s) in Thinking Mode.
   - Monitor real-time memory telemetry (GPU 0 VRAM, GPU 1 VRAM, System RAM).
   - Export structured outputs: `results.json`, `results.csv`, `sweet_spot.json`, and `summary.md`.

2. **Custom Runtime Engine**:
   - Stock `llama.cpp` does not support Hadamard activation rotation and rejects PQ2_0.
   - Automatically fetch official prebuilt Linux CUDA 12.8 binaries from [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10687-5d80cff) (`llama-prism-b10687-5d80cff-bin-linux-cuda-12.8-x64.tar.gz`) with CUDA 12.4 fallback.

3. **Production Inference Server**:
   - High-performance OpenAI-compatible server on 2x T4 with Tensor Parallelism (`-ts 1,1 -sm row`).
   - Dual Cloudflare Tunnel support (Named Tunnel e.g. `api.zexnoz.dev` with token, or free ephemeral Quick Tunnel).
   - Real-time telemetry reporting via `ntfy.sh`.

---

## 2. Hardware & VRAM Budget Analysis (2x Tesla T4)

| Component | 64k Context | 128k Context | 192k Context | 262k Native Context |
| :--- | :---: | :---: | :---: | :---: |
| **Model Weights (PQ2_0)** | ~7.21 GB (~3.6 GB/GPU) | ~7.21 GB (~3.6 GB/GPU) | ~7.21 GB (~3.6 GB/GPU) | ~7.21 GB (~3.6 GB/GPU) |
| **KV Cache (`q8_0`, -fa on)** | ~2.1 GB | ~4.2 GB | ~6.3 GB | ~8.6 GB |
| **Compute & Activation Buffer** | ~1.0 GB | ~1.2 GB | ~1.5 GB | ~2.0 GB |
| **Total VRAM Allocated** | **~10.3 GB** | **~12.6 GB** | **~15.0 GB** | **~17.8 GB** |
| **Remaining Free VRAM** | **~19.9 GB (66%)** | **~17.6 GB (58%)** | **~15.2 GB (50%)** | **~12.4 GB (41%)** |

*Note*: Because Qwen3.8 hybrid attention uses ~75% linear attention (constant/O(1) recurrent states) and only ~25% full attention, the KV cache footprint is ~75% smaller than conventional standard multi-head attention models of the same parameter scale. Even at 262k native context with `q8_0`, the total memory footprint comfortably fits within the 30.2 GB usable VRAM of 2x T4 GPUs without OOM.

---

## 3. Optimal Inference Parameters for Tesla T4

- **Tensor Parallelism**: `-ts 1,1 -sm row` (Symmetrical split across both T4 GPUs)
- **Flash Attention**: `-fa on` (Mandatory to avoid quadratic memory spikes during prefill)
- **Batching**:
  - `batch_size`: 1024 (Limits peak activation memory on T4's 16GB VRAM per chip)
  - `ubatch_size`: 512 (Matches Turing Tensor Core granularity for low-latency prefill)
- **KV Cache Quantization**: `--ctk q8_0 --ctv q8_0` (Optional fallback `--ctk q4_0 --ctv q4_0` if context is pushed further)
- **Sampling (Thinking / Reasoning Mode)**:
  - `temperature`: 1.0
  - `top_p`: 0.95
  - `top_k`: 20
  - `min_p`: 0.0
  - `reasoning_effort`: `high` / `xhigh`

---

## 4. File Structure & Components (`ternary_bonsai_2_27b/`)

```text
ternary_bonsai_2_27b/
├── benchmark_sweet_spot.py          # Benchmark harness sweeping 64k-262k context
├── launch.py                        # CLI controller for pushing & managing Kaggle kernels
├── kernel/
│   └── serve_bonsai.py              # Cloud script for llama-server + Cloudflare tunnel
├── notebook/
│   └── ternary_bonsai_2_27b_t4.ipynb # Complete Jupyter Notebook for Kaggle Web UI
└── README.md                        # Technical specs, run instructions, and benchmark documentation
```

### Component Details

1. **`benchmark_sweet_spot.py`**:
   - Detects 2x Tesla T4 environment (`nvidia-smi`).
   - Downloads PrismML `llama.cpp` CUDA 12.8 binaries and unpacks to `/kaggle/tmp/llm_server/bin`.
   - Downloads `Ternary-Bonsai-2-27B-PQ2_0.gguf` (~7.21 GB) with resume/aria2c/hf support.
   - For each context (`65536, 131072, 196608, 262144`):
     - Spawns `llama-server` with `-c <CTX> -b 1024 -ub 512 -ts 1,1 -fa on --ctk q8_0 --ctv q8_0`.
     - Waits for health endpoint `http://127.0.0.1:8080/health`.
     - Tests Short Generation (128 output tokens, measures decode tok/s).
     - Tests Long Context Prefill (measures prompt tok/s and TTFT).
     - Samples GPU VRAM and process RAM continuously.
     - Gracefully terminates server and records metrics.
   - Produces `results.json`, `results.csv`, `sweet_spot.json`, and `summary.md`.

2. **`launch.py`**:
   - Subcommands: `benchmark`, `serve`, `status`, `stop`.
   - Injects kernel metadata (`NvidiaTeslaT4`, `enable_internet=true`, `kernel_type=script`).
   - Submits directly to Kaggle via `KaggleApi` supporting token and key.
   - Listens to real-time events via `ntfy.sh` stream.

3. **`kernel/serve_bonsai.py`**:
   - Production server script designed to run headless on Kaggle.
   - Handles `cloudflared` tunnel setup (Named Tunnel `api.zexnoz.dev` or free Quick Tunnel).
   - Serves OpenAI-compatible `/v1/chat/completions` API endpoint with auth key.

4. **`notebook/ternary_bonsai_2_27b_t4.ipynb`**:
   - Cell 1: Environment & GPU verification.
   - Cell 2: Automated PrismML binary setup & model fetch.
   - Cell 3: Sweet-Spot Benchmark Runner.
   - Cell 4: Live 24/7 Server launcher with Cloudflare Tunnel & cURL testing.

---

## 5. Verification Plan

1. **Static Validation**:
   - Syntax validation via `python -m py_compile` on all generated Python scripts.
   - Jupyter Notebook JSON schema validation.
2. **CLI Parameter Verification**:
   - Verify `--help` and argument parsers for `launch.py` and `benchmark_sweet_spot.py`.
3. **URL & Binary Verification**:
   - Verify HTTP 200/302 for model download URL and PrismML release asset URLs.
