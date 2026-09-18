# Design Specification: Ornith-1.5-35B-A3B Benchmark Script (v2) for 2x Tesla T4 on Kaggle

- **Date**: 2026-09-08
- **Target Hardware**: Kaggle 2x Nvidia Tesla T4 (16GB GDDR6 each, total ~30.2GB usable VRAM, compute capability sm_75)
- **Target Model**: `ornith-ai/Ornith-1.5-35B-A3B-GGUF` (`Ornith-1.5-35B-Q4_K_M.gguf`, ~21.7GB)
- **Target Script**: `ornith15_35b_a3b/benchmark_q4_v2.py`

---

## 1. Background & Objectives

The existing `benchmark_q4_v2.py` script was designed to run an exhaustive parameter matrix across Phase B (KV cache permutations), Phase D (Context scaling limits), and Phase C (Batching & Concurrency). However:
1. **Excessive Runtime**: With over 41 server reboots and workloads, it takes 7–9 hours, risking session expiration on Kaggle (9–12 hr maximum session limit, network drops, or GPU timeouts).
2. **Guaranteed OOMs**: Permutations such as `kv=q8_0` at 128k context exceed the ~8.5GB free VRAM available on 2x T4 (after loading the 21.7GB model weights split 1:1), causing unrecoverable CUDA out-of-memory crashes.
3. **Missing Flags & Resilience**: Lacks the `-fit off` safeguard (which prevents llama.cpp from silently dropping context size), lacks binary compatibility between `llama-perplex` and `llama-perplexity`, and lacks checkpoint/resume capabilities to recover from intermittent Kaggle interruptions.

The objective is to refine and optimize `benchmark_q4_v2.py` into a robust, high-signal, production-oriented benchmark that executes within **1.5 to 2.0 hours** while identifying the exact sweet spot (Throughput, Latency, Concurrency, VRAM safety margin) for deploying Ornith-1.5-35B on Kaggle 2x T4.

---

## 2. Architecture & Dual-GPU Server Management

### 2.1 Dual-GPU Offloading Configuration
- **Binary**: Prebuilt `llama-server` (CUDA 12.8, sm_75 architecture).
- **Tensor Splitting**: `--tensor-split 1,1` divides the weights equally (~10.85GB per GPU), leaving ~4.25GB free per GPU for KV cache, CUDA buffers, and activations.
- **Offload Mode**: `-sm layer`, `-ngl 99` (full GPU offload).
- **Flash Attention**: Mandatory `-fa on` (or `--flash-attn on` depending on build help text) to eliminate quadratic activation growth at large context lengths.
- **Context Fit Override**: Automatically pass `-fit off` if supported by the binary to prevent silent context downsizing.
- **Environment**:
  - `GGML_CUDA_NO_VMM=1` to ensure stable memory virtualization on Kaggle's Linux kernel.
  - `CUDA_VISIBLE_DEVICES=0,1` explicit device mapping.

### 2.2 OOM Pruning Gate
- Combinations with `kv=q8_0` at `context > 32768` are hard-pruned before server initialization and recorded as skipped in output JSON.
- An idle VRAM sanity check: If idle VRAM exceeds 14.5GB on either GPU immediately after loading, long-context workloads for that configuration are aborted cleanly to prevent kernel panics.

### 2.3 Binary & Network Fallbacks
- Support both `llama-perplexity` (newer builds) and `llama-perplex` (older builds).
- Download logic includes a direct fallback URL and GitHub API `User-Agent` headers to bypass anonymous rate limits (HTTP 403) on Kaggle shared IPs.
- Seamless detection of pre-mounted Kaggle datasets at `/kaggle/input`.

---

## 3. Benchmark Phases & Targeted Matrices

### 3.1 Phase B: KV Cache Matrix & Quality Verification (~4 runs + PPL)
- **Configurations**:
  1. `ctx=32768, kv=q4_0, fa=on, reuse=off` (Baseline 32k)
  2. `ctx=131072, kv=q4_0, fa=on, reuse=off` (Target production 128k)
  3. `ctx=32768, kv=q8_0, fa=on, reuse=off` (Quality comparison baseline)
  4. `ctx=32768, kv=q4_0, fa=off, reuse=off` (Direct Flash Attention impact test)
- **Workload L1**:
  - 3x short completions (128 tokens) -> Generates median gen tokens/s and ITL p95.
  - 1x long completion with 8,192 prompt tokens -> Measures TTFT and prompt processing tokens/s.
- **Perplexity (Quality Regression Check)**:
  - Runs `llama-perplexity` against 40 GSM8K test questions at 4,096 context for `q4_0` vs `q8_0`.
  - Verifies that `q4_0` KV quantization causes zero significant degradation in reasoning precision.

### 3.2 Phase D: Context Scaling Step Probe (~5 runs)
- Replaces slow binary bisection with a 4-step progressive boundary probe:
  - Step 1: `ctx=32,768`
  - Step 2: `ctx=65,536`
  - Step 3: `ctx=98,304`
  - Step 4: `ctx=131,072`
- **Probe Workload**:
  - Prompt filled to `ctx - 512` tokens.
  - Model generates 64 tokens.
  - Measures TTFT, generation TPS, and peak VRAM across both GPUs.
- **Offload Boundary**:
  - Test `-ngl 99` vs `-ngl 80` at 128k context to quantify the impact of partial CPU offloading vs 100% GPU offloading.

### 3.3 Phase C: Batching & Concurrency (~5 cells)
- Fix context at 32,768 (or 131,072 with conservative batching) to evaluate server throughput under realistic traffic:
  1. `np=1, batch=1024, ubatch=512` (Single-user baseline)
  2. `np=4, batch=1024, ubatch=512` (Balanced production)
  3. `np=4, batch=2048, ubatch=512` (High prompt throughput)
  4. `np=8, batch=1024, ubatch=512` (High concurrency)
  5. `np=8, batch=512,  ubatch=512` (VRAM-conservative high concurrency)
- **Workloads**:
  - `L1`: Warmup and single-stream baseline.
  - `L2`: Closed-loop concurrent requests ($c \in \{1, 4, 8\}$).
  - `L3`: Open-loop Poisson arrival traffic ($\lambda \in \{1.0, 2.0, 3.0\}$ req/s with $n=16$ requests each).
- **Engine**: llama.cpp as default. vLLM disabled by default (opt-in via `--include-vllm`).

---

## 4. Checkpointing, Resilience & Metrics Collection

### 4.1 Checkpointing & Resume
- Saves intermediate state after every single run to `results_v2.json` and `checkpoint_v2.json`.
- When `--resume` is active (or enabled by default when existing checkpoint is found), already completed run IDs are skipped without re-launching the server.

### 4.2 Instrumentation
- **GpuSampler**:
  - Samples GPU0 VRAM, GPU1 VRAM, total VRAM, utilization (%), temperature (°C), power draw (W), and system RAM (GB) every 0.5s.
  - Outputs time series data to `samples_v2.csv`.
- **Request Client**:
  - Streams SSE chat completions (`/v1/chat/completions`) with chunk timestamping.
  - Records submit time, first token time, TTFT, generation TPS, ITL p95, prompt tokens, and completion tokens to `requests_v2.jsonl`.

### 4.3 Sweet-Spot Synthesis
- **SLO Criteria**:
  - `TTFT_p95 <= 5.0s`
  - `ITL_p95 <= 0.15s` (equivalent to >= 6.67 tok/s stream responsiveness)
  - Zero request failures/timeouts.
- **Selection Formula**:
  - Filter all cells passing SLOs.
  - Rank by maximum sustainable Poisson arrival rate ($\lambda$).
  - Break ties with higher aggregate system throughput (`system_tok_s`) and lower peak VRAM.
- **Output Files**:
  - `results_v2.json`: Complete raw metrics.
  - `results_v2.csv`: Tidy long-format table.
  - `samples_v2.csv`: GPU telemetry time-series.
  - `requests_v2.jsonl`: Individual request records.
  - `sweet_spot_v2.json`: Recommended parameters and ranking.
  - `summary_v2.md`: Human-readable markdown report.

---

## 5. Verification & Acceptance Criteria
1. Script compiles and passes syntax and linting checks (`python -m py_compile benchmark_q4_v2.py`).
2. Script accepts all CLI arguments (`--help`, `--quick`, `--resume`, `--skip`).
3. Under `--quick` mode, completes a smoke pass in ~15-20 minutes without crashing or hanging.
4. Correctly prunes OOM combinations (`q8_0` at 128k) and detects `-fit off` and `llama-perplexity`.
