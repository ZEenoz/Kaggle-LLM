# Benchmark Script (v2) 2x Tesla T4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `ornith15_35b_a3b/benchmark_q4_v2.py` for optimal, safe, and resilient execution on Kaggle 2x Tesla T4 GPUs within 1.5–2 hours.

**Architecture:** A streamlined multi-phase benchmark with pre-launch OOM filtering, step-based context probing, targeted batching/concurrency matrix, dual-GPU memory telemetry, and atomic state checkpointing.

**Tech Stack:** Python 3.10+, llama.cpp (`llama-server`, `llama-perplexity`), PyTorch / CUDA 12.x / sm_75, SSE streaming HTTP client.

## Global Constraints
- Target Hardware: 2x Tesla T4 (sm_75, 16GB GDDR6 each, total ~30.2GB usable VRAM).
- Model: `ornith-ai/Ornith-1.5-35B-A3B-GGUF` (`Ornith-1.5-35B-Q4_K_M.gguf`, ~21.7GB).
- Strict OOM Prevention: Combinations with `kv=q8_0` and `context > 32768` must be skipped.
- Context Fit Safeguard: Pass `-fit off` (when supported) to prevent silent context truncation.
- Binary Compatibility: Support both `llama-perplexity` and `llama-perplex`.
- No external heavy dependencies: Standard library + basic Python environment.

---

### Task 1: Server Management & Toolchain Resilience

**Files:**
- Modify: `ornith15_35b_a3b/benchmark_q4_v2.py`

**Interfaces:**
- `build_llama_cmd(cfg: RunCfg, server_bin: Path, model_path: Path, help_text: str, model_alias: str, api_key: str) -> list[str]`
- `install_llama_cpp(workdir: Path) -> Path`
- `run_perplexity(args: argparse.Namespace, bin_dir: Path, model_path: Path, corpus: str, kv: str) -> dict`

- [ ] **Step 1: Update `build_llama_cmd` to support `-fit off` and safe T4 flags**
  Add detection for `-fit` / `--fit` in `help_text` and append `["-fit", "off"]`. Ensure `-sm layer`, `--tensor-split 1,1`, and `-fa on` are correctly assembled.

- [ ] **Step 2: Update `install_llama_cpp` and `run_perplexity` for toolchain binary compatibility**
  Detect both `llama-perplexity` and `llama-perplex` in the extracted directory and link `bin/llama-perplexity`. Ensure `run_perplexity` invokes whichever binary exists.

- [ ] **Step 3: Add HTTP headers and fallback for GitHub releases**
  Ensure urllib requests to GitHub API pass a standard `User-Agent` and handle HTTP 403 Forbidden by falling back directly to `LLAMA_FALLBACK_URL`.

- [ ] **Step 4: Verify syntax**
  Run: `python -m py_compile ornith15_35b_a3b/benchmark_q4_v2.py`
  Expected: Returncode 0.

---

### Task 2: Matrix Streamlining & Safe OOM Pruning

**Files:**
- Modify: `ornith15_35b_a3b/benchmark_q4_v2.py`

**Interfaces:**
- `phase_b(args: argparse.Namespace, model_path: Path, bin_dir: Path, samples_out: list[dict], jsonl_path: Path, checkpoint: dict) -> list[dict]`
- `phase_d(args: argparse.Namespace, model_path: Path, bin_dir: Path, samples_out: list[dict], jsonl_path: Path, checkpoint: dict) -> dict`
- `phase_c(args: argparse.Namespace, model_path: Path, bin_dir: Path, samples_out: list[dict], jsonl_path: Path, checkpoint: dict) -> list[dict]`

- [ ] **Step 1: Implement pre-launch OOM filtering and streamline Phase B**
  Set `DEFAULT_B_CTX_SIZES = (32768, 131072)` and prune `(q8_0, 131072)`. Set `FA_MODES` and `REUSE_MODES` defaults to targeted runs (`q4_0` at 32k/128k, `q8_0` at 32k, and `fa=off` at 32k). Total runs = 4 + PPL.

- [ ] **Step 2: Replace binary bisection with 4-step progressive probe in Phase D**
  Define `DEFAULT_D_STEPS = (32768, 65536, 98304, 131072)`. Sequentially probe each context with `q4_0`. If a step fails, stop probing higher contexts. Sweep `-ngl 99` vs `-ngl 80` at the max good context.

- [ ] **Step 3: Streamline Phase C batching & concurrency cells**
  Define 5 production-grade cells:
  - `(np=1, b=1024, ub=512)`
  - `(np=4, b=1024, ub=512)`
  - `(np=4, b=2048, ub=512)`
  - `(np=8, b=1024, ub=512)`
  - `(np=8, b=512,  ubatch=512)`
  Workloads: L1 (warmup/single), L2 (c=1, 4, 8), L3 (rates=1.0, 2.0, 3.0 req/s, n=16). Disable vLLM by default.

- [ ] **Step 4: Verify syntax**
  Run: `python -m py_compile ornith15_35b_a3b/benchmark_q4_v2.py`
  Expected: Returncode 0.

---

### Task 3: State Checkpointing & Resume Support

**Files:**
- Modify: `ornith15_35b_a3b/benchmark_q4_v2.py`

**Interfaces:**
- `load_checkpoint(checkpoint_file: Path) -> dict`
- `save_checkpoint(checkpoint_file: Path, results: dict, samples: list[dict]) -> None`

- [ ] **Step 1: Implement `load_checkpoint` and `save_checkpoint`**
  Write an atomic saver (`.tmp` write and rename) to preserve intermediate results after each run/cell into `checkpoint_v2.json` and `results_v2.json`.

- [ ] **Step 2: Wire checkpointing into Phase B, Phase D, and Phase C runners**
  Check whether `run_id` already exists and was marked `fit` or completed in `checkpoint`. If present, skip the server boot and reuse the recorded metrics.

- [ ] **Step 3: Verify syntax**
  Run: `python -m py_compile ornith15_35b_a3b/benchmark_q4_v2.py`
  Expected: Returncode 0.

---

### Task 4: CLI Arguments, Quick Mode & End-to-End Verification

**Files:**
- Modify: `ornith15_35b_a3b/benchmark_q4_v2.py`

**Interfaces:**
- `apply_quick(args: argparse.Namespace) -> None`
- `parse_args(argv: list[str] | None) -> argparse.Namespace`
- `main(argv: list[str] | None) -> None`

- [ ] **Step 1: Update CLI arguments and `--quick` smoke profile**
  Add `--resume` flag (defaulting to True if checkpoint exists). Update `--quick` to run minimal smoke passes (1 cell per phase, ~15-20 min total).

- [ ] **Step 2: Update docstring and output summary report**
  Ensure `summary_v2.md` and `sweet_spot_v2.json` format the recommended production parameters ready for copy-pasting into `serve_ornith.py` / `launch.py`.

- [ ] **Step 3: End-to-End CLI smoke test**
  Run: `python ornith15_35b_a3b/benchmark_q4_v2.py --help`
  Expected: Clean display of arguments including `--quick`, `--resume`, `--skip`, `--include-vllm`.
