# Task 1 Brief: benchmark_sweet_spot.py

**Target File:** `c:\Users\nxg22304\Desktop\kaggle\ternary_bonsai_2_27b\benchmark_sweet_spot.py`
**Reference Pattern:** `c:\Users\nxg22304\Desktop\kaggle\qwen38_35b_a3b\benchmark_sweet_spot.py`

## Requirements:
1. Environment & GPU:
   - Must detect dual Tesla T4 GPUs using `nvidia-smi`.
   - Log GPU names and total memory.

2. PrismML llama.cpp Binaries:
   - Primary: `https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b10687-5d80cff/llama-prism-b10687-5d80cff-bin-linux-cuda-12.8-x64.tar.gz`
   - Fallback: `https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b10687-5d80cff/llama-prism-b10687-5d80cff-bin-linux-cuda-12.4-x64.tar.gz`
   - Extract to `/kaggle/tmp/llm_server` (or local `/tmp/llm_server`). Locate `llama-server` and ensure executable.

3. Model Download:
   - Repo: `prism-ml/Ternary-Bonsai-2-27B-gguf`
   - File: `Ternary-Bonsai-2-27B-PQ2_0.gguf`
   - URL: `https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PQ2_0.gguf`
   - Support download via `aria2c` / `curl` / `urllib` with progress and resume.

4. Context Sweep Parameters:
   - Default contexts: `(65536, 131072, 196608, 262144)`
   - Default batch size: `1024`
   - Default ubatch size: `512`
   - Default KV cache: `q8_0`
   - Tensor split: `-ts 1,1 -sm row`
   - Flash attention: `-fa on`
   - Offload: `-ngl 99`
   - Sampling defaults: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0`

5. Benchmarking Battery per Context:
   - Start `llama-server`.
   - Poll `/health` until ready (timeout 900s).
   - Start background thread measuring GPU 0 & GPU 1 VRAM (via `nvidia-smi` query) and RAM every 0.5s.
   - Run Warmup prompt.
   - Run Short Generation test (128 max tokens) -> measure Output tok/s.
   - Run Long Context Prefill test (Prompt scaled to ~80% of context window) -> measure Prompt tok/s and TTFT.
   - Terminate server cleanly.

6. Reporting:
   - Export `results.json`, `results.csv`, `sweet_spot.json`, and `summary.md`.

7. Verification:
   - Script must pass `python -m py_compile`.
   - `--help` flag must show all arguments with the specified defaults.
