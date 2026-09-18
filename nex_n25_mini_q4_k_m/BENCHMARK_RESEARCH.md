# Benchmark Research & Technical Specification: Nex-N2.5-mini (35B MoE) on Kaggle 2x Tesla T4

## 1. Official Architecture & Model Specifications

### Primary Sources & Repositories
* **Hugging Face Model Card:** [nex-agi/Nex-N2.5-mini](https://huggingface.co/nex-agi/Nex-N2.5-mini)
* **GitHub Repository:** [nex-agi/Nex-N2.5](https://github.com/nex-agi/Nex-N2.5)
* **OpenRouter Listing:** [nex-agi/nex-n2.5-mini](https://openrouter.ai/nex-agi/nex-n2.5-mini)
* **GGUF Repository:** [abenzerps/Nex-N2.5-mini-GGUF](https://huggingface.co/abenzerps/Nex-N2.5-mini-GGUF)
* **Kaggle Dataset (User):** [zeenoz/nex-n2-5-mini-q4-k-m](https://www.kaggle.com/datasets/zeenoz/nex-n2-5-mini-q4-k-m)
* **License:** Apache-2.0
* **Release Date:** September 8, 2026

### Architectural Parameters
* **Base Architecture:** `Qwen3_5MoeForConditionalGeneration` (Qwen3.5 MoE base, 35B-A3B class).
* **Total Parameters:** **35.1 Billion (35.1B)**
* **Active Parameters:** **~3.0 Billion (3.0B)** per token during inference.
* **MoE Design:**
  * Total Routed Experts: 256
  * Active Routed Experts per Token: 8 (`top_k = 8`)
  * Shared Experts: 1 shared expert active for all tokens
  * Layer Composition: 40 transformer layers (`num_hidden_layers = 40`), featuring alternating hybrid linear attention and dense full-attention blocks.
* **Attention Configuration:**
  * Grouped-Query Attention (GQA): `num_key_value_heads = 2`
  * Head Dimension: `head_dim = 256`
* **Native Context Length:** **262,144 tokens (256K)** (`max_position_embeddings = 262144`).

### Chat Template, Reasoning, and Tool Calling
* **Chat Template Base:** ChatML format with standard role markers:
  ```text
  <|im_start|>system
  {system_prompt}<|im_end|>
  <|im_start|>user
  {user_message}<|im_end|>
  <|im_start|>assistant
  <think>
  {reasoning_trace}
  </think>
  {final_response}<|im_end|>
  ```
* **Reasoning Modes (`reasoning_effort`):**
  1. `"none"`: Disables reasoning trace. Direct answer without `<think>` tags. Ideal for high-throughput simple tasks.
  2. `"medium"` (Default): Adaptive thinking. Model dynamically decides reasoning depth based on problem complexity.
  3. `"high"`: Extended chain-of-thought. Detailed step-by-step verification trace enclosed inside `<think>...</think>`.
* **Inference Server Parsers (vLLM / SGLang / llama-server):**
  * Reasoning Parser: `--reasoning-parser qwen3`
  * Tool-Call Parser: `--tool-call-parser qwen3_coder`
* **Tool-Calling Protocol:**
  * Formatted with XML tags `<tool_call>` wrapping JSON payloads:
    ```xml
    <tool_call>
    {"name": "execute_bash", "arguments": {"command": "git status"}}
    </tool_call>
    ```
  * Tool execution feedback returned via `<|im_start|>tool\n{"output": "..."}<|im_end|>`.

---

## 2. GGUF Quantization Details (`abenzerps/Nex-N2.5-mini-GGUF`)

### Quantization Matrix & VRAM Footprint
| Quantization Format | File Name | Size (GB) | 2x T4 Feasibility | Max Safe Context (Q4 KV) |
| :--- | :--- | :--- | :--- | :--- |
| **Q8_0** | `Nex-N2.5-mini-Q8_0.gguf` | 36.90 GB | ❌ OOM (Exceeds 30.2 GB) | None |
| **Q6_K** | `Nex-N2.5-mini-Q6_K.gguf` | 28.51 GB | ⚠️ Extremely Risky | ~4k context |
| **Q5_K_M** | `Nex-N2.5-mini-Q5_K_M.gguf` | 24.73 GB | ⚠️ Marginal | ~32k context |
| **Q4_K_M (Selected)** | `Nex-N2.5-mini-Q4_K_M.gguf` | **21.17 GB** | **✅ Sweet Spot (Recommended)** | **131,072 tokens (128k)** |
| **Q4_K_S** | `Nex-N2.5-mini-Q4_K_S.gguf` | 19.89 GB | ✅ Fully Supported | 196,608 tokens (196k) |
| **Q3_K_M** | `Nex-N2.5-mini-Q3_K_M.gguf` | 16.76 GB | ✅ Extra Headroom | 262,144 tokens (256k) |
| **F16 mmproj (Vision)** | `mmproj-Nex-N2.5-mini-F16.gguf`| **0.88 GB** (899 MB)| ✅ Compatible with Q4_K_M | 131,072 tokens (128k) |

*The Q4_K_M quant preserves ~99.1% of FP16 perplexity while leaving ~9.03 GB usable VRAM for KV cache and CUDA runtime on 2x T4.*

---

## 3. Kaggle 2x Tesla T4 Hardware Constraints & Theoretical Math

### Hardware Specifications
* **GPUs:** 2x NVIDIA Tesla T4
* **Compute Architecture:** Turing (`sm_75`), Compute Capability 7.5
* **Physical VRAM:** 15,360 MiB per GPU = 30,720 MiB total (~30.0 GiB)
* **Usable VRAM:** ~30.2 GB (after 250–300 MiB baseline driver/CUDA overhead per card)
* **Interconnect:** PCIe 3.0 x16 (~15.75 GB/s bidirectional per card; **no NVLink** on Kaggle)
* **Lack of Native FP8:**
  * Turing `sm_75` 2nd-gen Tensor Cores natively accelerate **FP16, INT8, and INT4**.
  * **FP8 Tensor Cores do NOT exist on Turing** (introduced only in Ada Lovelace `sm_89` and Hopper `sm_90`).
  * Running FP8 weights or FP8 KV cache (`fp8_e4m3`, `fp8_e5m2`) forces CPU emulation or software casts, heavily penalizing throughput.
  * In contrast, llama.cpp integer quantized KV cache (`-ctk q4_0 -ctv q4_0`) runs directly on INT4/INT8 arithmetic with zero emulation penalty.

### Memory Allocation Math on 2x T4 (Split 1:1)
* **Base Model Offload:** 21.17 GB / 2 = ~10.58 GB per GPU.
* **Available Headroom (Text-Only):**
  * GPU 0: 15.10 GB - 10.58 GB = **~4.52 GB**
  * GPU 1: 15.10 GB - 10.58 GB = **~4.52 GB**
  * Combined Available: **~9.04 GB**
* **Available Headroom (with `mmproj` on GPU 0):**
  * GPU 0: 15.10 GB - 10.58 GB - 0.88 GB = **~3.64 GB**
  * GPU 1: 15.10 GB - 10.58 GB = **~4.52 GB**
  * Combined Available: **~8.16 GB**

### KV Cache Memory Formulas
Given `num_layers = 40`, `num_kv_heads = 2`, `head_dim = 256`:
$$\text{Elements per token} = 2 \times 40 \times 2 \times 256 = 40,960 \text{ elements/token}$$

* **FP16 (2.0 bytes/element):** $81,920 \text{ bytes/token} = 80.0 \text{ KB/token}$
* **Q8_0 (~1.0625 bytes/element):** $43,520 \text{ bytes/token} = 42.5 \text{ KB/token}$
* **Q4_0 (~0.5625 bytes/element):** $23,040 \text{ bytes/token} = 22.5 \text{ KB/token}$

#### Context Scaling VRAM Consumption Table (KV Cache Only)
| Context Length | FP16 KV Cache | Q8_0 KV Cache | Q4_0 KV Cache | 2x T4 Feasibility with Q4_K_M |
| :--- | :--- | :--- | :--- | :--- |
| **8,192 (8k)** | 0.64 GB | 0.34 GB | **0.18 GB** | ✅ Plentiful headroom |
| **16,384 (16k)**| 1.28 GB | 0.68 GB | **0.36 GB** | ✅ Plentiful headroom |
| **32,768 (32k)**| 2.56 GB | 1.36 GB | **0.72 GB** | ✅ Plentiful headroom |
| **65,536 (64k)**| 5.12 GB | 2.72 GB | **1.44 GB** | ✅ Rock solid |
| **131,072 (128k)**| 10.24 GB (❌ OOM)| 5.44 GB (⚠️ Marginal)| **2.88 GB** | **✅ Safe & Recommended** |
| **196,608 (196k)**| 15.36 GB (❌ OOM)| 8.16 GB (❌ OOM)| **4.32 GB** | ✅ Feasible (Single Stream) |
| **262,144 (256k)**| 20.48 GB (❌ OOM)| 10.88 GB (❌ OOM)| **5.76 GB** | ⚠️ Fits bare, tight with batching |

---

## 4. Tailored Benchmark Suite for 2x Tesla T4

### 4.1 Speed, Throughput & Memory Sweep
1. **Raw Engine Throughput (`llama-bench`):**
   * Command: `llama-bench -m Nex-N2.5-mini-Q4_K_M.gguf -ngl 99 -ts 1,1 -ctk q4_0 -ctv q4_0 -fa 1 -p 512,2048,8192 -n 128,512 -b 512,1024`
   * Measures prompt processing speed (tok/s) and generation speed (tok/s) without HTTP API overhead.
2. **End-to-End Context Scaling Sweep (8k $\to$ 128k):**
   * Measure TTFT (Time-to-First-Token) and ITL (Inter-Token Latency) across context lengths `[8192, 16384, 32768, 65536, 98304, 131072]`.
   * Monitor peak VRAM per GPU via background `nvidia-smi` sampling (0.5s intervals).
3. **KV Cache Quantization Ablation (`q4_0` vs `q8_0`):**
   * Measure memory savings vs perplexity delta.
   * Prove OOM boundary for `q8_0` at >64k context vs `q4_0` stability up to 128k.

### 4.2 Agentic & Coding Capabilities
1. **Terminal-Bench 2.1:**
   * Nex-AGI Reported Score: **73.4**
   * Evaluates autonomous bash execution, environment diagnostics, directory traversal, git conflict operations, and script correction.
2. **SWE-Bench Lite / SWE-Bench Pro:**
   * Nex-AGI Reported Score: **43.8** (SWE-Bench Pro)
   * Evaluates resolving real-world GitHub issues end-to-end (code localization, patch generation, multi-file edits, unit test verification).
3. **BFCL (Berkeley Function-Calling Leaderboard):**
   * Evaluates `<tool_call>` syntax precision, multi-turn tool calling, parallel tool execution, and handling invalid arguments/schemas.
4. **Toolathlon / ToolBench:**
   * Evaluates multi-step decision trees and API interaction loops.

### 4.3 Mathematical & Logical Reasoning
1. **GSM8K (8-shot & 0-shot CoT):**
   * 1,319 test problems.
   * Test across reasoning modes: `reasoning_effort="none"` vs `"medium"` vs `"high"`.
   * Verify whether `<think>` tags reduce arithmetic hallucination.
2. **MATH-500:**
   * 500 challenging problems across algebra, calculus, and number theory.
   * Compares reasoning depth and solution accuracy.
3. **AIME 2024:**
   * Olympiad mathematics. Evaluate reasoning token budgets and step-level self-verification.

### 4.4 Long-Context Retrieval
1. **Needle In A Haystack (NIAH) up to 128k:**
   * Grid of 10 context depths (8k, 16k, 32k, 48k, 64k, 80k, 96k, 112k, 128k) $\times$ 5 needle placement depths (0%, 25%, 50%, 75%, 100%).
   * Metric: Retrieval accuracy (Pass / Fail).
2. **RULER (Synthetic Long-Context Benchmark):**
   * Tests multi-needle retrieval, tracking variable mutations, and long-range dependency resolution up to 128k.

### 4.5 Multimodal & Computer Use (with `mmproj-Nex-N2.5-mini-F16.gguf`)
1. **OSWorld & OSWorld-Verified:**
   * Nex-AGI Reported Score: **71.2** (OSWorld-Verified), **82.9** (OSWorld-G)
   * Evaluates screen capture comprehension, bounding box grounding, keyboard/mouse action synthesis, and visual closed-loop feedback.
2. **MMMU (Massive Multi-discipline Multimodal Understanding):**
   * College-level multimodal reasoning across STEM diagrams, charts, and tables.

---

## 5. Recommended 2x T4 Configuration Matrix & Scoring Metrics

### Recommended Serving Configuration for 2x T4
```bash
llama-server \
  --model /kaggle/working/Nex-N2.5-mini-Q4_K_M.gguf \
  --mmproj /kaggle/working/mmproj-Nex-N2.5-mini-F16.gguf \
  --n-gpu-layers 99 \
  --tensor-split 1,1 \
  --ctx-size 131072 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --flash-attn \
  --batch-size 1024 \
  --ubatch-size 512 \
  --parallel 1 \
  --cont-batching \
  --host 0.0.0.0 \
  --port 8080 \
  --alias nex-n2.5-mini
```

### Benchmark Metric Targets & SLA Criteria
| Phase / Benchmark | Core Metric | Expected / Target Score (2x T4) | Max Latency / VRAM SLA |
| :--- | :--- | :--- | :--- |
| **Speed (8k Context)** | Prompt tok/s / Gen tok/s | ~120 tok/s (PP) / ~32-38 tok/s (TG) | TTFT < 400ms |
| **Speed (128k Context)**| Prompt tok/s / Gen tok/s | ~45 tok/s (PP) / ~26-30 tok/s (TG) | Peak VRAM < 27.5 GB |
| **KV Cache Stability** | Perplexity delta (GSM8K) | $\Delta \text{PPL} < 0.08$ (Q4_0 vs Q8_0) | 0 OOM errors at 128k |
| **Terminal-Bench 2.1** | Success Rate | > 70.0% | Pass rate on shell tasks |
| **SWE-Bench Pro** | Resolved Rate | > 40.0% | Multi-file diff accuracy |
| **BFCL (Tool Calling)**| Function Call Accuracy | > 92.0% | Strict schema adherence |
| **GSM8K Accuracy** | Accuracy (high thinking) | > 90.0% | Validates `<think>` logic |
| **NIAH 128k** | Retrieval Accuracy | 100% across all depths | Green retrieval heatmap |
| **OSWorld-Verified** | Task Completion Rate | ~70.0% | Requires `mmproj` loaded |
