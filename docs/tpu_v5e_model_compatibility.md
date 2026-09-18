# TPU v5e-8 Model Compatibility & Serving Guide (Kaggle Environment)

## 1. Hardware Architecture & Memory Budget

On Kaggle, selecting **Accelerator: TPU v5e-8** provisions a single TPU VM slice equipped with **8 TPU v5e chips** (code name *ViperLite*).

### Specifications
* **Chip Count / Cores:** 8 chips (TensorCore v5e), `tp=8` (Tensor Parallelism = 8).
* **HBM (High Bandwidth Memory):** 16 GB HBM2 per chip = **128 GB total HBM**.
* **Memory Bandwidth:** 819 GB/s per chip (Aggregate ~6.55 TB/s).
* **Interconnect:** 2D Torus Inter-Chip Interconnect (ICI) with 400 Gbps per chip.
* **Compute Capabilities:**
  * Matrix Multiply Units (MXU) native to **BF16** (197 TFLOPS/chip, ~1,576 TFLOPS aggregate).
  * Native **INT8** support (393 TOPS/chip, ~3,144 TOPS aggregate).
* **Host Memory:** ~180 – 240 GB system CPU RAM.

### Memory Sizing Thresholds (128 GB HBM Total)
When serving with Tensor Parallelism (`tp=8`), model weights and KV caches are partitioned equally across the 8 chips:

| Parameter Size | Weight Format | Weights Memory | Remaining KV Cache | Feasibility on TPU v5e-8 |
| :--- | :--- | :--- | :--- | :--- |
| **7B – 9B** | BF16 (2B/p) | ~14 – 18 GB | ~110 GB | **Optimal** (Extreme batching / massive context up to 128k+) |
| **14B – 16B** | BF16 (2B/p) | ~28 – 32 GB | ~96 GB | **Optimal** (High speed, 64k – 128k context) |
| **27B – 32B** | BF16 (2B/p) | ~54 – 64 GB | ~64 – 74 GB | **Sweet Spot** (Full BF16 precision, 32k – 64k context) |
| **47B (MoE 8x7B)**| BF16 (2B/p) | ~94 GB | ~34 GB | **Supported** (Fast inference due to sparse activation) |
| **70B – 72B** | BF16 (2B/p) | ~140 – 144 GB | None (OOM) | **Not Feasible in unquantized BF16** (> 128 GB) |
| **70B – 72B** | INT8 / FP8 (1B/p)| ~70 – 72 GB | ~56 GB | **Supported** (Requires INT8/FP8 quantization) |
| **671B (DeepSeek-R1/V3)**| Any | > 350 GB | None (OOM) | **Impossible on single v5e-8** (Needs TPU pod slices like v5e-256) |

---

## 2. Serving Frameworks for TPU v5e

Unlike NVIDIA GPUs where CUDA and GGUF (`llama.cpp`) are standard, TPUs run compiled XLA / JAX graphs:

1. **vLLM TPU Backend (`vllm-tpu` / `tpu-inference`):**
   * Powered by Google's unified `tpu-inference` hardware plugin and PyTorch/XLA.
   * Compiles static computational buckets to prevent continuous recompilation.
   * Provides an OpenAI-compatible HTTP API server (`/v1/chat/completions`).
   * *Primary Source:* [vLLM Hardware Guide: TPU](https://docs.vllm.ai/en/latest/hardware/tpu.html) | [vLLM TPU-Inference Repo](https://github.com/vllm-project/tpu-inference)

2. **Google JetStream + MaxText:**
   * Google's flagship open-source inference engine written directly in JAX/XLA for TPUs.
   * Delivers maximum tokens/sec throughput for models explicitly converted to JAX checkpoints.
   * *Primary Source:* [Google Cloud MaxText](https://github.com/google/maxtext) | [Google Cloud JetStream](https://github.com/google/jetstream)

3. **Hugging Face Optimum TPU (`optimum-tpu`):**
   * Integrates Hugging Face Transformers pipelines with PyTorch/XLA.
   * *Primary Source:* [Hugging Face Optimum TPU](https://github.com/huggingface/optimum-tpu)

---

## 3. Recommended Models by Family

### A. DeepSeek Models
> The full **DeepSeek-V3** and **DeepSeek-R1 (671B MoE)** architectures **cannot** run on a single TPU v5e-8 due to the 128 GB memory limit and specialized MLA / MoE kernel compilation constraints.
>
> However, **DeepSeek-R1 Distilled models** use standard Dense Transformer architectures (Qwen2 and Llama-3) and run seamlessly on TPU v5e-8!

* **DeepSeek-R1-Distill-Qwen-32B (BF16):**
  * *Size:* ~64 GB.
  * *Suitability:* **Best Reasoning Model for TPU v5e-8**. Fits in full BF16 with ~64 GB HBM remaining for context and KV cache.
  * *Architecture:* Standard `Qwen2ForCausalLM` supported directly by `vllm-tpu` and `MaxText`.
* **DeepSeek-R1-Distill-Qwen-14B (BF16):**
  * *Size:* ~28 GB.
  * *Suitability:* Extremely fast decode speed (~150–220 tok/s), fits huge context windows (up to 128k).
* **DeepSeek-R1-Distill-Llama-8B (BF16):**
  * *Size:* ~16 GB.
  * *Suitability:* Ultra-lightweight reasoning engine, great for concurrent requests.
* **DeepSeek-Coder-V2-Lite-Instruct (16B total, 2.4B active MoE):**
  * *Size:* ~32 GB BF16.
  * *Suitability:* MoE code generation model that fits comfortably in memory.

---

### B. GLM Family (Zhipu AI)
* **GLM-4-9B-Chat / GLM-4-9B (BF16):**
  * *Size:* ~18 GB BF16.
  * *Suitability:* **Highly Recommended**. Leaves > 100 GB HBM free, enabling extensive long-context utilization (up to 128k–1M context tokens).
  * *Architecture:* Supported via `vllm` (`ChatGLM` / `GLM-4` model runner).
* **GLM-4-9B-1M:**
  * Tailor-made for long document analysis; TPU v5e-8 has ample memory to handle heavy KV caches.
* *Note on Larger GLMs (e.g. GLM-4-100B+):* Exceeds 128 GB HBM on a single node.

---

### C. Qwen Family (Alibaba)
* **Qwen2.5-32B-Instruct / Qwen2.5-Coder-32B-Instruct (BF16):**
  * *Size:* ~64 GB BF16.
  * *Suitability:* Top-tier general knowledge and coding capabilities. Considered the gold standard 32B model on TPU v5e-8.
* **Qwen2.5-14B-Instruct / Qwen2.5-7B-Instruct (BF16):**
  * *Size:* 28 GB and 14 GB respectively.
  * *Suitability:* Maximum throughput, ultra-low latency.
* **Qwen2.5-72B-Instruct (Quantized INT8 / FP8):**
  * *Size:* ~72 GB in 8-bit.
  * *Suitability:* Feasible on TPU v5e-8 using INT8 weight-only or FP8 quantization. Unquantized BF16 will OOM.

---

### D. Google Gemma Family (Native TPU Acceleration)
* **Gemma 2 27B (BF16):**
  * *Size:* ~54 GB BF16.
  * *Suitability:* **First-Class Optimization**. Built by Google DeepMind with specialized TPU kernel tuning, sliding window attention, and logit soft-capping. Supported natively in both MaxText and vLLM.
* **Gemma 2 9B (BF16):**
  * *Size:* ~18 GB BF16.
  * *Suitability:* High performance per watt, blazing inference speed.

---

### E. Meta Llama Family
* **Llama 3.1 8B-Instruct / Llama 3.2 3B (BF16):**
  * *Size:* ~16 GB and ~6 GB.
  * *Suitability:* Out-of-the-box support across all TPU serving engines with 128k context support.
* **Llama 3.1 70B-Instruct / Llama 3.3 70B-Instruct (INT8 / FP8):**
  * *Size:* ~70 GB in INT8.
  * *Suitability:* Runs with INT8 quantization, fitting inside the 128 GB memory envelope.

---

### F. Mistral AI Family
* **Mixtral 8x7B-Instruct-v0.1 (MoE, 46.7B total, 12.9B active):**
  * *Size:* ~94 GB in BF16.
  * *Suitability:* Supported in vLLM and MaxText. Yields fast token throughput due to sparse MoE execution while fitting inside 128 GB HBM.
* **Mistral-Nemo-Instruct-2407 (12B, BF16):**
  * *Size:* ~24 GB BF16.
  * *Suitability:* Excellent multilingual and reasoning capabilities with 128k context.
* **Codestral-22B (BF16):**
  * *Size:* ~44 GB BF16.
  * *Suitability:* Dedicated code generation engine, comfortably fits with ~84 GB HBM remaining for context.

---

## 4. Key Deployment Considerations on Kaggle

1. **Tensor Parallelism (`--tensor-parallel-size 8`):**
   * Must always specify `tp=8` to distribute the model across all 8 chips.
2. **XLA Graph Pre-compilation (Cold Start Mitigations):**
   * First-time XLA compilation can take 15–40 minutes on Kaggle.
   * To achieve ~3-4 minute fast startup, utilize pre-compiled XLA cache datasets or pinned batch size buckets (`--max-model-len 8192 --max-num-seqs 16`).
3. **Environment Packages:**
   * Kaggle TPU runtime includes standard `torch_xla`.
   * Install `vllm-tpu` or `tpu-inference` wheel corresponding to the Python/Torch version on Kaggle VM.
