# Ornith-1.5-35B-A3B on Kaggle 2x Tesla T4 GPUs 🚀

**Run Ornith-1.5-35B-A3B on Kaggle GPU T4 x 2 (32GB VRAM) with Cloudflare Tunnel (Custom Domain `api.zexnoz.dev` or free Quick Tunnel) and a 196,608 token (192k) Context**

---

## ⚡ Specs and Configuration

* **Model**: `gbuzhf/Ornith-1.5-35B-A3B-TIEL-Calibrated-MTPv2-ICE-GGUF` (`Ornith-1.5-35B-A3B-TIEL_Calibrated-MTPv2-21G-ICE.gguf`, 20.84 GB)
* **Model Features**: MoE Router in F32 (Zero Routing Degradation), Embedded MTPv2 Head (`--spec-type draft-mtp` enabled for ~22-28 tok/s)
* **Model Alias**: `ornith-1.5-35B-A3B`
* **GPU**: Kaggle Free Quota — 2x Nvidia Tesla T4 (Tensor Split 1:1, Pipeline Layer Parallel)
* **Context Window**: `196,608` tokens (192k)
* **KV Cache**: `q4_0` (Quantized KV Cache, saves VRAM)
* **Batch / UBatch**: `1024` / `512`
* **Network**:
  * **Static Custom Domain**: `https://api.zexnoz.dev/v1` (when `CF_TUNNEL_TOKEN` is set)
  * **Free Quick Tunnel**: `https://*.trycloudflare.com/v1` (automatic when no token is provided)
* **API Key**: `kilo-secret-key` (configurable)

---

## 💻 1. Run via Terminal (CLI Runner)

### Prerequisites
Install the kaggle package (if not installed):
```bash
pip install kaggle
```
Check that you have credentials at `~/.kaggle/access_token` or `~/.kaggle/kaggle.json`

### Deploy and stream live logs
Enter the `ornith15_35b_a3b` folder:
```bash
cd ornith15_35b_a3b
```

**Case 1: Run with a Named Tunnel Token (connects directly to `api.zexnoz.dev`)**
```bash
python launch.py serve --cf-token <YOUR_CLOUDFLARE_TUNNEL_TOKEN>
```
*(Or set the `CF_TUNNEL_TOKEN` environment variable first, then run `python launch.py serve`)*

**Case 2: Run with a free Quick Tunnel (no Token/Domain needed)**
```bash
python launch.py serve
```

**Case 3: Attach a Kaggle Dataset for 0-second boot (Instant Mount)**
```bash
python launch.py serve --dataset <username/dataset-slug>
```

**Case 4: Pass Hugging Face Token for fast authenticated downloads**
```bash
python launch.py serve --hf-token hf_xxxxxxxxxxxxxxxxx
```
*(Or set the `HF_TOKEN` environment variable, or add `HF_TOKEN` in Kaggle Secrets `Add-ons` -> `Secrets`)*

---

### Run Benchmark on Kaggle Cloud (2x T4)

Sweep context sizes (100k-160k+) with MTP enabled:
```bash
python launch.py benchmark
```

---

### Check status and stop the server

* **Check status**:
  ```bash
  python launch.py status
  ```
* **Stop (releases GPU quota immediately)**:
  ```bash
  python launch.py stop
  ```

---

## 📓 2. Run via Kaggle Notebook (Web Browser)

1. Open Kaggle -> click **+ Create** -> **New Notebook**
2. **File** -> **Import Notebook** -> select [`notebook/ornith15_35b_a3b_t4.ipynb`](notebook/ornith15_35b_a3b_t4.ipynb)
3. Right panel (**Settings**):
   * **Accelerator**: `GPU T4 x 2`
   * **Internet**: `On`
4. (Optional) For a Named Tunnel: go to **Add-ons** -> **Secrets** -> add a secret named `CF_TUNNEL_TOKEN`
5. Click **Run All** to start immediately

---

## 🔌 3. Connecting Coding Agents & API

### Claude Code
```bash
export OPENAI_BASE_URL="https://api.zexnoz.dev/v1"
export OPENAI_API_KEY="kilo-secret-key"
export OPENAI_MODEL="ornith-1.5-35B-A3B"
claude
```

### Cursor / VS Code / Cline
* **OpenAI Base URL**: `https://api.zexnoz.dev/v1`
* **API Key**: `kilo-secret-key`
* **Model ID**: `ornith-1.5-35B-A3B`

### Python OpenAI SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.zexnoz.dev/v1",
    api_key="kilo-secret-key"
)

response = client.chat.completions.create(
    model="ornith-1.5-35B-A3B",
    messages=[{"role": "user", "content": "Write a Python program that computes Fibonacci numbers"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```
