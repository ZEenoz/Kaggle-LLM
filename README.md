# kaggle-gpu-lab 🚀

**Serve Qwen3.8-27B on Kaggle's free 2x Tesla T4 GPUs (32GB VRAM), with an instant OpenAI-compatible endpoint and free Cloudflare Quick Tunnel.**

Works out of the box with **Claude Code, Cursor, Cline, Codex CLI, OpenCode**, or any tool that speaks the OpenAI API.

* **Free Tier**: Uses Kaggle's free quota of 30 GPU hours/week (GPU T4 x 2).
* **Speed to Live Endpoint**: **~2.5 - 3 minutes total** (model streamed at 180 MB/s to scratch storage).
* **Massive Context**: **131,072 tokens (128k)** via Flash Attention (`-fa on`) + quantized 4-bit KV cache (`-ctk q4_0 -ctv q4_0`).
* **Zero Configuration**: Free Public URL generated automatically via Cloudflare Quick Tunnel. No token, no domain, and no credit card required.

---

## ⚡ Quick Start — From Your Terminal

### 1. Requirements & Dependencies
Install the Kaggle CLI:
```bash
pip install -r requirements.txt
```

### 2. Kaggle API Token (One-time setup)
Place your Kaggle API Token in `~/.kaggle`:
* **Access Token (Recommended)**: `%USERPROFILE%\.kaggle\access_token`
* **Or Legacy JSON**: `%USERPROFILE%\.kaggle\kaggle.json`
*(Note: If you have already saved your token in `.kaggle/access_token`, the launcher detects it automatically!)*

### 3. Launch Server in Cloud
```bash
python launch.py serve
```

That's it! The launcher pushes the script to Kaggle Cloud and streams live progress right in your local terminal:

```text
[15:10:02]  Pushing kernel 'user/qwen38-2x-t4-server' to Kaggle Cloud (2x Tesla T4)...
[15:10:05]  Kernel pushed successfully!
[15:10:15]  Kaggle kernel initialized. Starting GPU environment...
[15:10:18]  GPU status: Detected 2 GPU(s): Tesla T4, 15360 MiB, Tesla T4, 15360 MiB
[15:10:35]  Fetching prebuilt llama.cpp CUDA 12 binaries...
[15:10:50]  Downloading Qwen3.8-27B GGUF (~17 GB) to scratch space (takes ~1.5 - 2 min)...
[15:12:15]  Model verified: 17.20 GB
[15:12:20]  Starting llama-server (2x Tesla T4 split, 128k context)...
[15:12:45]  Local server initialized and healthy.
[15:12:50]  Establishing free Cloudflare Quick Tunnel...

======================================================================
🎉 YOUR QWEN 3.8 (27B) ENDPOINT IS LIVE!
======================================================================
🌐 Public API Base   : https://xxxx-xxxx-xxxx.trycloudflare.com/v1
🧠 Model ID          : qwen3.8-27b
📏 Context Length    : 131,072 tokens (128k)
🔑 API Key           : Not required (open access)
======================================================================
```

Press `Ctrl+C` at any time to detach — the server remains online in Kaggle Cloud.

---

## ⚡ Instant Startup via Mounted Dataset (0s Download)

Instead of downloading the 17GB model from Hugging Face every run (~1.5 min), you can attach a Kaggle Dataset:
```bash
# Attach public dataset or your own dataset:
python launch.py serve --dataset drvivektrivedi34/qwen38-27b-q4km
```
When attached, Kaggle mounts the model into `/kaggle/input` in 0 seconds, bringing total launch time down to **under 45 seconds**!

---

## 🛠️ Managing Your Cloud Server

* **Check status**:
  ```bash
  python launch.py status
  ```
* **Stop session**:
  ```bash
  python launch.py stop
  ```

---

## 💻 Plug Into Coding Agents

### 1. Claude Code
```bash
export OPENAI_BASE_URL="https://<your-tunnel>.trycloudflare.com/v1"
export OPENAI_API_KEY="none"
export OPENAI_MODEL="qwen3.8-27b"
claude
```

### 2. Cursor / Cline / VS Code
* **OpenAI Base URL**: `https://<your-tunnel>.trycloudflare.com/v1`
* **API Key**: `none` (or any string)
* **Model**: `qwen3.8-27b`

### 3. Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<your-tunnel>.trycloudflare.com/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Write a Python script for quicksort"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

---

## 📓 Quick Start — As a Kaggle Notebook (Web UI)

If you prefer using the browser:
1. Open [kaggle.com](https://www.kaggle.com) -> Click **+ Create** -> **New Notebook**.
2. Go to **File** -> **Import Notebook** -> Select [`notebook/qwen38_27b_t4.ipynb`](notebook/qwen38_27b_t4.ipynb).
3. In the right panel (**Settings**):
   * **Accelerator**: `GPU T4 x 2`
   * **Internet**: `On`
4. Click **Run All**!
