# Qwen3.6-35B-A3B on Kaggle 2x Tesla T4 GPUs 🚀

**รัน Qwen3.6-35B-A3B บน Kaggle GPU T4 x 2 (32GB VRAM) พร้อม Cloudflare Tunnel (Custom Domain `api.zexnoz.dev` หรือ Quick Tunnel ฟรี) และ Context มหาศาล 196,608 tokens (196k)**

---

## ⚡ สเปกและคอนฟิกูเรชัน (Configuration)

* **Model**: `unsloth/Qwen3.6-35B-A3B-GGUF` (`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`)
* **Model Alias**: `qwen3.6-35B-A3B`
* **GPU**: Kaggle Free Quota — 2x Nvidia Tesla T4 (Tensor Split 1:1, Pipeline Layer Parallel)
* **Context Window**: `196,608` tokens (196k)
* **KV Cache**: `q4_0` (Quantized KV Cache ประหยัด VRAM)
* **Batch / UBatch**: `1024` / `512`
* **Network**:
  * **Static Custom Domain**: `https://api.zexnoz.dev/v1` (เมื่อใส่ `CF_TUNNEL_TOKEN`)
  * **Free Quick Tunnel**: `https://*.trycloudflare.com/v1` (อัตโนมัติหากไม่ระบุ Token)
* **API Key**: `kilo-secret-key` (ปรับแต่งได้)

---

## 💻 1. รันผ่าน Terminal (CLI Runner เหมือน Qwen 3.8 27B)

### ข้อกำหนดก่อนเริ่ม
ติดตั้ง kaggle package (หากยังไม่ได้ติดตั้ง):
```bash
pip install kaggle
```
ตรวจเช็คว่ามีไฟล์ยืนยันตัวตน `~/.kaggle/access_token` หรือ `~/.kaggle/kaggle.json`

### สั่ง Deploy และรับ Live Stream Log ทันที
เข้าสู่โฟลเดอร์ `qwen36_35b_a3b`:
```bash
cd qwen36_35b_a3b
```

**กรณีที่ 1: รันด้วย Named Tunnel Token (ต่อเข้า `api.zexnoz.dev` โดยตรง)**
```bash
python launch.py serve --cf-token <YOUR_CLOUDFLARE_TUNNEL_TOKEN>
```
*(หรือตั้ง Environment Variable `CF_TUNNEL_TOKEN` ไว้ล่วงหน้าแล้วรัน `python launch.py serve`)*

**กรณีที่ 2: รันแบบฟรี Quick Tunnel (ไม่ต้องใช้ Token/Domain)**
```bash
python launch.py serve
```

**กรณีที่ 3: แนบ Kaggle Dataset เพื่อเปิดเครื่องแบบ 0 วินาที (Instant Mount)**
```bash
python launch.py serve --dataset <username/dataset-slug>
```

---

### หน้าต่างแสดงผลใน Terminal

```text
[10:30:00]  Pushing kernel 'zeenoz/qwen36-35b-2x-t4-server' to Kaggle Cloud (2x Tesla T4)...
[10:30:04]  Kernel pushed successfully!
[10:30:15]  Kaggle kernel initialized. Starting GPU environment...
[10:30:18]  GPU status: Detected 2 GPU(s): Tesla T4, 15360 MiB, Tesla T4, 15360 MiB
[10:30:30]  Fetching prebuilt llama.cpp CUDA 12 binaries (sm_75)...
[10:30:45]  Downloading Qwen3.6-35B-A3B GGUF (~20 GB) to scratch space...
[10:32:30]  Model verified: Qwen3.6-35B-A3B-UD-Q4_K_M.gguf (20.50 GB)
[10:32:35]  Starting llama-server (2x Tesla T4 split, 196k context, batch 1024/512)...
[10:33:05]  Local server initialized and healthy.
[10:33:10]  Establishing Cloudflare Tunnel...

==========================================================================
🚀 QWEN3.6-35B-A3B SERVER ONLINE (2x TESLA T4)
==========================================================================
📡 Base URL       : https://api.zexnoz.dev/v1
🤖 Model ID       : qwen3.6-35B-A3B
🧠 Context Length : 196,608 tokens (196k)
⚡ Batch / UBatch : 1024 / 512 (KV: q4_0)
🔑 API Key        : kilo-secret-key
==========================================================================
```

กด `Ctrl+C` เพื่อกลับสู่ Terminal ได้ตลอดเวลา — เซิร์ฟเวอร์ยังคงทำงานอยู่บน Kaggle Cloud

---

### ตรวจสอบสถานะและหยุดเซิร์ฟเวอร์

* **ตรวจเช็คสถานะ**:
  ```bash
  python launch.py status
  ```
* **หยุดการทำงาน (คืนโควต้า GPU ทันที)**:
  ```bash
  python launch.py stop
  ```

---

## 📓 2. รันผ่าน Kaggle Notebook (Web Browser)

1. เข้าเว็บ Kaggle -> คลิก **+ Create** -> **New Notebook**
2. เลือก **File** -> **Import Notebook** -> เลือกไฟล์ [`notebook/qwen36_35b_a3b_t4.ipynb`](notebook/qwen36_35b_a3b_t4.ipynb)
3. เมนูขวา (**Settings**):
   * **Accelerator**: `GPU T4 x 2`
   * **Internet**: `On`
4. (ทางเลือก) หากใช้ Named Tunnel: ไปที่ **Add-ons** -> **Secrets** -> เพิ่ม Secret ชื่อ `CF_TUNNEL_TOKEN`
5. คลิก **Run All** เพื่อเริ่มใช้งานทันที

---

## 🔌 3. เชื่อมต่อกับ Coding Agents & API

### Claude Code
```bash
export OPENAI_BASE_URL="https://api.zexnoz.dev/v1"
export OPENAI_API_KEY="kilo-secret-key"
export OPENAI_MODEL="qwen3.6-35B-A3B"
claude
```

### Cursor / VS Code / Cline
* **OpenAI Base URL**: `https://api.zexnoz.dev/v1`
* **API Key**: `kilo-secret-key`
* **Model ID**: `qwen3.6-35B-A3B`

### Python OpenAI SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.zexnoz.dev/v1",
    api_key="kilo-secret-key"
)

response = client.chat.completions.create(
    model="qwen3.6-35B-A3B",
    messages=[{"role": "user", "content": "เขียนโปรแกรมคำนวณ Fibonacci ด้วย Python"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```
