# Nex-N2.5-mini (Q4_K_M) on Kaggle 2x Tesla T4 GPUs 🚀

**รัน Nex-N2.5-mini (35B Qwen3.5 MoE) บน Kaggle GPU T4 x 2 (32GB VRAM) พร้อม Cloudflare Tunnel (Custom Domain `api.zexnoz.dev` หรือ Quick Tunnel ฟรี) และ Context มหาศาล 131,072 tokens (128k)**

โมเดล Agentic ยุคใหม่สำหรับการเขียนโค้ด, รัน Terminal, Tool Calling และ Computer Use พร้อมสถาปัตยกรรม 35.1B MoE (~3B active parameters ต่อ token) ให้ความเร็วสูงและประหยัด VRAM

---

## ⚡ สเปกและคอนฟิกูเรชัน (Configuration)

* **Model**: `abenzerps/Nex-N2.5-mini-GGUF` (`Nex-N2.5-Mini-Q4_K_M.gguf`, ~21.17 GB)
* **Kaggle Dataset**: [`zeenoz/nex-n2-5-mini-q4-k-m`](https://www.kaggle.com/datasets/zeenoz/nex-n2-5-mini-q4-k-m) (รองรับ Instant 0s Mount ⚡)
* **Base Architecture**: `Qwen3_5MoeForConditionalGeneration` (35.1B Total, ~3.0B Active/Token, 256 Routed Experts / Top-8)
* **Model Alias**: `nex-n2.5-mini`
* **GPU**: Kaggle Free Quota — 2x Nvidia Tesla T4 (Tensor Split 1:1, Layer Parallel)
* **Context Window**: `131,072` tokens (128k) (สถาปัตยกรรมรองรับสูงสุดถึง 262,144 tokens / 256k)
* **KV Cache**: `q4_0` (Quantized KV Cache ประหยัด VRAM พอดีกับ 2x T4 โดยไม่มี OOM)
* **Batch / UBatch**: `1024` / `512`
* **Reasoning Format**: `<think>...</think>` tags พร้อม `reasoning_effort` ("none", "medium", "high")
* **Network**:
  * **Static Custom Domain**: `https://api.zexnoz.dev/v1` (เมื่อใส่ `CF_TUNNEL_TOKEN`)
  * **Free Quick Tunnel**: `https://*.trycloudflare.com/v1` (อัตโนมัติเมื่อไม่ใส่ Token)
* **API Key**: `kilo-secret-key` (ตั้งค่าได้)

---

## 💻 1. รันผ่าน Terminal บนเครื่องคุณ (CLI Runner)

### สิ่งที่ต้องเตรียม (Prerequisites)
ติดตั้ง Kaggle API CLI (หากยังไม่ได้ติดตั้ง):
```bash
pip install kaggle
```
ตรวจสอบว่ามี Token อยู่ที่ `~/.kaggle/access_token` หรือ `~/.kaggle/kaggle.json`

### สั่ง Deploy และดู Live Stream Log
เข้าไปที่โฟลเดอร์ `nex_n25_mini_q4_k_m`:
```bash
cd nex_n25_mini_q4_k_m
```

**กรณีที่ 1: รันพร้อมผูก Dataset บน Kaggle (0-Second Mount ไม่ต้องรอโหลด 21GB)**
```bash
python launch.py serve --dataset zeenoz/nex-n2-5-mini-q4-k-m
```

**กรณีที่ 2: รันด้วย Named Tunnel Token (ชี้เข้า Domain `api.zexnoz.dev`)**
```bash
python launch.py serve --cf-token <YOUR_CLOUDFLARE_TUNNEL_TOKEN>
```
*(หรือตั้งค่า `CF_TUNNEL_TOKEN` ใน Environment Variable)*

**กรณีที่ 3: รันแบบ Quick Tunnel ฟรี (ไม่ต้องใช้ Token/Domain)**
```bash
python launch.py serve
```

---

### ภาพรวม Terminal Output ขณะรัน

```text
[16:30:00]  Pushing kernel 'zeenoz/nex-n25-mini-2x-t4-server' to Kaggle Cloud (2x Tesla T4)...
[16:30:04]  Kernel pushed successfully!
[16:30:15]  Kaggle kernel initialized. Starting GPU environment...
[16:30:18]  GPU status: Detected 2 GPU(s): Tesla T4, 15360 MiB, Tesla T4, 15360 MiB
[16:30:30]  Fetching prebuilt llama.cpp CUDA 12 binaries (sm_75)...
[16:30:35]  Mounted from Dataset: Nex-N2.5-Mini-Q4_K_M.gguf (19.71 GB) [0s Mount ⚡]
[16:30:40]  Starting llama-server (2x Tesla T4 split, 128k context, batch 1024/512)...
[16:31:05]  Local server initialized and healthy.
[16:31:10]  Establishing Cloudflare Tunnel...

==========================================================================
🚀 NEX-N2.5-MINI SERVER ONLINE (2x TESLA T4)
==========================================================================
📡 Base URL       : https://api.zexnoz.dev/v1
🤖 Model ID       : nex-n2.5-mini
🧠 Context Length : 131,072 tokens (128k)
⚡ Batch / UBatch : 1024 / 512 (KV: q4_0)
🔑 API Key        : kilo-secret-key
==========================================================================
```

กด `Ctrl+C` ได้ตลอดเวลาเพื่อ Detach กลับมาที่ Terminal — เซิร์ฟเวอร์จะยังคงทำงานต่อไปบน Kaggle Cloud

---

### ตรวจสอบสถานะและหยุดเซิร์ฟเวอร์

* **เช็คสถานะการทำงาน**:
  ```bash
  python launch.py status
  ```
* **หยุดเซิร์ฟเวอร์ทันที (คืน GPU Quota)**:
  ```bash
  python launch.py stop
  ```

---

## 📓 2. รันผ่าน Kaggle Notebook (หน้าเว็บ Browser)

1. เปิด Kaggle -> กด **+ Create** -> **New Notebook**
2. **File** -> **Import Notebook** -> เลือกไฟล์ [`notebook/nex_n25_mini_t4.ipynb`](notebook/nex_n25_mini_t4.ipynb)
3. เมนูด้านขวา (**Settings**):
   * **Accelerator**: `GPU T4 x 2`
   * **Internet**: `On`
4. เมนูด้านขวา (**Input**):
   * กด **Add Input** -> ค้นหา `zeenoz/nex-n2-5-mini-q4-k-m` เพื่อ Mount โมเดลแบบ 0 วินาที
5. (ไม่บังคับ) สำหรับ Named Tunnel: ไปที่ **Add-ons** -> **Secrets** -> เพิ่ม Secret ชื่อ `CF_TUNNEL_TOKEN`
6. กด **Run All** เพื่อเริ่มใช้งานทันที

---

## 🔌 3. การเชื่อมต่อ Coding Agents & API

### Claude Code
```bash
export OPENAI_BASE_URL="https://api.zexnoz.dev/v1"
export OPENAI_API_KEY="kilo-secret-key"
export OPENAI_MODEL="nex-n2.5-mini"
claude
```

### Cursor / VS Code / Cline
* **OpenAI Base URL**: `https://api.zexnoz.dev/v1`
* **API Key**: `kilo-secret-key`
* **Model ID**: `nex-n2.5-mini`

### Python OpenAI SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.zexnoz.dev/v1",
    api_key="kilo-secret-key"
)

response = client.chat.completions.create(
    model="nex-n2.5-mini",
    messages=[{"role": "user", "content": "Write a Python script to verify palindromes with unit tests."}],
    extra_body={"reasoning_effort": "medium"},
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

---

## 📊 4. ชุด Benchmark ที่เหมาะสมสำหรับ 2x Tesla T4

รายละเอียดผลวิเคราะห์และงานวิจัยอย่างละเอียดดูได้ที่ [`BENCHMARK_RESEARCH.md`](BENCHMARK_RESEARCH.md)

### รัน Harness ทดสอบในเครื่องหรือบน Kaggle:
```bash
# รัน Benchmark พื้นฐาน (Speed sweep, VRAM, GSM8K reasoning)
python benchmark_q4.py

# รัน v2 Throughput & Concurrency Suite
python benchmark_q4_v2.py --quick
```

### สรุปเกณฑ์ Benchmark ที่แนะนำ:
1. **Speed & Context Scaling Sweep**: วัด TTFT และ Throughput ที่ Context 8k, 16k, 32k, 64k, 128k บน 2x T4
2. **KV Cache Efficiency**: เปรียบเทียบ `q4_0` (ใช้เพียง 2.88 GB ที่ 128k) กับ `q8_0` (5.44 GB)
3. **Agentic & Tool Calling**:
   * **Terminal-Bench 2.1**: คะแนนทางการของ Nex-N2.5-mini อยู่ที่ **73.4**
   * **SWE-Bench Pro**: คะแนนทางการอยู่ที่ **43.8**
   * **BFCL (Berkeley Function-Calling Leaderboard)**: ทดสอบความแม่นยำของ JSON Schema Tool Calling
4. **Reasoning Trace & Mathematics**:
   * **GSM8K & MATH-500**: ทดสอบความแม่นยำของการคิดวิเคราะห์ผ่าน `<think>...</think>` tag ด้วย `reasoning_effort` ระดับต่างๆ
5. **Needle In A Haystack (NIAH)**: ทดสอบการค้นหาข้อมูลใน Long Context สูงสุดถึง 128k tokens
