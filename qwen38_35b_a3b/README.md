# Qwen3.8-35B-A3B on Kaggle 2x Tesla T4 GPUs 🚀

ชุดสคริปต์สำหรับทดสอบค้นหา **Sweet-Spot Context Window (130k - 196k)** และรัน **Inference Server 24/7** สำหรับโมเดล **Qwen3.8-35B-A3B (Q4_K_M)** บน Kaggle Free GPU T4 x 2 (32GB VRAM) พร้อม Cloudflare Tunnel (Custom Domain `api.zexnoz.dev` หรือ Quick Tunnel ฟรี)

---

## 🔬 การวิเคราะห์ VRAM & สเปกทางเทคนิค (VRAM & Architecture Analysis)

* **Model**: `empero-ai/Qwen3.8-35B-A3B-Distill-GGUF` (`Qwen3.8-35B-A3B-Q4_K_M.gguf`)
* **Download URL**: `https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF/resolve/main/Qwen3.8-35B-A3B-Q4_K_M.gguf`
* **Architecture**: MoE (Mixture of Experts)
  * Total Parameters: ~35B
  * Active Parameters: ~3B (Active 3B per token -> decode ไวมาก)
  * Layers (Block Count): 41 layers
  * Attention Heads: 16 (Query) / **2 (KV Heads)**
  * Maximum Native Context: 262,144 tokens (256k)
* **GPU Hardware**: Kaggle 2x Nvidia Tesla T4 (รวม 32GB VRAM, ใช้งานได้จริง ~30.2GB)
* **Model Weights VRAM Footprint**: ~20.22 GB (แบ่ง Tensor Split 1:1 เหลือ ~10.1 GB ต่อ GPU)
* **KV Cache VRAM (`q8_0`) & Flash Attention (`-fa on`)**:
  * ด้วยสถาปัตยกรรม GQA ที่มี **KV Heads เพียง 2 heads** ทำให้แม้ใช้ `q8_0` ก็ยังประหยัด VRAM มาก:
    * **131,072 tokens (128k)**: ใช้ KV Cache ประมาณ **~2.84 GB**
    * **163,840 tokens (160k)**: ใช้ KV Cache ประมาณ **~3.55 GB**
    * **196,608 tokens (192k)**: ใช้ KV Cache ประมาณ **~4.25 GB**
  * **รวม VRAM เมื่อดัน Context 196k ด้วย `q8_0`**: ~20.5 GB (Weights) + ~4.25 GB (KV Cache) + ~1.0 GB (Compute buffer ด้วย Flash-Attn) = **~25.75 GB** จาก 30.2 GB ที่มี (เหลือ Headroom สบายๆ ~4.4 GB โดยไม่ชน OOM และได้ Attention Precision สูงสุด)

---

## 📁 โครงสร้างไฟล์ในโฟลเดอร์

```text
qwen38_35b_a3b/
├── benchmark_sweet_spot.py       # สคริปต์ค้นหา sweet-spot (sweep context 130k-196k, batch 1024/512, kv q4_0)
├── launch.py                     # CLI launcher สั่งงาน Kaggle kernel จากเครื่องคุณ (serve / benchmark / status / stop)
├── kernel/
│   └── serve_qwen.py             # สคริปต์รันเซิร์ฟเวอร์ llama.cpp + Cloudflare Tunnel บน Kaggle
├── notebook/
│   └── qwen38_35b_a3b_t4.ipynb   # Jupyter Notebook พร้อมใช้งานบน Kaggle
└── README.md                     # เอกสารคู่มือการใช้งาน
```

---

## 🧪 1. วิธีรันค้นหา Sweet-Spot (Context Sweep 130k-196k)

สคริปต์จะทดสอบ Context ขนาดต่างๆ (เช่น 131072, 147456, 163840, 180224, 196608 tokens) ที่ `batch 1024` / `ubatch 512` และ `kv q4_0`:
- ตรวจสอบว่าโมเดลโหลดขึ้น VRAM ได้โดยไม่ชน OOM หรือไม่
- วัดความเร็ว **Short Generation (Decode tok/s)**
- วัดความเร็ว **Long Context Prefill (Prompt tok/s)** และ **Time-To-First-Token (TTFT)**
- บันทึก Telemetry ของ VRAM แยกรายใบ (GPU 0 / GPU 1)
- สร้างสรุปผล `sweet_spot.json`, `results.json`, `results.csv` และ `summary.md`

### วิธีที่ 1.1: สั่งรันผ่าน Terminal CLI (ส่งไปรันบน Kaggle 2xT4 ทันที)
```bash
cd qwen38_35b_a3b
python launch.py benchmark
```
*(ระบบจะสร้าง Kernel บน Kaggle, ติดตั้ง llama.cpp, ดึงโมเดล และเริ่ม Sweep context ให้แบบอัตโนมัติ)*

### วิธีที่ 1.2: รันผ่าน Kaggle Notebook
1. อัปโหลดไฟล์ [`notebook/qwen38_35b_a3b_t4.ipynb`](notebook/qwen38_35b_a3b_t4.ipynb) ขึ้น Kaggle
2. ปรับ Accelerator เป็น **GPU T4 x 2** และเปิด **Internet: On**
3. รัน **Cell 1** เพื่อค้นหา Sweet-Spot

### วิธีที่ 1.3: รันตรงใน Shell หรือ Kaggle Terminal
```bash
python benchmark_sweet_spot.py --context-sizes 131072,147456,163840,180224,196608 --batch-size 1024 --ubatch-size 512 --kv-cache q8_0
```

---

## 🚀 2. วิธีรัน Production Server (เปิดให้ยิง API 24/7)

เมื่อได้ Sweet-Spot Context แล้ว (ค่าเริ่มต้นแนะนำที่ `196,608` tokens พร้อม KV `q8_0` และ Flash-Attn `-fa on`):

### สั่งรันผ่าน Terminal
```bash
# กรณีที่ 1: รันด้วย Named Tunnel (api.zexnoz.dev)
python launch.py serve --cf-token <YOUR_CLOUDFLARE_TUNNEL_TOKEN>

# กรณีที่ 2: รันด้วยฟรี Quick Tunnel (ไม่ต้องใส่ Token)
python launch.py serve

# กรณีที่ 3: ปรับขนาด Context หรือพารามิเตอร์เพิ่มเติม
python launch.py serve --context 196608 --batch-size 1024 --ubatch-size 512
```

### หน้าต่างแสดงผลเมื่อเซิร์ฟเวอร์พร้อมใช้งาน
```text
==========================================================================
🚀 QWEN3.8-35B-A3B SERVER ONLINE (2x TESLA T4)
==========================================================================
📡 Base URL       : https://api.zexnoz.dev/v1
🤖 Model ID       : qwen3.8-35B-A3B
🧠 Context Length : 196,608 tokens (196k)
⚡ Batch / UBatch : 1024 / 512 (KV: q8_0 | Flash-Attn: ON)
🔑 API Key        : kilo-secret-key
==========================================================================
```

### ตรวจสอบสถานะและหยุดเซิร์ฟเวอร์
```bash
# ตรวจสอบสถานะ
python launch.py status

# สั่งหยุดเซิร์ฟเวอร์เพื่อคืนโควต้า GPU
python launch.py stop
```

---

## 🔌 3. ตัวอย่างการเรียกใช้งาน (API Usage)

### ทดสอบผ่าน cURL
```bash
curl https://api.zexnoz.dev/v1/chat/completions \
  -H "Authorization: Bearer kilo-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-35B-A3B",
    "messages": [{"role": "user", "content": "สวัสดีครับ แนะนำตัวเองหน่อยครับ"}]
  }'
```

### ตั้งค่าใช้งานใน Cursor / Claude Code / Codex / OpenAI SDK
```bash
export OPENAI_BASE_URL="https://api.zexnoz.dev/v1"
export OPENAI_API_KEY="kilo-secret-key"
export OPENAI_MODEL="qwen3.8-35B-A3B"
```
