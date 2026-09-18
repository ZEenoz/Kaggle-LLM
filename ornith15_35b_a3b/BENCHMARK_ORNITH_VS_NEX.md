# Benchmark & Comparative Analysis: Ornith-1.5-35B-A3B vs. Nex-N2.5-mini (GGUF Q4_K_M)

การเปรียบเทียบเชิงลึกระหว่างสองโมเดลเปิด (Open-weight) สถาปัตยกรรม Sparse Mixture-of-Experts (MoE 35B-A3B) ยอดนิยมสำหรับ Agentic Workflows

---

## 1. ข้อมูลสเปกและสถาปัตยกรรม (Architectural Comparison)

| หัวข้อ | Ornith-1.5-35B-A3B | Nex-N2.5-mini |
| :--- | :--- | :--- |
| **ผู้พัฒนา** | Ornith-AI | Nex-AGI (เปิดตัว ก.ย. 2026) |
| **สถาปัตยกรรมหลัก** | Sparse MoE (~3B active params / token) | Qwen3.5-35B-A3B base (~3B active params) |
| **จำนวนพารามิเตอร์** | รวม ~35.5B (Active ~3.0B) | รวม ~35.0B (Active ~3.0B) |
| **จุดเน้น (Specialization)** | Autonomous Software Engineering, Coding Agent | Multimodal Agent, Computer Use, Terminal Loops |
| **Native Context Length** | 262,144 tokens (256k) | 262,144 tokens (256k) |
| **การรองรับ Multimodal** | Text / Code Only | Vision-Language (ผ่าน `mmproj` projector) |
| **ขนาดไฟล์ GGUF Q4_K_M** | ~21.7 GB | ~22.0 GB (+ mmproj ~1.5 GB สำหรับ Vision) |
| **VRAM ขั้นต่ำ (Inference)** | ~22 GB (เหมาะกับ 2x Tesla T4 หรือ 24GB GPU) | ~22-24 GB |

---

## 2. ตารางคะแนน Benchmark หลัก (Benchmark Scores)

| Benchmark Suite | Ornith-1.5-35B-A3B | Nex-N2.5-mini | ผลต่าง (Delta) | ผู้ชนะ |
| :--- | :---: | :---: | :---: | :---: |
| **SWE-bench Pro** | **59.6%** | 43.8% | **+15.8%** | 🏆 **Ornith-1.5** |
| **SWE-bench Verified** | **79.0%** | N/A (DeepSWE: 36.1%) | สูงกว่าอย่างมีนัยสำคัญ | 🏆 **Ornith-1.5** |
| **SWE-bench Multilingual** | **71.4%** | N/A | - | 🏆 **Ornith-1.5** |
| **NL2Repo (Repo Generation)**| **46.2%** | N/A | - | 🏆 **Ornith-1.5** |
| **Terminal-Bench 2.1** | 68.5% *(Claude Code)*<br>67.8% *(Terminus-2)* | **73.4%** | **+4.9% ถึง +5.6%** | 🏆 **Nex-N2.5-mini** |
| **AutomationBench v1.0.6** | N/A | **32.3%** | รองรับ GUI/Web | 🏆 **Nex-N2.5-mini** |

---

## 3. วิเคราะห์ความต่างของคะแนน (Score Analysis)

### 1. งานด้านการเขียนโค้ดและแก้บั๊กซอฟต์แวร์ (Software Engineering): **Ornith-1.5 เหนือกว่ามาก**
* บน **SWE-bench Pro** ซึ่งเป็นการทดสอบแก้ Issue ซอฟต์แวร์จริงในระดับ Repository:
  * **Ornith-1.5-35B** ทำคะแนนได้ **59.6%**
  * **Nex-N2.5-mini** ได้เพียง **43.8%**
  * **คะแนนต่างกันถึง 15.8%**: Ornith มีความเข้าใจ context ข้ามไฟล์ (cross-file dependencies), การวางแผน refactor, และการสร้าง git patch ที่ผ่าน unit test ได้แม่นยำกว่าอย่างเห็นได้ชัด

### 2. งานคำสั่ง Terminal และ Agentic Loops: **Nex-N2.5-mini นำอยู่เล็กน้อย**
* บน **Terminal-Bench 2.1** (การใช้คำสั่ง bash/terminal เพื่อแก้โจทย์สภาพแวดล้อม):
  * **Nex-N2.5-mini** ทำคะแนนได้ **73.4%**
  * **Ornith-1.5-35B** ได้ **68.5%** (Claude Code harness)
  * **คะแนนต่างกันประมาณ 4.9%**: Nex-N2.5-mini ถูกเทรนลูป `observe → act → check → correct` สำหรับงาน command-line execution และ GUI interaction มาเป็นพิเศษ จึงเลือกใช้เครื่องมือและ command syntax พื้นฐานได้คล่องตัวกว่า

### 3. ผลกระทบของ Quantization GGUF `Q4_K_M`
* ทั้งสองโมเดลเป็นสถาปัตยกรรม **35B-A3B MoE**:
  * การ quantize เป็น `Q4_K_M` มีค่า Perplexity (PPL) degradation ต่ำมาก (สูญเสียความแม่นยำไม่เกิน 1–3% เทียบกับ BF16)
  * Expert routing ไม่พังที่ Q4_K_M ต่างจากพวก Q2_K หรือ Q3_K
  * สัดส่วนความต่างของคะแนนระหว่างทั้งสองโมเดลบน GGUF Q4_K_M ยังคงรักษา Relative Ranking เหมือนเวอร์ชัน FP16 เดิม

---

## 4. คำแนะนำในการเลือกใช้งาน (Decision Guide)

| กรณีการใช้งาน | แนะนำโมเดล | เหตุผล |
| :--- | :---: | :--- |
| **ใช้กับ Claude Code / Cursor / Codex ในงานเขียนโค้ด** | **Ornith-1.5-35B-A3B** | SWE-bench ชนะขาดลอย (+15.8%) วิเคราะห์ codebase ซับซ้อนได้ลึกกว่ามาก |
| **งาน DevOps, Bash automation, Shell scripting ล้วนๆ** | **Nex-N2.5-mini** | Terminal-Bench สูงกว่า 73.4% ตัดสินใจรัน shell commands ได้ตรงจุด |
| **งาน Browser agent / Computer use / ต้องการ Vision** | **Nex-N2.5-mini** | รองรับ multimodal vision ผ่าน `mmproj` (Ornith รองรับเฉพาะ text/code) |

---
*แหล่งอ้างอิง: HuggingFace Model Cards (ornith-ai/Ornith-1.5-35B-A3B, Nex-AGI/Nex-N2.5-mini), SWE-bench Official Leaderboard, Terminal-Bench 2.1 evaluation reports.*
