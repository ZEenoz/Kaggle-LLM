# Qwen3.8-35B-A3B (Q4_K_M) 2x Tesla T4 Sweet-Spot Benchmark

**Hardware**: Kaggle 2x Nvidia Tesla T4 (32GB Total VRAM)
**Model**: `empero-ai/Qwen3.8-35B-A3B-Distill-GGUF` (`Qwen3.8-35B-A3B-Q4_K_M.gguf`)
**KV Cache**: `q8_0` | **Batch**: `1024` | **UBatch**: `512`

## 📊 Context Scaling & Performance Results

| Context Window | Fit | Peak VRAM (GB) | Short Gen tok/s | Long Prefill tok/s | Long TTFT (s) | Long Gen tok/s |
|---|---|---|---|---|---|---|
| 131,072 | ✅ Pass | 22.75 | n/a | 1243.34 | 22.16 | 310.54 |
| 147,456 | ✅ Pass | 23.11 | n/a | 1187.39 | 23.21 | 307.89 |
| 163,840 | ✅ Pass | 23.46 | n/a | 1128.47 | 24.42 | 282.52 |
| 180,224 | ✅ Pass | 23.81 | n/a | 1113.75 | 24.74 | 309.40 |
| 196,608 | ✅ Pass | 24.17 | n/a | 1132.35 | 24.33 | 300.61 |

## 🎯 Recommended Sweet Spot

- **Optimal Context Size**: `196,608` tokens
- **KV Cache Quantization**: `q8_0`
- **Batch / UBatch**: `1024` / `512`
- **Peak VRAM**: `24.17 GB` (fits within 2x T4 budget)
- **Decode Generation Speed**: `300.61 tok/s`
- **Prefill Processing Speed**: `1132.35 tok/s` (TTFT: `24.33 s`)

**Rationale**: Context 196,608 tokens achieves peak VRAM 24.17 GB with generation speed 300.61 tok/s and long prefill 1132.35 tok/s safely within Kaggle 2x T4 limits.