# Ornith-1.5-35B-A3B (TIEL-Calibrated MTPv2 ICE GGUF) 2x Tesla T4 Sweet-Spot Benchmark

**Hardware**: Kaggle 2x Nvidia Tesla T4 (32GB Total VRAM)
**Model**: `gbuzhf/Ornith-1.5-35B-A3B-TIEL-Calibrated-MTPv2-ICE-GGUF` (`Ornith-1.5-35B-A3B-TIEL_Calibrated-MTPv2-21G-ICE.gguf`)
**KV Cache**: `q4_0` | **Batch**: `1024` | **UBatch**: `512` | **MTP**: `Enabled`

## 📊 Context Scaling (100k+) & Performance Results

| Context Window | Fit | Peak VRAM (GB) | Short Gen tok/s | Long Prefill tok/s | Long TTFT (s) | Long Gen tok/s |
|---|---|---|---|---|---|---|
| 102,400 | ✅ Pass | 22.04 | n/a | 994.90 | 14.64 | 498.38 |
| 114,688 | ✅ Pass | 22.31 | n/a | 941.76 | 17.29 | 620.15 |
| 131,072 | ✅ Pass | 22.68 | n/a | 951.67 | 19.53 | 481.17 |
| 147,456 | ✅ Pass | 23.05 | n/a | 944.26 | 19.68 | 484.08 |
| 163,840 | ✅ Pass | 23.42 | n/a | 911.17 | 20.40 | 458.23 |

## 🎯 Recommended Sweet Spot

- **Optimal Context Size**: `114,688` tokens
- **KV Cache Quantization**: `q4_0`
- **Batch / UBatch**: `1024` / `512`
- **MTP Speculative Decoding**: `Enabled (--spec-type draft-mtp)`
- **Peak VRAM**: `22.31 GB` (fits within 2x T4 budget)
- **Decode Generation Speed**: `620.15 tok/s` (MTP accelerated)
- **Prefill Processing Speed**: `941.76 tok/s` (TTFT: `17.29 s`)

**Rationale**: Context 114,688 tokens achieves peak VRAM 22.31 GB with generation speed 620.15 tok/s (MTP accelerated) and long prefill 941.76 tok/s safely within Kaggle 2x T4 limits.