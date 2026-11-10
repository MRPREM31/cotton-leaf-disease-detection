# FINAL ANDROID ON-DEVICE BENCHMARK REPORT

**Status:** **PARTIAL / DESKTOP EMULATION ONLY (PHYSICAL ARM DEVICE: NOT TESTED)**  
**Audit Rule Applied:** No theoretical estimates converted into experimental hardware claims.

---

## 1. Verified Execution Runtimes (Local Desktop CPU Baseline)

*Measured on AMD64 CPU (100 warm iterations with standard 224x224 input tensors):*

| Model Precision | Export Artifact | File Size | Mean Latency | Median (P50) | 95th Percentile (P95) | Experimental Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **PyTorch FP32** | `cottonnet_best_model.pth` | 18.88 MB | **26.12 ms** | 25.40 ms | 32.51 ms | **VERIFIED (CPU)** |
| **ONNX FP32** | `cottonnet_fp32.onnx` | 18.56 MB | **6.56 ms** | 6.48 ms | 7.19 ms | **VERIFIED (CPU)** |
| **ONNX Dynamic INT8** | `cottonnet_int8.onnx` | **4.94 MB** | **33.80 ms** | 33.10 ms | 36.87 ms | **VERIFIED (CPU)** |

---

## 2. Target Android Deployment Architecture

```text
[CameraX Viewfinder (1080p)]
             │
             ▼
[Bitmap Center Crop -> 224x224]
             │
             ▼
[FloatBuffer Normalization (ImageNet Mean/Std)]
             │
             ▼
[ONNX Runtime Mobile Engine (cottonnet.onnx)]
             │
    ┌────────┴────────┐
    ▼                 ▼
[CPU Execution]   [NNAPI / XNNPACK]
 (ARM NEON)       (Qualcomm / MediaTek NPU)
```

---

## 3. Physical Android Device Status Table

| Device Benchmark Requirement | Measured Hardware Value | Experimental Status | Technical Reason |
|:---|:---:|:---:|:---|
| **Real Android Phone Model** | *N/A* | **NOT TESTED** | Physical Android device via ADB not connected in current build environment. |
| **SoC / Chipset Benchmark** | *N/A* | **NOT TESTED** | Requires execution on physical Snapdragon / MediaTek testbed. |
| **Physical RAM Consumption** | *~12 MB (Projected)* | **NOT TESTED** | Exact Android heap RSS requires Android Studio Profiler run. |
| **Cold Inference Latency** | *N/A* | **NOT TESTED** | Requires initial APK cold-launch profiling on physical hardware. |
| **Warm Inference Latency** | *N/A* | **NOT TESTED** | Requires 100 on-device camera-feed iterations. |

---

## 4. Why INT8 Latency is Higher on Desktop x86 than FP32
1. **Instruction Set Difference:** On desktop x86 CPUs without AVX-512 VNNI (Vector Neural Network Instructions), ONNX Runtime dynamically dequantizes INT8 weights to floating-point registers before performing Matrix Multiplication (GEMM). This adds runtime conversion overhead.
2. **Mobile Advantage (ARM NEON `dotprod`):** Target ARM Cortex-A78/A55 mobile cores natively execute INT8 dot products (`SDOT`/`UDOT`) directly in hardware, delivering optimal throughput and a **3.75x reduction in memory bandwidth** (4.94 MB vs. 18.56 MB).
