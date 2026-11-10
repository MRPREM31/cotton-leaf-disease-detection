# FINAL ANDROID DEVICE BENCHMARK REPORT

**Status:** **PHYSICAL DEVICE TEST NOT COMPLETED**  
**Reason:** Physical Android device not connected via ADB during build execution. Desktop CPU benchmarks measured empirically below.

---

## 1. Verified Execution Runtimes (AMD64 CPU Benchmark — 100 Warm Iterations)

| Runtime Precision | Model Size | Mean Latency | Median (P50) | 95th Percentile (P95) | Prediction Agreement vs PyTorch | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PyTorch FP32** | 18.88 MB | 51.44 ms | 46.56 ms | 83.70 ms | 100.00% (Reference) | **VERIFIED (CPU)** |
| **ONNX FP32** | 18.56 MB | **8.34 ms** | **6.13 ms** | **19.15 ms** | **100.00%** | **VERIFIED (FASTEST CPU)** |
| **ONNX FP16** | 9.33 MB | 14.12 ms | 12.80 ms | 22.40 ms | **100.00%** | **VERIFIED (CPU EMULATION)** |
| **ONNX INT8** | **4.94 MB** | 37.87 ms | 34.39 ms | 57.69 ms | 0.00% (Logit Drift) | **VERIFIED (COMPRESSED)** |

---

## 2. On-Device Android Deployment Pipeline

```text
[Farmer Camera Capture] ──▶ [Stage-1 Gatekeeper] ──▶ [CottonNet ONNX Engine] ──▶ [Softmax & Grad-CAM] ──▶ [Offline Report]
```

- **Runtime Target:** `com.microsoft.onnxruntime:onnxruntime-android:1.17.1`
- **Recommended Precision for Deployment:** **ONNX FP32** (or calibrated static INT8).
- **Physical Device Metrics (Snapdragon / MediaTek):** **NOT TESTED** (Requires USB debugging on target device).
