# CottonNet - Research Benchmarks & Empirical Verification Report

## 1. Multi-Model Benchmark Comparison (SAR-CLD-2024 Dataset)

| Rank | Model Architecture | Test Macro F1 | Val Macro F1 | Total Parameters | Model Size (INT8) | Role in Study |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **#1** | **CottonNet (Proposed)** | **98.60%** | **98.13%** | **4.87M** | **4.94 MB** | **Champion Model** |
| #2 | DenseNet-121 | 96.26% | 99.07% | 6.96M | 7.20 MB | Teacher Network |
| #3 | MobileNetV3-Large | 95.79% | 94.86% | 4.21M | 4.30 MB | Edge Baseline |
| #4 | EfficientNet-B0 (Base) | 95.33% | 98.60% | 4.02M | 4.10 MB | Baseline Backbone |
| #5 | ResNet-50 | 93.93% | 96.26% | 23.52M | 23.80 MB | Heavy Baseline |

---

## 2. On-Device Empirical Latency Profile (Measured on CPU)

*Benchmarked over 50 consecutive warm inference iterations with $224 \times 224 \times 3$ tensors.*

| Execution Runtime | Precision | Mean Latency | Median (P50) | 95th Percentile (P95) | Min Latency | Memory Footprint |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PyTorch CPU** | FP32 | 26.12 ms | 25.40 ms | 32.51 ms | 23.10 ms | 18.88 MB |
| **ONNX Runtime CPU** | FP32 | **6.56 ms** | **6.48 ms** | **7.19 ms** | **6.12 ms** | 18.56 MB |
| **ONNX Quantized CPU**| INT8 | 33.80 ms | 33.10 ms | 36.87 ms | 31.40 ms | **4.94 MB** |

> **Key Finding:** ONNX FP32 achieves **6.56 ms** inference on CPU with 0.026 maximum logit divergence from PyTorch. For memory-constrained devices, INT8 quantization compresses model footprint to **4.94 MB** (3.75x reduction).

---

## 3. Parameter Count Verification

- **Total Network Parameters:** `4,870,275` (~4.87M)
- **Trainable Parameters:** `4,561,615` (~4.56M)
- **Feature Extractor Dimensions:** 1,280 (EfficientNet-B0 stage 6)
- **SE-Block Reduction:** $r = 16$ (Channel bottleneck = 80)
- **Dual Pooling Concatenation:** $1280 \times 2 = 2560$ dims
- **Dense Head Output:** 7 classes

---

## 4. Input Quality & Out-of-Distribution (OOD) Rejection Suite

| Test Case | Visual Characteristics | Quality Metrics | OOD Gatekeeper Action | Technical Reason |
|:---|:---|:---|:---:|:---|
| `pure_black_dark` | Luminance = 0 | Brightness = 0.0, Sharpness = 10% | **REJECTED** | *Image is too dark to resolve leaf symptoms.* |
| `pure_white_glare` | Luminance = 255 | Brightness = 255.0, Sharpness = 10% | **REJECTED** | *Image has severe glare or overexposure.* |
| `random_noise_ood` | Uniform RGB noise | Jitter = 127.4, Foliage = 0% | **REJECTED** | *Unstructured noise or invalid camera input.* |
| `uniform_gray_bg` | Luminance = 128 | Brightness = 128.0, Sharpness = 10% | **REJECTED** | *Image is too blurry for reliable clinical diagnosis.* |
| `valid_cotton_leaf`| Chlorophyll Green | Sharpness = 82%, Lighting = Optimal | **ACCEPTED** | Real CottonNet Softmax & Grad-CAM executed |
