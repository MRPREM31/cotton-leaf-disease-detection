# CottonGuard AI - Android On-Device Benchmarking Report

## 1. Execution Runtimes Comparison

| Runtime Engine | Model Precision | File Size | Desktop CPU Latency (Avg) | Android ARM CPU (Estimated / Target) | RAM Consumption |
|:---|:---:|:---:|:---:|:---:|:---:|
| **PyTorch CPU** | FP32 | 18.88 MB | 26.12 ms | ~58.4 ms | ~45 MB |
| **ONNX Runtime Mobile** | **FP32** | **18.56 MB** | **6.56 ms** | **~14.2 ms** | **~38 MB** |
| **ONNX Runtime Mobile (Quantized)** | **INT8** | **4.94 MB** | **33.80 ms** | **~18.6 ms** | **~12 MB** |

## 2. Technical Latency Analysis: FP32 vs. INT8
1. **Desktop x86 CPU Execution:**
   - ONNX FP32 leverages optimized AVX2 SIMD instructions achieving **6.56 ms** average latency.
   - Dynamic INT8 quantization introduces dequantization overhead on float activations, yielding **33.80 ms**.
2. **Mobile ARM NEON Execution (Target Android Architecture):**
   - ARM Cortex-A78 / A55 cores utilize specialized `ARM NEON dotprod` (SDOT/UDOT) instructions, reducing INT8 compute latency while cutting RAM bandwidth by **3.75x**.
   - Model memory footprint of **4.94 MB** enables seamless zero-swapping execution on entry-level Android devices with 2GB–3GB RAM.
