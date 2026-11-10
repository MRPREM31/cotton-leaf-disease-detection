# CottonGuard AI: Empirical Reproducibility & Environment Specification

## 1. Environment & Library Versions
- **Python Version:** 3.11.8 (AMD64 / Windows 11 / Linux Compatible)
- **PyTorch:** 2.13.0
- **TorchVision:** 0.28.0
- **TIMM (PyTorch Image Models):** 1.0.29
- **ONNX Runtime:** 1.29.0
- **ONNX:** 1.22.0
- **NumPy:** 2.4.6
- **Pillow:** 12.3.0

## 2. Model Checkpoint Specifications
- **Checkpoint Path:** `cottonnet_best_model.pth`
- **Total Parameters:** 4,870,275 (4.87M)
- **Trainable Parameters:** 4,561,615 (4.56M)
- **Input Tensor Dimensions:** `[1, 3, 224, 224]`
- **Normalization Policy:**
  - $	ext{Mean} = [0.485, 0.456, 0.406]$
  - $	ext{Std} = [0.229, 0.224, 0.225]$
- **Random Seed:** 42 (training partition split)

## 3. Export & Conversion Commands
```bash
# 1. Export FP32 & INT8 Quantized ONNX
python export_onnx.py

# 2. Run Comprehensive Benchmarking
python benchmark_pipeline.py

# 3. Run Scientific Experiments (OOD, Robustness, Confidence Sweeps)
python run_experiments.py

# 4. Start Local Real-Time Inference Server
python server.py
```
