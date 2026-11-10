# PROJECT AUDIT: CottonGuard AI Research & Edge Deployment Suite

**Date:** 2026-09-01  
**Project Title:** CottonGuard AI: An Explainable, Robust and Offline Deep Learning System for Cotton Leaf Disease Diagnosis on Resource-Constrained Android Devices  
**Project Scope:** Final-Year B.Tech Thesis & Peer-Reviewed Research Publication  

---

## 1. System & Architectural Inventory

### 1.1 Model Checkpoint & Neural Architecture
- **Primary Model:** Custom `CottonNet` (`cottonnet_best_model.pth`, 18.88 MB).
  - Backbone: `timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)` (1,280 feature dimensions).
  - Attention Mechanism: Squeeze-and-Excitation (`SEBlock(1280, reduction=16)`).
  - Pooling: Dual spatial pooling (`AdaptiveAvgPool2d(1)` + `AdaptiveMaxPool2d(1)` concatenated to 2,560 dimensions).
  - Classification Head: `Dropout(0.4) -> Linear(2560, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.3) -> Linear(256, 7)`.
  - Distillation Target: Teacher `DenseNet-121` (96.26% test F1).
  - Parameter Breakdown: **4,870,275 total parameters** (**4,561,615 trainable**).
- **Other Checkpoints Available:**
  - `densenet121_best_model.pth` (28.47 MB, Teacher Model)
  - `mobilenetv3_large_best_model.pth` (17.06 MB, Fast Edge Baseline)
  - `efficientnet_b0_best_model.pth` (16.37 MB, Standard Backbone Baseline)
  - `resnet50_best_model.pth` (94.41 MB, Heavy Baseline)

### 1.2 Exact Preprocessing & Class Mapping
- **Input Dimensions:** `[1, 3, 224, 224]`, RGB channel format.
- **Normalization Constants:**
  - $\mu = [0.485, 0.456, 0.406]$
  - $\sigma = [0.229, 0.224, 0.225]$
- **Dataset Class Index Mapping (SAR-CLD-2024):**
  - `0`: `Bacterial Blight` (*Xanthomonas malvacearum*)
  - `1`: `Curl Virus` (Cotton Leaf Curl Virus / CLCuV)
  - `2`: `Healthy Leaf` (Normal foliage control)
  - `3`: `Herbicide Growth Damage` (Abiotic agrochemical drift)
  - `4`: `Leaf Hopper Jassids` (*Amrasca biguttula biguttula*)
  - `5`: `Leaf Redding` (Physiological Mg deficiency / cold stress)
  - `6`: `Leaf Variegation` (Nutritional micronutrient mosaic)

---

## 2. Current Pipeline Assessment

```text
[Camera / Gallery]
       │
       ▼
[Image Quality & Pre-Gatekeeper]
 (Laplacian blur, mean luminance, HSV foliage coverage)
       │
  ┌────┴────────────────────────┐
  ▼                             ▼
[REJECTED (OOD/Degraded)]   [ACCEPTED]
  "Undetermined / OOD"          │
                                ▼
                     [Exact Preprocessing]
                     (RGB 224x224, ImageNet Norm)
                                │
                                ▼
                     [CottonNet PyTorch / ONNX]
                                │
                                ▼
                     [Softmax Logits & Top-3]
                                │
                                ▼
                     [Grad-CAM Attribution]
```

---

## 3. Known Problems & Scientific Risks

1. **Closed-Set Overconfidence on Out-of-Distribution Inputs:**
   - *Risk:* Deep CNNs without explicit calibration or gatekeeping assign high softmax probability to random noise or other plant species.
   - *Mitigation:* Multi-stage gatekeeper (resolution, exposure, Laplacian blur, HSV plant manifold, spatial jitter coherence) + systematic confidence thresholding.
2. **Heuristic vs. Learned Gatekeeper Distinction:**
   - *Risk:* Calling the preliminary HSV/Laplacian check an "AI cotton detector" would fail peer review.
   - *Mitigation:* Explicitly classify it as "Stage-1 Heuristic Quality & Foliage Validation" and provide clear roadmap for a learned MobileNetV3-Small binary gatekeeper.
3. **Quantization Latency Trade-Offs:**
   - *Risk:* Assuming INT8 quantization is always faster. On x86 CPU without VNNI/AVX-512 INT8 instructions, dynamic quantization overhead can increase latency compared to optimized FP32 GEMM.
   - *Mitigation:* Document exact hardware profiling (x86 CPU vs ARM NEON mobile CPU) in `ANDROID_BENCHMARK.md`.
4. **Severity Labeling:**
   - *Risk:* Displaying "High Severity" based on classification confidence is scientifically invalid.
   - *Mitigation:* Purged all confidence-to-severity mappings. Display "Lesion Analysis: Active" or "Severity: Under research development".

---

## 4. Files That Should NOT Be Changed

| File | Reason |
|:---|:---|
| `cottonnet_best_model.pth` | Canonical research checkpoint with 98.60% test F1. |
| `class_mapping.json` | Dataset class order must remain strictly anchored to index 0..6. |
| `notebook9327109e29 (2).ipynb` | Research training provenance and teacher-student KD records. |

---

## 5. Files Requiring Implementation / Optimization

| Component | Target File | Action Required |
|:---|:---|:---|
| Model Verification | `verification_results.json` | Run fixed test set validation and output exact class probabilities. |
| Confidence Thresholds | `confidence_threshold_results.csv` | Sweep thresholds $0.30 \dots 0.95$ and calculate accuracy, FAR, FRR, coverage. |
| OOD Benchmarking | `OOD_RESULTS.md` | Test non-cotton, noise, glare, blur, and calculate rejection rates. |
| Robustness Sweeps | `ROBUSTNESS_RESULTS.csv` | Evaluate F1 degradation under noise, blur, low light, and JPEG compression. |
| Native Android Pipeline | `android/.../CottonNetClassifier.kt` | Implement offline ONNX Runtime Mobile + Stage-1 validation. |
| Reproducibility Spec | `REPRODUCIBILITY.md` | Document exact library versions, random seeds, and conversion commands. |
| Research Paper Outline | `RESEARCH_PAPER_OUTLINE.md` | Formulate research questions, ablation studies, and methodology. |
| Master Validation | `FINAL_VALIDATION_REPORT.md` | Final empirical summary of all experiments. |
