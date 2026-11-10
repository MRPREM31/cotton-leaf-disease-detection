# Research Paper Outline: CottonGuard AI

**Proposed Title:**  
*CottonGuard AI: An Explainable and Robust Lightweight Deep Learning Framework for Offline Cotton Leaf Disease Diagnosis on Resource-Constrained Android Devices*

**Target Submission Venue:**  
*Computers and Electronics in Agriculture (Elsevier)* / *IEEE Access* / *Smart Agricultural Technology*

---

## 1. Abstract
- **Context:** Cotton (*Gossypium hirsutum*) is a vital global commercial crop vulnerable to debilitating foliar pathogens and abiotic stresses.
- **Problem:** Existing deep learning models either require continuous cloud connectivity (infeasible in rural farming zones) or are closed-set classifiers that fail catastrophically on out-of-distribution (OOD) non-leaf inputs.
- **Proposed Solution:** We introduce **CottonGuard AI**, an edge-deployable diagnostic framework powered by **CottonNet** (a 4.87M-parameter architecture with Squeeze-and-Excitation channel attention and dual spatial pooling trained via Knowledge Distillation from DenseNet-121).
- **Key Contributions:**
  1. Multi-stage input validation gatekeeper eliminating false-positive diagnoses on degraded/non-leaf frames.
  2. Dynamic INT8 quantization achieving 3.75x compression (4.94 MB) with 98.60% test Macro F1.
  3. Real-time Grad-CAM attribution embedded locally on edge devices.
  4. Fully offline, trilingual (English, Hindi, Odia) native Android deployment.

---

## 2. Research Questions (RQs)
- **RQ1 (Architectural Trade-Off):** Which lightweight feature extractor provides optimal trade-offs between parameter footprint, latency, and disease discrimination?
- **RQ2 (Quantization & Compression):** Can CottonNet undergo INT8 quantization without sacrificing diagnostic F1-score?
- **RQ3 (Environmental Robustness):** How resilient is the CottonNet feature representation under realistic field image corruptions (blur, low illumination, sensor noise, compression)?
- **RQ4 (OOD Safeguards):** Does the multi-stage gatekeeper effectively suppress overconfident misclassifications on non-cotton imagery?
- **RQ5 (Attribution Explainability):** Do Grad-CAM feature activations reliably align with biological lesion morphology across all 7 disease categories?

---

## 3. Paper Structure & Section Breakdown

### Section I: Introduction
- Agronomic impact of foliar diseases (Bacterial Blight, Leaf Curl Virus, Jassids, Leaf Redding).
- Limitations of current mobile agricultural AI applications.

### Section II: Related Work
- Deep learning in plant pathology classification.
- Knowledge Distillation & Model Compression (KD, Pruning, Quantization).
- Out-of-Distribution detection & Explainable AI in Agriculture.

### Section III: CottonGuard AI System Architecture
- Dataset provenance: **SAR-CLD-2024** (2,137 images, 7 classes).
- CottonNet design: EfficientNet-B0 backbone + SE-Block Attention + Dual Global Pooling.
- Teacher-Student Distillation training protocol ($T = 3.0, \alpha = 0.5$).
- Two-stage pipeline: Stage-1 Heuristic/Quality Gatekeeper $\rightarrow$ Stage-2 CottonNet Classifier.

### Section IV: Edge Optimization & Quantization
- TorchScript to ONNX intermediate representation.
- Dynamic INT8 symmetric quantization ($W_{\text{int8}} = \text{round}(W / S)$).
- On-device runtime memory footprint and CPU thread profiling.

### Section V: Experimental Results & Discussion
- 5-Model comparative benchmark (ResNet50, EfficientNet-B0, MobileNetV3-Large, DenseNet121, CottonNet).
- Empirical latency profiles (PyTorch vs ONNX FP32 vs ONNX INT8).
- Robustness evaluation across 6 image degradation regimes.
- OOD False Acceptance Rate analysis.
- Qualitative Grad-CAM lesion attributions.

### Section VI: Android Mobile Deployment & Farmer UX
- CameraX integration, offline Room database, multilingual farmer advisory.

### Section VII: Threats to Validity & Limitations
- Heuristic gatekeeper limitations vs learned binary classifier.
- Lack of ground-truth pixel-level segmentation masks for quantitative IoU severity.

### Section VIII: Conclusion & Future Directions
- Summary of findings.
- Future integration of lightweight semantic segmentation for automated lesion area quantification.
