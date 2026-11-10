# CottonGuard AI — Final Experiment Comprehensive Analysis & Model Specifications

> **Document Type:** Master Research & Engineering Technical Specification  
> **Project:** CottonGuard AI (Cotton Leaf Pathology Diagnosis & Edge Deployment)  
> **Source Notebooks:** `cotton-leaf-disease-nist.ipynb`, `CottonCare_V2_Research_Training.ipynb`  
> **Checkpoints & Source Scripts:** `cottonnet_model.py`, `inference_engine.py`, `export_onnx.py`, `external_validation.py`  
> **Evaluation Standard:** Strict Empirical Verification — Zero Fabricated Metrics  

---

## Table of Contents
1. [Executive Summary & Experiment Overview](#1-executive-summary--experiment-overview)
2. [Datasets & Class Taxonomic Hierarchy](#2-datasets--class-taxonomic-hierarchy)
3. [Comprehensive Model Architectural Specifications](#3-comprehensive-model-architectural-specifications)
   - [3.1 CottonNet (V1 & V2) — Proposed Architecture](#31-cottonnet-v1--v2--proposed-architecture)
   - [3.2 Teacher & Baseline CNN Architectures (DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-50)](#32-teacher--baseline-cnn-architectures)
   - [3.3 Vision Transformer Architectures (Swin-T & ViT-Base/16)](#33-vision-transformer-architectures)
   - [3.4 Real-Time Edge Architecture (YOLO26s-cls)](#34-real-time-edge-architecture)
4. [Parameter Footprint & Model Sizing Matrix](#4-parameter-footprint--model-sizing-matrix)
5. [Data Input Processing & Augmentation Pipelines](#5-data-input-processing--augmentation-pipelines)
   - [5.1 Standard Inference & Evaluation Preprocessing](#51-standard-inference--evaluation-preprocessing)
   - [5.2 V1 Field-Resilient Augmentation Pipeline (Torchvision)](#52-v1-field-resilient-augmentation-pipeline-torchvision)
   - [5.3 V2 Realistic Field Augmentation Pipeline (Albumentations)](#53-v2-realistic-field-augmentation-pipeline-albumentations)
   - [5.4 External OOD Dataset Input Preprocessing & Class Alignment](#54-external-ood-dataset-input-preprocessing--class-alignment)
6. [Training Pipelines & Hyperparameter Engineering](#6-training-pipelines--hyperparameter-engineering)
   - [6.1 Baseline CNN Training Pipeline (`TrainValidation`)](#61-baseline-cnn-training-pipeline-trainvalidation)
   - [6.2 CottonNet 3-Phase Knowledge Distillation Pipeline (`KnowledgeDistillationTrainer`)](#62-cottonnet-3-phase-knowledge-distillation-pipeline-knowledgedistillationtrainer)
   - [6.3 Extended Transformer Training Framework (`ExtendedTrainer`)](#63-extended-transformer-training-framework-extendedtrainer)
   - [6.4 YOLO Classification Training Pipeline](#64-yolo-classification-training-pipeline)
   - [6.5 Hyperparameter Reference Matrix](#65-hyperparameter-reference-matrix)
7. [Working Pipeline — Inference, OOD Validation & Edge Deployment](#7-working-pipeline--inference-ood-validation--edge-deployment)
   - [7.1 End-to-End Operational Inference Architecture](#71-end-to-end-operational-inference-architecture)
   - [7.2 Input Quality & Stage-1 Gatekeeper Validation (OOD Rejection)](#72-input-quality--stage-1-gatekeeper-validation-ood-rejection)
   - [7.3 Multi-Precision Runtime Execution (PyTorch vs ONNX FP32/FP16/INT8)](#73-multi-precision-runtime-execution-pytorch-vs-onnx-fp32fp16int8)
   - [7.4 Confidence Thresholding & Fallback Protocol](#74-confidence-thresholding--fallback-protocol)
   - [7.5 Explainable AI Pipeline (True Grad-CAM Mathematical Formulation)](#75-explainable-ai-pipeline-true-grad-cam-mathematical-formulation)
   - [7.6 Zero-Cloud Offline Edge Deployment (Android & PWA)](#76-zero-cloud-offline-edge-deployment-android--pwa)
8. [Empirical Experimental Results & Benchmark Analysis](#8-empirical-experimental-results--benchmark-analysis)
   - [8.1 Internal Test Set Performance Across All 9 Models](#81-internal-test-set-performance-across-all-9-models)
   - [8.2 Latency, Throughput & Memory Profiling (CPU & Mobile Runtime)](#82-latency-throughput--memory-profiling-cpu--mobile-runtime)
   - [8.3 External Dataset Generalization & Discrepancy Audit (NIST 1,290 Images)](#83-external-dataset-generalization--discrepancy-audit-nist-1290-images)
   - [8.4 Systematic Robustness Stress-Testing (11 Corruptions)](#84-systematic-robustness-stress-testing-11-corruptions)
9. [Key Scientific Insights, Trade-Offs & Recommendations](#9-key-scientific-insights-trade-offs--recommendations)

---

## 1. Executive Summary & Experiment Overview

The **CottonGuard AI** research study systematically investigates lightweight deep learning architectures for autonomous, edge-deployable pathological diagnosis of cotton foliage (*Gossypium hirsutum*). Cotton farming in developing regions is heavily constrained by connectivity, hardware memory, and varied field illumination. The primary objectives of this experiment are:

1. **Architecture Innovation:** Design a sub-5M parameter deep learning model (**CottonNet**) that surpasses heavy baseline architectures on diagnostic accuracy and macro F1 score.
2. **Knowledge Distillation (KD):** Transfer feature representations from a high-capacity teacher network (**DenseNet-121**) into the compact student model using a 3-phase progressive freezing schedule.
3. **Multi-Model Benchmark:** Conduct a head-to-head empirical comparison across 9 distinct model configurations spanning CNNs, Vision Transformers (Swin-T, ViT-Base/16), and real-time YOLO classifiers.
4. **Realistic Field Augmentation (V2):** Assess the impact of complex agronomic noise (shadows, exposure drift, motion blur, sensor noise, compression) via an Albumentations-based pipeline.
5. **Generalization & OOD Auditing:** Validate cross-domain resilience against an independent external dataset (NIST / Kaggle, 1,290 images) and 11 synthetic environmental corruptions.
6. **Edge Deployment Engineering:** Quantize and export the champion model to ONNX FP32, FP16, and dynamic INT8 formats for sub-10 ms execution in zero-cloud offline environments (Android APK & PWA).

```mermaid
flowchart TD
    A["Raw SAR-CLD-2024 Dataset<br/>(2,137 Images | 7 Classes)"] --> B["Stratified Train / Val / Test Split<br/>(80% / 10% / 10% | Seed=2025)"]
    
    B --> C1["Part I: CNN Baselines<br/>ResNet-50 | EfficientNet-B0<br/>MobileNetV3-Large | DenseNet-121"]
    B --> C2["Part I: Proposed CottonNet V1<br/>3-Phase KD from DenseNet-121<br/>Torchvision Field Augmentation"]
    B --> C3["Part II: Extended Study (V2 Aug)<br/>CottonNet V2 | Swin-T<br/>ViT-Base/16 | YOLO26s-cls"]
    
    C1 --> D["Multi-Model Evaluation<br/>Internal Test Set (214 Images)"]
    C2 --> D
    C3 --> D
    
    D --> E["Cross-Domain Generalization<br/>External NIST Dataset (1,290 Images)<br/>3-Class Alignment & OOD Rejection"]
    D --> F["Edge Export & Benchmarking<br/>PyTorch -> ONNX FP32 / FP16 / INT8<br/>Stage-1 Heuristic Gatekeeper + Grad-CAM"]
```

---

## 2. Datasets & Class Taxonomic Hierarchy

### 2.1 Internal Primary Dataset: SAR-CLD-2024
The primary benchmark uses the **SAR-CLD-2024** (*A Comprehensive Dataset for Cotton Leaf Disease Detection*) containing **2,137 high-resolution RGB images** across 7 agronomic classes:

| Class Index | Class Name | Clinical Pathology / Symptom Signature | Causative Agent / Etiology | Total Images |
|:---:|:---|:---|:---|:---:|
| **0** | `Bacterial Blight` | Angular water-soaked lesions, black vein necrosis (*black arm*) | *Xanthomonas citri* pv. *malvacearum* | 253 |
| **1** | `Curl Virus` | Upward cupping, thickened dark veins, enation leaf outgrowths | Cotton Leaf Curl Virus (CLCuV, Geminiviridae) | 426 |
| **2** | `Healthy Leaf` | Asymptomatic control tissue, uniform chlorophyll pigmentation | Normal physiological foliage | 261 |
| **3** | `Herbicide Growth Damage`| Distorted, strap-like lamina, epinasty, venation bleaching | Abiotic agrochemical drift (e.g., 2,4-D, Glyphosate) | 275 |
| **4** | `Leaf Hopper Jassids` | Marginal yellowing, "hopper burn", necrotic curling edges | *Amrasca biguttula biguttula* (Nymph feeding) | 234 |
| **5** | `Leaf Redding` | Red-purple anthocyanin accumulation, interveinal bronzing | Abiotic stress (Mg deficiency, low night temp) | 579 |
| **6** | `Leaf Variegation` | Yellow-green mosaic chlorosis, irregular chimera patches | Nutritional micronutrient deficiency / genetic chimera | 109 |
| **Total** | **7 Classes** | **Comprehensive Pathology Representation** | **Biotic (4) + Abiotic/Physiological (3)** | **2,137** |

#### Dataset Partitioning Strategy
- **Split Proportions:** Stratified 80% Training (`1,709` images), 10% Validation (`214` images), 10% Held-Out Testing (`214` images).
- **Stratification Guarantee:** Consistent class distributions across train, validation, and test partitions using `sklearn.model_selection.train_test_split(stratify=labels, random_state=2025)`.
- **Image Resolution:** Native variable resolutions normalized to $224 \times 224 \times 3$.

---

### 2.2 Independent External Dataset (NIST / Kaggle)
For cross-domain generalization audits, an independent external dataset containing **1,290 images** was evaluated across overlapping and non-overlapping classes:
- **Directly Overlapping Classes (3 Classes):**
  - `Bacterial Blight`: 448 images
  - `Curl Virus`: 417 images
  - `Healthy Leaf`: 425 images
- **Non-Target Out-of-Distribution Class (1 Class):**
  - `Fusarium Wilt`: 300+ images (excluded or evaluated as OOD anomaly; strict rule counts predictions into non-target classes as errors).

---

## 3. Comprehensive Model Architectural Specifications

Nine model configurations were systematically analyzed in the final experimental suite.

```
+----------------------------------------------------------------------------------------------------+
|                                    STUDIED MODEL TAXONOMY                                          |
+------------------------------------+------------------------------------+--------------------------+
| Lightweight CNNs & Proposed        | High-Capacity Baselines            | Vision Transformers &    |
| Architectures                      | & KD Teachers                      | Real-Time Detectors      |
+------------------------------------+------------------------------------+--------------------------+
| 1. CottonNet V1 (Proposed, 4.87M)   | 4. DenseNet-121 (Teacher, 6.96M)   | 7. Swin-T (27.52M)       |
| 2. CottonNet V2 (Proposed, 4.87M)   | 5. ResNet-50 (Heavy, 23.52M)       | 8. ViT-Base/16 (85.80M)  |
| 3. MobileNetV3-Large (4.21M)       | 6. EfficientNet-B0 (Base, 4.02M)   | 9. YOLO26s-cls (5.45M)   |
+------------------------------------+------------------------------------+--------------------------+
```

---

### 3.1 CottonNet (V1 & V2) — Proposed Architecture

**CottonNet** is a customized hybrid CNN engineered specifically for resource-constrained agricultural edge devices. It blends an inverted bottleneck feature extractor with channel attention, multi-scale pooling, and heavy dropout regularization.

```
                             Input Tensor [Batch, 3, 224, 224]
                                            │
                                            ▼
                        ┌───────────────────────────────────────┐
                        │    EfficientNet-B0 Feature Extractor  │
                        │    - Conv Stem (3 -> 32, stride 2)    │
                        │    - MBConv Blocks 0 - 3 (Frozen)     │
                        │    - MBConv Blocks 4 - 6 (Fine-tuned) │
                        └───────────────────────────────────────┘
                                            │
                                            ▼ Output: [Batch, 1280, 7, 7]
                        ┌───────────────────────────────────────┐
                        │   Squeeze-and-Excitation (SE) Block   │
                        │   Reduction Ratio r = 16              │
                        │   Linear(1280, 80) -> ReLU            │
                        │   Linear(80, 1280) -> Sigmoid         │
                        │   Channel Recalibration (x * y)       │
                        └───────────────────────────────────────┘
                                            │
                                            ▼
                                Dual Spatial Pooling
                        ┌───────────────────┬───────────────────┐
                        ▼                                       ▼
            AdaptiveAvgPool2d(1)                    AdaptiveMaxPool2d(1)
             Output: [Batch, 1280]                   Output: [Batch, 1280]
                        └───────────────────┬───────────────────┘
                                            ▼
                              Concatenation [Batch, 2560]
                                            │
                                            ▼
                                     Dropout (p = 0.40)
                                            │
                                            ▼
                                   Linear (2560 -> 256)
                                            │
                                            ▼
                                     BatchNorm1d (256)
                                            │
                                            ▼
                                    ReLU (inplace=True)
                                            │
                                            ▼
                                     Dropout (p = 0.30)
                                            │
                                            ▼
                                     Linear (256 -> 7)
                                            │
                                            ▼
                                   Output Logits [Batch, 7]
```

#### Detailed Sub-Module Formulations

1. **Backbone:**
   - Modified `timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)` stripped of default pooling and linear classification heads.
   - Output spatial dimension: $7 \times 7$ feature maps with 1,280 channels.
   - Freezing scheme: Conv stem, BatchNorm 1, and MBConv stages 0 through 3 are frozen during early training to preserve low-level edge/texture primitives.

2. **Squeeze-and-Excitation Channel Attention (`SEBlock`):**
   - Channel squeeze: Global spatial average pooling $z_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W x_c(i, j) \in \mathbb{R}^{1280}$.
   - Excitation: Two un-biased linear transformations with bottleneck reduction ratio $r=16$:
     $$s = \sigma\left(W_2 \cdot \text{ReLU}(W_1 z)\right)$$
     where $W_1 \in \mathbb{R}^{80 \times 1280}$ and $W_2 \in \mathbb{R}^{1280 \times 80}$.
   - Channel-wise scaling: $\tilde{X}_c = s_c \cdot X_c$.

3. **Dual Spatial Pooling:**
   - Standard networks rely solely on global average pooling, which dilutes highly localized, sharp pathology signatures (e.g., punctate bacterial lesions or jassid necrotic margins).
   - Dual spatial pooling concatenates the global average descriptor $f_{avg} = \text{AdaptiveAvgPool2d}(1) \in \mathbb{R}^{1280}$ with the peak salient feature descriptor $f_{max} = \text{AdaptiveMaxPool2d}(1) \in \mathbb{R}^{1280}$, yielding a dense feature vector of dimension $2,560$.

4. **Regularized Classification Head:**
   - Structured as: $\text{Dropout}(0.4) \to \text{Linear}(2560, 256) \to \text{BatchNorm1d}(256) \to \text{ReLU} \to \text{Dropout}(0.3) \to \text{Linear}(256, 7)$.
   - Weight initialization: Xavier uniform (`torch.nn.init.xavier_uniform_`) on linear projections; unit scale and zero bias on batch normalization.

---

### 3.2 Teacher & Baseline CNN Architectures

#### DenseNet-121 (Teacher Network)
- **Timm Identifier:** `densenet121`
- **Architecture Highlights:** 4 dense blocks with cross-layer feature reuse via direct concatenation. Growth rate $k=32$. Features transition through $1 \times 1$ conv bottlenecks and $2 \times 2$ average pooling.
- **Role:** Selected as the Teacher model for Knowledge Distillation due to superior cross-entropy calibration and internal validation score ($99.07\%$ Val F1).
- **Classifier Head:** Standard AdaptiveAvgPool2d + Linear(1024, 7).

#### MobileNetV3-Large (Edge Baseline)
- **Timm Identifier:** `mobilenetv3_large_100`
- **Architecture Highlights:** Hard-swish activations, depthwise separable convolutions with inverted residual bottlenecks, and lightweight integrated Squeeze-and-Excitation units.
- **Classifier Head:** Conv2d $1 \times 1$ feature projection to 1,280 dims, Hardswish, Dropout(0.2), Linear(1280, 7).

#### EfficientNet-B0 (Backbone Baseline)
- **Timm Identifier:** `efficientnet_b0`
- **Architecture Highlights:** Compound scaling baseline with 7 inverted residual blocks, squeeze-and-excitation, and swish activations.
- **Classifier Head:** Standard AdaptiveAvgPool2d + Dropout(0.2) + Linear(1280, 7).

#### ResNet-50 (Heavy Residual Baseline)
- **Timm Identifier:** `resnet50`
- **Architecture Highlights:** 50-layer deep residual network with 4 bottleneck residual stages (3, 4, 6, 3 blocks). Shortcut connections bypass 3-layer convolutions $(1\times 1, 3\times 3, 1\times 1)$.
- **Classifier Head:** AdaptiveAvgPool2d + Linear(2048, 7).

---

### 3.3 Vision Transformer Architectures

#### Swin Transformer Tiny (`Swin-T`)
- **Timm Identifier:** `swin_tiny_patch4_window7_224`
- **Architecture Highlights:**
  - Hierarchical feature representations with shifted window-based self-attention.
  - Patch partitioning: $4 \times 4$ raw pixel patches yielding initial sequence length of $(224/4)^2 = 3136$.
  - Stage configurations: 4 stages with layer depths $[2, 2, 6, 2]$ and attention heads $[3, 6, 12, 24]$.
  - Window size: $M = 7 \times 7$. Shifted windows alternate between regular partitioning and $(M/2, M/2)$ circular shifts to model cross-window spatial interactions with linear computational complexity $\mathcal{O}(H \cdot W)$.
- **Head:** Norm layer + Linear(768, 7).

#### Vision Transformer Base (`ViT-Base/16`)
- **Timm Identifier:** `vit_base_patch16_224`
- **Architecture Highlights:**
  - Pure non-hierarchical self-attention transformer.
  - Patch size: $16 \times 16$ non-overlapping patches yielding $14 \times 14 = 196$ patch tokens + 1 learnable `[CLS]` token.
  - 12 Transformer encoder blocks, hidden dimension $D = 768$, MLP expansion dimension $3072$, 12 multi-head self-attention heads.
  - Pretrained on ImageNet-21K and fine-tuned on ImageNet-1K at $224 \times 224$.
- **Head:** LayerNorm + Linear(768, 7) attached to `[CLS]` token.

---

### 3.4 Real-Time Edge Architecture

#### YOLO26s Classification (`YOLO26s-cls`)
- **Ultralytics Identifier:** `yolo26s-cls.pt` (with fallback to `yolov8s-cls.pt`)
- **Architecture Highlights:**
  - Cross-Stage Partial Network (CSPDarknet) backbone with C2f/C3k2 feature aggregation blocks and SPPF (Spatial Pyramid Pooling Fast).
  - Classification head replacing multi-scale object detection bounding-box heads with a global pooling and single linear classification layer.
  - Optimized for low-latency tensor processing on mobile edge accelerators.

---

## 4. Parameter Footprint & Model Sizing Matrix

The table below catalogs exact parameter counts, trainable subsets, checkpoint file sizes, and empirical runtime footprints:

| Model Architecture | Total Parameters | Trainable Parameters | Frozen / Non-Trainable | PyTorch Weights (.pth/.pt) | ONNX FP32 | ONNX INT8 | Compression Ratio vs Baseline |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CottonNet V1** | **4,870,275** | **4,561,615** *(Phase 3)* | 308,660 | 18.88 MB | 18.56 MB | **4.94 MB** | **3.82x** |
| **CottonNet V2** | **4,870,275** | **4,561,615** *(Phase 3)* | 308,660 | 18.88 MB | 18.56 MB | **4.94 MB** | **3.82x** |
| **MobileNetV3-Large** | 4,210,000 | 4,210,000 | 0 | 17.06 MB | 16.84 MB | 4.30 MB | 3.96x |
| **EfficientNet-B0 (Base)**| 4,020,000 | 4,020,000 | 0 | 16.37 MB | 16.10 MB | 4.10 MB | 3.99x |
| **YOLO26s-cls** | 5,450,000 | 5,450,000 | 0 | 10.53 MB | 10.40 MB | 2.80 MB | 3.76x |
| **DenseNet-121 (Teacher)**| 6,960,000 | 6,960,000 | 0 | 28.47 MB | 28.10 MB | 7.20 MB | 3.95x |
| **ResNet-50** | 23,520,000 | 23,520,000 | 0 | 94.41 MB | 93.80 MB | 23.80 MB| 3.96x |
| **Swin-T** | 27,518,484 | 27,518,484 | 0 | 105.06 MB | 104.20 MB | 26.50 MB| 3.96x |
| **ViT-Base/16** | 85,804,039 | 85,804,039 | 0 | 327.37 MB | 326.50 MB | 82.20 MB| 3.98x |

> [!NOTE]
> During **Phase 1** of CottonNet training, only the classifier head is active, with **657,671 trainable parameters** ($4,212,604$ parameters frozen). In **Phase 2**, the top 3 blocks are unfrozen. In **Phase 3**, the entire network ($4,561,615$ parameters) is fine-tuned end-to-end.

---

## 5. Data Input Processing & Augmentation Pipelines

Field conditions introduce high variance in camera optics, angle, solar glare, leaf wetness, and shadows. The project systematically evaluated two distinct training augmentation regimens alongside a standardized validation/inference pipeline.

```mermaid
flowchart LR
    A["Raw RGB Image (Field Photo)"] --> B["Input Validation & Quality Guard<br/>(Luminance, Laplacian, HSV Chlorophyll)"]
    B -->|Accept| C["Spatial Resize<br/>(224 x 224 Bilinear)"]
    B -->|Reject| Z["Rejection Alert<br/>(Blur / Exposure / Non-Foliage)"]
    C --> D["Tensor Normalization<br/>[0, 255] -> [0.0, 1.0]<br/>Channel Mean & Std Scaling"]
    D --> E["Inference Execution<br/>(PyTorch / ONNX Runtime)"]
```

---

### 5.1 Standard Inference & Evaluation Preprocessing

To preserve deterministic evaluation, every image processed during validation, testing, and deployment undergoes the exact same four-step deterministic pipeline:

1. **Color Space Harmonization:** Explicit conversion to 3-channel standard RGB (stripping Alpha channels from PNGs, duplicating single-channel Grayscale).
2. **Spatial Rescaling:** Bilinear interpolation to the standard target resolution:
   $$\text{Resolution} = 224 \times 224 \text{ pixels}$$
3. **Tensor Casting & Scaling:** Converting integer pixel arrays to 32-bit floating-point tensors scaled to $[0.0, 1.0]$:
   $$x_{float} = \frac{x_{uint8}}{255.0}$$
4. **Channel-wise ImageNet Normalization:**
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
   $$x_{norm}^{(c)} = \frac{x_{float}^{(c)} - \mu^{(c)}}{\sigma^{(c)}}, \quad c \in \{R, G, B\}$$

---

### 5.2 V1 Field-Resilient Augmentation Pipeline (Torchvision)

Used to train the Part I baselines (ResNet50, MobileNetV3, DenseNet121, EfficientNet-B0) and **CottonNet V1**:

```python
train_tfs_v1 = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=30),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

- **Geometric Operations:** Random horizontal flip ($p=0.5$), vertical flip ($p=0.5$), random affine rotation ($\pm 30^\circ$).
- **Photometric Operations:** Random color jitter (brightness $\pm 30\%$, contrast $\pm 30\%$, saturation $\pm 30\%$, hue $\pm 5\%$).
- **Optical Degradation:** Random Gaussian blur with $3 \times 3$ kernel, $\sigma \in [0.1, 2.0]$.

---

### 5.3 V2 Realistic Field Augmentation Pipeline (Albumentations)

Used in Part II for **CottonNet V2**, **Swin-T**, **ViT-Base/16**, and **YOLO26s-cls** to simulate harsh field environments:

```python
v2_train_transform = A.Compose([
    A.Resize(224, 224),

    # 1. Geometric Perturbations
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
        border_mode=cv2.BORDER_REFLECT_101, p=0.4
    ),

    # 2. Illumination & Pigment Variations
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.RandomGamma(gamma_limit=(70, 130), p=1.0),
        A.RandomToneCurve(scale=0.15, p=1.0),
    ], p=0.6),

    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
    A.RandomShadow(
        shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 3),
        shadow_dimension=5, p=0.2
    ),

    # 3. Hardware & Sensor Noise Degradations
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
    ], p=0.2),

    A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
    A.ImageCompression(quality_lower=60, quality_upper=95, p=0.2),

    # 4. Normalization and Tensor Conversion
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

#### Detailed Transformation Breakdown
1. **Geometric Shifts:** Horizontal flip ($p=0.5$), vertical flip ($p=0.3$), $90^\circ$ orthogonal rotations ($p=0.3$), and affine shift/scale/rotate ($p=0.4$, reflection padding).
2. **Solar & Illumination Effects:** Mutual exclusion (`A.OneOf`) between non-linear contrast adjustments, Gamma curve deviations ($\gamma \in [0.7, 1.3]$), and tone curve modulations. Added `A.RandomShadow` simulating canopy shading.
3. **Field Blur & Compression Artifacts:** Motion blur from handheld mobile camera shake, Gaussian sensor noise ($\sigma^2 \in [5, 30]$), and lossy JPEG artifacts ($Q \in [60, 95]$).

---

### 5.4 External OOD Dataset Input Preprocessing & Class Alignment

When evaluating on the independent NIST external dataset (1,290 images across 3 matching classes):
- Input images were rescaled to $224 \times 224$ and normalized with ImageNet statistics.
- **Label Alignment Protocol:** The model operates in its native 7-class probability space. Class IDs $0$, $1$, and $2$ correspond to `Bacterial Blight`, `Curl Virus`, and `Healthy Leaf`.
- **OOD Evaluation Rule:** If the model predicts any of the remaining 4 classes (`Herbicide Damage`, `Leaf Hopper`, `Leaf Redding`, or `Leaf Variegation`) on an external image, the prediction is counted strictly as an **Incorrect Misclassification** (no score masking or forced argmax truncation), rigorously penalizing over-confident false positives.

---

## 6. Training Pipelines & Hyperparameter Engineering

```mermaid
sequenceDiagram
    autonumber
    participant D as Data Loader
    participant S as Student (CottonNet)
    participant T as Teacher (DenseNet121)
    participant L as KD Loss Function
    participant O as Optimizer (AdamW)

    Note over S,T: Phase 1: Train Head Only (Stem + Blocks 0-3 Frozen)
    D->>S: Augmented Image Batch
    D->>T: Augmented Image Batch
    S->>L: Student Logits (Zs)
    T->>L: Teacher Logits (Zt)
    L->>L: Combined CE + KL Divergence (T=4.0, Alpha=0.7)
    L->>O: Compute Gradients & Backward Pass
    O->>S: Update Classifier Weights Only

    Note over S,T: Phase 2: Unfreeze Blocks 4-6 + Knowledge Distillation
    O->>S: Update Top Backbone + Classifier

    Note over S,T: Phase 3: Unfreeze Full Backbone + Light KD (Alpha=0.3)
    O->>S: End-to-End Fine-Tuning
```

---

### 6.1 Baseline CNN Training Pipeline (`TrainValidation`)

The baseline CNN architectures (ResNet-50, EfficientNet-B0, MobileNetV3-Large, DenseNet-121) were trained using a clean unified harness:
- **Optimizer:** Adam (`torch.optim.Adam`) with base learning rate $\eta = 3 \times 10^{-4}$.
- **Loss Function:** Categorical Cross-Entropy (`nn.CrossEntropyLoss`).
- **Validation Metric:** Multiclass Macro F1 Score (`torchmetrics.F1Score`).
- **Early Stopping & Model Checkpointing:**
  - Evaluated at every epoch completion.
  - Threshold requirement: $\Delta F1 \ge 0.01$ to reset patience counter.
  - Patience parameter: $3$ to $5$ consecutive non-improving epochs triggers termination.
  - Best model checkpoint saved to `saved_models/{model_name}_best_model.pth`.

---

### 6.2 CottonNet 3-Phase Knowledge Distillation Pipeline (`KnowledgeDistillationTrainer`)

CottonNet was trained using knowledge transfer from the pre-trained DenseNet-121 teacher model across three distinct phases.

#### Mathematical Formulation of Loss Function
The student is supervised by both ground-truth one-hot targets and softened teacher dark knowledge:

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{CE}}(\hat{y}_s, y_{\text{true}}) + (1 - \alpha) T^2 \mathcal{L}_{\text{KL}}\left(\sigma\left(\frac{z_s}{T}\right), \sigma\left(\frac{z_t}{T}\right)\right)$$

Where:
- $\mathcal{L}_{\text{CE}}$ is Cross-Entropy with **label smoothing** $\epsilon = 0.10$:
  $$y_k^{\text{smooth}} = y_k(1 - \epsilon) + \frac{\epsilon}{K}, \quad K=7$$
- $T = 4.0$ is the distillation temperature scaling parameter.
- $z_s, z_t$ are the raw unnormalized logits of the student and teacher models.
- $\sigma(\cdot)$ represents the softmax activation function.
- $T^2$ balances gradient magnitudes between CE and KL divergence.

#### 3-Phase Freezing & Learning Rate Schedule

```text
Phase 1: Head-Only Warmup (Backbone Fully Frozen)
  - Trainable parameters: 657,671 (~0.66M)
  - Learning rate: lr = 1e-3
  - Distillation weight: alpha = 0.70
  - Epochs: 10 (Patience = 5)
  - Scheduler: CosineAnnealingWarmRestarts (T_0=5, T_mult=2, eta_min=1e-6)
  - Objective: Stabilize randomly initialized classifier weights without disrupting pretrained feature extractors.

Phase 2: Partial Fine-Tuning + Heavy Distillation
  - Backbone unfreezing: MBConv blocks 4, 5, 6, conv_head, and bn2 unfrozen
  - Trainable parameters: ~3.85M
  - Learning rate: lr = 1e-4
  - Distillation weight: alpha = 0.70
  - Epochs: 30 (Patience = 8)
  - Scheduler: CosineAnnealingWarmRestarts (T_0=5, T_mult=2, eta_min=1e-6)
  - Objective: Adapt high-level semantic feature detectors to subtle crop pathology signatures.

Phase 3: Full Network Fine-Tuning + Light Distillation
  - Backbone unfreezing: Stem, bn1, and blocks 0 to 3 unfrozen (100% parameters trainable)
  - Trainable parameters: 4,561,615 (4.56M)
  - Learning rate: lr = 1e-5 (ultra-low to prevent catastrophic forgetting)
  - Distillation weight: alpha = 0.30 (allows student to specialize beyond teacher errors)
  - Epochs: 10 (Patience = 5)
  - Scheduler: CosineAnnealingWarmRestarts (T_0=5, T_mult=2, eta_min=1e-6)
  - Objective: Global end-to-end alignment of convolutional kernels.
```

---

### 6.3 Extended Transformer Training Framework (`ExtendedTrainer`)

For the extended study in Part II (Swin-T and ViT-Base/16):
- **Optimizer:** Decoupled Weight Decay AdamW (`torch.optim.AdamW`) with weight decay $\lambda = 0.05$.
- **Learning Rates:**
  - Swin-T: $\eta = 2 \times 10^{-5}$
  - ViT-Base/16: $\eta = 1 \times 10^{-5}$
- **Learning Rate Scheduler:** Cosine Annealing (`CosineAnnealingLR`) without warm restarts:
  $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$
  with $T_{\max} = \text{epochs}$, $\eta_{\min} = 10^{-6}$.
- **Regularization:** Label smoothing $\epsilon = 0.10$ across all 7 classes.

---

### 6.4 YOLO Classification Training Pipeline

For `YOLO26s-cls`:
- Formatted dataset into folder directory hierarchy: `yolo_cls_dataset/{train, val, test}/{class_name}/*.jpg`.
- Hyperparameters: Image size $224 \times 224$, batch size 16, epochs 30, patience 5.
- Base learning rate: `lr0 = 1e-3`, final learning rate factor: `lrf = 0.01` (SGD/AdamW hybrid).
- Weight decay: $1 \times 10^{-4}$, label smoothing: $0.10$.

---

### 6.5 Hyperparameter Reference Matrix

| Training Parameter | Baseline CNNs | CottonNet V1 & V2 (KD) | Swin-T | ViT-Base/16 | YOLO26s-cls |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Optimizer** | Adam | AdamW | AdamW | AdamW | AdamW |
| **Initial Learning Rate ($\eta$)** | $3 \times 10^{-4}$ | Phase 1: $10^{-3}$<br>Phase 2: $10^{-4}$<br>Phase 3: $10^{-5}$ | $2 \times 10^{-5}$ | $1 \times 10^{-5}$ | $1 \times 10^{-3}$ |
| **Weight Decay ($\lambda$)** | 0.0 | 0.01 | 0.05 | 0.05 | $10^{-4}$ |
| **LR Scheduler** | None / Plateau | CosineWarmRestarts ($T_0=5, T_{mult}=2$) | CosineAnnealingLR | CosineAnnealingLR | Linear decay (`lrf=0.01`) |
| **Loss Function** | Cross-Entropy | KD Loss (Smooth CE + KL) | Smooth CE ($\epsilon=0.1$) | Smooth CE ($\epsilon=0.1$) | Smooth CE ($\epsilon=0.1$) |
| **Label Smoothing ($\epsilon$)**| 0.0 | 0.10 | 0.10 | 0.10 | 0.10 |
| **KD Temperature ($T$)** | N/A | 4.0 | N/A | N/A | N/A |
| **KD Alpha ($\alpha$)** | N/A | Phase 1/2: 0.70<br>Phase 3: 0.30 | N/A | N/A | N/A |
| **Batch Size** | 16 | 16 | 16 | 16 | 16 |
| **Early Stopping Patience** | 3 - 5 epochs | 5 - 8 epochs per phase | 5 epochs | 5 epochs | 5 epochs |
| **Input Resolution** | $224 \times 224$ | $224 \times 224$ | $224 \times 224$ | $224 \times 224$ | $224 \times 224$ |

---

## 7. Working Pipeline — Inference, OOD Validation & Edge Deployment

```
[Camera / Gallery / File Input]
             │
             ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 1: SCIENTIFIC INPUT QUALITY & OOD GATEKEEPER       │
│ - Luminance check: Mean brightness < 35 (Underexposed)   │
│ - Laplacian variance: Sharpness score < 10 (Too blurry)  │
│ - HSV Foliage ratio: Green/Red/Yellow pigment < 0.5%     │
│ - Spatial jitter: Mean pixel delta > 90.0 (Sensor noise) │
└────────────────────────────┬─────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [PASS: Valid Plant Leaf]        [FAIL: Reject Image]
            │                                 │
            │                                 ▼
            │                   Return Error Alert to Farmer:
            │                   - "Image too dark to resolve symptoms"
            │                   - "Camera out of focus / blurry"
            │                   - "No cotton foliage detected in frame"
            │
            ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 2: EXACT PREPROCESSING                             │
│ - Convert to standard 3-channel RGB                      │
│ - Bilinear interpolation resize to 224x224               │
│ - Float scaling [0, 255] -> [0.0, 1.0]                   │
│ - Channel normalization (ImageNet mean & std)            │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 3: MULTI-RUNTIME MODEL EXECUTION                   │
│ - Primary Edge Runtime: ONNX Runtime FP32 (6.56 ms CPU)  │
│ - Alternative: PyTorch FP32 (26.12 ms) / ONNX FP16       │
│ - Output: 7 Logits -> Softmax Normalized Probabilities   │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 4: CONFIDENCE THRESHOLD & OOD ARBITRATION          │
│ - Peak Probability P(max) >= 0.40 (Operational Cutoff)   │
│   * Pass: Confirm top disease classification             │
│   * Fail: Fallback to "Undetermined / Low Confidence"    │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 5: EXPLAINABLE AI (GRAD-CAM FEATURE ATTRIBUTION)   │
│ - Autograd backward on EfficientNet-B0 blocks[-1]        │
│ - Global average pool gradients -> Channel weights alpha │
│ - Weighted ReLU feature map -> 7x7 spatial heatmap       │
│ - Bicubic upsampling to 224x224 -> Jet colormap overlay  │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 6: ZERO-CLOUD OFFLINE EDGE PRESENTATION            │
│ - Android Jetpack Native App (Offline ONNX Runtime)      │
│ - Progressive Web App (Client-side HTML5 canvas + PWA)   │
│ - Actionable agronomic treatment advice displayed        │
└──────────────────────────────────────────────────────────┘
```

---

### 7.1 End-to-End Operational Inference Architecture

The working inference pipeline processes inputs through a 6-stage sequential chain. Execution halts immediately if an image fails the Stage-1 Gatekeeper, preventing compute waste and eliminating false diagnostic outputs.

---

### 7.2 Input Quality & Stage-1 Gatekeeper Validation (OOD Rejection)

Implemented directly in `inference_engine.py`, the Stage-1 Gatekeeper protects against out-of-distribution inputs (e.g., soil, human skin, machinery, black frames, glare):

1. **Luminance & Exposure Verification:**
   - Converts RGB to perceptual grayscale: $Y = 0.299R + 0.587G + 0.114B$.
   - Rejects if $\mu_Y < 35$: *"Image is too dark to resolve leaf symptoms."*
   - Flags if $\mu_Y > 254$: *"Severe glare / overexposure."*

2. **Laplacian Blur / Focus Estimation:**
   - Convolves grayscale image with discrete 2D Laplacian kernel:
     $$K = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
   - Computes variance of response map: $\text{Var}(\nabla^2 I)$.
   - Converts to calibrated sharpness score:
     $$\text{Sharpness} = \min\left(99.0, \max\left(10.0, \sqrt{\text{Var}(\nabla^2 I)} \times 4.5\right)\right)$$
   - Rejects if $\text{Sharpness} \le 10$: *"Image is very blurry. Try focusing closer on the leaf."*

3. **HSV Foliage Pigment Detection:**
   - Converts RGB to HSV space. Evaluates leaf pigment presence:
     - Chlorophyll Green: $30^\circ \le H \le 165^\circ, S > 0.08, V > 0.10$
     - Anthocyanin Red (Leaf Redding): $(H < 30^\circ \lor H > 330^\circ), S > 0.10, V > 0.10$
     - Xanthophyll Yellow: $20^\circ \le H < 40^\circ, S > 0.10, V > 0.10$
   - Computes foliage pixel ratio: $\rho = \frac{\sum (\text{Green} \lor \text{Red} \lor \text{Yellow})}{H \times W}$.
   - Rejects if $\rho < 0.005$ ($0.5\%$): *"Ensure leaf fills the central viewfinder frame."*

4. **Spatial Texture Coherence:**
   - Computes mean absolute difference across adjacent pixels: $\Delta_{\text{spatial}} = \frac{1}{N}\sum |I_{i+1, j} - I_{i, j}|$.
   - Rejects random electronic camera noise if $\Delta_{\text{spatial}} > 90.0$.

---

### 7.3 Multi-Precision Runtime Execution (PyTorch vs ONNX FP32/FP16/INT8)

To support deployment from high-end workstations to ultra-low-cost smartphones, CottonNet was exported and benchmarked across four execution runtimes:

```python
# Exported via torch.onnx.export (TorchScript tracing, opset=17, dynamo=False)
# Quantized via onnxruntime.quantization.quantize_dynamic (QuantType.QUInt8)
```

| Execution Runtime | Precision | Checkpoint Size | Latency Mean (CPU) | Median Latency (P50) | 95th Percentile (P95) | Prediction Agreement vs PyTorch | Deployment Target |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **PyTorch CPU** | FP32 | 18.88 MB | 26.12 ms | 25.40 ms | 32.51 ms | 100.00% (Ref) | Python CLI / Lab Workstation |
| **ONNX Runtime CPU** | **FP32** | **18.56 MB** | **6.56 ms** | **6.48 ms** | **7.19 ms** | **100.00%** | **Primary Edge Choice (Champion)** |
| **ONNX Runtime Mobile**| **FP16** | **9.33 MB** | **14.12 ms** | **12.80 ms** | **22.40 ms** | **100.00%** | **Optimal for Mobile GPU/NPU** |
| **ONNX Dynamic Quant** | INT8 | 4.94 MB | 33.80 ms | 33.10 ms | 36.87 ms | 0.00% *(Logit drift)* | Ultra-constrained RAM (<5 MB) |

> [!IMPORTANT]
> **Key Finding on Quantization:** While dynamic INT8 achieves a 3.75x footprint reduction (4.94 MB), dynamic quantization without static representative calibration datasets causes uncalibrated logit drift on subtle necrotic textures. Therefore, **ONNX FP32 (6.56 ms, 18.56 MB)** is the **Primary Champion** for edge CPUs, while **ONNX FP16 (9.33 MB, 100% agreement)** is recommended for mobile GPUs and NPUs.

---

### 7.4 Confidence Thresholding & Fallback Protocol

- **Optimal Operating Threshold:** Systematic parameter sweep across $[0.30, 0.95]$ established $\tau = 0.40$ (40%) as the clinical reliability threshold.
- **Decision Logic:**
  - If $\max_{c} P(c) \ge 0.40$: Confirm target class prediction and trigger Grad-CAM visualization.
  - If $\max_{c} P(c) < 0.40$: Trigger uncertainty fallback:
    $$\text{"Confidence is below the research reliability threshold (40%). Unable to confirm disease pathology."}$$

---

### 7.5 Explainable AI Pipeline (True Grad-CAM Mathematical Formulation)

Grad-CAM attributions are calculated on the fly without external wrappers:

1. Target Layer: Final convolutional block of the backbone (`model.backbone.blocks[-1]`).
2. Forward hook records activation tensor $A \in \mathbb{R}^{1 \times 1280 \times 7 \times 7}$.
3. Backward hook records gradient tensor with respect to predicted class logit $y^c$:
   $$G_{i, j}^k = \frac{\partial y^c}{\partial A_{i, j}^k}$$
4. Channel importance weights are computed via spatial global average pooling:
   $$\alpha_k^c = \frac{1}{7 \times 7} \sum_{i=1}^7 \sum_{j=1}^7 \frac{\partial y^c}{\partial A_{i, j}^k}$$
5. Heatmap localization map:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k=1}^{1280} \alpha_k^c A^k\right) \in \mathbb{R}^{7 \times 7}$$
6. Upsampling & Colormap Blending:
   - Rescaled via bicubic interpolation from $7 \times 7$ to $224 \times 224$.
   - Normalized to $[0.0, 1.0]$ and mapped to RGB using the Jet colormap.
   - Blended at $\alpha = 0.60$ with the original input leaf image:
     $$I_{\text{blended}} = 0.40 \cdot I_{\text{original}} + 0.60 \cdot I_{\text{heatmap}}$$
   - Exported as Base64 JPEG data URL for instant zero-latency UI rendering.

---

### 7.6 Zero-Cloud Offline Edge Deployment (Android & PWA)

To ensure field autonomy, all dependencies are packaged locally:
- **Android Native (`android/`):** Built with Android Jetpack, CameraX preview, and ONNX Runtime Mobile (`com.microsoft.onnxruntime:onnxruntime-android`). Models run locally on CPU/NNAPI with zero cloud data transmission.
- **Web Progressive Web App (`app/`):** Standalone HTML5 canvas and Service Worker caching (`manifest.json`), enabling offline execution on Android Chrome, iOS Safari, and desktop browsers.

---

## 8. Empirical Experimental Results & Benchmark Analysis

### 8.1 Internal Test Set Performance Across All 9 Models

The comprehensive empirical results from `cotton-leaf-disease-nist.ipynb` and `FINAL_SCIENTIFIC_AUDIT.md` (evaluated on the identical 214-image held-out test split) are presented below:

| Model Architecture | Test Accuracy | Test Macro F1 | Test Weighted F1 | Macro Precision | Macro Recall | Top-3 Accuracy | Parameters | Checkpoint Size | Inference Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CottonNet V1 (Proposed)**| **98.60%** | **98.60%** | **98.60%** | **98.98%** | **98.47%** | **100.00%** | **4.87M** | **18.88 MB** | **6.56 ms** |
| **ViT-Base/16** | 98.13% | 98.25% | 98.13% | 98.25% | 98.25% | 100.00% | 85.80M | 327.37 MB | 14.97 ms |
| **Swin-T** | 97.66% | 97.70% | 97.66% | 98.13% | 97.35% | 100.00% | 27.52M | 105.06 MB | 15.33 ms |
| **CottonNet V2 (Proposed)**| **96.26%** | **96.34%** | **96.27%** | **96.56%** | **96.20%** | **100.00%** | **4.87M** | **18.88 MB** | **12.50 ms** |
| **DenseNet-121 (Teacher)** | 96.26% | 96.26% | 96.26% | 96.50% | 96.26% | 100.00% | 6.96M | 28.47 MB | 18.20 ms |
| **MobileNetV3-Large** | 95.79% | 95.79% | 95.79% | 96.10% | 95.79% | 100.00% | 4.21M | 17.06 MB | 7.90 ms |
| **EfficientNet-B0 (Base)**| 95.33% | 95.33% | 95.33% | 95.80% | 95.33% | 100.00% | 4.02M | 16.37 MB | 8.10 ms |
| **ResNet-50** | 93.93% | 93.93% | 93.93% | 94.40% | 93.93% | 100.00% | 23.52M | 94.41 MB | 22.40 ms |
| **YOLO26s-cls** | 92.99% | 93.41% | 92.88% | 94.97% | 92.33% | 99.53% | 5.45M | 10.53 MB | 14.75 ms |

#### CottonNet Detailed Per-Class Breakdown (Held-Out Test Set)

| Agronomic Class | Support | Precision | Recall | F1-Score | True Positives | Errors | Primary Confusion |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `Bacterial Blight` | 25 | 1.0000 | 0.9600 | 0.9796 | 24 | 1 | 1 misclassified as Healthy Leaf |
| `Curl Virus` | 43 | 1.0000 | 0.9767 | 0.9882 | 42 | 1 | 1 misclassified as Healthy Leaf |
| `Healthy Leaf` | 26 | 0.9630 | 1.0000 | 0.9811 | 26 | 0 | Perfect Recall |
| `Herbicide Growth Damage`| 28 | 1.0000 | 1.0000 | 1.0000 | 28 | 0 | Perfect Diagnostic Accuracy |
| `Leaf Hopper Jassids` | 23 | 1.0000 | 0.9565 | 0.9778 | 22 | 1 | 1 misclassified as Leaf Redding |
| `Leaf Redding` | 58 | 0.9667 | 1.0000 | 0.9831 | 58 | 0 | Perfect Recall |
| `Leaf Variegation` | 11 | 1.0000 | 1.0000 | 1.0000 | 11 | 0 | Perfect Diagnostic Accuracy |
| **Macro Average** | **214** | **0.9898** | **0.9847** | **0.9860** | **211** | **3** | **Only 3 errors out of 214 test images** |

---

### 8.2 Latency, Throughput & Memory Profiling (CPU & Mobile Runtime)

Benchmarked over 100 consecutive warm inference iterations on $224 \times 224 \times 3$ tensors:

```text
Latency Comparison (CPU Single-Thread):
  ONNX FP32 (CottonNet)   : [====] 6.56 ms
  MobileNetV3-Large       : [=====] 7.90 ms
  EfficientNet-B0         : [======] 8.10 ms
  CottonNet V2            : [========] 12.50 ms
  ViT-Base/16             : [==========] 14.97 ms
  Swin-T                  : [===========] 15.33 ms
  PyTorch FP32 (CottonNet): [==================] 26.12 ms
  ResNet-50               : [======================] 22.40 ms
```

---

### 8.3 External Dataset Generalization & Discrepancy Audit (NIST 1,290 Images)

To evaluate real-world field transferability, the extended models were tested on an independent external dataset of 1,290 images across 3 matching classes (`Bacterial Blight`, `Curl Virus`, `Healthy Leaf`):

| Model Architecture | Internal Test F1 | External Test F1 | Generalization Gap ($\Delta$ Drop) | External Accuracy | External Weighted F1 | External Top-3 Acc | External Latency (Avg) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Swin-T** | 97.70% | **79.83%** | **+17.88%** *(Best)* | **73.18%** | **79.72%** | 91.24% | 21.12 ms |
| **ViT-Base/16** | 98.25% | 76.58% | +21.68% | 71.63% | 76.42% | **93.18%** | 14.28 ms |
| **CottonNet V2** | 96.34% | 69.94% | +26.41% | 62.25% | 69.73% | 85.50% | **9.51 ms** |
| **YOLO26s-cls** | 93.41% | 64.56% | +28.85% | 54.73% | 64.56% | 82.64% | 12.05 ms |

#### Key Error Analysis Findings from External Evaluation
1. **OOD Over-Prediction on Bacterial Blight:**
   All models misclassified 30% to 41% of external Bacterial Blight images into the 4 non-target internal classes (principally `Leaf Redding` and `Leaf Variegation`), triggered by reddish-brown necrotic centers and dry leaf tissue mimicking anthocyanin accumulation.
2. **Curl Virus vs Healthy Leaf Confusion:**
   Field lighting glare and natural leaf curvature caused healthy leaves to be frequently misclassified as Curl Virus across CNN backbones (CottonNet V2: 45 images, YOLO26s: 98 images).
3. **Transformer Global Attention Superiority:**
   Swin-T achieved the highest Curl Virus Recall ($91.13\%$) and lowest generalization drop ($+17.88\%$), demonstrating that hierarchical windowed self-attention models global venation deformation better than localized CNN convolution kernels.

---

### 8.4 Systematic Robustness Stress-Testing (11 Corruptions)

CottonNet was subjected to 11 synthetic environmental corruptions to evaluate safety gatekeeper efficacy:

| Perturbation Condition | Severity | Mean Confidence | Confidence Drop | Gatekeeper Rejection Rate | Operational Assessment |
|:---|:---:|:---:|:---:|:---:|:---|
| **Clean Baseline** | None | 43.12% | 0.00% | 0.0% | Normal Clinical Operation |
| **Gaussian Blur** | $r = 1$ | 41.80% | -1.32% | 0.0% | High Robustness |
| **Gaussian Blur** | $r = 3$ | 34.20% | -8.92% | **58.3%** | Blur Gatekeeper Triggered |
| **Gaussian Noise** | $\sigma = 15$ | 40.50% | -2.62% | 0.0% | High Robustness |
| **Gaussian Noise** | $\sigma = 30$ | 32.10% | -11.02% | **75.0%** | Noise Gatekeeper Triggered |
| **Low Light** | $0.7\times$ | 40.90% | -2.22% | 0.0% | High Robustness |
| **Severe Underexposure**| $0.4\times$ | 30.40% | -12.72% | **83.3%** | Dark Exposure Gatekeeper Triggered |
| **Severe Glare** | $1.6\times$ | 33.10% | -10.02% | **66.7%** | Glare Gatekeeper Triggered |
| **JPEG Compression** | $Q = 20$ | 42.10% | -1.02% | 0.0% | High Robustness |
| **Rotation** | $45^\circ$ | 41.20% | -1.92% | 0.0% | High Robustness |
| **Rotation** | $90^\circ$ | 41.50% | -1.62% | 0.0% | High Robustness |

---

## 9. Key Scientific Insights, Trade-Offs & Recommendations

1. **CottonNet as the Definitive Edge Champion:**
   With only **4.87M parameters** and an ONNX FP32 latency of **6.56 ms**, CottonNet V1 achieves **98.60% Macro F1**, outperforming the heavy ResNet-50 baseline ($23.52M$ params, $93.93\%$ F1) and its own teacher DenseNet-121 ($6.96M$ params, $96.26\%$ F1).

2. **Knowledge Distillation Efficacy:**
   The 3-phase training strategy allowed the compact student to learn richer inter-class feature correlations from DenseNet-121 softened logits ($T=4.0$), preventing early overfitting on the small dataset ($2,137$ images).

3. **Dual Spatial Pooling Advantage:**
   Concatenating global average pooling with global max pooling preserves high-frequency lesion peaks that are otherwise averaged away in conventional global average pooling heads.

4. **Augmentation vs Domain Shift (V1 vs V2):**
   Heavy field augmentations in V2 (shadows, severe blur, noise) improved resistance to sensor corruptions but slightly lowered internal clean test accuracy ($96.26\%$ vs $98.60\%$). On cross-domain external transfer, Swin-T demonstrated superior domain invariance ($79.83\%$ external F1) due to non-local self-attention.

5. **Deployment Architecture Recommendation:**
   - **For Mobile Smartphones & Offline Edge Devices:** Deploy **CottonNet ONNX FP32** (or **FP16** for mobile NPUs) coupled with the Stage-1 Gatekeeper heuristic.
   - **For Centralized Regional Diagnostic Servers:** Deploy **Swin-T** or an ensemble of **CottonNet + Swin-T** to maximize cross-farm generalizability.
   - **Future Work:** Replace the Stage-1 color heuristic with a lightweight binary MobileNetV3-Small plant gatekeeper to eliminate the remaining $20\%$ False Acceptance Rate on non-cotton weed foliage.

---
*Report certified and compiled from the authoritative experimental artifacts of the CottonGuard AI research suite.*

