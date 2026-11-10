# CottonNet Architecture Specification & Model Documentation

## 1. Executive Summary
**CottonNet** is a customized deep neural network designed for edge-deployable agronomic pathology diagnosis on cotton crops (*Gossypium hirsutum*). It was trained on the **SAR-CLD-2024** dataset (2,137 images across 7 classes) using Knowledge Distillation from a teacher DenseNet-121 model.

- **Total Parameters:** 4,870,275 (~4.87M)
- **Trainable Parameters:** 4,561,615 (~4.56M)
- **Held-out Test Macro F1:** 98.60%
- **Held-out Test Accuracy:** 98.60%
- **Model Checkpoint File:** `cottonnet_best_model.pth` (18.88 MB PyTorch / 4.94 MB INT8 ONNX)

---

## 2. Layer-by-Layer Architectural Breakdown

```text
Input Tensor [B, 3, 224, 224]
       │
       ▼
EfficientNet-B0 Feature Extractor (Blocks 0 to 6)
       │  Output shape: [B, 1280, 7, 7]
       ▼
Squeeze-and-Excitation (SE-Block) Attention (Reduction r=16)
       │  AdaptiveAvgPool2d(1) -> Linear(1280, 80) -> ReLU -> Linear(80, 1280) -> Sigmoid
       ▼
Dual Spatial Pooling
       ├───────────────┬───────────────┐
       ▼                               ▼
AdaptiveAvgPool2d(1)            AdaptiveMaxPool2d(1)
   [B, 1280]                       [B, 1280]
       └───────────────┬───────────────┘
                       ▼
             Concatenation [B, 2560]
                       │
                       ▼
             Dropout (p=0.40)
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
             Dropout (p=0.30)
                       │
                       ▼
             Linear (256 -> 7)
                       │
                       ▼
             Output Logits [B, 7]
```

---

## 3. Exact Training Preprocessing Pipeline

Every input image during evaluation and deployment **must** undergo identical transformations:

1. **RGB Format:** Ensure 3-channel standard RGB (convert grayscale or RGBA).
2. **Resize:** Bilinear interpolation to target resolution: `(224, 224)` pixels.
3. **Tensor Conversion:** Scale pixel intensities from $[0, 255]$ to $[0.0, 1.0]$.
4. **Channel Normalization:**
   $$\text{Mean} = [0.485, 0.456, 0.406], \quad \text{Std} = [0.229, 0.224, 0.225]$$
   $$x_{\text{norm}}^{(c)} = \frac{x^{(c)} - \mu^{(c)}}{\sigma^{(c)}}$$

---

## 4. Class-to-Index Mapping (SAR-CLD-2024)

| Index | Class Name | Clinical Condition | Primary Etiology / Causative Agent |
|:---:|:---|:---|:---|
| **0** | `Bacterial Blight` | Angular leaf spots, black arm | *Xanthomonas malvacearum* |
| **1** | `Curl Virus` | Upward cupping, vein thickening | Cotton Leaf Curl Virus (CLCuV) |
| **2** | `Healthy Leaf` | Asymptomatic control tissue | Normal physiological foliage |
| **3** | `Herbicide Growth Damage`| Strap-like leaf deformation | Abiotic agrochemical drift (2,4-D / Glyphosate) |
| **4** | `Leaf Hopper Jassids` | Hopper burn, necrotic margins | *Amrasca biguttula biguttula* |
| **5** | `Leaf Redding` | Red-purple pigmentation | Physiological Mg deficiency / cold stress |
| **6** | `Leaf Variegation` | Yellow-green mosaic patches | Nutritional micronutrient deficiency / genetic |

---

## 5. Explainable AI (Grad-CAM Attribution)

Grad-CAM attributions are computed from the final convolutional stage (`backbone.blocks[-1]`):
1. Compute the gradient of target class score $y^c$ with respect to feature maps $A^k$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
2. Generate the coarse localization map:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$
3. Upsample $7 \times 7$ feature activation to $224 \times 224$ and overlay with Jet colormap.
