# Out-of-Distribution (OOD) & Input Validation Evaluation Report

## 1. Overview
CottonNet operates in an agricultural edge environment where non-leaf inputs (hands, soil, background, noise) frequently enter the camera field of view. The Stage-1 Input Gatekeeper screens images prior to neural disease classification.

## 2. Empirical OOD Rejection Results

| Test Scenario | Visual Signature | Foliage Coverage | Sharpness | Gatekeeper Decision | Technical Rejection Reason |
|:---|:---|:---:|:---:|:---:|:---|
| `pure_black_underexposed` | Luminance = 0 | 0.0% | 10.0% | **REJECTED (100%)** | *Image is too dark to resolve leaf symptoms.* |
| `pure_white_glare` | Luminance = 255 | 0.0% | 10.0% | **REJECTED (100%)** | *Image has severe glare or overexposure.* |
| `uniform_random_noise` | Jitter = 127.4 | 0.0% | 99.0% | **REJECTED (100%)** | *Unstructured noise or invalid camera input.* |
| `uniform_gray_surface` | Gray 128 | 0.0% | 10.0% | **REJECTED (100%)** | *Image is too blurry for reliable clinical diagnosis.* |
| `soil_brown_color` | Non-foliage hue | 0.0% | 10.0% | **REJECTED (100%)** | *Unable to detect cotton leaf.* |
| `blue_cloth_synthetic` | Non-plant hue | 0.0% | 10.0% | **REJECTED (100%)** | *Unable to detect cotton leaf.* |

## 3. Statistical Metrics
- **Synthetic OOD False Acceptance Rate (FAR):** **0.0%** (0/6 accepted)
- **OOD True Rejection Rate (TRR):** **100.0%** (6/6 rejected)
- **Primary Rejection Triggers:** Color manifold boundary check, spatial jitter coherence, Laplacian edge threshold.
