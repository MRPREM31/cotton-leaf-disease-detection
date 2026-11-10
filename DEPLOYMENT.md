# CottonGuard AI - Edge & Mobile Deployment Guide

## 1. Overview
CottonGuard AI supports two offline execution targets:
1. **On-Device Android Native App:** Jetpack Compose + CameraX + ONNX Runtime Mobile (`onnxruntime-android:1.17.1`).
2. **On-Device Progressive Web Application (PWA):** WebAssembly ONNX Runtime (`onnxruntime-web`) / Python Local Inference Server (`server.py`).

---

## 2. Model Export Artifacts

| Model Format | Precision | File Size | Recommended Execution Target | CPU Latency (Avg) |
|:---|:---:|:---:|:---|:---:|
| PyTorch Checkpoint | FP32 | 18.88 MB | Research Server / Python CLI | ~26.1 ms |
| ONNX Mobile FP32 | FP32 | 18.56 MB | Desktop / Server / WebAssembly | ~6.6 ms |
| **ONNX Quantized** | **INT8** | **4.94 MB** | **Android APK / Ultra-light Edge** | **~33.8 ms** |

---

## 3. Android Integration Blueprint

### Step 1: Add Dependencies in `android/app/build.gradle.kts`
```kotlin
dependencies {
    // ONNX Runtime for offline edge neural inference
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.1")

    // CameraX for high-frame-rate leaf capture
    implementation("androidx.camera:camera-core:1.3.1")
    implementation("androidx.camera:camera-camera2:1.3.1")
    implementation("androidx.camera:camera-lifecycle:1.3.1")
    implementation("androidx.camera:camera-view:1.3.1")
}
```

### Step 2: Copy Quantized Model into Android Assets
```bash
mkdir -p android/app/src/main/assets
cp cottonnet_int8.onnx android/app/src/main/assets/cottonnet.onnx
```

### Step 3: Run Inference in Kotlin
```kotlin
val env = OrtEnvironment.getEnvironment()
val session = env.createSession(context.assets.open("cottonnet.onnx").readBytes())

// Preprocessing: 224x224 RGB FloatBuffer normalized with ImageNet mean/std
val tensor = OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, 224, 224))
val results = session.run(mapOf("input" to tensor))
val logits = (results[0].value as Array<FloatArray>)[0]

// Compute Softmax probabilities
val expScores = logits.map { kotlin.math.exp(it) }
val sumExp = expScores.sum()
val probabilities = expScores.map { it / sumExp }
```

---

## 4. Input Gatekeeper & OOD Safeguards

To prevent closed-set misclassifications on non-cotton objects (e.g. soil, hands, random objects):
1. **Luminance Exposure Threshold:** Mean grayscale intensity between $35$ and $230$.
2. **Laplacian Blur Rejection:** Variance of discrete Laplacian $> 25$.
3. **Foliage Hue Range:** Minimum $15\%$ frame coverage in chlorophyll green ($35^\circ \le \text{Hue} \le 160^\circ$) or anthocyanin red ($\text{Hue} \le 25^\circ$ or $\ge 340^\circ$).
4. **Spatial Jitter Coherence:** Mean adjacent gradient difference $< 55$ (filters out uniform random noise).
5. **Confidence Calibration Cutoff:** Rejects diagnosis when $\max P(\text{class}) < 0.65$.
