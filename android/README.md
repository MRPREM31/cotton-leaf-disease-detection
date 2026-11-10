# CottonGuard AI - Android Native Project (Kotlin + Jetpack Compose)

This Android project runs **CottonNet** (4.87M parameters, 98.60% Test F1) fully offline using **ONNX Runtime Mobile**.

---

## 📁 Directory Structure

```text
android/
├── app/
│   ├── src/main/
│   │   ├── java/com/cottonguard/ai/
│   │   │   ├── MainActivity.kt          # Jetpack Compose UI & Viewfinder
│   │   │   └── ml/
│   │   │       └── CottonNetClassifier.kt # ONNX Runtime Mobile Classifier
│   │   ├── assets/
│   │   │   └── cottonnet_int8.onnx       # Quantized 4.9MB CottonNet Model
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts
└── README.md
```

---

## 🚀 How to Build & Run in Android Studio

1. **Export the ONNX Model**:
   Run the Python export script in the root directory:
   ```bash
   python export_onnx.py
   ```
2. **Copy the Exported ONNX to Android Assets**:
   ```bash
   mkdir -p android/app/src/main/assets
   cp models_exported/cottonnet_int8.onnx android/app/src/main/assets/
   ```
3. **Open in Android Studio**:
   - Open Android Studio -> `File` -> `Open...` -> Select the `android/` directory.
   - Wait for Gradle sync to finish.
   - Connect your Android device or start an emulator (Android 7.0+ / API 24+).
   - Press **Run** (Green Play button).

---

## 📊 Class Mapping

The model outputs 7 probabilities corresponding to:
1. `0: Bacterial Blight`
2. `1: Curl Virus`
3. `2: Healthy Leaf`
4. `3: Herbicide Growth Damage`
5. `4: Leaf Hopper Jassids`
6. `5: Leaf Redding`
7. `6: Leaf Variegation`
