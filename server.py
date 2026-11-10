import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os
import io
import json
import base64
import time
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from inference_engine import CottonNetInferenceEngine

app = Flask(__name__, static_folder="app")
CORS(app)

print("Initializing CottonNet Inference Engine with real PyTorch weights...")
engine = CottonNetInferenceEngine(checkpoint_path="cottonnet_best_model.pth")
print("[OK] Engine loaded successfully and ready for real diagnostic requests!")

@app.route("/")
def index():
    return send_from_directory("app", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("app", path)

@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    """
    Receives real image and executes authentic CottonNet inference pipeline.
    """
    try:
        image = None
        
        # 1. Check if multipart file upload
        if "image" in request.files:
            file = request.files["image"]
            image = Image.open(file.stream).convert("RGB")
            
        # 2. Check if JSON with base64 payload
        elif request.is_json:
            data = request.get_json()
            if "image_base64" in data:
                b64_str = data["image_base64"]
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                img_bytes = base64.b64decode(b64_str)
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        if image is None:
            return jsonify({
                "success": False,
                "error": "No valid image provided in request. Send multipart/form-data 'image' or JSON 'image_base64'."
            }), 400

        # Execute real CottonNet inference
        result = engine.predict(image, compute_gradcam=True)
        return jsonify(result)

    except Exception as e:
        print(f"Error during inference: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal inference error: {str(e)}"
        }), 500

@app.route("/api/model-info", methods=["GET"])
def model_info():
    """
    Returns authentic research benchmark parameters & dataset specifications.
    """
    return jsonify({
        "model_name": "CottonNet",
        "backbone": "EfficientNet-B0 (Modified)",
        "attention": "Squeeze-and-Excitation (SE-Block)",
        "pooling": "Dual Pooling (AdaptiveAvgPool2d + AdaptiveMaxPool2d)",
        "parameters": {
            "total": 4870275,
            "trainable": 4561615,
            "total_millions": 4.87,
            "trainable_millions": 4.56
        },
        "dataset": "SAR-CLD-2024 (2,137 Cotton Leaf Images)",
        "test_performance": {
            "test_macro_f1": "98.60%",
            "test_accuracy": "98.60%",
            "val_test_generalization_gap": "-0.47%",
            "baseline_comparison": {
                "CottonNet": {"f1": "98.60%", "params": "4.87M"},
                "DenseNet121": {"f1": "96.26%", "params": "6.96M"},
                "MobileNetV3-Large": {"f1": "95.79%", "params": "4.21M"},
                "EfficientNet-B0": {"f1": "95.33%", "params": "4.02M"},
                "ResNet50": {"f1": "93.93%", "params": "23.52M"}
            }
        },
        "classes": CottonNetInferenceEngine.CLASS_MAPPING,
        "input_specs": {
            "input_shape": [1, 3, 224, 224],
            "normalization_mean": CottonNetInferenceEngine.MEAN,
            "normalization_std": CottonNetInferenceEngine.STD
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
