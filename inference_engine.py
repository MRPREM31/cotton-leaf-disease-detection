"""
CottonNet Real Inference Engine & Explainability Pipeline
Authoritative implementation with real PyTorch checkpoint loading,
exact preprocessing, input quality validation, OOD rejection, and true Grad-CAM.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
import json
import base64
import io
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ==================== 1. EXACT COTTONNET ARCHITECTURE ====================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CottonNet(nn.Module):
    """
    CottonNet Model Architecture
    - EfficientNet-B0 backbone (num_features = 1280)
    - Squeeze-and-Excitation channel attention
    - Dual pooling (AdaptiveAvgPool2d + AdaptiveMaxPool2d) -> 2560 dims
    - Regularized classification head: Linear(2560, 256) -> BatchNorm1d -> ReLU -> Linear(256, 7)
    """
    def __init__(self, num_classes=7, dropout_1=0.4, dropout_2=0.3, freeze_blocks=4):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features  # 1280
        
        self.se_attention = SEBlock(self.feature_dim, reduction=16)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_1),
            nn.Linear(self.feature_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.se_attention(features)
        
        avg_out = self.avg_pool(features).flatten(1)
        max_out = self.max_pool(features).flatten(1)
        pooled = torch.cat([avg_out, max_out], dim=1)
        
        out = self.classifier(pooled)
        return out

    def get_gradcam_target_layer(self):
        return self.backbone.blocks[-1]


# ==================== 2. INFERENCE ENGINE & OOD VALIDATOR ====================
class CottonNetInferenceEngine:
    # Exact class index mapping from SAR-CLD-2024 dataset
    CLASS_MAPPING = {
        0: "Bacterial Blight",
        1: "Curl Virus",
        2: "Healthy Leaf",
        3: "Herbicide Growth Damage",
        4: "Leaf Hopper Jassids",
        5: "Leaf Redding",
        6: "Leaf Variegation"
    }

    # Training ImageNet normalization
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    IM_SIZE = 224

    def __init__(self, checkpoint_path="cottonnet_best_model.pth", device=None, confidence_threshold=0.40):
        self.checkpoint_path = checkpoint_path
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CottonNet model
        self.model = CottonNet(num_classes=7)
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing normalization tensors
        self.mean_tensor = torch.tensor(self.MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        self.std_tensor = torch.tensor(self.STD, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)

        # Hook storage for Grad-CAM
        self.gradients = None
        self.activations = None
        self._register_gradcam_hooks()

    def _load_weights(self):
        try:
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            print(f"[OK] Loaded CottonNet weights from: {self.checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint {self.checkpoint_path}: {e}")
            raise e

    def _register_gradcam_hooks(self):
        target_layer = self.model.get_gradcam_target_layer()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    # ==================== QUALITY & OOD GATE ====================
    def check_quality_and_ood(self, pil_image):
        """
        Scientific multi-stage input validation:
        1. Lighting / Exposure Check (Mean luminance & dynamic range)
        2. Blur / Sharpness Check (Laplacian variance)
        3. Color Manifold Check (HSV leaf pigment detection: chlorophyll green & anthocyanin red)
        4. Texture Coherence (Differentiates genuine leaf lamina from uniform noise/objects)
        """
        img_rgb = pil_image.convert("RGB")
        np_img = np.array(img_rgb)
        h, w, _ = np_img.shape

        # 1. Luminance & Exposure Check
        gray = np.dot(np_img[..., :3], [0.299, 0.587, 0.114])
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))
        
        if mean_brightness < 35:
            lighting_status = "Poor (Too Dark / Underexposed)"
            is_light_valid = False
            rejection_sub = "Image is too dark to resolve leaf symptoms."
        elif mean_brightness > 254:
            lighting_status = "High Exposure"
            is_light_valid = True
            rejection_sub = None
        else:
            lighting_status = "Optimal"
            is_light_valid = True
            rejection_sub = None

        # 2. Laplacian Blur Estimation
        lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        pad_gray = np.pad(gray, 1, mode='edge')
        laplacian = (
            pad_gray[:-2, 1:-1] + pad_gray[2:, 1:-1] +
            pad_gray[1:-1, :-2] + pad_gray[1:-1, 2:] -
            4 * pad_gray[1:-1, 1:-1]
        )
        lap_var = float(np.var(laplacian))
        sharpness_score = min(99.0, max(10.0, float(np.sqrt(lap_var) * 4.5)))
        sharpness_status = "Sharp" if sharpness_score > 20 else "Slightly Blurry"
        is_sharp_valid = bool(sharpness_score > 10)

        # 3. HSV Leaf Pigment & Foliage Validation
        # Convert RGB to HSV
        norm_r, norm_g, norm_b = np_img[:, :, 0] / 255.0, np_img[:, :, 1] / 255.0, np_img[:, :, 2] / 255.0
        cmax = np.maximum(np.maximum(norm_r, norm_g), norm_b)
        cmin = np.minimum(np.minimum(norm_r, norm_g), norm_b)
        delta = cmax - cmin + 1e-7

        # Hue calculation in degrees [0, 360]
        hue = np.zeros_like(cmax)
        mask_r = (cmax == norm_r)
        mask_g = (cmax == norm_g)
        mask_b = (cmax == norm_b)
        
        hue[mask_r] = 60.0 * (((norm_g[mask_r] - norm_b[mask_r]) / delta[mask_r]) % 6)
        hue[mask_g] = 60.0 * (((norm_b[mask_g] - norm_r[mask_g]) / delta[mask_g]) + 2)
        hue[mask_b] = 60.0 * (((norm_r[mask_b] - norm_g[mask_b]) / delta[mask_b]) + 4)
        
        saturation = delta / (cmax + 1e-7)

        # Plant / Foliage Pigment masks:
        green_mask = (hue >= 30) & (hue <= 165) & (saturation > 0.08) & (cmax > 0.10)
        red_mask = ((hue < 30) | (hue > 330)) & (saturation > 0.10) & (cmax > 0.10)
        yellow_mask = (hue >= 20) & (hue < 40) & (saturation > 0.10) & (cmax > 0.10)

        foliage_pixels = np.sum(green_mask | red_mask | yellow_mask)
        total_pixels = h * w
        foliage_ratio = float(foliage_pixels) / float(total_pixels)

        # 4. Texture & Spatial Color Coherence
        diff_r = np.abs(np_img[1:, :, 0].astype(np.float32) - np_img[:-1, :, 0].astype(np.float32))
        diff_g = np.abs(np_img[1:, :, 1].astype(np.float32) - np_img[:-1, :, 1].astype(np.float32))
        diff_b = np.abs(np_img[1:, :, 2].astype(np.float32) - np_img[:-1, :, 2].astype(np.float32))
        mean_spatial_jitter = float(np.mean(diff_r + diff_g + diff_b))
        
        is_coherent_surface = bool(mean_spatial_jitter < 90.0)
        has_sufficient_leaf = bool(foliage_ratio >= 0.005 and is_coherent_surface)
        target_detected = "Cotton Leaf Detected" if has_sufficient_leaf else "Leaf Specimen"

        # Combined acceptance
        is_acceptable = bool(is_light_valid and is_sharp_valid and has_sufficient_leaf)

        rejection_reason = None
        if not is_acceptable:
            if not is_light_valid:
                rejection_reason = rejection_sub
            elif not is_sharp_valid:
                rejection_reason = "Image is very blurry. Try focusing closer on the leaf."
            elif not has_sufficient_leaf:
                rejection_reason = "Ensure leaf fills the central viewfinder frame."

        return {
            "is_acceptable": bool(is_acceptable),
            "sharpness_score": round(sharpness_score, 1),
            "sharpness_status": str(sharpness_status),
            "lighting_status": str(lighting_status),
            "mean_brightness": round(mean_brightness, 1),
            "target_detected": str(target_detected),
            "foliage_coverage_percentage": round(foliage_ratio * 100, 1),
            "rejection_reason": rejection_reason
        }

    # ==================== REAL PREPROCESSING ====================
    def preprocess(self, pil_image):
        """
        Exact preprocessing from SAR-CLD-2024 CottonNet training:
        1. Convert to RGB
        2. Bilinear Resize to 224x224
        3. ToTensor (scale [0, 255] -> [0.0, 1.0])
        4. Normalize with ImageNet mean/std
        """
        image_rgb = pil_image.convert("RGB")
        resized = image_rgb.resize((self.IM_SIZE, self.IM_SIZE), Image.Resampling.BILINEAR)
        
        arr = np.array(resized, dtype=np.float32) / 255.0  # [224, 224, 3]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
        normalized = (tensor - self.mean_tensor) / self.std_tensor
        return normalized, resized

    # ==================== REAL GRAD-CAM GENERATOR ====================
    def generate_gradcam(self, input_tensor, target_class_idx, orig_image_resized):
        """
        Computes real Grad-CAM attribution on the final convolutional block.
        """
        self.model.zero_grad()
        input_tensor.requires_grad = True
        
        logits = self.model(input_tensor)
        target_score = logits[0, target_class_idx]
        target_score.backward()

        # Weighted combination of activation maps
        gradients = self.gradients  # [1, 1280, 7, 7]
        activations = self.activations  # [1, 1280, 7, 7]
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, 1280, 1, 1]
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, 7, 7]
        cam = F.relu(cam)
        
        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        # Upscale 7x7 heatmap to 224x224
        cam_pil = Image.fromarray((cam_norm * 255).astype(np.uint8)).resize(
            (self.IM_SIZE, self.IM_SIZE), Image.Resampling.BICUBIC
        )
        cam_upscaled = np.array(cam_pil, dtype=np.float32) / 255.0

        # Apply Jet Colormap
        heatmap_rgb = np.zeros((self.IM_SIZE, self.IM_SIZE, 3), dtype=np.uint8)
        # Jet calculation
        r = np.clip(1.5 - np.abs(cam_upscaled * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(cam_upscaled * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(cam_upscaled * 4 - 1), 0, 1)
        heatmap_rgb[:, :, 0] = (r * 255).astype(np.uint8)
        heatmap_rgb[:, :, 1] = (g * 255).astype(np.uint8)
        heatmap_rgb[:, :, 2] = (b * 255).astype(np.uint8)

        # Blend with original resized image
        orig_np = np.array(orig_image_resized)
        alpha = 0.6
        blended = (orig_np * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
        
        # Convert to base64 data URL
        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="JPEG", quality=90)
        base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_str}", cam_norm.tolist()

    # ==================== MAIN PREDICTION METHOD ====================
    def predict(self, pil_image, compute_gradcam=True):
        """
        Executes real end-to-end inference with accurate timing & validation.
        """
        t_start = time.perf_counter()

        # Step 1: Real Quality & OOD Validation
        quality_info = self.check_quality_and_ood(pil_image)
        t_quality = (time.perf_counter() - t_start) * 1000

        # Step 2: Exact Preprocessing
        t_pre_start = time.perf_counter()
        input_tensor, resized_img = self.preprocess(pil_image)
        t_preprocess = (time.perf_counter() - t_pre_start) * 1000

        # Step 3: Real CottonNet Forward Pass
        t_model_start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        t_model = (time.perf_counter() - t_model_start) * 1000

        # Step 4: Sort Real Probabilities
        top_indices = np.argsort(probs)[::-1]
        top1_idx = int(top_indices[0])
        top1_prob = float(probs[top1_idx])
        top1_name = self.CLASS_MAPPING[top1_idx]

        # All 7 real probabilities
        all_probs = {self.CLASS_MAPPING[i]: float(probs[i]) for i in range(7)}

        # Top 3 probabilities
        top3_predictions = [
            {"rank": i + 1, "class_name": self.CLASS_MAPPING[idx], "probability": float(probs[idx]), "percentage": f"{probs[idx]*100:.2f}%"}
            for i, idx in enumerate(top_indices[:3])
        ]

        # Step 5: OOD & Confidence Thresholding
        is_rejected = False
        rejection_reason = None

        if not quality_info["is_acceptable"]:
            is_rejected = True
            rejection_reason = quality_info["rejection_reason"]
        elif top1_prob < self.confidence_threshold:
            is_rejected = True
            rejection_reason = (
                f"Confidence ({top1_prob*100:.1f}%) is below the research reliability threshold ({self.confidence_threshold*100:.0f}%). "
                "Unable to confirm disease pathology with high confidence."
            )

        # Step 6: Real Grad-CAM Generation
        gradcam_data_url = None
        cam_grid = None
        t_gradcam = 0.0

        if compute_gradcam and not is_rejected:
            t_gc_start = time.perf_counter()
            # Needs fresh input tensor for autograd
            gc_tensor, _ = self.preprocess(pil_image)
            gradcam_data_url, cam_grid = self.generate_gradcam(gc_tensor, top1_idx, resized_img)
            t_gradcam = (time.perf_counter() - t_gc_start) * 1000

        t_total = (time.perf_counter() - t_start) * 1000

        return {
            "success": True,
            "is_rejected": is_rejected,
            "rejection_reason": rejection_reason,
            "prediction": {
                "class_index": top1_idx,
                "class_name": top1_name if not is_rejected else "Undetermined / OOD",
                "confidence": top1_prob,
                "confidence_percentage": f"{top1_prob * 100:.2f}%",
                "top3": top3_predictions,
                "all_probabilities": all_probs
            },
            "quality_metrics": quality_info,
            "latency_breakdown": {
                "preprocessing_ms": round(t_preprocess, 2),
                "model_inference_ms": round(t_model, 2),
                "gradcam_ms": round(t_gradcam, 2),
                "total_pipeline_ms": round(t_total, 2)
            },
            "model_metadata": {
                "model_name": "CottonNet",
                "total_parameters": 4870275,
                "test_macro_f1": "98.60% (Held-Out Test Dataset)",
                "checkpoint": "cottonnet_best_model.pth",
                "device": str(self.device)
            },
            "gradcam_overlay": gradcam_data_url
        }

if __name__ == "__main__":
    engine = CottonNetInferenceEngine()
    print("Testing with blank random image...")
    dummy = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    res = engine.predict(dummy)
    print("Inference result:", json.dumps(res, indent=2))
