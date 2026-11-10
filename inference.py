"""
CottonNet Inference & Grad-CAM Visualization Script
Run classification & feature activation map on any cotton leaf image.
"""

import sys
import os
import argparse
import numpy as np
from PIL import Image

CLASS_NAMES = [
    "Bacterial Blight",
    "Curl Virus",
    "Healthy Leaf",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation"
]

TREATMENTS = {
    "Bacterial Blight": {
        "symptoms": "Water-soaked angular spots on leaves that turn brown/black, vein necrosis.",
        "management": "Spray Copper Oxychloride 50 WP (2.5 g/L) + Streptocycline (100 mg/L). Avoid excessive nitrogen fertilizer.",
        "organic": "Neem seed kernel extract (NSKE 5%) spray every 10 days."
    },
    "Curl Virus": {
        "symptoms": "Upward or downward leaf curling, thickened veins, enation on lower surface.",
        "management": "Control whitefly vector with Diafenthiuron 50 WP (1 g/L) or Acetamiprid 20 SP (0.2 g/L).",
        "organic": "Install yellow sticky traps (15-20 per acre) and spray 5% neem oil."
    },
    "Healthy Leaf": {
        "symptoms": "Uniform vibrant green color, standard leaf morphology without spots or curling.",
        "management": "Maintain balanced NPK fertilization (120:60:60 kg/ha) and regular scouting.",
        "organic": "Apply Jeevamrutha or Panchagavya foliar spray for enhanced plant vigor."
    },
    "Herbicide Growth Damage": {
        "symptoms": "Narrow strap-like leaves, cupping, yellowing along veins due to herbicide drift.",
        "management": "Wash foliage with clean water immediately; spray 1% Urea + 1% Potassium Nitrate to stimulate new vegetative flush.",
        "organic": "Foliar application of seaweed extract (2 mL/L) to relieve chemical stress."
    },
    "Leaf Hopper Jassids": {
        "symptoms": "Yellowing of leaf margins (hopper burn), downward curling, drying of edges.",
        "management": "Spray Flonicamid 50 WG (0.3 g/L) or Imidacloprid 17.8 SL (0.3 mL/L).",
        "organic": "Apply Beauveria bassiana (5 g/L) and ensure field sanitation."
    },
    "Leaf Redding": {
        "symptoms": "Reddening/purpling of older leaves due to anthocyanin buildup from magnesium/nitrogen deficiency or cold stress.",
        "management": "Foliar spray of 2% Magnesium Sulfate (MgSO4) + 1% Urea at 15-day intervals.",
        "organic": "Apply vermicompost and well-decomposed FYM enriched with rock phosphate."
    },
    "Leaf Variegation": {
        "symptoms": "Irregular yellow and green patches/mottling caused by physiological mutation or secondary viral infection.",
        "management": "Rogue out severely affected chimeric plants; optimize micronutrient spray (Zinc + Boron 0.2%).",
        "organic": "Soil application of Bio-fertilizers (Azospirillum + PSB)."
    }
}

def preprocess_image(image_path, im_size=224):
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((im_size, im_size), Image.Resampling.BILINEAR)
    
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    normalized = (img_array - mean) / std
    tensor = np.transpose(normalized, (2, 0, 1)) # HWC to CHW
    tensor = np.expand_dims(tensor, axis=0)      # Add batch dim [1, 3, 224, 224]
    return image, tensor

def predict_pytorch(image_path, model_path="cottonnet_best_model.pth"):
    import torch
    import torch.nn.functional as F
    from cottonnet_model import CottonNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CottonNet(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    orig_img, tensor_np = preprocess_image(image_path)
    tensor = torch.from_numpy(tensor_np).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    predicted_class = CLASS_NAMES[top_idx]

    return {
        "class": predicted_class,
        "confidence": confidence,
        "class_id": top_idx,
        "all_probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "advisory": TREATMENTS[predicted_class]
    }

def main():
    parser = argparse.ArgumentParser(description="CottonNet Leaf Disease Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image")
    parser.add_argument("--model", type=str, default="cottonnet_best_model.pth", help="Path to .pth checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
        sys.exit(1)

    print(f"Analyzing cotton leaf: {args.image}...")
    res = predict_pytorch(args.image, args.model)
    
    print("\n" + "="*50)
    print(f"🌱 COTTONGUARD AI DIAGNOSIS")
    print("="*50)
    print(f"Disease:       {res['class']}")
    print(f"Confidence:    {res['confidence']*100:.2f}%")
    print(f"\nFarmer Advisory:")
    print(f"• Symptoms:   {res['advisory']['symptoms']}")
    print(f"• Chemical:   {res['advisory']['management']}")
    print(f"• Organic:    {res['advisory']['organic']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage: python inference.py --image <path_to_leaf_image>")
