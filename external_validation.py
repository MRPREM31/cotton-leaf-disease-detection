"""
CottonCare / CottonNet External Validation Script
Scientific Validation on Independent External Datasets (Dataset_A, Dataset_B)

Strict Guidelines:
- Baseline external validation only (CottonCare V1)
- No training, no fine-tuning, no weight modification
- Preprocessing: RGB -> Resize(224, 224) -> ToTensor() -> Normalize(ImageNet mean & std)
- Exact semantic matching only: Bacterial Blight, Curl Virus, Healthy Leaf
- Nested severity folders (mild, moderate, critical) mapped to parent disease class
- Unsupported classes excluded and tracked separately
- SHA-256 duplicate detection without silent removal
- Corrupt/unreadable file tracking
- True accuracy calculated independently (correct / total)
- Macro F1 calculated via scikit-learn
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_DIR = Path(r"D:\CottonDiseaseAI")
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Also add fallback directory if needed
FALLBACK_DIR = Path(r"D:\cotton leaf")
if str(FALLBACK_DIR) not in sys.path:
    sys.path.insert(0, str(FALLBACK_DIR))

DOWNLOADS_DIR = Path(r"C:\Users\mrpre\Downloads\cotton leaf")
if str(DOWNLOADS_DIR) not in sys.path:
    sys.path.insert(0, str(DOWNLOADS_DIR))

from cottonnet_model import CottonNet

# -----------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -----------------------------------------------------------------------------
DATA_DIR = PROJECT_DIR / "data" / "External_Test"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_CSV = RESULTS_DIR / "external_validation_predictions.csv"
REPORT_TXT = RESULTS_DIR / "external_validation_report.txt"
CONFUSION_MATRIX_PNG = RESULTS_DIR / "external_confusion_matrix.png"
CLASSIFICATION_REPORT_CSV = RESULTS_DIR / "external_classification_report.csv"
SUMMARY_JSON = RESULTS_DIR / "external_validation_summary.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COTTONNET_CLASSES = [
    "Bacterial Blight",         # 0
    "Curl Virus",               # 1
    "Healthy Leaf",             # 2
    "Herbicide Growth Damage",  # 3
    "Leaf Hopper Jassids",      # 4
    "Leaf Redding",             # 5
    "Leaf Variegation"          # 6
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(COTTONNET_CLASSES)}

# Semantic matching dictionary per dataset (folder name lowercased -> standard class name)
DATASET_A_MAPPING = {
    "bacterial blight": "Bacterial Blight",
    "healthy leaf": "Healthy Leaf"
}

DATASET_B_MAPPING = {
    "bacterial blight": "Bacterial Blight",
    "curl virus": "Curl Virus",
    "healthy": "Healthy Leaf"
}

SEVERITY_FOLDERS = {"mild", "moderate", "critical", "severe"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Preprocessing transform
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# -----------------------------------------------------------------------------
# CHECKPOINT LOCATOR & VALIDATION
# -----------------------------------------------------------------------------
def locate_checkpoint() -> Path:
    """Locate the cottonnet_best_model.pth checkpoint across known locations."""
    candidate_paths = [
        PROJECT_DIR / "cottonnet_best_model.pth",
        PROJECT_DIR / "models" / "cottonnet_best_model.pth",
        FALLBACK_DIR / "cottonnet_best_model.pth",
        Path(r"C:\Users\mrpre\Downloads\cotton leaf\cottonnet_best_model.pth")
    ]
    for p in candidate_paths:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(f"Checkpoint cottonnet_best_model.pth not found in candidates: {[str(c) for c in candidate_paths]}")


def load_and_verify_model(checkpoint_path: Path, device: torch.device) -> CottonNet:
    """Instantiate CottonNet, load weights, and verify 7-class architecture."""
    model = CottonNet(num_classes=7)
    
    # Checkpoint loading
    try:
        ckpt = torch.load(str(checkpoint_path), map_location=device)
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to load checkpoint file {checkpoint_path}: {e}")
    
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError(f"FATAL: Checkpoint format invalid: {type(ckpt)}")
    
    # Strip any potential module. prefixes if trained with DataParallel
    cleaned_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Verify classifier output dimension matches num_classes=7
    if "classifier.5.weight" in cleaned_state_dict:
        out_features = cleaned_state_dict["classifier.5.weight"].shape[0]
        if out_features != 7:
            raise RuntimeError(f"FATAL: Model classifier weight has {out_features} output logits, expected 7.")
    
    try:
        model.load_state_dict(cleaned_state_dict, strict=True)
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to load state_dict into CottonNet: {e}")
    
    model.to(device)
    model.eval()
    
    # Test dummy inference forward pass
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    
    if dummy_output.shape != (1, 7):
        raise RuntimeError(f"FATAL: Model forward pass returned shape {dummy_output.shape}, expected (1, 7).")
    
    return model


# -----------------------------------------------------------------------------
# DATASET DISCOVERY & AUDIT
# -----------------------------------------------------------------------------
def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of an image file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_external_dataset(data_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Recursively scan external datasets, parse labels, detect severity subfolders,
    identify duplicates via SHA-256, and flag corrupt/unreadable files.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"External test data directory does not exist: {data_dir}")

    eligible_images = []
    excluded_images = []
    corrupt_images = []
    
    hash_seen: Dict[str, Path] = {}
    duplicates_info: List[Dict[str, Any]] = []

    for file_path in data_dir.rglob("*"):
        if not file_path.is_file():
            continue
        
        ext = file_path.suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        
        # Test file readability and image validity
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as e:
            corrupt_images.append({
                "path": str(file_path),
                "error": str(e)
            })
            continue
        
        # Compute SHA-256 for duplicate tracking
        try:
            sha256 = compute_sha256(file_path)
            if sha256 in hash_seen:
                duplicates_info.append({
                    "duplicate_path": str(file_path),
                    "original_path": str(hash_seen[sha256]),
                    "sha256": sha256
                })
            else:
                hash_seen[sha256] = file_path
        except Exception as e:
            corrupt_images.append({
                "path": str(file_path),
                "error": f"SHA256 read error: {e}"
            })
            continue

        # Parse relative path to identify dataset and disease folder
        rel_path = file_path.relative_to(data_dir)
        parts = rel_path.parts  # e.g., ('Dataset_A', 'Bacterial Blight', 'img.jpg') or ('Dataset_B', 'curl virus', 'mild', 'img.jpg')
        
        if len(parts) < 2:
            excluded_images.append({
                "path": str(file_path),
                "reason": "File directly in root folder without dataset structure",
                "folder": str(rel_path.parent)
            })
            continue
        
        dataset_name = parts[0]
        folder_class_candidate = parts[1].strip()
        
        # Determine semantic match based on dataset
        dataset_name_lower = dataset_name.lower()
        matched_class = None
        
        if "dataset_a" in dataset_name_lower or dataset_name == "Dataset_A":
            matched_class = DATASET_A_MAPPING.get(folder_class_candidate.lower())
        elif "dataset_b" in dataset_name_lower or dataset_name == "Dataset_B":
            matched_class = DATASET_B_MAPPING.get(folder_class_candidate.lower())
        else:
            # Check either mapping as fallback
            matched_class = DATASET_A_MAPPING.get(folder_class_candidate.lower()) or DATASET_B_MAPPING.get(folder_class_candidate.lower())

        # If severity folder exists as parts[2], record severity for auditing
        severity = None
        if len(parts) >= 4 and parts[2].lower() in SEVERITY_FOLDERS:
            severity = parts[2].lower()
        elif len(parts) == 3 and parts[2].lower() in SEVERITY_FOLDERS:
            severity = parts[2].lower()

        if matched_class is None:
            excluded_images.append({
                "path": str(file_path),
                "dataset": dataset_name,
                "folder": folder_class_candidate,
                "reason": f"Class '{folder_class_candidate}' not supported by CottonNet (7 classes only)"
            })
        else:
            eligible_images.append({
                "file_path": file_path,
                "dataset": dataset_name,
                "folder_class": folder_class_candidate,
                "actual_class": matched_class,
                "severity": severity,
                "sha256": sha256
            })

    audit_meta = {
        "unique_sha256_count": len(hash_seen),
        "duplicate_count": len(duplicates_info),
        "duplicates": duplicates_info
    }
    
    return eligible_images, excluded_images, corrupt_images, audit_meta


# -----------------------------------------------------------------------------
# SANITY CHECK (INDIVIDUAL SAMPLES)
# -----------------------------------------------------------------------------
def run_sanity_check(model: CottonNet, eligible_images: List[Dict[str, Any]], device: torch.device) -> None:
    """Run sanity check on 5 diverse images individually and print detailed output."""
    print("\n----------------------------------------------------")
    print("SANITY CHECK: INDIVIDUAL INFERENCE ON 5 SAMPLES")
    print("----------------------------------------------------")
    
    # Pick diverse sample images across datasets and classes
    samples = []
    seen_classes_datasets = set()
    for item in eligible_images:
        key = (item["dataset"], item["actual_class"])
        if key not in seen_classes_datasets:
            samples.append(item)
            seen_classes_datasets.add(key)
        if len(samples) >= 5:
            break
            
    # If fewer than 5 unique keys, pad with remaining items
    idx = 0
    while len(samples) < min(5, len(eligible_images)):
        if eligible_images[idx] not in samples:
            samples.append(eligible_images[idx])
        idx += 1

    for i, sample in enumerate(samples, 1):
        file_path = sample["file_path"]
        actual_class = sample["actual_class"]
        
        with Image.open(file_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
            tensor = INFERENCE_TRANSFORM(rgb_img).unsqueeze(0).to(device)
            
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            
        top1_idx = torch.argmax(probs).item()
        pred_class = COTTONNET_CLASSES[top1_idx]
        confidence = probs[top1_idx].item()
        
        print(f"Sample {i}:")
        print(f"  Image:      {file_path.name} ({sample['dataset']})")
        print(f"  Actual:     {actual_class}")
        print(f"  Prediction: {pred_class}")
        print(f"  Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")
        print(f"  Match:      {'CORRECT' if pred_class == actual_class else 'WRONG'}")
    print("----------------------------------------------------\n")


# -----------------------------------------------------------------------------
# MAIN INFERENCE PIPELINE
# -----------------------------------------------------------------------------
def run_external_validation():
    start_time_total = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Locating CottonNet checkpoint...")
    checkpoint_path = locate_checkpoint()
    
    print(f"Scanning external dataset directory: {DATA_DIR} ...")
    eligible_images, excluded_images, corrupt_images, audit_meta = scan_external_dataset(DATA_DIR)
    
    excluded_classes_list = sorted(list({ex["folder"] for ex in excluded_images if "folder" in ex}))
    excluded_str = ", ".join(excluded_classes_list) if excluded_classes_list else "None"
    
    # ----------------------------------------------------
    # REQUIRED PRE-INFERENCE BANNER
    # ----------------------------------------------------
    print("====================================================")
    print("COTTONCARE EXTERNAL VALIDATION")
    print("====================================================")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {DATA_DIR}")
    print(f"Number of eligible images: {len(eligible_images)}")
    print(f"Excluded classes: {excluded_str}")
    print("Preprocessing: RGB -> Resize(224, 224) -> ToTensor() -> Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
    print(f"Device: {device}")
    print(f"Total Unique SHA-256 Hashes: {audit_meta['unique_sha256_count']}")
    print(f"Duplicate Files Detected (SHA-256): {audit_meta['duplicate_count']}")
    print(f"Corrupt/Unreadable Images: {len(corrupt_images)}")
    print(f"Excluded Images: {len(excluded_images)}")
    print("====================================================")

    if len(eligible_images) == 0:
        raise RuntimeError("No eligible images found for external validation! Stopping.")
        
    print("\nLoading CottonNet model and verifying architecture...")
    model = load_and_verify_model(checkpoint_path, device)
    print("Model loaded successfully. Verification check passed (7 classes output).\n")
    
    # Run Sanity Check
    run_sanity_check(model, eligible_images, device)
    
    # ----------------------------------------------------
    # RUN EVALUATION OVER ALL ELIGIBLE IMAGES
    # ----------------------------------------------------
    print(f"Starting inference on {len(eligible_images)} images...")
    predictions_records = []
    
    for idx, item in enumerate(eligible_images, 1):
        file_path = item["file_path"]
        actual_class = item["actual_class"]
        dataset_name = item["dataset"]
        
        try:
            with Image.open(file_path) as pil_img:
                rgb_img = pil_img.convert("RGB")
                tensor = INFERENCE_TRANSFORM(rgb_img).unsqueeze(0).to(device)
        except Exception as e:
            corrupt_images.append({"path": str(file_path), "error": f"Runtime load error: {e}"})
            continue

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            t_infer = (time.perf_counter() - t0) * 1000.0  # ms
            probs = torch.softmax(logits, dim=1).squeeze(0)

        # Top 1
        pred_idx = torch.argmax(probs).item()
        pred_class = COTTONNET_CLASSES[pred_idx]
        confidence = probs[pred_idx].item()
        is_top1_correct = (pred_class == actual_class)
        
        # Top 3
        top3_vals, top3_indices = torch.topk(probs, k=min(3, len(COTTONNET_CLASSES)))
        top3_classes = [COTTONNET_CLASSES[i.item()] for i in top3_indices]
        top3_probs = [round(v.item(), 4) for v in top3_vals]
        is_top3_correct = (actual_class in top3_classes)

        record = {
            "dataset": dataset_name,
            "image_path": str(file_path),
            "actual_class": actual_class,
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "top1_correct": "correct" if is_top1_correct else "wrong",
            "top3_predictions": "; ".join(top3_classes),
            "top3_probabilities": "; ".join(map(str, top3_probs)),
            "inference_time_ms": round(t_infer, 2),
            "is_top1_correct_bool": is_top1_correct,
            "is_top3_correct_bool": is_top3_correct,
            "actual_idx": CLASS_TO_IDX[actual_class],
            "pred_idx": pred_idx
        }
        predictions_records.append(record)
        
        if idx % 250 == 0 or idx == len(eligible_images):
            print(f"Processed {idx}/{len(eligible_images)} images ({idx/len(eligible_images)*100:.1f}%)...")

    # ----------------------------------------------------
    # SAVE PREDICTIONS CSV
    # ----------------------------------------------------
    df = pd.DataFrame(predictions_records)
    export_df = df[[
        "dataset",
        "image_path",
        "actual_class",
        "predicted_class",
        "confidence",
        "top1_correct",
        "top3_predictions",
        "top3_probabilities",
        "inference_time_ms"
    ]]
    export_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nSaved detailed prediction logs to: {PREDICTIONS_CSV}")

    # ----------------------------------------------------
    # METRIC CALCULATIONS
    # ----------------------------------------------------
    def compute_metrics_subset(sub_df: pd.DataFrame) -> Dict[str, Any]:
        total = len(sub_df)
        if total == 0:
            return {}
        
        y_true = sub_df["actual_class"].tolist()
        y_pred = sub_df["predicted_class"].tolist()
        
        # Independent TRUE accuracy calculation
        correct = int(sub_df["is_top1_correct_bool"].sum())
        incorrect = total - correct
        true_accuracy = correct / total
        top3_correct = int(sub_df["is_top3_correct_bool"].sum())
        top3_acc = top3_correct / total
        
        # Scikit-learn Macro & Weighted metrics
        present_classes = sorted(list(set(y_true)))
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        
        # Per class metrics
        per_class = {}
        for cls_name in present_classes:
            cls_df = sub_df[sub_df["actual_class"] == cls_name]
            cls_total = len(cls_df)
            cls_correct = int(cls_df["is_top1_correct_bool"].sum())
            cls_incorrect = cls_total - cls_correct
            
            # Binary precision, recall, F1 for this class
            y_true_bin = [1 if y == cls_name else 0 for y in y_true]
            y_pred_bin = [1 if y == cls_name else 0 for y in y_pred]
            
            p = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            r = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            
            per_class[cls_name] = {
                "total_images": cls_total,
                "correct": cls_correct,
                "incorrect": cls_incorrect,
                "accuracy": round(cls_correct / cls_total, 4) if cls_total > 0 else 0.0,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1_score": round(f1, 4)
            }

        # Confusion Matrix
        cm_labels = present_classes
        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
        
        return {
            "total_images": total,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": round(true_accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "top3_accuracy": round(top3_acc, 4),
            "per_class_metrics": per_class,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": cm_labels
        }

    # Combined metrics
    combined_metrics = compute_metrics_subset(df)
    
    # Dataset A metrics
    df_a = df[df["dataset"] == "Dataset_A"]
    dataset_a_metrics = compute_metrics_subset(df_a)
    
    # Dataset B metrics
    df_b = df[df["dataset"] == "Dataset_B"]
    dataset_b_metrics = compute_metrics_subset(df_b)

    # ----------------------------------------------------
    # GENERATE CLASSIFICATION REPORT CSV
    # ----------------------------------------------------
    clf_report_dict = classification_report(
        df["actual_class"].tolist(),
        df["predicted_class"].tolist(),
        output_dict=True,
        zero_division=0
    )
    clf_report_df = pd.DataFrame(clf_report_dict).transpose()
    clf_report_df.to_csv(CLASSIFICATION_REPORT_CSV)
    print(f"Saved classification report CSV to: {CLASSIFICATION_REPORT_CSV}")

    # ----------------------------------------------------
    # GENERATE CONFUSION MATRIX PLOT
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    def plot_cm(ax, cm_data, labels, title):
        cm_arr = np.array(cm_data)
        sns.heatmap(
            cm_arr,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=False
        )
        ax.set_title(title, fontsize=13, pad=10, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("Actual Class", fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plot_cm(axes[0], combined_metrics["confusion_matrix"], combined_metrics["confusion_matrix_labels"], "Combined External Test")
    if dataset_a_metrics:
        plot_cm(axes[1], dataset_a_metrics["confusion_matrix"], dataset_a_metrics["confusion_matrix_labels"], "Dataset A")
    if dataset_b_metrics:
        plot_cm(axes[2], dataset_b_metrics["confusion_matrix"], dataset_b_metrics["confusion_matrix_labels"], "Dataset B")
        
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {CONFUSION_MATRIX_PNG}")

    # ----------------------------------------------------
    # GENERATE SUMMARY JSON
    # ----------------------------------------------------
    summary_data = {
        "metadata": {
            "validation_type": "Baseline External Scientific Validation (CottonCare V1)",
            "checkpoint": str(checkpoint_path),
            "external_test_dir": str(DATA_DIR),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "unique_sha256_images": audit_meta["unique_sha256_count"],
            "duplicates_detected": audit_meta["duplicate_count"],
            "corrupted_images_count": len(corrupt_images),
            "excluded_images_count": len(excluded_images),
            "excluded_classes": excluded_classes_list
        },
        "total_images": combined_metrics["total_images"],
        "correct": combined_metrics["correct"],
        "incorrect": combined_metrics["incorrect"],
        "accuracy": combined_metrics["accuracy"],
        "macro_f1": combined_metrics["macro_f1"],
        "weighted_f1": combined_metrics["weighted_f1"],
        "macro_precision": combined_metrics["macro_precision"],
        "macro_recall": combined_metrics["macro_recall"],
        "top3_accuracy": combined_metrics["top3_accuracy"],
        "per_class_metrics": combined_metrics["per_class_metrics"],
        "dataset_metrics": {
            "Dataset_A": dataset_a_metrics,
            "Dataset_B": dataset_b_metrics
        }
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved validation summary JSON to: {SUMMARY_JSON}")

    # ----------------------------------------------------
    # GENERATE REPORT TXT
    # ----------------------------------------------------
    with open(REPORT_TXT, "w") as f:
        f.write("====================================================\n")
        f.write("COTTONCARE EXTERNAL VALIDATION SCIENTIFIC REPORT\n")
        f.write("====================================================\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Dataset Path: {DATA_DIR}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Eligible Images: {combined_metrics['total_images']}\n")
        f.write(f"Total Unique SHA-256 Images: {audit_meta['unique_sha256_count']}\n")
        f.write(f"Duplicate Files Detected (SHA-256): {audit_meta['duplicate_count']}\n")
        f.write(f"Excluded Images: {len(excluded_images)}\n")
        f.write(f"Corrupt/Unreadable Images: {len(corrupt_images)}\n")
        f.write("====================================================\n\n")
        
        f.write("OVERALL METRICS (COMBINED DATASET A & B)\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Total Images:     {combined_metrics['total_images']}\n")
        f.write(f"Correct:          {combined_metrics['correct']}\n")
        f.write(f"Incorrect:        {combined_metrics['incorrect']}\n")
        f.write(f"True Accuracy:    {combined_metrics['accuracy']:.4f} ({combined_metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Macro F1:         {combined_metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1:      {combined_metrics['weighted_f1']:.4f}\n")
        f.write(f"Macro Precision:  {combined_metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall:     {combined_metrics['macro_recall']:.4f}\n")
        f.write(f"Top-3 Accuracy:   {combined_metrics['top3_accuracy']:.4f} ({combined_metrics['top3_accuracy']*100:.2f}%)\n\n")
        
        f.write("PER-CLASS BREAKDOWN (COMBINED)\n")
        f.write("----------------------------------------------------\n")
        for cls_name, m in combined_metrics["per_class_metrics"].items():
            f.write(f"Class: {cls_name}\n")
            f.write(f"  Total Images: {m['total_images']}\n")
            f.write(f"  Correct:      {m['correct']}\n")
            f.write(f"  Incorrect:    {m['incorrect']}\n")
            f.write(f"  Accuracy:     {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)\n")
            f.write(f"  Precision:    {m['precision']:.4f}\n")
            f.write(f"  Recall:       {m['recall']:.4f}\n")
            f.write(f"  F1 Score:     {m['f1_score']:.4f}\n\n")

        f.write("DATASET A METRICS\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Total Images:     {dataset_a_metrics.get('total_images', 0)}\n")
        f.write(f"Correct:          {dataset_a_metrics.get('correct', 0)}\n")
        f.write(f"Incorrect:        {dataset_a_metrics.get('incorrect', 0)}\n")
        f.write(f"True Accuracy:    {dataset_a_metrics.get('accuracy', 0):.4f} ({dataset_a_metrics.get('accuracy', 0)*100:.2f}%)\n")
        f.write(f"Macro F1:         {dataset_a_metrics.get('macro_f1', 0):.4f}\n")
        f.write(f"Weighted F1:      {dataset_a_metrics.get('weighted_f1', 0):.4f}\n\n")
        
        f.write("DATASET B METRICS\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Total Images:     {dataset_b_metrics.get('total_images', 0)}\n")
        f.write(f"Correct:          {dataset_b_metrics.get('correct', 0)}\n")
        f.write(f"Incorrect:        {dataset_b_metrics.get('incorrect', 0)}\n")
        f.write(f"True Accuracy:    {dataset_b_metrics.get('accuracy', 0):.4f} ({dataset_b_metrics.get('accuracy', 0)*100:.2f}%)\n")
        f.write(f"Macro F1:         {dataset_b_metrics.get('macro_f1', 0):.4f}\n")
        f.write(f"Weighted F1:      {dataset_b_metrics.get('weighted_f1', 0):.4f}\n\n")

        f.write("CONFUSION MATRIX (COMBINED)\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Labels: {combined_metrics['confusion_matrix_labels']}\n")
        for row, label in zip(combined_metrics["confusion_matrix"], combined_metrics["confusion_matrix_labels"]):
            f.write(f"{label:20s}: {row}\n")
            
        if audit_meta["duplicate_count"] > 0:
            f.write("\nDUPLICATE SAMPLES DETECTED (FIRST 20):\n")
            f.write("----------------------------------------------------\n")
            for dup in audit_meta["duplicates"][:20]:
                f.write(f"Duplicate: {dup['duplicate_path']} == Original: {dup['original_path']} (SHA: {dup['sha256']})\n")

    print(f"Saved comprehensive report to: {REPORT_TXT}")

    # ----------------------------------------------------
    # REQUIRED TERMINAL REPORT
    # ----------------------------------------------------
    print("\n====================================================")
    print("FINAL EXTERNAL VALIDATION RESULTS")
    print("====================================================")
    print(f"Total images:    {combined_metrics['total_images']}")
    print(f"Correct:         {combined_metrics['correct']}")
    print(f"Incorrect:       {combined_metrics['incorrect']}")
    print()
    print(f"Accuracy:        {combined_metrics['accuracy']:.4f} ({combined_metrics['accuracy']*100:.2f}%)")
    print(f"Macro F1:        {combined_metrics['macro_f1']:.4f}")
    print(f"Weighted F1:     {combined_metrics['weighted_f1']:.4f}")
    print(f"Macro Precision: {combined_metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {combined_metrics['macro_recall']:.4f}")
    print(f"Top-3 Accuracy:  {combined_metrics['top3_accuracy']:.4f} ({combined_metrics['top3_accuracy']*100:.2f}%)")
    print()
    print("PER CLASS")
    print("----------------------------------------------------")
    for cls_name, m in combined_metrics["per_class_metrics"].items():
        print(f"{cls_name}:")
        print(f"  Count: {m['total_images']} | Correct: {m['correct']} | Incorrect: {m['incorrect']}")
        print(f"  Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1_score']:.4f}")
    print()
    print("DATASET A")
    print("----------------------------------------------------")
    print(f"Total Images: {dataset_a_metrics.get('total_images', 0)}")
    print(f"Accuracy:     {dataset_a_metrics.get('accuracy', 0):.4f} ({dataset_a_metrics.get('accuracy', 0)*100:.2f}%)")
    print(f"Macro F1:     {dataset_a_metrics.get('macro_f1', 0):.4f}")
    print(f"Weighted F1:  {dataset_a_metrics.get('weighted_f1', 0):.4f}")
    print()
    print("DATASET B")
    print("----------------------------------------------------")
    print(f"Total Images: {dataset_b_metrics.get('total_images', 0)}")
    print(f"Accuracy:     {dataset_b_metrics.get('accuracy', 0):.4f} ({dataset_b_metrics.get('accuracy', 0)*100:.2f}%)")
    print(f"Macro F1:     {dataset_b_metrics.get('macro_f1', 0):.4f}")
    print(f"Weighted F1:  {dataset_b_metrics.get('weighted_f1', 0):.4f}")
    print()
    print("CONFUSION MATRIX")
    print("----------------------------------------------------")
    labels = combined_metrics["confusion_matrix_labels"]
    hdr = "Actual \\ Pred"
    print(f"{hdr:<20} | " + " | ".join([f"{l:<16}" for l in labels]))
    print("-" * (20 + 3 + len(labels) * 19))
    for row, label in zip(combined_metrics["confusion_matrix"], labels):
        row_str = " | ".join([f"{val:<16}" for val in row])
        print(f"{label:<20} | {row_str}")
    print("====================================================")
    print()
    print("EXTERNAL VALIDATION COMPLETE")
    print(f"- Number of images tested:             {combined_metrics['total_images']}")
    print(f"- Accuracy:                             {combined_metrics['accuracy']:.4f} ({combined_metrics['accuracy']*100:.2f}%)")
    print(f"- Macro F1:                             {combined_metrics['macro_f1']:.4f}")
    print(f"- Per-class F1:                         { {k: v['f1_score'] for k, v in combined_metrics['per_class_metrics'].items()} }")
    print(f"- Dataset A accuracy:                   {dataset_a_metrics.get('accuracy', 0):.4f} ({dataset_a_metrics.get('accuracy', 0)*100:.2f}%)")
    print(f"- Dataset B accuracy:                   {dataset_b_metrics.get('accuracy', 0):.4f} ({dataset_b_metrics.get('accuracy', 0)*100:.2f}%)")
    print(f"- Number of excluded images:            {len(excluded_images)}")
    print(f"- Number of corrupted/unreadable images:{len(corrupt_images)}")
    print("====================================================")


if __name__ == "__main__":
    run_external_validation()
