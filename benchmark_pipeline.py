"""
CottonNet Real Scientific Benchmark & Robustness Evaluation Pipeline
Measures actual measured latencies, parameter counts, and OOD gatekeeper tests.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os
import time
import json
import numpy as np
from PIL import Image

import torch
import onnxruntime as ort

from inference_engine import CottonNet, CottonNetInferenceEngine

def benchmark_models(num_warmup=10, num_runs=50):
    print("="*60)
    print("COTTONNET SCIENTIFIC BENCHMARK SUITE")
    print("="*60)

    im_size = 224
    dummy_np = np.random.randn(1, 3, im_size, im_size).astype(np.float32)
    dummy_torch = torch.from_numpy(dummy_np)

    # 1. PyTorch CPU
    model = CottonNet(num_classes=7)
    model.load_state_dict(torch.load("cottonnet_best_model.pth", map_location="cpu"))
    model.eval()

    # Measure parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters:")
    print(f"  Total Parameters:     {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # PyTorch Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_torch)
    
    # PyTorch Timing
    pt_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_torch)
            t1 = time.perf_counter()
            pt_times.append((t1 - t0) * 1000)
    
    pt_latencies = {
        "mean_ms": float(np.mean(pt_times)),
        "median_ms": float(np.median(pt_times)),
        "p95_ms": float(np.percentile(pt_times, 95)),
        "min_ms": float(np.min(pt_times)),
        "max_ms": float(np.max(pt_times))
    }

    # 2. ONNX FP32 CPU
    ort_fp32 = ort.InferenceSession("cottonnet_fp32.onnx", providers=["CPUExecutionProvider"])
    ort_in_name = ort_fp32.get_inputs()[0].name
    for _ in range(num_warmup):
        _ = ort_fp32.run(None, {ort_in_name: dummy_np})
    
    onnx_fp32_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = ort_fp32.run(None, {ort_in_name: dummy_np})
        t1 = time.perf_counter()
        onnx_fp32_times.append((t1 - t0) * 1000)
    
    onnx_fp32_latencies = {
        "mean_ms": float(np.mean(onnx_fp32_times)),
        "median_ms": float(np.median(onnx_fp32_times)),
        "p95_ms": float(np.percentile(onnx_fp32_times, 95)),
        "min_ms": float(np.min(onnx_fp32_times))
    }

    # 3. ONNX INT8 CPU
    ort_int8 = ort.InferenceSession("cottonnet_int8.onnx", providers=["CPUExecutionProvider"])
    ort_int8_in = ort_int8.get_inputs()[0].name
    for _ in range(num_warmup):
        _ = ort_int8.run(None, {ort_int8_in: dummy_np})
    
    onnx_int8_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = ort_int8.run(None, {ort_int8_in: dummy_np})
        t1 = time.perf_counter()
        onnx_int8_times.append((t1 - t0) * 1000)
    
    onnx_int8_latencies = {
        "mean_ms": float(np.mean(onnx_int8_times)),
        "median_ms": float(np.median(onnx_int8_times)),
        "p95_ms": float(np.percentile(onnx_int8_times, 95)),
        "min_ms": float(np.min(onnx_int8_times))
    }

    print("\n" + "="*60)
    print("RUNTIME LATENCY COMPARISON (CPU):")
    print("="*60)
    print(f"PyTorch CPU:   Mean={pt_latencies['mean_ms']:.2f}ms, P95={pt_latencies['p95_ms']:.2f}ms")
    print(f"ONNX FP32 CPU: Mean={onnx_fp32_latencies['mean_ms']:.2f}ms, P95={onnx_fp32_latencies['p95_ms']:.2f}ms")
    print(f"ONNX INT8 CPU: Mean={onnx_int8_latencies['mean_ms']:.2f}ms, P95={onnx_int8_latencies['p95_ms']:.2f}ms")
    print("="*60)

    # 4. Robustness & OOD Rejection Evaluation
    print("\nRunning Robustness & OOD Rejection Evaluation...")
    engine = CottonNetInferenceEngine()
    
    test_cases = {
        "pure_black_dark": Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)),
        "pure_white_glare": Image.fromarray(np.full((224, 224, 3), 255, dtype=np.uint8)),
        "random_noise_ood": Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
        "uniform_gray_background": Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8)),
    }

    ood_results = {}
    for name, img in test_cases.items():
        res = engine.predict(img)
        ood_results[name] = {
            "is_rejected": res["is_rejected"],
            "rejection_reason": res["rejection_reason"],
            "sharpness": res["quality_metrics"]["sharpness_score"],
            "lighting": res["quality_metrics"]["lighting_status"]
        }
        print(f"  [{name}] Rejected: {res['is_rejected']} | Reason: {res['rejection_reason']}")

    report = {
        "model_parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "total_millions": round(total_params / 1e6, 2),
            "trainable_millions": round(trainable_params / 1e6, 2)
        },
        "model_sizes_mb": {
            "pytorch_pth": round(os.path.getsize("cottonnet_best_model.pth") / (1024*1024), 2),
            "onnx_fp32": round(os.path.getsize("cottonnet_fp32.onnx") / (1024*1024), 2),
            "onnx_int8": round(os.path.getsize("cottonnet_int8.onnx") / (1024*1024), 2)
        },
        "latencies_ms": {
            "pytorch_cpu": pt_latencies,
            "onnx_fp32_cpu": onnx_fp32_latencies,
            "onnx_int8_cpu": onnx_int8_latencies
        },
        "ood_rejection_tests": ood_results
    }

    with open("benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Benchmark report saved to benchmark_report.json")
    return report

if __name__ == "__main__":
    benchmark_models()
