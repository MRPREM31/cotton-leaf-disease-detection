"""
CottonNet ONNX Export & Multi-Precision Quantization Suite
Converts PyTorch cottonnet_best_model.pth to:
1. cottonnet_fp32.onnx (Full precision)
2. cottonnet_int8.onnx (Dynamic INT8 Quantization for Mobile/Edge)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os
import torch
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from inference_engine import CottonNet

def export_cottonnet_to_onnx(
    pth_path="cottonnet_best_model.pth",
    output_fp32="cottonnet_fp32.onnx",
    output_int8="cottonnet_int8.onnx",
    im_size=224
):
    print(f"Loading CottonNet checkpoint: {pth_path}...")
    model = CottonNet(num_classes=7)
    state_dict = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("[OK] Successfully loaded PyTorch checkpoint weights.")

    dummy_input = torch.randn(1, 3, im_size, im_size, dtype=torch.float32)

    # 1. Export FP32 ONNX using TorchScript tracing (dynamo=False) for maximum mobile compatibility
    print(f"Exporting FP32 ONNX to {output_fp32}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_fp32,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False
    )

    # Check validity
    onnx_model = onnx.load(output_fp32)
    onnx.checker.check_model(onnx_model)
    fp32_size_mb = os.path.getsize(output_fp32) / (1024 * 1024)
    print(f"[OK] FP32 ONNX export verified. Size: {fp32_size_mb:.2f} MB")

    # Verify numerical parity with PyTorch
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    ort_session = ort.InferenceSession(output_fp32, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(torch_out - ort_out))
    print(f"[OK] PyTorch vs ONNX FP32 Max Absolute Difference: {max_diff:.6f}")

    # 2. Export INT8 Dynamic Quantization
    print(f"Exporting Dynamic INT8 Quantization to {output_int8}...")
    quantize_dynamic(
        model_input=output_fp32,
        model_output=output_int8,
        weight_type=QuantType.QUInt8
    )
    int8_size_mb = os.path.getsize(output_int8) / (1024 * 1024)
    print(f"[OK] INT8 Quantized ONNX verified. Size: {int8_size_mb:.2f} MB (Compression: {fp32_size_mb/int8_size_mb:.2f}x)")

    # Test INT8 inference
    ort_session_int8 = ort.InferenceSession(output_int8, providers=["CPUExecutionProvider"])
    int8_out = ort_session_int8.run(None, ort_inputs)[0]
    int8_diff = np.max(np.abs(torch_out - int8_out))
    print(f"[OK] PyTorch vs INT8 Max Absolute Difference: {int8_diff:.4f}")

    print("\n" + "="*50)
    print("COTTONNET ONNX EXPORT SUMMARY:")
    print("="*50)
    print(f"1. PyTorch Baseline (.pth): {os.path.getsize(pth_path)/(1024*1024):.2f} MB")
    print(f"2. ONNX FP32 Model:         {output_fp32} ({fp32_size_mb:.2f} MB)")
    print(f"3. ONNX INT8 Quantized:     {output_int8} ({int8_size_mb:.2f} MB)")
    print("="*50)

if __name__ == "__main__":
    export_cottonnet_to_onnx()
