import os
import argparse
import numpy as np
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(model_path, output_path, op_types_to_quantize=None):
    """
    Dynamically quantize an ONNX model to uint8
    
    Args:
        model_path: Path to the input ONNX model
        output_path: Path to save the quantized model
        op_types_to_quantize: List of operator types to quantize, or None for default operators
    """
    print(f"Loading model from {model_path}")
    
    # Default operator types to quantize if not specified
    if op_types_to_quantize is None:
        op_types_to_quantize = [
            'Conv',
            'MatMul',
            'Gemm',
            'Add',
            'Mul',
            'Relu'
        ]
    
    # Perform dynamic quantization - this doesn't require calibration data
    print(f"Dynamically quantizing model to uint8...")
    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=op_types_to_quantize,
        extra_options={'MatMulConstBOnly': True}
    )
    
    print(f"Quantized model saved to {output_path}")
    
    # Print file size comparison
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size)*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamically quantize ONNX model to uint8")
    parser.add_argument("--input", required=True, help="Input ONNX model path")
    parser.add_argument("--output", required=True, help="Output quantized model path")
    parser.add_argument("--op_types", default=None, help="Comma-separated list of operator types to quantize")
    
    args = parser.parse_args()
    
    # Parse op_types if provided
    op_types = None
    if args.op_types:
        op_types = [op.strip() for op in args.op_types.split(',')]
    
    quantize_model(
        args.input,
        args.output,
        op_types
    )
