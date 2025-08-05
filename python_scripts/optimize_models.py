#!/usr/bin/env python3

import os
import argparse
import time
import json
from onnxruntime.quantization import quantize_dynamic, QuantType

def get_file_size_mb(file_path):
    """Get file size in megabytes"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0

def quantize_model(input_model_path, output_model_path=None):
    """Quantize an ONNX model using dynamic quantization"""
    if not os.path.exists(input_model_path):
        print(f"ERROR: Input model {input_model_path} does not exist")
        return None
    
    if output_model_path is None:
        # Create default output filename by adding '_quantized' before the extension
        name, ext = os.path.splitext(input_model_path)
        output_model_path = f"{name}_quantized{ext}"
    
    print(f"Quantizing model: {input_model_path}")
    print(f"Output model: {output_model_path}")
    
    original_size = get_file_size_mb(input_model_path)
    print(f"Original model size: {original_size:.2f} MB")
    
    start_time = time.time()
    
    # Perform dynamic quantization
    try:
        quantize_dynamic(
            model_input=input_model_path,
            model_output=output_model_path,
            weight_type=QuantType.QUInt8,
            optimize_model=True
        )
    except Exception as e:
        print(f"ERROR during quantization: {str(e)}")
        return None
    
    end_time = time.time()
    quantization_time = end_time - start_time
    
    if os.path.exists(output_model_path):
        quantized_size = get_file_size_mb(output_model_path)
        size_reduction = (original_size - quantized_size) / original_size * 100
        
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Size reduction: {size_reduction:.2f}%")
        print(f"Quantization completed in {quantization_time:.2f} seconds")
        
        return {
            "input_model": input_model_path,
            "output_model": output_model_path,
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "size_reduction_percent": size_reduction,
            "quantization_time_seconds": quantization_time
        }
    else:
        print("ERROR: Quantization failed, output file not created")
        return None

def batch_quantize_models(model_paths, output_dir=None):
    """Quantize multiple models and generate a report"""
    results = []
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"WARNING: Model {model_path} not found, skipping")
            continue
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            name = os.path.basename(model_path)
            base_name, ext = os.path.splitext(name)
            output_model_path = os.path.join(output_dir, f"{base_name}_quantized{ext}")
        else:
            name, ext = os.path.splitext(model_path)
            output_model_path = f"{name}_quantized{ext}"
        
        result = quantize_model(model_path, output_model_path)
        if result:
            results.append(result)
    
    # Generate report
    if results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = f"quantization_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump({"quantization_results": results}, f, indent=2)
        
        print(f"\nQuantization report saved to: {report_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Optimize ONNX models with dynamic quantization")
    parser.add_argument("--model", "-m", help="Path to input ONNX model file", nargs='+')
    parser.add_argument("--output", "-o", help="Path for output quantized model file or directory (for multiple models)")
    parser.add_argument("--batch", "-b", action="store_true", help="Process multiple models in batch mode")
    
    args = parser.parse_args()
    
    if args.batch and args.model:
        # Batch process multiple models
        batch_quantize_models(args.model, args.output)
    elif args.model:
        if len(args.model) == 1:
            # Single model
            quantize_model(args.model[0], args.output)
        else:
            print("ERROR: For multiple models, use --batch flag")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
