#!/usr/bin/env python3
# Script to convert FasNet PyTorch models to ONNX format

import os
import torch
import argparse

# Import model definitions
from FasNetBackbone import MiniFASNetV2, MiniFASNetV1SE

def convert_pytorch_to_onnx(model_path, output_path, model_type):
    """Convert a PyTorch model to ONNX format.
    
    Args:
        model_path (str): Path to PyTorch model weights
        output_path (str): Path to save ONNX model
        model_type (str): Type of model, either 'MiniFASNetV2' or 'MiniFASNetV1SE'
    """
    print(f"Converting {model_type} model from {model_path} to {output_path}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model based on type
    if model_type == 'MiniFASNetV2':
        model = MiniFASNetV2(embedding_size=128, conv6_kernel=(5, 5), drop_p=0.2, num_classes=3)
    elif model_type == 'MiniFASNetV1SE':
        model = MiniFASNetV1SE(embedding_size=128, conv6_kernel=(5, 5), drop_p=0.75, num_classes=3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights
    state_dict = torch.load(model_path, map_location=device)
    
    # Check if state dict needs to be modified (remove "module." prefix)
    keys = iter(state_dict)
    first_layer_name = next(keys)
    
    if first_layer_name.find("module.") >= 0:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            name_key = key[7:]  # remove 'module.' prefix
            new_state_dict[name_key] = value
        state_dict = new_state_dict
    
    # Load state dictionary into model
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Create dummy input for tracing
    dummy_input = torch.randn(1, 3, 80, 80, device=device)  # Batch size 1, 3 channels, 80x80 image
    
    # Export the model to ONNX format
    torch.onnx.export(
        model,                            # model being run
        dummy_input,                      # model input (or a tuple for multiple inputs)
        output_path,                      # where to save the model
        export_params=True,               # store the trained parameter weights inside the model file
        opset_version=18,                 # the ONNX version to export the model to
        do_constant_folding=True,         # whether to execute constant folding for optimization
        input_names=['input'],            # the model's input names
        output_names=['output'],          # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},   # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model successfully converted and saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FasNet PyTorch models to ONNX format")
    parser.add_argument("--model_path", type=str, required=True, help="Path to PyTorch model weights")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save ONNX model")
    parser.add_argument("--model_type", type=str, required=True, choices=['MiniFASNetV2', 'MiniFASNetV1SE'], 
                        help="Type of model to convert")
    
    args = parser.parse_args()
    
    convert_pytorch_to_onnx(args.model_path, args.output_path, args.model_type)
