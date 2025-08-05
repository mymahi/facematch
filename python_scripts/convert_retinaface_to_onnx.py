#!/usr/bin/env python3
# Script to convert RetinaFace model from h5 to ONNX format

import os
import tensorflow as tf
import tf2onnx
import numpy as np

# Import our modified retinaface_model
import retinaface_model

def convert_retinaface_to_onnx(h5_path='retinaface.h5', onnx_path='retinaface.onnx'):
    """
    Convert RetinaFace model from H5 format to ONNX format
    
    Args:
        h5_path: Path to the H5 model file
        onnx_path: Path where to save the ONNX model file
    """
    print(f"Building RetinaFace model...")
    
    # Build the model using the structure defined in retinaface_model.py
    model = retinaface_model.build_model()
    
    # Load pre-trained weights
    if os.path.exists(h5_path):
        model = retinaface_model.load_weights(model, h5_path)
        print(f"Model weights loaded from {h5_path}")
    else:
        print(f"Warning: Model file {h5_path} not found.")
        return None

    # Save the Keras model with weights to a separate file before ONNX conversion
    # keras_save_path = "retinaface.keras"
    # print(f"Saving Keras model with weights to {keras_save_path}...")
    # model.save(keras_save_path)
    # print(f"Keras model saved to {keras_save_path}")

    # Create input with dynamic shape for conversion
    spec = (tf.TensorSpec((1, None, None, 3), tf.float32, name="data"),) # type: ignore

    print(f"Converting model to ONNX format and saving to {onnx_path}...")

    # Convert the model to ONNX with opset (18)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=18)

    # Save the model
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    print(f"Model successfully converted and saved to {onnx_path}")
    return onnx_path

if __name__ == "__main__":
    convert_retinaface_to_onnx()
