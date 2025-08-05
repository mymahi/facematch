#!/usr/bin/env python3
# Script to convert both FasNet models to ONNX

import os
from convert_fasnet_to_onnx import convert_pytorch_to_onnx
from convert_facenet_to_onnx import convert_facenet_to_onnx
from convert_retinaface_to_onnx import convert_retinaface_to_onnx

def main():
    # Define model paths and types
    models = [
        {
            "model_path": os.path.join(".", "source_weights", "2.7_80x80_MiniFASNetV2.pth"),
            "output_path": os.path.join(".", "exported_models", "fasnet_v2.onnx"),
            "model_type": "MiniFASNetV2"
        },
        {
            "model_path": os.path.join(".", "source_weights", "4_0_0_80x80_MiniFASNetV1SE.pth"),
            "output_path": os.path.join(".", "exported_models", "fasnet_v1se.onnx"),
            "model_type": "MiniFASNetV1SE"
        }
    ]
    
    # Convert each model
    for model in models:
        # Check if input file exists
        if not os.path.exists(model["model_path"]):
            print(f"Error: Model file {model['model_path']} not found")
            continue
            
        try:
            convert_pytorch_to_onnx(
                model["model_path"],
                model["output_path"],
                model["model_type"]
            )
            print(f"Successfully converted {model['model_path']} to {model['output_path']}")
        except Exception as e:
            print(f"Error converting {model['model_path']}: {str(e)}")

    # Convert FaceNet model
    try:
        facenet_h5 = os.path.join(".", "source_weights", "facenet512_weights.h5")
        facenet_onnx = os.path.join(".", "exported_models", "facenet512.onnx")
        if os.path.exists(facenet_h5):
            convert_facenet_to_onnx(facenet_h5, facenet_onnx)
            print(f"Successfully converted {facenet_h5} to {facenet_onnx}")
        else:
            print(f"Error: Model file {facenet_h5} not found")
    except Exception as e:
        print(f"Error converting FaceNet model: {str(e)}")

    # Convert RetinaFace model
    try:
        retinaface_h5 = os.path.join(".", "source_weights", "retinaface.h5")
        retinaface_onnx = os.path.join(".", "exported_models", "retinaface.onnx")
        if os.path.exists(retinaface_h5):
            convert_retinaface_to_onnx(retinaface_h5, retinaface_onnx)
            print(f"Successfully converted {retinaface_h5} to {retinaface_onnx}")
        else:
            print(f"Error: Model file {retinaface_h5} not found")
    except Exception as e:
        print(f"Error converting RetinaFace model: {str(e)}")

    print("Conversion process completed")
if __name__ == "__main__":
    main()
