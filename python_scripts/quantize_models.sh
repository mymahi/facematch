#!/bin/bash

# This script dynamically quantizes all ONNX models in the workspace to QUint8 format

# Ensure the script exits on any error
set -e

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== Starting ONNX model dynamic quantization =====${NC}"

# Check if the required Python packages are installed
echo -e "${BLUE}Checking dependencies...${NC}"
python -c "import onnx, onnxruntime" || { echo "Error: onnx and onnxruntime packages are required"; exit 1; }

# Quantize the RetinaFace model
echo -e "${GREEN}Dynamically quantizing RetinaFace model...${NC}"
python ./python_scripts/quantize_onnx_model.py \
    --input ./exported_models/retinaface.onnx \
    --output ./exported_models/retinaface_uint8.onnx

# Quantize the FasNet V1SE model
echo -e "${GREEN}Dynamically quantizing FasNet V1SE model...${NC}"
python ./python_scripts/quantize_onnx_model.py \
    --input ./exported_models/fasnet_v1se.onnx \
    --output ./exported_models/fasnet_v1se_uint8.onnx

# Quantize the FasNet V2 model
echo -e "${GREEN}Dynamically quantizing FasNet V2 model...${NC}"
python ./python_scripts/quantize_onnx_model.py \
    --input ./exported_models/fasnet_v2.onnx \
    --output ./exported_models/fasnet_v2_uint8.onnx

# Quantize the Facenet model
echo -e "${GREEN}Dynamically quantizing Facenet model...${NC}"
python ./python_scripts/quantize_onnx_model.py \
    --input ./exported_models/facenet512.onnx \
    --output ./exported_models/facenet512_uint8.onnx

echo -e "${BLUE}===== Dynamic quantization completed =====${NC}"
echo -e "${BLUE}Quantized models saved with _uint8 suffix${NC}"
