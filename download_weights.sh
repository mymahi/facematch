#!/bin/bash
# Script to download model weights for Face Detection and Anti-Spoofing

set -e

# Create the destination directory if it doesn't exist
DEST_DIR="source_weights"
mkdir -p "$DEST_DIR"

# Download weights
wget -O "$DEST_DIR/facenet512_weights.h5" "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5"
wget -O "$DEST_DIR/retinaface.h5" "https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5"
wget -O "$DEST_DIR/2.7_80x80_MiniFASNetV2.pth" "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
wget -O "$DEST_DIR/4_0_0_80x80_MiniFASNetV1SE.pth" "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth"

echo "All weights downloaded to $DEST_DIR."
