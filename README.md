# Face Recognition & Anti-Spoofing Models

This repository provides scripts and tools for converting, optimizing, and testing face recognition and anti-spoofing models. It includes ONNX model exports, quantization utilities, and Python scripts for handling various model architectures such as FaceNet, RetinaFace, and MiniFASNet. 

It also comes with a complete TypeScript implementation for face matching, enabling fast and flexible inference in JavaScript environments.

## Repository Structure

- `exported_models/` — Contains ONNX and quantized models for FaceNet, RetinaFace, and MiniFASNet variants.
- `python_scripts/` — Python scripts for model conversion, optimization, quantization, and backbone definitions.
- `python_scripts/keras_layers/` — Custom Keras layers.
- `source_weights/` — Original model weights in PyTorch and Keras h5 formats.
- `ts_src/` — TypeScript source files for face matching, image utilities, and model inference.
- `test_photos/` — Sample images for testing model inference and annotation.
- `test_results/` — Output images from model inference and annotation scripts.

## Setup

### Setup (TypeScript & Python)

First, install both Node.js and Python dependencies:
```bash
npm install
```
This will automatically install Python dependencies using a postinstall script defined in `package.json`.

Then, download the required model weights:
```bash
npm run download-weights
```

## Usage

### Model Conversion & Quantization
To convert models, use the npm script:
```bash
npm run convert-models
```
This will run the necessary Python scripts for model conversion automatically.

To quantize ONNX models, use the npm script:
```bash
npm run quantize-models
```

### Running TypeScript Inference
- Use files in `ts_src/` for face matching and anti-spoofing inference.

To run face matching tests, use the npm script:
```bash
npm run test-face-match
```
This will execute the TypeScript test script for face matching.

### Testing
- Test images are in `test_photos/`.
- Results are saved in `test_results/`.


## License

This repository is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Acknowledgements

This repository uses models and weights from the following sources:

- **FaceNet** and **RetinaFace** weights: https://github.com/serengil/deepface_models
- **MiniFASNet (Anti-Spoofing)**: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing

Please refer to the respective repositories for their original licenses and citations.
