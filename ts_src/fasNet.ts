import { InferenceSession, Tensor } from 'onnxruntime-node';
import { cropAndResizeBilinear } from './image_utils';
import { ImageData } from './types';

export interface FasNetResult {
    isReal: boolean;
    antispoofScore: number;
}

/**
 * FasNet - Face Anti-Spoofing Network implementation using ONNX models
 * Based on MiniFASNetV1SE and MiniFASNetV2 models from Silent-Face-Anti-Spoofing
 */
export default class FasNet {
    private v1Session: InferenceSession;
    private v2Session: InferenceSession;
    private inputSize: number;

    /**
     * Create a FasNet instance
     * @param v1Session - ONNX InferenceSession for MiniFASNetV1SE model
     * @param v2Session - ONNX InferenceSession for MiniFASNetV2 model
     * @param inputSize - Input size for the models (default: 80x80)
     */
    constructor(v1Session: InferenceSession, v2Session: InferenceSession, inputSize = 80) {
        this.v1Session = v1Session;
        this.v2Session = v2Session;
        this.inputSize = inputSize;
    }

    /**
     * Process the image for a specific model
     * @param imageData - Original image data
     * @param facialArea - Facial area coordinates [x, y, width, height]
     * @param scale - Scale factor for cropping (2.7 for first model, 4.0 for second)
     * @returns Processed image ready for model input
     */
    private async cropAndResize(
        imageData: ImageData,
        facialArea: [number, number, number, number],
        scale: number
    ): Promise<ImageData> {
        const [x, y, w, h] = facialArea;
        
        // Calculate the center of the face
        const centerX = x + w / 2;
        const centerY = y + h / 2;
        
        // Calculate new box dimensions
        const newWidth = w * scale;
        const newHeight = h * scale;
        
        // Calculate new box coordinates
        let left = Math.floor(centerX - newWidth / 2);
        let top = Math.floor(centerY - newHeight / 2);
        let right = Math.floor(centerX + newWidth / 2);
        let bottom = Math.floor(centerY + newHeight / 2);
        
        // Ensure coordinates are within image bounds
        if (left < 0) {
            right -= left;
            left = 0;
        }
        if (top < 0) {
            bottom -= top;
            top = 0;
        }
        if (right > imageData.width - 1) {
            left -= right - imageData.width + 1;
            right = imageData.width - 1;
        }
        if (bottom > imageData.height - 1) {
            top -= bottom - imageData.height + 1;
            bottom = imageData.height - 1;
        }
        
        // Ensure coordinates are valid after adjustments
        left = Math.max(0, left);
        top = Math.max(0, top);
        const extractWidth = Math.min(imageData.width - left, right - left);
        const extractHeight = Math.min(imageData.height - top, bottom - top);

        // Use cropAndResizeBilinear from image_utils.ts
        const dst = cropAndResizeBilinear(
            imageData,
            left,
            top,
            extractWidth,
            extractHeight,
            this.inputSize,
            this.inputSize
        );

        return {
            data: dst,
            width: this.inputSize,
            height: this.inputSize
        };
    }
    
    /**
     * Prepare input tensor for model inference
     * @param imageData - Processed image data
     * @returns Tensor for model inference
     */
    private prepareInputTensor(imageData: ImageData): Tensor {
        // Create a float32 array for the RGB channels (NCHW format)
        const data = new Float32Array(3 * this.inputSize * this.inputSize);
        const len = this.inputSize * this.inputSize;
        
        // Convert from interleaved RGB to planar format (NCHW)
        for (let i = 0; i < len; i++) {
            // R channel
            data[i] = imageData.data[i * 4];
            // G channel
            data[i + len] = imageData.data[i * 4 + 1];
            // B channel
            data[i + len * 2] = imageData.data[i * 4 + 2];
        }
        
        // Create tensor with shape [1, 3, height, width] (batch, channels, height, width)
        return new Tensor('float32', data, [1, 3, this.inputSize, this.inputSize]);
    }
    
    /**
     * Apply softmax to model output
     * @param output - Raw model output
     * @returns Softmax probabilities
     */
    private softmax(output: Float32Array): Float32Array {
        const maxVal = Math.max(...Array.from(output));
        const expValues = Array.from(output).map(val => Math.exp(val - maxVal));
        const sumExp = expValues.reduce((acc, val) => acc + val, 0);
        
        return new Float32Array(expValues.map(val => val / sumExp));
    }
    
    /**
     * Analyze an image to detect if it's a real face or spoofed
     * @param imageData - Image data to analyze
     * @param facialArea - Facial area coordinates [x, y, width, height]
     * @returns Analysis result with is_real flag and confidence score
     */
    async analyze(imageData: ImageData, facialArea: [number, number, number, number]): Promise<FasNetResult> {
        try {
            // Process images for both models with different scales
            const firstImageData = await this.cropAndResize(imageData, facialArea, 2.7);
            const secondImageData = await this.cropAndResize(imageData, facialArea, 4.0);
            
            // Prepare input tensors
            const firstInputTensor = this.prepareInputTensor(firstImageData);
            const secondInputTensor = this.prepareInputTensor(secondImageData);
            
            // Run inference on both models
            const firstModelResult = await this.v2Session.run({ input: firstInputTensor });
            const secondModelResult = await this.v1Session.run({ input: secondInputTensor });
            
            // Get and process outputs
            const firstOutput = await firstModelResult.output.getData() as Float32Array;
            const secondOutput = await secondModelResult.output.getData() as Float32Array;

            // Apply softmax to get probabilities
            const firstProbs = this.softmax(firstOutput);
            const secondProbs = this.softmax(secondOutput);
            
            // Combine predictions (sum the probabilities)
            const combinedPrediction = new Float32Array(3);
            for (let i = 0; i < 3; i++) {
                combinedPrediction[i] = firstProbs[i] + secondProbs[i];
            }
            
            // Get the class with highest probability
            let maxIndex = 0;
            let maxValue = combinedPrediction[0];
            for (let i = 1; i < combinedPrediction.length; i++) {
                if (combinedPrediction[i] > maxValue) {
                    maxValue = combinedPrediction[i];
                    maxIndex = i;
                }
            }
            
            // Class 1 represents a real face (as in the Python implementation)
            const isReal = maxIndex === 1;
            
            // The score is the probability of the predicted class divided by 2
            const score = maxValue / 2;
            
            return {
                isReal: isReal,
                antispoofScore: score
            };
        } catch (error) {
            console.error('Error in FasNet analysis:', error);
            throw error;
        }
    }
}
