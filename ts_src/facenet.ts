import { InferenceSession, Tensor } from 'onnxruntime-node';
import { cropAndResizeBilinear } from './image_utils';
import { ImageData } from './types';

/**
 * Facenet model for face embedding
 * Converts a face image to a 512-dimensional embedding vector
 */
export default class Facenet {
    private session: InferenceSession;

    constructor(session: InferenceSession) {
        this.session = session;
    }

    /**
     * Convert a face image to an embedding vector
     * @param imageData Image data with dimensions
     * @returns 512-dimensional embedding vector as Float32Array
     */
    async imageVectorization(imageData: ImageData): Promise<Float32Array> {
        try {
            const { data: img, width, height } = imageData;
            const data = await this.imageTransfer(img, width, height);
            const inputName = this.session.inputNames[0];
            const outputName = this.session.outputNames[0];
            
            // Create input tensor
            const inputTensor = new Tensor('float32', data, [1, 160, 160, 3]);
            
            // Run inference
            const results = await this.session.run({ [inputName]: inputTensor });
            
            // Return embedding vector
            return await results[outputName].getData() as Float32Array;
        } catch (error) {
            throw new Error(`Image vectorization error: ${(error as Error).message.split("\n")[0]}`);
        }
    }

    /**
     * Process image for the model input
     * Resizes to 160x160 and normalizes pixel values
     * @param rawImageData Raw image data as Uint8Array
     * @param width Image width
     * @param height Image height
     * @returns Processed image data as Float32Array
     */
    private async imageTransfer(rawImageData: Uint8Array, width: number, height: number): Promise<Float32Array> {
        try {
            // Use cropAndResizeBilinear to resize the image to 160x160
            const dstW = 160, dstH = 160;
            const resized = cropAndResizeBilinear(
                { data: rawImageData, width, height },
                0, // cropL
                0, // cropT
                width, // cropW
                height, // cropH
                dstW,
                dstH
            );

            // Convert to Float32Array and normalize to [0,1], interleaved RGB (ignore alpha)
            const dst = new Float32Array(dstW * dstH * 3);
            for (let i = 0; i < dstW * dstH; i++) {
                dst[i * 3 + 0] = resized[i * 4 + 0] / 255.0; // R
                dst[i * 3 + 1] = resized[i * 4 + 1] / 255.0; // G
                dst[i * 3 + 2] = resized[i * 4 + 2] / 255.0; // B
            }

            return dst;
        } catch (error) {
            throw new Error(`Image transfer error: ${(error as Error).message.split("\n")[0]}`);
        }
    }
}
