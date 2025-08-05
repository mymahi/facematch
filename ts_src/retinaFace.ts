import { InferenceSession, Tensor } from 'onnxruntime-node';
import { ImageData } from './types';
import { cropAndResizeBilinear } from './image_utils';

export interface FaceDetectionResult {
    facial_area: [number, number, number, number];
    landmarks: {
        left_eye: [number, number];
        right_eye: [number, number];
        nose: [number, number];
        mouth_left: [number, number];
        mouth_right: [number, number];
    };
    score: number;
};


export default class RetinaFace {
    public static readonly _anchors_fpn = {
        "stride32": [
            [-248.0, -248.0, 263.0, 263.0],
            [-120.0, -120.0, 135.0, 135.0]
        ],
        "stride16": [
            [-56.0, -56.0, 71.0, 71.0],
            [-24.0, -24.0, 39.0, 39.0]
        ],
        "stride8": [
            [-8.0, -8.0, 23.0, 23.0],
            [0.0, 0.0, 15.0, 15.0]
        ]
    }

    private session: InferenceSession;

    /**
     * Create a Retinaface instance
     * @param session - ONNX InferenceSession
     */
    constructor(session: InferenceSession) {
        this.session = session;
    }

    /**
     * Generate face proposals from model outputs (private method)
     */
    private async generateProposals(
        anchors: number[][],
        featStride: number,
        scoreT: Tensor,
        bboxT: Tensor,
        landmarkT: Tensor,
        probThreshold: number
    ): Promise<FaceDetectionResult[]> {
        // Get dimensions using NHWC layout (batch, height, width, channels)
        const scoreDims = scoreT.dims;
        const h = scoreDims[1]; // height is dimension 1 in NHWC
        const w = scoreDims[2]; // width is dimension 2 in NHWC
        const numClasses = 2; // 2 classes: background and face

        // Get tensor data in parallel to improve performance
        const [score, bbox, landmark] = await Promise.all([
            scoreT.getData(),
            bboxT.getData(),
            landmarkT.getData()
        ]) as [Float32Array, Float32Array, Float32Array];

        // Initialize results
        const faces: FaceDetectionResult[] = [];
        
        // Setup constants
        const numAnchors = anchors.length;
        const LANDMARK_POINTS_PER_FACE = 5; // 5 landmarks: right eye, left eye, nose, mouth right, mouth left
        const VALUES_PER_LANDMARK = 2;      // x and y coordinates per landmark
        const LANDMARK_VALUES_PER_FACE = LANDMARK_POINTS_PER_FACE * VALUES_PER_LANDMARK; // 10 total values

        // Process each anchor
        for (let a = 0; a < numAnchors; a++) {
            const anchor = anchors[a];
            let anchorY = anchor[1];
            
            // Use Python-style inclusive coordinates (width = x2-x1+1, height = y2-y1+1)
            const anchorW = anchor[2] - anchor[0] + 1.0;
            const anchorH = anchor[3] - anchor[1] + 1.0;

            for (let y = 0; y < h; y++) {
                let anchorX = anchor[0];

                for (let x = 0; x < w; x++) {
                    // Calculate index for foreground score directly
                    const fgIdx = ((0 * h + y) * w + x) * (numClasses * numAnchors) + numAnchors + a;
                    
                    // Just use the foreground probability directly without considering background
                    const fgProb = score[fgIdx];
                    const faceProbability = fgProb;
                    if (faceProbability >= probThreshold) {
                        // Calculate bbox indices using NHWC layout
                        const BBOX_VALUES_PER_ANCHOR = 4; // 4 values per bbox: dx, dy, dw, dh
                        const bboxBaseIdx = ((0 * h + y) * w + x) * (BBOX_VALUES_PER_ANCHOR * numAnchors);
                        
                        // Extract bbox regression values
                        const dx = bbox[bboxBaseIdx + a * BBOX_VALUES_PER_ANCHOR + 0]; // dx for this anchor
                        const dy = bbox[bboxBaseIdx + a * BBOX_VALUES_PER_ANCHOR + 1]; // dy for this anchor
                        const dw = bbox[bboxBaseIdx + a * BBOX_VALUES_PER_ANCHOR + 2]; // dw for this anchor
                        const dh = bbox[bboxBaseIdx + a * BBOX_VALUES_PER_ANCHOR + 3]; // dh for this anchor
                        // Calculate center point with Python-style anchor width/height
                        const cx = anchorX + 0.5 * (anchorW - 1.0);
                        const cy = anchorY + 0.5 * (anchorH - 1.0);
                        
                        // Apply regression deltas exactly as in Python
                        const pbCx = cx + anchorW * dx;
                        const pbCy = cy + anchorH * dy;
                        const pbW = anchorW * Math.exp(dw);
                        const pbH = anchorH * Math.exp(dh);
                        
                        // Calculate box coordinates consistent with Python
                        const x0 = pbCx - 0.5 * (pbW - 1.0);
                        const y0 = pbCy - 0.5 * (pbH - 1.0);
                        const x1 = pbCx + 0.5 * (pbW - 1.0);
                        const y1 = pbCy + 0.5 * (pbH - 1.0);
                        
                        // Use anchorW and anchorH directly for landmark mapping (Python logic)
                        const w2 = anchorW;
                        const h2 = anchorH;

                        // Calculate landmark indices using NHWC layout
                        const landmarkBaseIdx = ((0 * h + y) * w + x) * (LANDMARK_VALUES_PER_FACE * numAnchors);
                        
                        faces.push({
                            facial_area: [x0, y0, x1, y1],
                            landmarks: {
                                right_eye: [
                                    cx + w2 * landmark[landmarkBaseIdx + a * LANDMARK_VALUES_PER_FACE + 0],
                                    cy + h2 * landmark[landmarkBaseIdx + a * LANDMARK_VALUES_PER_FACE + 1]
                                ],
                                left_eye: [
                                    cx + w2 * landmark[landmarkBaseIdx + a * LANDMARK_VALUES_PER_FACE + 2],
                                    cy + h2 * landmark[landmarkBaseIdx + a * LANDMARK_VALUES_PER_FACE + 3]
                                ],
                                nose: [
                                    cx + w2 * landmark[landmarkBaseIdx + a * LANDMARK_VALUES_PER_FACE + 4],
                                    cy + h2 * landmark[landmarkBaseIdx + a * LANDMARK_VALUES_PER_FACE + 5]
                                ],
                                mouth_right: [
                                    cx + w2 * landmark[landmarkBaseIdx + a * LANDMARK_VALUES_PER_FACE + 6],
                                    cy + h2 * landmark[landmarkBaseIdx + a * LANDMARK_VALUES_PER_FACE + 7]
                                ],
                                mouth_left: [
                                    cx + w2 * landmark[landmarkBaseIdx + a * LANDMARK_VALUES_PER_FACE + 8],
                                    cy + h2 * landmark[landmarkBaseIdx + a * LANDMARK_VALUES_PER_FACE + 9]
                                ]
                            },
                            score: faceProbability
                        });
                    }
                    anchorX += featStride;
                }
                anchorY += featStride;
            }
        }

        return faces;
    }

    /**
     * Process a stride of model outputs (private method)
     */
    private async processStride(
        results: InferenceSession.OnnxValueMapType,
        faceProposals: FaceDetectionResult[],
        probThreshold: number,
        stride: number,
        anchors: number[][]
    ) {
        const score = results['face_rpn_cls_prob_reshape_stride' + stride];
        const bbox = results['face_rpn_bbox_pred_stride' + stride];
        const landmark = results['face_rpn_landmark_pred_stride' + stride];
        const featStride = stride;
        // Get proposals using foreground scores only (no background normalization)
        const newProposals = await this.generateProposals(anchors, featStride, score, bbox, landmark, probThreshold);
        
        // Apply additional filtering
        const filteredProposals = newProposals.filter(face => face.score >= probThreshold);
        
        // Add filtered proposals to the face proposals array
        faceProposals.push(...filteredProposals);
    }

    /**
     * Non-maximum suppression for face bounding boxes (private method)
     */
    private nmsSortedBboxes(faceObjects: FaceDetectionResult[], nmsThreshold: number): number[] {
        const n = faceObjects.length;
        if (n === 0) {
            return [];
        }
        // Filter out faces that are too small (width or height < 5 pixels)
        // This matches the Python implementation's filtering
        const validFaces: number[] = [];
        
        // Check each face detection
        for (let i = 0; i < n; i++) {
            const box = faceObjects[i].facial_area;
            const width = Math.round(box[2] - box[0] + 1);  // +1 to match Python's inclusive coordinates
            const height = Math.round(box[3] - box[1] + 1);
            
            // Only keep faces that have sufficient size
            if (width >= 5 && height >= 5) {
                validFaces.push(i);
            }
        }
        if (validFaces.length === 0) return [];
        // Calculate areas for all valid boxes (adding +1 to match Python implementation)
        const areas = validFaces.map(i => {
            const obj = faceObjects[i];
            return (obj.facial_area[2] - obj.facial_area[0] + 1) * (obj.facial_area[3] - obj.facial_area[1] + 1);
        });
        
        // Create indices array and sort by scores (descending)
        const order = Array.from({ length: validFaces.length }, (_, i) => i)
            .sort((a, b) => faceObjects[validFaces[b]].score - faceObjects[validFaces[a]].score);
            
        // Track suppressed boxes
        const suppressed = new Array(validFaces.length).fill(0);
        const keep: number[] = [];
        
        // Process boxes in order of decreasing confidence
        for (let _i = 0; _i < validFaces.length; _i++) {
            const i = order[_i];
            
            // Skip already suppressed boxes
            if (suppressed[i] === 1) continue;
            
            keep.push(validFaces[i]); // Keep the original index
            const faceI = faceObjects[validFaces[i]];
            const ix1 = faceI.facial_area[0];
            const iy1 = faceI.facial_area[1];
            const ix2 = faceI.facial_area[2];
            const iy2 = faceI.facial_area[3];
            const iarea = areas[i];
            // Check against remaining boxes
            for (let _j = _i + 1; _j < validFaces.length; _j++) {
                const j = order[_j];
                
                // Skip already suppressed boxes
                if (suppressed[j] === 1) continue;
                
                const faceJ = faceObjects[validFaces[j]];
                
                // Calculate intersection
                const xx1 = Math.max(ix1, faceJ.facial_area[0]);
                const yy1 = Math.max(iy1, faceJ.facial_area[1]);
                const xx2 = Math.min(ix2, faceJ.facial_area[2]);
                const yy2 = Math.min(iy2, faceJ.facial_area[3]);
                
                // Add +1 to match Python implementation
                const w = Math.max(0.0, xx2 - xx1 + 1);
                const h = Math.max(0.0, yy2 - yy1 + 1);
                const inter = w * h;
                const ovr = inter / (iarea + areas[j] - inter);
                
                // Suppress boxes with IoU above threshold
                if (ovr >= nmsThreshold) {
                    suppressed[j] = 1;
                }
            }
        }
        return keep;
    }

    /**
     * Process the image for face detection
     * @param imageData - Image data to process
     * @param targetSize - Target size for the shorter side of the image
     * @param maxSize - Maximum size for the longer side of the image
     * @param rect - Optional rectangle to crop the image before processing
     */
    protected async processImage(
        imageData: ImageData,
        targetSize: number,
        maxSize: number,
        rect: { left?: number; top?: number; width?: number; height?: number } = {}
    ): Promise<[ImageData, number]> {
        // Pure TypeScript crop and resize (bilinear interpolation)
        const cropRegion = {
            left: rect.left || 0,
            top: rect.top || 0,
            width: rect.width || imageData.width,
            height: rect.height || imageData.height
        };
        const imSizeMin = Math.min(cropRegion.width, cropRegion.height);
        const imSizeMax = Math.max(cropRegion.width, cropRegion.height);
        let imScale = targetSize / imSizeMin;
        if (Math.round(imScale * imSizeMax) > maxSize) {
            imScale = maxSize / imSizeMax;
        }
        
        // Calculate new dimensions based on the scale
        const newWidth = Math.round(cropRegion.width * imScale);
        const newHeight = Math.round(cropRegion.height * imScale);

        // Use common crop and resize function
        const dst = cropAndResizeBilinear(
            imageData,
            cropRegion.left,
            cropRegion.top,
            cropRegion.width,
            cropRegion.height,
            newWidth,
            newHeight
        );

        const outImageData: ImageData = {
            data: dst,
            width: newWidth,
            height: newHeight
        };

        return [outImageData, imScale];
    }

    /**
     * Detect faces in the image
     * @param imageData - Image data to process
     * @param targetSize - Target size for the shorter side of the image
     * @param maxSize - Maximum size for the longer side of the image
     * @param probThreshold - Probability threshold for face detection
     * @param nmsThreshold - Non-maximum suppression threshold
     */
    public async detect(
        imageData: ImageData,
        targetSize = 480,
        maxSize = 1980,
        probThreshold = 0.9,
        nmsThreshold = 0.4
    ): Promise<FaceDetectionResult[]> {
        // Process the image using dynamic scaling
        const [processedImage, processingScale] = await this.processImage(imageData, targetSize, maxSize);
        
        // Convert image data to Float32 tensor data
        const data = new Float32Array(processedImage.width * processedImage.height * 3);
        // Convert RGBA to BGR format for NHWC to match Python implementation (ignore alpha)
        for (let i = 0; i < processedImage.width * processedImage.height; i++) {
            // processedImage.data is RGBA, so channels are [R,G,B,A]
            data[i * 3 + 0] = processedImage.data[i * 4 + 2]; // B (from R)
            data[i * 3 + 1] = processedImage.data[i * 4 + 1]; // G
            data[i * 3 + 2] = processedImage.data[i * 4 + 0]; // R (from B)
        }
        
        const inputName = this.session.inputNames[0];
        const inputTensor = new Tensor('float32', data, [1, processedImage.height, processedImage.width, 3]);
        const results = await this.session.run({ [inputName]: inputTensor });
        const faceProposals: FaceDetectionResult[] = [];
        
        // Process each stride with different anchor sizes
        await this.processStride(results, faceProposals, probThreshold, 32, RetinaFace._anchors_fpn.stride32);
        await this.processStride(results, faceProposals, probThreshold, 16, RetinaFace._anchors_fpn.stride16);
        await this.processStride(results, faceProposals, probThreshold, 8, RetinaFace._anchors_fpn.stride8);
        
        const scaledProposals = faceProposals.map(obj => {
            // Clip and scale coordinates before NMS
            const x0 = Math.max(Math.min(obj.facial_area[0], processedImage.width - 1), 0) / processingScale;
            const y0 = Math.max(Math.min(obj.facial_area[1], processedImage.height - 1), 0) / processingScale;
            const x1 = Math.max(Math.min(obj.facial_area[2], processedImage.width - 1), 0) / processingScale;
            const y1 = Math.max(Math.min(obj.facial_area[3], processedImage.height - 1), 0) / processingScale;
            return {
                ...obj,
                facial_area: [x0, y0, x1, y1] as [number, number, number, number],
                landmarks: {
                    right_eye: [obj.landmarks.right_eye[0] / processingScale, obj.landmarks.right_eye[1] / processingScale] as [number, number],
                    left_eye: [obj.landmarks.left_eye[0] / processingScale, obj.landmarks.left_eye[1] / processingScale] as [number, number],
                    nose: [obj.landmarks.nose[0] / processingScale, obj.landmarks.nose[1] / processingScale] as [number, number],
                    mouth_right: [obj.landmarks.mouth_right[0] / processingScale, obj.landmarks.mouth_right[1] / processingScale] as [number, number],
                    mouth_left: [obj.landmarks.mouth_left[0] / processingScale, obj.landmarks.mouth_left[1] / processingScale] as [number, number]
                }
            };
        });
        scaledProposals.sort((a, b) => b.score - a.score);
        const picked = this.nmsSortedBboxes(scaledProposals as FaceDetectionResult[], nmsThreshold);
        
        // Get final results from picked indices
        const result = picked.map(i => scaledProposals[i]);
        
        return result;
    }
}

