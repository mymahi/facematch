import sharp from 'sharp';
import { InferenceSession } from 'onnxruntime-node';
import RetinaFace from './retinaFace';
import FasNet from './fasNet';
import Facenet from './facenet';
import { 
    extractRegion, 
    alignImgWrtAngle,
    projectFacialArea,
    extractSubImage
} from './image_utils';
import { ImageData, FacialArea, FaceResult, FaceLandmarks, FaceRect, FacePose } from './types';

export interface ProcessImageOptions {
    imgData: Uint8Array | Buffer;
    /**
     * How much to expand the facial area (as a percentage)
     */
    expandPercentage?: number;
    /**
     * Maximum number of faces to detect, or null for all faces
     */
    maxFaces?: number | null;
    /**
     * Whether to use fast detection
     */
    fastDetection?: boolean;
}

/**
 * Options for detecting faces in an image
 */
export interface DetectFacesOptions {
    /**
     * The image data to process
     */
    imageData: ImageData;
    /**
     * How much to expand the facial area (as a percentage)
     */
    expandPercentage?: number;
    /**
     * Maximum number of faces to detect, or null for all faces
     */
    maxFaces?: number | null;
    /**
     * Whether to use fast detection
     */
    fastDetection?: boolean;
}

export class FaceMatch {
    private retinafaceSession: InferenceSession;
    private fasnetV1Session: InferenceSession;
    private fasnetV2Session: InferenceSession;
    private facenetSession: InferenceSession;

    /**
     * Create a new FaceMatch instance
     * @param retinafaceSession - The ONNX inference session for the RetinaFace model
     * @param fasnetV1Session - The ONNX inference session for the FasNet V1SE model
     * @param fasnetV2Session - The ONNX inference session for the FasNet V2 model
     * @param facenetSession - The ONNX inference session for the FaceNet model
     */
    constructor(
        retinafaceSession: InferenceSession,
        fasnetV1Session: InferenceSession,
        fasnetV2Session: InferenceSession,
        facenetSession: InferenceSession
    ) {
        this.retinafaceSession = retinafaceSession;
        this.fasnetV1Session = fasnetV1Session;
        this.fasnetV2Session = fasnetV2Session;
        this.facenetSession = facenetSession;
    }

    /**
     * Create and initialize a new FaceMatch with all models loaded
     * @returns A fully initialized FaceMatch instance
     */
    public static async create(): Promise<FaceMatch> {
        // RetinaFace Model Optimized Configuration
        const retinafaceSessionOptions: InferenceSession.SessionOptions = {
            graphOptimizationLevel: 'all',
            executionMode: 'sequential',
            intraOpNumThreads: 4,
            enableCpuMemArena: true,
            enableMemPattern: true,
            executionProviders: ['cpu']
        };

        // FasNet Models Optimized Configuration
        const fasnetSessionOptions: InferenceSession.SessionOptions = {
            graphOptimizationLevel: 'all',
            executionMode: 'sequential',
            intraOpNumThreads: 4,
            executionProviders: ['cpu']
        };

        // Facenet Model Optimized Configuration
        const facenetSessionOptions: InferenceSession.SessionOptions = {
            graphOptimizationLevel: 'all',
            executionMode: 'sequential',
            intraOpNumThreads: 4,
            executionProviders: ['cpu']
        };

        const retinafaceSession = await InferenceSession.create('./exported_models/retinaface.onnx', retinafaceSessionOptions);
        const fasnetV1Session = await InferenceSession.create('./exported_models/fasnet_v1se.onnx', fasnetSessionOptions);
        const fasnetV2Session = await InferenceSession.create('./exported_models/fasnet_v2.onnx', fasnetSessionOptions);
        const facenetSession = await InferenceSession.create('./exported_models/facenet512.onnx', facenetSessionOptions);

        return new FaceMatch(
            retinafaceSession, 
            fasnetV1Session, 
            fasnetV2Session,
            facenetSession
        );
    }

    /**
     * Create and initialize a new FaceMatch with custom model paths
     * @param retinafacePath - Path to the RetinaFace ONNX model
     * @param fasnetV1Path - Path to the FasNet V1SE ONNX model
     * @param fasnetV2Path - Path to the FasNet V2 ONNX model
     * @param facenetPath - Path to the FaceNet ONNX model
     * @returns A fully initialized FaceMatch instance
     */
    public static async createWithModels(
        retinafacePath: string,
        fasnetV1Path: string,
        fasnetV2Path: string,
        facenetPath: string
    ): Promise<FaceMatch> {
        // RetinaFace Model Optimized Configuration
        const retinafaceSessionOptions: InferenceSession.SessionOptions = {
            graphOptimizationLevel: 'all',
            executionMode: 'sequential',
            intraOpNumThreads: 4,
            enableCpuMemArena: true,
            enableMemPattern: true,
            executionProviders: ['cpu']
        };

        // FasNet Models Optimized Configuration
        const fasnetSessionOptions: InferenceSession.SessionOptions = {
            graphOptimizationLevel: 'all',
            intraOpNumThreads: 4,
            executionProviders: ['cpu']
        };

        // Facenet Model Optimized Configuration
        const facenetSessionOptions: InferenceSession.SessionOptions = {
            graphOptimizationLevel: 'all',
            intraOpNumThreads: 4,
            enableCpuMemArena: true,
            enableMemPattern: true,
            executionProviders: ['cpu']
        };

        const retinafaceSession = await InferenceSession.create(retinafacePath, retinafaceSessionOptions);
        const fasnetV1Session = await InferenceSession.create(fasnetV1Path, fasnetSessionOptions);
        const fasnetV2Session = await InferenceSession.create(fasnetV2Path, fasnetSessionOptions);
        const facenetSession = await InferenceSession.create(facenetPath, facenetSessionOptions);
        
        return new FaceMatch(
            retinafaceSession, 
            fasnetV1Session, 
            fasnetV2Session,
            facenetSession
        );
    }

    /**
     * Check if a landmark coordinate is within valid image bounds.
     */
    public static isValidLandmark(
        coord: number[],
        width: number,
        height: number
    ): boolean {
        if (!coord) return false;
        if (!Array.isArray(coord) || coord.length !== 2) return false;
        const [x, y] = coord;
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    /**
     * Find the pose of a face based on landmark coordinates
     * @param points - Coordinates of landmarks for the selected faces
     * @returns [angle, yaw, pitch] - Rotation angle in degrees and frontal measures
     */
    private findPose(landmarks: FaceLandmarks): { roll: number; yaw: number; pitch: number } {
        // Order: right_eye, left_eye, nose, mouth_right, mouth_left
        const LMx = [
            landmarks.right_eye[0],
            landmarks.left_eye[0],
            landmarks.nose[0],
            landmarks.mouth_right[0],
            landmarks.mouth_left[0]
        ];
        const LMy = [
            landmarks.right_eye[1],
            landmarks.left_eye[1],
            landmarks.nose[1],
            landmarks.mouth_right[1],
            landmarks.mouth_left[1]
        ];

        // Calculate roll angle (in degrees)
        const dPx_eyes = LMx[1] - LMx[0];
        const dPy_eyes = LMy[1] - LMy[0];
        const angle = Math.atan2(dPy_eyes, dPx_eyes);
        const roll = -angle * 180 / Math.PI;

        // Rotate landmarks to correct roll, using nose as center
        const [centerX, centerY] = [LMx[2], LMy[2]];
        const cosA = Math.cos(-angle);
        const sinA = Math.sin(-angle);
        const LMxr = [];
        const LMyr = [];
        for (let i = 0; i < LMx.length; i++) {
            const dx = LMx[i] - centerX;
            const dy = LMy[i] - centerY;
            LMxr.push(centerX + dx * cosA - dy * sinA);
            LMyr.push(centerY + dx * sinA + dy * cosA);
        }

        // Average distances for yaw/pitch estimation
        const dXtot = ((LMxr[1] - LMxr[0]) + (LMxr[4] - LMxr[3])) * 0.5;
        const dYtot = ((LMyr[3] - LMyr[0]) + (LMyr[4] - LMyr[1])) * 0.5;
        const dXnose = ((LMxr[1] - LMxr[2]) + (LMxr[4] - LMxr[2])) * 0.5;
        const dYnose = ((LMyr[3] - LMyr[2]) + (LMyr[4] - LMyr[2])) * 0.5;

        // Frontal measures
        const Xfrontal = dXtot !== 0 ? (-90 + 180 * dXnose / dXtot) : 0;
        const Yfrontal = dYtot !== 0 ? (-90 + 180 * dYnose / dYtot) : 0;

        return { roll, yaw: Xfrontal, pitch: Yfrontal };
    }

    /**
     * Extract a face region from an image.
     */
    private async extractFaceImageData({
        facialArea,
        pose,
        imageData,
        expandPercentage
    }: {
        facialArea: FacialArea;
        pose: FacePose;
        imageData: ImageData;
        expandPercentage: number;
    }): Promise<ImageData> {
        let { x, y, w, h } = facialArea;

        // Apply expansion if needed
        if (expandPercentage > 0) {
            const expandedW = w + Math.floor(w * expandPercentage / 100);
            const expandedH = h + Math.floor(h * expandPercentage / 100);
            x = Math.max(0, x - Math.floor((expandedW - w) / 2));
            y = Math.max(0, y - Math.floor((expandedH - h) / 2));
            w = Math.min(imageData.width - x, expandedW);
            h = Math.min(imageData.height - y, expandedH);
        }

        // Extract the unaligned face region from the raw image data
        let detectedFace = extractRegion(
            imageData,
            x, y, w, h
        );

        // Always align is true
        // Extract a sub-image with margin for alignment
        const { subImg, subImgWidth, subImgHeight, relativeX, relativeY } = await extractSubImage({
            imageData: imageData,
            facialArea: [x, y, w, h]
        });

        // Align the sub-image with respect to the eyes
        const { alignedSubImg, alignedImgHeight, alignedImgWidth } = await alignImgWrtAngle({
            imageData: {
                data: subImg,
                width: subImgWidth,
                height: subImgHeight
            },
            angle: pose.roll, // Use the roll angle from the pose
        });

        // Project the facial area coordinates after rotation
        const [rotatedX1, rotatedY1, rotatedX2, rotatedY2] = projectFacialArea({
            facialArea: [relativeX, relativeY, relativeX + w, relativeY + h],
            angle: pose.roll,
            size: [subImgHeight, subImgWidth],
            newSize: [alignedImgHeight, alignedImgWidth]
        });

        // Calculate rotated width and height
        const rotatedW = Math.abs(rotatedX2 - rotatedX1);
        const rotatedH = Math.abs(rotatedY2 - rotatedY1);

        // Extract the aligned face from the rotated sub-image
        detectedFace = extractRegion(
            {
                data: alignedSubImg,
                width: alignedImgWidth,
                height: alignedImgHeight
            },
            rotatedX1,
            rotatedY1,
            rotatedW,
            rotatedH
        );

        // Return the extracted face image data
        return {
            data: detectedFace,
            width: rotatedW,
            height: rotatedH
        };
    }

    /**
     * Detect faces in an image
     * @param options - Options for face detection
     */
    public async detectFaces({
        imageData,
        expandPercentage = 0,
        maxFaces = null,
        fastDetection = true
    }: DetectFacesOptions): Promise<{ facialExtract: ImageData; faceRect: FaceRect; landmarks: FaceLandmarks; confidence: number; pose: FacePose }[]> {
        // Granular performance tracking for large images
        const { data: img, width, height } = imageData;

        if (expandPercentage < 0) {
            throw new Error(
                `Expand percentage cannot be negative but you set it to ${expandPercentage}.`
            );
        }

        const heightBorder = Math.floor(0.25 * height);
        const widthBorder = Math.floor(0.25 * width);

        let paddedImg: Uint8Array;
        const channels = 4; // Now assuming RGBA format

        // Always align is true
        paddedImg = new Uint8Array((width + 2 * widthBorder) * (height + 2 * heightBorder) * channels);
        const paddedWidth = width + 2 * widthBorder;
        const paddedImg32 = new Uint32Array(paddedImg.buffer);

        // Fill entire image with opaque black
        paddedImg32.fill(0xFF000000);
        // Copy valid image rows into center
        for (let y = 0; y < height; y++) {
            const srcRowStart = y * width * channels;
            const dstRowStart = ((y + heightBorder) * paddedWidth + widthBorder) * channels;
            paddedImg.set(img.subarray(srcRowStart, srcRowStart + width * channels), dstRowStart);
        }

        const detectorImageData: ImageData = {
            data: paddedImg,
            width: width + 2 * widthBorder,
            height: height + 2 * heightBorder
        };

        const retinaface = new RetinaFace(this.retinafaceSession);

        const targetSize = fastDetection ? 240 : 640; // Use fast detection for better performance
        const faces = await retinaface.detect(detectorImageData, targetSize);

        let facialAreas: FacialArea[] = faces.map(face => {
            const [x1, y1, x2, y2] = face.facial_area;
            const x = Math.round(x1);
            const y = Math.round(y1);
            const w = Math.round(x2 - x1);
            const h = Math.round(y2 - y1);
            return {
                x,
                y,
                w,
                h,
                left_eye: [Math.round(face.landmarks.left_eye[0]), Math.round(face.landmarks.left_eye[1])],
                right_eye: [Math.round(face.landmarks.right_eye[0]), Math.round(face.landmarks.right_eye[1])],
                nose: [Math.round(face.landmarks.nose[0]), Math.round(face.landmarks.nose[1])],
                mouth_left: [Math.round(face.landmarks.mouth_left[0]), Math.round(face.landmarks.mouth_left[1])],
                mouth_right: [Math.round(face.landmarks.mouth_right[0]), Math.round(face.landmarks.mouth_right[1])],
                confidence: face.score,
            };
        });

        if (maxFaces && maxFaces < facialAreas.length) {
            facialAreas.sort((a, b) => (b.w * b.h) - (a.w * a.h));
            facialAreas = facialAreas.slice(0, maxFaces);
        }

        const results = await Promise.all(facialAreas.map(async (facialArea) => {
            // Restore x, y coordinates (adjusted for border)
            const faceRect: FaceRect = [facialArea.x - widthBorder, facialArea.y - heightBorder, facialArea.w, facialArea.h];

            // Adjust landmark coordinates for the border
            const landmarks: FaceLandmarks = {
                left_eye: [facialArea.left_eye[0] - widthBorder, facialArea.left_eye[1] - heightBorder],
                right_eye: [facialArea.right_eye[0] - widthBorder, facialArea.right_eye[1] - heightBorder],
                nose: [facialArea.nose[0] - widthBorder, facialArea.nose[1] - heightBorder],
                mouth_left: [facialArea.mouth_left[0] - widthBorder, facialArea.mouth_left[1] - heightBorder],
                mouth_right: [facialArea.mouth_right[0] - widthBorder, facialArea.mouth_right[1] - heightBorder]
            };

            const pose = this.findPose(landmarks);

            const facialExtract = await this.extractFaceImageData({
                facialArea,
                pose,
                imageData: detectorImageData,
                expandPercentage,
            });

            return {
                facialExtract,
                faceRect,
                landmarks,
                confidence: facialArea.confidence,
                pose
            }
        }));

        return results;
    }

    
    /**
     * Analyze if a face is real or spoofed.
     * 
     * @param imageData - The original image data containing the face
     * @param faceRect - The face rectangle coordinates [x, y, width, height]
     * @returns Object containing isReal flag and antispoofScore
     */
    public async analyzeAntispoof(
        imageData: ImageData, 
        faceRect: [number, number, number, number]
    ): Promise<{ isReal: boolean; antispoofScore: number }> {
        // Create FasNet model instance
        const fasNet = new FasNet(
            this.fasnetV1Session,
            this.fasnetV2Session
        );
        
        // Run the anti-spoofing analysis
        return await fasNet.analyze(imageData, faceRect);
    }

    /**
     * Represent an extracted facial image as multi-dimensional vector embedding.
     * 
     * @param facialExtract - The extracted face image data
     * 
     * @returns Result with embedding vector and facial area information
     */
    public async represent(facialExtract: ImageData): Promise<number[]> {
        // Create facenet model instance
        const model = new Facenet(this.facenetSession);
        
        // Get embedding vector from the model
        const rawEmbedding = await model.imageVectorization(facialExtract);
        
        // Convert Float32Array to number[] if needed
        const embedding = Array.from(rawEmbedding);
        
        // Return the result for this face
        return embedding;
    }

    /**
     * Extract faces from given image data using the default detector backend.
     * @param imgData - Buffer or Uint8Array containing the image data
     * @param align - Whether to align detected faces based on eye positions
     * @param expandPercentage - How much to expand the facial area (as a percentage)
     * @param maxFaces - Maximum number of faces to detect, or null for all faces
     * @param fastDetection - Whether to use fast detection
     * @returns Array of detected face objects with image data and metadata
     */
    public async analyzeImage({
        imgData,
        expandPercentage = 0,
        maxFaces = null,
        fastDetection = true
    }: ProcessImageOptions): Promise<FaceResult[]> {
        const sharpImg = sharp(imgData);
        const metadata = await sharpImg.metadata();
        const height = metadata.height!;
        const width = metadata.width!;

        if (!sharpImg || !height || !width) {
            throw new Error('Exception while loading image data');
        }

        // Convert Sharp to 4 channel raw Uint8Array
        const rawImgBuffer = await sharpImg.ensureAlpha().raw().toBuffer();
        const imageData: ImageData = {
            data: new Uint8Array(rawImgBuffer),
            width,
            height
        };

        // Use RetinaFace for face detection
        const faceObjs = await this.detectFaces({
            imageData,
            expandPercentage,
            maxFaces,
            fastDetection
        });

        const respObjs: FaceResult[] = [];
        for (const faceObj of faceObjs) {
            // Get the original image data
            let currentImg = faceObj.facialExtract.data;
            let currentImgWidth = faceObj.facialExtract.width;
            let currentImgHeight = faceObj.facialExtract.height;

            // Skip images with invalid dimensions
            if (currentImgWidth <= 0 || currentImgHeight <= 0) {
                continue;
            }

            // Perform anti-spoofing analysis
            const { isReal, antispoofScore } = await this.analyzeAntispoof(
                imageData,
                faceObj.faceRect
            );

            // Create embedding vector
            const embedding = await this.represent(faceObj.facialExtract);

            const respObj: FaceResult = {
                extract: {
                    data: currentImg,
                    width: currentImgWidth,
                    height: currentImgHeight
                },
                rect: faceObj.faceRect,
                landmarks: faceObj.landmarks,
                confidence: Math.round(Number(faceObj.confidence || 0) * 100) / 100,
                pose: faceObj.pose,
                isReal,
                antispoofScore,
                embedding
            };

            respObjs.push(respObj);
        }

        return respObjs;
    }
}