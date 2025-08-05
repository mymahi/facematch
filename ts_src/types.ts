/**
 * Common types used across the face detection modules
 */

/**
 * Interface representing raw RGBA pixel data with its dimensions
 */
export interface ImageData {
    data: Uint8Array;
    width: number;
    height: number;
}

/**
 * Interface for facial area regions
 */
export interface FacialArea {
    x: number;
    y: number;
    w: number;
    h: number;
    left_eye: number[];
    right_eye: number[];
    nose: number[];
    mouth_left: number[];
    mouth_right: number[];
    confidence: number;
}

export type FaceRect = [x: number, y: number, width: number, height: number];
export type FaceLandmarks = {
    left_eye: number[];
    right_eye: number[];
    nose: number[];
    mouth_left: number[];
    mouth_right: number[];
};
export type FacePose = {
    roll: number;
    yaw: number;
    pitch: number;
}

/**
 * Interface for extracted face results
 */
export interface FaceResult {
    /**
     * The extracted face raw image data
     */
    extract: ImageData;
    /**
     * The bounding box of the detected face in the original image
     */
    rect: FaceRect;
    /**
     * The facial landmarks detected in the face image
     */
    landmarks: FaceLandmarks;
    /**
     * Confidence score of the face detection
     */
    confidence: number;
    /**
     * The pose of the face, including roll, yaw, and pitch angles
     */
    pose: FacePose;
    /**
     * Indicates if the face is real or spoofed
     */
    isReal: boolean;
    /**
     * Anti-spoofing score
     */
    antispoofScore: number;
    /**
     * Embedding vector representing the face
     */
    embedding: number[];
}
