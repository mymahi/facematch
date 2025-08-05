import { promises as fs } from 'fs';
import * as path from 'path';
import sharp from 'sharp';
import { FaceMatch } from './faceMatch';

/**
 * Test script for FaceMatch functionality
 * 
 * This script demonstrates:
 * 1. Loading the FaceMatch class with ONNX models
 * 2. Detecting faces in images
 * 3. Checking if detected faces are real (anti-spoofing)
 * 4. Generating face embeddings
 * 5. Comparing face embeddings for similarity
 */

async function main() {
    try {
        console.time('Total Test Time');
        console.time('Model Load Time');
        console.log('Loading FaceMatch with default model paths...');
        const faceMatch = await FaceMatch.create();
        console.timeEnd('Model Load Time');

        console.time('Image Load Time');
        console.log('Loading test images...');
        // Dynamically get all files in the test_photos subfolder
        const testPhotosDir = './test_photos';
        const files = await fs.readdir(testPhotosDir);
        const imagePaths = files
            .filter(f => /\.(jpg|jpeg|png|webp)$/i.test(f))
            .map(f => path.join(testPhotosDir, f));
        const imageBuffers: Buffer[] = [];
        for (const imagePath of imagePaths) {
            try {
                const imageBuffer = await fs.readFile(imagePath);
                imageBuffers.push(imageBuffer);
                console.log(`Loaded ${imagePath}`);
            } catch (error) {
                console.error(`Error loading ${imagePath}:`, error);
            }
        }
        console.timeEnd('Image Load Time');

        if (imageBuffers.length === 0) {
            throw new Error('No images could be loaded');
        }

        const allResults = [];
        for (let i = 0; i < imageBuffers.length; i++) {
            console.log(`\nProcessing image: ${imagePaths[i]}`);
            console.time(`Face Detection & Extraction Time for ${imagePaths[i]}`);
            const faceResults = await faceMatch.analyzeImage({
                imgData: imageBuffers[i],
                expandPercentage: 10,
                maxFaces: 1,
                fastDetection: true
            });
            console.timeEnd(`Face Detection & Extraction Time for ${imagePaths[i]}`);

            console.log(`Found ${faceResults.length} faces in ${imagePaths[i]}`);

            if (faceResults.length > 0) {
                const face = faceResults[0];
                // Anti-spoofing and embedding are already included in processImage, but we can time them separately if needed
                // For demonstration, let's time embedding extraction again
                console.time(`Embedding Extraction Time for ${imagePaths[i]}`);
                const embedding = await faceMatch.represent(face.extract);
                console.timeEnd(`Embedding Extraction Time for ${imagePaths[i]}`);

                console.log(`Face detection confidence: ${face.confidence}`);
                console.log(`Face pose - roll: ${face.pose.roll.toFixed(2)}°, yaw: ${face.pose.yaw.toFixed(2)}°, pitch: ${face.pose.pitch.toFixed(2)}°`);
                console.log(`Is real face: ${face.isReal} (antispoofing score: ${face.antispoofScore.toFixed(4)})`);
                if (face.rect && Array.isArray(face.rect) && face.rect.length === 4) {
                    console.log(`Face rect: [${face.rect.map(v => v.toFixed(2)).join(', ')}]`);
                }
                if (face.landmarks) {
                    console.log('Landmarks:');
                    const keys: (keyof typeof face.landmarks)[] = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right'];
                    for (const key of keys) {
                        const pt = face.landmarks[key];
                        if (Array.isArray(pt) && pt.length === 2) {
                            console.log(`  ${key}: [${pt[0].toFixed(2)}, ${pt[1].toFixed(2)}]`);
                        }
                    }
                }

                try {
                    console.time(`Annotated Image Save Time for ${imagePaths[i]}`);
                    const origImageBuffer = imageBuffers[i];
                    const origSharp = sharp(origImageBuffer);
                    const metadata = await origSharp.metadata();
                    const width = metadata.width || 0;
                    const height = metadata.height || 0;
                    if (width > 0 && height > 0 && face.rect && face.landmarks) {
                        const [x, y, w, h] = face.rect;
                        const boxColor = '#00ff00';
                        const boxStroke = 3;
                        const landmarkColors = ['#00eaff', '#ff00ea', '#ffe600', '#00ff5a', '#ff5a00'];
                        const radius = Math.max(3, Math.round(Math.min(width, height) / 80));
                        const keys: (keyof typeof face.landmarks)[] = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right'];
                        const circles = keys.map((key, idx) => {
                            const pt = face.landmarks[key];
                            if (Array.isArray(pt) && pt.length === 2) {
                                return `<circle cx="${pt[0]}" cy="${pt[1]}" r="${radius}" fill="${landmarkColors[idx % landmarkColors.length]}" stroke="white" stroke-width="1" />`;
                            }
                            return '';
                        }).join('\n');
                        const rectSvg = `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="${boxColor}" stroke-width="${boxStroke}" />`;
                        const svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">${rectSvg}${circles}</svg>`;
                        const resultsDir = './test_results';
                        await fs.mkdir(resultsDir, { recursive: true });
                        const annotatedFilename = path.join(resultsDir, `annotated_${path.basename(imagePaths[i])}`);
                        await origSharp.composite([{ input: Buffer.from(svg), top: 0, left: 0 }]).toFile(annotatedFilename);
                        console.log(`Saved annotated image to ${annotatedFilename}`);
                    }
                    console.timeEnd(`Annotated Image Save Time for ${imagePaths[i]}`);
                } catch (err) {
                    console.error('Error saving annotated image:', err);
                }

                const resultsDir = './test_results';
                await fs.mkdir(resultsDir, { recursive: true });
                const extractedFilename = path.join(resultsDir, `extracted_${path.basename(imagePaths[i])}`);
                let landmarkPoints: { x: number, y: number }[] = [];
                if (face.landmarks && face.rect && face.extract && face.pose && face.extract.width && face.extract.height) {
                    const keys: (keyof typeof face.landmarks)[] = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right'];
                    const rect = face.rect;
                    const roll = face.pose.roll || 0;
                    const angleRad = roll * Math.PI / 180;
                    const cropW = rect[2];
                    const cropH = rect[3];
                    const cx = cropW / 2;
                    const cy = cropH / 2;
                    const corners = [
                        [0, 0],
                        [cropW, 0],
                        [cropW, cropH],
                        [0, cropH]
                    ];
                    function rotatePoint(x: number, y: number, cx: number, cy: number, angleRad: number) {
                        const cos = Math.cos(angleRad);
                        const sin = Math.sin(angleRad);
                        const nx = cos * (x - cx) - sin * (y - cy) + cx;
                        const ny = sin * (x - cx) + cos * (y - cy) + cy;
                        return [nx, ny];
                    }
                    const rotatedCorners = corners.map(([x, y]) => rotatePoint(x, y, cx, cy, angleRad));
                    const xs = rotatedCorners.map(([x, _]) => x);
                    const ys = rotatedCorners.map(([_, y]) => y);
                    const minX = Math.floor(Math.min(...xs));
                    const minY = Math.floor(Math.min(...ys));
                    for (const key of keys) {
                        const pt = face.landmarks[key];
                        if (Array.isArray(pt) && pt.length === 2 && typeof pt[0] === 'number' && typeof pt[1] === 'number') {
                            let x = pt[0] - rect[0];
                            let y = pt[1] - rect[1];
                            let rotatedX = x;
                            let rotatedY = y;
                            if (Math.abs(roll) > 1e-3) {
                                const dx = x - cx;
                                const dy = y - cy;
                                rotatedX = Math.cos(angleRad) * dx - Math.sin(angleRad) * dy + cx;
                                rotatedY = Math.sin(angleRad) * dx + Math.cos(angleRad) * dy + cy;
                            }
                            rotatedX -= minX;
                            rotatedY -= minY;
                            landmarkPoints.push({ x: rotatedX, y: rotatedY });
                        }
                    }
                }
                console.time(`Extracted Face Save Time for ${imagePaths[i]}`);
                await saveExtractedFace(face.extract.data, face.extract.width, face.extract.height, extractedFilename, landmarkPoints);
                console.timeEnd(`Extracted Face Save Time for ${imagePaths[i]}`);
                console.log(`Saved extracted face to ${extractedFilename}`);

                allResults.push({
                    filename: imagePaths[i],
                    face: face,
                    embedding
                });
            }
        }

        if (allResults.length >= 2) {
            console.time('Face Comparison Time');
            console.log('\n--- Face Distance Comparison ---');
            for (let i = 0; i < allResults.length; i++) {
                for (let j = i + 1; j < allResults.length; j++) {
                    const distance = calculateCosineDistance(
                        allResults[i].face.embedding,
                        allResults[j].face.embedding
                    );
                    console.log(`Distance between ${allResults[i].filename} and ${allResults[j].filename}: ${distance.toFixed(4)}`);
                    console.log(`Match: ${distance < 0.37 ? 'YES' : 'NO'} (threshold: 0.37)`);
                    console.log(`Real: ${allResults[i].face.isReal && allResults[j].face.isReal ? 'YES' : 'NO'}`);
                }
            }
            console.timeEnd('Face Comparison Time');
        }

        console.timeEnd('Total Test Time');
    } catch (error) {
        console.error('Error in face matching test:', error);
    }
}

/**
 * Save an extracted face as an image file
 */
/**
 * Save an extracted face as an image file, drawing landmarks if provided
 */
async function saveExtractedFace(
    imageData: Uint8Array,
    width: number,
    height: number,
    filename: string,
    landmarks?: Array<{ x: number, y: number }>
): Promise<void> {
    try {
        let faceSharp = sharp(imageData, {
            raw: {
                width,
                height,
                channels: 4
            }
        });

        if (landmarks && Array.isArray(landmarks) && landmarks.length > 0) {
            // Assign a color to each landmark (order: left_eye, right_eye, nose, mouth_left, mouth_right)
            const landmarkColors = ['#00eaff', '#ff00ea', '#ffe600', '#00ff5a', '#ff5a00'];
            const radius = Math.max(2, Math.round(Math.min(width, height) / 40));
            const svgCircles = landmarks.map((l, i) => `<circle cx="${l.x}" cy="${l.y}" r="${radius}" fill="${landmarkColors[i % landmarkColors.length]}" stroke="white" stroke-width="1" />`).join('\n');
            const svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">${svgCircles}</svg>`;
            faceSharp = faceSharp.composite([{ input: Buffer.from(svg), top: 0, left: 0 }]);
        }

        await faceSharp.toFile(filename);
    } catch (error) {
        console.error(`Error saving extracted face to ${filename}:`, error);
    }
}

/**
 * Calculate cosine distance between two embedding vectors
 * Values close to 0 indicate similar faces (lower is more similar)
 * Distance = 1 - similarity
 */
function calculateCosineDistance(
    embedding1: number[],
    embedding2: number[]
): number {
    if (embedding1.length !== embedding2.length) {
        throw new Error('Embedding vectors must have the same length');
    }
    
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;
    
    for (let i = 0; i < embedding1.length; i++) {
        dotProduct += embedding1[i] * embedding2[i];
        norm1 += embedding1[i] * embedding1[i];
        norm2 += embedding2[i] * embedding2[i];
    }
    
    norm1 = Math.sqrt(norm1);
    norm2 = Math.sqrt(norm2);
    
    if (norm1 === 0 || norm2 === 0) {
        return 1; // Maximum distance when one vector is zero
    }
    
    // Convert similarity to distance: distance = 1 - similarity
    return 1 - (dotProduct / (norm1 * norm2));
}

// Run the test
main()
    .then(() => console.log('\nTest completed successfully'))
    .catch(error => console.error('Test failed:', error));
