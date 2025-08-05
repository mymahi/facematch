import { ImageData } from './types';

/**
 * Crop and resize a region from RGBA image data using bilinear interpolation.
 * @param src RGBA image data (Uint8Array)
 * @param srcW Source image width
 * @param srcH Source image height
 * @param cropL Crop left
 * @param cropT Crop top
 * @param cropW Crop width
 * @param cropH Crop height
 * @param dstW Destination width
 * @param dstH Destination height
 * @returns Cropped and resized RGBA image data (Uint8Array)
 */
export function cropAndResizeBilinear(
    imageData: ImageData,
    cropL: number,
    cropT: number,
    cropW: number,
    cropH: number,
    dstW: number,
    dstH: number
): Uint8Array {
    const src = imageData.data;
    const srcW = imageData.width;
    const srcH = imageData.height;
    const dst = new Uint8Array(dstW * dstH * 4);

    const cropYMax = cropT + cropH - 1;
    const cropXMax = cropL + cropW - 1;
    for (let y = 0; y < dstH; y++) {
        const srcY = cropT + (y + 0.5) * cropH / dstH - 0.5;
        const y0 = srcY | 0; // Faster Math.floor
        const y1 = y0 + 1 > cropYMax ? cropYMax : y0 + 1;
        const wy = srcY - y0;
        for (let x = 0; x < dstW; x++) {
            const srcX = cropL + (x + 0.5) * cropW / dstW - 0.5;
            const x0 = srcX | 0;
            const x1 = x0 + 1 > cropXMax ? cropXMax : x0 + 1;
            const wx = srcX - x0;
            const dstBase = (y * dstW + x) * 4;
            for (let c = 0; c < 4; c++) {
                const idx00 = (y0 * srcW + x0) * 4 + c;
                const idx01 = (y0 * srcW + x1) * 4 + c;
                const idx10 = (y1 * srcW + x0) * 4 + c;
                const idx11 = (y1 * srcW + x1) * 4 + c;
                const v00 = src[idx00];
                const v01 = src[idx01];
                const v10 = src[idx10];
                const v11 = src[idx11];
                const v0 = v00 * (1 - wx) + v01 * wx;
                const v1 = v10 * (1 - wx) + v11 * wx;
                const v = v0 * (1 - wy) + v1 * wy;
                dst[dstBase + c] = Math.round(v);
            }
        }
    }

    return dst;
}

/**
 * Extract a region from an image
 * @param img Source image
 * @param y1 Start Y coordinate
 * @param y2 End Y coordinate
 * @param x1 Start X coordinate
 * @param x2 End X Coordinate
 * @param imgWidth Width of source image
 * @returns Extracted region
 */
export function sliceImage(
    img: Uint8Array,
    y1: number,
    y2: number,
    x1: number,
    x2: number,
    imgWidth: number,
): Uint8Array {
    const width = x2 - x1;
    const height = y2 - y1;
    const result = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) {
        const srcRowStart = ((y + y1) * imgWidth + x1) * 4;
        const dstRowStart = (y * width) * 4;
        result.set(img.subarray(srcRowStart, srcRowStart + width * 4), dstRowStart);
    }
    return result;
}

/**
 * Extract a region from an image.
 * @param imageData The source image data
 * @param x X coordinate of the region to extract
 * @param y Y coordinate of the region to extract
 * @param width Width of the region to extract
 * @param height Height of the region to extract
 * @returns A new Uint8Array containing only the extracted region
 */
export function extractRegion(
    imageData: ImageData,
    x: number,
    y: number,
    width: number,
    height: number
): Uint8Array {
    const result = new Uint8Array(width * height * 4);
    const result32 = new Uint32Array(result.buffer);

    // Calculate the valid region to copy
    const x1 = Math.max(0, x);
    const y1 = Math.max(0, y);
    const x2 = Math.min(imageData.width, x + width);
    const y2 = Math.min(imageData.height, y + height);
    const copyWidth = x2 - x1;
    const copyHeight = y2 - y1;
    const padLeft = x1 - x;
    const padTop = y1 - y;

    // Fast row-wise copy for valid region, fill pure padding rows and row ends with opaque black
    for (let row = 0; row < height; row++) {
        if (row >= padTop && row < padTop + copyHeight) {
            // Valid row to copy from source
            const srcRowStart = ((y1 + (row - padTop)) * imageData.width + x1) * 4;
            const dstRowStart = (row * width + padLeft) * 4;
            // Fill left padding
            if (padLeft > 0) {
                result32.fill(0xFF000000, row * width, row * width + padLeft);
            }
            // Copy valid region
            result.set(
                imageData.data.subarray(srcRowStart, srcRowStart + copyWidth * 4),
                dstRowStart
            );
            // Fill right padding
            const padRight = width - (padLeft + copyWidth);
            if (padRight > 0) {
                result32.fill(0xFF000000, row * width + padLeft + copyWidth, (row + 1) * width);
            }
        } else {
            // Entire row is padding: fill with opaque black
            result32.fill(0xFF000000, row * width, (row + 1) * width);
        }
    }
    return result;
}

/**
 * Align a given image horizontally with respect to their face angle.
 */
export async function alignImgWrtAngle({
    imageData,
    angle
}: {
    imageData: ImageData;
    angle: number;
}): Promise<{ alignedSubImg: Uint8Array; alignedImgWidth: number; alignedImgHeight: number }> {
    const width = imageData.width;
    const height = imageData.height;
    const cx = width / 2;
    const cy = height / 2;
    const rad = angle * Math.PI / 180.0;
    const cosA = Math.cos(rad);
    const sinA = Math.sin(rad);

    // Find bounding box of rotated image
    const corners = [
        [0, 0], [width, 0], [0, height], [width, height]
    ];
    const rotated = corners.map(([x, y]) => {
        const dx = x - cx;
        const dy = y - cy;
        return [
            cosA * dx - sinA * dy + cx,
            sinA * dx + cosA * dy + cy
        ];
    });
    const xs = rotated.map(([x, _]) => x);
    const ys = rotated.map(([_, y]) => y);
    const minX = Math.floor(Math.min(...xs));
    const maxX = Math.ceil(Math.max(...xs));
    const minY = Math.floor(Math.min(...ys));
    const maxY = Math.ceil(Math.max(...ys));
    const outW = maxX - minX;
    const outH = maxY - minY;

    const src = imageData.data;
    const dst = new Uint8Array(outW * outH * 4);

    for (let y = 0; y < outH; y++) {
        for (let x = 0; x < outW; x++) {
            // Map output (x, y) to input (srcX, srcY) using inverse rotation
            const outX = x + minX;
            const outY = y + minY;
            // Shift to center
            const dx = outX - cx;
            const dy = outY - cy;
            // Inverse rotation (rotate by -angle)
            const srcX =  cosA * dx + sinA * dy + cx;
            const srcY = -sinA * dx + cosA * dy + cy;
            const ix = Math.round(srcX);
            const iy = Math.round(srcY);
            const outIdx = (y * outW + x) * 4;
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                const inIdx = (iy * width + ix) * 4;
                dst[outIdx] = src[inIdx];
                dst[outIdx + 1] = src[inIdx + 1];
                dst[outIdx + 2] = src[inIdx + 2];
                dst[outIdx + 3] = src[inIdx + 3];
            } else {
                dst[outIdx] = 0;
                dst[outIdx + 1] = 0;
                dst[outIdx + 2] = 0;
                dst[outIdx + 3] = 255;
            }
        }
    }

    return {
        alignedSubImg: dst,
        alignedImgWidth: outW,
        alignedImgHeight: outH
    };
}

/**
 * Project facial area coordinates after rotation (for alignment).
 * Update pre-calculated facial area coordinates after image itself
 * rotated with respect to the eyes.
 */
export function projectFacialArea({
    facialArea,
    angle,
    size,
    newSize
}: {
    facialArea: [number, number, number, number];
    angle: number;
    size: [number, number];
    newSize: [number, number];
}): [number, number, number, number] {
    // If no rotation, just return the area limited to newSize
    if (!angle || angle % 360 === 0) {
        const [newH, newW] = newSize;
        const x1 = Math.max(Math.floor(facialArea[0]), 0);
        const y1 = Math.max(Math.floor(facialArea[1]), 0);
        const x2 = Math.min(Math.floor(facialArea[2]), newW);
        const y2 = Math.min(Math.floor(facialArea[3]), newH);
        return [x1, y1, x2, y2];
    }

    // Convert angle to radians
    const theta = angle * Math.PI / 180;
    const [height, width] = size;
    const [newH, newW] = newSize;

    // Get the four corners of the original face area
    const corners = [
        [facialArea[0], facialArea[1]],
        [facialArea[2], facialArea[1]],
        [facialArea[2], facialArea[3]],
        [facialArea[0], facialArea[3]]
    ];

    // Center of original image
    const cx = width / 2;
    const cy = height / 2;

    // Center of new image
    const ncx = newW / 2;
    const ncy = newH / 2;

    // For each corner, rotate about center and then shift to new image
    const rotated = corners.map(([x, y]) => {
        // Shift to center
        const dx = x - cx;
        const dy = y - cy;
        // Rotate by theta
        const rx =  Math.cos(theta) * dx + Math.sin(theta) * dy;
        const ry = -Math.sin(theta) * dx + Math.cos(theta) * dy;
        // Shift to new center
        return [rx + ncx, ry + ncy];
    });

    // Find bounding box of rotated corners
    const xs = rotated.map(([x, _]) => x);
    const ys = rotated.map(([_, y]) => y);
    const x1 = Math.max(Math.floor(Math.min(...xs)), 0);
    const y1 = Math.max(Math.floor(Math.min(...ys)), 0);
    const x2 = Math.min(Math.ceil(Math.max(...xs)), newW);
    const y2 = Math.min(Math.ceil(Math.max(...ys)), newH);
    return [x1, y1, x2, y2];
}

/**
 * Extract a sub-image region from an image, with margin.
 * This function doubles the height and width of the face region,
 * and adds black pixels if necessary.
 */
export async function extractSubImage({
    imageData,
    facialArea,
}: {
    imageData: ImageData;
    facialArea: [number, number, number, number];
}): Promise<{ subImg: Uint8Array; subImgWidth: number; subImgHeight: number; relativeX: number; relativeY: number }> {
    const [x, y, w, h] = facialArea;
    const relativeX = Math.floor(0.5 * w);
    const relativeY = Math.floor(0.5 * h);

    // Calculate expanded coordinates
    let x1 = x - relativeX;
    let y1 = y - relativeY;
    let x2 = x + w + relativeX;
    let y2 = y + h + relativeY;

    // Width and height of the expanded region
    const expandedWidth = w + 2 * relativeX;
    const expandedHeight = h + 2 * relativeY;

    // Clamp coordinates to image bounds
    const srcX1 = Math.max(0, x1);
    const srcY1 = Math.max(0, y1);
    const srcX2 = Math.min(imageData.width, x2);
    const srcY2 = Math.min(imageData.height, y2);
    const copyWidth = srcX2 - srcX1;
    const copyHeight = srcY2 - srcY1;

    // Calculate padding
    const padLeft = srcX1 - x1;
    const padTop = srcY1 - y1;

    // Allocate output buffer (leave uninitialized for speed)
    const subImg = new Uint8Array(expandedWidth * expandedHeight * 4);
    const subImg32 = new Uint32Array(subImg.buffer);

    // Fast row-wise copy for valid region, fill pure padding rows and row ends with opaque black
    for (let row = 0; row < expandedHeight; row++) {
        if (row >= padTop && row < padTop + copyHeight) {
            // Valid row to copy from source
            const srcRowStart = ((srcY1 + (row - padTop)) * imageData.width + srcX1) * 4;
            const dstRowStart = (row * expandedWidth + padLeft) * 4;
            // Fill left padding
            if (padLeft > 0) {
                subImg32.fill(0xFF000000, row * expandedWidth, row * expandedWidth + padLeft);
            }
            // Copy valid region
            subImg.set(
                imageData.data.subarray(srcRowStart, srcRowStart + copyWidth * 4),
                dstRowStart
            );
            // Fill right padding
            const padRight = expandedWidth - (padLeft + copyWidth);
            if (padRight > 0) {
                subImg32.fill(0xFF000000, row * expandedWidth + padLeft + copyWidth, (row + 1) * expandedWidth);
            }
        } else {
            // Entire row is padding: fill with opaque black
            subImg32.fill(0xFF000000, row * expandedWidth, (row + 1) * expandedWidth);
        }
    }

    return {
        subImg,
        subImgWidth: expandedWidth,
        subImgHeight: expandedHeight,
        relativeX,
        relativeY
    };
}
