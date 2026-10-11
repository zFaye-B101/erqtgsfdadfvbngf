"""Reusable SOP image transforms and feature extraction helpers.

This module is imported by 01_SOP_Preprocessing.ipynb and can also be imported
from a separate experiment script in the same project folder.
"""

import cv2
import numpy as np
from PIL import Image


HSV_BINS = 16
HOG_GRID = 4
HOG_BINS = 9
FEATURE_DIM = 3 * HSV_BINS + HOG_GRID * HOG_GRID * HOG_BINS


def resize_for_model(pil_image, size=224):
    """Aspect-preserving resize with white padding for a square model input."""
    image = pil_image.convert("RGB")
    width, height = image.size
    scale = min(size / float(width), size / float(height))
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    try:
        resampling = Image.Resampling.LANCZOS
    except AttributeError:  # Pillow versions before 9.1
        resampling = Image.LANCZOS
    resized = image.resize(new_size, resampling)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    offset = ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def image_feature_vector(rgb_image, feature_image_size=128):
    """Return 48 HSV histogram values and 144 spatial HOG values."""
    rgb = np.asarray(rgb_image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    color_parts = []
    channel_ranges = [(0, 180), (0, 256), (0, 256)]
    for channel, value_range in enumerate(channel_ranges):
        hist, _ = np.histogram(
            hsv[:, :, channel], bins=HSV_BINS, range=value_range
        )
        hist = hist.astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        color_parts.append(hist)

    small = cv2.resize(
        rgb, (feature_image_size, feature_image_size), interpolation=cv2.INTER_AREA
    )
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = np.mod(angle, 180.0)
    bin_index = np.floor(angle * HOG_BINS / 180.0).astype(np.int32)
    bin_index = np.clip(bin_index, 0, HOG_BINS - 1)

    hog_parts = []
    height, width = gray.shape
    for row in range(HOG_GRID):
        y0, y1 = row * height // HOG_GRID, (row + 1) * height // HOG_GRID
        for col in range(HOG_GRID):
            x0, x1 = col * width // HOG_GRID, (col + 1) * width // HOG_GRID
            bins = bin_index[y0:y1, x0:x1].ravel()
            weights = magnitude[y0:y1, x0:x1].ravel()
            hist = np.bincount(bins, weights=weights, minlength=HOG_BINS)
            hist = hist.astype(np.float32)
            hist /= max(float(np.linalg.norm(hist)), 1e-8)
            hog_parts.append(hist)

    vector = np.concatenate(color_parts + hog_parts).astype(np.float32)
    if vector.shape != (FEATURE_DIM,) or not np.isfinite(vector).all():
        raise ValueError("Image feature dimension or values are invalid.")
    return vector


def sift_pair_features(image_a, image_b, ratio=0.75, max_keypoints=1000):
    """Return SIFT match count, rate, median distance and spatial coverages.

    Call this only for pairs generated inside the relevant training or
    validation fold, after grouping the data by product ID.
    """
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    gray_a = cv2.cvtColor(np.asarray(image_a.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(np.asarray(image_b.convert("RGB")), cv2.COLOR_RGB2GRAY)
    key_a, desc_a = sift.detectAndCompute(gray_a, None)
    key_b, desc_b = sift.detectAndCompute(gray_b, None)

    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return np.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    candidates = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [m for pair in candidates if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]
    denominator = max(1, min(len(desc_a), len(desc_b)))
    median_distance = float(np.median([m.distance for m in good])) if good else 0.0

    def grid_coverage(points, shape, grid=4):
        height, width = shape[:2]
        occupied = set()
        for x, y in points:
            cell_x = min(grid - 1, int(x * grid / max(width, 1)))
            cell_y = min(grid - 1, int(y * grid / max(height, 1)))
            occupied.add(cell_y * grid + cell_x)
        return len(occupied) / float(grid * grid)

    points_a = [key_a[m.queryIdx].pt for m in good]
    points_b = [key_b[m.trainIdx].pt for m in good]
    return np.asarray([
        float(len(good)),
        float(len(good) / denominator),
        median_distance,
        grid_coverage(points_a, gray_a.shape),
        grid_coverage(points_b, gray_b.shape),
    ], dtype=np.float32)
