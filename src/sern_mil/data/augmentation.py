"""Training-only orientation and photometric augmentation from preprocessing."""

from __future__ import annotations

import cv2
import numpy as np


def dominant_orientation(image):
    gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    xx = cv2.GaussianBlur(ix * ix, (0, 0), 1).mean()
    yy = cv2.GaussianBlur(iy * iy, (0, 0), 1).mean()
    xy = cv2.GaussianBlur(ix * iy, (0, 0), 1).mean()
    return float(np.degrees(0.5 * np.arctan2(2 * xy, xx - yy)))


def rotation_angles(dominant_degrees, gp4_fraction=0, gp5_fraction=0):
    high_grade = gp5_fraction >= 0.10 or gp4_fraction >= 0.35
    step, count = (22.5, 16) if high_grade else (45, 8)
    return [k * step - dominant_degrees for k in range(count)]


def rotate_tile(image, degrees, labels=False):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), degrees, 1)
    source = image.astype(np.float32) if labels else image
    result = cv2.warpAffine(
        source,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST if labels else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0 if labels else (255, 255, 255),
    )
    return result.astype(image.dtype)


def photometric_jitter(image, seed):
    rng = np.random.default_rng(seed)
    x = image.astype(np.float32) / 255
    if rng.random() < 0.5:
        x *= rng.uniform(0.8, 1.2)
    if rng.random() < 0.5:
        mean = x.mean((0, 1), keepdims=True)
        x = (x - mean) * rng.uniform(0.8, 1.2) + mean
    x = np.clip(x, 0, 1)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    if rng.random() < 0.3:
        hsv[..., 1] *= rng.uniform(0.85, 1.15)
    if rng.random() < 0.3:
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-0.05, 0.05) * 360) % 360
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 1)
    x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if rng.random() < 0.3:
        x = np.clip(x, 0, 1) ** rng.uniform(0.9, 1.1)
    if rng.random() < 0.5:
        x *= rng.uniform(0.9, 1.1, (1, 1, 3))
    if rng.random() < 0.2:
        x += rng.normal(0, rng.uniform(0, 0.02), x.shape).astype(np.float32)
    x = np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)
    if rng.random() < 0.2:
        x = cv2.medianBlur(x, 3)
    return x
