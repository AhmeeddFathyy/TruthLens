from __future__ import annotations
import os
import numpy as np
import cv2


def read_image_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def to_float01(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32)
    if x.max() > 1.5:
        x /= 255.0
    return np.clip(x, 0.0, 1.0)


def rgb_to_gray01(rgb01: np.ndarray) -> np.ndarray:
    # ITU-R BT.601
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.float32)


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)
