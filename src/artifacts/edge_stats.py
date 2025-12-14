from __future__ import annotations
import numpy as np
import cv2
from ..utils import normalize01


def edge_features(gray01: np.ndarray) -> dict:
    g8 = (np.clip(gray01, 0, 1) * 255).astype(np.uint8)

    # Laplacian variance: blur vs oversharp clue
    lap = cv2.Laplacian(g8, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap))

    # Gradient magnitude map
    gx = cv2.Sobel(g8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    mag01 = normalize01(mag)

    # Heuristic: extremely low or extremely high lap_var can be suspicious
    # (depends on image; we score "out-of-middle" ranges)
    low = np.clip((60.0 - lap_var) / 60.0, 0.0, 1.0)
    high = np.clip((lap_var - 900.0) / 900.0, 0.0, 1.0)
    score = float(np.clip(0.5 * low + 0.5 * high, 0.0, 1.0))

    return {
        "lap_var": lap_var,
        "score": score,
        "edge_map": mag01,
    }
