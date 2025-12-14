from __future__ import annotations
import numpy as np
import cv2
from ..utils import normalize01


def make_heatmap_overlay(rgb01: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat01 = normalize01(heat01)
    heat8 = (heat01 * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat8, cv2.COLORMAP_JET)  # BGR
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    base = rgb01.astype(np.float32)
    out = (1 - alpha) * base + alpha * heat_color
    return np.clip(out, 0.0, 1.0)
