from __future__ import annotations
import numpy as np
import cv2
from ..utils import normalize01


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


def patch_repetition_features(gray01: np.ndarray, patch: int = 24, stride: int = 12) -> dict:
    h, w = gray01.shape

    # Downscale for speed (keeps textures)
    scale = 512.0 / max(h, w) if max(h, w) > 512 else 1.0
    if scale < 1.0:
        small = cv2.resize(gray01, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = gray01.copy()

    hs, ws = small.shape
    patches = []
    coords = []

    for y in range(0, hs - patch + 1, stride):
        for x in range(0, ws - patch + 1, stride):
            p = small[y:y + patch, x:x + patch].astype(np.float32)
            p = p - float(np.mean(p))
            patches.append(p.ravel())
            coords.append((y, x))

    if len(patches) < 10:
        return {"max_sim": 0.0, "score": 0.0, "rep_map": np.zeros_like(gray01, dtype=np.float32)}

    P = np.stack(patches, axis=0)  # [N, D]
    N = P.shape[0]

    # Sample comparisons for speed
    rng = np.random.default_rng(0)
    sample_pairs = min(3500, N * 6)
    max_sim = -1.0
    hot = np.zeros((hs, ws), dtype=np.float32)

    for _ in range(sample_pairs):
        i = int(rng.integers(0, N))
        j = int(rng.integers(0, N))
        if i == j:
            continue
        sim = _cosine_sim(P[i], P[j])
        if sim > max_sim:
            max_sim = sim

        # highlight unusually similar pairs
        if sim > 0.92:
            yi, xi = coords[i]
            yj, xj = coords[j]
            hot[yi:yi + patch, xi:xi + patch] += 1.0
            hot[yj:yj + patch, xj:xj + patch] += 1.0

    # Normalize repetition map and upsample back
    hot = normalize01(hot)
    if scale < 1.0:
        rep_map = cv2.resize(hot, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        rep_map = hot

    # Score: high max similarity OR large hot regions
    max_sim = float(np.clip(max_sim, -1.0, 1.0))
    max_sim_score = np.clip((max_sim - 0.85) / (0.97 - 0.85), 0.0, 1.0)
    hot_score = float(np.clip(np.mean(rep_map) / 0.12, 0.0, 1.0))
    score = float(np.clip(0.55 * max_sim_score + 0.45 * hot_score, 0.0, 1.0))

    return {
        "max_sim": max_sim,
        "score": score,
        "rep_map": rep_map.astype(np.float32),
    }
