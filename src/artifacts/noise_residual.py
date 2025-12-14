from __future__ import annotations
import numpy as np
import cv2
from ..utils import normalize01


def noise_residual_features(rgb01: np.ndarray) -> dict:
    # Work in uint8 for denoiser stability
    rgb8 = (np.clip(rgb01, 0, 1) * 255.0).astype(np.uint8)

    # Denoise (fast + decent)
    den = cv2.fastNlMeansDenoisingColored(rgb8, None, 7, 7, 7, 21)
    den01 = den.astype(np.float32) / 255.0

    resid = (rgb01 - den01).astype(np.float32)
    resid_mag = np.mean(np.abs(resid), axis=2)

    # Simple stats
    rstd = float(np.std(resid_mag))
    rmean = float(np.mean(resid_mag))

    # Autocorr at 1-pixel shift (camera noise tends to be less structured)
    a = resid_mag[:, :-1]
    b = resid_mag[:, 1:]
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 10 else 0.0
    corr = float(np.clip(corr, -1.0, 1.0))

    # Score heuristic
    # - very low residual => over-smooth (common in some gens)
    # - very structured residual (high corr) => suspicious
    smooth_score = np.clip((0.010 - rmean) / (0.010 - 0.003), 0.0, 1.0)
    corr_score = np.clip((corr - 0.10) / (0.45 - 0.10), 0.0, 1.0)
    score = float(np.clip(0.6 * corr_score + 0.4 * smooth_score, 0.0, 1.0))

    return {
        "resid_mean": rmean,
        "resid_std": rstd,
        "resid_corr_1px": corr,
        "score": score,
        "resid_map": normalize01(resid_mag),
    }
