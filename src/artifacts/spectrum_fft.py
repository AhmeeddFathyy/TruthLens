from __future__ import annotations
import numpy as np
import cv2


def _radial_profile(mag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    r_max = min(cy, cx)
    r = np.clip(r, 0, r_max)

    # radial mean
    tbin = np.bincount(r.ravel(), mag.ravel())
    nr = np.bincount(r.ravel())
    radial_mean = tbin / np.maximum(nr, 1)
    radii = np.arange(len(radial_mean))
    return radii[1:], radial_mean[1:]  # skip r=0


def spectrum_features(gray01: np.ndarray) -> dict:
    # windowing reduces border artifacts
    h, w = gray01.shape
    win_y = np.hanning(h).reshape(-1, 1)
    win_x = np.hanning(w).reshape(1, -1)
    win = (win_y * win_x).astype(np.float32)

    x = (gray01 * win).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(x))
    mag = np.log1p(np.abs(f)).astype(np.float32)

    radii, rp = _radial_profile(mag)

    # Fit log(r) vs log(profile) => natural images ~ 1/f^alpha
    r = radii.astype(np.float32)
    y = rp.astype(np.float32)

    # avoid zeros
    r = np.clip(r, 1.0, None)
    y = np.clip(y, 1e-6, None)

    lx = np.log(r)
    ly = np.log(y)

    # linear fit
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    slope = float(coef[0])
    intercept = float(coef[1])

    pred = slope * lx + intercept
    resid = ly - pred
    resid_std = float(np.std(resid))

    # Heuristic: AI often has "too smooth" or "odd" spectral roll-off
    # Larger residual std => more "non-natural" spectrum
    score = float(np.clip((resid_std - 0.12) / (0.30 - 0.12), 0.0, 1.0))

    return {
        "slope": slope,
        "resid_std": resid_std,
        "score": score,
    }
