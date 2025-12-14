from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .utils import to_float01, rgb_to_gray01, sigmoid, normalize01
from .artifacts.spectrum_fft import spectrum_features
from .artifacts.noise_residual import noise_residual_features
from .artifacts.patch_repetition import patch_repetition_features
from .artifacts.edge_stats import edge_features

# Dynamic calibration helpers (loaded if calibration.json exists)
from .calibration import load_calibration, get_thresholds, verdict_from_likelihood


@dataclass
class TruthLensResult:
    verdict: str
    confidence: float
    ai_likelihood: float
    evidence: list[str]
    scores: dict
    heatmap01: np.ndarray


def analyze_image(rgb: np.ndarray) -> TruthLensResult:
    rgb01 = to_float01(rgb)
    gray01 = rgb_to_gray01(rgb01)

    spec = spectrum_features(gray01)
    noi = noise_residual_features(rgb01)
    rep = patch_repetition_features(gray01)
    edg = edge_features(gray01)

    # Weighted combine (MVP weights)
    w_spec, w_noi, w_rep, w_edg = 0.30, 0.30, 0.25, 0.15
    combined = (
        w_spec * spec["score"] +
        w_noi  * noi["score"] +
        w_rep  * rep["score"] +
        w_edg  * edg["score"]
    )

    # Map to likelihood (smooth)
    ai_likelihood = sigmoid((combined - 0.50) * 6.0)  # center around 0.5
    ai_likelihood = float(np.clip(ai_likelihood, 0.0, 1.0))

    # ---- Dynamic verdict rules (use calibration.json if available)
    repo_root = Path(__file__).resolve().parents[1]  # TruthLens/
    calib = load_calibration(repo_root)
    likely_real_max, likely_ai_min = get_thresholds(calib)

    verdict, confidence = verdict_from_likelihood(
        ai_likelihood=ai_likelihood,
        likely_real_max=likely_real_max,
        likely_ai_min=likely_ai_min,
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))

    # Evidence (explainable)
    evidence: list[str] = []
    if spec["score"] > 0.55:
        evidence.append(
            f"Non-natural frequency spectrum (residual_std={spec['resid_std']:.3f})"
        )
    if noi["score"] > 0.55:
        evidence.append(
            f"Suspicious noise residual (corr@1px={noi['resid_corr_1px']:.2f}, resid_mean={noi['resid_mean']:.4f})"
        )
    if rep["score"] > 0.55:
        evidence.append(
            f"Patch self-similarity / repetition (max_sim={rep['max_sim']:.2f})"
        )
    if edg["score"] > 0.55:
        evidence.append(
            f"Edge statistics out of expected range (lap_var={edg['lap_var']:.1f})"
        )

    if not evidence:
        evidence.append(
            "No strong forensic artifacts detected by current heuristics (MVP)."
        )

    # Heatmap: combine artifact maps
    heat = (
        0.45 * noi["resid_map"] +
        0.40 * rep["rep_map"] +
        0.15 * edg["edge_map"]
    )
    heat = normalize01(heat)

    scores = {
        "ai_likelihood": ai_likelihood,
        "combined_score": float(combined),
        "weights": {
            "spectrum": w_spec,
            "noise": w_noi,
            "repetition": w_rep,
            "edges": w_edg,
        },
        "thresholds_used": {
            "likely_real_max": float(likely_real_max),
            "likely_ai_min": float(likely_ai_min),
            "has_calibration_file": bool(calib),
        },
        "spectrum": spec,
        "noise": {k: noi[k] for k in ["resid_mean", "resid_std", "resid_corr_1px", "score"]},
        "repetition": {k: rep[k] for k in ["max_sim", "score"]},
        "edges": {k: edg[k] for k in ["lap_var", "score"]},
    }

    return TruthLensResult(
        verdict=verdict,
        confidence=confidence,
        ai_likelihood=ai_likelihood,
        evidence=evidence,
        scores=scores,
        heatmap01=heat,
    )
