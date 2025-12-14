from __future__ import annotations

import json
from pathlib import Path


def load_calibration(repo_root: Path) -> dict | None:
    path = repo_root / "out" / "calibration.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_thresholds(calib: dict | None) -> tuple[float, float]:
    """
    returns (likely_real_max, likely_ai_min)
    """
    if not calib:
        return 0.30, 0.70
    th = calib.get("thresholds", {})
    return float(th.get("likely_real_max", 0.30)), float(th.get("likely_ai_min", 0.70))


def verdict_from_likelihood(ai_likelihood: float, likely_real_max: float, likely_ai_min: float) -> tuple[str, float]:
    """
    Returns (verdict, confidence) with honest uncertainty.
    """
    x = float(ai_likelihood)

    if x >= likely_ai_min:
        return "Likely AI-generated", min(1.0, max(0.0, x))
    if x <= likely_real_max:
        return "Likely Real", min(1.0, max(0.0, 1.0 - x))

    # Uncertain zone
    # Confidence is lower near 0.5
    conf = 1.0 - abs(x - 0.5) * 2.0
    return "Uncertain", min(1.0, max(0.0, conf))
