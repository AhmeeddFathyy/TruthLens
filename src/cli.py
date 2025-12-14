from __future__ import annotations
import argparse
import json
import os
import cv2
import numpy as np

from .utils import read_image_rgb, ensure_dir, to_float01
from .pipeline import analyze_image
from .explain.heatmap import make_heatmap_overlay


def save_rgb01(path: str, rgb01: np.ndarray) -> None:
    rgb8 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def main() -> None:
    ap = argparse.ArgumentParser(description="TruthLens CLI - Explainable AI image forensics (MVP)")
    ap.add_argument("--image", required=True, help="Path to image")
    ap.add_argument("--out", default="out", help="Output folder")
    args = ap.parse_args()

    ensure_dir(args.out)

    rgb = read_image_rgb(args.image)
    res = analyze_image(rgb)

    rgb01 = to_float01(rgb)
    overlay01 = make_heatmap_overlay(rgb01, res.heatmap01, alpha=0.45)

    base = os.path.splitext(os.path.basename(args.image))[0]
    out_overlay = os.path.join(args.out, f"{base}_truthlens_heatmap.png")
    out_json = os.path.join(args.out, f"{base}_truthlens_report.json")

    save_rgb01(out_overlay, overlay01)

    report = {
        "verdict": res.verdict,
        "confidence": round(res.confidence, 4),
        "ai_likelihood": round(res.ai_likelihood, 4),
        "evidence": res.evidence,
        "scores": res.scores,
        "outputs": {"heatmap_overlay": out_overlay},
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
