from __future__ import annotations

import os
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .utils import read_image_rgb, ensure_dir, to_float01
from .pipeline import analyze_image
from .explain.heatmap import make_heatmap_overlay


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def save_rgb01(path: str, rgb01: np.ndarray) -> None:
    rgb8 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def infer_labels_from_path(p: Path) -> tuple[str, str]:
    """
    Expected structure:
      demo/sample_images/<split>/<category>/<file>
    split = real | ai | borderline
    category = Architecture | Indoor | Nature | Objects (or any folder name)
    """
    parts = [x.lower() for x in p.parts]
    split = "unknown"
    category = "unknown"

    # find split
    for s in ("real", "ai", "borderline"):
        if s in parts:
            split = s
            break

    # category = folder after split
    if split in parts:
        idx = parts.index(split)
        if idx + 1 < len(parts):
            category = parts[idx + 1]

    return split, category


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]  # TruthLens/
    img_root = repo_root / "demo" / "sample_images"
    out_root = repo_root / "out"

    overlays_dir = out_root / "overlays"
    json_dir = out_root / "json"

    ensure_dir(str(out_root))
    ensure_dir(str(overlays_dir))
    ensure_dir(str(json_dir))

    if not img_root.exists():
        raise FileNotFoundError(f"Image root not found: {img_root}")

    images = []
    for p in img_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            images.append(p)

    images = sorted(images)
    if not images:
        print(f"No images found under: {img_root}")
        return

    csv_path = out_root / "batch_report.csv"
    rows = []

    for p in images:
        rel = p.relative_to(repo_root).as_posix()
        split, category = infer_labels_from_path(p)

        try:
            rgb = read_image_rgb(str(p))
            res = analyze_image(rgb)

            rgb01 = to_float01(rgb)
            overlay01 = make_heatmap_overlay(rgb01, res.heatmap01, alpha=0.45)

            base = p.stem
            overlay_path = overlays_dir / f"{base}_heatmap.png"
            report_path = json_dir / f"{base}_report.json"

            save_rgb01(str(overlay_path), overlay01)

            report = {
                "image": rel,
                "split": split,
                "category": category,
                "verdict": res.verdict,
                "confidence": round(res.confidence, 4),
                "ai_likelihood": round(res.ai_likelihood, 4),
                "evidence": res.evidence,
                "scores": res.scores,
                "outputs": {"heatmap_overlay": overlay_path.as_posix()},
            }

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            rows.append({
                "image": rel,
                "split": split,
                "category": category,
                "verdict": res.verdict,
                "confidence": res.confidence,
                "ai_likelihood": res.ai_likelihood,
                "evidence": " | ".join(res.evidence),
                "overlay": overlay_path.as_posix(),
                "json": report_path.as_posix(),
            })

            print(f"[OK] {rel} -> {res.verdict} (ai={res.ai_likelihood:.2f})")

        except Exception as e:
            print(f"[ERR] {rel}: {e}")
            rows.append({
                "image": rel,
                "split": split,
                "category": category,
                "verdict": "ERROR",
                "confidence": "",
                "ai_likelihood": "",
                "evidence": str(e),
                "overlay": "",
                "json": "",
            })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Done. CSV saved to: {csv_path}")
    print(f"✅ Overlays: {overlays_dir}")
    print(f"✅ JSON reports: {json_dir}")


if __name__ == "__main__":
    main()
