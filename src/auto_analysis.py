from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(x: str, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def pct(values: list[float], p: float, fallback: float) -> float:
    if not values:
        return fallback
    return float(np.percentile(np.array(values, dtype=np.float32), p))


def clamp_thresholds(likely_real_max: float, likely_ai_min: float, margin: float = 0.05) -> tuple[float, float]:
    """
    Ensures thresholds are not inverted and leaves a reasonable "Uncertain" gap.
    If inverted or too close, we center around the midpoint and force a margin.
    """
    # If inverted or too close, force a valid gap
    if likely_real_max >= likely_ai_min - 1e-9:
        mid = 0.5 * (likely_real_max + likely_ai_min)
        likely_real_max = mid - margin
        likely_ai_min = mid + margin

    # Hard clamp to [0,1]
    likely_real_max = float(np.clip(likely_real_max, 0.0, 1.0))
    likely_ai_min = float(np.clip(likely_ai_min, 0.0, 1.0))

    # If still invalid due to edge clipping, use conservative defaults
    if likely_real_max >= likely_ai_min - 1e-9:
        likely_real_max, likely_ai_min = 0.30, 0.70

    return likely_real_max, likely_ai_min


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_root = repo_root / "out"
    csv_path = out_root / "batch_report.csv"
    if not csv_path.exists():
        raise FileNotFoundError("out/batch_report.csv not found. Run: python -m src.batch_run")

    rows = load_csv(csv_path)

    # collect likelihoods per split
    real = [to_float(r["ai_likelihood"]) for r in rows if r["split"] == "real" and r["verdict"] != "ERROR"]
    ai = [to_float(r["ai_likelihood"]) for r in rows if r["split"] == "ai" and r["verdict"] != "ERROR"]
    border = [to_float(r["ai_likelihood"]) for r in rows if r["split"] == "borderline" and r["verdict"] != "ERROR"]

    # baseline calibration (no ML)
    # - real threshold: 90th percentile (most real should be below this)
    # - ai threshold: 10th percentile (most ai should be above this)
    likely_real_max = pct(real, 90, 0.30)
    likely_ai_min = pct(ai, 10, 0.70)

    # Safety clamp (prevents inverted thresholds)
    likely_real_max, likely_ai_min = clamp_thresholds(likely_real_max, likely_ai_min, margin=0.05)

    calibration = {
        "thresholds": {
            "likely_real_max": round(likely_real_max, 3),
            "likely_ai_min": round(likely_ai_min, 3),
            "uncertain_range": [round(likely_real_max, 3), round(likely_ai_min, 3)],
        },
        "stats": {
            "count_real": len(real),
            "count_ai": len(ai),
            "count_borderline": len(border),
            "real_mean": round(float(np.mean(real)), 3) if real else None,
            "ai_mean": round(float(np.mean(ai)), 3) if ai else None,
            "borderline_mean": round(float(np.mean(border)), 3) if border else None,
            "real_p10": round(pct(real, 10, 0.0), 3) if real else None,
            "real_p90": round(pct(real, 90, 0.0), 3) if real else None,
            "ai_p10": round(pct(ai, 10, 0.0), 3) if ai else None,
            "ai_p90": round(pct(ai, 90, 0.0), 3) if ai else None,
        },
    }

    def best(rows_subset: list[dict], key, n=3, reverse=False):
        return sorted(rows_subset, key=key, reverse=reverse)[:n]

    real_rows = [r for r in rows if r["split"] == "real" and r["verdict"] != "ERROR"]
    ai_rows = [r for r in rows if r["split"] == "ai" and r["verdict"] != "ERROR"]
    all_ok = [r for r in rows if r["verdict"] != "ERROR"]

    # best examples:
    # - real: lowest ai_likelihood
    # - ai: highest ai_likelihood
    # - uncertain: closest to 0.5
    top_real = best(real_rows, key=lambda r: to_float(r["ai_likelihood"]), n=3, reverse=False)
    top_ai = best(ai_rows, key=lambda r: to_float(r["ai_likelihood"]), n=3, reverse=True)
    top_uncertain = best(all_ok, key=lambda r: abs(to_float(r["ai_likelihood"]) - 0.5), n=3, reverse=False)

    top_examples = {"real": top_real, "ai": top_ai, "uncertain": top_uncertain}

    # save outputs
    (out_root / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    (out_root / "top_examples.json").write_text(json.dumps(top_examples, indent=2), encoding="utf-8")

    # print summary
    print("\n🔍 TruthLens Auto-Analysis Summary\n")
    print("Calibration thresholds (safe):")
    for k, v in calibration["thresholds"].items():
        print(f"  - {k}: {v}")

    print("\nDataset stats:")
    for k, v in calibration["stats"].items():
        print(f"  - {k}: {v}")

    print("\nTop AI examples:")
    for r in top_ai:
        print(f"  {r['image']} | ai={to_float(r['ai_likelihood']):.2f} | overlay={r.get('overlay','')}")

    print("\nTop Real examples:")
    for r in top_real:
        print(f"  {r['image']} | ai={to_float(r['ai_likelihood']):.2f} | overlay={r.get('overlay','')}")

    print("\nTop Uncertain examples:")
    for r in top_uncertain:
        print(f"  {r['image']} | ai={to_float(r['ai_likelihood']):.2f} | overlay={r.get('overlay','')}")

    print("\n✅ Saved: out/calibration.json + out/top_examples.json\n")


if __name__ == "__main__":
    main()
