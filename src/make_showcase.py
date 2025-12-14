from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import cv2


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def put_text(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    # background bar
    h, w = out.shape[:2]
    bar_h = max(28, h // 12)
    cv2.rectangle(out, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.putText(out, text, (10, int(bar_h * 0.70)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def resize_keep(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = r
    return canvas


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_root = repo_root / "out"
    top_path = out_root / "top_examples.json"
    calib_path = out_root / "calibration.json"

    if not top_path.exists() or not calib_path.exists():
        raise FileNotFoundError("Run: python -m src.auto_analysis (needs top_examples.json + calibration.json)")

    top = json.loads(top_path.read_text(encoding="utf-8"))
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    th = calib["thresholds"]

    # pick 1 best from each bucket
    real = top["real"][0]
    ai = top["ai"][0]
    unc = top["uncertain"][0]

    # use overlays if exist (better for showcase), else fallback to original image
    def pick_image(row: dict) -> Path:
        p = Path(row.get("overlay", "")).as_posix()
        if p:
            overlay_path = repo_root / p
            if overlay_path.exists():
                return overlay_path
        return repo_root / row["image"]

    imgs = [
        ("REAL (lowest AI score)", real),
        ("AI (highest AI score)", ai),
        ("UNCERTAIN (closest to 0.5)", unc),
    ]

    tiles = []
    for title, r in imgs:
        img = read_rgb(pick_image(r))
        label = f"{title} | ai={float(r['ai_likelihood']):.2f}"
        tiles.append(put_text(img, label))

    # layout 1x3
    tile_w, tile_h = 520, 360
    tiles = [resize_keep(t, tile_w, tile_h) for t in tiles]
    grid = np.concatenate(tiles, axis=1)

    out_img = out_root / "showcase_grid.png"
    cv2.imwrite(str(out_img), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    md = f"""# TruthLens Showcase (Auto-generated)

## Calibration thresholds (from your dataset)
- likely_real_max: **{th['likely_real_max']}**
- likely_ai_min: **{th['likely_ai_min']}**
- uncertain_range: **{th['uncertain_range'][0]} → {th['uncertain_range'][1]}**

## Best examples (auto-picked)
### Real (lowest AI likelihood)
- `{real['image']}` (ai={float(real['ai_likelihood']):.2f})
- overlay: `{real['overlay']}`

### AI (highest AI likelihood)
- `{ai['image']}` (ai={float(ai['ai_likelihood']):.2f})
- overlay: `{ai['overlay']}`

### Uncertain (closest to 0.5)
- `{unc['image']}` (ai={float(unc['ai_likelihood']):.2f})
- overlay: `{unc['overlay']}`

## Showcase grid
![TruthLens Showcase](showcase_grid.png)
"""
    (out_root / "README_SHOWCASE.md").write_text(md, encoding="utf-8")

    print(f"✅ Wrote: {out_img}")
    print(f"✅ Wrote: {out_root / 'README_SHOWCASE.md'}")


if __name__ == "__main__":
    main()
