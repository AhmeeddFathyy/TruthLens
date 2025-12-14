# TruthLens 🔍 — Explainable AI Image Forensics (MVP)

TruthLens does not only say "AI / Not AI".
It shows **why**, using visual forensic clues + heatmap.

## Features (MVP)
- Works on **any image** (nature, indoor, objects, etc.)
- Explainable evidence:
  - Frequency spectrum anomalies (FFT)
  - Noise residual structure
  - Patch repetition / self-similarity
  - Edge statistics
- Outputs:
  - Verdict + confidence
  - Evidence list
  - Heatmap overlay

## Run (CLI)
```bash
pip install -r requirements.txt
python -m src.cli --image path/to/img.jpg --out out
