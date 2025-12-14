import streamlit as st
import numpy as np
from PIL import Image

from src.pipeline import analyze_image
from src.utils import to_float01
from src.explain.heatmap import make_heatmap_overlay

st.set_page_config(page_title="TruthLens - AI Image Lie Detector", layout="wide")

st.title("TruthLens 🔍 — Explainable AI Image Forensics (MVP)")
st.caption("Uploads any image → verdict + evidence + heatmap (frequency/noise/repetition/edges)")

up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

if up:
    img = Image.open(up).convert("RGB")
    rgb = np.array(img)

    res = analyze_image(rgb)
    overlay = make_heatmap_overlay(to_float01(rgb), res.heatmap01, alpha=0.45)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader("Forensic Heatmap Overlay")
        st.image((overlay * 255).astype(np.uint8), use_container_width=True)

    st.divider()
    a, b, c = st.columns(3)
    a.metric("Verdict", res.verdict)
    b.metric("AI Likelihood", f"{res.ai_likelihood:.2f}")
    c.metric("Confidence", f"{res.confidence:.2f}")

    st.subheader("Evidence (Explainable)")
    for e in res.evidence:
        st.write("•", e)

    with st.expander("Raw Scores (debug)"):
        st.json(res.scores)
else:
    st.info("Upload an image to start.")
