"""
Explainability layer
--------------------
Utilities for visual and human-readable explanations,
including heatmaps and forensic reports.
"""

from .heatmap import make_heatmap_overlay

__all__ = [
    "make_heatmap_overlay",
]
