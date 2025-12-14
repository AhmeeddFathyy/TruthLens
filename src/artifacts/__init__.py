"""
Artifacts modules
-----------------
Each module extracts a specific visual forensic artifact
used to assess whether an image is AI-generated.

Artifacts included:
- Frequency spectrum analysis (FFT)
- Noise residual analysis
- Patch repetition / self-similarity
- Edge statistics
"""

from .spectrum_fft import spectrum_features
from .noise_residual import noise_residual_features
from .patch_repetition import patch_repetition_features
from .edge_stats import edge_features

__all__ = [
    "spectrum_features",
    "noise_residual_features",
    "patch_repetition_features",
    "edge_features",
]
