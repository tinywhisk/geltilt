"""geltilt - automatic rotation correction for gel electrophoresis images.

The core idea: GelGenie band segmentation produces band centroids; centroids are
clustered into lanes, each lane is fitted with a straight line, and the per-lane
angles are pooled by inverse variance into one robust image tilt. Rotating by the
negative of that angle makes the lanes vertical. The same lane model is the only
rotation estimator used by the CLI and the desktop app.
"""

from .core import RotationEstimate, auto_correct, correct_rotation, estimate_rotation
from .imageio import load_array, load_gray, save_array

__all__ = [
    "RotationEstimate",
    "estimate_rotation",
    "correct_rotation",
    "auto_correct",
    "load_array",
    "load_gray",
    "save_array",
]

__version__ = "1.2.0"