"""geltilt - automatic rotation correction for gel electrophoresis images.

The core idea: lanes and their band smears form vertical structures. When a gel
image is tilted, the vertical projection (column sums) of the band signal is
smeared; when the lanes are vertical that projection is maximally sharp. We scan
candidate rotation angles and pick the one that maximises the sharpness of this
projection, which straightens the lanes.
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

__version__ = "0.1.0"