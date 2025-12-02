"""Ready-to-use wrappers for individual preprocessing steps.

Each module exposes `run(payload: dict)` which accepts:
{
    "image_path": "<path>",
    "params": {...},
    "output_dir": "<optional>",
    "target_dpi": <optional int>
}
and returns a JSON-friendly dict with step result metadata.
"""

from .grayscale import run as run_grayscale
from .crop_to_content import run as run_crop_to_content
from .deskew import run as run_deskew
from .resize import run as run_resize
from .normalize_to_a4 import run as run_normalize_to_a4
from .denoise import run as run_denoise
from .contrast import run as run_contrast
from .adaptive_binarization import run as run_adaptive_binarization

__all__ = [
    "run_grayscale",
    "run_crop_to_content",
    "run_deskew",
    "run_resize",
    "run_normalize_to_a4",
    "run_denoise",
    "run_contrast",
    "run_adaptive_binarization",
]
