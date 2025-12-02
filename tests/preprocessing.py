"""Utility runner to test all preprocessing steps independently on the same image."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Callable

from src.preprocessing.services import (
    run_grayscale,
    run_crop_to_content,
    run_deskew,
    run_resize,
    run_normalize_to_a4,
    run_denoise,
    run_contrast,
    run_adaptive_binarization,
)


def run_all_steps(image_path: Path, output_dir: Path, target_dpi: int) -> None:
    steps: List[Tuple[str, Callable[[Dict[str, object]], Dict[str, object]], Dict[str, object]]] = [
        ("grayscale", run_grayscale, {}),
        ("crop_to_content", run_crop_to_content, {"threshold": 235, "margin": 16}),
        ("deskew", run_deskew, {"max_angle": 5.0, "min_angle": 3.0}),
        ("resize", run_resize, {"max_width": 2480, "max_height": 3508}),
        ("normalize_to_a4", run_normalize_to_a4, {"enabled": True, "target_dpi": target_dpi}),
        ("denoise", run_denoise, {"strength": "medium"}),
        ("contrast", run_contrast, {}),
        ("adaptive_binarization", run_adaptive_binarization, {"window_size": 35, "offset": 10}),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, fn, params in steps:
        res = fn(
            {
                "image_path": str(image_path),
                "params": params,
                "output_dir": str(output_dir),
                "target_dpi": target_dpi,
            }
        )
        print(name, res)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all preprocessing steps independently on one image.")
    parser.add_argument("--image", type=Path, default=Path("input/x.jpg"), help="Path to input image")
    parser.add_argument("--output", type=Path, default=Path("output/check_steps_single"), help="Output directory")
    parser.add_argument("--target-dpi", type=int, default=300, help="Target DPI for saved PNGs")
    args = parser.parse_args()

    run_all_steps(args.image, args.output, args.target_dpi)


if __name__ == "__main__":  # pragma: no cover
    main()
