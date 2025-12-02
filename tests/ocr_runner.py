"""Quick OCR smoke test using the Tesseract service wrapper."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.ocr.service_tesseract import run as run_tesseract


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR (Tesseract) on one image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to preprocessed image (PNG/JPG).")
    parser.add_argument("--output", type=Path, default=Path("output/ocr_test"), help="Output directory for OCR JSON.")
    parser.add_argument("--lang", type=str, default="eng", help="Tesseract languages.")
    parser.add_argument("--psm", type=int, default=6, help="Page segmentation mode.")
    parser.add_argument("--oem", type=int, default=3, help="OEM mode.")
    args = parser.parse_args()

    res = run_tesseract(
        {
            "image_path": str(args.image),
            "params": {
                "languages": args.lang,
                "page_segmentation_mode": args.psm,
                "oem": args.oem,
            },
            "output_dir": str(args.output),
        }
    )
    print(res["engine"], res["word_count"], res["json_path"])


if __name__ == "__main__":  # pragma: no cover
    main()
