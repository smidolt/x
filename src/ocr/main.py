"""Standalone OCR runner (Tesseract-first) with JSON-friendly output."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from src.config import OCRConfig
from src.ocr.engine import run_ocr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR on a single image and emit JSON.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image (PNG/JPG).")
    parser.add_argument("--output-dir", type=Path, default=Path("output/ocr_single"), help="Directory for OCR JSON.")
    parser.add_argument("--languages", type=str, default="eng", help="Tesseract languages, e.g. 'eng+deu'.")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract PSM mode.")
    parser.add_argument("--oem", type=int, default=3, help="Tesseract OEM mode.")
    parser.add_argument("--engine", type=str, default="tesseract", help="Engine string: tesseract|paddle|auto|list")
    args = parser.parse_args()

    cfg = OCRConfig(
        engine=args.engine,
        languages=args.languages,
        page_segmentation_mode=args.psm,
        oem=args.oem,
        enable_stub_fallback=True,
    )
    result = run_ocr(args.image, args.output_dir, cfg)
    payload_path = result.json_path
    print(f"OCR done. Engine={result.engine} words={len(result.words)} json={payload_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
