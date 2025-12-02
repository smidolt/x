"""Lightweight orchestrator CLI that runs modular services in sequence.

Steps per document:
1) Preprocess (grayscale -> crop -> deskew -> resize -> normalize_to_a4 -> denoise -> contrast -> adaptive_binarization)
2) OCR (Tesseract)
3) Layout (stub by default; real model if enabled)
4) Parsing classic (meta + items + rule validation)
5) Emit a combined JSON with paths to artifacts
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
from src.ocr.service_tesseract import run as run_tesseract
from src.layout.services import run_layout
from src.parsing.services import run_classic as run_parsing_classic

LOGGER = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    steps: List[Dict[str, object]]
    final_image: Path


def _iter_documents(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    for path in sorted(p for p in input_path.rglob("*") if p.is_file()):
        yield path


def run_preprocess(image_path: Path, output_dir: Path, target_dpi: int) -> PreprocessResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = []
    current = image_path
    for name, fn, params in [
        ("grayscale", run_grayscale, {}),
        ("crop_to_content", run_crop_to_content, {"threshold": 235, "margin": 16}),
        ("deskew", run_deskew, {"max_angle": 5.0, "min_angle": 3.0}),
        ("resize", run_resize, {"max_width": 2480, "max_height": 3508}),
        ("normalize_to_a4", run_normalize_to_a4, {"enabled": True, "target_dpi": target_dpi}),
        ("denoise", run_denoise, {"strength": "medium"}),
        ("contrast", run_contrast, {}),
        ("adaptive_binarization", run_adaptive_binarization, {"window_size": 35, "offset": 10}),
    ]:
        res = fn(
            {
                "image_path": str(current),
                "params": params,
                "output_dir": str(output_dir),
                "target_dpi": target_dpi,
            }
        )
        steps.append(res)
        current = Path(res["output_path"])
    return PreprocessResult(steps=steps, final_image=current)


def run_document(
    document: Path,
    output_root: Path,
    seller_name: str,
    seller_tax_id: str,
    layout_enabled: bool,
    target_dpi: int,
    ocr_lang: str,
    ocr_psm: int,
    ocr_oem: int,
) -> Dict[str, object]:
    stem = document.stem
    preprocess_dir = output_root / "preprocessed"
    ocr_dir = output_root / "ocr"
    layout_dir = output_root / "layout"
    json_dir = output_root / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    doc_payload: Dict[str, object] = {"file_name": document.name}

    # Preprocess
    LOGGER.info("Preprocess: %s", document)
    prep_start = time.time()
    prep_result = run_preprocess(document, preprocess_dir, target_dpi)
    doc_payload["preprocessing"] = {
        "steps": prep_result.steps,
        "final_image": str(prep_result.final_image),
        "elapsed_seconds": time.time() - prep_start,
    }

    # OCR
    LOGGER.info("OCR: %s", prep_result.final_image)
    ocr_res = run_tesseract(
        {
            "image_path": str(prep_result.final_image),
            "params": {
                "languages": ocr_lang,
                "page_segmentation_mode": ocr_psm,
                "oem": ocr_oem,
            },
            "output_dir": str(ocr_dir),
        }
    )
    doc_payload["ocr"] = {
        "engine": ocr_res["engine"],
        "word_count": ocr_res["word_count"],
        "json_path": ocr_res["json_path"],
        "elapsed_seconds": ocr_res["elapsed_seconds"],
    }

    # Layout (optional)
    layout_path: Optional[Path] = None
    layout_data: Optional[Dict[str, object]] = None
    if layout_enabled:
        LOGGER.info("Layout: %s", ocr_res["json_path"])
        layout_path = layout_dir / f"{stem}.layout.json"
        layout_dir.mkdir(parents=True, exist_ok=True)
        layout_data = run_layout(
            {
                "ocr_json_path": ocr_res["json_path"],
                "params": {"enabled": True, "model_name": "microsoft/layoutlmv3-base", "offline": True},
            }
        )
        layout_path.write_text(json.dumps(layout_data, indent=2), encoding="utf-8")
        doc_payload["layout"] = {
            "json_path": str(layout_path),
            "engine": layout_data.get("engine"),
            "model_name": layout_data.get("model_name"),
        }
    else:
        doc_payload["layout"] = {"engine": "disabled"}

    # Parsing classic
    LOGGER.info("Parsing: %s", ocr_res["json_path"])
    parsing_out = run_parsing_classic(
        {
            "ocr_json_path": ocr_res["json_path"],
            "seller_name": seller_name,
            "seller_tax_id": seller_tax_id,
            "layout_json_path": str(layout_path) if layout_path else None,
        }
    )
    doc_payload["parsing"] = parsing_out

    # Persist combined JSON
    output_json_path = json_dir / f"{stem}.json"
    output_json_path.write_text(json.dumps(doc_payload, indent=2), encoding="utf-8")
    doc_payload["output_json"] = str(output_json_path)
    LOGGER.info("Done %s -> %s", document, output_json_path)
    return doc_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight pipeline orchestrator using modular services.")
    parser.add_argument("--input", type=Path, default=Path("input"), help="File or directory with images.")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output root directory.")
    parser.add_argument("--seller-name", type=str, required=True, help="Expected seller name.")
    parser.add_argument("--seller-tax-id", type=str, required=True, help="Expected seller tax ID.")
    parser.add_argument("--layout-enabled", action="store_true", default=True, help="Enable LayoutLM (stub otherwise).")
    parser.add_argument("--target-dpi", type=int, default=300, help="Target DPI for preprocessing output.")
    parser.add_argument("--ocr-lang", type=str, default="eng", help="Tesseract languages.")
    parser.add_argument("--ocr-psm", type=int, default=6, help="Tesseract PSM mode.")
    parser.add_argument("--ocr-oem", type=int, default=3, help="Tesseract OEM mode.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    for doc in _iter_documents(args.input):
        results.append(
            run_document(
                document=doc,
                output_root=output_root,
                seller_name=args.seller_name,
                seller_tax_id=args.seller_tax_id,
                layout_enabled=args.layout_enabled,
                target_dpi=args.target_dpi,
                ocr_lang=args.ocr_lang,
                ocr_psm=args.ocr_psm,
                ocr_oem=args.ocr_oem,
            )
        )

    summary_path = output_root / "summary_orchestrator.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":  # pragma: no cover
    main()
