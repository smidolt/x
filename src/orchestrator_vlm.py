"""VLM-focused orchestrator: preprocess -> OCR -> VLM blocks -> VLM reasoner -> combined JSON."""
from __future__ import annotations

import argparse
import json
import logging
import time
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
from src.vlm.services import run_blocks, run_reasoner

LOGGER = logging.getLogger(__name__)


def _iter_documents(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    for path in sorted(p for p in input_path.rglob("*") if p.is_file()):
        yield path


def run_preprocess(image_path: Path, output_dir: Path, target_dpi: int) -> Dict[str, object]:
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
    return {"steps": steps, "final_image": current}


def run_document(
    document: Path,
    output_root: Path,
    target_dpi: int,
    ocr_lang: str,
    ocr_psm: int,
    ocr_oem: int,
    vlm_backend_blocks: str,
    vlm_model_reasoner: str,
    vlm_device: str,
    vlm_max_tokens: int,
    vlm_temperature: float,
    vlm_prompt: Optional[str],
) -> Dict[str, object]:
    stem = document.stem
    preprocess_dir = output_root / "preprocessed"
    ocr_dir = output_root / "ocr"
    vlm_dir = output_root / "vlm"
    vlm_dir.mkdir(parents=True, exist_ok=True)
    json_dir = output_root / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, object] = {"file_name": document.name}

    # Preprocess
    LOGGER.info("Preprocess: %s", document)
    prep_start = time.time()
    prep = run_preprocess(document, preprocess_dir, target_dpi)
    payload["preprocessing"] = {
        "steps": prep["steps"],
        "final_image": str(prep["final_image"]),
        "elapsed_seconds": time.time() - prep_start,
    }

    # OCR (for blocks)
    LOGGER.info("OCR for blocks: %s", prep["final_image"])
    ocr_res = run_tesseract(
        {
            "image_path": str(prep["final_image"]),
            "params": {
                "languages": ocr_lang,
                "page_segmentation_mode": ocr_psm,
                "oem": ocr_oem,
            },
            "output_dir": str(ocr_dir),
        }
    )
    payload["ocr"] = {
        "engine": ocr_res["engine"],
        "word_count": ocr_res["word_count"],
        "json_path": ocr_res["json_path"],
        "elapsed_seconds": ocr_res["elapsed_seconds"],
    }

    # VLM blocks
    LOGGER.info("VLM blocks: %s", ocr_res["json_path"])
    blocks_path = vlm_dir / f"{stem}.blocks.json"
    blocks = run_blocks({"ocr_json_path": ocr_res["json_path"], "output_path": str(blocks_path), "backend": vlm_backend_blocks})
    payload["vlm_blocks"] = {
        "json_path": str(blocks_path),
        "backend": blocks.get("backend"),
        "blocks": blocks.get("blocks"),
    }

    # VLM reasoner
    LOGGER.info("VLM reasoner: %s", prep["final_image"])
    reasoner_params = {
        "model_name": vlm_model_reasoner,
        "device": vlm_device,
        "max_new_tokens": vlm_max_tokens,
        "temperature": vlm_temperature,
    }
    if vlm_prompt:
        reasoner_params["prompt"] = vlm_prompt
    reasoner_res = run_reasoner({"image_path": str(prep["final_image"]), "params": reasoner_params})
    payload["vlm_reasoner"] = reasoner_res

    # Persist combined JSON
    output_json_path = json_dir / f"{stem}.vlm.json"
    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["output_json"] = str(output_json_path)
    LOGGER.info("Done %s -> %s", document, output_json_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM-first pipeline orchestrator.")
    parser.add_argument("--input", type=Path, default=Path("input"), help="File or directory with images.")
    parser.add_argument("--output", type=Path, default=Path("output_vlm"), help="Output root directory.")
    parser.add_argument("--target-dpi", type=int, default=300, help="Target DPI for preprocessing output.")
    parser.add_argument("--ocr-lang", type=str, default="eng", help="Tesseract languages.")
    parser.add_argument("--ocr-psm", type=int, default=6, help="Tesseract PSM mode.")
    parser.add_argument("--ocr-oem", type=int, default=3, help="Tesseract OEM mode.")
    parser.add_argument("--vlm-backend-blocks", type=str, default="heuristic", help="VLM blocks backend.")
    parser.add_argument("--vlm-model-reasoner", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="VLM reasoner model.")
    parser.add_argument("--vlm-device", type=str, default="auto", help="Device: auto|cuda|mps|cpu.")
    parser.add_argument("--vlm-max-tokens", type=int, default=256, help="Max new tokens for VLM reasoner.")
    parser.add_argument("--vlm-temperature", type=float, default=0.1, help="Temperature for VLM reasoner.")
    parser.add_argument("--vlm-prompt", type=str, default=None, help="Custom prompt for VLM reasoner.")
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
                target_dpi=args.target_dpi,
                ocr_lang=args.ocr_lang,
                ocr_psm=args.ocr_psm,
                ocr_oem=args.ocr_oem,
                vlm_backend_blocks=args.vlm_backend_blocks,
                vlm_model_reasoner=args.vlm_model_reasoner,
                vlm_device=args.vlm_device,
                vlm_max_tokens=args.vlm_max_tokens,
                vlm_temperature=args.vlm_temperature,
                vlm_prompt=args.vlm_prompt,
            )
        )

    summary_path = output_root / "summary_vlm_orchestrator.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":  # pragma: no cover
    main()
