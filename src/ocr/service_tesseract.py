"""Standalone Tesseract OCR service entrypoint with JSON-friendly contract."""
from __future__ import annotations

from pathlib import Path
from typing import Dict
import logging

from src.config import OCRConfig
from src.ocr.engine import run_ocr

LOGGER = logging.getLogger(__name__)


def run(payload: Dict[str, object]) -> Dict[str, object]:
    image_path_raw = payload.get("image_path")
    if not image_path_raw:
        raise ValueError("payload must include 'image_path'")
    params = payload.get("params") or {}
    output_dir = Path(payload.get("output_dir") or Path(image_path_raw).parent / "ocr_single")

    engine = str(params.get("engine", "tesseract"))
    languages = str(params.get("languages", "eng"))
    psm = int(params.get("page_segmentation_mode", 6))
    oem = int(params.get("oem", 3))
    enable_stub_fallback = bool(params.get("enable_stub_fallback", True))
    paddle_lang = str(params.get("paddle_lang", "en"))

    cfg = OCRConfig(
        engine=engine,
        languages=languages,
        page_segmentation_mode=psm,
        oem=oem,
        enable_stub_fallback=enable_stub_fallback,
        paddle_lang=paddle_lang,
    )
    image_path = Path(str(image_path_raw))
    result = run_ocr(image_path, output_dir, cfg)

    return {
        "engine": result.engine,
        "engine_attempts": result.engine_attempts,
        "elapsed_seconds": result.elapsed_seconds,
        "json_path": str(result.json_path),
        "word_count": len(result.words),
        "words": [
            {
                "text": w.text,
                "bbox": w.bbox,
                "confidence": w.confidence,
                "page_num": w.page_num,
                "block_num": w.block_num,
                "line_num": w.line_num,
                "word_num": w.word_num,
            }
            for w in result.words
        ],
    }
