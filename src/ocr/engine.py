"""OCR integration using Tesseract (with Paddle fallback stubs)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import csv
import io
import json
import logging
import shutil
import subprocess
import time

from src.config import OCRConfig as AppOCRConfig

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from paddleocr import PaddleOCR  # type: ignore
except Exception:  # pragma: no cover - dependency guard
    PaddleOCR = None


class OCRDependencyError(RuntimeError):
    """Raised when an OCR engine cannot start due to missing deps."""


@dataclass(slots=True)
class OCRWord:
    text: str
    bbox: List[int]
    confidence: float | None
    page_num: int
    block_num: int
    line_num: int
    word_num: int


@dataclass(slots=True)
class OCRResult:
    words: List[OCRWord]
    json_path: Path
    engine: str
    elapsed_seconds: float
    engine_attempts: List[str] = field(default_factory=list)

    def to_payload(self, source_path: Path) -> Dict[str, Any]:
        return {
            "source": str(source_path),
            "engine": self.engine,
            "engine_attempts": self.engine_attempts,
            "elapsed_seconds": self.elapsed_seconds,
            "word_count": len(self.words),
            "words": [
                {
                    "text": word.text,
                    "bbox": word.bbox,
                    "confidence": word.confidence,
                    "page_num": word.page_num,
                    "block_num": word.block_num,
                    "line_num": word.line_num,
                    "word_num": word.word_num,
                }
                for word in self.words
            ],
        }


class BaseOCREngine:
    name: str

    def recognize(self, image_path: Path) -> List[OCRWord]:  # pragma: no cover - interface
        raise NotImplementedError


class TesseractOCREngine(BaseOCREngine):
    name = "tesseract"

    def __init__(self, languages: str, psm: int, oem: int) -> None:
        self.languages = languages
        self.psm = psm
        self.oem = oem

    def recognize(self, image_path: Path) -> List[OCRWord]:
        if shutil.which("tesseract") is None:
            raise OCRDependencyError("Tesseract binary is not available on PATH")

        cmd = [
            "tesseract",
            str(image_path),
            "stdout",
            "-l",
            self.languages,
            "--psm",
            str(self.psm),
            "--oem",
            str(self.oem),
            "tsv",
        ]
        LOGGER.debug("Executing Tesseract command: %s", " ".join(cmd))
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as proc:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"Tesseract failed: {stderr.strip()}")

        reader = csv.DictReader(io.StringIO(stdout), delimiter="\t")
        words: List[OCRWord] = []
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            try:
                left = int(row.get("left", 0) or 0)
                top = int(row.get("top", 0) or 0)
                width = int(row.get("width", 0) or 0)
                height = int(row.get("height", 0) or 0)
            except ValueError:
                continue
            conf_raw = row.get("conf")
            try:
                confidence = float(conf_raw) if conf_raw not in {"", None} else None
            except ValueError:
                confidence = None
            bbox = [left, top, left + width, top + height]
            words.append(
                OCRWord(
                    text=text,
                    bbox=bbox,
                    confidence=confidence,
                    page_num=int(row.get("page_num", 1) or 1),
                    block_num=int(row.get("block_num", 0) or 0),
                    line_num=int(row.get("line_num", 0) or 0),
                    word_num=int(row.get("word_num", 0) or 0),
                )
            )

        LOGGER.debug("Tesseract OCR produced %d tokens", len(words))
        return words


class PaddleOCREngine(BaseOCREngine):  # pragma: no cover - heavy dep
    name = "paddleocr"

    def __init__(self, lang: str) -> None:
        if PaddleOCR is None:
            raise OCRDependencyError("paddleocr package is not installed")
        # PaddleOCR can be slow to initialize; reuse the instance.
        self._client = PaddleOCR(lang=lang, show_log=False, use_angle_cls=True)

    def recognize(self, image_path: Path) -> List[OCRWord]:
        result = self._client.ocr(str(image_path), cls=True)
        words: List[OCRWord] = []
        page_num = 1
        for line_idx, (bbox, (text, confidence)) in enumerate(result, start=1):
            x_coords = [int(point[0]) for point in bbox]
            y_coords = [int(point[1]) for point in bbox]
            bbox_rect = [
                min(x_coords),
                min(y_coords),
                max(x_coords),
                max(y_coords),
            ]
            words.append(
                OCRWord(
                    text=text,
                    bbox=bbox_rect,
                    confidence=float(confidence),
                    page_num=page_num,
                    block_num=line_idx,
                    line_num=line_idx,
                    word_num=line_idx,
                )
            )
        LOGGER.debug("PaddleOCR produced %d tokens", len(words))
        return words


class StubOCREngine(BaseOCREngine):
    name = "stub"

    def __init__(self, reason: str = "not_configured") -> None:
        self.reason = reason

    def recognize(self, image_path: Path) -> List[OCRWord]:
        LOGGER.warning("Using stub OCR engine (%s) for %s", self.reason, image_path)
        return []


def run_ocr(image_path: Path, output_dir: Path, config: AppOCRConfig) -> OCRResult:
    LOGGER.info("Running OCR on %s", image_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{image_path.stem}.ocr.json"
    engines = _build_engine_pipeline(config)

    words: List[OCRWord] = []
    engine_used = "stub"
    elapsed = 0.0
    engine_attempts = [engine.name for engine in engines]

    for engine in engines:
        start = time.perf_counter()
        try:
            words = engine.recognize(image_path)
            elapsed = time.perf_counter() - start
            engine_used = engine.name
            LOGGER.info(
                "OCR via %s completed in %.2fs (tokens=%d)",
                engine.name,
                elapsed,
                len(words),
            )
            break
        except OCRDependencyError as exc:
            LOGGER.warning("Skipping OCR engine %s: %s", engine.name, exc)
            continue
        except Exception as exc:  # pragma: no cover - runtime safety
            LOGGER.exception("OCR engine %s failed: %s", engine.name, exc)
            continue
    else:
        LOGGER.warning("All OCR engines failed; falling back to stub output")
        stub = StubOCREngine("no_engine_succeeded")
        words = stub.recognize(image_path)
        engine_used = stub.name
        engine_attempts.append(stub.name)

    result = OCRResult(
        words=words,
        json_path=json_path,
        engine=engine_used,
        elapsed_seconds=elapsed,
        engine_attempts=engine_attempts,
    )
    payload = result.to_payload(image_path)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result


def _build_engine_pipeline(config: AppOCRConfig) -> List[BaseOCREngine]:
    engine_field = config.engine.strip().lower()
    if engine_field == "auto":
        requested = ["tesseract", "paddle"]
    elif "," in config.engine:
        requested = [item.strip().lower() for item in config.engine.split(",") if item.strip()]
    else:
        requested = [engine_field]

    engines: List[BaseOCREngine] = []
    for name in requested:
        if name == "tesseract":
            engines.append(
                TesseractOCREngine(
                    languages=config.languages,
                    psm=config.page_segmentation_mode,
                    oem=config.oem,
                )
            )
        elif name == "paddle":
            try:
                engines.append(PaddleOCREngine(config.paddle_lang))
            except OCRDependencyError as exc:
                LOGGER.warning("Cannot initialize PaddleOCR: %s", exc)
        elif name:
            LOGGER.warning("Unknown OCR engine '%s' requested; skipping", name)

    if not engines:
        if config.enable_stub_fallback:
            engines.append(StubOCREngine("no_valid_engine"))
        else:
            raise RuntimeError("No OCR engines available and stub fallback disabled")
    return engines
