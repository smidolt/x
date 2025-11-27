"""VLM-aware block detector for invoices.

Currently supports:
- heuristic mode: derive header/table/totals regions from OCR bboxes.
- qwen2_vl mode (placeholder): uses heuristic fallback, ready to plug real encoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import json
import logging

from src.ocr.engine import OCRWord
from src.config import VLMConfig

LOGGER = logging.getLogger(__name__)

BlockType = Literal["header", "table", "totals"]


@dataclass(slots=True)
class BlockBox:
    kind: BlockType
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float


@dataclass(slots=True)
class BlockDetectionResult:
    blocks: List[BlockBox]
    backend: str
    elapsed_seconds: float
    raw: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "elapsed_seconds": self.elapsed_seconds,
            "blocks": [
                {"kind": b.kind, "bbox": b.bbox, "confidence": b.confidence}
                for b in self.blocks
            ],
            "raw": self.raw,
        }


class BlockDetector:
    def __init__(self, cfg: VLMConfig) -> None:
        self.cfg = cfg

    def run(self, words: List[OCRWord], output_path: Path) -> Optional[BlockDetectionResult]:
        if not self.cfg.enabled:
            LOGGER.info("VLM block detection disabled; skipping")
            return None

        backend = (self.cfg.backend or "heuristic").lower()
        if backend == "heuristic" or not words:
            result = self._heuristic_blocks(words)
        else:
            # Placeholder for future qwen2_vl backend; fall back for now.
            LOGGER.info("VLM backend %s not implemented; using heuristic fallback", backend)
            result = self._heuristic_blocks(words)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result

    def _heuristic_blocks(self, words: List[OCRWord]) -> BlockDetectionResult:
        if not words:
            empty = BlockDetectionResult(blocks=[], backend="heuristic", elapsed_seconds=0.0, raw={})
            return empty

        xs1 = [w.bbox[0] for w in words]
        ys1 = [w.bbox[1] for w in words]
        xs2 = [w.bbox[2] for w in words]
        ys2 = [w.bbox[3] for w in words]
        min_x, max_x = min(xs1), max(xs2)
        min_y, max_y = min(ys1), max(ys2)
        height = max(1, max_y - min_y)

        header_top = min_y
        header_bottom = min_y + int(0.22 * height)
        totals_top = max_y - int(0.18 * height)
        totals_bottom = max_y
        table_top = header_bottom
        table_bottom = totals_top

        blocks = [
            BlockBox("header", [min_x, header_top, max_x, header_bottom], confidence=0.35),
            BlockBox("table", [min_x, table_top, max_x, table_bottom], confidence=0.25),
            BlockBox("totals", [min_x, totals_top, max_x, totals_bottom], confidence=0.35),
        ]

        raw = {
            "source": "heuristic_from_ocr",
            "word_count": len(words),
            "bbox_extent": [min_x, min_y, max_x, max_y],
        }
        return BlockDetectionResult(blocks=blocks, backend="heuristic", elapsed_seconds=0.0, raw=raw)
