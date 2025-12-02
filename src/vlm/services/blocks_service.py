"""Service entrypoint for heuristic block detection."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json

from src.vlm.blocks import BlockDetector
from src.config import VLMConfig
from src.ocr.engine import OCRWord


def _load_words(path: Path) -> List[OCRWord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    words: List[OCRWord] = []
    for w in data.get("words", []):
        words.append(
            OCRWord(
                text=w.get("text", ""),
                bbox=w.get("bbox", [0, 0, 0, 0]),
                confidence=w.get("confidence"),
                page_num=w.get("page_num", 1),
                block_num=w.get("block_num", 0),
                line_num=w.get("line_num", 0),
                word_num=w.get("word_num", 0),
            )
        )
    return words


def run(payload: Dict[str, object]) -> Dict[str, object]:
    ocr_json_raw = payload.get("ocr_json_path")
    if not ocr_json_raw:
        raise ValueError("payload must include 'ocr_json_path'")
    output_path = Path(payload.get("output_path") or Path(str(ocr_json_raw)).with_suffix(".blocks.json"))
    backend = str(payload.get("backend", "heuristic"))

    words = _load_words(Path(str(ocr_json_raw)))
    cfg = VLMConfig(enabled=True, backend=backend)
    detector = BlockDetector(cfg)
    result = detector.run(words, output_path)
    if result is None:
        return {}
    return result.to_dict()
