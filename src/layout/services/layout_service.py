"""Service wrapper for LayoutAnalyzer."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.layout import LayoutAnalyzer


def run(payload: Dict[str, object]) -> Dict[str, object]:
    ocr_json_raw = payload.get("ocr_json_path")
    if not ocr_json_raw:
        raise ValueError("payload must include 'ocr_json_path'")
    params = payload.get("params") or {}

    analyzer = LayoutAnalyzer(
        model_name=str(params.get("model_name", "microsoft/layoutlmv3-base")),
        enabled=bool(params.get("enabled", False)),
        offline=bool(params.get("offline", False)),
    )
    annotations = analyzer.run(Path(str(ocr_json_raw)))
    return {
        "tokens": annotations.tokens,
        "embeddings": annotations.embeddings,
        "model_name": annotations.model_name,
        "engine": annotations.engine,
    }
