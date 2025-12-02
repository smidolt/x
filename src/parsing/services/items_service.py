"""Standalone items parser service."""
from __future__ import annotations

from pathlib import Path
from typing import Dict
import json

from src.parsing.items import ItemsParser


def run(payload: Dict[str, object]) -> Dict[str, object]:
    ocr_json_path_raw = payload.get("ocr_json_path")
    if not ocr_json_path_raw:
        raise ValueError("payload must include 'ocr_json_path'")
    layout_json_path_raw = payload.get("layout_json_path")
    currency_hint = payload.get("currency_hint")

    items_parser = ItemsParser()
    layout_data = {}
    if layout_json_path_raw:
        try:
            layout_data = json.loads(Path(str(layout_json_path_raw)).read_text(encoding="utf-8"))
        except Exception:
            layout_data = {}

    items = items_parser.run(Path(str(ocr_json_path_raw)), layout_data, currency_hint=currency_hint)
    return items.__dict__
