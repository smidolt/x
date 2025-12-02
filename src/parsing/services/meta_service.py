"""Standalone meta parser service."""
from __future__ import annotations

from pathlib import Path
from typing import Dict
import json

from src.parsing.meta import MetaParser


def run(payload: Dict[str, object]) -> Dict[str, object]:
    ocr_json_path_raw = payload.get("ocr_json_path")
    if not ocr_json_path_raw:
        raise ValueError("payload must include 'ocr_json_path'")
    seller_name = str(payload.get("seller_name", ""))
    seller_tax_id = str(payload.get("seller_tax_id", ""))
    layout_json_path_raw = payload.get("layout_json_path")

    meta_parser = MetaParser(seller_name, seller_tax_id)
    layout_data = {}
    if layout_json_path_raw:
        try:
            layout_data = json.loads(Path(str(layout_json_path_raw)).read_text(encoding="utf-8"))
        except Exception:
            layout_data = {}

    meta = meta_parser.run(Path(str(ocr_json_path_raw)), layout_data)
    return meta.raw
