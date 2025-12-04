"""Service wrapper for VLM validation."""
from __future__ import annotations

from typing import Dict

from src.validation.vlm.main import validate_vlm


def run(payload: Dict[str, object]) -> Dict[str, object]:
    data = payload.get("data") or {}
    tolerance = float(payload.get("tolerance", 0.5))
    meta = data.get("meta") if isinstance(data, dict) else {}
    items = data.get("items") if isinstance(data, dict) else {}
    return validate_vlm(meta or {}, items or {}, tol=tolerance)
