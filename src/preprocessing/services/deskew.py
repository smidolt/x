"""Standalone deskew step with direct implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict
import logging
import time

from src.preprocessing import steps

LOGGER = logging.getLogger(__name__)


def run(payload: Dict[str, object]) -> Dict[str, object]:
    image_path_raw = payload.get("image_path")
    if not image_path_raw:
        raise ValueError("payload must include 'image_path'")
    params = payload.get("params") or {}
    output_dir = Path(payload.get("output_dir") or Path(image_path_raw).parent)
    target_dpi = payload.get("target_dpi")
    try:
        target_dpi_int = int(target_dpi) if target_dpi is not None else None
    except Exception:
        target_dpi_int = None

    max_angle = float(params.get("max_angle", 5.0))
    min_angle = float(params.get("min_angle", 1.0))

    image_path = Path(str(image_path_raw))
    output_dir.mkdir(parents=True, exist_ok=True)

    image = steps.load_image(image_path)
    start = time.perf_counter()
    result = steps.deskew_image(image, max_angle=max_angle, min_angle=min_angle)
    elapsed = time.perf_counter() - start

    output_path = output_dir / f"{image_path.stem}__deskew.png"
    save_kwargs = {}
    if target_dpi_int and target_dpi_int > 0:
        save_kwargs["dpi"] = (target_dpi_int, target_dpi_int)
    result.image.save(output_path, format="PNG", **save_kwargs)

    LOGGER.info(
        "deskew applied=%s elapsed=%.2fs output=%s warning=%s",
        result.applied,
        elapsed,
        output_path,
        result.warning,
    )
    return {
        "step": "deskew",
        "applied": bool(result.applied),
        "warning": result.warning,
        "elapsed_seconds": elapsed,
        "output_path": str(output_path),
    }
