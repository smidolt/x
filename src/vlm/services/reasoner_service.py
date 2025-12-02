"""Service entrypoint for Qwen2-VL reasoner."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.vlm.reasoner import VLMReasoner
from src.config import VLMConfig


def run(payload: Dict[str, object]) -> Dict[str, object]:
    image_path_raw = payload.get("image_path")
    if not image_path_raw:
        raise ValueError("payload must include 'image_path'")
    params = payload.get("params") or {}

    cfg = VLMConfig(
        enabled=True,
        backend="qwen2_vl",
        model_name=str(params.get("model_name", "Qwen/Qwen2-VL-2B-Instruct")),
        device=str(params.get("device", "auto")),
        max_new_tokens=int(params.get("max_new_tokens", 256)),
        temperature=float(params.get("temperature", 0.1)),
        prompt=str(params.get("prompt") or ""),
    )
    reasoner = VLMReasoner(cfg)
    res = reasoner.run(Path(str(image_path_raw)))
    return {
        "raw_response": res.raw_response,
        "parsed": res.parsed,
        "error": res.error,
        "elapsed_seconds": res.elapsed_seconds,
    }
