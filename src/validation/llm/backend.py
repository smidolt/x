"""LLM validation backend stubs and local transformer integration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

import json
import logging

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from transformers import pipeline
except Exception:  # pragma: no cover
    pipeline = None


class LLMBackend(Protocol):
    def validate(self, meta: dict, items: dict) -> dict:
        ...


@dataclass(slots=True)
class StubLLMBackend:
    """Simple backend that echoes a placeholder response."""

    name: str = "stub"

    def validate(self, meta: dict, items: dict) -> dict:
        LOGGER.info("Running stub LLM backend")
        return {
            "backend": self.name,
            "notes": "LLM validation disabled or not configured.",
            "meta_fields_checked": len(meta),
            "items_checked": len(items.get("rows", [])) if isinstance(items, dict) else 0,
            "issues": [],
        }


@dataclass(slots=True)
class LocalTransformerBackend:
    model_name: str

    def __post_init__(self) -> None:
        if pipeline is None:
            raise RuntimeError("transformers is not installed")
        try:
            self._pipeline = pipeline("text-classification", model=self.model_name)
        except Exception as exc:  # pragma: no cover - heavy dep
            raise RuntimeError(f"Unable to load transformer model {self.model_name}: {exc}") from exc

    def validate(self, meta: dict, items: dict) -> dict:
        LOGGER.info("Running local transformer backend (%s)", self.model_name)
        text = _compose_invoice_summary(meta, items)
        try:
            prediction = self._pipeline(text[:512])[0]
        except Exception as exc:
            LOGGER.error("Transformer inference failed: %s", exc)
            return {
                "backend": "local_transformer",
                "model": self.model_name,
                "notes": "Transformer inference failed; falling back to stub-style response.",
                "issues": [
                    {
                        "code": "transformer-error",
                        "message": str(exc),
                    }
                ],
            }

        return {
            "backend": "local_transformer",
            "model": self.model_name,
            "issues": [
                {
                    "code": prediction.get("label", "transformer-score"),
                    "message": f"confidence={prediction.get('score'):.3f}",
                }
            ],
            "notes": "Heuristic transformer classification only.",
        }


def build_llm_backend(config: Any | None) -> LLMBackend:
    if not config:
        return StubLLMBackend(name="stub")
    backend_name = getattr(config, "backend", "stub") or "stub"
    model_name = getattr(config, "model_name", None) or "sshleifer/tiny-distilbert-base-cased-distilled-sst-2"
    backend_normalized = backend_name.lower()
    if backend_normalized == "local_transformer":
        try:
            return LocalTransformerBackend(model_name=model_name)
        except RuntimeError as exc:
            LOGGER.warning("Failed to initialize local transformer backend: %s", exc)
            return StubLLMBackend(name="stub")
    return StubLLMBackend(name=backend_name)


def _compose_invoice_summary(meta: Dict[str, Any], items: Dict[str, Any]) -> str:
    try:
        snippet = json.dumps({"meta": meta, "items": items.get("rows", [])[:3]}, ensure_ascii=False)
    except Exception:
        snippet = str(meta)
    return snippet
