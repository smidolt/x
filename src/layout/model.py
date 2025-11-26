"""LayoutLM integration with graceful fallback."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import json
import logging
import os
from PIL import Image
from huggingface_hub import snapshot_download

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import torch
    from transformers import AutoModelForTokenClassification, AutoProcessor

    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - dependency guard
    TRANSFORMERS_AVAILABLE = False
    torch = None  # type: ignore
    AutoModelForTokenClassification = AutoProcessor = None  # type: ignore


LABELS = [
    "other",
    "header",
    "key",
    "value",
    "table_header",
    "table_row",
    "footer",
]


@dataclass(slots=True)
class LayoutAnnotations:
    tokens: List[Dict[str, Any]]
    embeddings: List[List[float]] | None = None
    model_name: str | None = None
    engine: str = "stub"


class LayoutAnalyzer:
    """Runs LayoutLM (v3 or compatible) on OCR tokens to classify layout zones."""

    def __init__(
        self,
        model_name: str,
        enabled: bool = True,
        label_schema: Sequence[str] | None = None,
        offline: bool = False,
    ) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self.label_schema = list(label_schema) if label_schema else LABELS
        self.offline = offline
        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self._backend: BaseLayoutBackend = self._build_backend()

    def _build_backend(self) -> "BaseLayoutBackend":
        if not self.enabled:
            return StubLayoutBackend("disabled")
        if not TRANSFORMERS_AVAILABLE:
            LOGGER.warning("Transformers/torch not available; layout model will be stubbed")
            return StubLayoutBackend("missing_dependencies")
        try:
            return HuggingFaceLayoutBackend(
                self.model_name,
                self.label_schema,
                offline=self.offline,
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            LOGGER.warning("Failed to initialize layout model %s: %s", self.model_name, exc)
            return StubLayoutBackend("model_init_failed")

    def run(self, ocr_result_path: Path) -> LayoutAnnotations:
        LOGGER.info("Running layout model (%s) for %s", self.model_name, ocr_result_path)
        annotations = self._backend.annotate(ocr_result_path)
        return annotations


class BaseLayoutBackend:
    name = "base"

    def annotate(self, ocr_result_path: Path) -> LayoutAnnotations:  # pragma: no cover - interface
        raise NotImplementedError


class StubLayoutBackend(BaseLayoutBackend):
    name = "stub"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def annotate(self, ocr_result_path: Path) -> LayoutAnnotations:
        LOGGER.warning("Using stub layout backend (%s) for %s", self.reason, ocr_result_path)
        data = _load_tokens(ocr_result_path)
        for token in data["tokens"]:
            token["label"] = "other"
            token["score"] = 0.0
        return LayoutAnnotations(tokens=data["tokens"], embeddings=None, model_name=None, engine=self.name)


class HuggingFaceLayoutBackend(BaseLayoutBackend):  # pragma: no cover - heavy dep
    name = "huggingface"

    def __init__(self, model_name: str, labels: Sequence[str], offline: bool) -> None:
        self.model_name = model_name
        self.labels = list(labels)
        self.offline = offline
        LOGGER.info("Loading LayoutLM model %s", model_name)
        model_source = model_name
        if offline:
            model_source = snapshot_download(model_name, local_files_only=True)
        self.processor = _load_hf_component(
            AutoProcessor,
            model_source,
            local_first=True,
            local_only=offline,
        )
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor and hasattr(image_processor, "apply_ocr"):
            image_processor.apply_ocr = False
        self.model = _load_hf_component(
            AutoModelForTokenClassification,
            model_source,
            local_first=True,
            local_only=offline,
            config={"num_labels": len(labels)},
        )
        self.model.eval()

    def annotate(self, ocr_result_path: Path) -> LayoutAnnotations:
        data = _load_tokens(ocr_result_path)
        tokens = data["tokens"]
        image = _load_image(data.get("image_path"))
        if not tokens or image is None:
            return LayoutAnnotations(tokens=[], embeddings=None, model_name=self.model_name, engine=self.name)

        texts = [token["text"] for token in tokens]
        boxes = [token["bbox_layoutlm"] for token in tokens]
        encoding = self.processor(
            images=image,
            text=texts,
            boxes=boxes,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**encoding, output_hidden_states=True)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            max_scores, pred_ids = torch.max(probs, dim=-1)

        word_ids = encoding.word_ids()
        annotated_tokens: List[Dict[str, Any]] = []
        embeddings: List[List[float]] = []
        if word_ids is None:
            index_pairs = [(idx, idx) for idx in range(min(len(tokens), 512))]
        else:
            index_pairs = []
            used_words = set()
            for idx, word_id in enumerate(word_ids):
                if word_id is None or word_id in used_words:
                    continue
                used_words.add(word_id)
                if word_id >= len(tokens):
                    continue
                index_pairs.append((idx, word_id))

        for logit_idx, token_idx in index_pairs:
            label_id = int(pred_ids[logit_idx].item())
            score = float(max_scores[logit_idx].item())
            label = self.labels[label_id] if label_id < len(self.labels) else "other"
            token = tokens[token_idx]
            annotated_tokens.append({
                "text": token["text"],
                "bbox": token["bbox"],
                "label": label,
                "score": score,
            })
            if outputs.hidden_states:
                embeddings.append(outputs.hidden_states[-1][0, logit_idx].tolist())

        return LayoutAnnotations(
            tokens=annotated_tokens,
            embeddings=embeddings if embeddings else None,
            model_name=self.model_name,
            engine=self.name,
        )


def _load_tokens(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.error("Failed to read OCR payload from %s", path)
        return {"tokens": [], "image_path": None}

    words = payload.get("words", [])
    if not words:
        return {"tokens": [], "image_path": payload.get("source")}

    max_x = max((word.get("bbox", [0, 0, 0, 0])[2] for word in words), default=1) or 1
    max_y = max((word.get("bbox", [0, 0, 0, 0])[3] for word in words), default=1) or 1
    scale_x = 1000.0 / max_x
    scale_y = 1000.0 / max_y

    tokens: List[Dict[str, Any]] = []
    for word in words:
        text = (word.get("text") or "").strip()
        if not text:
            continue
        bbox = word.get("bbox", [0, 0, 0, 0])
        bbox_layoutlm = _normalize_bbox(bbox, scale_x, scale_y)
        tokens.append(
            {
                "text": text,
                "bbox": bbox,
                "bbox_layoutlm": bbox_layoutlm,
            }
        )
    return {"tokens": tokens, "image_path": payload.get("source")}


def _normalize_bbox(bbox: Sequence[int], scale_x: float, scale_y: float) -> List[int]:
    coords = list(bbox) + [0, 0, 0, 0]
    left, top, right, bottom = coords[:4]
    left = int(max(0, min(1000, left * scale_x)))
    right = int(max(0, min(1000, right * scale_x)))
    top = int(max(0, min(1000, top * scale_y)))
    bottom = int(max(0, min(1000, bottom * scale_y)))
    return [left, top, right, bottom]


def _load_hf_component(
    cls,
    model_name: str,
    local_first: bool = False,
    local_only: bool = False,
    config: Dict[str, Any] | None = None,
):
    params = config or {}
    if local_first or local_only:
        try:
            return cls.from_pretrained(model_name, local_files_only=True, **params)
        except Exception as exc:
            if local_only:
                raise RuntimeError(
                    f"Model {model_name} not found in local cache while offline"
                ) from exc
            LOGGER.debug("Local cache miss for %s; trying remote", model_name)
    if local_only:
        raise RuntimeError(f"Model {model_name} requested offline but not cached")
    return cls.from_pretrained(model_name, **params)


def _load_image(path_str: str | None) -> Image.Image | None:
    if not path_str:
        return None
    image_path = Path(path_str)
    if not image_path.exists():
        LOGGER.warning("Layout image path not found: %s", image_path)
        return None
    try:
        with Image.open(image_path) as img:
            return img.convert("RGB")
    except Exception as exc:
        LOGGER.warning("Failed to load image %s: %s", image_path, exc)
        return None
