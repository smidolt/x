"""VLM reasoner using Qwen2-VL with official processing utils."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from src.config import VLMConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class VLMReasonerResult:
    raw_response: str
    elapsed_seconds: float
    parsed: Optional[Dict[str, object]]
    error: Optional[str] = None


def _build_prompt(custom_prompt: str | None = None) -> str:
    base = (
        "Return ONLY valid JSON with keys: meta, items, notes. Example: "
        '{"meta": {"seller_name": "...", "buyer_name": "...", "invoice_number": "...", "currency": "...", '
        '"total_net": 0, "total_vat": 0, "total_gross": 0}, '
        '"items": [{"description": "...", "quantity": 1, "unit_price": 0, "amount": 0, "currency": "..."}], '
        '"notes": []}'
    )
    return custom_prompt.strip() if custom_prompt else base


def _prepare_inputs(image_path: Path, processor: AutoProcessor, prompt: str) -> Dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs


class VLMReasoner:
    def __init__(self, cfg: VLMConfig) -> None:
        self.cfg = cfg
        self.model_name = cfg.model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.device in {"auto", "cuda"} else "cpu")
        LOGGER.info("Loading VLM model %s on %s", self.model_name, self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
        )
        if self.device.type == "cpu":
            self.model = self.model.to(self.device)

    def run(self, image_path: Path) -> VLMReasonerResult:
        prompt = _build_prompt(self.cfg.prompt)
        inputs = _prepare_inputs(image_path, self.processor, prompt)
        inputs = inputs.to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
        }

        start = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - start

        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw = output_text[0] if output_text else ""

        parsed = None
        error = None
        cleaned = raw.strip()
        # Remove markdown fences if present
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        # Try to extract JSON substring between first { and last }
        if "{" in cleaned and "}" in cleaned:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            cleaned = cleaned[start:end]
        try:
            parsed = json.loads(cleaned)
        except Exception as exc:
            error = f"Failed to parse JSON: {exc}"
        return VLMReasonerResult(raw_response=raw, elapsed_seconds=elapsed, parsed=parsed, error=error)
