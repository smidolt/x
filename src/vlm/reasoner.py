"""Vision-Language reasoner for invoices (Qwen2-VL)."""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image
import torch
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

from src.config import VLMConfig

LOGGER = logging.getLogger(__name__)

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION_IMPORT", "1")
os.environ.setdefault("FLASH_ATTENTION_SKIP", "1")
os.environ.setdefault("USE_FLASH_ATTENTION_2", "0")
os.environ.setdefault("USE_FLASH_ATTENTION", "0")
os.environ.setdefault("FLASH_ATTENTION_FORCE_DISABLE", "1")
os.environ.setdefault("ATTN_IMPLEMENTATION", "eager")


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" or (device_arg == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if device_arg == "mps" or (device_arg == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class VLMReasonerResult:
    raw_response: str
    elapsed_seconds: float
    parsed: Optional[Dict[str, object]]
    error: Optional[str] = None


class VLMReasoner:
    def __init__(self, cfg: VLMConfig) -> None:
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        dtype = torch.float16 if self.device.type in {"cuda", "mps"} else torch.float32

        config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "eager"
        for attr in ("use_flash_attention_2", "use_flash_attn", "flash_attn", "flash_attention"):
            if hasattr(config, attr):
                setattr(config, attr, False)

        self.processor = AutoProcessor.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=dtype,
            config=config,
        )
        self.model.to(self.device, dtype=dtype)
        self.model.eval()

    def _build_inputs(self, image: Image.Image, prompt: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            text=chat_prompt,
            images=[image],
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def run(self, image_path: Path) -> VLMReasonerResult:
        image = Image.open(image_path).convert("RGB")
        inputs = self._build_inputs(image, self.cfg.prompt.strip())
        gen_kwargs = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "do_sample": False,
            "use_cache": False,
        }

        start = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - start

        decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        raw = decoded[0] if decoded else ""

        parsed = None
        error = None
        try:
            parsed = json.loads(raw)
        except Exception as exc:
            error = f"Failed to parse JSON: {exc}"
        return VLMReasonerResult(raw_response=raw, elapsed_seconds=elapsed, parsed=parsed, error=error)
