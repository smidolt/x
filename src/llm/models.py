"""Model registry and lightweight loaders for local LLM validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

import logging

try:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
except Exception as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("transformers/torch are required to run llm models") from exc

try:
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:  # pragma: no cover - optional quant
    BitsAndBytesConfig = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelSpec:
    key: str
    model_id: str
    quant: str = "4bit"  # 4bit|8bit|fp16|bf16|auto
    max_new_tokens: int = 512
    temperature: float = 0.2
    chat_template: Optional[str] = None


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "llama31-8b": ModelSpec("llama31-8b", "meta-llama/Meta-Llama-3.1-8B-Instruct", "4bit"),
    "llama31-70b": ModelSpec("llama31-70b", "meta-llama/Meta-Llama-3.1-70B-Instruct", "4bit"),
    "qwen2-7b": ModelSpec("qwen2-7b", "Qwen/Qwen2-7B-Instruct", "4bit"),
    "qwen2-72b": ModelSpec("qwen2-72b", "Qwen/Qwen2-72B-Instruct", "4bit"),
    "mistral-7b": ModelSpec("mistral-7b", "mistralai/Mistral-7B-Instruct-v0.3", "4bit"),
    "mixtral-8x7b": ModelSpec("mixtral-8x7b", "mistralai/Mixtral-8x7B-Instruct-v0.1", "4bit"),
    "deepseek-7b": ModelSpec("deepseek-7b", "deepseek-ai/deepseek-llm-7b-chat", "4bit"),
    "phi3-medium": ModelSpec("phi3-medium", "microsoft/Phi-3-medium-4k-instruct", "4bit"),
}


def _build_quant_config(mode: str) -> Any:
    if mode in {"4bit", "8bit"} and BitsAndBytesConfig is not None:
        if mode == "4bit":
            return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_generator(spec: ModelSpec):
    """Return a text-generation pipeline for the given spec."""
    quant_cfg = _build_quant_config(spec.quant)
    dtype = torch.bfloat16 if spec.quant in {"4bit", "8bit", "bf16"} else torch.float16
    LOGGER.info("Loading model %s (%s)", spec.model_id, spec.quant)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        device_map="auto",
        torch_dtype=dtype,
        quantization_config=quant_cfg,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
    )


def get_spec(key: str) -> ModelSpec:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key '{key}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]
