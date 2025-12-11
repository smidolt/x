"""Single-model runner for LLM-based invoice validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import logging

from src.llm.models import get_spec, load_generator

LOGGER = logging.getLogger(__name__)


def load_payload(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read payload {path}: {exc}") from exc


def build_prompt(payload: Dict[str, Any]) -> str:
    meta = payload.get("meta", {})
    items = payload.get("items", {})
    required_fields = payload.get("required_fields", [])
    tolerance = payload.get("amount_tolerance", 0.5)
    raw_vlm = payload.get("raw_vlm") or ""
    schema_errors = payload.get("schema_errors", [])

    return (
        "You are an invoice validation model. Given extracted meta/items, "
        "validate and fix them. Steps:\n"
        "1) Check all required fields exist and are non-empty.\n"
        "2) Check sums: net + vat ≈ gross (tolerance), sum(items.net_amount) ≈ total_net, "
        "sum(items.vat_amount) ≈ total_vat. Use amount_tolerance.\n"
        "3) Fix missing/incorrect fields if obvious from context; prefer leaving unknown as null.\n"
        "4) Return ONLY JSON with keys: meta, items, validation. "
        "validation.issues is an array of {code, message, field}. "
        "Do not include markdown or explanations outside JSON.\n\n"
        f"required_fields: {required_fields}\n"
        f"amount_tolerance: {tolerance}\n"
        f"schema_errors: {schema_errors}\n"
        "meta:\n"
        f"{json.dumps(meta, ensure_ascii=False, indent=2)}\n"
        "items:\n"
        f"{json.dumps(items, ensure_ascii=False, indent=2)}\n"
        "raw_vlm_snippet:\n"
        f"{raw_vlm[:1200]}\n"
        "Return JSON now:\n"
    )


def run_model(
    model_key: str,
    payload: Dict[str, Any],
    max_new_tokens: int | None,
    temperature: float | None,
) -> Dict[str, Any]:
    spec = get_spec(model_key)
    gen = load_generator(spec)
    prompt = build_prompt(payload)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens or spec.max_new_tokens,
        "temperature": temperature if temperature is not None else spec.temperature,
        "do_sample": True,
        "top_p": 0.9,
    }
    LOGGER.info("Running model %s", model_key)
    result = gen(prompt, **gen_kwargs)[0]["generated_text"]
    if result.startswith(prompt):
        result = result[len(prompt):]
    return {
        "model": model_key,
        "output": result.strip(),
        "prompt_used": prompt,
        "gen_kwargs": gen_kwargs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one LLM validator on a payload JSON.")
    parser.add_argument("--model", type=str, required=True, help="Model key (see models.py).")
    parser.add_argument("--payload", type=Path, required=True, help="Path to payload JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Where to write model output.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max_new_tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    payload = load_payload(args.payload)
    res = run_model(args.model, payload, args.max_new_tokens, args.temperature)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote result to {args.output}")
    else:
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
