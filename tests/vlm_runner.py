"""Quick VLM smoke test (reasoner only, optional blocks).

Designed for invoice extraction expectations:
- meta should include seller/buyer names/addresses, tax IDs (DDV), dates (issue/supply), invoice number,
  currency, total_net, total_vat, total_gross, and VAT exemption reason if applicable.
- items should list description, quantity, unit_price, amount, currency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.vlm.services import run_blocks, run_reasoner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM reasoner (and optional blocks) on one image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to image.")
    parser.add_argument("--ocr-json", type=Path, help="Optional OCR JSON for blocks (uses heuristic backend).")
    parser.add_argument("--output", type=Path, default=Path("output/vlm_single.json"), help="Where to save VLM result.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="VLM model name.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto|cuda|mps|cpu.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--prompt", type=str, default="", help="Custom prompt (optional).")
    args = parser.parse_args()

    # Prompt that nudges required invoice fields (Slovenia DDV context)
    prompt = (
        "Return ONLY valid JSON with keys: meta, items, notes. "
        "meta must include seller_name, seller_address, seller_tax_id (DDV if exists), "
        "buyer_name, buyer_address, buyer_tax_id (if exists), invoice_number, issue_date, supply_date, "
        "currency, total_net, total_vat, total_gross, vat_exemption_reason (if applicable). "
        "items must include description, quantity, unit_price, amount, currency. notes as array. "
        "No markdown, no extra text. Close all braces."
    )

    payload = {
        "image_path": str(args.image),
        "params": {
            "model_name": args.model_name,
            "device": args.device,
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
            "prompt": args.prompt or prompt,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
        },
    }
    result = {}
    if args.ocr_json:
        blocks = run_blocks({"ocr_json_path": str(args.ocr_json), "backend": "heuristic"})
        result["vlm_blocks"] = blocks
    reasoner_res = run_reasoner(payload)
    result["vlm_reasoner"] = {
        "raw_response": reasoner_res.get("raw_response"),
        "parsed": reasoner_res.get("parsed"),
        "error": reasoner_res.get("error"),
        "elapsed_seconds": reasoner_res.get("elapsed_seconds"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved VLM result to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
