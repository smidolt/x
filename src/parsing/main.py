"""Standalone classic parsing runner (meta + items + rule validation)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.parsing.meta import MetaParser
from src.parsing.items import ItemsParser
from src.validation.rules import RuleValidator
from src.validation.llm import build_llm_backend


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_classic(
    ocr_json: Path,
    seller_name: str,
    seller_tax_id: str,
    layout_json: Path | None = None,
    required_fields: List[str] | None = None,
    amount_tolerance: float = 0.5,
) -> Dict[str, Any]:
    layout_data = _load_json(layout_json) if layout_json else {}

    meta_parser = MetaParser(seller_name, seller_tax_id)
    items_parser = ItemsParser()
    rule_validator = RuleValidator(
        required_fields=required_fields
        or [
            "seller_name",
            "seller_address",
            "buyer_name",
            "buyer_address",
            "invoice_number",
            "issue_date",
            "supply_date",
            "payment_date",
            "total_net",
            "total_vat",
            "total_gross",
            "seller_tax_id",
        ],
        seller_name=seller_name,
        seller_tax_id=seller_tax_id,
        amount_tolerance=amount_tolerance,
    )
    llm_backend = build_llm_backend(None)  # stub by default

    meta = meta_parser.run(ocr_json, layout_data)
    items = items_parser.run(ocr_json, layout_data)
    validation = rule_validator.run(meta.raw, items.__dict__)
    llm_validation = llm_backend.validate(meta.raw, items.__dict__)

    return {
        "meta": meta.raw,
        "items": items.__dict__,
        "validation": {
            "rules": {
                "errors": validation.errors,
                "warnings": validation.warnings,
                "is_valid": validation.is_valid,
            },
            "llm": llm_validation,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classic parsing (meta + items + validation) on OCR JSON.")
    parser.add_argument("--ocr-json", type=Path, required=True, help="Path to OCR JSON file.")
    parser.add_argument("--seller-name", type=str, required=True, help="Expected seller name.")
    parser.add_argument("--seller-tax-id", type=str, required=True, help="Expected seller tax ID.")
    parser.add_argument("--layout-json", type=Path, help="Optional layout annotations JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    result = run_classic(
        ocr_json=args.ocr_json,
        seller_name=args.seller_name,
        seller_tax_id=args.seller_tax_id,
        layout_json=args.layout_json,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote parsing result to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
