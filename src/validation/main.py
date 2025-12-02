"""Standalone validation runner (rule-based + LLM stub)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.validation.rules import RuleValidator
from src.validation.llm import build_llm_backend


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_validation(
    meta: Dict[str, Any],
    items: Dict[str, Any],
    seller_name: str,
    seller_tax_id: str,
    required_fields: List[str] | None = None,
    amount_tolerance: float = 0.5,
) -> Dict[str, Any]:
    validator = RuleValidator(
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
    llm_backend = build_llm_backend(None)  # stub unless configured

    rules_result = validator.run(meta, items)
    llm_result = llm_backend.validate(meta, items)
    return {
        "rules": {
            "errors": rules_result.errors,
            "warnings": rules_result.warnings,
            "is_valid": rules_result.is_valid,
        },
        "llm": llm_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation on meta/items JSON payloads.")
    parser.add_argument("--meta-json", type=Path, required=True, help="Path to meta JSON.")
    parser.add_argument("--items-json", type=Path, required=True, help="Path to items JSON.")
    parser.add_argument("--seller-name", type=str, required=True, help="Expected seller name.")
    parser.add_argument("--seller-tax-id", type=str, required=True, help="Expected seller tax ID.")
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    parser.add_argument("--amount-tolerance", type=float, default=0.5, help="Tolerance for sums.")
    args = parser.parse_args()

    meta = _load_json(args.meta_json)
    items = _load_json(args.items_json)

    result = run_validation(
        meta=meta,
        items=items,
        seller_name=args.seller_name,
        seller_tax_id=args.seller_tax_id,
        amount_tolerance=args.amount_tolerance,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote validation to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
