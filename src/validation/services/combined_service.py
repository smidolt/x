"""Rule + LLM validation service (LLM stub by default)."""
from __future__ import annotations

from typing import Dict, List

from src.validation.rules import RuleValidator
from src.validation.llm import build_llm_backend


def run(payload: Dict[str, object]) -> Dict[str, object]:
    meta = payload.get("meta") or {}
    items = payload.get("items") or {}
    seller_name = str(payload.get("seller_name", ""))
    seller_tax_id = str(payload.get("seller_tax_id", ""))
    required_fields = payload.get("required_fields")
    amount_tolerance = float(payload.get("amount_tolerance", 0.5))
    llm_cfg = payload.get("llm_config")

    required: List[str] = list(required_fields) if isinstance(required_fields, list) else [
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
    ]

    validator = RuleValidator(
        required_fields=required,
        seller_name=seller_name,
        seller_tax_id=seller_tax_id,
        amount_tolerance=amount_tolerance,
    )
    llm_backend = build_llm_backend(llm_cfg)

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
