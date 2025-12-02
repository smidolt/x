"""Rule-only validation service."""
from __future__ import annotations

from typing import Dict, List

from src.validation.rules import RuleValidator


def run(payload: Dict[str, object]) -> Dict[str, object]:
    meta = payload.get("meta") or {}
    items = payload.get("items") or {}
    seller_name = str(payload.get("seller_name", ""))
    seller_tax_id = str(payload.get("seller_tax_id", ""))
    required_fields = payload.get("required_fields")
    amount_tolerance = float(payload.get("amount_tolerance", 0.5))

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
    result = validator.run(meta, items)
    return {
        "errors": result.errors,
        "warnings": result.warnings,
        "is_valid": result.is_valid,
    }
