"""Business validation for VLM outputs (invoice requisites)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import logging

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ValidationResult:
    errors: List[str]
    warnings: List[str]

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(slots=True)
class VLMValidator:
    required_fields: Iterable[str]
    seller_name: str
    seller_tax_id: str
    amount_tolerance: float = 0.5

    def run(self, meta: Dict[str, Any], items: Dict[str, Any]) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        errors.extend(_check_required_fields(meta, items, self.required_fields))
        amount_errors = _check_amounts(meta, items, self.amount_tolerance)
        errors.extend(amount_errors["errors"])
        warnings.extend(amount_errors["warnings"])

        identity_error = _check_seller_identity(meta, self.seller_name, self.seller_tax_id)
        if identity_error:
            errors.append(identity_error)

        vat_warning = _check_vat_exemption(meta)
        if vat_warning:
            warnings.append(vat_warning)

        return ValidationResult(errors=errors, warnings=warnings)


def _check_required_fields(meta: Dict[str, Any], items: Dict[str, Any], required_fields: Iterable[str]) -> List[str]:
    missing = [field for field in required_fields if not meta.get(field)]
    errors = [f"Missing required field: {field}" for field in missing]

    rows = items.get("rows", []) if isinstance(items, dict) else []
    if not rows:
        errors.append("No item rows detected")
    else:
        for idx, row in enumerate(rows):
            if not row.get("description"):
                errors.append(f"Item {idx} missing description")
            if row.get("unit_price") is None and row.get("net_amount") is None:
                errors.append(f"Item {idx} missing price/net amount")
    return errors


def _check_amounts(meta: Dict[str, Any], items: Dict[str, Any], tolerance: float) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    rows = items.get("rows", []) if isinstance(items, dict) else []
    net_sum = sum(row.get("net_amount", 0) or 0 for row in rows if row.get("net_amount") is not None)
    vat_sum = sum(row.get("vat_amount", 0) or 0 for row in rows if row.get("vat_amount") is not None)
    total_net = meta.get("total_net")
    total_vat = meta.get("total_vat")
    total_gross = meta.get("total_gross")

    if total_net is not None and rows:
        if abs(net_sum - total_net) > tolerance:
            errors.append(f"Line items net sum {net_sum:.2f} does not match meta total_net {total_net:.2f}")

    if total_vat is not None and vat_sum:
        if abs(vat_sum - total_vat) > tolerance:
            errors.append(f"Line items VAT sum {vat_sum:.2f} does not match meta total_vat {total_vat:.2f}")

    if total_net is not None and total_vat is not None and total_gross is not None:
        if abs((total_net + total_vat) - total_gross) > tolerance:
            errors.append(
                f"total_net + total_vat ({(total_net + total_vat):.2f}) does not equal total_gross {total_gross:.2f}"
            )
    return {"errors": errors, "warnings": warnings}


def _check_seller_identity(meta: Dict[str, Any], expected_name: str, expected_tax_id: str) -> str | None:
    if not expected_name:
        return None
    seller_name = meta.get("seller_name")
    seller_tax_id = meta.get("seller_tax_id")
    if seller_name and expected_name.lower() not in seller_name.lower():
        return f"Seller name '{seller_name}' does not match expected '{expected_name}'"
    if seller_tax_id and expected_tax_id and expected_tax_id.lower() not in seller_tax_id.lower():
        return f"Seller tax ID '{seller_tax_id}' does not match expected '{expected_tax_id}'"
    return None


def _check_vat_exemption(meta: Dict[str, Any]) -> str | None:
    reason = meta.get("vat_exemption_reason")
    total_vat = meta.get("total_vat")
    if reason and total_vat is not None and abs(total_vat) > 0.01:
        return "VAT exemption stated but total_vat is not zero"
    return None
