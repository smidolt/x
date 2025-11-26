"""Rule-based validation for invoices."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import logging

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RuleValidationResult:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(slots=True)
class RuleValidator:
    required_fields: Sequence[str]
    seller_name: str
    seller_tax_id: str
    amount_tolerance: float = 0.5

    def run(self, meta: Dict[str, any], items: Dict[str, any]) -> RuleValidationResult:
        LOGGER.info("Running rule-based validation")
        errors: List[str] = []
        warnings: List[str] = []

        errors.extend(_check_required_fields(meta, self.required_fields))
        amount_errors = _check_amounts(meta, items, self.amount_tolerance)
        errors.extend(amount_errors["errors"])
        warnings.extend(amount_errors["warnings"])

        identity_error = _check_seller_identity(meta, self.seller_name, self.seller_tax_id)
        if identity_error:
            errors.append(identity_error)

        vat_warning = _check_vat_exemption(meta)
        if vat_warning:
            warnings.append(vat_warning)

        return RuleValidationResult(errors=errors, warnings=warnings)


def _check_required_fields(meta: Dict[str, any], required_fields: Iterable[str]) -> List[str]:
    missing = [field for field in required_fields if not meta.get(field)]
    return [f"Missing required field: {field}" for field in missing]


def _check_amounts(meta: Dict[str, any], items: Dict[str, any], tolerance: float) -> Dict[str, List[str]]:
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
            errors.append(
                f"Line items net sum {net_sum:.2f} does not match meta total_net {total_net:.2f}"
            )

    if total_vat is not None and vat_sum:
        if abs(vat_sum - total_vat) > tolerance:
            errors.append(
                f"Line items VAT sum {vat_sum:.2f} does not match meta total_vat {total_vat:.2f}"
            )

    if total_net is not None and total_vat is not None and total_gross is not None:
        if abs((total_net + total_vat) - total_gross) > tolerance:
            errors.append(
                f"total_net + total_vat ({total_net + total_vat:.2f}) does not equal total_gross {total_gross:.2f}"
            )
    return {"errors": errors, "warnings": warnings}


def _check_seller_identity(meta: Dict[str, any], expected_name: str, expected_tax_id: str) -> str | None:
    if not expected_name:
        return None
    seller_name = meta.get("seller_name")
    seller_tax_id = meta.get("seller_tax_id")
    if seller_name and expected_name.lower() not in seller_name.lower():
        return f"Seller name '{seller_name}' does not match configured '{expected_name}'"
    if seller_tax_id and expected_tax_id and expected_tax_id.lower() not in seller_tax_id.lower():
        return f"Seller tax ID '{seller_tax_id}' does not match configured '{expected_tax_id}'"
    return None


def _check_vat_exemption(meta: Dict[str, any]) -> str | None:
    reason = meta.get("vat_exemption_reason")
    total_vat = meta.get("total_vat")
    if reason and total_vat and abs(total_vat) > 0.01:
        return "VAT exemption stated but total_vat is not zero"
    return None
