"""Reusable validation checks for VLM JSON."""
from __future__ import annotations

from typing import Any, Dict, List


def _missing_fields(meta: Dict[str, Any], required: List[str]) -> List[str]:
    return [f for f in required if not meta.get(f)]


def check_required(meta: Dict[str, Any], items: Dict[str, Any]) -> List[str]:
    required_meta = [
        "seller_name",
        "seller_address",
        "buyer_name",
        "buyer_address",
        "invoice_number",
        "issue_date",
        "supply_date",
        "currency",
        "total_net",
        "total_vat",
        "total_gross",
    ]
    errors = [f"Missing required meta field: {f}" for f in _missing_fields(meta, required_meta)]
    rows = items.get("items") or items.get("rows") or items.get("items", [])
    if not rows:
        errors.append("Missing items")
    else:
        for idx, row in enumerate(rows):
            if not row.get("description"):
                errors.append(f"Item {idx} missing description")
            if row.get("amount") is None and row.get("net_amount") is None and row.get("gross_amount") is None:
                errors.append(f"Item {idx} missing amount")
    return errors


def check_math(meta: Dict[str, Any], items: Dict[str, Any], tol: float = 0.5) -> List[str]:
    errors: List[str] = []
    rows = items.get("items") or items.get("rows") or []
    net_sum = 0.0
    vat_sum = 0.0
    for row in rows:
        net = row.get("net_amount")
        gross = row.get("gross_amount") or row.get("amount")
        vat = row.get("vat_amount")
        if net is not None:
            net_sum += float(net)
        if vat is not None:
            vat_sum += float(vat)
        if net is not None and vat is not None and gross is not None:
            if abs((net + vat) - float(gross)) > tol:
                errors.append("Item net+vat != gross")

    total_net = meta.get("total_net")
    total_vat = meta.get("total_vat")
    total_gross = meta.get("total_gross")

    if total_net is not None and abs(net_sum - float(total_net)) > tol:
        errors.append("Sum items net != total_net")
    if total_vat is not None and abs(vat_sum - float(total_vat)) > tol:
        errors.append("Sum items vat != total_vat")
    if total_net is not None and total_vat is not None and total_gross is not None:
        if abs((float(total_net) + float(total_vat)) - float(total_gross)) > tol:
            errors.append("total_net + total_vat != total_gross")
    return errors


def check_vat_exemption(meta: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    reason = meta.get("vat_exemption_reason")
    total_vat = meta.get("total_vat")
    if reason and total_vat is not None and abs(float(total_vat)) > 0.5:
        warnings.append("VAT exemption reason given but total_vat not near zero")
    return warnings
