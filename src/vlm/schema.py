"""Schema helpers and light normalization for VLM outputs.

This module avoids heavy dependencies (no pydantic) and provides:
- key normalization (camelCase → snake_case)
- type coercion for dates/numbers/currency
- structural validation with human-readable errors
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

DATE_PATTERNS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
]

MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "SEPT": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

KEY_ALIASES = {
    "sellername": "seller_name",
    "seller": "seller_name",
    "selleraddress": "seller_address",
    "sellertaxid": "seller_tax_id",
    "seller_taxid": "seller_tax_id",
    "buyername": "buyer_name",
    "buyer": "buyer_name",
    "buyeraddress": "buyer_address",
    "buyertaxid": "buyer_tax_id",
    "invoiceid": "invoice_number",
    "invoicenumber": "invoice_number",
    "date": "issue_date",
    "issuedate": "issue_date",
    "supplydate": "supply_date",
    "duedate": "payment_date",
    "paymentdate": "payment_date",
    "total": "total_gross",
    "totalgross": "total_gross",
    "totalnet": "total_net",
    "totalvat": "total_vat",
    "vatrate": "vat_rate",
    "vatreason": "vat_exemption_reason",
    "vat_exemption": "vat_exemption_reason",
}

CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}


def normalize_currency(value: Any, hint: str | None = None) -> str | None:
    if isinstance(value, str):
        upper = value.strip().upper()
        if upper in {"USD", "EUR", "GBP"}:
            return upper
        for symbol, code in CURRENCY_SYMBOLS.items():
            if symbol in value:
                return code
    if hint:
        return hint.upper()
    return None


def normalize_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        cleaned = cleaned.replace("\u00a0", "").replace(" ", "")
        cleaned = cleaned.replace(",", "")
        cleaned = cleaned.lstrip("$€£")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None
    if isinstance(value, str):
        text = value.strip()
        for pattern in DATE_PATTERNS:
            try:
                return datetime.strptime(text, pattern).strftime("%Y-%m-%d")
            except ValueError:
                continue
        # Textual month (e.g., Jan 2, 2024)
        match = re.search(r"([A-Za-z]{3,})\s+(\d{1,2}),\s*(\d{4})", text)
        if match:
            month = MONTHS.get(match.group(1).upper()[:3])
            if month:
                try:
                    return datetime(int(match.group(3)), month, int(match.group(2))).strftime("%Y-%m-%d")
                except ValueError:
                    return None
    return None


def normalize_meta(raw: Dict[str, Any], currency_hint: str | None = None) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for key, value in raw.items():
        normalized_key = KEY_ALIASES.get(key.lower().replace(" ", "").replace("-", "").replace("_", ""), key)
        meta[normalized_key] = value

    meta["seller_name"] = _str(meta.get("seller_name"))
    meta["seller_address"] = _str(meta.get("seller_address"))
    meta["seller_tax_id"] = _str(meta.get("seller_tax_id"))
    meta["buyer_name"] = _str(meta.get("buyer_name"))
    meta["buyer_address"] = _str(meta.get("buyer_address"))
    meta["buyer_tax_id"] = _str(meta.get("buyer_tax_id"))
    meta["invoice_number"] = _str(meta.get("invoice_number"))
    meta["currency"] = normalize_currency(meta.get("currency"), hint=currency_hint)
    meta["total_net"] = normalize_number(meta.get("total_net"))
    meta["total_vat"] = normalize_number(meta.get("total_vat"))
    meta["total_gross"] = normalize_number(meta.get("total_gross"))
    meta["vat_rate"] = normalize_number(meta.get("vat_rate"))
    meta["vat_exemption_reason"] = _str(meta.get("vat_exemption_reason"))
    meta["payment_method"] = _str(meta.get("payment_method"))
    meta["payment_date"] = normalize_date(meta.get("payment_date"))
    meta["issue_date"] = normalize_date(meta.get("issue_date"))
    meta["supply_date"] = normalize_date(meta.get("supply_date"))
    if meta.get("notes") is None:
        meta["notes"] = []
    elif isinstance(meta.get("notes"), list):
        meta["notes"] = [_str(n) for n in meta.get("notes") if _str(n)]
    else:
        meta["notes"] = [_str(meta.get("notes"))] if _str(meta.get("notes")) else []
    return meta


def normalize_items(raw: Any, currency_hint: str | None = None) -> Dict[str, Any]:
    rows_in = raw or []
    rows_out: List[Dict[str, Any]] = []
    for row in rows_in:
        if not isinstance(row, dict):
            continue
        currency = normalize_currency(row.get("currency"), hint=currency_hint)
        amount = normalize_number(row.get("amount"))
        net_amount = normalize_number(row.get("net_amount"))
        gross_amount = normalize_number(row.get("gross_amount"))
        # Fallback: use amount if explicit net/gross are missing
        if net_amount is None and amount is not None:
            net_amount = amount
        if gross_amount is None and amount is not None:
            gross_amount = amount
        rows_out.append(
            {
                "description": _str(row.get("description")),
                "quantity": normalize_number(row.get("quantity")),
                "unit_of_measure": _str(row.get("unit_of_measure")),
                "unit_price": normalize_number(row.get("unit_price")),
                "net_amount": net_amount,
                "vat_rate": normalize_number(row.get("vat_rate")),
                "vat_amount": normalize_number(row.get("vat_amount")),
                "gross_amount": gross_amount,
                "currency": currency,
            }
        )
    return {"rows": rows_out}


def validate_structure(meta: Dict[str, Any], items: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(meta, dict):
        errors.append("meta must be an object")
        return errors
    if not isinstance(items, dict):
        errors.append("items must be an object")
        return errors
    rows = items.get("rows", [])
    if not isinstance(rows, list):
        errors.append("items.rows must be an array")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"items.rows[{idx}] must be an object")
            continue
        if not row.get("description"):
            errors.append(f"items.rows[{idx}].description is missing")
    return errors


def _str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
