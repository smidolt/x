"""Post-processing for VLM responses: clean text, parse JSON, normalize, and hydrate missing totals."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from src.vlm.schema import normalize_currency, normalize_items, normalize_meta, validate_structure

LOGGER = logging.getLogger(__name__)


def strip_to_json(text: str) -> Tuple[str | None, str | None]:
    """Extract JSON substring from raw text, return (json_text, error)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip("`\n ")
    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        cleaned = cleaned[start:end]
    try:
        json.loads(cleaned)
        return cleaned, None
    except Exception as exc:
        return None, f"Failed to parse JSON substring: {exc}"


def parse_raw_response(raw: str) -> Tuple[Dict[str, Any] | None, str | None]:
    json_text, err = strip_to_json(raw)
    if err:
        return None, err
    try:
        parsed = json.loads(json_text or "{}")
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            return None, "Parsed JSON is not an object"
        return parsed, None
    except Exception as exc:
        return None, f"Failed to load JSON: {exc}"


def normalize_payload(parsed: Dict[str, Any], currency_hint: str | None = None) -> Tuple[Dict[str, Any], List[str]]:
    mapped = _map_common_fields(parsed)
    meta_raw = mapped.get("meta") or parsed.get("meta") or parsed
    items_raw = mapped.get("items") or parsed.get("items") or parsed.get("lines") or []
    if isinstance(items_raw, dict) and "rows" in items_raw:
        items_raw = items_raw.get("rows", [])

    meta = normalize_meta(meta_raw, currency_hint=currency_hint)
    items = normalize_items(items_raw, currency_hint=meta.get("currency") or currency_hint)

    schema_errors = validate_structure(meta, items)
    meta, items = _hydrate_totals(meta, items)
    return {"meta": meta, "items": items}, schema_errors


def _hydrate_totals(meta: Dict[str, Any], items: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rows = items.get("rows", []) if isinstance(items, dict) else []
    if rows:
        net_sum = 0.0
        gross_sum = 0.0
        vat_sum = 0.0
        for row in rows:
            if not isinstance(row, dict):
                continue
            net = row.get("net_amount")
            gross = row.get("gross_amount")
            amt = row.get("amount") if "amount" in row else None
            vat = row.get("vat_amount")
            if net is None and amt is not None:
                net = amt
            if gross is None and amt is not None:
                gross = amt
            net_sum += net or 0
            gross_sum += gross or 0
            vat_sum += vat or 0

        if meta.get("total_net") is None or meta.get("total_net") == 0:
            meta["total_net"] = net_sum
        if (meta.get("total_vat") is None or meta.get("total_vat") == 0) and vat_sum:
            meta["total_vat"] = vat_sum
        if meta.get("total_gross") is None or meta.get("total_gross") == 0:
            meta["total_gross"] = gross_sum or ((meta.get("total_net") or 0) + (meta.get("total_vat") or 0))
        if meta.get("currency") is None:
            currencies = {row.get("currency") for row in rows if row.get("currency")}
            if len(currencies) == 1:
                meta["currency"] = currencies.pop()
    return meta, items


# -----------------------------------------------------------
# Mapping helpers: adapt common nested invoice shapes into meta/items


def _map_common_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    mapped: Dict[str, Any] = {"meta": {}, "items": []}
    currency_hint = None

    invoice = parsed.get("invoice", {}) if isinstance(parsed.get("invoice"), dict) else {}
    company = parsed.get("company", {}) if isinstance(parsed.get("company"), dict) else {}
    billing_address = parsed.get("billing_address", {}) if isinstance(parsed.get("billing_address"), dict) else {}
    summary = parsed.get("summary", {}) if isinstance(parsed.get("summary"), dict) else {}
    taxes = parsed.get("taxes", {}) if isinstance(parsed.get("taxes"), dict) else {}
    google_ws = parsed.get("googleWorkspace", {}) if isinstance(parsed.get("googleWorkspace"), dict) else {}
    details = parsed.get("details", {}) if isinstance(parsed.get("details"), dict) else {}

    meta = {}
    # Invoice basics
    meta["invoice_number"] = invoice.get("number") or invoice.get("id")
    meta["issue_date"] = invoice.get("date") or invoice.get("issue_date")
    meta["supply_date"] = invoice.get("supply_date")
    # Totals from invoice/summary/google_ws/details
    totals_candidates = [
        invoice.get("total"),
        invoice.get("total_amount_due"),
        summary.get("total"),
        summary.get("total_amount_due"),
        google_ws.get("total"),
        google_ws.get("total_in_usd"),
    ]
    total_gross, currency_from_gross = _parse_first_amount(totals_candidates)
    meta["total_gross"] = total_gross
    currency_hint = currency_hint or currency_from_gross

    subtotal, currency_from_sub = _parse_first_amount(
        [
            summary.get("subtotal"),
            summary.get("subtotal_in_usd"),
            google_ws.get("subtotal"),
            google_ws.get("subtotal_in_usd"),
        ]
    )
    meta["total_net"] = subtotal
    currency_hint = currency_hint or currency_from_sub

    vat_total, currency_from_tax = _parse_first_amount(
        [
            taxes.get("sales_tax"),
            summary.get("stateSalesTax"),
            summary.get("localSalesTax"),
            summary.get("state_sales_tax"),
            summary.get("local_sales_tax"),
        ]
    )
    meta["total_vat"] = vat_total
    currency_hint = currency_hint or currency_from_tax

    # Seller info
    if company:
        meta["seller_name"] = company.get("name")
        meta["seller_tax_id"] = company.get("tax_id")
        address_parts = [
            company.get("address"),
            company.get("city"),
            company.get("state"),
            company.get("zip"),
            company.get("country"),
        ]
        meta["seller_address"] = ", ".join([str(p) for p in address_parts if p])

    # Buyer info
    if billing_address:
        meta["buyer_name"] = billing_address.get("name") or billing_address.get("company")
        address_parts = [
            billing_address.get("street"),
            billing_address.get("address"),
            billing_address.get("city"),
            billing_address.get("state"),
            billing_address.get("zip"),
            billing_address.get("country"),
        ]
        meta["buyer_address"] = ", ".join([str(p) for p in address_parts if p])

    # Items: products list
    items_rows: List[Dict[str, Any]] = []
    products = parsed.get("products") or parsed.get("items") or []
    if isinstance(products, list):
        for row in products:
            if not isinstance(row, dict):
                continue
            unit_price, cur_from_unit = _parse_first_amount([row.get("unit_price")])
            total_price, cur_from_total = _parse_first_amount([row.get("total_price"), row.get("amount")])
            currency_hint = currency_hint or cur_from_total or cur_from_unit
            items_rows.append(
                {
                    "description": row.get("description") or row.get("product"),
                    "quantity": row.get("quantity"),
                    "unit_price": unit_price,
                    "net_amount": total_price,
                    "vat_rate": None,
                    "vat_amount": None,
                    "gross_amount": total_price,
                    "currency": cur_from_total or cur_from_unit,
                }
            )

    # Fallback: if summary has subtotal/taxes/total, synthesize a single item
    if not items_rows and subtotal is not None:
        items_rows.append(
            {
                "description": "Invoice total",
                "quantity": 1,
                "unit_price": subtotal,
                "net_amount": subtotal,
                "vat_rate": None,
                "vat_amount": vat_total,
                "gross_amount": total_gross or subtotal + (vat_total or 0),
                "currency": currency_hint,
            }
        )

    mapped["meta"] = meta
    mapped["items"] = items_rows
    if currency_hint:
        mapped.setdefault("meta", {})["currency"] = currency_hint
    return mapped


def _parse_first_amount(candidates: List[Any]) -> Tuple[float | None, str | None]:
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value), None
        if isinstance(value, str):
            text = value.strip()
            currency = None
            if "$" in text:
                currency = "USD"
            elif "€" in text:
                currency = "EUR"
            elif "£" in text:
                currency = "GBP"
            try:
                cleaned = text.replace("$", "").replace("€", "").replace("£", "").replace(",", "").strip()
                return float(cleaned), currency
            except ValueError:
                continue
    return None, None
