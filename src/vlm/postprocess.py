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
    meta_raw = parsed.get("meta") or parsed
    items_raw = parsed.get("items") or parsed.get("lines") or []
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
