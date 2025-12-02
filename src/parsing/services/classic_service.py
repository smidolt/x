"""Combined classic parsing (meta + items + rule validation) service."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json

from src.parsing.meta import MetaParser
from src.parsing.items import ItemsParser
from src.validation.rules import RuleValidator
from src.validation.llm import build_llm_backend


def run(payload: Dict[str, object]) -> Dict[str, object]:
    ocr_json_path_raw = payload.get("ocr_json_path")
    if not ocr_json_path_raw:
        raise ValueError("payload must include 'ocr_json_path'")
    seller_name = str(payload.get("seller_name", ""))
    seller_tax_id = str(payload.get("seller_tax_id", ""))
    layout_json_path_raw = payload.get("layout_json_path")
    required_fields = payload.get("required_fields")
    amount_tolerance = float(payload.get("amount_tolerance", 0.5))

    layout_data = {}
    if layout_json_path_raw:
        try:
            layout_data = json.loads(Path(str(layout_json_path_raw)).read_text(encoding="utf-8"))
        except Exception:
            layout_data = {}

    required: List[str] = (
        list(required_fields)
        if isinstance(required_fields, list)
        else [
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
    )

    meta_parser = MetaParser(seller_name, seller_tax_id)
    items_parser = ItemsParser()
    rule_validator = RuleValidator(
        required_fields=required,
        seller_name=seller_name,
        seller_tax_id=seller_tax_id,
        amount_tolerance=amount_tolerance,
    )
    llm_backend = build_llm_backend(None)  # stub

    meta = meta_parser.run(Path(str(ocr_json_path_raw)), layout_data)
    items = items_parser.run(Path(str(ocr_json_path_raw)), layout_data, currency_hint=meta.raw.get("currency"))
    items_dict = items.__dict__ if hasattr(items, "__dict__") else {"rows": getattr(items, "rows", [])}
    validation = rule_validator.run(meta.raw, items_dict)
    llm_validation = llm_backend.validate(meta.raw, items_dict)

    return {
        "meta": meta.raw,
        "items": items_dict,
        "validation": {
            "rules": {
                "errors": validation.errors,
                "warnings": validation.warnings,
                "is_valid": validation.is_valid,
            },
            "llm": llm_validation,
        },
    }
