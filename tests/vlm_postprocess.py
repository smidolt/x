"""Simple checks for VLM postprocess/validation."""
from __future__ import annotations

from pprint import pprint

from src.vlm.postprocess import normalize_payload, parse_raw_response
from src.vlm.validation import VLMValidator


def run_demo() -> None:
    raw_response = """
    Here is the JSON:
    ```json
    {
      "meta": {
        "seller": "Demo Seller d.o.o.",
        "sellerAddress": "Street 1, City",
        "sellerTaxId": "SI123",
        "buyer": "Demo Buyer",
        "buyerAddress": "Buyer St 2",
        "invoice_number": "INV-1",
        "issue_date": "2024-12-01",
        "supply_date": "2024/12/01",
        "currency": "EUR",
        "total_net": "100",
        "total_vat": "22",
        "total_gross": "122"
      },
      "items": [
        {"description": "Service", "quantity": 1, "unit_price": 100, "vat_rate": 22, "vat_amount": 22, "gross_amount": 122, "currency": "EUR"}
      ]
    }
    ```
    """
    parsed, err = parse_raw_response(raw_response)
    assert err is None
    normalized, schema_errors = normalize_payload(parsed, currency_hint=None)
    print("Normalized:")
    pprint(normalized)
    validator = VLMValidator(
        required_fields=[
            "seller_name",
            "seller_address",
            "buyer_name",
            "buyer_address",
            "seller_tax_id",
            "invoice_number",
            "issue_date",
            "supply_date",
            "currency",
            "total_net",
            "total_vat",
            "total_gross",
        ],
        seller_name="Demo Seller d.o.o.",
        seller_tax_id="SI123",
        amount_tolerance=0.5,
    )
    result = validator.run(normalized["meta"], normalized["items"])
    print("Schema errors:", schema_errors)
    print("Validation errors:", result.errors)
    print("Validation warnings:", result.warnings)


if __name__ == "__main__":  # pragma: no cover
    run_demo()
