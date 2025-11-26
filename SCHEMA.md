# eSlog-lite JSON Schema

This MVP schema captures the mandatory invoice requisites defined by the Slovenian VAT
law (ZDDV-1) and is aligned with the eSlog structure. Each processed invoice produces a
single JSON document:

```
{
  "meta": { ... },
  "items": { ... },
  "validation": { ... },
  "preprocessing": { ... },
  "artifacts": { ... }
}
```

## meta

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `seller_name` | string | yes | Legal name of the supplier. Must match the configured client company for internal validation. |
| `seller_address` | string | yes | Full address (street, postal code, city, country). |
| `seller_tax_id` | string | yes | DDV or VAT identification number of the seller (e.g., `SI12345678`). |
| `seller_tax_id_type` | string | no | For differentiation (`DDV`, `VAT`, `TIN`, ...). |
| `buyer_name` | string | yes | Legal name of the customer. |
| `buyer_address` | string | yes | Customer billing address. |
| `buyer_tax_id` | string | conditional | Required if the buyer is registered for VAT; empty otherwise. |
| `invoice_number` | string | yes | Unique sequential invoice ID. |
| `issue_date` | string (ISO date) | yes | Date of issuance. |
| `supply_date` | string (ISO date) | conditional | Required when different from `issue_date`. |
| `payment_date` | string (ISO date) | optional | Payment due date or actual payment date if available. |
| `currency` | string (ISO 4217) | yes | Currency code (EUR, USD, ...). |
| `total_net` | number | yes | Total taxable amount without VAT. |
| `total_vat` | number | yes | Total VAT amount. Must be `≈ 0` if VAT-exempt. |
| `total_gross` | number | yes | Total amount payable (`total_net + total_vat`). |
| `vat_rate` | number | optional | Dominant VAT rate in percent. |
| `vat_exemption_reason` | string | conditional | Required whenever VAT is not charged; should quote the legal article (e.g., "Neobdavčeno po 1. točki 42. člena ZDDV-1" / "Exempt under Art. 42(1) ZDDV-1"). |
| `payment_method` | string | optional | e.g., `credit card`, `wire transfer`. |
| `notes` | array of strings | optional | Additional textual remarks detected on the document. |
| `source_language` | string | optional | Detected language of the invoice (sl, en, ...). |

## items

```
"items": {
  "rows": [ItemRow]
}

ItemRow = {
  "description": string,
  "quantity": number | null,
  "unit_of_measure": string | null,
  "unit_price": number | null,
  "net_amount": number | null,
  "vat_rate": number | null,
  "vat_amount": number | null,
  "gross_amount": number | null,
  "currency": string | null
}
```

At least one row is required whenever the invoice contains a table of goods/services.

## validation

```
"validation": {
  "rules": {
    "is_valid": bool,
    "errors": [ValidationIssue],
    "warnings": [ValidationIssue]
  },
  "llm": {
    "backend": string,
    "notes": string,
    "issues": [ValidationIssue]
  }
}

ValidationIssue = {
  "code": string,
  "message": string,
  "field": string | null
}
```

Rule-based validation must cover:

1. Presence of mandatory fields listed above.
2. Mathematical checks: `sum(line.net_amount) ≈ total_net`, `sum(line.vat_amount) ≈ total_vat`, `total_net + total_vat ≈ total_gross`.
3. Seller identity match against configuration.
4. VAT exemptions: when `vat_exemption_reason` exists, enforce `total_vat ≈ 0`.
5. Optional cross-checks such as buyer VAT ID format or invoice numbering consistency.

LLM validation is a pluggable interface (stub by default) that can provide free-form
reasoning about inconsistencies; the interface preserves space for future backends.

## preprocessing / artifacts sections

Already included in the JSON payload for traceability:

| Field | Type | Description |
| --- | --- | --- |
| `preprocessing.steps` | array | Ordered list of preprocessing steps applied. |
| `preprocessing.warnings` | array | Any warnings (missing deps, skipped operations). |
| `preprocessing.elapsed_seconds` | number | Time spent in preprocessing. |
| `artifacts.preprocessed_path` | string | Path (relative) to the normalized image fed into OCR. |
| `artifacts.ocr_json` | string | Path to the raw OCR result used by downstream modules. |

## Example (simplified)

```
{
  "meta": {
    "seller_name": "Semrush Inc.",
    "seller_address": "7 Neshaminy Interplex, Ste 301, Trevose, PA 19053-6980, USA",
    "seller_tax_id": "US 90-0897948",
    "buyer_name": "Samethinkers Inc.",
    "buyer_address": "211 East 7th Street, Suite 620, Austin, TX 78701, USA",
    "invoice_number": "42383818908",
    "issue_date": "2020-12-03",
    "currency": "USD",
    "total_net": 99.95,
    "total_vat": 8.25,
    "total_gross": 108.20,
    "vat_rate": 8.25,
    "payment_method": "credit card"
  },
  "items": {
    "rows": [
      {
        "description": "Pro 1 month recurring",
        "quantity": 1,
        "unit_price": 99.95,
        "net_amount": 99.95,
        "vat_rate": 8.25,
        "vat_amount": 8.25,
        "gross_amount": 108.20,
        "currency": "USD"
      }
    ]
  },
  "validation": {
    "rules": {
      "is_valid": true,
      "errors": [],
      "warnings": []
    },
    "llm": {
      "backend": "stub",
      "notes": "LLM validation disabled.",
      "issues": []
    }
  },
  "preprocessing": {
    "steps": ["grayscale", "crop_to_content", "deskew", "normalize_to_a4", "denoise", "contrast"],
    "warnings": [],
    "elapsed_seconds": 0.70
  },
  "artifacts": {
    "preprocessed_path": "output/preprocessed/x_preprocessed.png",
    "ocr_json": "output/ocr/x_preprocessed.ocr.json"
  }
}
```

This document acts as the contract between every stage of the pipeline and external
consumers (validation dashboards, downstream ERP integrations, etc.). Future steps
(tasks 6–10) will populate the schema automatically.
