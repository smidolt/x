# Parsing (classic meta + items)

CLI:
```bash
python -m src.parsing.main \
  --ocr-json output/ocr_single/google__normalize_to_a4.ocr.json \
  --seller-name "Example Seller d.o.o." \
  --seller-tax-id "SI12345678" \
  --output output/parse_single/google.json
```

Services (Python):
```python
from src.parsing.services import run_meta, run_items, run_classic
meta = run_meta({"ocr_json_path": "...", "seller_name": "...", "seller_tax_id": "..."})
items = run_items({"ocr_json_path": "...", "currency_hint": meta.get("currency")})
classic = run_classic({"ocr_json_path": "...", "seller_name": "...", "seller_tax_id": "..."})
```

Sample output snippet (classic):
```
{
  "meta": {"seller_name": "...", "invoice_number": "...", ...},
  "items": {"rows": [...]},
  "validation": {"rules": {"is_valid": true, "errors": [], "warnings": []}, "llm": {...}}
}
```
