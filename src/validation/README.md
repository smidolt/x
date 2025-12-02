# Validation (rules + LLM stub)

CLI:
```bash
python -m src.validation.main \
  --meta-json output/parse_single/meta.json \
  --items-json output/parse_single/items.json \
  --seller-name "Example Seller d.o.o." \
  --seller-tax-id "SI12345678" \
  --output output/parse_single/validation.json
```

Services (Python):
```python
from src.validation.services import run_rules, run_validation
rules = run_rules({"meta": {...}, "items": {...}, "seller_name": "...", "seller_tax_id": "..."})
combined = run_validation({"meta": {...}, "items": {...}, "seller_name": "...", "seller_tax_id": "..."})
```

Sample output snippet:
```
{"rules": {"is_valid": true, "errors": [], "warnings": []}, "llm": {...}}
```
