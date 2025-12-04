# Validation

Rule-based checks (+ LLM stub) on meta/items JSON: required fields, math sums, seller identity, VAT exemption consistency.

Run from repo root:
```bash
PYTHONPATH=. python -m src.validation.main \
  --meta-json output/parse_single/meta.json \
  --items-json output/parse_single/items.json \
  --seller-name "Example Seller d.o.o." \
  --seller-tax-id "SI12345678" \
  --output output/parse_single/validation.json
```

Smoke test: `tests/validation_runner.py`. Output JSON: rules {is_valid, errors, warnings} and llm stub block.
