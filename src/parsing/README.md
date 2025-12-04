# Parsing (classic)

Parses meta + items from OCR (optionally layout), then runs rule/LLM stub validation.

Run from repo root:
```bash
PYTHONPATH=. python -m src.parsing.main \
  --ocr-json output/ocr_single/google__normalize_to_a4.ocr.json \
  --seller-name "Example Seller d.o.o." \
  --seller-tax-id "SI12345678" \
  --layout-json output/layout/google.layout.json \
  --output output/parse_single/google.json
```

Smoke test: `tests/parsing_runner.py` (also writes meta.json/items.json). Output includes meta/items and validation blocks.
