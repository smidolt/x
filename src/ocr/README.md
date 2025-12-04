# OCR

Tesseract-based OCR producing `<name>.ocr.json` with tokens/bboxes.

Run from repo root:
```bash
PYTHONPATH=. python -m src.ocr.main --image output/check_steps_single/google__normalize_to_a4.png --output output/ocr_single --languages eng --psm 6 --oem 3
```

Smoke test: `tests/ocr_runner.py`. Output JSON path is printed; includes engine, word_count, and words array.
