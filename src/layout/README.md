# Layout (LayoutLM or stub)

CLI (stub by default; add `--enabled` to load real model):
```bash
python -m src.layout.main --ocr-json output/ocr_single/google__normalize_to_a4.ocr.json --output output/layout/google.layout.json
```

Service (Python):
```python
from src.layout.services import run_layout
res = run_layout({
  "ocr_json_path": "output/ocr_single/google__normalize_to_a4.ocr.json",
  "params": {"enabled": False}  # set True if model is available
})
```

Sample output snippet (stub):
```
{"engine": "stub", "tokens": [...], "model_name": null}
```
