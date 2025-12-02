# OCR (Tesseract service)

CLI:
```bash
python -m src.ocr.main --image output/check_steps_single/google__normalize_to_a4.png --output output/ocr_single --languages eng --psm 6 --oem 3
```

Service call (Python):
```python
from src.ocr.service_tesseract import run as run_tesseract
res = run_tesseract({
  "image_path": "output/check_steps_single/google__normalize_to_a4.png",
  "params": {"languages": "eng", "page_segmentation_mode": 6, "oem": 3},
  "output_dir": "output/ocr_single"
})
print(res["json_path"], res["word_count"])
```

Sample output snippet:
```
OCR done. Engine=tesseract words=XXX json=output/ocr_single/google__normalize_to_a4.ocr.json
```
