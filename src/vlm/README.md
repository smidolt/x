# VLM (blocks + Qwen2-VL reasoner)

Blocks (heuristic) CLI:
```bash
python -m src.vlm.main blocks --ocr-json output/ocr_single/google__normalize_to_a4.ocr.json --output output/vlm_blocks/google.blocks.json
```

Reasoner CLI:
```bash
python -m src.vlm.main reasoner \
  --image output/check_steps_single/google__normalize_to_a4.png \
  --model-name Qwen/Qwen2-VL-2B-Instruct \
  --device auto --max-tokens 256 --temperature 0.1
```

Services (Python):
```python
from src.vlm.services import run_blocks, run_reasoner
blocks = run_blocks({"ocr_json_path": "...", "output_path": "output/vlm_blocks/page.blocks.json"})
res = run_reasoner({"image_path": "...", "params": {"model_name": "...", "device": "auto"}})
```

Sample blocks output snippet:
```
{"backend": "heuristic", "blocks": [{"kind": "header", "bbox": [...]}, ...]}
```
