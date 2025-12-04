# VLM

Qwen2-VL reasoner (official HF/qwen_vl_utils flow) + optional heuristic blocks. Default VLM orchestrator uses preprocess → reasoner (OCR disabled).

Orchestrator run from repo root (OCR disabled):
```bash
PYTHONPATH=. python -m src.orchestrator_vlm \
  --input input/google.jpg \
  --output output_vlm \
  --vlm-model-reasoner Qwen/Qwen2-VL-2B-Instruct \
  --vlm-max-tokens 256 \
  --vlm-temperature 0.2
```

Smoke test: `tests/vlm_runner.py` (reasoner only, optional blocks if OCR JSON provided). Output includes raw_response, parsed JSON (if valid), and timing.
