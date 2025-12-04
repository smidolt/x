# OCR / VLM Invoice Pipeline

Modular pipeline for invoices: preprocessing, OCR (classic path), layout (stub/model), parsing, validation, plus a VLM-only path. Classic runs Tesseract + layout + parsing/validation. VLM runs Qwen2-VL directly on the image (OCR/blocks optional/disabled by default in the VLM orchestrator).

## Project layout
```
src/
  preprocessing/   # resize/crop/deskew/denoise
  ocr/             # OCR (Tesseract)
  layout/          # LayoutLM wrapper (stub or HF model)
  parsing/         # meta/items extraction
  validation/      # rule-based + LLM stub
  vlm/             # Qwen2-VL reasoner + optional heuristic blocks
  orchestrator.py        # classic pipeline runner
  orchestrator_vlm.py    # VLM-only runner (OCR disabled)
config.yaml        # runtime configuration (classic path)
tests/             # smoke runners for each module
```

## Environment setup
```bash
python3 -m venv myenv1
source myenv1/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -r requirements-vlm.txt
```

## Models
- **LayoutLM (classic)**: `microsoft/layoutlmv3-base` (stub if not available).
- **OCR**: Tesseract (system binary).
- **VLM**: Qwen2-VL (2B or 7B). Reasoner uses `qwen_vl_utils` + HF transformers.
- **LLM validation**: stub by default; optional tiny HF classifier (see `validation/llm`).

## Running (classic)
```bash
python -m src.orchestrator \
  --input input/google.jpg \
  --output output/orchestrated \
  --seller-name "Example Seller d.o.o." \
  --seller-tax-id "SI12345678" \
  --layout-enabled \
  --verbose
```
Outputs:
- `output/orchestrated/json/<name>.json`
- `output/orchestrated/summary_orchestrator.json`
- artifacts under `output/orchestrated/preprocessed/*`, `ocr/*`, `layout/*`

## Running (VLM-only, no OCR/blocks)
```bash
python -m src.orchestrator_vlm \
  --input input/google.jpg \
  --output output_vlm \
  --vlm-model-reasoner Qwen/Qwen2-VL-7B-Instruct \
  --vlm-max-tokens 256 \
  --vlm-temperature 0.2
```
Output: `output_vlm/json/<name>.vlm.json` and `summary_vlm_orchestrator.json`.

## Smoke tests
- Preproc: `PYTHONPATH=. python tests/preprocessing.py --image input/google.jpg --output output/check_steps_single --target-dpi 300`
- OCR: `PYTHONPATH=. python tests/ocr_runner.py ...`
- Layout: `PYTHONPATH=. python tests/layout_runner.py ...`
- Parsing: `PYTHONPATH=. python tests/parsing_runner.py ...`
- Validation: `PYTHONPATH=. python tests/validation_runner.py ...`
- VLM: `PYTHONPATH=. python tests/vlm_runner.py --image input/google.jpg ...`

## Notes
- Classic path uses `config.yaml` toggles (layout/llm/etc.).
- VLM path bypasses OCR/blocks by default to reduce GPU load; switch model name if you want 7B and have enough memory.
