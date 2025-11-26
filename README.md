# OCR Invoice Pipeline

Batch pipeline that normalizes invoice scans, runs OCR (Tesseract), infers document
structure with LayoutLM, extracts meta fields plus line items, and validates the result
against an eSlog-lite JSON contract.

## Project layout
```
src/
  preprocessing/  # resize/crop/deskew/denoise utilities
  ocr/            # OCR engine drivers
  layout/         # LayoutLM wiring (HuggingFace or stub)
  parsing/meta/   # header/meta extraction
  parsing/items/  # table & line-item parsing
  validation/     # rule-based checks + LLM interface
cli.py            # batch entry point
config.yaml       # runtime configuration
```

## Environment setup
```bash
python3 -m venv myenv1
source myenv1/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Models
1. **LayoutLM (offline cache recommended)**
   ```bash
   hf download microsoft/layoutlmv3-base --cache-dir ~/.cache/huggingface
   export HF_HUB_OFFLINE=1  # optional: force offline mode after the download
   ```
2. **LLM validation backend (optional)**
   The default backend is `stub`. To run a tiny local transformer add to `config.yaml`:
   ```yaml
   llm:
     enabled: true
     backend: local_transformer
     model_name: sshleifer/tiny-distilbert-base-cased-distilled-sst-2
   ```
   Any HuggingFace text-classification checkpoint will work.

## Config highlights
- `input.path` — folder with images/PDFs to process (default `data/input`).
- `company` — expected seller identity for validation.
- `features.layoutlm_enabled`, `llm.enabled` — global toggles.
- `preprocessing`, `ocr`, `layoutlm`, `llm` sections — fine-grained knobs for each stage.

## Running the pipeline
```bash
myenv1/bin/python -m src.cli --config config.yaml --verbose
```
The CLI scans `input.path` (e.g., `x.jpg`, `google.jpg`) and produces:
- `output/json/<name>.json` — full eSlog-lite payload.
- `output/summary.csv` — per-file summary (`is_valid`, totals, error counts, etc.).
- `output/preprocessed/*`, `output/ocr/*` — intermediate artifacts for debugging.

## Verifying results
1. Inspect `output/json/<name>.json` to review `meta`, `items`, and `validation` blocks.
2. Any rule failures are listed in `validation.rules.errors` (missing fields, math issues,
seller mismatch, VAT exemptions, …).
3. Compare against the original image in `data/input` to confirm seller/buyer details,
dates, invoice number, amounts (net/VAT/gross), and extracted line items.

## Adding invoices
Drop new files or folders under `data/input` and rerun the CLI. Every file is processed
recursively; results land in `output/json` and the summary CSV.

## Troubleshooting
- **LayoutLM fails to load** — ensure the checkpoint is cached (`hf download`) or unset
  `HF_HUB_OFFLINE` if network access is required. The pipeline automatically falls back
  to the stub backend when the model cannot be loaded.
- **Tesseract missing** — install it system-wide (macOS: `brew install tesseract`).
- **Need more logs** — keep `--verbose` enabled or toggle individual steps in
  `config.yaml`.
