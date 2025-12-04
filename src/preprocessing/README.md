# Preprocessing

Per-step image cleanup: grayscale, crop, deskew, resize, normalize_to_a4, denoise, contrast, adaptive binarization. Each step can be called independently; CLI runs them all on the same source.

Run from repo root:
```bash
PYTHONPATH=. python tests/preprocessing.py --image input/google.jpg --output output/check_steps_single --target-dpi 300
```

Smoke test: `tests/preprocessing.py`. Outputs are individual PNGs per step in the chosen `--output` dir.
