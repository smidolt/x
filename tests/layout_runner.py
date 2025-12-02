"""Quick Layout smoke test (stub by default; enable real model with --enabled)."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from src.layout.services import run_layout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Layout analyzer on OCR JSON.")
    parser.add_argument("--ocr-json", type=Path, required=True, help="Path to OCR JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path.")
    parser.add_argument("--enabled", action="store_true", help="Enable real model (otherwise stub).")
    parser.add_argument("--model-name", type=str, default="microsoft/layoutlmv3-base", help="Layout model name.")
    parser.add_argument("--offline", action="store_true", help="Use HF cache only.")
    args = parser.parse_args()

    res = run_layout(
        {
            "ocr_json_path": str(args.ocr_json),
            "params": {
                "enabled": args.enabled,
                "model_name": args.model_name,
                "offline": args.offline,
            },
        }
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Saved layout to {args.output}")
    else:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
