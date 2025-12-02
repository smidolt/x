"""Standalone Layout runner (LayoutLM or stub) with JSON-friendly output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.layout import LayoutAnalyzer


def to_payload(annotations) -> dict:
    return {
        "tokens": annotations.tokens,
        "embeddings": annotations.embeddings,
        "model_name": annotations.model_name,
        "engine": annotations.engine,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run layout analyzer on OCR JSON.")
    parser.add_argument("--ocr-json", type=Path, required=True, help="Path to OCR JSON with words/bboxes.")
    parser.add_argument("--model-name", type=str, default="microsoft/layoutlmv3-base", help="Layout model name.")
    parser.add_argument("--enabled", action="store_true", help="Enable real model (otherwise stub).")
    parser.add_argument("--offline", action="store_true", help="Use offline HF cache.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    analyzer = LayoutAnalyzer(
        model_name=args.model_name,
        enabled=args.enabled,
        offline=args.offline,
    )
    annotations = analyzer.run(args.ocr_json)
    payload = to_payload(annotations)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote layout annotations to {args.output} (engine={payload.get('engine')})")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
