"""Smoke test for VLM validation (schema + math)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.validation.vlm.services import run_vlm_validation


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM validation on a VLM JSON file.")
    parser.add_argument("--input", type=Path, required=True, help="Path to VLM JSON (with meta/items).")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path.")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Tolerance for math checks.")
    args = parser.parse_args()

    data = _load(args.input)
    res = run_vlm_validation({"data": data, "tolerance": args.tolerance})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Wrote VLM validation to {args.output}")
    else:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
