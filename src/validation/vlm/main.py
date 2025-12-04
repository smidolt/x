"""VLM JSON validation (schema + math) as a standalone module."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .checks import check_required, check_math, check_vat_exemption


def _load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def validate_vlm(meta: Dict[str, Any], items: Dict[str, Any], tol: float = 0.5) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    errors.extend(check_required(meta, items))
    errors.extend(check_math(meta, items, tol=tol))
    warnings.extend(check_vat_exemption(meta))
    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VLM JSON (schema + math).")
    parser.add_argument("--input", type=Path, required=True, help="Path to VLM JSON (with meta/items).")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for validation result.")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Tolerance for math checks.")
    args = parser.parse_args()

    data = _load(args.input)
    meta = data.get("meta") or {}
    items = data.get("items") or data.get("items", {})
    result = validate_vlm(meta, items, tol=args.tolerance)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote validation to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
