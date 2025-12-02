"""Quick validation smoke test (rules + LLM stub)."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from src.validation.services import run_validation


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation on meta/items JSON.")
    parser.add_argument("--meta-json", type=Path, required=True)
    parser.add_argument("--items-json", type=Path, required=True)
    parser.add_argument("--seller-name", type=str, required=True)
    parser.add_argument("--seller-tax-id", type=str, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    meta = _load(args.meta_json)
    items = _load(args.items_json)
    res = run_validation(
        {
            "meta": meta,
            "items": items,
            "seller_name": args.seller_name,
            "seller_tax_id": args.seller_tax_id,
        }
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Saved validation to {args.output}")
    else:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
