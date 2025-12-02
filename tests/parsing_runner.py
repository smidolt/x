"""Quick parsing smoke test (classic meta+items+validation stub)."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from src.parsing.services import run_classic


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classic parsing on OCR JSON.")
    parser.add_argument("--ocr-json", type=Path, required=True, help="Path to OCR JSON.")
    parser.add_argument("--seller-name", type=str, required=True, help="Seller name.")
    parser.add_argument("--seller-tax-id", type=str, required=True, help="Seller tax ID.")
    parser.add_argument("--layout-json", type=Path, help="Optional layout JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path.")
    args = parser.parse_args()

    res = run_classic(
        {
            "ocr_json_path": str(args.ocr_json),
            "seller_name": args.seller_name,
            "seller_tax_id": args.seller_tax_id,
            "layout_json_path": str(args.layout_json) if args.layout_json else None,
        }
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(res, indent=2), encoding="utf-8")
        # Also emit meta/items separately for validation runner convenience
        meta_path = args.output.parent / "meta.json"
        items_path = args.output.parent / "items.json"
        meta_path.write_text(json.dumps(res.get("meta", {}), indent=2), encoding="utf-8")
        items_path.write_text(json.dumps(res.get("items", {}), indent=2), encoding="utf-8")
        print(f"Saved parsing result to {args.output}")
        print(f"Saved meta to {meta_path}")
        print(f"Saved items to {items_path}")
    else:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
