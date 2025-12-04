"""Smoke test for eval: compare predicted JSON vs gold JSON on selected fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.main import eval_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval predicted vs gold JSON.")
    parser.add_argument("--pred", type=Path, required=True, help="Predicted JSON.")
    parser.add_argument("--gold", type=Path, required=True, help="Gold JSON.")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["seller_name", "buyer_name", "invoice_number", "currency", "total_net", "total_vat", "total_gross"],
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    scores = eval_pair(args.pred, args.gold, args.fields)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        print(f"Wrote scores to {args.output}")
    else:
        print(json.dumps(scores, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
