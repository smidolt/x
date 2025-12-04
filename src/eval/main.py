"""Simple eval script comparing predicted vs gold JSON fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _compare_field(pred: Dict[str, Any], gold: Dict[str, Any], key: str) -> bool:
    return pred.get(key) == gold.get(key)


def eval_pair(pred_path: Path, gold_path: Path, fields: List[str]) -> Dict[str, Any]:
    pred = _load(pred_path)
    gold = _load(gold_path)
    meta_pred = pred.get("meta", pred)
    meta_gold = gold.get("meta", gold)
    scores = {}
    for f in fields:
        scores[f] = 1.0 if _compare_field(meta_pred, meta_gold, f) else 0.0
    scores["avg"] = sum(scores.values()) / len(fields) if fields else 0.0
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted JSON vs gold JSON on selected fields.")
    parser.add_argument("--pred", type=Path, required=True, help="Predicted JSON path.")
    parser.add_argument("--gold", type=Path, required=True, help="Gold JSON path.")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["seller_name", "buyer_name", "invoice_number", "currency", "total_net", "total_vat", "total_gross"],
        help="Fields to compare in meta block.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for scores.")
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
