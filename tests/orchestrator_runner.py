"""Smoke test for the classic orchestrator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import orchestrator  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classic orchestrator for smoke test.")
    parser.add_argument("--input", type=Path, default=Path("input/google.jpg"))
    parser.add_argument("--output", type=Path, default=Path("output/orchestrated_test"))
    parser.add_argument("--seller-name", type=str, default="Example Seller d.o.o.")
    parser.add_argument("--seller-tax-id", type=str, default="SI12345678")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    argv = [
        "orchestrator",
        "--input",
        str(args.input),
        "--output",
        str(args.output),
        "--seller-name",
        args.seller_name,
        "--seller-tax-id",
        args.seller_tax_id,
    ]
    if args.verbose:
        argv.append("--verbose")
    sys.argv = argv
    orchestrator.main()


if __name__ == "__main__":  # pragma: no cover
    main()
