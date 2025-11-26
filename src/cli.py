"""Batch CLI entry point for the invoice parsing MVP."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import argparse
import csv
import json
import logging
import time

from .config import AppConfig, load_config
from .layout import LayoutAnalyzer
from .ocr import run_ocr
from .parsing.items import ItemsParser
from .parsing.meta import MetaParser
from .preprocessing import PreprocessingPipeline
from .validation.llm import build_llm_backend
from .validation.rules import RuleValidator

LOGGER = logging.getLogger(__name__)


def _iter_documents(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    for path in sorted(p for p in input_path.rglob("*") if p.is_file()):
        yield path


def _compose_output_paths(base_output: Path, document: Path) -> Dict[str, Path]:
    name = document.stem
    document_output = base_output / name
    return {
        "document": document_output,
        "preprocessed": base_output / "preprocessed",
        "ocr": base_output / "ocr",
        "json": base_output / "json" / f"{name}.json",
    }


def run_pipeline(config: AppConfig) -> None:
    config.output.base_path.mkdir(parents=True, exist_ok=True)
    config.output.json_dir.mkdir(parents=True, exist_ok=True)

    preprocessing = PreprocessingPipeline(
        config.output.base_path / "preprocessed",
        config.preprocessing,
    )
    ocr_output_dir = config.output.base_path / "ocr"
    layout_analyzer = LayoutAnalyzer(
        model_name=config.layoutlm.model_name,
        enabled=config.layoutlm.enabled and config.features.layoutlm_enabled,
        offline=config.layoutlm.offline,
    )
    meta_parser = MetaParser(config.company.name, config.company.tax_id)
    items_parser = ItemsParser()
    rule_validator = RuleValidator(
        required_fields=[
            "seller_name",
            "seller_address",
            "buyer_name",
            "buyer_address",
            "invoice_number",
            "issue_date",
            "supply_date",
            "payment_date",
            "total_net",
            "total_vat",
            "total_gross",
            "seller_tax_id",
        ],
        seller_name=config.company.name,
        seller_tax_id=config.company.tax_id,
        amount_tolerance=0.5,
    )
    llm_backend = build_llm_backend(config.llm if config.llm.enabled else None)

    summary_rows: List[Dict[str, str]] = []

    for document in _iter_documents(config.input.path):
        LOGGER.info("Processing document: %s", document)
        doc_start = time.time()
        outputs = _compose_output_paths(config.output.base_path, document)

        preprocessed = preprocessing.run(document)
        ocr_result = run_ocr(preprocessed.processed_path, ocr_output_dir, config.ocr)
        layout_annotations = layout_analyzer.run(ocr_result.json_path)
        layout_data = asdict(layout_annotations)
        meta = meta_parser.run(ocr_result.json_path, layout_data)
        items = items_parser.run(
            ocr_result.json_path,
            layout_data,
            currency_hint=meta.raw.get("currency"),
        )
        rule_validation = rule_validator.run(meta.raw, asdict(items))
        llm_validation = llm_backend.validate(meta.raw, asdict(items))

        document_payload = {
            "meta": meta.raw,
            "items": asdict(items),
            "validation": {
                "rules": {
                    "errors": rule_validation.errors,
                    "warnings": rule_validation.warnings,
                    "is_valid": rule_validation.is_valid,
                },
                "llm": llm_validation,
            },
            "preprocessing": {
                "steps": preprocessed.steps_applied,
                "warnings": preprocessed.warnings,
                "elapsed_seconds": preprocessed.elapsed_seconds,
            },
            "artifacts": {
                "preprocessed_path": str(preprocessed.processed_path),
                "ocr_json": str(ocr_result.json_path),
            },
        }
        outputs["json"].parent.mkdir(parents=True, exist_ok=True)
        outputs["json"].write_text(json.dumps(document_payload, indent=2), encoding="utf-8")

        summary_rows.append(
            {
                "file_name": document.name,
                "is_valid_invoice": str(rule_validation.is_valid),
                "errors_count": str(len(rule_validation.errors)),
                "warnings_count": str(len(rule_validation.warnings)),
                "total_gross": str(meta.raw.get("total_gross", "")),
                "seller_name": meta.raw.get("seller_name", ""),
                "invoice_number": meta.raw.get("invoice_number", ""),
                "elapsed_seconds": f"{time.time() - doc_start:.2f}",
            }
        )

    _write_summary_csv(config.output.summary_csv, summary_rows)


def _write_summary_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        LOGGER.warning("No documents processed; summary file will be empty")
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Summary CSV written to %s", path)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Invoice parsing MVP")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    LOGGER.info("Loading config from %s", args.config)
    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":  # pragma: no cover
    main()
