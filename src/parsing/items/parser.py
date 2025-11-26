"""Line items parser using OCR + layout information."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import json
import logging
import re

LOGGER = logging.getLogger(__name__)

AMOUNT_PATTERN = re.compile(r"(?P<currency>[$€£])?\s*(?P<value>\d+[\d,.]*)")
PERCENT_PATTERN = re.compile(r"([0-9]+\.?[0-9]*)%")
CURRENCY_SYMBOLS = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
}
SUMMARY_LINE_KEYWORDS = {
    "SUBTOTAL",
    "TOTAL AMOUNT DUE",
    "TOTAL DUE",
    "TOTAL:",
    "AMOUNT DUE",
    "CREDITS APPLIED",
    "PAYMENTS APPLIED",
    "THE CREDIT CARD",
    "INVOICE NOTES",
    "NOTE",
}


@dataclass(slots=True)
class ParsedItems:
    rows: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class TableLine:
    text: str
    words: List[Dict[str, Any]]
    line_num: int
    min_x: int
    max_x: int
    center_y: float


@dataclass(slots=True)
class TableRegion:
    lines: List[TableLine]
    header: TableLine | None
    top: float
    bottom: float


class ItemsParser:
    """Parse invoice items from OCR text."""

    def run(
        self,
        ocr_result_path: Path,
        layout_annotations: Dict[str, Any] | None = None,
        currency_hint: str | None = None,
    ) -> ParsedItems:
        LOGGER.info("Parsing table line items")
        payload = _load_ocr_payload(ocr_result_path)
        words = payload.get("words", [])
        lines = _build_lines(words)
        if not lines:
            LOGGER.warning("No OCR lines available for items parsing")
            return ParsedItems()

        region = detect_table_region(lines, layout_annotations or {})
        if not region:
            LOGGER.warning("Failed to detect table region")
            return ParsedItems()

        rows = _parse_region(region, currency_hint)
        return ParsedItems(rows=rows)


# ---------------------------------------------------------------------------
def detect_table_region(
    lines: List[TableLine], layout_annotations: Dict[str, Any]
) -> TableRegion | None:
    layout_tokens = layout_annotations.get("tokens") if layout_annotations else None
    if layout_tokens:
        region = _detect_via_layout(lines, layout_tokens)
        if region:
            return region
    return _detect_via_heuristics(lines)


def _detect_via_layout(
    lines: List[TableLine], tokens: Sequence[Dict[str, Any]]
) -> TableRegion | None:
    table_tokens = [t for t in tokens if t.get("label") in {"table_row", "table_header"}]
    if not table_tokens:
        return None
    top = min(t.get("bbox", [0, 0, 0, 0])[1] for t in table_tokens)
    bottom = max(t.get("bbox", [0, 0, 0, 0])[3] for t in table_tokens)
    region_lines = [
        line
        for line in lines
        if line.center_y >= top - 10 and line.center_y <= bottom + 10
    ]
    if not region_lines:
        return None
    return TableRegion(lines=region_lines, header=None, top=top, bottom=bottom)


def _detect_via_heuristics(lines: List[TableLine]) -> TableRegion | None:
    if not lines:
        return None
    header_idx = _find_line_with_keywords(lines, ["PRODUCT", "DESCRIPTION", "ITEM"])
    if header_idx is None:
        header_idx = _find_line_with_keywords(lines, ["AMOUNT"])
    amount_indices = [idx for idx, line in enumerate(lines) if _line_has_amount(line)]
    if header_idx is None:
        if not amount_indices:
            return None
        header_idx = max(0, amount_indices[0] - 1)

    total_idx = None
    for idx in range(header_idx + 1, len(lines)):
        upper = lines[idx].text.upper()
        if "TOTAL" in upper or "CHECKOUT" in upper:
            total_idx = idx
            break
    if total_idx is None:
        total_idx = len(lines)

    start_idx = header_idx + 1
    if header_idx is None and amount_indices:
        start_idx = amount_indices[0]
    region_lines = [line for line in lines[start_idx:total_idx] if line.text.strip()]
    if not region_lines:
        return None
    top = min(line.center_y for line in region_lines)
    bottom = max(line.center_y for line in region_lines)
    header_line = lines[header_idx] if header_idx < len(lines) else None
    return TableRegion(lines=region_lines, header=header_line, top=top, bottom=bottom)


def _parse_region(region: TableRegion, currency_hint: str | None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in region.lines:
        upper = line.text.upper()
        if _is_summary_line(upper):
            continue
        amount_tokens = _extract_amount_tokens(line)
        if not amount_tokens:
            continue
        raw_amounts = _extract_amounts_from_text(line.text)
        description_tokens = _description_tokens(line.words, amount_tokens)
        description = " ".join(description_tokens).strip()
        if not description:
            description = line.text.strip()
        currency = (amount_tokens[-1]["currency"] or amount_tokens[0]["currency"] or currency_hint)
        line_total = raw_amounts[-1] if raw_amounts else amount_tokens[-1]["value"]

        if "TAX" in upper and "TOTAL" not in upper:
            vat_rate = _extract_percentage(line.text)
            rows.append(
                {
                    "description": description,
                    "quantity": None,
                    "unit_of_measure": None,
                    "unit_price": None,
                    "net_amount": None,
                    "vat_rate": vat_rate,
                    "vat_amount": line_total,
                    "gross_amount": line_total,
                    "currency": currency,
                }
            )
            continue

        description_upper = description.upper()
        if description_upper in {"PURCHASES", "TOTAL"}:
            continue

        quantity = _extract_quantity(description)
        unit_price = None
        if len(raw_amounts) > 1:
            unit_price = raw_amounts[-2]
        elif len(amount_tokens) > 1:
            unit_price = amount_tokens[-2]["value"]
        elif quantity and quantity > 0:
            unit_price = line_total / quantity

        row = {
            "description": description,
            "quantity": quantity,
            "unit_of_measure": None,
            "unit_price": unit_price if unit_price is not None else line_total,
            "net_amount": line_total,
            "vat_rate": None,
            "vat_amount": None,
            "gross_amount": line_total,
            "currency": currency,
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
def _load_ocr_payload(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.exception("Failed to read OCR JSON %s", path)
        return {"words": []}


def _build_lines(words: List[Dict[str, Any]]) -> List[TableLine]:
    lines: Dict[int, TableLine] = {}
    for word in words:
        line_id = word.get("line_num", 0)
        entry = lines.get(line_id)
        if not entry:
            entry = TableLine(
                text="",
                words=[],
                line_num=line_id,
                min_x=10_000,
                max_x=0,
                center_y=0.0,
            )
            lines[line_id] = entry
        entry.words.append(word)
        bbox = word.get("bbox", [0, 0, 0, 0])
        entry.min_x = min(entry.min_x, bbox[0])
        entry.max_x = max(entry.max_x, bbox[2])
    for entry in lines.values():
        entry.words.sort(key=lambda w: w.get("bbox", [0])[0])
        entry.text = " ".join(word.get("text", "") for word in entry.words).strip()
        if entry.words:
            centers = [
                (w.get("bbox", [0, 0, 0, 0])[1] + w.get("bbox", [0, 0, 0, 0])[3]) / 2
                for w in entry.words
            ]
            entry.center_y = sum(centers) / len(centers)
        else:
            entry.center_y = 0.0
    return [lines[line_id] for line_id in sorted(lines)]


def _line_has_amount(line: TableLine) -> bool:
    return bool(_extract_amount_tokens(line))


def _extract_amount_tokens(line: TableLine) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    width = max(line.max_x - line.min_x, 1)
    threshold = line.min_x + width * 0.55
    for idx, word in enumerate(line.words):
        text = (word.get("text") or "").strip()
        if not text:
            continue
        bbox = word.get("bbox", [0, 0, 0, 0])
        center = (bbox[0] + bbox[2]) / 2
        if center < threshold:
            continue
        match = AMOUNT_PATTERN.fullmatch(text)
        if not match:
            continue
        raw_value = match.group("value").replace(",", "")
        currency_symbol = (match.group("currency") or "").strip()
        has_decimal = "." in raw_value
        if not currency_symbol and not has_decimal:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        results.append(
            {
                "value": value,
                "currency": CURRENCY_SYMBOLS.get(currency_symbol),
                "index": idx,
            }
        )
    return results


def _description_tokens(words: List[Dict[str, Any]], amounts: List[Dict[str, Any]]) -> List[str]:
    first_amount_idx = min(amount["index"] for amount in amounts)
    tokens = []
    for idx, word in enumerate(words):
        if idx >= first_amount_idx:
            break
        tokens.append(word.get("text", ""))
    return tokens


def _extract_percentage(text: str) -> float | None:
    match = PERCENT_PATTERN.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _extract_quantity(description: str) -> float | None:
    match = re.search(r"(\d+(?:[\.,]\d+)?)", description)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def _find_line_with_keywords(lines: List[TableLine], keywords: Iterable[str]) -> int | None:
    keyword_set = [kw.upper() for kw in keywords]
    for idx, line in enumerate(lines):
        upper = line.text.upper()
        if any(keyword in upper for keyword in keyword_set):
            return idx
    return None


def _is_summary_line(text_upper: str) -> bool:
    if "TAX" in text_upper and "TOTAL" not in text_upper:
        return False
    return any(keyword in text_upper for keyword in SUMMARY_LINE_KEYWORDS)


def _extract_amounts_from_text(text: str) -> List[float]:
    results: List[float] = []
    for match in AMOUNT_PATTERN.finditer(text):
        value_raw = match.group("value").replace(",", "")
        currency_symbol = match.group("currency")
        has_decimal = "." in value_raw
        if not has_decimal and not currency_symbol:
            continue
        try:
            results.append(float(value_raw))
        except ValueError:
            continue
    return results
