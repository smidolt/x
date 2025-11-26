"""Parse meta-information from OCR/Layout annotations."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import json
import logging
import re

LOGGER = logging.getLogger(__name__)


DATE_PATTERN_YMD = re.compile(
    r"(?P<year>20\d{2})\s*[-\./]\s*(?P<month>\d{1,2})\s*[-\./]\s*(?P<day>\d{1,2})"
)
DATE_PATTERN_MDY = re.compile(
    r"(?P<month>\d{1,2})\s*[-\./]\s*(?P<day>\d{1,2})\s*[-\./]\s*(?P<year>20\d{2})"
)
DATE_TEXT_PATTERN = re.compile(
    r"(?P<month_name>[A-Za-z]{3,})\s+(?P<day>\d{1,2}),\s+(?P<year>20\d{2})"
)
MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "SEPT": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
AMOUNT_PATTERN = re.compile(r"(?P<currency>[$€£])?\s*(?P<value>\d+[\d,.]*)")

SELLER_LABELS = {"PRODUCT(S) PROVIDED", "PROVIDER", "SUPPLIER", "SELLER", "REMIT"}
BUYER_LABELS = {"BUYER", "BILLED", "CUSTOMER", "BILL TO"}
TOTAL_LABELS = {"total", "checkout", "amount"}
TAX_LABELS = {"tax", "ddv", "vat"}
COMPANY_MARKERS = [
    " INC",
    " LLC",
    " LTD",
    " D.O.O",
    " D.D.",
    " GMBH",
    " BV",
    " SAS",
    " PLC",
    " CORP",
    " COMPANY",
    " LLP",
]
COMPANY_PATTERN = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9&\.,\- ]{0,80}?(?:INC|LLC|LTD|D\.O\.O|D\.D\.|GMBH|BV|SAS|PLC|CORP|COMPANY|LLP))",
    re.IGNORECASE,
)


@dataclass(slots=True)
class MetaData:
    raw: Dict[str, Any] = field(default_factory=dict)


class MetaParser:
    """Parse invoice meta fields using OCR text and optional layout hints."""

    def __init__(self, company_name: str, company_tax_id: str) -> None:
        self.company_name = company_name
        self.company_tax_id = company_tax_id

    def run(self, ocr_result_path: Path, layout_annotations: Dict[str, Any] | None = None) -> MetaData:
        LOGGER.info("Parsing meta fields from %s", ocr_result_path)
        payload = _load_json(ocr_result_path)
        words = payload.get("words", [])
        lines = _aggregate_lines(words)
        annotations = layout_annotations.get("tokens", []) if layout_annotations else []

        meta: Dict[str, Any] = {
            "document_type": "invoice",
            "ocr_word_count": len(words),
            "seller_name": self.company_name,
            "seller_tax_id": self.company_tax_id,
        }

        meta.update(self._extract_parties(lines, annotations))
        meta.update(self._extract_dates(lines))
        meta.update(self._extract_numbers(lines))
        meta.update(self._extract_totals(lines))
        meta.update(self._extract_tax_info(lines))

        if layout_annotations:
            meta["layout_tokens"] = len(annotations)
            meta["layout_engine"] = layout_annotations.get("engine")
            meta["layout_model"] = layout_annotations.get("model_name")

        return MetaData(raw=meta)

    # ------------------------------------------------------------------
    def _extract_parties(
        self, lines: Dict[int, Dict[str, Any]], annotations: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        buyer_header = _find_block_by_phrase(
            lines,
            ["BILL TO", "BILLED TO", "SOLD TO", "BUYER", "CUSTOMER"],
        )
        if buyer_header:
            section = _collect_section(lines, buyer_header["line_num"], side="left", max_lines=4)
            if section:
                result.update(_make_party("buyer", section))
            sold_header = _find_block_by_phrase(lines, ["SOLD TO", "CUSTOMER ADDRESS", "SHIP TO"])
            if sold_header:
                sold_section = _collect_section(lines, sold_header["line_num"], side="left", max_lines=5)
                if sold_section:
                    address = _extract_address_block(sold_section["text"])
                    if address and _needs_address_upgrade(result.get("buyer_address")):
                        result["buyer_address"] = address

        seller_header = _find_labeled_block(lines, SELLER_LABELS)
        if seller_header:
            section = _collect_section(lines, seller_header["line_num"], side="right")
            if section:
                result.update(_make_party("seller", section))
        elif buyer_header:
            section = _collect_preceding_section(lines, buyer_header["line_num"], side=None)
            if section:
                result.update(_make_party("seller", section))
        if "seller_name" not in result:
            section = _collect_leading_section(lines, buyer_header["line_num"] if buyer_header else None)
            if section:
                result.update(_make_party("seller", section))

        # Layout labels override heuristics if available
        if annotations:
            seller_lines = _collect_labelled_lines(
                annotations,
                "header",
                max_relative_y=0.45,
                exclude_keywords={"TOTAL", "SUBTOTAL", "TAX"},
            )
            if seller_lines and "seller_name" not in result:
                result.update(_make_party("seller", seller_lines))
            buyer_lines = _collect_labelled_lines(
                annotations,
                "value",
                max_relative_y=0.65,
                exclude_keywords={"TOTAL", "SUBTOTAL"},
            )
            if buyer_lines and "buyer_name" not in result:
                result.update(_make_party("buyer", buyer_lines))

        refined_seller = _pick_company_name(lines, result.get("seller_name"))
        if refined_seller:
            result["seller_name"] = refined_seller
        return result

    def _extract_dates(self, lines: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, standard_name in [
            ("INVOICE ISSUE DATE", "issue_date"),
            ("INVOICE DATE", "issue_date"),
            ("DATE", "issue_date"),
            ("PAYMENT DATE", "payment_date"),
            ("DUE DATE", "payment_date"),
            ("ACTIVATION DATE", "supply_date"),
        ]:
            line = _find_block(lines, key)
            if not line:
                continue
            date = _extract_first_date(line["text"])
            if date:
                result[standard_name] = date
        return result

    def _extract_numbers(self, lines: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        invoice_line = (
            _find_block(lines, "INVOICE NUMBER")
            or _find_block(lines, "INVOICE #")
            or _find_block(lines, "INVOICE NO")
            or _find_block(lines, "INVOICE")
        )
        if invoice_line:
            match = re.search(
                r"(?:INVOICE(?:\s+NUMBER|\s+NO\.?|\s*#)?[:#\s]+)([A-Za-z0-9-]{4,})",
                invoice_line["text"],
                re.IGNORECASE,
            )
            if match:
                result["invoice_number"] = match.group(1).strip()
            else:
                extracted = re.findall(r"([A-Za-z0-9-]{5,})", invoice_line["text"])
                if extracted:
                    result["invoice_number"] = extracted[-1]

        sale_line = _find_block(lines, "SALE")
        if sale_line:
            extracted = re.findall(r"(\d{4,})", sale_line["text"])
            if extracted:
                result["sale_reference"] = extracted[0]
        return result

    def _extract_totals(self, lines: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        line = _find_block(lines, "TOTAL", last=True) or _find_block(
            lines,
            "TOTAL AT CHECKOUT",
            last=True,
        )
        totals: Dict[str, Any] = {}
        if line:
            amount = _extract_amount(line["text"], strict=True)
            if amount is not None:
                totals["total_gross"] = amount
            currency = _detect_currency(line["text"])
            if currency:
                totals["currency"] = currency
        net_line = (
            _find_block(lines, "PURCHASES")
            or _find_block(lines, "SUBTOTAL")
            or _find_block(lines, "SUB TOTAL")
        )
        if net_line:
            net = _extract_amount(net_line["text"], strict=True)
            if net is not None:
                totals["total_net"] = net
            if "currency" not in totals:
                currency = _detect_currency(net_line["text"])
                if currency:
                    totals["currency"] = currency
        return totals

    def _extract_tax_info(self, lines: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        vat_total = 0.0
        rates: List[float] = []
        for tax_line in _find_blocks(lines, "TAX"):
            upper = tax_line["text"].upper()
            if "TAX ID" in upper or "TAX 1D" in upper:
                continue
            amount = _extract_amount(tax_line["text"], strict=True)
            if amount is not None:
                vat_total += amount
            match = re.search(r"([0-9]+\.?[0-9]*)%", tax_line["text"])
            if match:
                try:
                    rates.append(float(match.group(1)))
                except ValueError:
                    pass
        if vat_total:
            result["total_vat"] = vat_total
        if rates:
            result["vat_rate"] = max(rates)
        return result


# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.exception("Failed to read OCR JSON %s", path)
        return {"words": []}


def _aggregate_lines(words: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    lines: Dict[int, Dict[str, Any]] = {}
    for word in words:
        line_id = word.get("line_num", 0)
        entry = lines.setdefault(
            line_id,
            {
                "words": [],
                "text": "",
                "line_num": line_id,
                "min_x": 10_000,
                "max_x": 0,
                "min_y": 10_000,
                "max_y": 0,
            },
        )
        entry["words"].append(word)
        bbox = word.get("bbox", [0, 0, 0, 0])
        entry["min_x"] = min(entry["min_x"], bbox[0])
        entry["max_x"] = max(entry["max_x"], bbox[2])
        entry["min_y"] = min(entry["min_y"], bbox[1])
        entry["max_y"] = max(entry["max_y"], bbox[3])
    for entry in lines.values():
        entry["words"].sort(key=lambda w: w.get("bbox", [0])[0])
        entry["text"] = " ".join(w.get("text", "") for w in entry["words"]).strip()
    return lines


def _find_block(
    lines: Dict[int, Dict[str, Any]], keyword: str, *, last: bool = False
) -> Dict[str, Any] | None:
    blocks = list(_find_blocks(lines, keyword))
    if not blocks:
        return None
    return blocks[-1] if last else blocks[0]


def _find_blocks(lines: Dict[int, Dict[str, Any]], keyword: str) -> Iterable[Dict[str, Any]]:
    keyword_upper = keyword.upper()
    for line_num in sorted(lines):
        entry = lines[line_num]
        if keyword_upper in entry["text"].upper():
            yield entry


def _extract_first_date(text: str) -> str | None:
    for pattern in (DATE_PATTERN_YMD, DATE_PATTERN_MDY):
        match = pattern.search(text)
        if match:
            year = int(match.group("year"))
            month = int(match.group("month"))
            day = int(match.group("day"))
            break
    else:
        match = DATE_TEXT_PATTERN.search(text)
        if not match:
            return None
        month_name = match.group("month_name").strip().upper()[:3]
        month = MONTHS.get(month_name)
        if not month:
            return None
        day = int(match.group("day"))
        year = int(match.group("year"))
    try:
        dt = datetime(year, month, day)
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def _extract_amount(text: str, strict: bool = False) -> float | None:
    matches = list(AMOUNT_PATTERN.finditer(text))
    if not matches:
        return None
    candidates: List[float] = []
    for match in matches:
        value_raw = match.group("value").replace(",", "")
        currency = match.group("currency")
        has_decimal = "." in value_raw
        if strict and not (currency or has_decimal):
            continue
        try:
            candidates.append(float(value_raw))
        except ValueError:
            continue
    if not candidates:
        return None
    return candidates[-1]


def _collect_section(
    lines: Dict[int, Dict[str, Any]],
    start_line: int,
    side: str,
    max_lines: int = 6,
) -> Dict[str, Any] | None:
    collected: List[str] = []
    current = start_line + 1
    stop_words = {
        "DETAIL",
        "SUMMARY",
        "INVOICE",
        "DOMAIN",
        "TOTAL",
        "SUBTOTAL",
        "SALES TAX",
        "CREDIT",
        "PAYMENT",
        "TAX",
        "NOTE",
    }
    for _ in range(max_lines):
        entry = lines.get(current)
        if not entry:
            break
        full_text = entry.get("text", "")
        if any(keyword in full_text.upper() for keyword in stop_words):
            current += 1
            continue
        text = _entry_text(entry, side)
        if not text:
            current += 1
            continue
        collected.append(text)
        current += 1
    if not collected:
        return None
    return {"text": "\n".join(collected), "words": []}


def _entry_text(entry: Dict[str, Any], side: str | None = None) -> str:
    words = entry.get("words", [])
    filtered: List[str] = []
    for word in words:
        bbox = word.get("bbox", [0, 0, 0, 0])
        center = (bbox[0] + bbox[2]) / 2
        if side == "left" and center > 900:
            continue
        if side == "right" and center < 900:
            continue
        filtered.append(word.get("text", ""))
    if side and not filtered:
        return ""
    if not filtered:
        filtered = [w.get("text", "") for w in words]
    return " ".join(filtered).strip()


def _collect_preceding_section(
    lines: Dict[int, Dict[str, Any]],
    end_line: int,
    side: str | None,
    max_lines: int = 6,
) -> Dict[str, Any] | None:
    collected: List[str] = []
    stop_words = {"DETAIL", "SUMMARY", "INVOICE", "DOMAIN", "TOTAL", "SUBTOTAL", "SALES TAX", "CREDIT", "PAYMENT", "TAX"}
    for line_num in range(max(1, end_line - max_lines), end_line):
        entry = lines.get(line_num)
        if not entry:
            continue
        full_text = entry.get("text", "")
        if any(keyword in full_text.upper() for keyword in stop_words):
            continue
        text = _entry_text(entry, side)
        if text:
            collected.append(text)
    if not collected:
        return None
    return {"text": "\n".join(collected), "words": []}


def _collect_leading_section(
    lines: Dict[int, Dict[str, Any]],
    stop_line: int | None,
    max_lines: int = 4,
) -> Dict[str, Any] | None:
    upper_bound = stop_line if stop_line is not None else max_lines + 1
    collected: List[str] = []
    for line_num in range(1, min(upper_bound, max_lines + 1)):
        entry = lines.get(line_num)
        if not entry:
            continue
        text = _entry_text(entry, None)
        if text:
            collected.append(text)
    if not collected:
        return None
    return {"text": "\n".join(collected), "words": []}


def _detect_currency(text: str) -> str | None:
    upper = text.upper()
    if "$" in text or "USD" in upper or "US DOLLAR" in upper:
        return "USD"
    if "€" in text or "EUR" in upper or "EVRO" in upper:
        return "EUR"
    if "£" in text or "GBP" in upper:
        return "GBP"
    return None


def _find_labeled_block(lines: Dict[int, Dict[str, Any]], labels: Sequence[str]) -> Dict[str, Any] | None:
    upper_labels = [label.upper() for label in labels]
    for entry in lines.values():
        text_upper = entry.get("text", "").upper()
        if any(label in text_upper for label in upper_labels):
            return entry
    return None


def _find_block_by_phrase(
    lines: Dict[int, Dict[str, Any]], phrases: Sequence[str]
) -> Dict[str, Any] | None:
    upper_phrases = [phrase.upper() for phrase in phrases]
    for line_num in sorted(lines):
        entry = lines[line_num]
        text_upper = entry.get("text", "").upper()
        for phrase in upper_phrases:
            if phrase in text_upper:
                return entry
    return None


def _make_party(prefix: str, block: Dict[str, Any]) -> Dict[str, Any]:
    text = block.get("text", "")
    parts = [part.strip() for part in re.split(r";|\n", text) if part.strip()]
    filtered: List[str] = []
    for part in parts:
        upper = part.upper()
        if any(kw in upper for kw in ["BILL TO", "SOLD TO", "TOTAL", "AMOUNT", "SUMMARY", "DETAIL"]):
            continue
        if any(ch.isalpha() for ch in part):
            filtered.append(part)
    parts = filtered or parts or [text]
    if not parts:
        parts = [text]
    name = parts[0]
    address = "; ".join(parts[1:]) if len(parts) > 1 else ""
    return {
        f"{prefix}_name": name.strip(":"),
        f"{prefix}_address": address,
    }


def _collect_labelled_lines(
    annotations: Sequence[Dict[str, Any]],
    target_label: str,
    max_relative_y: float | None = None,
    exclude_keywords: Sequence[str] | None = None,
) -> Dict[str, Any] | None:
    tokens = [token for token in annotations if token.get("label") == target_label]
    if not tokens:
        return None
    usable_tokens: List[Dict[str, Any]] = []
    max_bottom = max((token.get("bbox", [0, 0, 0, 0])[3] for token in tokens), default=1) or 1
    threshold = max_bottom * max_relative_y if max_relative_y else None
    excludes = {kw.upper() for kw in exclude_keywords or []}
    for token in tokens:
        bbox = token.get("bbox", [0, 0, 0, 0])
        if threshold is not None and bbox[1] > threshold:
            continue
        text_upper = (token.get("text") or "").upper()
        if excludes and any(keyword in text_upper for keyword in excludes):
            continue
        usable_tokens.append(token)
    if not usable_tokens:
        usable_tokens = tokens

    lines: Dict[int, List[str]] = {}
    for token in usable_tokens:
        bbox = token.get("bbox", [0, 0, 0, 0])
        line_id = bbox[1] // 20
        lines.setdefault(line_id, []).append(token["text"])
    if not lines:
        return None
    best_line = max(lines.items(), key=lambda item: len(item[1]))
    return {"text": " ".join(best_line[1]), "words": []}


def _extract_address_block(text: str) -> str:
    lines = [line.strip(" ,") for line in text.split("\n") if line.strip()]
    cleaned: List[str] = []
    skip_tokens = {"FEE", "PERIOD", "UNIT PRICE", "DESCRIPTION"}
    for line in lines:
        upper = line.upper()
        if any(token in upper for token in skip_tokens):
            continue
        cleaned.append(line)
    return "; ".join(cleaned)


def _needs_address_upgrade(address: str | None) -> bool:
    if not address:
        return True
    if "@" in address and not any(ch.isdigit() for ch in address):
        return True
    return False


def _looks_like_address(text: str | None) -> bool:
    if not text:
        return False
    digits = any(ch.isdigit() for ch in text)
    has_street = any(token in text.upper() for token in {"ST", "AVE", "ROAD", "BLVD", "UL.", "ULICA"})
    return digits or has_street


def _pick_company_name(lines: Dict[int, Dict[str, Any]], current: str | None) -> str | None:
    if current and not _looks_like_address(current):
        return current
    for entry in sorted(lines.values(), key=lambda item: item["line_num"]):
        text = entry.get("text", "").strip()
        if not text:
            continue
        upper = text.upper()
        if any(marker in upper for marker in COMPANY_MARKERS):
            if any(keyword in upper for keyword in ["BILL TO", "SOLD TO", "TOTAL", "SUBTOTAL", "AMOUNT DUE"]):
                continue
            snippet = _extract_company_text(text)
            cleaned = re.sub(r"^[^A-Za-z0-9]+", "", snippet).strip(": ")
            return cleaned or snippet.strip(":")
    return current


def _extract_company_text(text: str) -> str:
    upper = text.upper()
    for marker in COMPANY_MARKERS:
        idx = upper.find(marker)
        if idx == -1:
            continue
        start = idx
        while start > 0 and (
            text[start - 1].isalnum() or text[start - 1] in {",", ".", "&", "-", "/"}
        ):
            start -= 1
        snippet = text[start:].strip()
        if snippet:
            return snippet
    match = COMPANY_PATTERN.search(text)
    if match:
        return match.group(1)
    return text
