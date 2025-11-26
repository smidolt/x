"""Configuration helpers for the invoice parsing MVP."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import logging

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CompanyConfig:
    name: str
    tax_id: str


@dataclass(slots=True)
class InputConfig:
    path: Path
    archive: Optional[Path] = None


@dataclass(slots=True)
class OutputConfig:
    base_path: Path
    json_dir: Path
    summary_csv: Path


@dataclass(slots=True)
class FeaturesConfig:
    layoutlm_enabled: bool = True
    llm_validation_enabled: bool = False


@dataclass(slots=True)
class LayoutLMConfig:
    enabled: bool = True
    model_name: str = "microsoft/layoutlmv3-base"
    offline: bool = False


@dataclass(slots=True)
class LLMConfig:
    enabled: bool = False
    backend: str = "stub"
    model_name: Optional[str] = None


@dataclass(slots=True)
class OCRConfig:
    engine: str = "tesseract"  # "tesseract", "paddle", or comma-separated list/"auto"
    languages: str = "eng"
    page_segmentation_mode: int = 6
    oem: int = 3
    enable_stub_fallback: bool = True
    paddle_lang: str = "en"


@dataclass(slots=True)
class PreprocessingConfig:
    enabled: bool = True
    grayscale: bool = True
    resize_max_width: int = 2480
    resize_max_height: int = 3508
    normalize_dpi: bool = True
    target_dpi: int = 300
    normalize_to_a4: bool = True
    deskew_enabled: bool = True
    deskew_min_angle: float = 3.0
    deskew_max_angle: float = 5.0
    crop_content_enabled: bool = True
    crop_threshold: int = 240
    crop_margin: int = 12
    denoise_enabled: bool = True
    denoise_strength: str = "medium"
    contrast_enhance_enabled: bool = True
    adaptive_binarization_enabled: bool = False
    binarization_window_size: int = 35
    binarization_offset: int = 10


@dataclass(slots=True)
class AppConfig:
    company: CompanyConfig
    input: InputConfig
    output: OutputConfig
    document_type: str = "invoice"
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    layoutlm: LayoutLMConfig = field(default_factory=LayoutLMConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)


def _validate_dict(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in raw:
        raise KeyError(f"Missing '{key}' section in config.yaml")
    if not isinstance(raw[key], dict):
        raise TypeError(f"Section '{key}' must be a mapping")
    return raw[key]


def load_config(path: Path) -> AppConfig:
    """Load and validate configuration from the YAML file."""
    raw_config = _load_yaml_file(path)

    company_cfg = _validate_dict(raw_config, "company")
    input_cfg = _validate_dict(raw_config, "input")
    output_cfg = _validate_dict(raw_config, "output")
    features_cfg = raw_config.get("features", {})
    layoutlm_cfg = raw_config.get("layoutlm", {})
    llm_cfg = raw_config.get("llm", {})
    ocr_cfg = raw_config.get("ocr", {})
    preprocessing_cfg = raw_config.get("preprocessing", {})

    config = AppConfig(
        company=CompanyConfig(
            name=str(company_cfg.get("name", "")),
            tax_id=str(company_cfg.get("tax_id", "")),
        ),
        input=InputConfig(
            path=Path(input_cfg.get("path", "input")),
            archive=Path(input_cfg["archive"]) if input_cfg.get("archive") else None,
        ),
        output=OutputConfig(
            base_path=Path(output_cfg.get("path", "output")),
            json_dir=Path(output_cfg.get("json_dir", "output/json")),
            summary_csv=Path(output_cfg.get("summary", "output/summary.csv")),
        ),
        document_type=str(raw_config.get("document_type", "invoice")),
        features=FeaturesConfig(
            layoutlm_enabled=bool(features_cfg.get("layoutlm_enabled", True)),
            llm_validation_enabled=bool(features_cfg.get("llm_validation_enabled", False)),
        ),
        layoutlm=LayoutLMConfig(
            enabled=bool(layoutlm_cfg.get("enabled", True)),
            model_name=str(layoutlm_cfg.get("model_name", "microsoft/layoutlmv3-base")),
            offline=bool(layoutlm_cfg.get("offline", False)),
        ),
        llm=LLMConfig(
            enabled=bool(llm_cfg.get("enabled", False)),
            backend=str(llm_cfg.get("backend", "stub")),
            model_name=llm_cfg.get("model_name"),
        ),
        ocr=OCRConfig(
            engine=str(ocr_cfg.get("engine", "tesseract")),
            languages=str(ocr_cfg.get("languages", "eng")),
            page_segmentation_mode=int(ocr_cfg.get("page_segmentation_mode", 6)),
            oem=int(ocr_cfg.get("oem", 3)),
            enable_stub_fallback=bool(ocr_cfg.get("enable_stub_fallback", True)),
            paddle_lang=str(ocr_cfg.get("paddle_lang", "en")),
        ),
        preprocessing=PreprocessingConfig(
            enabled=bool(preprocessing_cfg.get("enabled", True)),
            grayscale=bool(preprocessing_cfg.get("grayscale", True)),
            resize_max_width=int(preprocessing_cfg.get("resize_max_width", 2480)),
            resize_max_height=int(preprocessing_cfg.get("resize_max_height", 3508)),
            normalize_dpi=bool(preprocessing_cfg.get("normalize_dpi", True)),
            target_dpi=int(preprocessing_cfg.get("target_dpi", 300)),
            normalize_to_a4=bool(preprocessing_cfg.get("normalize_to_a4", True)),
            deskew_enabled=bool(preprocessing_cfg.get("deskew_enabled", True)),
            deskew_min_angle=float(preprocessing_cfg.get("deskew_min_angle", 1.5)),
            deskew_max_angle=float(preprocessing_cfg.get("deskew_max_angle", 5.0)),
            crop_content_enabled=bool(preprocessing_cfg.get("crop_content_enabled", True)),
            crop_threshold=int(preprocessing_cfg.get("crop_threshold", 240)),
            crop_margin=int(preprocessing_cfg.get("crop_margin", 12)),
            denoise_enabled=bool(preprocessing_cfg.get("denoise_enabled", True)),
            denoise_strength=str(preprocessing_cfg.get("denoise_strength", "medium")),
            contrast_enhance_enabled=bool(preprocessing_cfg.get("contrast_enhance_enabled", True)),
            adaptive_binarization_enabled=bool(
                preprocessing_cfg.get("adaptive_binarization_enabled", False)
            ),
            binarization_window_size=int(preprocessing_cfg.get("binarization_window_size", 35)),
            binarization_offset=int(preprocessing_cfg.get("binarization_offset", 10)),
        ),
    )

    LOGGER.debug("Loaded configuration: %s", config)
    return config


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        LOGGER.debug("Using PyYAML to parse %s", path)
        return yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        LOGGER.warning(
            "PyYAML not installed; falling back to minimal YAML parser for %s", path
        )
        return _minimal_yaml_load(text)


def _minimal_yaml_load(text: str) -> Dict[str, Any]:
    """Very small subset YAML parser (mapping-only)."""
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        value = value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        if not value:
            nested: Dict[str, Any] = {}
            current[key] = nested
            stack.append((indent, nested))
        else:
            current[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    return value
