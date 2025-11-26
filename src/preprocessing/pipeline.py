"""Invoice-focused preprocessing pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import logging
import time

from src.config import PreprocessingConfig
from . import steps

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PreprocessingResult:
    source_path: Path
    processed_path: Path
    steps_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class PreprocessingPipeline:
    """Sequence of preprocessing steps tailored for invoice documents."""

    def __init__(self, output_dir: Path, config: PreprocessingConfig) -> None:
        self.output_dir = output_dir
        self.config = config
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, document_path: Path) -> PreprocessingResult:
        LOGGER.info("Preprocessing %s", document_path)
        start = time.perf_counter()
        steps_applied: List[str] = []
        warnings: List[str] = []

        if not self.config.enabled:
            warning = "Preprocessing disabled via config"
            warnings.append(warning)
            LOGGER.warning(warning)
            return PreprocessingResult(
                source_path=document_path,
                processed_path=document_path,
                steps_applied=steps_applied,
                warnings=warnings,
                elapsed_seconds=0.0,
            )

        image = steps.load_image(document_path)

        if self.config.grayscale:
            image = steps.record_step(steps_applied, warnings, "grayscale", steps.to_grayscale(image))

        if self.config.crop_content_enabled:
            image = steps.record_step(
                steps_applied,
                warnings,
                "crop_to_content",
                steps.crop_to_content(
                    image,
                    threshold=self.config.crop_threshold,
                    margin=self.config.crop_margin,
                ),
            )

        if self.config.deskew_enabled:
            image = steps.record_step(
                steps_applied,
                warnings,
                "deskew",
                steps.deskew_image(
                    image,
                    max_angle=self.config.deskew_max_angle,
                    min_angle=self.config.deskew_min_angle,
                ),
            )

        image = steps.record_step(
            steps_applied,
            warnings,
            "resize",
            steps.resize_to_limits(image, self.config.resize_max_width, self.config.resize_max_height),
        )

        image = steps.record_step(
            steps_applied,
            warnings,
            "normalize_to_a4",
            steps.normalize_to_a4(image, self.config.target_dpi, self.config.normalize_to_a4),
        )

        if self.config.denoise_enabled:
            image = steps.record_step(
                steps_applied,
                warnings,
                "denoise",
                steps.denoise_image(image, self.config.denoise_strength),
            )

        if self.config.contrast_enhance_enabled:
            image = steps.record_step(steps_applied, warnings, "contrast", steps.enhance_contrast(image))

        if self.config.adaptive_binarization_enabled:
            image = steps.record_step(
                steps_applied,
                warnings,
                "adaptive_binarization",
                steps.adaptive_binarize(
                    image,
                    window_size=self.config.binarization_window_size,
                    offset=self.config.binarization_offset,
                ),
            )

        processed_path = self._persist_image(document_path, image)
        elapsed = time.perf_counter() - start
        LOGGER.info(
            "Preprocessing finished for %s in %.2fs (steps=%s)",
            document_path.name,
            elapsed,
            ", ".join(steps_applied) if steps_applied else "none",
        )
        return PreprocessingResult(
            source_path=document_path,
            processed_path=processed_path,
            steps_applied=steps_applied,
            warnings=warnings,
            elapsed_seconds=elapsed,
        )

    def _persist_image(self, document_path: Path, image: "steps.Image") -> Path:
        target = self.output_dir / f"{document_path.stem}_preprocessed.png"
        save_kwargs = {}
        if self.config.normalize_dpi and self.config.target_dpi > 0:
            save_kwargs["dpi"] = (self.config.target_dpi, self.config.target_dpi)
        LOGGER.debug("Writing preprocessed image to %s", target)
        image.save(target, format="PNG", **save_kwargs)
        return target
