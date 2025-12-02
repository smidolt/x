"""Image preprocessing helpers for invoices."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import logging
import math

try:  # Pillow is our primary dependency for preprocessing
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat
except ImportError as exc:  # pragma: no cover - dependency guard
    PIL_AVAILABLE = False
    PIL_IMPORT_ERROR = exc
else:  # pragma: no cover - dependency guard
    PIL_AVAILABLE = True
    PIL_IMPORT_ERROR = None

try:  # numpy is optional but enables faster math and OpenCV interop
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - dependency guard
    np = None


try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover - dependency guard
    cv2 = None


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class StepResult:
    image: "Image.Image"
    applied: bool
    warning: str | None = None


def ensure_pillow_available() -> None:
    if not PIL_AVAILABLE:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Pillow is required for preprocessing steps. Install it via 'pip install pillow'."
        ) from PIL_IMPORT_ERROR


def load_image(path: Path) -> "Image.Image":
    ensure_pillow_available()
    with Image.open(path) as image:
        return image.convert("RGB")


def to_grayscale(image: "Image.Image") -> StepResult:
    if image.mode == "L":
        return StepResult(image=image, applied=False)
    return StepResult(image=ImageOps.grayscale(image), applied=True)


def resize_to_limits(image: "Image.Image", max_width: int, max_height: int) -> StepResult:
    if max_width <= 0 or max_height <= 0:
        return StepResult(image=image, applied=False)

    width, height = image.size
    ratio = min(max_width / float(width), max_height / float(height), 1.0)
    if ratio >= 0.999:
        return StepResult(image=image, applied=False)
    new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
    LOGGER.debug("Resizing from %s to %s", (width, height), new_size)
    return StepResult(image=image.resize(new_size, Image.BICUBIC), applied=True)


def normalize_to_a4(image: "Image.Image", target_dpi: int, enabled: bool) -> StepResult:
    if not enabled:
        return StepResult(image=image, applied=False)
    if target_dpi <= 0:
        return StepResult(image=image, applied=False)
    # A4 dimensions in inches (210 x 297 mm)
    width_px = int(round(8.27 * target_dpi))
    height_px = int(round(11.69 * target_dpi))
    if image.size == (width_px, height_px):
        return StepResult(image=image, applied=False)
    LOGGER.debug("Normalizing page to A4 @ %sdpi: %s -> %s", target_dpi, image.size, (width_px, height_px))
    return StepResult(image=image.resize((width_px, height_px), Image.BILINEAR), applied=True)


def crop_to_content(
    image: "Image.Image", threshold: int = 240, margin: int = 10
) -> StepResult:
    gray = ImageOps.grayscale(image)
    # Create a mask of non-background regions
    mask = gray.point(lambda x: 255 if x < threshold else 0)
    bbox = mask.getbbox()
    if not bbox:
        return StepResult(image=image, applied=False)
    left = max(0, bbox[0] - margin)
    upper = max(0, bbox[1] - margin)
    right = min(image.width, bbox[2] + margin)
    lower = min(image.height, bbox[3] + margin)
    expanded_box = (left, upper, right, lower)
    cropped = image.crop(expanded_box)
    LOGGER.debug("Cropping image bbox=%s margin=%s -> %s", bbox, margin, expanded_box)
    return StepResult(image=cropped, applied=True)


def deskew_image(image: "Image.Image", max_angle: float, min_angle: float = 1.0) -> StepResult:
    gray = ImageOps.grayscale(image)
    mask = gray.point(lambda x: 255 if x < 240 else 0)
    bbox = mask.getbbox()
    if not bbox:
        return StepResult(image=image, applied=False, warning="Deskew skipped (no content detected).")

    img_area = image.width * image.height or 1
    content_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    coverage = content_area / img_area
    if coverage < 0.3:
        return StepResult(
            image=image,
            applied=False,
            warning="Deskew skipped (sparse content; avoiding over-rotation).",
        )

    angles: list[float] = []

    # Prefer Hough-based angle (more stable on invoices with many horizontal lines)
    hough_angle = _estimate_angle_via_hough(gray, max_angle)
    if hough_angle is not None:
        angles.append(hough_angle)

    # Regression-based estimate (conservative)
    regression_angle = _estimate_angle_via_regression(gray, max_angle)
    if regression_angle is not None:
        angles.append(regression_angle)

    # OpenCV minAreaRect estimate
    if cv2 and np:
        arr = np.array(gray)
        coords = np.column_stack(np.where(arr < 255))
        if coords.size:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            angles.append(angle)
    elif not (cv2 and np):
        LOGGER.debug("OpenCV/NumPy not available; skipping cv2-based deskew.")

    if not angles:
        warning = None
        if not (cv2 and np):
            warning = "Deskew skipped (install numpy + opencv-python for improved accuracy)."
        return StepResult(image=image, applied=False, warning=warning)

    # Choose the smallest absolute angle to avoid over-rotation
    chosen = min(angles, key=lambda a: abs(a))
    if abs(chosen) < max(min_angle, 0.1):
        return StepResult(image=image, applied=False)

    limited_angle = max(-max_angle, min(max_angle, chosen))
    LOGGER.debug(
        "Deskew angle candidates=%s chosen=%.2f limited=%.2f",
        [round(a, 2) for a in angles],
        chosen,
        limited_angle,
    )
    rotated = image.rotate(limited_angle, resample=Image.BICUBIC, fillcolor=255)
    return StepResult(image=rotated, applied=True)


def _estimate_angle_via_regression(gray_image: "Image.Image", max_angle: float) -> float | None:
    width, height = gray_image.size
    pixels = gray_image.load()
    samples = max(16, width // 40)
    threshold = 240
    coords: list[tuple[float, float]] = []

    for index in range(samples):
        x = int(index * (width - 1) / max(samples - 1, 1))
        top = None
        bottom = None
        for y in range(height):
            if pixels[x, y] < threshold:
                top = y
                break
        for y in range(height - 1, -1, -1):
            if pixels[x, y] < threshold:
                bottom = y
                break
        if top is None or bottom is None:
            continue
        coords.append((float(x), (top + bottom) / 2.0))

    if len(coords) < 8:
        return None

    mean_x = sum(x for x, _ in coords) / len(coords)
    mean_y = sum(y for _, y in coords) / len(coords)
    sum_xx = sum((x - mean_x) ** 2 for x, _ in coords)
    sum_xy = sum((x - mean_x) * (y - mean_y) for x, y in coords)

    if sum_xx == 0:
        return None

    slope = sum_xy / sum_xx
    angle_deg = math.degrees(math.atan(slope))
    limited_angle = max(-max_angle, min(max_angle, angle_deg))
    LOGGER.debug("Deskew regression slope=%.4f angle=%.2f coords=%d", slope, angle_deg, len(coords))
    return limited_angle


def _estimate_angle_via_hough(gray_image: "Image.Image", max_angle: float) -> float | None:
    if not (cv2 and np):
        return None
    arr = np.array(gray_image)
    # Boost contrast and binarize to emphasize text/lines
    arr_blur = cv2.GaussianBlur(arr, (3, 3), 0)
    _, binary = cv2.threshold(arr_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 1800.0,  # fine resolution (~0.1 deg)
        threshold=80,
        minLineLength=max(arr.shape[1], arr.shape[0]) * 0.15,
        maxLineGap=20,
    )
    if lines is None or len(lines) == 0:
        return None

    angles: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        # Normalize to [-90, 90]
        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180
        if abs(angle) <= max_angle + 1:
            angles.append(angle)

    if not angles:
        return None
    # Use median for robustness
    angles.sort()
    median = angles[len(angles) // 2]
    return median


def denoise_image(image: "Image.Image", strength: str = "medium") -> StepResult:
    strength_map = {
        "low": 3,
        "medium": 5,
        "high": 7,
    }
    kernel = strength_map.get(strength, 5)

    if cv2 and np:
        arr = np.array(ImageOps.grayscale(image))
        denoised = cv2.fastNlMeansDenoising(arr, h=kernel * 5)
        return StepResult(image=Image.fromarray(denoised), applied=True)

    LOGGER.debug("Using PIL median filter for denoising (kernel=%s)", kernel)
    return StepResult(image=image.filter(ImageFilter.MedianFilter(size=kernel | 1)), applied=True)


def enhance_contrast(image: "Image.Image") -> StepResult:
    enhancer = ImageEnhance.Contrast(image)
    contrasted = enhancer.enhance(1.2)
    return StepResult(image=contrasted, applied=True)


def adaptive_binarize(
    image: "Image.Image", window_size: int = 35, offset: int = 10
) -> StepResult:
    if not window_size or window_size < 3:
        return StepResult(image=image, applied=False)

    window_size = window_size if window_size % 2 == 1 else window_size + 1
    gray = ImageOps.grayscale(image)

    if cv2 and np:
        arr = np.array(gray)
        binary = cv2.adaptiveThreshold(
            arr,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            window_size,
            offset,
        )
        return StepResult(image=Image.fromarray(binary), applied=True)

    LOGGER.debug("Adaptive binarization fallback using blurred threshold")
    blurred = gray.filter(ImageFilter.BoxBlur(max(1, window_size // 4)))
    stat = ImageStat.Stat(blurred)
    threshold = stat.mean[0] - offset
    binary = gray.point(lambda x: 255 if x > threshold else 0)
    return StepResult(image=binary, applied=True)


def record_step(
    steps: List[str],
    warnings: List[str],
    name: str,
    step_result: StepResult,
) -> "Image.Image":
    if step_result.applied:
        steps.append(name)
    if step_result.warning:
        warnings.append(step_result.warning)
    return step_result.image
