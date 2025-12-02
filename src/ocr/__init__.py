"""OCR package."""

from .engine import OCRResult, run_ocr
from .service_tesseract import run as run_tesseract

__all__ = ["OCRResult", "run_ocr", "run_tesseract"]
