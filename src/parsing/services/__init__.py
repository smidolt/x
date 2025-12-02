"""Service-style entrypoints for parsing."""

from .meta_service import run as run_meta
from .items_service import run as run_items
from .classic_service import run as run_classic

__all__ = ["run_meta", "run_items", "run_classic"]
