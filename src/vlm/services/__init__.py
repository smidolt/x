"""Service-style entrypoints for VLM components."""

from .blocks_service import run as run_blocks
from .reasoner_service import run as run_reasoner

__all__ = ["run_blocks", "run_reasoner"]
