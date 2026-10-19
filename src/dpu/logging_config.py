"""Logging setup (replaces print)."""

from __future__ import annotations

import logging
from typing import Final

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root handler once and set the package logger to ``level`` (INFO by default)."""
    logging.basicConfig(level=level, format=LOG_FORMAT)
    logging.getLogger("dpu").setLevel(level)
