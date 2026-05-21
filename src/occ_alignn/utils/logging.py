"""Logging setup."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a console logger with a compact format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)
