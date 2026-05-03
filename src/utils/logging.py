"""Logging helpers."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """Return a Rich-backed logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger(name)
