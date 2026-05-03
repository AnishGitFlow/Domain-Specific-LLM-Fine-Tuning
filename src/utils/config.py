"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve a project-relative path."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with resolve_path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it."""
    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
