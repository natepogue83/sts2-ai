"""Canonical identifiers shared by offline and live data adapters."""

from __future__ import annotations

import re
from typing import Any


CHARACTER_NAMES = {
    "THE IRONCLAD": "CHARACTER.IRONCLAD",
    "IRONCLAD": "CHARACTER.IRONCLAD",
    "THE SILENT": "CHARACTER.SILENT",
    "SILENT": "CHARACTER.SILENT",
    "THE DEFECT": "CHARACTER.DEFECT",
    "DEFECT": "CHARACTER.DEFECT",
    "THE NECROBINDER": "CHARACTER.NECROBINDER",
    "NECROBINDER": "CHARACTER.NECROBINDER",
    "THE REGENT": "CHARACTER.REGENT",
    "REGENT": "CHARACTER.REGENT",
}


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_")
    return text.upper() or "UNKNOWN"


def canonical_character(value: Any) -> str:
    text = str(value or "").strip()
    if text.upper().startswith("CHARACTER."):
        return text.upper()
    return CHARACTER_NAMES.get(text.upper(), f"CHARACTER.{_slug(text)}")


def canonical_entity(value: Any, prefix: str) -> str:
    text = str(value or "").strip()
    if text.upper().startswith(prefix.upper() + "."):
        return text.upper()
    return f"{prefix.upper()}.{_slug(text)}"
