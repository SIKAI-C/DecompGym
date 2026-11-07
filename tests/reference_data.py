"""Locations of the published files that some recipe tests compare against (not needed to generate instances).

The tests fail when a file is missing, so a missing download is never mistaken for a pass. Set
DECOMPGYM_ALLOW_MISSING_REFERENCE_DATA=1 to turn the failure into an explicit skip (for example on a machine without
the downloads); see README "Reference data for recipe tests" for sources and placement.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "outputs" / "reference_data"


def reference_path(variable: str, default: Path, what: str) -> Path:
    path = Path(os.environ.get(variable, default))
    if not path.exists():
        message = f"{what} not found at {path} (set {variable} or see README 'Reference data for recipe tests')"
        if os.environ.get("DECOMPGYM_ALLOW_MISSING_REFERENCE_DATA") == "1":
            pytest.skip(message)
        pytest.fail(message)
    return path
