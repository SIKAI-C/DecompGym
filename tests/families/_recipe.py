"""Shared checks of the small-scale recipe regression tests (tests/families/test_{ufl,km,fctp}.py); solver-free."""
from __future__ import annotations

import hashlib
from pathlib import Path

from tests.reference_data import reference_path


REPO = Path(__file__).resolve().parents[2]
SIZES = ("small", "medium", "large")
SEEDS = (0, 1, 2)


def fresh_matches_pins_and_stored(generate, family: str, size: str, seed: int, pins: dict, tmp_path: Path) -> Path:
    """Generate one instance into tmp_path; its data.json must hash to the pinned value and, together with source.json,
    equal the stored final instance of outputs/litgen_data (missing: fail, or skip under the reference-data override)."""
    directory = tmp_path / f"{family}_{size}_seed{seed:03d}"
    generate(seed, size, directory)
    fresh = (directory / "data.json").read_bytes()
    assert hashlib.sha256(fresh).hexdigest() == pins[size, seed]
    stored = reference_path("DECOMPGYM_SMALL_SCALE_DATA", REPO / "outputs" / "litgen_data", "small-scale dataset (outputs/litgen_data)")
    final = stored / "bd" / family / directory.name
    assert (final / "data.json").read_bytes() == fresh
    assert (final / "source.json").read_bytes() == (directory / "source.json").read_bytes()
    return directory
