"""1D bin packing instances generated with the Scholl, Klein and Juergens (1997) data set 1 recipe.

Source: A. Scholl, R. Klein, C. Juergens (1997), "BISON: A fast hybrid procedure for exactly solving the
one-dimensional bin packing problem", Computers & Operations Research 24(7):627-645, data set 1, as distributed in
BPPLIB (M. Delorme, M. Iori, S. Martello (2018), "BPPLIB: a library for bin packing and cutting stock problems",
Optimization Letters 12:235-250; https://github.com/mdelorme2/BPPLIB, "Scholl 1": 720 instances, n between 50 and
500, c between 100 and 150).

Data set 1 is a full factorial design (20 instances per cell, names N<a>C<b>W<c>_<letter>):
  - n in {50, 100, 200, 500} items (N1..N4),
  - capacity c in {100, 120, 150} (C1, C2, C3),
  - integer weights drawn independently and uniformly from [1, 100], [20, 100] or [30, 100] (W1, W2, W4).

This generator follows one cell family exactly: SCHOLL_CLASS (capacity and weight range), weights i.i.d. uniform
integers. Only n, the size factor that data set 1 itself varies, is set per label: small/medium/large =
200/500/1000 items, i.e. N3, N4 and twice N4 (round-2 recalibration, 2026-11-05; n = 1000 goes beyond the published
grid, which is parametric in n; the round-1 table was N1/N2/N3 = 50/100/200). The arc-flow graph of
src/cg/bp/arcflow.py depends on c and the weight range, not on n; on pilot seed 100 the MIP proved optimality in about
0.15 s (full process) at n = 500 and n = 2000, and litgen_suite rejects any instance where it does not (Threads = 1).
The reference CG aggregates equal weights into at most 81 item types; on the same pilot its full-process time was about
0.6 s at both n = 500 and n = 2000 (a pilot observation: near-constant in n, so the CG time targets cannot be reached
by n alone within class C2W2).
Weights come from Python's random.Random(seed).randint; BISON's random-number
stream is not reproduced (the published files are not used as evaluation instances).
The paper-era generator (factor-scaled sizes of 10-60 % of a random capacity) is preserved in the paper code base.
"""
from __future__ import annotations
# ruff: noqa: I001

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin

# ---------------------------------------------------------------------------
# Problem config (Scholl data set 1; see the module docstring)
# ---------------------------------------------------------------------------

SCHOLL_CLASS = {"name": "scholl1-C2W2", "capacity": 120, "weight_range": (20, 100)}

DEFAULT_CONFIG = {
    "bin_unit": "volume_units",
    "item_name_prefix": "Shipment_",
}

# Number of items per size label: Scholl N3, N4 and 2 x N4 (round 2; see the module docstring).
SIZE_SCALING = {
    "small": {"n_items": 200},
    "medium": {"n_items": 500},
    "large": {"n_items": 1000},
}


@dataclass
class BPItem(JsonDataMixin):
    id: int
    name: str
    size: int

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.size = int(self.size)
        if self.size <= 0:
            raise ValueError("item size must be positive")
        if not isinstance(self.name, str):
            raise TypeError("item name must be a string")


@dataclass
class BinPackingInstance(JsonDataMixin):
    """1D bin packing instance."""

    instance_size: str
    bin_capacity: int
    bin_unit: str
    items: list[BPItem]

    def __post_init__(self) -> None:
        self.instance_size = str(self.instance_size)
        self.bin_capacity = int(self.bin_capacity)
        self.bin_unit = str(self.bin_unit)
        if self.items and isinstance(self.items[0], dict):
            self.items = [BPItem(**it) for it in self.items]
        if self.bin_capacity <= 0:
            raise ValueError("bin_capacity must be positive")
        for it in self.items:
            if it.size > self.bin_capacity:
                raise ValueError(f"item {it.id} size {it.size} exceeds capacity {self.bin_capacity}")

    @classmethod
    def from_json_file(cls, path: str | Path) -> BinPackingInstance:
        return cast("BinPackingInstance", super().from_json_file(path))


def first_fit_decreasing_bins(sizes: list[int], demands: list[int], capacity: int) -> int:
    """Number of bins used by first-fit decreasing: a valid upper bound on the optimum."""
    loads: list[int] = []
    for size, count in sorted(zip(sizes, demands), reverse=True):
        for _ in range(count):
            for b, load in enumerate(loads):
                if load + size <= capacity:
                    loads[b] += size
                    break
            else:
                loads.append(size)
    return len(loads)


class BinPackingGenerator(BaseInstanceGenerator[BinPackingInstance]):
    """Scholl data set 1 bin packing generator (SCHOLL_CLASS); n is set by the size label."""

    def __init__(self, *, size_label: str = "medium", seed: int = 0) -> None:
        self.default_size_label = size_label
        self.default_seed = seed

    def generate_instance(
        self,
        *,
        seed: int | None = None,
        size_label: str | None = None,
    ) -> BinPackingInstance:
        size = (size_label or self.default_size_label).lower()
        if size not in SIZE_SCALING:
            raise ValueError(f"Unknown size_label '{size}'. Expected one of {list(SIZE_SCALING)}.")

        cfg = DEFAULT_CONFIG
        rng = random.Random(seed if seed is not None else self.default_seed)
        w_lo, w_hi = SCHOLL_CLASS["weight_range"]
        items = [
            BPItem(id=i, name=f"{cfg['item_name_prefix']}{i}", size=rng.randint(w_lo, w_hi))
            for i in range(SIZE_SCALING[size]["n_items"])
        ]
        return BinPackingInstance(
            instance_size=size,
            bin_capacity=SCHOLL_CLASS["capacity"],
            bin_unit=cfg["bin_unit"],
            items=items,
        )


def recipe_record(seed: int, size: str) -> dict[str, Any]:
    """Recipe metadata of a benchmark instance (for meta.json)."""
    return {"recipe": SCHOLL_CLASS["name"], "bin_capacity": SCHOLL_CLASS["capacity"],
            "weight_range": list(SCHOLL_CLASS["weight_range"]), "n_items": SIZE_SCALING[size]["n_items"]}


def generate_bp_instance(
    seed: int,
    size: str = "medium",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience wrapper matching the benchmark pipeline."""
    gen = BinPackingGenerator(size_label=size, seed=seed)
    inst = gen.generate_instance(seed=seed, size_label=size)
    data = inst.to_dict()

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")

    return data
