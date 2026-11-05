"""Bin packing with conflicts: pack items into the fewest bins of one capacity; conflicting items may not share a bin.

Weights follow the classical uniform recipe (Gendreau, Laporte and Semet 2004; Muritiba et al. 2010):
weights in [20, 100], capacity 150, and a random conflict graph of the given density.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


SIZE_CONFIG = {
    "small": {"n_range": (20, 24)},
    "medium": {"n_range": (30, 35)},
    "large": {"n_range": (40, 45)},
}

DEFAULT_CONFIG = {"weight_range": (20, 100), "capacity": 150, "density_range": (0.1, 0.4)}


@dataclass
class BPPCInstance(JsonDataMixin):
    instance_size: str
    n_items: int
    capacity: int
    weights: list[int]
    conflicts: list[list[int]]

    def __post_init__(self) -> None:
        if len(self.weights) != self.n_items or any(w <= 0 or w > self.capacity for w in self.weights):
            raise ValueError("weights must be positive and fit in a bin")
        for a, b in self.conflicts:
            if not (0 <= a < self.n_items and 0 <= b < self.n_items) or a == b:
                raise ValueError("invalid conflict")

    @classmethod
    def from_json_file(cls, path: str | Path) -> BPPCInstance:
        return cast("BPPCInstance", super().from_json_file(path))


class BPPCGenerator(BaseInstanceGenerator[BPPCInstance]):
    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> BPPCInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        cfg = DEFAULT_CONFIG
        rng = random.Random(seed)
        n = rng.randint(*SIZE_CONFIG[size_label]["n_range"])
        density = rng.uniform(*cfg["density_range"])
        weights = [rng.randint(*cfg["weight_range"]) for _ in range(n)]
        conflicts = [[a, b] for a in range(n) for b in range(a + 1, n) if rng.random() < density]
        return BPPCInstance(size_label, n, cfg["capacity"], weights, conflicts)


def generate_bppc_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst = BPPCGenerator().generate_instance(seed=seed, size_label=size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return inst.to_dict()
