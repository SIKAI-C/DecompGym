from __future__ import annotations
# ruff: noqa: I001

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin

# ---------------------------------------------------------------------------
# Problem config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "n_items_range": (12, 25),
    "bin_capacity_range": (80, 160),
    "size_min_frac": 0.10,
    "size_max_frac": 0.60,
    "bin_unit": "volume_units",
    "item_name_prefix": "Shipment_",
}

SIZE_SCALING = {
    "small": {"n_item_factor": 0.7, "capacity_factor": 0.8},
    "medium": {"n_item_factor": 1.0, "capacity_factor": 1.0},
    "large": {"n_item_factor": 1.4, "capacity_factor": 1.2},
}


def _scale_range(rng: tuple[int, int], factor: float, *, minimum: int = 1) -> tuple[int, int]:
    low, high = rng
    new_low = max(minimum, round(low * factor))
    new_high = max(new_low, round(high * factor))
    return new_low, new_high


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


class BinPackingGenerator(BaseInstanceGenerator[BinPackingInstance]):
    """Synthetic bin packing generator (identical bins, size-labeled)."""

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
        scale = SIZE_SCALING[size]
        rng = random.Random(seed if seed is not None else self.default_seed)

        n_items_range = _scale_range(cfg["n_items_range"], scale["n_item_factor"], minimum=3)
        cap_range = _scale_range(cfg["bin_capacity_range"], scale["capacity_factor"], minimum=5)

        n_items = rng.randint(*n_items_range)
        bin_capacity = rng.randint(*cap_range)

        items: list[BPItem] = []
        for i in range(n_items):
            min_size = max(1, int(cfg["size_min_frac"] * bin_capacity))
            max_size = max(min_size, int(cfg["size_max_frac"] * bin_capacity))
            size_val = rng.randint(min_size, max_size)
            size_val = min(size_val, bin_capacity)
            items.append(BPItem(id=i, name=f"{cfg.get('item_name_prefix', 'Item_')}{i}", size=size_val))

        return BinPackingInstance(
            instance_size=size,
            bin_capacity=bin_capacity,
            bin_unit=cfg.get("bin_unit", "units"),
            items=items,
        )


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
