from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


# Problem config
DEFAULT_CONFIG = {
    "n_item_types_range": (3, 6),
    "stock_size_range": (150, 250),
    "size_min_frac": 0.10,
    "size_max_frac": 0.60,
    "demand_range": (10, 60),
    "stock_unit": "cm",
    "item_name_prefix": "PaperProduct_",
}

SIZE_SCALING = {
    "small": {"n_item_factor": 0.7, "stock_factor": 0.8, "demand_factor": 0.6},
    "medium": {"n_item_factor": 1.0, "stock_factor": 1.0, "demand_factor": 1.0},
    "large": {"n_item_factor": 1.4, "stock_factor": 1.2, "demand_factor": 1.3},
}


def _scale_range(rng: tuple[int, int], factor: float, *, minimum: int = 1) -> tuple[int, int]:
    low, high = rng
    new_low = max(minimum, round(low * factor))
    new_high = max(new_low, round(high * factor))
    return new_low, new_high


@dataclass
class CSPItemType(JsonDataMixin):
    id: int
    name: str
    size: int
    demand: int

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.size = int(self.size)
        self.demand = int(self.demand)
        if self.size <= 0:
            raise ValueError("size must be positive")
        if self.demand <= 0:
            raise ValueError("demand must be positive")


@dataclass
class CuttingStockInstance(JsonDataMixin):
    instance_size: str
    stock_size: int
    stock_unit: str
    item_types: list[CSPItemType]

    def __post_init__(self) -> None:
        self.instance_size = str(self.instance_size)
        self.stock_size = int(self.stock_size)
        if self.item_types and isinstance(self.item_types[0], dict):
            self.item_types = [CSPItemType(**it) for it in self.item_types]
        if self.stock_size <= 0:
            raise ValueError("stock_size must be positive")
        for it in self.item_types:
            if it.size > self.stock_size:
                raise ValueError(f"item {it.id} width {it.size} exceeds stock_size {self.stock_size}")

    @classmethod
    def from_json_file(cls, path: str | Path) -> CuttingStockInstance:
        return cast("CuttingStockInstance", super().from_json_file(path))


class CuttingStockGenerator(BaseInstanceGenerator[CuttingStockInstance]):
    def __init__(self, *, size_label: str = "medium", seed: int = 0) -> None:
        self.default_size_label = size_label
        self.default_seed = seed

    def generate_instance(
        self,
        *,
        seed: int | None = None,
        size_label: str | None = None,
    ) -> CuttingStockInstance:
        size = (size_label or self.default_size_label).lower()
        if size not in SIZE_SCALING:
            raise ValueError(f"Unknown size_label '{size}'. Expected one of {list(SIZE_SCALING)}.")

        cfg = DEFAULT_CONFIG
        scale = SIZE_SCALING[size]
        rng = random.Random(seed if seed is not None else self.default_seed)

        n_item_types = rng.randint(*_scale_range(cfg["n_item_types_range"], scale["n_item_factor"], minimum=2))
        stock_size = rng.randint(*_scale_range(cfg["stock_size_range"], scale["stock_factor"], minimum=10))
        demand_range = _scale_range(cfg["demand_range"], scale["demand_factor"], minimum=1)

        sizes = []
        for _ in range(n_item_types):
            min_size = max(1, int(cfg["size_min_frac"] * stock_size))
            max_size = max(min_size, int(cfg["size_max_frac"] * stock_size))
            size_val = rng.randint(min_size, max_size)
            size_val = min(size_val, stock_size)
            sizes.append(size_val)

        demands = [rng.randint(*demand_range) for _ in range(n_item_types)]

        items = [
            CSPItemType(
                id=i,
                name=f"{cfg.get('item_name_prefix', 'Item_')}{i}",
                size=sizes[i],
                demand=demands[i],
            )
            for i in range(n_item_types)
        ]

        return CuttingStockInstance(
            instance_size=size,
            stock_size=stock_size,
            stock_unit=cfg.get("stock_unit", "units"),
            item_types=items,
        )


def generate_csp_instance(
    seed: int,
    size: str = "medium",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    gen = CuttingStockGenerator(size_label=size, seed=seed)
    inst = gen.generate_instance(seed=seed, size_label=size)
    data = inst.to_dict()
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return data
