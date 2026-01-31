from __future__ import annotations
# ruff: noqa: I001

from dataclasses import dataclass
from pathlib import Path
import math
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin

SIZE_SCALING = {"small": 0.7, "medium": 1.0, "large": 1.1}

DEFAULT_CONFIG = {
    "T_range": (8, 12),
    "demand_min": 2,
    "demand_max": 6,
    "peak_multiplier": 1.6,
    "shift_min_length": 2,
    "shift_max_length": 5,
    "time_label_start": 8,
    "time_label_step": 1,
}


def _scale_value(val: int, scale: float, minimum: int = 1) -> int:
    return max(minimum, round(val * scale))


@dataclass
class ShiftSlot(JsonDataMixin):
    id: int
    label: str

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.label = str(self.label)


@dataclass
class ShiftInstance(JsonDataMixin):
    instance_size: str
    time_slots: list[ShiftSlot]
    demand: list[int]
    shift_min_length: int
    shift_max_length: int
    shift_cost: float

    def __post_init__(self) -> None:
        self.instance_size = str(self.instance_size)
        self.shift_min_length = int(self.shift_min_length)
        self.shift_max_length = int(self.shift_max_length)
        self.shift_cost = float(self.shift_cost)
        if len(self.demand) != len(self.time_slots):
            raise ValueError("demand length must match number of time slots")

    @classmethod
    def from_json_file(cls, path: str | Path) -> ShiftInstance:
        return cast("ShiftInstance", super().from_json_file(path))


class ShiftGenerator(BaseInstanceGenerator[ShiftInstance]):
    def __init__(self, *, size_label: str = "medium", seed: int = 0) -> None:
        self.default_size_label = size_label
        self.default_seed = seed

    def generate_instance(
        self,
        *,
        seed: int | None = None,
        size_label: str | None = None,
    ) -> ShiftInstance:
        size = (size_label or self.default_size_label).lower()
        if size not in SIZE_SCALING:
            raise ValueError(f"Unknown size_label '{size}'. Expected one of {list(SIZE_SCALING)}.")

        cfg = DEFAULT_CONFIG
        scale = SIZE_SCALING[size]
        rng = random.Random(seed if seed is not None else self.default_seed)

        T_low = _scale_value(cfg["T_range"][0], scale, minimum=6)
        T_high = _scale_value(cfg["T_range"][1], scale, minimum=T_low)
        T = rng.randint(T_low, T_high)

        demand: list[int] = []
        for t in range(T):
            base = rng.randint(cfg["demand_min"], cfg["demand_max"])
            if T // 3 <= t <= 2 * T // 3:
                base = math.ceil(base * cfg["peak_multiplier"])
            demand.append(max(1, base))

        L_min = cfg["shift_min_length"]
        L_max = min(cfg["shift_max_length"], T)
        time_slots = []
        start_hour = cfg["time_label_start"]
        step = cfg["time_label_step"]
        for t in range(T):
            h1 = start_hour + t * step
            h2 = start_hour + (t + 1) * step
            label = f"{h1:02d}:00-{h2:02d}:00"
            time_slots.append(ShiftSlot(id=t, label=label))

        return ShiftInstance(
            instance_size=size,
            time_slots=time_slots,
            demand=demand,
            shift_min_length=L_min,
            shift_max_length=L_max,
            shift_cost=1.0,
        )


def generate_shift_instance(
    seed: int,
    size: str = "medium",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    gen = ShiftGenerator(size_label=size, seed=seed)
    inst = gen.generate_instance(seed=seed, size_label=size)
    data = inst.to_dict()
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return data
