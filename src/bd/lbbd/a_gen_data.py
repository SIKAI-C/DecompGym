"""Task assignment + single-machine scheduling instances for logic-based Benders.

n tasks are assigned to m unrelated machines; each machine processes one task at a time without
preemption. Task j has a release time r_j, a due date d_j and a tardiness weight w_j; on machine i
it takes p_ij time units and costs c_ij. Objective: total processing cost plus weighted tardiness.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


SIZE_CONFIG = {
    "small": {"n_tasks_range": (8, 10), "n_machines": 2},
    "medium": {"n_tasks_range": (11, 12), "n_machines": 3},
    "large": {"n_tasks_range": (13, 15), "n_machines": 3},
}

DEFAULT_CONFIG = {
    "base_time_range": (3, 9),
    "speed_factors": (0.7, 1.0, 1.4),  # machine i takes speed * base time; faster machines cost more
    "cost_per_unit_range": (2, 4),
    "release_spread": 0.5,  # releases spread over this share of the per-machine workload
    "due_slack_range": (1.0, 2.5),  # due date = release + slack * base time
    "weight_range": (1, 5),
}


@dataclass
class LBBDInstance(JsonDataMixin):
    instance_size: str
    n_tasks: int
    n_machines: int
    release: list[int]
    due: list[int]
    weight: list[int]
    proc_time: list[list[int]]  # proc_time[i][j]: machine i, task j
    proc_cost: list[list[int]]  # proc_cost[i][j]

    def __post_init__(self) -> None:
        if len(self.release) != self.n_tasks or len(self.due) != self.n_tasks or len(self.weight) != self.n_tasks:
            raise ValueError("task vectors must have length n_tasks")
        if len(self.proc_time) != self.n_machines or any(len(row) != self.n_tasks for row in self.proc_time):
            raise ValueError("proc_time must be n_machines x n_tasks")
        if any(p <= 0 for row in self.proc_time for p in row):
            raise ValueError("processing times must be positive")
        if any(w < 0 for w in self.weight):
            raise ValueError("weights must be nonnegative")

    @classmethod
    def from_json_file(cls, path: str | Path) -> LBBDInstance:
        return cast("LBBDInstance", super().from_json_file(path))

    def horizon(self) -> int:
        """Every task set has an optimal schedule that finishes by max release + total max processing time."""
        return max(self.release) + sum(max(self.proc_time[i][j] for i in range(self.n_machines)) for j in range(self.n_tasks))


class LBBDGenerator(BaseInstanceGenerator[LBBDInstance]):
    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> LBBDInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        cfg, size = DEFAULT_CONFIG, SIZE_CONFIG[size_label]
        rng = random.Random(seed)
        n = rng.randint(*size["n_tasks_range"])
        m = size["n_machines"]
        speeds = list(cfg["speed_factors"][:m]) if m <= len(cfg["speed_factors"]) else [rng.uniform(0.7, 1.4) for _ in range(m)]
        base = [rng.randint(*cfg["base_time_range"]) for _ in range(n)]
        proc_time = [[max(1, round(speeds[i] * base[j] + rng.choice((-1, 0, 1)))) for j in range(n)] for i in range(m)]
        unit = [rng.randint(*cfg["cost_per_unit_range"]) for _ in range(n)]
        # Faster machines charge more per task, so the cheapest plan overloads the slow machine.
        proc_cost = [[round(unit[j] * base[j] * (2.2 - speeds[i])) for j in range(n)] for i in range(m)]
        workload = sum(base) / m
        release = sorted(rng.randint(0, max(1, round(cfg["release_spread"] * workload))) for _ in range(n))
        rng.shuffle(release)
        due = [release[j] + max(1, round(rng.uniform(*cfg["due_slack_range"]) * base[j])) for j in range(n)]
        weight = [rng.randint(*cfg["weight_range"]) for _ in range(n)]
        return LBBDInstance(size_label, n, m, release, due, weight, proc_time, proc_cost)


def generate_lbbd_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst = LBBDGenerator().generate_instance(seed=seed, size_label=size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return inst.to_dict()
