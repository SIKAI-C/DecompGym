from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


SIZE_SCALING = {"small": 0.7, "medium": 1.0, "large": 1.15}


DEFAULT_CONFIG = {
    "n_jobs_range": (5, 8),
    "n_machines_range": (2, 3),
    "resource_min": 2,
    "resource_max": 8,
    "base_cost_min": 8.0,
    "base_cost_max": 18.0,
    "machine_bias_scale": 0.25,
    "capacity_tightness": 0.9,
    "job_name_prefix": "Order_",
    "machine_name_prefix": "Plant_",
}


def _scale_range(rng: tuple[int, int], scale: float, minimum: int = 1) -> tuple[int, int]:
    low, high = rng
    new_low = max(minimum, round(low * scale))
    new_high = max(new_low, round(high * scale))
    return new_low, new_high


@dataclass
class GAPAssignment(JsonDataMixin):
    machine_id: int
    resource: int
    cost: float

    def __post_init__(self) -> None:
        self.machine_id = int(self.machine_id)
        self.resource = int(self.resource)
        self.cost = float(self.cost)
        if self.resource <= 0:
            raise ValueError("resource must be positive")


@dataclass
class GAPJob(JsonDataMixin):
    id: int
    name: str
    assignments: list[GAPAssignment]

    def __post_init__(self) -> None:
        self.id = int(self.id)


@dataclass
class GAPMachine(JsonDataMixin):
    id: int
    name: str
    capacity: int

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.capacity = int(self.capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")


@dataclass
class GAPInstance(JsonDataMixin):
    instance_size: str
    machines: list[GAPMachine]
    jobs: list[GAPJob]

    def __post_init__(self) -> None:
        self.instance_size = str(self.instance_size)
        if self.machines and isinstance(self.machines[0], dict):
            self.machines = [GAPMachine(**m) for m in self.machines]
        if self.jobs and isinstance(self.jobs[0], dict):
            jobs: list[GAPJob] = []
            for job in self.jobs:
                assigns = job.get("assignments", [])
                if assigns and isinstance(assigns[0], dict):
                    job = {**job, "assignments": [GAPAssignment(**a) for a in assigns]}
                jobs.append(GAPJob(**job))
            self.jobs = jobs

    @classmethod
    def from_json_file(cls, path: str | Path) -> GAPInstance:
        return cast("GAPInstance", super().from_json_file(path))


class GAPGenerator(BaseInstanceGenerator[GAPInstance]):
    def __init__(self, *, size_label: str = "medium", seed: int = 0) -> None:
        self.default_size_label = size_label
        self.default_seed = seed

    def generate_instance(
        self,
        *,
        seed: int | None = None,
        size_label: str | None = None,
    ) -> GAPInstance:
        size = (size_label or self.default_size_label).lower()
        if size not in SIZE_SCALING:
            raise ValueError(f"Unknown size_label '{size}'. Expected one of {list(SIZE_SCALING)}.")

        cfg = DEFAULT_CONFIG
        scale = SIZE_SCALING[size]
        rng = random.Random(seed if seed is not None else self.default_seed)

        n_jobs = rng.randint(*_scale_range(cfg["n_jobs_range"], scale, minimum=2))
        n_machines = rng.randint(*_scale_range(cfg["n_machines_range"], scale, minimum=2))

        a = [[0] * n_machines for _ in range(n_jobs)]
        c = [[0.0] * n_machines for _ in range(n_jobs)]
        for i in range(n_jobs):
            base_cost = rng.uniform(cfg["base_cost_min"], cfg["base_cost_max"])
            for j in range(n_machines):
                a_ij = rng.randint(cfg["resource_min"], cfg["resource_max"])
                bias = (rng.random() - 0.5) * 2 * cfg["machine_bias_scale"] * base_cost
                a[i][j] = a_ij
                c[i][j] = round(base_cost + bias, 2)

        total_per_machine = [sum(a[i][j] for i in range(n_jobs)) for j in range(n_machines)]
        capacities = []
        for j in range(n_machines):
            base = total_per_machine[j] * cfg["capacity_tightness"]
            cap = max(
                max(a[i][j] for i in range(n_jobs)),
                int(base) + rng.randint(0, cfg["resource_max"]),
            )
            capacities.append(cap)

        machines = [
            GAPMachine(id=j, name=f"{cfg.get('machine_name_prefix', 'Machine_')}{j}", capacity=int(capacities[j]))
            for j in range(n_machines)
        ]

        jobs: list[GAPJob] = []
        for i in range(n_jobs):
            assignments = [
                GAPAssignment(machine_id=j, resource=int(a[i][j]), cost=float(c[i][j])) for j in range(n_machines)
            ]
            jobs.append(GAPJob(id=i, name=f"{cfg.get('job_name_prefix', 'Job_')}{i}", assignments=assignments))

        return GAPInstance(instance_size=size, machines=machines, jobs=jobs)


def generate_gap_instance(
    seed: int,
    size: str = "medium",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    gen = GAPGenerator(size_label=size, seed=seed)
    inst = gen.generate_instance(seed=seed, size_label=size)
    data = inst.to_dict()
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return data
