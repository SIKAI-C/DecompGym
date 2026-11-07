"""Generalized assignment (minimization) instances generated with the Chu-Beasley / Yagiura type D and E recipes.

Sources:
  - P. C. Chu, J. E. Beasley (1997), "A genetic algorithm for the generalised assignment problem", Computers &
    Operations Research 24(1):17-23 (types A-D; OR-Library gapinfo, https://people.brunel.ac.uk/~mastjjb/jeb/orlib/gapinfo.html);
  - M. Laguna, J. P. Kelly, J. L. Gonzalez-Velarde, F. Glover (1995), EJOR 81:176-189 (type E);
  - M. Yagiura, T. Ibaraki, F. Glover (2004), INFORMS JOC 16(2):133-151, and (2006) EJOR 169:548-569 (types D and E
    as minimization problems; instance archive https://www-or.amp.i.kyoto-u.ac.jp/~yagiura/gap/).
Notation here: machines j (Yagiura's agents), jobs i, resource a[i][j], cost c[i][j], capacity b[j], m machines.
  - Type D: a_ij ~ U{1, ..., 100}; c_ij = 111 - a_ij + e_ij with e_ij ~ U{-10, ..., 10}.
  - Type E: a_ij = floor(1 - 10 ln x_ij), x_ij ~ U(0, 1]; c_ij = floor(1000 / a_ij - 10 y_ij), y_ij ~ U[0, 1).
  - Both: b_j = floor(0.8 * sum_i a_ij / m).
The integer conventions (floor for E's a and c and for b, integer noise for D) were checked against the published
files (Yagiura's d*/e* files as redistributed by fontanf/generalizedassignmentsolver): D: c + a - 111 spans exactly
[-10, 10]; E: a in [1, 91] with P(a = k) matching floor (not rounding), c - 1000 / a in (-11, 0]; b = floor(0.8 sum / m).
Draw order: per job i, per machine j (a_ij then the cost noise), with Python's random.Random.

Adaptations (named): (1) size: Yagiura's D/E sets have n >= 100 and m in {5, ..., 80}; here the
published ratio n/m = 5 (e.g. (m, n) = (20, 100)) is kept and m is shrunk (SIZE_SCALING): (m, n) = (8, 40) / (10, 50) /
(12, 60) (round-2 recalibration, 2026-11-05; round 1 used (4, 20) / (6, 30) / (8, 40)). The size is bounded by the proof of
the monolithic MIP (Threads = 1, TimeLimit 1800 s; src.utils.instance_suite rejects any instance where it does not prove optimality):
type D at (14, 70) and (15, 75) did not prove optimality within 1800 s on a pilot seed, while (12, 60) proved on pilot
seeds 100, 102, 103, 104 and 106 in at most about 210 s; the full-process reference CG took about 9-23 s at (12, 60)
on pilot seeds 100-104 and 106.
(2) type mix: even benchmark seeds draw type D, odd seeds type E (seeds 0-2 give D, E, D).
(3) feasibility policy: the instance with redraw index k = 0 is drawn from random.Random(f"gap-litgen:{seed}:{k}");
if it has no feasible assignment, k is increased and the instance is redrawn (the type and redraw index are recorded in meta.json).
Feasibility is decided by a greedy sufficient check and, if that fails, an exact feasibility MIP.
The paper-era generator (independent costs with machine bias, capacity 0.9 sum a) is preserved in the paper code base.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


# Small-scale sizes: (machines m, jobs n) with the published ratio n / m = 5 (round 2; see the module docstring).
SIZE_SCALING = {"small": {"m": 8, "n": 40}, "medium": {"m": 10, "n": 50}, "large": {"m": 12, "n": 60}}

GAP_TYPES = {
    "D": {"resource_range": (1, 100), "cost_base": 111, "cost_noise": 10, "capacity_share": 0.8},
    "E": {"exp_scale": 10.0, "cost_numerator": 1000.0, "cost_noise": 10.0, "capacity_share": 0.8},
}

DEFAULT_CONFIG = {
    "job_name_prefix": "Order_",
    "machine_name_prefix": "Plant_",
}


def gap_type_for_seed(seed: int) -> str:
    return "D" if seed % 2 == 0 else "E"


def draw_type_de(rng: random.Random, gap_type: str, n_jobs: int, n_machines: int) -> tuple[list[list[int]], list[list[float]], list[int]]:
    """One type D or E draw: a[i][j], c[i][j], b[j] (see the module docstring)."""
    cfg = GAP_TYPES[gap_type]
    a = [[0] * n_machines for _ in range(n_jobs)]
    c = [[0.0] * n_machines for _ in range(n_jobs)]
    for i in range(n_jobs):
        for j in range(n_machines):
            if gap_type == "D":
                a[i][j] = rng.randint(*cfg["resource_range"])
                c[i][j] = float(cfg["cost_base"] - a[i][j] + rng.randint(-cfg["cost_noise"], cfg["cost_noise"]))
            else:
                x = 1.0 - rng.random()  # U(0, 1]
                a[i][j] = math.floor(1.0 - cfg["exp_scale"] * math.log(x))
                c[i][j] = float(math.floor(cfg["cost_numerator"] / a[i][j] - cfg["cost_noise"] * rng.random()))
    b = [math.floor(cfg["capacity_share"] * sum(a[i][j] for i in range(n_jobs)) / n_machines) for j in range(n_machines)]
    return a, c, b


def _greedy_feasible(a: list[list[int]], b: list[int]) -> bool:
    """Sufficient check: jobs by decreasing minimum resource, each to the fitting machine with the most slack left."""
    left = list(b)
    for i in sorted(range(len(a)), key=lambda i: -min(a[i])):
        fits = [j for j in range(len(b)) if a[i][j] <= left[j]]
        if not fits:
            return False
        j = max(fits, key=lambda j: (left[j] - a[i][j]) / max(1, b[j]))
        left[j] -= a[i][j]
    return True


def gap_feasible(a: list[list[int]], b: list[int]) -> bool:
    """Exact feasibility: the greedy check, else a feasibility MIP (Threads = 1)."""
    if _greedy_feasible(a, b):
        return True
    import gurobipy as gp
    from gurobipy import GRB

    with gp.Env(params={"OutputFlag": 0, "Threads": 1, "TimeLimit": 30.0}) as env, gp.Model("gap_feasibility", env=env) as model:
        n, m = len(a), len(b)
        x = model.addVars(n, m, vtype=GRB.BINARY)
        model.addConstrs(x.sum(i, "*") == 1 for i in range(n))
        model.addConstrs(gp.quicksum(a[i][j] * x[i, j] for i in range(n)) <= b[j] for j in range(m))
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            return True
        if model.Status == GRB.INFEASIBLE:
            return False
        raise RuntimeError(f"GAP feasibility check undecided (status {model.Status})")


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
    machines: list[GAPMachine]
    jobs: list[GAPJob]

    def __post_init__(self) -> None:
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
        bench_seed = seed if seed is not None else self.default_seed
        n_machines, n_jobs = SIZE_SCALING[size]["m"], SIZE_SCALING[size]["n"]
        gap_type = gap_type_for_seed(bench_seed)
        for k in range(1000):
            rng = random.Random(f"gap-litgen:{bench_seed}:{k}")
            a, c, b = draw_type_de(rng, gap_type, n_jobs, n_machines)
            if gap_feasible(a, b):
                break
        else:
            raise RuntimeError("no feasible GAP draw within 1000 redraws")

        machines = [GAPMachine(id=j, name=f"{cfg['machine_name_prefix']}{j}", capacity=b[j]) for j in range(n_machines)]
        jobs = [
            GAPJob(id=i, name=f"{cfg['job_name_prefix']}{i}",
                   assignments=[GAPAssignment(machine_id=j, resource=a[i][j], cost=c[i][j]) for j in range(n_machines)])
            for i in range(n_jobs)
        ]
        self.last_recipe = {"recipe": "yagiura-type-" + gap_type, "gap_type": gap_type, "redraw_index": k,
                            "n_machines": n_machines, "n_jobs": n_jobs}
        return GAPInstance(machines=machines, jobs=jobs)


def generate_gap_instance_with_recipe(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """(data, recipe metadata) from one generation; the recipe goes to meta.json, never to data.json."""
    gen = GAPGenerator(size_label=size, seed=seed)
    inst = gen.generate_instance(seed=seed, size_label=size)
    data = inst.to_dict()
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return data, gen.last_recipe


def generate_gap_instance(
    seed: int,
    size: str = "medium",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    return generate_gap_instance_with_recipe(seed, size, output_dir)[0]


# Extra-large scale-up tier (exploration).
# Chu-Beasley type D: resource and cost are inversely correlated and capacities are tight,
# which keeps the instances hard for monolithic MIP.
XL_CONFIG = {"n_machines_range": (10, 20), "jobs_per_machine": (10, 20), "resource_range": (1, 100), "cost_noise": 10, "capacity_share": 0.8}


def generate_gap_xl_instance(seed: int, output_dir: str | Path | None = None) -> dict[str, Any]:
    cfg = XL_CONFIG
    rng = random.Random(seed)
    n_machines = rng.randint(*cfg["n_machines_range"])
    n_jobs = n_machines * rng.randint(*cfg["jobs_per_machine"])
    a = [[rng.randint(*cfg["resource_range"]) for _ in range(n_machines)] for _ in range(n_jobs)]
    c = [[float(111 - a[i][j] + rng.randint(-cfg["cost_noise"], cfg["cost_noise"])) for j in range(n_machines)] for i in range(n_jobs)]
    capacities = [int(cfg["capacity_share"] * sum(a[i][j] for i in range(n_jobs)) / n_machines) for j in range(n_machines)]
    machines = [GAPMachine(id=j, name=f"Plant_{j}", capacity=capacities[j]) for j in range(n_machines)]
    jobs = [
        GAPJob(id=i, name=f"Order_{i}", assignments=[GAPAssignment(machine_id=j, resource=a[i][j], cost=c[i][j]) for j in range(n_machines)])
        for i in range(n_jobs)
    ]
    inst = GAPInstance(machines=machines, jobs=jobs)
    data = inst.to_dict()
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return data


def gap_instance_from_arrays(cost: list[list[float]], resource: list[list[int]], capacity: list[int]) -> GAPInstance:
    """Build a GAPInstance from agent-major arrays cost[k][i], resource[k][i] (k = machine, i = job)."""
    m, n = len(capacity), len(cost[0])
    machines = [GAPMachine(id=k, name=f"Plant_{k}", capacity=int(capacity[k])) for k in range(m)]
    jobs = [GAPJob(id=i, name=f"Order_{i}", assignments=[GAPAssignment(machine_id=k, resource=int(resource[k][i]), cost=float(cost[k][i])) for k in range(m)]) for i in range(n)]
    return GAPInstance(machines=machines, jobs=jobs)


def load_orlib_gap_file(path: str | Path) -> list[GAPInstance]:
    """Read an OR-Library / Yagiura GAP file: problem count (OR-Library only), then per problem m n, costs (m x n), resources (m x n), capacities (m)."""
    tokens = [int(t) for t in Path(path).read_text().split()]
    pos, out = 0, []
    # OR-Library files start with the problem count; Yagiura's single-problem files start with m n.
    count = 1
    if len(tokens) > 2 and tokens[0] < 100 and len(tokens) != 2 + 2 * tokens[0] * tokens[1] + tokens[0]:
        count, pos = tokens[0], 1
    for _ in range(count):
        m, n = tokens[pos], tokens[pos + 1]
        pos += 2
        cost = [tokens[pos + k * n: pos + (k + 1) * n] for k in range(m)]
        pos += m * n
        resource = [tokens[pos + k * n: pos + (k + 1) * n] for k in range(m)]
        pos += m * n
        capacity = tokens[pos: pos + m]
        pos += m
        out.append(gap_instance_from_arrays(cost, resource, capacity))
    if pos != len(tokens):
        raise ValueError(f"{path}: {len(tokens) - pos} trailing tokens")
    return out
