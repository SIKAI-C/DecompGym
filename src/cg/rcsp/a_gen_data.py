"""Railway crew scheduling (RCSP): small instances cut from the van Rossum (2026) benchmark.

Problem (B.T.C. van Rossum, "Branch and Price for Railway Crew Scheduling: Benchmark Instances and Computational
Study", 2026; semantics as in the scale-up RCSP): choose a minimum-cost set of duties covering every task at least
once. A duty is a path SOURCE -> CONNECTION* -> SINK in the connection graph whose SOURCE and SINK use the same crew
base, and it must
- last at most 9.5 h (34 200 s; the duty length is the sum of the connection durations, SOURCE and SINK included);
- contain a break connection a (timeBeforeBreak(a) >= 0; a 30-min meal break) with
  before(a) = (durations of the connections before a) + timeBeforeBreak(a) <= 5.5 h (19 800 s) and
  after(a) = timeAfterBreak(a) + (durations of the connections after a) <= 5.5 h.
Cost: the sum of the connection costs (1 per second of duty time plus 28 800 on every SOURCE connection).

Instances. The benchmark's smallest instances (90+ tasks) are too large for an exact 60 s MIP, so each instance is
an induced sub-instance of one benchmark instance (instance and best-known solution files vendored in `benchmark/`,
MIT licence). A plain time window does not work: the benchmark connects only some task pairs, so most tasks of a
short window lie in no feasible duty. The truncation rule is therefore:
1. sort the duties of the benchmark's published best-known solution of the variant by the start of their first task;
2. from a uniformly random position, take consecutive duties until their tasks reach the lower end of the size
   range; the draw is repeated when the task count exceeds the upper end;
3. keep every connection of the variant between two kept tasks and every SOURCE / SINK connection of a kept task,
   with all their fields (type, break times, duration, cost); nothing is recomputed. The chosen best-known duties stay
   feasible (so every task is covered), and the other kept connections let duties recombine.
Tasks are renumbered 0..n-1 by (start, end, original index) and connections 0..m-1 in their original order; the
original indices are kept in `source`. Stations and crew bases keep their benchmark indices.
Sizes: small 12-15 tasks, medium 18-22, large 25-30. Seeds 0 and 2 are single-depot (s2_t90, s3_t172), seed 1 is
multi-depot (s2_t122).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


MAX_DUTY = 34_200  # 9.5 h
MAX_WORK = 19_800  # 5.5 h
BENCHMARK = Path(__file__).resolve().parent / "benchmark"
SOURCES = ("s2_t90", "s2_t122", "s3_t172")

SIZE_CONFIG = {
    "small": {"tasks": (12, 15)},
    "medium": {"tasks": (18, 22)},
    "large": {"tasks": (25, 30)},
}

DEFAULT_CONFIG = {"max_draws": 500}


@dataclass
class RCSPInstance(JsonDataMixin):
    instance_size: str
    variant: str
    n_tasks: int
    tasks: list[list[int]]  # [start, end, start_station, end_station, duration] per task
    stations: list[int]
    crew_bases: list[int]
    connections: list[list[Any]]  # [index, type, startBlock, endBlock, timeBeforeBreak, timeAfterBreak, duration, cost]
    source: dict[str, Any]

    def __post_init__(self) -> None:
        if len(self.tasks) != self.n_tasks:
            raise ValueError("tasks must have n_tasks entries")
        if [c[0] for c in self.connections] != list(range(len(self.connections))):
            raise ValueError("connection indices must be 0..m-1")

    @classmethod
    def from_json_file(cls, path: str | Path) -> RCSPInstance:
        return cast("RCSPInstance", super().from_json_file(path))

    def graph(self) -> dict[str, dict]:
        """Connection lookups: source[(base, task)], conn[(task, task)], sink[(task, base)] -> connection row."""
        out: dict[str, dict] = {"SOURCE": {}, "CONNECTION": {}, "SINK": {}}
        for c in self.connections:
            out[c[1]][(c[2], c[3])] = c
        return out


def duty_cost(chain: list[list[Any]]) -> int | None:
    """Cost of a duty given as its connection rows (SOURCE ... SINK), or None if it breaks a labour rule."""
    length = sum(c[6] for c in chain)
    if length > MAX_DUTY:
        return None
    elapsed = 0
    for c in chain:
        if c[4] >= 0 and elapsed + c[4] <= MAX_WORK and c[5] + (length - elapsed - c[6]) <= MAX_WORK:
            return sum(c[7] for c in chain)
        elapsed += c[6]
    return None


def enumerate_duties(inst: RCSPInstance, limit: int = 5_000_000) -> list[tuple[int, tuple[int, ...], int]]:
    """Every feasible duty (base, task sequence, cost), by depth-first search from every SOURCE connection.

    The search only prunes on duty length (connection durations are nonnegative, so a too-long prefix stays too
    long) and on a prefix that is past 5.5 h without any break arc that could still start in time.
    """
    g = inst.graph()
    succ: dict[int, list[list[Any]]] = {}
    for (i, _), c in g["CONNECTION"].items():
        succ.setdefault(i, []).append(c)
    sinks: dict[int, list[list[Any]]] = {}
    for (i, _), c in g["SINK"].items():
        sinks.setdefault(i, []).append(c)
    out = []
    for (base, first), src in g["SOURCE"].items():
        stack = [([src], (first,), src[6], False)]
        while stack:
            chain, path, length, had_break = stack.pop()
            for snk in sinks.get(path[-1], []):
                if snk[3] == base:
                    cost = duty_cost(chain + [snk])
                    if cost is not None:
                        out.append((base, path, cost))
                        if len(out) > limit:
                            raise RuntimeError(f"more than {limit} feasible duties")
            for c in succ.get(path[-1], []):
                nl = length + c[6]
                brk = had_break or (c[4] >= 0 and length + c[4] <= MAX_WORK)
                if nl > MAX_DUTY or (not brk and nl > MAX_WORK):
                    continue
                stack.append((chain + [c], path + (c[3],), nl, brk))
    return out


def _read(name: str, variant: str) -> dict[str, Any]:
    d = BENCHMARK / name
    rows = lambda f: [line.split() for line in (d / f).read_text().splitlines() if line.strip()]
    tasks = {int(r[0]): [int(x) for x in r[1:6]] for r in rows("tasks.txt")}
    conns = [[int(r[0]), r[1], *(int(x) for x in r[2:8])] for r in rows(f"connections_{variant}_depot.txt")]
    conns.sort(key=lambda c: c[0])
    by_index = {c[0]: c for c in conns}
    best = [[by_index[int(x)][3] for x in line[:-1]] for line in rows(f"solution_{variant}_depot.txt")]
    return {"tasks": tasks, "connections": conns, "stations": sorted(int(r[0]) for r in rows("stations.txt")),
            "bases": sorted(int(r[0]) for r in rows(f"crewbases_{variant}_depot.txt")), "best_duties": best}


def cut(raw: dict[str, Any], keep: list[int], variant: str, size: str, source: dict[str, Any]) -> RCSPInstance:
    """The induced sub-instance on the kept task ids (renumbered by start, end, original id)."""
    keep = sorted(keep, key=lambda t: (raw["tasks"][t][0], raw["tasks"][t][1], t))
    new = {t: k for k, t in enumerate(keep)}
    conns = []
    for c in raw["connections"]:
        kind, a, b = c[1], c[2], c[3]
        if kind == "SOURCE" and b in new:
            row = [kind, a, new[b]]
        elif kind == "SINK" and a in new:
            row = [kind, new[a], b]
        elif kind == "CONNECTION" and a in new and b in new:
            row = [kind, new[a], new[b]]
        else:
            continue
        conns.append([len(conns), *row, *c[4:8], c[0]])
    source = {**source, "task_ids": keep, "connection_ids": [c.pop() for c in conns]}
    return RCSPInstance(size, variant, len(keep), [raw["tasks"][t] for t in keep], raw["stations"], raw["bases"], conns, source)


class RCSPGenerator(BaseInstanceGenerator[RCSPInstance]):
    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> RCSPInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        seed = 0 if seed is None else int(seed)
        rng = random.Random(f"rcsp-{size_label}-{seed}")
        name = SOURCES[seed % len(SOURCES)]
        variant = "multi" if seed % 3 == 1 else "single"
        raw = _read(name, variant)
        lo_n, hi_n = SIZE_CONFIG[size_label]["tasks"]
        duties = sorted(raw["best_duties"], key=lambda d: (raw["tasks"][d[0]][0], d))
        for draw in range(DEFAULT_CONFIG["max_draws"]):
            first = rng.randrange(len(duties))
            keep: set[int] = set()
            last = first
            while last < len(duties) and len(keep) < lo_n:
                keep |= set(duties[last])
                last += 1
            if not lo_n <= len(keep) <= hi_n:
                continue
            inst = cut(raw, sorted(keep), variant, size_label, {"benchmark": name, "best_known_duties": [first, last], "draw": draw})
            if {t for _, path, _ in enumerate_duties(inst) for t in path} != set(range(inst.n_tasks)):
                raise RuntimeError("a kept best-known duty is infeasible in the sub-instance")
            return inst
        raise RuntimeError(f"no admissible {size_label} sub-instance for seed {seed}")


def generate_rcsp_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst = RCSPGenerator().generate_instance(seed=seed, size_label=size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return inst.to_dict()
