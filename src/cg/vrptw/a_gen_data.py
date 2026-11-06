"""Capacitated vehicle routing with time windows (VRPTW): random customer subsets of Solomon's 100-customer instances.

Literature construction. Solomon (1987, Operations Research 35(2):254-265) built 56 instances of 100 customers in six
classes (R1, R2, C1, C2, RC1, RC2: random / clustered / mixed geography, short / long scheduling horizon); the
customary 25- and 50-customer instances are the FIRST 25 or 50 customers of each 100-customer file, with the depot,
capacity and fleet unchanged. Exact VRPTW studies (e.g. Kohl, Desrosiers, Madsen, Solomon & Soumis 1999,
Transportation Science 33(1)) solve these subsets with distances truncated to one decimal. Files: `solomon.py`.

Our sampling protocol (a new protocol on Solomon's data, frozen here; it adapts the "first k customers" rule):
- parent file: instance slot k = 3 * size_index + seed (size_index 0/1/2 for small/medium/large) takes the class
  CLASSES[k mod 6] = (R1, R2, C1, C2, RC1, RC2)[k mod 6], so seeds 0..2 at the three sizes cover every class at
  least once (small: R1, R2, C1; medium: C2, RC1, RC2; large: R1, R2, C1); the file inside the class is drawn
  uniformly by the instance's own random stream;
- customers: a uniformly random subset of n_customers of the parent's 100 customers (without replacement), kept in
  the parent's order and renumbered 1..n; the depot is the parent's depot. Coordinates, demands, ready times, due
  dates and service times are copied unchanged (all integers), so every subset keeps its class's geography, window
  widths and horizon; the vehicle capacity is the parent's (200, 700 or 1000);
- random stream: Python's random.Random("solomon-subset/<size>/<seed>"); no instance is selected or redrawn on
  solver outcomes;
- fleet: at most the parent's 25 vehicles (data field "vehicles"; the MIP and the verifier enforce it; the
  branch-and-price master omits the fleet row, so its optimum is a relaxation's and is accepted only if it uses at
  most 25 routes, else the reference fails and the instance is not admitted - a declared rule, never a redraw);
- objective: total distance, no vehicle cost (SINTEF ranks vehicles first; that hierarchy is NOT used here, so its
  best-known values are not optima of this model; the references are recomputed);
- precision: every distance and travel time is the Euclidean distance truncated to one decimal, floor(10 d) / 10 (the
  scale-up VRPTW's DIMACS convention); windows and service times are integers, so all tests are exact in tenths;
- every sampled customer is individually serviceable (depot -> customer -> depot within all windows) in every
  Solomon file; generation asserts it.
Size adaptation: Solomon's smallest standard subset has 25 customers; n_customers depends on the parent's class
(R1 and RC1: 16 / 21 / 26; C1: 10 / 13 / 15; C2: 11 / 14 / 16; R2 and RC2: 10 / 12 / 14 for small / medium / large),
calibrated on pilot seeds 100+ (never the evaluation seeds 0-2) so that the branch-and-price reference proves
optimality within its 60 s budget with one thread (the monolithic MIP only has to prove optimality, limit 1800 s).
The classes differ because files with mostly wide windows and long routes (c103/c104 in C1; r204/r208, rc204/rc208 in
R2/RC2) make exact elementary pricing hard: c104 times out at 16 customers, r208 and rc204 at 16.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, cast

from src.cg.vrptw.solomon import CLASSES, CLASS_FILES, load_solomon
from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


SIZES = ("small", "medium", "large")
# Keyed by the parent's class (calibrated on pilot seeds 100+, the reference is the binding limit; full-process
# seconds): R1/RC1 at 26 customers <= 19 s (16 pilots; MIP <= 245 s); C1 at 15 <= 29.3 s (c104; c104 times out at 16);
# C2 at 16 <= 19.1 s; R2/RC2 at 14 <= 37.4 s (r208; r208 and rc204 time out at 16).
SIZE_CONFIG = {
    "small": {"n_customers": {"R1": 16, "RC1": 16, "C1": 10, "C2": 11, "R2": 10, "RC2": 10}},
    "medium": {"n_customers": {"R1": 21, "RC1": 21, "C1": 13, "C2": 14, "R2": 12, "RC2": 12}},
    "large": {"n_customers": {"R1": 26, "RC1": 26, "C1": 15, "C2": 16, "R2": 14, "RC2": 14}},
}


@dataclass
class VRPTWInstance(JsonDataMixin):
    instance_size: str
    n_customers: int
    capacity: int
    x: list[float]  # node coordinates, index 0 = depot
    y: list[float]
    demand: list[int]  # demand[0] = 0
    ready: list[float]
    due: list[float]
    service: list[float]
    vehicles: int  # fleet limit: at most this many routes

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Any:
        # Instances written before the fleet field existed have an unlimited fleet (at most one route per customer).
        return super().from_dict({"vehicles": payload.get("n_customers"), **payload})

    def __post_init__(self) -> None:
        n = self.n_customers + 1
        for name in ("x", "y", "demand", "ready", "due", "service"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have n_customers + 1 entries")

    @classmethod
    def from_json_file(cls, path: str | Path) -> VRPTWInstance:
        return cast("VRPTWInstance", super().from_json_file(path))

    def dist(self, i: int, j: int) -> float:
        return math.floor(10.0 * math.hypot(self.x[i] - self.x[j], self.y[i] - self.y[j])) / 10.0


def n_customers(size_label: str, parent: str) -> int:
    """The size rule: an int, or a dict keyed by the parent's Solomon class ("R1", "C1", ...), falling back to its
    horizon type ("1"/"2", r1xx -> "1") for a class without its own entry."""
    cfg = SIZE_CONFIG[size_label]["n_customers"]
    if isinstance(cfg, int):
        return cfg
    cls = next(c for c in CLASSES if parent in CLASS_FILES[c])
    return cfg[cls] if cls in cfg else cfg[parent[-3]]  # r101 -> "1", rc208 -> "2"


def solomon_parent(seed: int, size_label: str) -> tuple[str, random.Random]:
    """The declared parent schedule: (parent file name, the instance's random stream after drawing it)."""
    rng = random.Random(f"solomon-subset/{size_label}/{seed}")
    cls = CLASSES[(3 * SIZES.index(size_label) + seed) % len(CLASSES)]
    return rng.choice(CLASS_FILES[cls]), rng


def truncated(a: tuple[int, int], b: tuple[int, int]) -> int:
    """floor(10 x Euclidean distance) for integer points, exactly (integer square root)."""
    return math.isqrt(100 * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


class VRPTWGenerator(BaseInstanceGenerator[VRPTWInstance]):
    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> VRPTWInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        inst, _ = sample_subset(int(seed or 0), size_label)
        return inst


def sample_subset(seed: int, size_label: str) -> tuple[VRPTWInstance, dict[str, Any]]:
    parent, rng = solomon_parent(seed, size_label)
    sol = load_solomon(parent)
    n = n_customers(size_label, parent)
    customers = sorted(rng.sample(range(1, 101), n))
    rows = [sol.rows[0]] + [sol.rows[i] for i in customers]
    depot = rows[0]
    for x, y, q, a, b, s in rows[1:]:
        go = truncated(depot[:2], (x, y))
        if max(10 * depot[3] + go, 10 * a) > 10 * b or max(10 * depot[3] + go, 10 * a) + 10 * s + go > 10 * depot[4]:
            raise ValueError(f"{parent}: a customer cannot be served on its own")
    inst = VRPTWInstance(size_label, n, sol.capacity, *([float(r[k]) for r in rows] for k in (0, 1)), [r[2] for r in rows],
                         *([float(r[k]) for r in rows] for k in (3, 4, 5)), sol.vehicles)
    source = {"recipe": "Solomon (1987) random customer subset", "parent": parent, "parent_class": next(c for c in CLASSES if parent in CLASS_FILES[c]),
              "parent_customers": customers, "parent_vehicles": sol.vehicles, "stream": f"solomon-subset/{size_label}/{seed}"}
    return inst, source


def generate_vrptw_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst, source = sample_subset(seed, size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
        # Provenance stays out of data.json (the only file the agent sees).
        (out_dir / "source.json").write_text(json.dumps(source, indent=2))
    return inst.to_dict()
