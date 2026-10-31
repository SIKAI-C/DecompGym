"""Planning and scheduling instances for the logic-based Benders (LBBD) family of the scale-up study.

Problem (Hooker 2004, 2005, 2007; Cire, Coban and Hooker 2016, KER 31(5):440-451, Section 4): assign each of n jobs
to one of m facilities (machines) and schedule the jobs of every facility so that
  - job j starts at an integer time s_j with r_j <= s_j and s_j + p_ij <= d_j on its facility i (release date,
    deadline, facility-dependent processing time p_ij >= 1),
  - on facility i the jobs in process at any time consume at most C_i units of a renewable resource (job j consumes
    c_ij while it runs, on [s_j, s_j + p_ij): cumulative scheduling),
minimizing the assignment cost sum_j f_{a(j) j}. All data are integers, so integer start times lose nothing (a
feasible schedule can be left-shifted until every start is a release date or a completion time), and the cost is an
integer.

Instances
- Hooker's published sets (http://public.tepper.cmu.edu/jnh/planningandscheduling.zip, OPL .dat files: `machjob` =
  per facility the triples <p, c, f> of every job, `Limit` = C_i, `job` = <r_j, d_j>): class c (10-38 jobs, 2-4
  facilities, common window [0, d]; facility i is about m - i + 1 times slower and i times cheaper), class e
  (n = 5m jobs, 2-10 facilities, window [0, 33], speeds within a factor 1.7), classes de/df (3 facilities, different
  deadlines / release dates). `parse_hooker_dat` reads them; `convert` writes our JSON.
- `generate`: larger instances following the published recipes, with the ranges read off the published files
  (documented per class below; half-open integer ranges [lo, hi)):
    c: p_ij in [k, 10k) with k = m - i (0-based i), c_j in [1, 10) equal on all facilities, f_ij in [10(i+1), 20(i+1)),
       C_i = 10, r_j = 0, d_j = ceil(1.65 (m + 1) n / m) (reproduces 165 of the 180 published c deadlines exactly,
       the others within one unit);
    e: p_ij in [2, P_i] with P_i = round(24 - 10 i / (m - 1)) (reproduces the largest published p of every facility
       for m = 4-10), f_ij in [L_i, 2 L_i) with L_i = round(16 + 11 i / (m - 1)) (the published m = 10 ranges; for
       other m an interpolation), c_j in [1, 10), C_i = 10, r_j = 0, d_j = 33 (the published e instances have n = 5m;
       the generator takes any n);
  numpy default_rng(seed), drawn facility by facility (p, then f), then c. Generated instances have no literature
  values.

File format (JSON, `write_instance`): {"format": "ps-v1", "name": str, "capacity": [m], "release": [n],
"deadline": [n], "proc": [[n] * m], "rate": [[n] * m], "cost": [[n] * m], "meta": {...}} (all integers).

Solutions (every method publishes this, the runner and the verifier check it): a list of n pairs [facility, start],
one per job. `schedule_cost` returns the cost of a feasible solution and None otherwise (exact integer checks).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class PSInstance:
    proc: np.ndarray      # (m, n) int, p_ij >= 1
    rate: np.ndarray      # (m, n) int, c_ij >= 0
    cost: np.ndarray      # (m, n) int, f_ij >= 0
    capacity: np.ndarray  # (m,) int, C_i >= 1
    release: np.ndarray   # (n,) int, r_j >= 0
    deadline: np.ndarray  # (n,) int, d_j > r_j
    name: str = field(default="")
    meta: dict = field(default_factory=dict)

    @property
    def m(self) -> int:
        return self.proc.shape[0]

    @property
    def n(self) -> int:
        return self.proc.shape[1]

    @property
    def horizon(self) -> int:
        return int(self.deadline.max())

    @property
    def eligible(self) -> np.ndarray:
        """(m, n) bool: job j fits its window and the capacity on facility i."""
        return (self.proc <= (self.deadline - self.release)[None, :]) & (self.rate <= self.capacity[:, None])

    @property
    def energy(self) -> np.ndarray:
        """(m, n) p_ij c_ij."""
        return self.proc * self.rate

    def validate(self) -> None:
        m, n = self.proc.shape
        if m < 1 or n < 1:
            raise ValueError("empty instance")
        for name, arr, shape in (("rate", self.rate, (m, n)), ("cost", self.cost, (m, n)), ("capacity", self.capacity, (m,)),
                                 ("release", self.release, (n,)), ("deadline", self.deadline, (n,))):
            if arr.shape != shape:
                raise ValueError(f"{name}: shape {arr.shape}, expected {shape}")
        for name in ("proc", "rate", "cost", "capacity", "release", "deadline"):
            arr = getattr(self, name)
            if arr.dtype.kind not in "iu":
                raise ValueError(f"{name}: integer data required")
        if np.any(self.proc < 1) or np.any(self.rate < 0) or np.any(self.cost < 0) or np.any(self.capacity < 1):
            raise ValueError("p >= 1, c >= 0, f >= 0 and C >= 1 required")
        if np.any(self.release < 0) or np.any(self.deadline <= self.release):
            raise ValueError("0 <= r_j < d_j required")
        # A job that fits on no facility makes the instance infeasible; that is allowed (methods report it).


def _ints(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind == "f":
        if not np.all(arr == np.round(arr)):
            raise ValueError("integer data required")
        arr = arr.astype(np.int64)
    return arr.astype(np.int64)


# ---------------------------------------------------------------- Hooker's OPL .dat files

def parse_hooker_dat(text: str, name: str = "") -> PSInstance:
    """Hooker's OPL data file (see the module docstring)."""
    n = int(re.search(r"JobCount\s*=\s*(\d+)", text).group(1))
    m = int(re.search(r"MachineCount\s*=\s*(\d+)", text).group(1))
    block = text[text.index("machjob"):text.index("Limit")]
    rows = re.findall(r"\[([^\[\]]*)\]", block)
    trip = [[[int(v) for v in t.split()] for t in re.findall(r"<([^<>]*)>", row)] for row in rows]
    if len(trip) != m or any(len(r) != n or any(len(t) != 3 for t in r) for r in trip):
        raise ValueError(f"{name}: machjob does not have {m} rows of {n} triples")
    limit = [int(v) for v in re.search(r"Limit\s*=\s*\[([^\]]*)\]", text).group(1).split()]
    jobs = [[int(v) for v in t.split()] for t in re.findall(r"<([^<>]*)>", text[text.index("job ="):])]
    if len(limit) != m or len(jobs) != n or any(len(t) != 2 for t in jobs):
        raise ValueError(f"{name}: Limit or job list has the wrong length")
    arr = np.array(trip, dtype=np.int64)  # (m, n, 3)
    inst = PSInstance(proc=arr[:, :, 0].copy(), rate=arr[:, :, 1].copy(), cost=arr[:, :, 2].copy(),
                      capacity=np.array(limit, dtype=np.int64), release=np.array([t[0] for t in jobs], dtype=np.int64),
                      deadline=np.array([t[1] for t in jobs], dtype=np.int64), name=name,
                      meta={"source": "Hooker planning and scheduling instances (planningandscheduling.zip)", "file": f"{name}.dat"})
    inst.validate()
    return inst


def load_hooker_dat(path: str | Path) -> PSInstance:
    return parse_hooker_dat(Path(path).read_text(), name=Path(path).stem)


# ---------------------------------------------------------------- generator

def generate(kind: str, n: int, m: int, seed: int) -> PSInstance:
    """Instances following Hooker's published recipes (see the module docstring)."""
    rng = np.random.default_rng(seed)
    p = np.zeros((m, n), dtype=np.int64)
    f = np.zeros((m, n), dtype=np.int64)
    if kind == "c":
        for i in range(m):
            k = m - i
            p[i] = rng.integers(k, 10 * k, size=n)
            f[i] = rng.integers(10 * (i + 1), 20 * (i + 1), size=n)
        d = math.ceil(1.65 * (m + 1) * n / m - 1e-9)
    elif kind == "e":
        if m < 2:
            raise ValueError("class e needs m >= 2")
        for i in range(m):
            top = math.floor(24 - 10 * i / (m - 1) + 0.5)
            low = math.floor(16 + 11 * i / (m - 1) + 0.5)
            p[i] = rng.integers(2, top + 1, size=n)
            f[i] = rng.integers(low, 2 * low, size=n)
        d = 33
    else:
        raise ValueError(f"unknown class {kind!r}")
    c = rng.integers(1, 10, size=n)
    inst = PSInstance(proc=p, rate=np.tile(c, (m, 1)), cost=f, capacity=np.full(m, 10, dtype=np.int64),
                      release=np.zeros(n, dtype=np.int64), deadline=np.full(n, d, dtype=np.int64),
                      name=f"g{kind}{n}j{m}m_s{seed}",
                      meta={"generator": f"a_data.generate (Hooker class-{kind} recipe, ranges from the published files)",
                            "kind": kind, "n": n, "m": m, "seed": seed})
    inst.validate()
    return inst


def random_instance(rng: np.random.Generator, n: int, m: int, horizon: int = 10, pmax: int = 5, capacity: int = 4,
                    windows: bool = True) -> PSInstance:
    """A small random instance for tests (short horizons, so brute force over start times is cheap). Every job fits on
    at least one facility."""
    while True:
        p = rng.integers(1, pmax + 1, size=(m, n))
        c = rng.integers(1, capacity + 1, size=(m, n))
        f = rng.integers(1, 30, size=(m, n))
        if windows:
            r = rng.integers(0, max(1, horizon // 3), size=n)
            d = np.minimum(horizon, r + rng.integers(2, horizon + 1, size=n))
            d = np.maximum(d, r + 1)
        else:
            r, d = np.zeros(n, dtype=np.int64), np.full(n, horizon, dtype=np.int64)
        cap = np.full(m, capacity, dtype=np.int64)
        inst = PSInstance(proc=p.astype(np.int64), rate=c.astype(np.int64), cost=f.astype(np.int64), capacity=cap,
                          release=np.asarray(r, dtype=np.int64), deadline=np.asarray(d, dtype=np.int64),
                          name=f"rand{n}j{m}m", meta={"generator": "random_instance (tests)"})
        if inst.eligible.any(axis=0).all():
            inst.validate()
            return inst


# ---------------------------------------------------------------- JSON files

def instance_dict(inst: PSInstance) -> dict:
    return {"format": "ps-v1", "name": inst.name, "capacity": inst.capacity.tolist(), "release": inst.release.tolist(),
            "deadline": inst.deadline.tolist(), "proc": inst.proc.tolist(), "rate": inst.rate.tolist(),
            "cost": inst.cost.tolist(), "meta": inst.meta}


def write_instance(inst: PSInstance, path: str | Path) -> str:
    text = json.dumps(instance_dict(inst), separators=(",", ":"))
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def parse_instance(obj: dict, name: str = "") -> PSInstance:
    if obj.get("format") != "ps-v1":
        raise ValueError("not a ps-v1 instance")
    inst = PSInstance(proc=_ints(obj["proc"]), rate=_ints(obj["rate"]), cost=_ints(obj["cost"]),
                      capacity=_ints(obj["capacity"]), release=_ints(obj["release"]), deadline=_ints(obj["deadline"]),
                      name=obj.get("name") or name, meta=obj.get("meta", {}))
    inst.validate()
    return inst


def load_instance(path: str | Path) -> PSInstance:
    path = Path(path)
    if path.suffix == ".dat":
        return load_hooker_dat(path)
    return parse_instance(json.loads(path.read_text()), name=path.stem)


# ---------------------------------------------------------------- solutions

def normalize_solution(sol: Iterable) -> list[list[int]]:
    out = []
    for pair in sol:
        i, s = pair
        if isinstance(i, bool) or isinstance(s, bool):
            raise ValueError("facility and start must be integers")
        if float(i) != int(i) or float(s) != int(s):
            raise ValueError("facility and start must be integers")
        out.append([int(i), int(s)])
    return out


def machine_load_ok(inst: PSInstance, i: int, jobs: list[int], starts: list[int]) -> bool:
    """Windows and the cumulative resource of facility i for the given jobs and integer starts (exact)."""
    events: list[tuple[int, int]] = []
    for j, s in zip(jobs, starts):
        p = int(inst.proc[i, j])
        if s < inst.release[j] or s + p > inst.deadline[j] or inst.rate[i, j] > inst.capacity[i]:
            return False
        events.append((s, int(inst.rate[i, j])))
        events.append((s + p, -int(inst.rate[i, j])))
    events.sort()  # at equal times releases (negative) come first: [s, s + p) intervals
    load = 0
    for _, delta in events:
        load += delta
        if load > inst.capacity[i]:
            return False
    return True


def schedule_cost(inst: PSInstance, sol: Iterable) -> float | None:
    """Cost of a feasible solution (list of [facility, start] per job), None if infeasible or malformed."""
    try:
        pairs = normalize_solution(sol)
    except (TypeError, ValueError):
        return None
    if len(pairs) != inst.n:
        return None
    per: dict[int, tuple[list[int], list[int]]] = {}
    total = 0
    for j, (i, s) in enumerate(pairs):
        if not 0 <= i < inst.m:
            return None
        per.setdefault(i, ([], []))
        per[i][0].append(j)
        per[i][1].append(s)
        total += int(inst.cost[i, j])
    for i, (jobs, starts) in per.items():
        if not machine_load_ok(inst, i, jobs, starts):
            return None
    return float(total)


def assignment_cost(inst: PSInstance, assign: Iterable[int]) -> int:
    return int(sum(int(inst.cost[i, j]) for j, i in enumerate(assign)))


def convert(src_dir: str | Path, out_dir: str | Path, pattern: str = "*.dat") -> list[dict]:
    """Convert Hooker .dat files to ps-v1 JSON; returns [{name, sha256}] (the odd generator-config file is skipped)."""
    out = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(src_dir).glob(pattern)):
        text = path.read_text()
        if "JobCount" not in text:
            continue
        inst = parse_hooker_dat(text, name=path.stem)
        inst.meta["dat_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        out.append({"name": inst.name, "sha256": write_instance(inst, Path(out_dir) / f"{inst.name}.json")})
    return out


def main() -> None:
    """python -m src.bd.lbbd.a_data --hooker <dat dir> --out <dir>   (convert the published instances)
       python -m src.bd.lbbd.a_data --gen e 100 20 1 --out <dir>       (kind n m seed; repeatable)"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hooker", default=None)
    ap.add_argument("--gen", nargs=4, action="append", default=[], metavar=("KIND", "N", "M", "SEED"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.hooker:
        for row in convert(a.hooker, a.out):
            print(json.dumps(row))
    for kind, n, m, seed in a.gen:
        inst = generate(kind, int(n), int(m), int(seed))
        print(json.dumps({"name": inst.name, "sha256": write_instance(inst, Path(a.out) / f"{inst.name}.json")}))


if __name__ == "__main__":
    main()
