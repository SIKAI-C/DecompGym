"""Planning and scheduling with a cumulative resource (Hooker 2004, 2007; Cire, Coban and Hooker 2016), small scale.

Assign each of n jobs to one of m facilities and schedule every facility's jobs: job j starts at an integer time s_j
with r_j <= s_j and s_j + p_ij <= d_j on its facility i; while it runs, on [s_j, s_j + p_ij), it consumes c_ij units of
the facility's renewable resource, and the jobs in process on facility i never consume more than C_i at any time.
Minimize the assignment cost sum_j f_{a(j) j}. All data are integers.

Recipe: Hooker's published generator for the minimum-cost classes c and e.
  Sources: J. N. Hooker, "Planning and scheduling by logic-based Benders decomposition", Operations Research 55(3):
  588-602 (2007), Section 9 "Problem Generation"; the instance page https://johnhooker.tepper.cmu.edu/instances.htm
  ("Problem generation" table, footnote D(a) = 5aN(M+1)/M); the instance archive
  http://public.tepper.cmu.edu/jnh/planningandscheduling.zip (OPL .dat files cNjMmK.dat, eNjMmK.dat).
  Common to both classes: C_i = 10; c_ij = c_j drawn once per job on [1, 10] and equal on every facility; r_j = 0;
  one common deadline d.
  class c (archive: m = 2, 3, 4; n = 10, 12, ..., 38 / 32 / 32): facility speeds differ by up to a factor m, faster
    facilities cost more. With rows r = 0..m-1 in archive order (row 0 slowest) and k = m - r (Hooker's facility
    number i = k):  p_rj on [k, 10k],  f_rj on [10(r + 1), 20(r + 1)],  d = 5 alpha n (m + 1) / m with alpha = 1/3,
    rounded to the nearest integer (Hooker 2007: "the deadline for every task was set to alpha L (rounded to the
    nearest integer), where L = 5n(m + 1)/m. We used alpha = 1/3").
  class e (archive: m = 2..10, n = 5m, plus n = 12 at m = 2): speeds within a factor 25/15; with
    P2_r = 25 - 10 r / (m - 1) (the page's 25 - (i - 1)(10/(M - 1)), row r = i - 1, row 0 slowest):
    p_rj on [2, P2_r],  f_rj on [400 / P2_r, 800 / P2_r],  d = 33 for n = 5m (the page); the archive's e12j2m* files
    have d = 40, so d = 33 n / (5m) rounded, which reproduces every e file.
Integer conventions: an archive-compatible reconstruction of the published recipes. Every value of every c and e
archive file (195 c, 50 e) lies in these ranges, windows and capacities match exactly, and every range endpoint is
attained somewhere in the archive except one (e, m = 3, row 0: p tops at 23 of 24 in the five e15j3m files); checked
in tests/test_harder_lbbd_ps.py when the archive is present: a uniform draw on
[lo, hi] is an integer in the half-open range [lo, hi) (no c file has p = 10k or c = 10, no e file has c = 10), with
  c: p_rj in [k, 10k), f_rj in [10(r + 1), 20(r + 1)), c_j in [1, 10);
  e: p_rj in [2, round(P2_r)), f_rj in [L_r, 2 L_r) with L_r = round(400 / P2_r) for m <= 8 and
     L_r = round(16 + 11 r / (m - 1)) for m = 9, 10 (the archive's m = 9, 10 files use this linear spacing instead);
     round = half up;
  d (class e) = round(6.6 n / m);
  d (class c) = nearest integer to 5n(m + 1)/(3m), ties rounded DOWN (the archive's c18j4m*, c30j4m* have
     d = 37, 62 for 37.5, 62.5; the cause in Hooker's code is unknown, the archive fixes the rule).
Deviation from the printed recipe, in favour of the archive: the instance page, OR 2007 Section 9 and CP 2004 print
the class-c cost range [2(M - i + 1), 20(M - i + 1)]; every archive c file has f in [10(M - i + 1), 20(M - i + 1)).
The class-e deadline 33 is printed for n = 5m; round(6.6 n / m) reproduces the archive (incl. e12j2m, d = 40) and
is an extrapolation for other n (not used by SIZE_CONFIG).
Random stream: numpy default_rng(seed), facility by facility (p, then f), then c (the scale-up generator's order).
Classes de/df (different deadlines / release dates, m = 3) are not mimicked: their archive files do not follow one
fixed rule (de: the facility-0 ranges change between job counts; df: the five df20j3m* files hold 18 jobs). Even
seeds use class c, odd seeds class e.
Small-scale adaptation: only (class, n, m) combinations that occur in the archive are generated (SIZE_CONFIG); the
archive's sizes are kept, so no size is shrunk below the published ones.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


# (n_jobs, n_facilities) per class; every pair is an archive size. Calibrated on pilot seeds >= 100 (time-indexed
# MIP, 1 thread, 60 s): c14j3m 20 pilots 0.02-1.6 s; c16j4m 26 pilots 0.05-1.0 s; e20j4m 6 pilots 0.2-3.5 s;
# e25j5m 19 pilots 0.2-16.2 s. Rejected for heavy tails: c18j4m (1 of 20 pilots unproved at 60 s), c20j4m (14 pilots,
# 1 unproved, 1 at 51 s; evaluation seed 0 took 59 s in a first generation), c24j4m (3 of 6 unproved), e30j6m (1 of 6).
SIZE_CONFIG = {
    "small": {"c": (10, 2), "e": (10, 2)},
    "medium": {"c": (14, 3), "e": (20, 4)},
    "large": {"c": (16, 4), "e": (25, 5)},
}


@dataclass
class PSInstance(JsonDataMixin):
    instance_size: str
    kind: str
    n_jobs: int
    n_facilities: int
    capacity: list[int]  # C_i
    release: list[int]  # r_j
    deadline: list[int]  # d_j
    proc: list[list[int]]  # p_ij, [facility][job]
    rate: list[list[int]]  # c_ij, [facility][job]
    cost: list[list[int]]  # f_ij, [facility][job]

    def __post_init__(self) -> None:
        m, n = self.n_facilities, self.n_jobs
        if len(self.capacity) != m or len(self.release) != n or len(self.deadline) != n:
            raise ValueError("capacity needs n_facilities entries; release and deadline need n_jobs entries")
        for name in ("proc", "rate", "cost"):
            if len(getattr(self, name)) != m or any(len(row) != n for row in getattr(self, name)):
                raise ValueError(f"{name} must be n_facilities x n_jobs")
        if any(p < 1 for row in self.proc for p in row) or any(c < 0 for row in self.rate for c in row) or any(C < 1 for C in self.capacity):
            raise ValueError("p >= 1, c >= 0 and C >= 1 required")
        if any(r < 0 or d <= r for r, d in zip(self.release, self.deadline)):
            raise ValueError("0 <= r_j < d_j required")

    @classmethod
    def from_json_file(cls, path: str | Path) -> PSInstance:
        return cast("PSInstance", super().from_json_file(path))

    def eligible(self, i: int, j: int) -> bool:
        """Job j fits its window and the capacity on facility i."""
        return self.proc[i][j] <= self.deadline[j] - self.release[j] and self.rate[i][j] <= self.capacity[i]

    @property
    def horizon(self) -> int:
        return max(self.deadline)


def _half_up(x: float) -> int:
    return math.floor(x + 0.5)


def class_ranges(kind: str, n: int, m: int) -> tuple[list[tuple[int, int, int, int]], int]:
    """Per facility row r (archive order): half-open integer ranges (p_lo, p_hi, f_lo, f_hi), and the common deadline."""
    if kind == "c":
        rows = [(m - r, 10 * (m - r), 10 * (r + 1), 20 * (r + 1)) for r in range(m)]
        return rows, math.ceil(Fraction(5 * n * (m + 1), 3 * m) - Fraction(1, 2))  # nearest integer, ties down
    if kind == "e":
        if m < 2:
            raise ValueError("class e needs m >= 2")
        rows = []
        for r in range(m):
            top = 25 - 10 * r / (m - 1)
            low = _half_up(400 / top) if m <= 8 else _half_up(16 + 11 * r / (m - 1))
            rows.append((2, _half_up(top), low, 2 * low))
        return rows, _half_up(6.6 * n / m)
    raise ValueError(f"unknown class {kind!r}")


def generate(kind: str, n: int, m: int, seed: int) -> dict[str, Any]:
    """Hooker's class-c / class-e recipe (see the module docstring), returning plain integer lists."""
    rows, d = class_ranges(kind, n, m)
    rng = np.random.default_rng(seed)
    p = np.zeros((m, n), dtype=np.int64)
    f = np.zeros((m, n), dtype=np.int64)
    for i, (plo, phi, flo, fhi) in enumerate(rows):
        p[i] = rng.integers(plo, phi, size=n)
        f[i] = rng.integers(flo, fhi, size=n)
    c = rng.integers(1, 10, size=n)
    return {"kind": kind, "n_jobs": n, "n_facilities": m, "capacity": [10] * m, "release": [0] * n, "deadline": [d] * n,
            "proc": p.tolist(), "rate": np.tile(c, (m, 1)).tolist(), "cost": f.tolist()}


class PSGenerator(BaseInstanceGenerator[PSInstance]):
    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> PSInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        seed = 0 if seed is None else int(seed)
        kind = "c" if seed % 2 == 0 else "e"
        n, m = SIZE_CONFIG[size_label][kind]
        return PSInstance(instance_size=size_label, **generate(kind, n, m, seed))


def generate_lbbd_ps_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst = PSGenerator().generate_instance(seed=seed, size_label=size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return inst.to_dict()
