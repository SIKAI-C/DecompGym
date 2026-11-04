"""Planning and scheduling with a cumulative resource (Hooker 2004, 2005, 2007; Cire, Coban and Hooker 2016), small scale.

Assign each of n jobs to one of m facilities and schedule every facility's jobs: job j starts at an integer time s_j
with r_j <= s_j and s_j + p_ij <= d_j on its facility i; while it runs, on [s_j, s_j + p_ij), it consumes c_ij units of
the facility's renewable resource, and the jobs in process on facility i never consume more than C_i at any time.
Minimize the assignment cost sum_j f_{a(j) j}. All data are integers.
Semantics and generator follow the scale-up family (DecompBench-scaleup src/bd/lbbd/a_data.py `generate`, Hooker's
published recipes, numpy default_rng(seed), drawn facility by facility (p, then f), then c):
    class c: p_ij in [k, 10k) with k = m - i, f_ij in [10(i+1), 20(i+1)), c_j in [1, 10) on every facility, C_i = 10,
             r_j = 0, d_j = ceil(1.65 (m + 1) n / m);
    class e: p_ij in [2, P_i], P_i = round(24 - 10 i / (m - 1)), f_ij in [L_i, 2 L_i), L_i = round(16 + 11 i / (m - 1)),
             c_j in [1, 10), C_i = 10, r_j = 0, d_j = 33.
Ranges are half-open unless closed brackets are written. Even seeds use class c, odd seeds class e.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


SIZE_CONFIG = {
    "small": {"n_jobs": 10, "n_facilities": 2},
    "medium": {"n_jobs": 15, "n_facilities": 3},
    "large": {"n_jobs": 18, "n_facilities": 4},
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


def generate(kind: str, n: int, m: int, seed: int) -> dict[str, Any]:
    """The scale-up generator (`a_data.generate`), returning plain integer lists."""
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
    return {"kind": kind, "n_jobs": n, "n_facilities": m, "capacity": [10] * m, "release": [0] * n, "deadline": [d] * n,
            "proc": p.tolist(), "rate": np.tile(c, (m, 1)).tolist(), "cost": f.tolist()}


class PSGenerator(BaseInstanceGenerator[PSInstance]):
    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> PSInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        seed = 0 if seed is None else int(seed)
        size = SIZE_CONFIG[size_label]
        return PSInstance(instance_size=size_label, **generate("c" if seed % 2 == 0 else "e", size["n_jobs"], size["n_facilities"], seed))


def generate_lbbd_ps_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst = PSGenerator().generate_instance(seed=seed, size_label=size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return inst.to_dict()
