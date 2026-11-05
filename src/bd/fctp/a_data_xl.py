"""Fixed-charge transportation problem (FCTP) instances for the scale-up study (the paper's FCTP family in this
directory, a_gen_data.py and friends, is unchanged and unrelated to these files).

File format (Klose's ".FCTP" text format, used for the Glover, Amini and Kochenberger (2005) instances; read as in
Klose's FCTPbas.java): whitespace-separated numbers
    m n
    s_1 ... s_m                         (integer supplies)
    d_1 ... d_n                         (integer demands)
    c_11 ... c_1n c_21 ... c_mn         (unit transportation costs, arc i*n + j)
    f_11 ... f_1n f_21 ... f_mn         (fixed costs, same arc order).

Problem (balanced: sum_i s_i = sum_j d_j): choose flows x_ij >= 0 with sum_j x_ij = s_i for every supplier i and
sum_i x_ij = d_j for every customer j; an arc with x_ij > 0 pays f_ij. Objective sum_{x_ij > 0} f_ij + sum c_ij x_ij.
Every arc may be used (complete bipartite graph); its flow is at most m_ij = min(s_i, d_j) in any feasible solution,
so the binary model links x_ij <= m_ij y_ij. With integer supplies and demands every vertex of the transportation
polytope (also restricted to a subset of arcs) is integral, so some optimal solution has integer flows and, with
integer costs, the optimal value is an integer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FCTPInstance:
    supply: np.ndarray   # (m,) supplies s_i > 0
    demand: np.ndarray   # (n,) demands d_j > 0
    tcost: np.ndarray    # (m, n) unit costs c_ij >= 0
    fcost: np.ndarray    # (m, n) fixed costs f_ij >= 0
    name: str = field(default="")

    @property
    def m(self) -> int:
        return len(self.supply)

    @property
    def n(self) -> int:
        return len(self.demand)

    @property
    def cap(self) -> np.ndarray:
        """(m, n) implied arc bounds m_ij = min(s_i, d_j)."""
        return np.minimum(self.supply[:, None], self.demand[None, :])

    @property
    def integral(self) -> bool:
        """Integer supplies, demands and costs: integer optimal flows exist and objective values are integers."""
        return all(bool(np.all(a == np.round(a))) for a in (self.supply, self.demand, self.tcost, self.fcost))

    def validate(self) -> None:
        m, n = self.m, self.n
        if m < 1 or n < 1:
            raise ValueError("empty instance")
        if self.tcost.shape != (m, n) or self.fcost.shape != (m, n):
            raise ValueError(f"cost matrices must have shape ({m}, {n})")
        for name, arr in (("supply", self.supply), ("demand", self.demand), ("tcost", self.tcost), ("fcost", self.fcost)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name}: non-finite values")
        if np.any(self.supply <= 0) or np.any(self.demand <= 0):
            raise ValueError("supplies and demands must be positive")
        if np.any(self.tcost < 0) or np.any(self.fcost < 0):
            raise ValueError("costs must be nonnegative")
        if abs(float(self.supply.sum()) - float(self.demand.sum())) > 1e-9 * max(1.0, float(self.supply.sum())):
            raise ValueError(f"unbalanced instance: total supply {self.supply.sum()} != total demand {self.demand.sum()}")


def parse_fctp(text: str, name: str = "") -> FCTPInstance:
    tok = text.split()
    if len(tok) < 2:
        raise ValueError("missing header")
    m, n = int(tok[0]), int(tok[1])
    expected = 2 + m + n + 2 * m * n
    if len(tok) != expected:
        raise ValueError(f"{len(tok)} numbers, expected {expected} for m={m}, n={n}")
    vals = np.array([float(t) for t in tok[2:]])
    k = 0
    supply = vals[k:k + m]; k += m
    demand = vals[k:k + n]; k += n
    tcost = vals[k:k + m * n].reshape(m, n); k += m * n
    fcost = vals[k:k + m * n].reshape(m, n)
    inst = FCTPInstance(supply=supply, demand=demand, tcost=tcost, fcost=fcost, name=name)
    inst.validate()
    return inst


def load_fctp(path: str | Path) -> FCTPInstance:
    return parse_fctp(Path(path).read_text(), name=Path(path).stem)


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def write_fctp(inst: FCTPInstance, path: str | Path) -> None:
    """Write an instance in the same format (used for generated test instances)."""
    rows = [f"{inst.m} {inst.n}", " ".join(_fmt(v) for v in inst.supply), " ".join(_fmt(v) for v in inst.demand),
            " ".join(_fmt(v) for v in inst.tcost.ravel()), " ".join(_fmt(v) for v in inst.fcost.ravel())]
    Path(path).write_text("\n".join(rows) + "\n")


def random_instance(rng: np.random.Generator, m: int, n: int, fixed_scale: float = 1.0, total: int | None = None) -> FCTPInstance:
    """A small Glover-like instance (tests): integer supplies and demands summing to the same total, unit costs in
    [3, 8], fixed costs in [50, 200) * fixed_scale."""
    total = total or 10 * m * n
    cuts_s = np.sort(rng.choice(np.arange(1, total), size=m - 1, replace=False)) if m > 1 else np.zeros(0, dtype=int)
    cuts_d = np.sort(rng.choice(np.arange(1, total), size=n - 1, replace=False)) if n > 1 else np.zeros(0, dtype=int)
    supply = np.diff(np.concatenate([[0], cuts_s, [total]])).astype(float)
    demand = np.diff(np.concatenate([[0], cuts_d, [total]])).astype(float)
    tcost = rng.integers(3, 9, size=(m, n)).astype(float)
    fcost = np.floor(rng.integers(50, 200, size=(m, n)) * fixed_scale).astype(float)
    inst = FCTPInstance(supply=supply, demand=demand, tcost=tcost, fcost=fcost, name=f"rand{m}x{n}")
    inst.validate()
    return inst
