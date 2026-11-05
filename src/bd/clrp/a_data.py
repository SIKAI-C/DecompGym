"""Capacitated location-routing (CLRP) instances in Prodhon's format (scale-up study).

File layout: n, m; m depot coordinates; n customer coordinates; vehicle capacity; m depot capacities;
n demands; m depot opening costs; vehicle fixed cost; cost-type flag. Prodhon's tables report costs
with travel costs as Euclidean distances multiplied by 100; `rounding` selects how they are made
integral (the convention is fixed by reproducing the published optima).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CLRPInstance:
    depots: np.ndarray       # (m, 2)
    customers: np.ndarray    # (n, 2)
    vehicle_capacity: int
    depot_capacity: np.ndarray  # (m,)
    demand: np.ndarray       # (n,)
    opening_cost: np.ndarray  # (m,)
    vehicle_cost: float
    cost_type: int

    @property
    def n(self) -> int:
        return len(self.customers)

    @property
    def m(self) -> int:
        return len(self.depots)

    def travel(self, rounding: str = "ceil") -> np.ndarray:
        """(m + n) x (m + n) travel costs, depots first, as 100 x Euclidean distance."""
        pts = np.vstack([self.depots, self.customers]).astype(float)
        d = 100.0 * np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        return {"ceil": np.ceil(d - 1e-9), "round": np.round(d), "floor": np.floor(d + 1e-9), "none": d}[rounding]


def load_prodhon(path: str | Path) -> CLRPInstance:
    tok = [float(t) for t in Path(path).read_text().split()]
    n, m = int(tok[0]), int(tok[1])
    pos = 2
    take = lambda k: tok[pos:pos + k]
    depots = np.array(take(2 * m)).reshape(m, 2); pos += 2 * m
    customers = np.array(tok[pos:pos + 2 * n]).reshape(n, 2); pos += 2 * n
    q = int(tok[pos]); pos += 1
    cap = np.array(tok[pos:pos + m]); pos += m
    dem = np.array(tok[pos:pos + n]); pos += n
    open_cost = np.array(tok[pos:pos + m]); pos += m
    vcost, ctype = tok[pos], int(tok[pos + 1]); pos += 2
    if pos != len(tok):
        raise ValueError(f"{path}: {len(tok) - pos} trailing tokens")
    return CLRPInstance(depots, customers, q, cap, dem, open_cost, vcost, ctype)
