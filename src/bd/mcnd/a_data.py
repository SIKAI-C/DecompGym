"""Multicommodity capacitated fixed-charge network design (MCND) instances in the Canad ".dow" format (scale-up study).

File layout (written by Canad/Mulgen/outdat.f with the `dow` option): the header line "MULTIGEN.DAT:", then
n (nodes), m (arcs), K (commodities); then one line per arc
    from  to  variable_cost  capacity  fixed_cost  1  generator_arc_index
and one line per commodity
    origin  destination  demand.
Nodes are numbered 1..n in the files and 0..n-1 here. The generator writes the same variable cost for every
commodity on an arc and the individual capacities min(u_a, d_k); the trailing two arc fields are not data.

Problem: open arcs y_a in {0, 1} at fixed cost f_a and route every commodity k (d_k units from o_k to t_k, splittable,
continuous) over open arcs at unit cost c_a, with sum_k x_ak <= u_a y_a. Objective: sum_a f_a y_a + sum_{a,k} c_a x_ak.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class MCNDInstance:
    n: int
    tail: np.ndarray      # (m,) int, 0-based start node of each arc
    head: np.ndarray      # (m,) int, 0-based end node
    cost: np.ndarray      # (m,) variable cost per unit of flow (any commodity)
    cap: np.ndarray       # (m,) arc capacity u_a
    fixed: np.ndarray     # (m,) fixed opening cost f_a
    orig: np.ndarray      # (K,) int, 0-based origin of each commodity
    dest: np.ndarray      # (K,) int, 0-based destination
    demand: np.ndarray    # (K,) demand d_k
    name: str = field(default="")

    @property
    def m(self) -> int:
        return len(self.tail)

    @property
    def K(self) -> int:
        return len(self.orig)

    def supply(self) -> np.ndarray:
        """(K, n) node balances b: +d_k at the origin, -d_k at the destination (outflow - inflow = b)."""
        b = np.zeros((self.K, self.n))
        b[np.arange(self.K), self.orig] += self.demand
        b[np.arange(self.K), self.dest] -= self.demand
        return b

    def link_bound(self) -> np.ndarray:
        """(K, m) strong-linking coefficients min(d_k, u_a)."""
        return np.minimum(self.demand[:, None], self.cap[None, :])

    def validate(self) -> None:
        n, m, K = self.n, self.m, self.K
        for name, arr, size in (("head", self.head, m), ("cost", self.cost, m), ("cap", self.cap, m), ("fixed", self.fixed, m),
                                ("dest", self.dest, K), ("demand", self.demand, K)):
            if len(arr) != size:
                raise ValueError(f"{name}: {len(arr)} entries, expected {size}")
        if m and (self.tail.min() < 0 or self.head.min() < 0 or max(self.tail.max(), self.head.max()) >= n):
            raise ValueError("arc end node out of range")
        if K and (self.orig.min() < 0 or self.dest.min() < 0 or max(self.orig.max(), self.dest.max()) >= n):
            raise ValueError("commodity node out of range")
        if np.any(self.tail == self.head):
            raise ValueError("self-loop arc")
        if np.any(self.orig == self.dest):
            raise ValueError("commodity with origin == destination")
        # Nonnegative costs make cycle-free flows optimal, which the strong linking x_ak <= min(d_k, u_a) y_a relies on.
        if np.any(self.cost < 0) or np.any(self.fixed < 0) or np.any(self.cap <= 0) or np.any(self.demand <= 0):
            raise ValueError("negative costs, nonpositive capacities or nonpositive demands are not supported")


def load_dow(path: str | Path) -> MCNDInstance:
    text = Path(path).read_text()
    lines = text.splitlines()
    if not lines or lines[0].strip() != "MULTIGEN.DAT:":
        raise ValueError(f"{path}: missing 'MULTIGEN.DAT:' header")
    tok = [int(t) for t in " ".join(lines[1:]).split()]
    n, m, K = tok[:3]
    if len(tok) != 3 + 7 * m + 3 * K:
        raise ValueError(f"{path}: {len(tok)} numbers, expected {3 + 7 * m + 3 * K} for n={n}, m={m}, K={K}")
    arcs = np.array(tok[3:3 + 7 * m], dtype=np.int64).reshape(m, 7)
    com = np.array(tok[3 + 7 * m:], dtype=np.int64).reshape(K, 3)
    if np.any(arcs[:, 5] != 1):
        raise ValueError(f"{path}: arc field 6 is not 1 (unexpected layout)")
    inst = MCNDInstance(n=n, tail=arcs[:, 0] - 1, head=arcs[:, 1] - 1, cost=arcs[:, 2].astype(float), cap=arcs[:, 3].astype(float),
                        fixed=arcs[:, 4].astype(float), orig=com[:, 0] - 1, dest=com[:, 1] - 1, demand=com[:, 2].astype(float),
                        name=Path(path).stem)
    inst.validate()
    return inst


def write_dow(inst: MCNDInstance, path: str | Path) -> None:
    """Write an instance in the same layout (used for generated test instances)."""
    rows = ["MULTIGEN.DAT:", f"{inst.n:8d}{inst.m:8d}{inst.K:8d}"]
    for a in range(inst.m):
        rows.append("".join(f"{int(v):8d}" for v in (inst.tail[a] + 1, inst.head[a] + 1, inst.cost[a], inst.cap[a], inst.fixed[a], 1, a + 1)))
    for k in range(inst.K):
        rows.append("".join(f"{int(v):8d}" for v in (inst.orig[k] + 1, inst.dest[k] + 1, inst.demand[k])))
    Path(path).write_text("\n".join(rows) + "\n")
