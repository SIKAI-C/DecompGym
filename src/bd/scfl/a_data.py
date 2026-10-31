"""Two-stage stochastic capacitated facility location (SCFL) instances for the scale-up study.

Problem (Louveaux 1986; the formulation (15) of Bodur, Dash, Guenluek and Luedtke 2017, INFORMS JoC 29(1):77-91):
    first stage   open facilities y_i in {0, 1} (fixed cost f_i, capacity u_i), subject to
                  sum_i u_i y_i >= max_s sum_j d_sj                      ((15d): relatively complete recourse)
    second stage  in scenario s (probability p_s, customer demands d_sj) ship x_sij >= 0 with
                  sum_i x_sij = d_sj  (every customer's demand is met),  sum_j x_sij <= u_i y_i  (capacity),
                  at cost Q_s(y) = min sum_ij c_ij x_sij  (c_ij = unit cost)
    minimize      f.y + sum_s p_s Q_s(y).
For a binary y, every scenario is feasible iff (15d) holds (all customers can be served by every open facility), so
(15d) is the exact feasibility condition of a design, not an extra restriction.

Instances (`make_stochastic`) follow Bodur et al. (2017, Section 4.4): the deterministic OR-Library CAP instances
cap101-cap134 (Beasley 1990; 25 or 50 facilities, 50 customers) give f, u, the mean demands dbar_j and the cost of
serving all of customer j from facility i, so the unit cost is c_ij = cost_ij / dbar_j. For each customer a standard
deviation sigma_j ~ U(0.1 dbar_j, 0.3 dbar_j) is drawn once, then demands d_sj ~ N(dbar_j, sigma_j) independently,
with equal scenario probabilities. Our documented choices where the paper is silent: demands are rounded to the
nearest integer and truncated at 0 (integer demands and capacities make every vertex of the recourse integral), and
the random stream is numpy's default_rng(seed) (sigma first, then the S x J demand matrix in row order). Bodur et al.
did not publish their sampled instances or per-instance optima, so generated instances have no literature values.

File format (JSON, written by `write_instance`):
    {"format": "scfl-v1", "name": str, "capacity": [I], "fixed_cost": [I], "unit_cost": [[J] * I],
     "demand": [[J] * S], "probability": [S] (optional; uniform 1/S when absent), "meta": {...}}.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SCFLInstance:
    capacity: np.ndarray   # (I,) u_i > 0
    fixed: np.ndarray      # (I,) f_i >= 0
    cost: np.ndarray       # (I, J) unit costs c_ij >= 0
    demand: np.ndarray     # (S, J) scenario demands d_sj >= 0
    prob: np.ndarray       # (S,) probabilities, summing to 1
    name: str = field(default="")
    meta: dict = field(default_factory=dict)

    @property
    def I(self) -> int:  # noqa: E743 - the paper's notation
        return len(self.capacity)

    @property
    def J(self) -> int:
        return self.cost.shape[1]

    @property
    def S(self) -> int:
        return self.demand.shape[0]

    @property
    def total_demand(self) -> np.ndarray:
        """(S,) total demand per scenario."""
        return self.demand.sum(axis=1)

    @property
    def max_demand(self) -> float:
        return float(self.total_demand.max())

    @property
    def mean_demand(self) -> np.ndarray:
        """(J,) expected demand per customer."""
        return self.prob @ self.demand

    def feasible(self, y: np.ndarray) -> bool:
        """(15d) for a binary design: every scenario's total demand fits into the open capacity."""
        return float(self.capacity @ np.asarray(y, dtype=float)) >= self.max_demand - 1e-9 * max(1.0, self.max_demand)

    def validate(self) -> None:
        I, J, S = self.I, self.J, self.S
        if I < 1 or J < 1 or S < 1:
            raise ValueError("empty instance")
        if self.fixed.shape != (I,) or self.cost.shape != (I, J) or self.demand.shape != (S, J) or self.prob.shape != (S,):
            raise ValueError("inconsistent array shapes")
        for name, arr in (("capacity", self.capacity), ("fixed", self.fixed), ("cost", self.cost), ("demand", self.demand),
                          ("prob", self.prob)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name}: non-finite values")
        if np.any(self.capacity <= 0) or np.any(self.fixed < 0) or np.any(self.cost < 0) or np.any(self.demand < 0):
            raise ValueError("capacities must be positive; costs and demands nonnegative")
        if np.any(self.prob <= 0) or abs(float(self.prob.sum()) - 1.0) > 1e-9:
            raise ValueError("probabilities must be positive and sum to 1")
        if not self.feasible(np.ones(I)):
            raise ValueError("infeasible instance: total capacity below the largest scenario demand")


# ---------------------------------------------------------------- OR-Library CAP files

def read_orlib_cap(path: str | Path) -> dict:
    """OR-Library capacitated warehouse location file: 'm n', m lines 'capacity fixed_cost', then for each customer
    its demand followed by the m costs of allocating all of its demand to each warehouse."""
    tok = Path(path).read_text().split()
    m, n = int(tok[0]), int(tok[1])
    if len(tok) != 2 + 2 * m + n * (m + 1):
        raise ValueError(f"{path}: {len(tok)} tokens, expected {2 + 2 * m + n * (m + 1)} for m={m}, n={n}")
    vals = [float(t) for t in tok[2:]]
    cap = np.array(vals[0:2 * m:2])
    fixed = np.array(vals[1:2 * m:2])
    rest = np.array(vals[2 * m:]).reshape(n, m + 1)
    demand = rest[:, 0]
    alloc = rest[:, 1:].T  # (m, n): cost of serving all of customer j from facility i
    if np.any(demand <= 0):
        raise ValueError("OR-Library customers must have positive demand")
    return {"capacity": cap, "fixed": fixed, "demand": demand, "alloc_cost": alloc, "name": Path(path).stem}


def deterministic_instance(base: dict) -> SCFLInstance:
    """The deterministic CAP instance as a one-scenario SCFL (unit costs alloc_cost / demand). Its optimum is the
    OR-Library value (capopt.txt), a check of the unit-cost conversion."""
    inst = SCFLInstance(capacity=base["capacity"].copy(), fixed=base["fixed"].copy(),
                        cost=base["alloc_cost"] / base["demand"][None, :], demand=base["demand"][None, :].copy(),
                        prob=np.ones(1), name=base["name"], meta={"base": base["name"], "scenarios": 1})
    inst.validate()
    return inst


def make_stochastic(base: dict, scenarios: int, seed: int, sigma: tuple[float, float] = (0.1, 0.3)) -> SCFLInstance:
    """Bodur et al. (2017) stochastic CAP instance (see the module docstring)."""
    rng = np.random.default_rng(seed)
    dbar = base["demand"]
    sig = rng.uniform(sigma[0] * dbar, sigma[1] * dbar)
    raw = rng.normal(dbar[None, :], sig[None, :], size=(scenarios, len(dbar)))
    demand = np.maximum(np.rint(raw), 0.0)
    inst = SCFLInstance(capacity=base["capacity"].copy(), fixed=base["fixed"].copy(),
                        cost=base["alloc_cost"] / dbar[None, :], demand=demand, prob=np.full(scenarios, 1.0 / scenarios),
                        name=f"{base['name']}_s{scenarios}_g{seed}",
                        meta={"base": base["name"], "scenarios": scenarios, "seed": seed, "sigma": list(sigma),
                              "generator": "Bodur et al. (2017) Sec. 4.4: sigma_j ~ U(0.1, 0.3) dbar_j, d_sj ~ N(dbar_j, sigma_j); "
                                           "rounded to the nearest integer, truncated at 0; numpy default_rng(seed)",
                              "truncated_values": int(np.sum(raw < -0.5))})
    if not inst.feasible(np.ones(inst.I)):
        raise ValueError("generated instance is infeasible (total capacity below a scenario demand)")
    inst.validate()
    return inst


def make_cst(I: int, J: int, ratio: float, scenarios: int, seed: int, sigma: tuple[float, float] = (0.1, 0.3)) -> SCFLInstance:
    """Larger generated instances: the capacitated facility location generator of Cornuejols, Sridharan and Thizy
    (1991) as used by Avella and Boccia (2009) and Fischetti, Ljubic and Sinnl (2016): facilities and customers
    uniform in the unit square, mean demands dbar_j ~ U{5..35}, capacities s_i ~ U[10, 160] rescaled so that
    sum_i s_i = ratio * sum_j dbar_j (then rounded up to integers), fixed costs f_i = U[0, 90] + U[100, 110] sqrt(s_i)
    (from the rescaled capacities; our choice where descriptions differ), unit costs 10 * Euclidean distance; then
    Bodur et al.'s stochastic demands as in `make_stochastic` (same rounding and truncation)."""
    rng = np.random.default_rng(seed)
    fpos, cpos = rng.random((I, 2)), rng.random((J, 2))
    dbar = rng.integers(5, 36, size=J).astype(float)
    cap = rng.uniform(10, 160, size=I)
    cap = np.ceil(cap * ratio * dbar.sum() / cap.sum())
    fixed = rng.uniform(0, 90, size=I) + rng.uniform(100, 110, size=I) * np.sqrt(cap)
    cost = 10.0 * np.sqrt(((fpos[:, None, :] - cpos[None, :, :]) ** 2).sum(axis=2))
    sig = rng.uniform(sigma[0] * dbar, sigma[1] * dbar)
    raw = rng.normal(dbar[None, :], sig[None, :], size=(scenarios, J))
    demand = np.maximum(np.rint(raw), 0.0)
    inst = SCFLInstance(capacity=cap, fixed=fixed, cost=cost, demand=demand, prob=np.full(scenarios, 1.0 / scenarios),
                        name=f"cst{I}x{J}_r{ratio:g}_s{scenarios}_g{seed}",
                        meta={"base": "CST (Cornuejols, Sridharan and Thizy 1991)", "I": I, "J": J, "ratio": ratio,
                              "scenarios": scenarios, "seed": seed, "sigma": list(sigma),
                              "generator": "make_cst: CST CFL + Bodur et al. (2017) stochastic demands; numpy default_rng(seed)",
                              "truncated_values": int(np.sum(raw < -0.5))})
    inst.validate()
    return inst


# ---------------------------------------------------------------- JSON files

def _num(v: float):
    v = float(v)
    return int(v) if v.is_integer() and abs(v) < 2 ** 53 else v


def instance_dict(inst: SCFLInstance, uniform: bool | None = None) -> dict:
    if uniform is None:
        uniform = bool(np.all(inst.prob == 1.0 / inst.S))
    out = {"format": "scfl-v1", "name": inst.name, "capacity": [_num(v) for v in inst.capacity],
           "fixed_cost": [_num(v) for v in inst.fixed], "unit_cost": [[_num(v) for v in row] for row in inst.cost],
           "demand": [[_num(v) for v in row] for row in inst.demand], "meta": inst.meta}
    if not uniform:
        out["probability"] = [float(v) for v in inst.prob]
    return out


def write_instance(inst: SCFLInstance, path: str | Path) -> str:
    """Write the JSON file (floats via repr, exact round trip); returns its sha256."""
    text = json.dumps(instance_dict(inst), separators=(",", ":"))
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def parse_instance(obj: dict, name: str = "") -> SCFLInstance:
    if obj.get("format") != "scfl-v1":
        raise ValueError("not an scfl-v1 instance")
    demand = np.array(obj["demand"], dtype=float)
    S = demand.shape[0]
    prob = np.array(obj["probability"], dtype=float) if obj.get("probability") is not None else np.full(S, 1.0 / S)
    inst = SCFLInstance(capacity=np.array(obj["capacity"], dtype=float), fixed=np.array(obj["fixed_cost"], dtype=float),
                        cost=np.array(obj["unit_cost"], dtype=float), demand=demand, prob=prob,
                        name=obj.get("name") or name, meta=obj.get("meta", {}))
    inst.validate()
    return inst


def load_instance(path: str | Path) -> SCFLInstance:
    return parse_instance(json.loads(Path(path).read_text()), name=Path(path).stem)


# ---------------------------------------------------------------- small instances (tests)

def random_instance(rng: np.random.Generator, I: int, J: int, S: int, tightness: float = 1.6, uniform: bool = True,
                    integer_costs: bool = False) -> SCFLInstance:
    """A small random instance: integer demands in [0, 30], integer capacities summing to about `tightness` times
    the largest scenario demand (so some designs are infeasible), unit costs from planar distances."""
    fpos, cpos = rng.random((I, 2)), rng.random((J, 2))
    cost = 10.0 * np.sqrt(((fpos[:, None, :] - cpos[None, :, :]) ** 2).sum(axis=2)) + rng.random((I, J))
    if integer_costs:
        cost = np.round(cost)
    demand = rng.integers(0, 31, size=(S, J)).astype(float)
    demand[:, 0] = np.maximum(demand[:, 0], 1.0)
    dmax = demand.sum(axis=1).max()
    share = rng.dirichlet(np.ones(I))
    capacity = np.maximum(np.ceil(tightness * dmax * (0.3 / I + 0.7 * share)), 1.0)
    while capacity.sum() < dmax:
        capacity[int(rng.integers(I))] += dmax
    fixed = np.round(rng.uniform(20, 120, size=I) + 2.0 * np.sqrt(capacity))
    prob = np.full(S, 1.0 / S) if uniform else rng.dirichlet(np.ones(S))
    inst = SCFLInstance(capacity=capacity, fixed=fixed, cost=cost, demand=demand, prob=prob,
                        name=f"rand{I}x{J}x{S}", meta={"generator": "random_instance (tests)"})
    inst.validate()
    return inst


def main() -> None:
    """Generate stochastic CAP instances: python -m src.bd.scfl.a_data --raw <orlib dir> --out <dir>
    --bases cap111 cap121 --scenarios 250 500 --seeds 1 (prints name, sha256 and the truncation count)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None, help="OR-Library directory (needed with --bases)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bases", nargs="+", default=[])
    ap.add_argument("--cst", nargs=3, type=float, action="append", default=[], metavar=("I", "J", "RATIO"))
    ap.add_argument("--scenarios", nargs="+", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for I, J, ratio in a.cst:
        for S in a.scenarios:
            for seed in a.seeds:
                inst = make_cst(int(I), int(J), ratio, S, seed)
                h = write_instance(inst, Path(a.out) / f"{inst.name}.json")
                print(json.dumps({"name": inst.name, "sha256": h, "max_demand": inst.max_demand,
                                  "mean_total_demand": float(inst.total_demand.mean()),
                                  "truncated_values": inst.meta["truncated_values"]}))
    for b in a.bases:
        base = read_orlib_cap(Path(a.raw) / f"{b}.txt")
        for S in a.scenarios:
            for seed in a.seeds:
                inst = make_stochastic(base, S, seed)
                h = write_instance(inst, Path(a.out) / f"{inst.name}.json")
                print(json.dumps({"name": inst.name, "sha256": h, "max_demand": inst.max_demand,
                                  "mean_total_demand": float(inst.total_demand.mean()),
                                  "truncated_values": inst.meta["truncated_values"]}))


if __name__ == "__main__":
    main()
