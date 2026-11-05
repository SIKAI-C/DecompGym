"""Two-stage stochastic production planning (TSSO) at scale: instances, generator and the two-stage view (scale-up
study; the paper's TSSO files in this directory are unchanged).

Problem (the paper's model, `a_gold_monolithic.py`, unchanged): products i, periods t, scenarios s with probability
p_s. First stage: activate regular capacity y_it in {0, 1} at fixed cost F_it. Second stage, per scenario: regular
production xr_its <= K_it y_it (unit cost c_reg_it), overtime xo_its >= 0 (c_ot_it, uncapacitated), inventory
inv_its >= 0 (h_it) and backlog back_its >= 0 (b_it) with the balance
    xr_its + xo_its - inv_its + back_its = d_its - inv_i,t-1,s + back_i,t-1,s   (inv_i,-1 = I0_i, back_i,-1 = B0_i).
Objective: F.y + sum_s p_s (c_reg.xr_s + c_ot.xo_s + h.inv_s + b.back_s). Backlog left at the last period is never
served (it pays b_i,T-1 once, a lost-sales penalty), and inventory left at the end pays h_i,T-1 -- both as in the
paper.

Two-stage view (declared, `VIEW`): the recourse separates by product AND scenario (no row couples two products), so
the subproblems are the pairs k = (i, s) with ID k = i * S + s and trusted weight p_s:
    Q_is(y_i) = min { c_reg_i.xr + c_ot_i.xo + h_i.inv + b_i.back :
                      bal_t:  xr_t + xo_t - inv_t + back_t + inv_t-1 - back_t-1 = d_its   (t = 0..T-1),
                      0 <= xr_t <= K_it y_it,  xo, inv, back >= 0 }
    z(y) = sum_it F_it y_it + sum_(i,s) p_s Q_is(y_i).
XL instances have I0 = B0 = 0 (as every paper instance) and nonnegative costs, so Q_is >= 0 (`trivial_lower`).

Implied bounds (used by the certificate checker's Neumaier-Shcherbina safe cuts): with D_is = sum_t d_its (the
total-horizon demand of the subproblem), adding 0 <= xr_t, xo_t, inv_t, back_t <= D_is does not change Q_is(y) for
any y in [0, 1]^T. Proof: the subproblem is a min-cost flow on the path 0..T-1 (production arcs from a super source
into t, inventory arcs t -> t+1 with cost h_t, backlog arcs t+1 -> t with cost b_t, inv_T-1 an arc from T-1 to a
disposal sink, back_T-1 an arc from an 'unmet' source into T-1, demand d_t leaving t). Take any optimal flow and
decompose it into paths and cycles. Every cycle and every path ending in the disposal sink has nonnegative cost
(all costs are nonnegative) and only upper bounds (K_it y_it on xr) restrict the flow, so deleting them keeps the
flow feasible and optimal. What remains is a set of paths, each ending in one unit of demand, of total amount D_is;
so every arc -- every xr, xo, inv and back variable -- carries at most D_is. (A bound such as "production <=
remaining demand" is not implied: production may clear earlier backlog.)

Proven upper bound: U_is = sum_t c_ot_it d_its (all demand by overtime in its own period is feasible for every y),
so Q_is(y) <= U_is on [0, 1]^T.

Generator (`generate`): the paper's recipe verbatim -- `TSSOGenerator.generate_instance` of `a_gen_data.py` is
called with the requested sizes and seed, so every distribution (p ~ U(0, 1) normalized, F ~ U{30..89},
K ~ U{25..69}, c_reg ~ U(1, 3), c_ot = c_reg + U(0.8, 2), h ~ U(0.2, 0.8), b ~ U(2, 4.5), d ~ U{12..55}, I0 = B0 = 0)
and the random stream are the paper's; only the sizes are larger.

File format (JSON, `write_instance`): the paper's data.json keys (n_products, n_periods, n_scenarios, prob, F, K,
c_reg, c_ot, h, b, demand [I][T][S], I0, B0) plus "format": "tsso-xl-v1", "name" and "meta"; the paper's loader
reads it too.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT = "tsso-xl-v1"
VIEW = {
    "id": "tsso-xl-compact-v1",
    "subproblems": "(i, s) for product i in 0..I-1 and scenario s in 0..S-1; ID k = i * S + s; weight p_s",
    "rows": "bal_t (t = 0..T-1, equality, dual pi_t free); cap_t: xr_t <= K_it y_it (dual mu_t >= 0)",
    "columns": "xr_t, xo_t, inv_t, back_t in [0, D_is] with D_is = sum_t d_its",
    "theta_box": "[0, U_is] with U_is = sum_t c_ot_it d_its",
}


@dataclass
class TSSOXLInstance:
    F: np.ndarray       # (I, T) fixed activation costs
    K: np.ndarray       # (I, T) regular capacities
    c_reg: np.ndarray   # (I, T)
    c_ot: np.ndarray    # (I, T)
    h: np.ndarray       # (I, T) holding
    b: np.ndarray       # (I, T) backlog
    demand: np.ndarray  # (I, T, S)
    prob: np.ndarray    # (S,)
    I0: np.ndarray      # (I,) zeros
    B0: np.ndarray      # (I,) zeros
    name: str = field(default="")
    meta: dict = field(default_factory=dict)

    @property
    def I(self) -> int:  # noqa: E743 - the paper's notation
        return self.F.shape[0]

    @property
    def T(self) -> int:
        return self.F.shape[1]

    @property
    def S(self) -> int:
        return self.demand.shape[2]

    @property
    def total_demand(self) -> np.ndarray:
        """(I, S) D_is = sum_t d_its, the implied bound of every recourse variable."""
        return self.demand.sum(axis=1)

    @property
    def upper(self) -> np.ndarray:
        """(I, S) U_is = sum_t c_ot_it d_its >= Q_is(y) for every y in [0, 1]^T."""
        return np.einsum("it,its->is", self.c_ot, self.demand)

    def validate(self) -> None:
        I, T, S = self.I, self.T, self.S
        if I < 1 or T < 1 or S < 1:
            raise ValueError("empty instance")
        for name in ("K", "c_reg", "c_ot", "h", "b"):
            if getattr(self, name).shape != (I, T):
                raise ValueError(f"{name} must have shape ({I}, {T})")
        if self.demand.shape != (I, T, S) or self.prob.shape != (S,) or self.I0.shape != (I,) or self.B0.shape != (I,):
            raise ValueError("inconsistent array shapes")
        for name in ("F", "K", "c_reg", "c_ot", "h", "b", "demand", "prob", "I0", "B0"):
            arr = getattr(self, name)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name}: non-finite values")
            if np.any(arr < 0):
                raise ValueError(f"{name}: negative values")
        if np.any(self.I0 != 0) or np.any(self.B0 != 0):
            raise ValueError("XL instances have zero initial inventory and backlog")
        if np.any(self.prob <= 0) or abs(float(self.prob.sum()) - 1.0) > 1e-9:
            raise ValueError("probabilities must be positive and sum to 1")
        if np.any(self.c_ot < self.c_reg):
            raise ValueError("overtime must not be cheaper than regular production")


def from_paper(inst, name: str = "", meta: dict | None = None) -> TSSOXLInstance:
    """A paper TSSOInstance (a_gen_data) as an XL instance (same arrays)."""
    out = TSSOXLInstance(F=np.asarray(inst.F, float), K=np.asarray(inst.K, float), c_reg=np.asarray(inst.c_reg, float),
                         c_ot=np.asarray(inst.c_ot, float), h=np.asarray(inst.h, float), b=np.asarray(inst.b, float),
                         demand=np.asarray(inst.demand, float), prob=np.asarray(inst.prob, float).reshape(-1),
                         I0=np.asarray(inst.I0, float).reshape(-1), B0=np.asarray(inst.B0, float).reshape(-1),
                         name=name, meta=dict(meta or {}))
    out.validate()
    return out


def generate(products: int, periods: int, scenarios: int, seed: int) -> TSSOXLInstance:
    """The paper's generator (a_gen_data.TSSOGenerator.generate_instance) at the requested sizes."""
    from src.bd.tsso.a_gen_data import TSSOGenerator
    raw = TSSOGenerator(seed=seed).generate_instance(n_products=products, n_periods=periods, n_scenarios=scenarios, seed=seed)
    name = f"tsso_p{products}_t{periods}_s{scenarios}_g{seed}"
    return from_paper(raw, name=name, meta={"products": products, "periods": periods, "scenarios": scenarios, "seed": seed,
                                            "generator": "src.bd.tsso.a_gen_data.TSSOGenerator.generate_instance (paper recipe)"})


def write_instance(inst: TSSOXLInstance, path: str | Path) -> Path:
    inst.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"format": FORMAT, "name": inst.name, "n_products": inst.I, "n_periods": inst.T, "n_scenarios": inst.S,
           "prob": inst.prob.tolist(), "F": inst.F.tolist(), "K": inst.K.tolist(), "c_reg": inst.c_reg.tolist(),
           "c_ot": inst.c_ot.tolist(), "h": inst.h.tolist(), "b": inst.b.tolist(),
           "demand": [[[int(v) if float(v).is_integer() else float(v) for v in row] for row in prod] for prod in inst.demand],
           "I0": inst.I0.tolist(), "B0": inst.B0.tolist(), "meta": inst.meta}
    path.write_text(json.dumps(obj, separators=(",", ":")))
    return path


def load_instance(path: str | Path) -> TSSOXLInstance:
    obj = json.loads(Path(path).read_text())
    if obj.get("format") not in (None, FORMAT):
        raise ValueError(f"not a {FORMAT} instance")
    inst = TSSOXLInstance(F=np.asarray(obj["F"], float), K=np.asarray(obj["K"], float), c_reg=np.asarray(obj["c_reg"], float),
                          c_ot=np.asarray(obj["c_ot"], float), h=np.asarray(obj["h"], float), b=np.asarray(obj["b"], float),
                          demand=np.asarray(obj["demand"], float), prob=np.asarray(obj["prob"], float).reshape(-1),
                          I0=np.asarray(obj.get("I0", [0.0] * len(obj["F"])), float).reshape(-1),
                          B0=np.asarray(obj.get("B0", [0.0] * len(obj["F"])), float).reshape(-1),
                          name=obj.get("name", Path(path).stem), meta=obj.get("meta", {}))
    if abs(float(inst.prob.sum()) - 1.0) > 1e-9 and np.all(inst.prob > 0):
        inst.prob = inst.prob / inst.prob.sum()  # the paper's loader renormalizes the same way
    inst.validate()
    return inst


# ---------------------------------------------------------------- shared basic initializer

def initial_design(inst: TSSOXLInstance) -> np.ndarray:
    """Myopic activation shared by the naive and engineered Benders and the heuristic controls: open (i, t) iff the
    expected regular-production saving in its own period, E_s[min(K_it, d_its)] (c_ot_it - c_reg_it), covers F_it."""
    use = np.einsum("s,its->it", inst.prob, np.minimum(inst.K[:, :, None], inst.demand))
    return use * (inst.c_ot - inst.c_reg) >= inst.F


def design_key(y: np.ndarray) -> bytes:
    return np.packbits(np.asarray(y, dtype=bool), axis=None).tobytes()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate scale-up TSSO instances with the paper's recipe.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", nargs="+", required=True, help="PxTxS, e.g. 40x30x1000")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    a = ap.parse_args()
    for size in a.sizes:
        P, T, S = (int(v) for v in size.lower().split("x"))
        for seed in a.seeds:
            inst = generate(P, T, S, seed)
            path = write_instance(inst, Path(a.out) / f"{inst.name}.json")
            print(json.dumps({"name": inst.name, "path": str(path), "I": P, "T": T, "S": S, "seed": seed}))


if __name__ == "__main__":
    main()
