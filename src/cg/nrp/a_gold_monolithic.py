from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.cg.nrp.a_gen_data import NRPInstance


class NRPBaseline:
    """Assignment MIP x[n, d, s] with all rostering rules and soft coverage."""

    def __init__(self, *, params: dict[str, Any] | None = None) -> None:
        self.params = {"Threads": 1, **(params or {})}

    def run(self, path: str) -> dict[str, Any]:
        inst = NRPInstance.from_json_file(path)
        N, D, S = range(inst.n_nurses), range(inst.n_days), range(3)
        m = gp.Model("nrp_monolithic")
        m.Params.OutputFlag = 0
        for k, v in self.params.items():
            m.setParam(k, v)
        x = m.addVars(N, D, S, vtype=GRB.BINARY)
        short = m.addVars(D, S, lb=0.0)
        m.addConstrs(x.sum(n, d, "*") <= 1 for n in N for d in D)
        m.addConstrs(x.sum("*", d, s) + short[d, s] >= inst.demand[d][s] for d in D for s in S)
        for n in N:
            m.addConstr(x.sum(n, "*", "*") >= inst.min_days[n])
            m.addConstr(x.sum(n, "*", "*") <= inst.max_days[n])
            m.addConstr(x.sum(n, "*", 2) <= inst.max_nights[n])
            k = inst.max_consecutive
            for d0 in range(inst.n_days - k):
                m.addConstr(gp.quicksum(x[n, d, s] for d in range(d0, d0 + k + 1) for s in S) <= k)
            for a, b in inst.forbidden_successions:
                m.addConstrs(x[n, d, a] + x[n, d + 1, b] <= 1 for d in range(inst.n_days - 1))
        m.setObjective(gp.quicksum(inst.cost[n][d][s] * x[n, d, s] for n in N for d in D for s in S) + inst.shortage_penalty * short.sum(), GRB.MINIMIZE)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Baseline model not optimal (status={m.Status})")
        rosters = [[next((s for s in S if x[n, d, s].X > 0.5), -1) for d in D] for n in N]
        return {"best_objective": float(m.ObjVal), "global_lb": float(m.ObjBound), "global_ub": float(m.ObjVal), "rosters": rosters}
