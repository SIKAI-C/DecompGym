"""Exact column generation for CLRP: root column generation over (depot, route) columns, then enumeration.

Master (LP), with lam_r for a route r at depot d(r) and 0 <= y_d <= 1:
    min  sum_d O_d y_d + sum_r c_r lam_r                        (c_r = vehicle cost + travel cost)
    s.t. sum_{r ∋ i} lam_r = 1                         (pi_i)    every customer covered once
         sum_{r at d} q(r) lam_r - W_d y_d <= 0         (mu_d)    depot capacity
         sum_{r at d, r ∋ i} lam_r - y_d <= 0           (nu_id)   a customer is served only from an open depot
Phase I uses one artificial per customer. Pricing is exact, one problem per depot: every capacity-feasible
customer subset has its cheapest route from that depot precomputed (`src.utils.subset_routes`), and its
reduced cost is c_r - sum_{i in r} (pi_i + q_i mu_d + nu_id). At the converged root LP value LB, any plan
cheaper than an upper bound UB uses only columns with reduced cost <= UB - LB (the y_d terms are
nonnegative once the dual of y_d <= 1 is accounted for), so the integer master over those columns and binary y is exact.
`enumerate=False` is the price-and-branch shortcut: root column generation, then one integer master.
"""
from __future__ import annotations

import math
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.clrp.a_gen_data import CLRPInstance
from src.utils.reference_runtime import ReferenceRuntime
from src.utils.subset_routes import RouteTable


Column = tuple[int, int]  # (depot, row of that depot's RouteTable)


class CLRPColumnGeneration:
    def __init__(self, *, time_limit: float = 60.0, enumerate: bool = True, batch: int = 30) -> None:
        self.runtime = ReferenceRuntime(time_limit)
        self.enumerate = enumerate
        if batch < 1:
            raise ValueError("batch must be a positive number of columns per pricing round")
        self.batch = batch

    def _rows(self, model, inst, tables, cols, lam, y, art=None):
        n, m = inst.n_customers, inst.n_depots
        cover = model.addConstrs(gp.quicksum(lam[k] for k, (d, c) in enumerate(cols) if tables[d].members[c, i]) + (art[i] if art is not None else 0) == 1 for i in range(n))
        cap = model.addConstrs(gp.quicksum(float(tables[d].load[c]) * lam[k] for k, (d, c) in enumerate(cols) if d == e) - inst.depot_capacity[e] * y[e] <= 0 for e in range(m))
        link = model.addConstrs(gp.quicksum(lam[k] for k, (d, c) in enumerate(cols) if d == e and tables[d].members[c, i]) - y[e] <= 0 for i in range(n) for e in range(m))
        model.setObjective(gp.quicksum(inst.opening_cost[e] * y[e] for e in range(m)) + gp.quicksum(self.cost(inst, tables, col) * lam[k] for k, col in enumerate(cols))
                           + (self.big * art.sum() if art is not None else 0), GRB.MINIMIZE)
        return cover, cap, link

    @staticmethod
    def cost(inst, tables, col: Column) -> float:
        return float(tables[col[0]].cost[col[1]] + inst.vehicle_cost)

    def _integer_master(self, inst, tables, cols: list[Column], limit: float):
        ip = gp.Model("clrp_integer_master")
        ip.Params.OutputFlag = 0
        ip.Params.TimeLimit = max(0.5, min(limit, self.runtime.remaining()))
        ip.Params.MIPGap, ip.Params.MIPGapAbs = 0.0, 0.5  # integer costs: exact once the bound is within 0.5
        lam = ip.addVars(len(cols), vtype=GRB.BINARY)
        y = ip.addVars(inst.n_depots, vtype=GRB.BINARY)
        self._rows(ip, inst, tables, cols, lam, y)
        self.runtime.optimize(ip, role="integer_master", allow_incumbent=True)
        self.last_master_bound = math.ceil(ip.ObjBound - 1e-6) if ip.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else None
        self.last_master_optimal = ip.Status == GRB.OPTIMAL and ip.SolCount > 0 and self.last_master_bound >= ip.ObjVal - 1e-6
        if ip.SolCount == 0:
            return None, math.inf
        chosen = [col for k, col in enumerate(cols) if lam[k].X > 0.5]
        value = sum(self.cost(inst, tables, col) for col in chosen) + sum(inst.opening_cost[d] for d in {d for d, _ in chosen})
        return chosen, float(value)

    def run(self, instance_name: str, *, started_at: float | None = None) -> dict[str, Any]:
        self.runtime.start(started_at)
        started = time.perf_counter()
        inst = CLRPInstance.from_json_file(instance_name)
        n, m = inst.n_customers, inst.n_depots
        c = np.array([[inst.cost(u, v) for v in range(m + n)] for u in range(m + n)])
        demand = np.array(inst.demand)
        tables = [RouteTable(c[d, m:], c[m:, m:], c[m:, d], demand, inst.vehicle_capacity) for d in range(m)]
        self.runtime.check()
        pool: list[Column] = [(d, tables[d].index[1 << i]) for d in range(m) for i in range(n)]
        in_pool = set(pool)
        self.big = 1.0 + sum(inst.opening_cost) + sum(self.cost(inst, tables, col) for col in pool)
        lb, prizes, termination, iters = None, None, "lp_converged", 0
        self.runtime.stage_reserve = min(10.0, self.runtime.time_limit / 6)
        try:
            while True:
                iters += 1
                lp = gp.Model("clrp_rmp")
                lp.Params.OutputFlag = 0
                lam = lp.addVars(len(pool), lb=0.0)
                y = lp.addVars(m, lb=0.0, ub=1.0)
                art = lp.addVars(n, lb=0.0)
                cover, cap, link = self._rows(lp, inst, tables, pool, lam, y, art)
                self.runtime.optimize(lp, role="restricted_master", allow_incumbent=False)
                if lp.Status != GRB.OPTIMAL:
                    raise RuntimeError(f"restricted master LP not optimal (status={lp.Status})")
                pi = np.array([cover[i].Pi for i in range(n)])
                prizes = [pi + demand * cap[d].Pi + np.array([link[i, d].Pi for i in range(n)]) for d in range(m)]
                new: list[Column] = []
                min_rc = 0.0
                for d in range(m):
                    rc = tables[d].reduced_costs(prizes[d], inst.vehicle_cost)
                    min_rc = min(min_rc, float(rc.min()))
                    order = np.argsort(rc)[: self.batch + len(pool)]
                    new += [(d, int(k)) for k in order if rc[k] < -1e-6 and (d, int(k)) not in in_pool][: self.batch]
                if not new:
                    # Lagrangian bound: a plan has at most n routes, each with reduced cost >= min(0, min rc).
                    lb = float(lp.ObjVal) + n * min_rc if all(art[i].X <= 1e-6 for i in range(n)) else None
                    break
                pool += new
                in_pool.update(new)
        except Exception as exc:
            if "budget" not in str(exc).lower() and "time limit" not in str(exc).lower():
                raise
            termination, lb = "time_limit", None
        self.runtime.stage_reserve = 0.0
        chosen, ub = self._integer_master(inst, tables, pool, 10.0)
        proved = lb is not None and chosen is not None and ub <= math.ceil(lb - 1e-6) + 1e-9
        if self.enumerate and lb is not None and chosen is not None and not proved:
            candidates = set(in_pool)
            for d in range(m):
                rc = tables[d].reduced_costs(prizes[d], inst.vehicle_cost)
                candidates |= {(d, int(k)) for k in np.flatnonzero(rc <= ub - lb + 1e-6)}
            enumerated, value = self._integer_master(inst, tables, sorted(candidates), self.runtime.remaining())
            if enumerated is not None and value < ub - 1e-9:
                chosen, ub = enumerated, value
            proved = self.last_master_optimal
            termination = "enumerated" if proved else "time_limit"
        if chosen is None:
            raise RuntimeError("column generation found no feasible location-routing plan")
        return {
            "best_objective": ub, "global_lb": ub if proved else (math.ceil(lb - 1e-6) if lb is not None else None), "global_ub": ub,
            "root_lp": lb, "termination_reason": "optimal" if proved else ("root_only" if not self.enumerate else termination), "iters": iters,
            "n_columns": len(pool), "total_time": time.perf_counter() - started,
            "routes": [{"depot": d, "customers": list(tables[d].path(k))} for d, k in chosen],
        }
