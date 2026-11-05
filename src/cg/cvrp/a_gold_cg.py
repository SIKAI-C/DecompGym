"""Exact column generation for CVRP: root column generation, then reduced-cost route enumeration.

Master: set partitioning over routes (one artificial per customer in phase I). Pricing is exact: every
capacity-feasible customer subset has its cheapest route precomputed by a Held-Karp dynamic program
(`src.utils.subset_routes`), so the most negative reduced-cost routes are found by one vectorized product.
When pricing finds no negative column, the root LP value LB is a valid bound. An integer master over the
generated routes gives an upper bound UB; every route of a plan cheaper than UB has reduced cost at most
UB - LB under the root duals, so the integer master over all such routes (plus the generated ones) is
exact. `enumerate=False` is the price-and-branch shortcut: root column generation, then one integer master.
"""
from __future__ import annotations

import math
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.cvrp.a_gen_data import CVRPInstance
from src.utils.reference_runtime import ReferenceRuntime
from src.utils.subset_routes import RouteTable


class CVRPColumnGeneration:
    max_customers = 21

    def __init__(self, *, time_limit: float = 60.0, enumerate: bool = True, batch: int = 50) -> None:
        self.runtime = ReferenceRuntime(time_limit)
        self.enumerate = enumerate
        if batch < 1:
            raise ValueError("batch must be a positive number of columns per pricing round")
        self.batch = batch

    def _integer_master(self, table: RouteTable, columns: list[int], limit: float) -> tuple[list[int] | None, float]:
        ip = gp.Model("cvrp_integer_master")
        ip.Params.OutputFlag = 0
        ip.Params.TimeLimit = max(0.5, min(limit, self.runtime.remaining()))
        ip.Params.MIPGap, ip.Params.MIPGapAbs = 0.0, 0.5  # integer costs: exact once the bound is within 0.5
        lam = ip.addVars(len(columns), vtype=GRB.BINARY)
        ip.addConstrs(gp.quicksum(lam[k] for k, c in enumerate(columns) if table.members[c, i]) == 1 for i in range(table.n))
        ip.setObjective(gp.quicksum(float(table.cost[c]) * lam[k] for k, c in enumerate(columns)), GRB.MINIMIZE)
        self.runtime.optimize(ip, role="integer_master", allow_incumbent=True)
        self.last_master_bound = math.ceil(ip.ObjBound - 1e-6) if ip.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else None
        self.last_master_optimal = ip.Status == GRB.OPTIMAL and ip.SolCount > 0 and self.last_master_bound >= ip.ObjVal - 1e-6
        if ip.SolCount == 0:
            return None, math.inf
        chosen = [c for k, c in enumerate(columns) if lam[k].X > 0.5]
        return chosen, float(sum(table.cost[c] for c in chosen))

    def run(self, instance_name: str, *, started_at: float | None = None) -> dict[str, Any]:
        self.runtime.start(started_at)
        started = time.perf_counter()
        inst = CVRPInstance.from_json_file(instance_name)
        n = inst.n_customers
        if n > self.max_customers:
            # The subset table holds 2^n x n int64 labels plus a dense 2^n x n membership matrix (about 1.4 GB at n = 22).
            raise ValueError(f"the subset-table reference supports at most {self.max_customers} customers (got {n})")
        d = np.array([[inst.dist(i, j) for j in range(n + 1)] for i in range(n + 1)])
        table = RouteTable(d[0, 1:], d[1:, 1:], d[1:, 0], np.array(inst.demand[1:]), inst.capacity)
        self.runtime.check()
        pool = [table.index[1 << i] for i in range(n)]  # one-customer routes (every demand fits a vehicle)
        in_pool = set(pool)
        big = 1.0 + 2.0 * float(table.cost[pool].sum())
        lb, duals, termination, iters = None, None, "lp_converged", 0
        self.runtime.stage_reserve = min(10.0, self.runtime.time_limit / 6)
        try:
            while True:
                iters += 1
                lp = gp.Model("cvrp_rmp")
                lp.Params.OutputFlag = 0
                lam = lp.addVars(len(pool), lb=0.0)
                art = lp.addVars(n, lb=0.0)
                cover = lp.addConstrs(gp.quicksum(lam[k] for k, c in enumerate(pool) if table.members[c, i]) + art[i] == 1 for i in range(n))
                lp.setObjective(gp.quicksum(float(table.cost[c]) * lam[k] for k, c in enumerate(pool)) + big * art.sum(), GRB.MINIMIZE)
                self.runtime.optimize(lp, role="restricted_master", allow_incumbent=False)
                if lp.Status != GRB.OPTIMAL:
                    raise RuntimeError(f"restricted master LP not optimal (status={lp.Status})")
                duals = np.array([cover[i].Pi for i in range(n)])
                rc = table.reduced_costs(duals)
                order = np.argsort(rc)[: self.batch + len(pool)]
                new = [int(c) for c in order if rc[c] < -1e-6 and int(c) not in in_pool][: self.batch]
                if not new:
                    # Lagrangian bound from the duals themselves (valid whatever the LP's primal accuracy): a plan has at
                    # most n routes, each with reduced cost >= min(0, min rc) (pricing is exhaustive over subsets), and an
                    # artificial column has reduced cost big - dual.
                    lb = (math.fsum(duals) + n * min(0.0, float(rc.min()) - 1e-9) + math.fsum(min(0.0, big - v) for v in duals)
                          if all(art[i].X <= 1e-6 for i in range(n)) else None)
                    break
                pool += new
                in_pool.update(new)
        except Exception as exc:
            if "budget" not in str(exc).lower() and "time limit" not in str(exc).lower():
                raise
            termination, lb = "time_limit", None
        self.runtime.stage_reserve = 0.0
        chosen, ub = self._integer_master(table, pool, 10.0)
        proved = lb is not None and chosen is not None and ub <= math.ceil(lb - 1e-6) + 1e-9
        if self.enumerate and lb is not None and chosen is not None and not proved:
            # Every route of a plan cheaper than UB has reduced cost <= UB - LB under the root duals.
            candidates = set(int(c) for c in np.flatnonzero(table.reduced_costs(duals) <= ub - lb + 1e-6)) | in_pool
            enumerated, value = self._integer_master(table, sorted(candidates), self.runtime.remaining())
            if enumerated is not None and value < ub - 1e-9:
                chosen, ub = enumerated, value
            proved = self.last_master_optimal
            termination = "enumerated" if proved else "time_limit"
        if chosen is None:
            raise RuntimeError("column generation found no feasible routing plan")
        return {
            "best_objective": ub, "global_lb": ub if proved else (math.ceil(lb - 1e-6) if lb is not None else None), "global_ub": ub,
            "root_lp": lb, "termination_reason": "optimal" if proved else ("root_only" if not self.enumerate else termination), "iters": iters,
            "n_columns": len(pool), "total_time": time.perf_counter() - started, "routes": [[i + 1 for i in table.path(c)] for c in chosen],
        }
