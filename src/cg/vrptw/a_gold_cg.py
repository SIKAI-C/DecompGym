"""Root column generation for VRPTW with exact ESPPRC labeling pricing.

`RootColumnGeneration.run` is the price-and-branch shortcut screened for this family: solve the root LP
of the set-partitioning master to optimality with exact pricing, then solve the master once as an integer
program over the generated routes. Its LP value is a valid lower bound; its integer value is only an
upper bound, because no branching is done.
"""
from __future__ import annotations

import math
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.cg.vrptw.a_gen_data import VRPTWInstance
from src.utils.reference_runtime import ReferenceRuntime


def route_cost(inst: VRPTWInstance, route: tuple[int, ...]) -> float:
    nodes = (0, *route, 0)
    return sum(inst.dist(a, b) for a, b in zip(nodes, nodes[1:]))


def espprc(inst: VRPTWInstance, duals: list[float], runtime: ReferenceRuntime, max_routes: int = 30) -> list[tuple[float, tuple[int, ...]]]:
    """Exact elementary shortest path with time windows and capacity; returns negative reduced-cost routes."""
    n, Q = inst.n_customers, inst.capacity
    d = [[inst.dist(i, j) for j in range(n + 1)] for i in range(n + 1)]
    # label: (reduced cost, time, load, visited bitmask, path)
    labels: dict[int, list[tuple[float, float, int, int, tuple[int, ...]]]] = {i: [] for i in range(1, n + 1)}
    queue: list[tuple[int, tuple[float, float, int, int, tuple[int, ...]]]] = []
    for j in range(1, n + 1):
        t = max(d[0][j], inst.ready[j])
        if t <= inst.due[j] and inst.demand[j] <= Q:
            lab = (d[0][j] - duals[j], t, inst.demand[j], 1 << j, (j,))
            labels[j].append(lab)
            queue.append((j, lab))
    found: list[tuple[float, tuple[int, ...]]] = []
    checks = 0
    while queue:
        node, (rc, t, load, mask, path) = queue.pop()
        if (rc, t, load, mask, path) not in labels[node]:
            continue  # dominated after it was queued
        checks += 1
        if checks % 2000 == 0:
            runtime.check()
        back = rc + d[node][0]
        if back < -1e-6:
            found.append((back, path))
        for j in range(1, n + 1):
            if mask >> j & 1 or load + inst.demand[j] > Q:
                continue
            tj = max(t + inst.service[node] + d[node][j], inst.ready[j])
            if tj > inst.due[j]:
                continue
            new = (rc + d[node][j] - duals[j], tj, load + inst.demand[j], mask | 1 << j, path + (j,))
            bucket = labels[j]
            if any(o[0] <= new[0] + 1e-9 and o[1] <= new[1] and o[2] <= new[2] and (o[3] | new[3]) == new[3] for o in bucket):
                continue
            labels[j] = [o for o in bucket if not (new[0] <= o[0] + 1e-9 and new[1] <= o[1] and new[2] <= o[2] and (new[3] | o[3]) == o[3])]
            labels[j].append(new)
            queue.append((j, new))
    found.sort()
    return found[:max_routes]


class RootColumnGeneration:
    def __init__(self, *, time_limit: float = 60.0, max_iters: int = 10_000, verbose: bool = False) -> None:
        self.runtime = ReferenceRuntime(time_limit)
        self.max_iters = max_iters
        self.verbose = verbose

    def run(self, instance_name: str) -> dict[str, Any]:
        self.runtime.start()
        self.runtime.stage_reserve = min(10.0, self.runtime.time_limit / 6)
        inst = VRPTWInstance.from_json_file(instance_name)
        C = range(1, inst.n_customers + 1)
        routes: list[tuple[int, ...]] = [(i,) for i in C]
        lp_bound, termination, started, it = None, "max_iters", time.perf_counter(), 0
        try:
            for it in range(1, self.max_iters + 1):
                lp = gp.Model("vrptw_rmp")
                lp.Params.OutputFlag = 0
                lam = lp.addVars(len(routes), lb=0.0)
                cover = lp.addConstrs(gp.quicksum(lam[r] for r, route in enumerate(routes) if i in route) == 1 for i in C)
                lp.setObjective(gp.quicksum(route_cost(inst, route) * lam[r] for r, route in enumerate(routes)), GRB.MINIMIZE)
                self.runtime.optimize(lp, role="restricted_master", allow_incumbent=False)
                duals = [0.0] + [cover[i].Pi for i in C]
                new = [r for _, r in espprc(inst, duals, self.runtime) if r not in routes]
                if self.verbose:
                    print(f"[cg {it}] lp={lp.ObjVal:.3f} new={len(new)}")
                if not new:
                    lp_bound, termination = float(lp.ObjVal), "lp_converged"
                    break
                routes.extend(new)
        except Exception as exc:
            if "budget" not in str(exc).lower() and "time limit" not in str(exc).lower():
                raise
            termination = "time_limit"
        self.runtime.stage_reserve = 0.0
        ip = gp.Model("vrptw_integer_rmp")
        ip.Params.OutputFlag = 0
        lam = ip.addVars(len(routes), vtype=GRB.BINARY)
        ip.addConstrs(gp.quicksum(lam[r] for r, route in enumerate(routes) if i in route) == 1 for i in C)
        ip.setObjective(gp.quicksum(route_cost(inst, route) * lam[r] for r, route in enumerate(routes)), GRB.MINIMIZE)
        self.runtime.optimize(ip, role="integer_master", allow_incumbent=True)
        if ip.SolCount == 0:
            raise RuntimeError("Integer restricted master found no solution")
        chosen = [list(routes[r]) for r in range(len(routes)) if lam[r].X > 0.5]
        return {
            "best_objective": float(ip.ObjVal), "global_lb": math.ceil(lp_bound * 10 - 1e-6) / 10 if lp_bound is not None else None,
            "global_ub": float(ip.ObjVal), "root_lp": lp_bound, "termination_reason": termination, "iters": it,
            "n_columns": len(routes), "total_time": time.perf_counter() - started, "routes": chosen,
        }
