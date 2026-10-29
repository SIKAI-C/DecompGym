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


def time_grid(inst, n_nodes: int):
    """Distances, windows and service times in integer tenths, so feasibility tests are exact.

    Distances are floor(10 * euclidean) by definition; windows and service times must lie on the 0.1 grid.
    """
    to10 = lambda v: int(round(10 * v))
    for v in (*inst.ready, *inst.due, *inst.service):
        if abs(10 * v - round(10 * v)) > 1e-6:
            raise ValueError("time data must lie on the 0.1 grid")
    V = range(n_nodes + 1)
    d10 = [[math.floor(10.0 * math.hypot(inst.x[i] - inst.x[j], inst.y[i] - inst.y[j])) for j in V] for i in V]
    return d10, [to10(v) for v in inst.ready], [to10(v) for v in inst.due], [to10(v) for v in inst.service]


def artificial_cost(inst, n_nodes: int) -> float:
    """Exceeds the cost of any route set: every route costs at most twice its customers' depot distances (+ rounding)."""
    return 1.0 + sum(2 * inst.dist(0, i) + 0.2 for i in range(1, n_nodes + 1)) * 2


def espprc(inst: VRPTWInstance, duals: list[float], runtime: ReferenceRuntime, max_routes: int = 30, *, forbidden: frozenset = frozenset(),
           neighbors: dict[int, list[int]] | None = None) -> list[tuple[float, tuple[int, ...]]]:
    """Elementary shortest path with time windows and capacity; returns negative reduced-cost routes.

    Exact unless `neighbors` restricts each extension to a few nearest customers (heuristic pricing).
    `forbidden` holds arcs removed by branching (depot = 0).
    """
    n, Q = inst.n_customers, inst.capacity
    d10, R, D, S = time_grid(inst, n)
    d = [[v / 10.0 for v in row] for row in d10]
    # label: (reduced cost, time in tenths, load, visited bitmask, path)
    labels: dict[int, list[tuple[float, int, int, int, tuple[int, ...]]]] = {i: [] for i in range(1, n + 1)}
    queue: list[tuple[int, tuple[float, int, int, int, tuple[int, ...]]]] = []
    for j in range(1, n + 1):
        t = max(R[0] + d10[0][j], R[j])
        if t <= D[j] and inst.demand[j] <= Q and (0, j) not in forbidden:
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
        if back < -1e-6 and (node, 0) not in forbidden and t + S[node] + d10[node][0] <= D[0]:
            found.append((back, path))
        for j in (neighbors[node] if neighbors else range(1, n + 1)):
            if mask >> j & 1 or load + inst.demand[j] > Q or (node, j) in forbidden:
                continue
            tj = max(t + S[node] + d10[node][j], R[j])
            if tj > D[j]:
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

    def run(self, instance_name: str, *, started_at: float | None = None) -> dict[str, Any]:
        self.runtime.start(started_at)
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


class BranchAndPrice:
    """Exact branch-and-price: set-partitioning master, heuristic-then-exact ESPPRC pricing, branching on arc flows.

    Child x_ij = 0 forbids arc (i, j); child x_ij = 1 forbids every other arc leaving i and entering j
    (depot arcs excepted). Node bounds are rounded up to the one-decimal distance grid.
    """

    instance_class = VRPTWInstance

    def __init__(self, *, time_limit: float = 60.0, n_neighbors: int = 6) -> None:
        self.runtime = ReferenceRuntime(time_limit)
        self.n_neighbors = n_neighbors

    def pricer(self, inst, duals, *, forbidden, neighbors):
        return espprc(inst, duals, self.runtime, forbidden=forbidden, neighbors=neighbors)

    def n_nodes(self, inst) -> int:
        return inst.n_customers

    def initial_routes(self, inst) -> list[tuple[int, ...]]:
        return [(i,) for i in range(1, self.n_nodes(inst) + 1)]

    def route_feasible(self, inst, route: tuple[int, ...]) -> bool:
        from src.cg.vrptw.d_verifier import route_check

        return route_check(inst.to_dict(), list(route))[0] is None

    def _node_lp(self, inst, pool, forbidden):
        C = range(1, self.n_nodes(inst) + 1)
        while True:
            cols = [r for r in pool if not any(a in forbidden for a in zip((0, *r), (*r, 0)))]
            lp = gp.Model("vrptw_node")
            lp.Params.OutputFlag = 0
            lam = lp.addVars(len(cols), lb=0.0)
            art = lp.addVars(C, lb=0.0)
            cover = lp.addConstrs(gp.quicksum(lam[k] for k, r in enumerate(cols) if i in r) + art[i] == 1 for i in C)
            lp.setObjective(gp.quicksum(route_cost(inst, r) * lam[k] for k, r in enumerate(cols)) + artificial_cost(inst, len(C)) * art.sum(), GRB.MINIMIZE)
            self.runtime.optimize(lp, role="restricted_master", allow_incumbent=False)
            duals = [0.0] + [cover[i].Pi for i in C]
            new = [r for _, r in self.pricer(inst, duals, forbidden=forbidden, neighbors=self.neighbors) if r not in self.seen]
            if not new:
                new = [r for _, r in self.pricer(inst, duals, forbidden=forbidden, neighbors=None) if r not in self.seen]
            if not new:
                if any(art[i].X > 1e-6 for i in C):
                    return None, {}
                return float(lp.ObjVal), {cols[k]: lam[k].X for k in range(len(cols)) if lam[k].X > 1e-6}
            for r in new:
                self.seen.add(r)
                pool.append(r)

    def _integer_master(self, inst, pool, limit):
        C = range(1, self.n_nodes(inst) + 1)
        ip = gp.Model("vrptw_integer_master")
        ip.Params.OutputFlag = 0
        ip.Params.Threads = 1
        ip.Params.TimeLimit = max(0.5, min(limit, self.runtime.remaining()))
        lam = ip.addVars(len(pool), vtype=GRB.BINARY)
        ip.addConstrs(gp.quicksum(lam[k] for k, r in enumerate(pool) if i in r) == 1 for i in C)
        ip.setObjective(gp.quicksum(route_cost(inst, r) * lam[k] for k, r in enumerate(pool)), GRB.MINIMIZE)
        ip.optimize()
        if ip.SolCount == 0:
            return None, math.inf
        return [list(pool[k]) for k in range(len(pool)) if lam[k].X > 0.5], float(ip.ObjVal)

    def run(self, instance_name: str, *, started_at: float | None = None) -> dict[str, Any]:
        self.runtime.start(started_at)
        started = time.perf_counter()
        inst = self.instance_class.from_json_file(instance_name)
        n = self.n_nodes(inst)
        self.neighbors = {i: sorted((j for j in range(1, n + 1) if j != i), key=lambda j: inst.dist(i, j))[: self.n_neighbors] for i in range(n + 1)}
        # Seed routes enter the master and the incumbent only if feasible; otherwise phase I (the artificial
        # columns) and pricing must cover the customers.
        seeds = self.initial_routes(inst)
        pool: list[tuple[int, ...]] = [r for r in seeds if self.route_feasible(inst, r)]
        self.seen = set(pool)
        if len(pool) == len(seeds):
            best, ub = [list(r) for r in pool], sum(route_cost(inst, r) for r in pool)
        else:
            best, ub = None, math.inf
        stack, root_lb, nodes, termination = [frozenset()], None, 0, "optimal"
        snap = lambda v: math.ceil(v * 10 - 1e-6) / 10
        try:
            while stack:
                forbidden = stack.pop()
                nodes += 1
                lp, lam = self._node_lp(inst, pool, forbidden)
                if lp is None or snap(lp) >= ub - 1e-9:
                    root_lb = root_lb if root_lb is not None or lp is None else snap(lp)
                    continue
                root_lb = snap(lp) if root_lb is None else root_lb
                routes, value = self._integer_master(inst, pool, 2.0)
                if routes is not None and value < ub - 1e-9:
                    best, ub = routes, value
                if snap(lp) >= ub - 1e-9:
                    continue
                flow: dict[tuple[int, int], float] = {}
                for r, v in lam.items():
                    for a in zip((0, *r), (*r, 0)):
                        flow[a] = flow.get(a, 0.0) + v
                frac = [(abs(v - 0.5), a) for a, v in flow.items() if 1e-6 < v < 1 - 1e-6]
                if not frac:
                    routes = [list(r) for r, v in lam.items() if v > 0.5]
                    value = sum(route_cost(inst, tuple(r)) for r in routes)
                    if value < ub - 1e-9:
                        best, ub = routes, value
                    continue
                i, j = min(frac)[1]
                one = set(forbidden)
                one |= {(i, k) for k in range(n + 1) if k != j and i != 0}
                one |= {(k, j) for k in range(n + 1) if k != i and j != 0}
                stack.append(forbidden | {(i, j)})
                stack.append(frozenset(one))
        except Exception as exc:
            if "budget" not in str(exc).lower() and "time limit" not in str(exc).lower():
                raise
            termination = "time_limit"
        if best is None:
            raise RuntimeError("branch-and-price found no feasible routing plan")
        proved = termination == "optimal" and not stack
        return {
            "best_objective": round(ub, 6), "global_lb": round(ub, 6) if proved else root_lb, "global_ub": round(ub, 6),
            "root_lp": root_lb, "termination_reason": "optimal" if proved else termination, "iters": nodes,
            "n_columns": len(pool), "total_time": time.perf_counter() - started, "routes": best,
        }
