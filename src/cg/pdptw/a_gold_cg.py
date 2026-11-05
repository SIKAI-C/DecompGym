"""Branch-and-price for PDPTW: the VRPTW branch-and-price with pickup-and-delivery pricing.

Pricing is an elementary shortest path where a label also carries its open requests (picked up, not yet
delivered): a delivery is allowed only when its request is open, and a route may return to the depot only
when nothing is open. Dominance requires equal open sets.
"""
from __future__ import annotations

from src.cg.pdptw.a_gen_data import PDPTWInstance
from src.cg.vrptw.a_gold_cg import BranchAndPrice, time_grid
from src.utils.reference_runtime import ReferenceRuntime


def pdp_espprc(inst: PDPTWInstance, duals: list[float], runtime: ReferenceRuntime, max_routes: int = 30, *, forbidden=frozenset(), neighbors=None):
    n, Q, N = inst.n_requests, inst.capacity, 2 * inst.n_requests
    d10, R, D, S = time_grid(inst, N)
    d = [[v / 10.0 for v in row] for row in d10]
    labels: dict[int, list[tuple]] = {i: [] for i in range(1, N + 1)}
    queue: list[tuple[int, tuple]] = []
    for p in range(1, n + 1):
        t = max(R[0] + d10[0][p], R[p])
        if t <= D[p] and inst.demand[p] <= Q and (0, p) not in forbidden:
            lab = (d[0][p] - duals[p], t, inst.demand[p], 1 << p, 1 << p, (p,))
            labels[p].append(lab)
            queue.append((p, lab))
    found, checks = [], 0
    while queue:
        node, lab = queue.pop()
        if lab not in labels[node]:
            continue
        rc, t, load, mask, open_, path = lab
        checks += 1
        if checks % 2000 == 0:
            runtime.check()
        if open_ == 0 and (node, 0) not in forbidden and t + S[node] + d10[node][0] <= D[0]:
            back = rc + d[node][0]
            if back < -1e-6:
                found.append((back, path))
        candidates = neighbors[node] + [n + k for k in range(1, n + 1) if open_ >> k & 1] if neighbors else range(1, N + 1)
        for j in candidates:
            if mask >> j & 1 or (node, j) in forbidden:
                continue
            if j > n and not open_ >> (j - n) & 1:
                continue  # delivery of a request that is not on board
            if j <= n and load + inst.demand[j] > Q:
                continue
            tj = max(t + S[node] + d10[node][j], R[j])
            if tj > D[j]:
                continue
            new_open = open_ | (1 << j) if j <= n else open_ & ~(1 << (j - n))
            new = (rc + d[node][j] - duals[j], tj, load + inst.demand[j], mask | 1 << j, new_open, path + (j,))
            bucket = labels[j]
            if any(o[4] == new_open and o[0] <= new[0] + 1e-9 and o[1] <= new[1] and (o[3] | new[3]) == new[3] for o in bucket):
                continue
            labels[j] = [o for o in bucket if not (o[4] == new_open and new[0] <= o[0] + 1e-9 and new[1] <= o[1] and (new[3] | o[3]) == o[3])]
            labels[j].append(new)
            queue.append((j, new))
    found.sort()
    return found[:max_routes]


class PDPTWBranchAndPrice(BranchAndPrice):
    instance_class = PDPTWInstance

    def n_nodes(self, inst) -> int:
        return 2 * inst.n_requests

    def pricer(self, inst, duals, *, forbidden, neighbors):
        # Heuristic neighbor lists cover pickups; deliveries of open requests are always allowed.
        nb = {i: [j for j in neighbors[i] if j <= inst.n_requests] for i in neighbors} if neighbors else None
        return pdp_espprc(inst, duals, self.runtime, forbidden=forbidden, neighbors=nb)

    def initial_routes(self, inst):
        return [(k, inst.n_requests + k) for k in range(1, inst.n_requests + 1)]

    def route_feasible(self, inst, route):
        from src.cg.pdptw.d_verifier import route_issue

        return route_issue(inst.to_dict(), list(route)) is None

class PDPTWRootColumnGeneration(PDPTWBranchAndPrice):
    """The price-and-branch shortcut: root column generation, then one integer master over its routes."""

    def run(self, instance_name: str, *, started_at: float | None = None):
        import math
        import time

        self.runtime.start(started_at)
        started = time.perf_counter()
        inst = self.instance_class.from_json_file(instance_name)
        n = self.n_nodes(inst)
        self.neighbors = {i: sorted((j for j in range(1, n + 1) if j != i), key=lambda j: inst.dist(i, j))[: self.n_neighbors] for i in range(n + 1)}
        seeds = self.initial_routes(inst)
        pool = [r for r in seeds if self.route_feasible(inst, r)]
        self.seen = set(pool)
        seed_plan = [list(r) for r in pool] if len(pool) == len(seeds) else None
        from src.cg.vrptw.a_gold_cg import route_cost
        seed_value = sum(route_cost(inst, r) for r in pool) if seed_plan is not None else math.inf
        self.runtime.stage_reserve = min(10.0, self.runtime.time_limit / 6)
        try:
            lp, _ = self._node_lp(inst, pool, frozenset())
            termination = "root_only"
        except Exception as exc:
            if "budget" not in str(exc).lower() and "time limit" not in str(exc).lower():
                raise
            lp, termination = None, "time_limit"  # keep the routes generated so far
        self.runtime.stage_reserve = 0.0
        routes, value = self._integer_master(inst, pool, self.runtime.remaining())
        if routes is None or value > seed_value:
            routes, value = seed_plan, seed_value  # keep the verified seed plan unless the master improves it
        if routes is None:
            raise RuntimeError("root column generation found no feasible routing plan")
        return {"best_objective": round(value, 6), "global_lb": math.ceil(lp * 10 - 1e-6) / 10 if lp is not None else None, "global_ub": round(value, 6), "termination_reason": termination,
                "iters": 1, "n_columns": len(pool), "total_time": time.perf_counter() - started, "routes": routes}
