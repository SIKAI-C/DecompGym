"""Branch-and-price for "partition items into the fewest feasible groups" problems.

A group is feasible when no two of its items conflict and (if a capacity is given) its total weight
fits. Graph coloring (groups = independent sets, no capacity) and bin packing with conflicts
(groups = conflict-free bins) are both of this form.

Master (covering):  min sum_S lambda_S   s.t.  sum_{S contains i} lambda_S >= 1,  lambda >= 0.
Feasible groups are closed under taking subsets, so a covering solution can always be turned into a
partition by deleting repeated items.
Pricing:  max sum_i pi_i z_i  over feasible groups (an independent-set / conflict-knapsack MIP);
a column improves the LP iff its value exceeds 1.
Branching (Ryan-Foster): pick items i, j whose joint coverage is fractional; one child requires them
in the same group (z_i = z_j in pricing), the other in different groups (z_i + z_j <= 1).
Each node's bound is ceil(LP - eps), valid because the objective counts groups.
"""
from __future__ import annotations

import math
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.utils.reference_runtime import ReferenceRuntime


EPS = 1e-6
ARTIFICIAL_COST = 1e4


class PartitionBranchAndPrice:
    def __init__(self, n_items: int, conflicts: list[tuple[int, int]], *, weights: list[int] | None = None, capacity: int | None = None,
                 time_limit: float = 60.0, root_only: bool = False, max_columns_per_pricing: int = 10) -> None:
        self.n = n_items
        self.conflicts = sorted({(min(a, b), max(a, b)) for a, b in conflicts if a != b})
        self.weights = weights
        self.capacity = capacity
        self.runtime = ReferenceRuntime(time_limit)
        self.root_only = root_only  # True: the price-and-branch shortcut (root CG + one integer master)
        self.k = max_columns_per_pricing
        self.adj = [set() for _ in range(n_items)]
        for a, b in self.conflicts:
            self.adj[a].add(b)
            self.adj[b].add(a)

    # --- feasibility --------------------------------------------------------------
    def feasible(self, group: frozenset[int]) -> bool:
        if any(self.adj[i] & group for i in group):
            return False
        return self.capacity is None or sum(self.weights[i] for i in group) <= self.capacity

    def respects(self, group: frozenset[int], same, differ) -> bool:
        return all((i in group) == (j in group) for i, j in same) and not any(i in group and j in group for i, j in differ)

    # --- heuristics -----------------------------------------------------------------
    def greedy(self, same=(), differ=()) -> list[frozenset[int]] | None:
        """First-fit over merged 'same' units, largest first; None if some unit is infeasible."""
        parent = list(range(self.n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, j in same:
            parent[find(i)] = find(j)
        units: dict[int, set[int]] = {}
        for i in range(self.n):
            units.setdefault(find(i), set()).add(i)
        blocked = {(min(a, b), max(a, b)) for a, b in differ}
        size = (lambda u: sum(self.weights[i] for i in u)) if self.weights else (lambda u: len(set().union(*(self.adj[i] for i in u))))
        groups: list[set[int]] = []
        for unit in sorted(units.values(), key=size, reverse=True):
            if not self.feasible(frozenset(unit)):
                return None
            for g in groups:
                cand = frozenset(g | unit)
                if self.feasible(cand) and not any((min(a, b), max(a, b)) in blocked for a in unit for b in g):
                    g |= unit
                    break
            else:
                groups.append(set(unit))
        return [frozenset(g) for g in groups]

    # --- pricing ----------------------------------------------------------------------
    def price(self, duals: list[float], same, differ) -> tuple[list[frozenset[int]], float]:
        """Exact pricing; returns improving groups and the optimal pricing value (for the LP bound)."""
        m = gp.Model("pricing")
        m.Params.OutputFlag = 0
        z = m.addVars(self.n, vtype=GRB.BINARY)
        for a, b in self.conflicts:
            m.addConstr(z[a] + z[b] <= 1)
        if self.capacity is not None:
            m.addConstr(gp.quicksum(self.weights[i] * z[i] for i in range(self.n)) <= self.capacity)
        for i, j in same:
            m.addConstr(z[i] == z[j])
        for i, j in differ:
            m.addConstr(z[i] + z[j] <= 1)
        m.setObjective(gp.quicksum(duals[i] * z[i] for i in range(self.n)), GRB.MAXIMIZE)
        m.Params.PoolSolutions = self.k
        m.Params.PoolSearchMode = 1
        self.runtime.optimize(m, role="pricing", allow_incumbent=False)
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"pricing not optimal (status {m.Status})")
        found = []
        for s in range(m.SolCount):
            m.Params.SolutionNumber = s
            if m.PoolObjVal > 1 + EPS:
                found.append(frozenset(i for i in range(self.n) if z[i].Xn > 0.5))
        return found, float(m.ObjBound)

    # --- one node -------------------------------------------------------------------------
    def column_generation(self, pool: list[frozenset[int]], same, differ) -> tuple[float | None, dict[frozenset[int], float]]:
        """Solve the node LP to optimality; returns (LP value or None if infeasible, positive lambdas)."""
        while True:
            cols = [c for c in pool if self.respects(c, same, differ)]
            lp = gp.Model("rmp")
            lp.Params.OutputFlag = 0
            lam = lp.addVars(len(cols), lb=0.0)
            art = lp.addVars(self.n, lb=0.0)
            cover = [lp.addConstr(gp.quicksum(lam[k] for k, c in enumerate(cols) if i in c) + art[i] >= 1) for i in range(self.n)]
            lp.setObjective(lam.sum() + ARTIFICIAL_COST * art.sum(), GRB.MINIMIZE)
            self.runtime.optimize(lp, role="restricted_master", allow_incumbent=False)
            duals = [c.Pi for c in cover]
            new, best = self.price(duals, same, differ)
            new = [c for c in new if c not in pool]
            if not new:
                if any(art[i].X > EPS for i in range(self.n)):
                    return None, {}
                # With complete pricing, the LP value is the node's LP optimum.
                return float(lp.ObjVal), {cols[k]: lam[k].X for k in range(len(cols)) if lam[k].X > EPS}
            pool.extend(new)

    def integer_master(self, pool: list[frozenset[int]], limit: float) -> list[frozenset[int]] | None:
        ip = gp.Model("integer_master")
        ip.Params.OutputFlag = 0
        lam = ip.addVars(len(pool), vtype=GRB.BINARY)
        for i in range(self.n):
            ip.addConstr(gp.quicksum(lam[k] for k, c in enumerate(pool) if i in c) >= 1)
        ip.setObjective(lam.sum(), GRB.MINIMIZE)
        ip.Params.TimeLimit = max(0.5, min(limit, self.runtime.remaining()))
        ip.Params.Threads = 1
        ip.optimize()
        if ip.SolCount == 0:
            return None
        return self.to_partition([pool[k] for k in range(len(pool)) if lam[k].X > 0.5])

    def to_partition(self, groups: list[frozenset[int]]) -> list[frozenset[int]]:
        seen: set[int] = set()
        out = []
        for g in groups:
            g = frozenset(g - seen)
            if g:
                out.append(g)
                seen |= g
        return out

    # --- driver -----------------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        self.runtime.start()
        started = time.perf_counter()
        best = self.greedy()
        pool = list({frozenset([i]) for i in range(self.n)} | set(best))
        stack = [((), ())]
        root_bound, nodes, termination = None, 0, "optimal"
        try:
            while stack:
                same, differ = stack.pop()
                nodes += 1
                lp, lam = self.column_generation(pool, same, differ)
                if lp is None:
                    continue
                bound = math.ceil(lp - EPS)
                if root_bound is None:
                    root_bound = bound
                if bound >= len(best):
                    continue
                # Cheap primal step: integer master over all columns generated so far.
                cand = self.integer_master(pool, limit=2.0 if not self.root_only else self.runtime.remaining())
                if cand is not None and len(cand) < len(best):
                    best = cand
                if self.root_only or bound >= len(best):
                    if self.root_only:
                        termination = "root_only"
                    continue
                # Ryan-Foster: the pair whose joint coverage is closest to 1/2.
                pair, score = None, 1.0
                for i in range(self.n):
                    for j in range(i + 1, self.n):
                        if j in self.adj[i]:
                            continue
                        together = sum(v for c, v in lam.items() if i in c and j in c)
                        if EPS < together < 1 - EPS and abs(together - 0.5) < score:
                            pair, score = (i, j), abs(together - 0.5)
                if pair is None:
                    # LP integral up to repeated cover; the LP columns themselves form a solution.
                    sol = self.to_partition([c for c, v in lam.items() if v > 0.5])
                    if sum(len(g) for g in sol) == self.n and len(sol) < len(best):
                        best = sol
                    continue
                stack.append((same, differ + (pair,)))
                stack.append((same + (pair,), differ))
        except Exception as exc:
            if "budget" not in str(exc).lower() and "time limit" not in str(exc).lower():
                raise
            termination = "time_limit"
        # Proved when the search finished, or when the best plan meets the root bound.
        proved = (termination == "optimal" and not stack and not self.root_only) or (root_bound is not None and len(best) <= root_bound)
        return {
            "best_objective": len(best), "global_lb": len(best) if proved else root_bound, "global_ub": len(best),
            "root_bound": root_bound, "termination_reason": "optimal" if proved else termination,
            "nodes": nodes, "n_columns": len(pool), "total_time": time.perf_counter() - started,
            "groups": [sorted(g) for g in best],
        }


def solve_partition_mip(n_items: int, conflicts: list[tuple[int, int]], *, weights: list[int] | None = None, capacity: int | None = None,
                        upper_bound: int, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assignment MIP: x[i, g] puts item i in group g, y[g] opens group g (g < upper_bound).

    Symmetry breaking: groups open in order (y_g >= y_{g+1}) and item i may only use groups 0..i.
    """
    G = range(upper_bound)
    m = gp.Model("partition_monolithic")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    for k, v in (params or {}).items():
        m.setParam(k, v)
    x = m.addVars([(i, g) for i in range(n_items) for g in G if g <= i], vtype=GRB.BINARY)
    y = m.addVars(G, vtype=GRB.BINARY)
    m.addConstrs(x.sum(i, "*") == 1 for i in range(n_items))
    m.addConstrs(x[i, g] <= y[g] for (i, g) in x)
    m.addConstrs(y[g] >= y[g + 1] for g in G if g + 1 < upper_bound)
    for a, b in conflicts:
        for g in G:
            if (a, g) in x and (b, g) in x:
                m.addConstr(x[a, g] + x[b, g] <= y[g])
    if capacity is not None:
        m.addConstrs(gp.quicksum(weights[i] * x[i, g] for i in range(n_items) if (i, g) in x) <= capacity * y[g] for g in G)
    m.setObjective(y.sum(), GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Baseline model not optimal (status={m.Status})")
    groups = [sorted(i for i in range(n_items) if (i, g) in x and x[i, g].X > 0.5) for g in G]
    return {"best_objective": float(m.ObjVal), "global_lb": float(m.ObjBound), "global_ub": float(m.ObjVal), "groups": [g for g in groups if g], "runtime": m.Runtime}
