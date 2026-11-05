"""Exact column generation for RCSP: root column generation with labelling pricing, then reduced-cost duty enumeration.

Master: set covering over duties (sum over duties containing task t >= 1; one artificial per task in phase I), so
the duals pi are nonnegative. Pricing is a resource-constrained shortest path on the acyclic connection graph with the
labels of van Rossum (2026, Section 3.1.1), as in the scale-up pricer: (reduced cost c, duty length w, time without a
break beta, break flag b, crew base). Along a connection a = (i, j): c += cost_a - pi_j, w += dur_a; the break is
taken on a break connection whenever w + timeBeforeBreak(a) <= 5.5 h (beta = timeAfterBreak(a), b = 1), otherwise
beta += dur_a; labels need w <= 9.5 h and beta <= 5.5 h (before the break beta = w). Taking the latest admissible
break is exact: a later break arc only lowers the time after the break. A duty closes on a SINK at its own base with
b = 1 and the SINK duration within both limits. Dominance at a task (same base): c, w, beta smaller or equal and b
larger or equal. Labels whose cost plus the cheapest resource-free completion cannot reach the threshold are dropped.

Bounds. For duals pi >= 0 and any plan P without redundant duties (at most n of them),
cost(P) >= sum_t pi_t + sum_{k in P} rc_k, so LB = LP + n * min(0, min rc) is valid, and every duty of a plan of cost
at most UB has reduced cost <= UB - LB. After root convergence an integer master over the generated duties gives UB;
when it is not proved optimal, every duty with reduced cost <= UB - LB is enumerated by labelling without dominance
and the integer master over them is exact. `enumerate=False` is the price-and-branch shortcut (root CG, one master).
"""
from __future__ import annotations

import math
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.cg.rcsp.a_gen_data import MAX_DUTY, MAX_WORK, RCSPInstance
from src.utils.reference_runtime import ReferenceRuntime

Duty = tuple[int, tuple[int, ...]]  # (crew base, task sequence)


class DutyPricer:
    """Labelling over the connection graph; see the module docstring."""

    def __init__(self, inst: RCSPInstance) -> None:
        n = self.n = inst.n_tasks
        self.bases = list(inst.crew_bases)
        g = inst.graph()
        self.src = {(b, j): (c[6], c[7]) for (b, j), c in g["SOURCE"].items()}
        self.snk = {(i, b): (c[6], c[7]) for (i, b), c in g["SINK"].items()}
        self.into: list[list[tuple[int, int, int, int, int]]] = [[] for _ in range(n)]  # (tail, dur, cost, tbb, tab)
        self.out: list[list[tuple[int, int]]] = [[] for _ in range(n)]  # (head, cost)
        indeg = [0] * n
        for (i, j), c in g["CONNECTION"].items():
            self.into[j].append((i, c[6], c[7], c[4], c[5]))
            self.out[i].append((j, c[7]))
            indeg[j] += 1
        order, ready = [], sorted(k for k in range(n) if indeg[k] == 0)
        while ready:
            k = ready.pop(0)
            order.append(k)
            for j, _ in self.out[k]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    ready.append(j)
        if len(order) != n:
            raise ValueError("the connection graph has a cycle")
        self.order = order

    def completion_bound(self, pi: np.ndarray) -> dict[int, list[float]]:
        """V[base][k]: the cheapest reduced-cost completion from task k to a SINK at the base, ignoring resources."""
        V = {b: [self.snk[(k, b)][1] if (k, b) in self.snk else math.inf for k in range(self.n)] for b in self.bases}
        for k in reversed(self.order):
            for b in self.bases:
                for j, cost in self.out[k]:
                    V[b][k] = min(V[b][k], cost - pi[j] + V[b][j])
        return V

    def price(self, pi: np.ndarray, threshold: float, *, enumerate_all: bool = False, max_labels: int = 2_000_000) -> list[tuple[float, Duty]]:
        """Duties with reduced cost < threshold (dominance on), or every duty with reduced cost <= threshold
        (enumerate_all: no dominance), as (rc, (base, tasks)) sorted by rc."""
        V = self.completion_bound(pi)
        eps = 1e-9
        labels: list[list[tuple]] = [[] for _ in range(self.n)]  # (c, w, beta, b, base, tasks)
        found: list[tuple[float, Duty]] = []
        total = 0
        for k in self.order:
            cand = []
            for b in self.bases:
                if (b, k) in self.src:
                    dur, cost = self.src[(b, k)]
                    cand.append((cost - pi[k], dur, dur, 0, b, (k,)))
            for (i, dur, cost, tbb, tab) in self.into[k]:
                for (c, w, beta, brk, b, tasks) in labels[i]:
                    reset = tbb >= 0 and w + tbb <= MAX_WORK
                    cand.append((c + cost - pi[k], w + dur, tab if reset else beta + dur, 1 if reset else brk, b, tasks + (k,)))
            keep = [l for l in cand if l[1] <= MAX_DUTY and l[2] <= MAX_WORK and l[0] + V[l[4]][k] <= threshold + eps]
            if not enumerate_all:
                keep.sort(key=lambda l: (l[4], l[0], l[1], l[2], -l[3]))
                front: list[tuple] = []
                for l in keep:
                    if not any(f[4] == l[4] and f[0] <= l[0] + eps and f[1] <= l[1] and f[2] <= l[2] and f[3] >= l[3] for f in front):
                        front.append(l)
                keep = front
            labels[k] = keep
            total += len(keep)
            if total > max_labels:
                raise RuntimeError(f"pricing exceeded {max_labels} labels")
            for (c, w, beta, brk, b, tasks) in keep:
                if brk and (k, b) in self.snk:
                    dur, cost = self.snk[(k, b)]
                    rc = c + cost
                    if w + dur <= MAX_DUTY and beta + dur <= MAX_WORK and (rc <= threshold + eps if enumerate_all else rc < threshold):
                        found.append((rc, (b, tasks)))
        found.sort(key=lambda t: t[0])
        return found


class RCSPColumnGeneration:
    def __init__(self, *, time_limit: float = 60.0, enumerate: bool = True, batch: int = 30, tol: float = 1e-6) -> None:
        self.runtime = ReferenceRuntime(time_limit)
        self.enumerate = enumerate
        if batch < 1:
            raise ValueError("batch must be a positive number of columns per pricing round")
        self.batch, self.tol = batch, tol

    @staticmethod
    def duty_cost(inst: RCSPInstance, duty: Duty, g: dict[str, dict] | None = None) -> int:
        g = inst.graph() if g is None else g
        b, tasks = duty
        chain = [g["SOURCE"][(b, tasks[0])]] + [g["CONNECTION"][p] for p in zip(tasks, tasks[1:])] + [g["SINK"][(tasks[-1], b)]]
        return int(sum(c[7] for c in chain))

    def _integer_master(self, n: int, cols: list[Duty], cost: dict[Duty, int], limit: float):
        ip = gp.Model("rcsp_integer_master")
        ip.Params.OutputFlag = 0
        ip.Params.TimeLimit = max(0.5, min(limit, self.runtime.remaining()))
        ip.Params.MIPGap, ip.Params.MIPGapAbs = 0.0, 0.5  # integer costs: exact once the bound is within 0.5
        lam = ip.addVars(len(cols), vtype=GRB.BINARY)
        for t in range(n):
            ip.addConstr(gp.quicksum(lam[k] for k, (_, tasks) in enumerate(cols) if t in tasks) >= 1)
        ip.setObjective(gp.quicksum(cost[col] * lam[k] for k, col in enumerate(cols)), GRB.MINIMIZE)
        self.runtime.optimize(ip, role="integer_master", allow_incumbent=True)
        self.last_master_bound = math.ceil(ip.ObjBound - 1e-6) if ip.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else None
        self.last_master_optimal = ip.Status == GRB.OPTIMAL and ip.SolCount > 0 and self.last_master_bound >= ip.ObjVal - 1e-6
        if ip.SolCount == 0:
            return None, math.inf
        chosen = [col for k, col in enumerate(cols) if lam[k].X > 0.5]
        return chosen, float(sum(cost[col] for col in chosen))

    def run(self, instance_name: str, *, started_at: float | None = None) -> dict[str, Any]:
        self.runtime.start(started_at)
        started = time.perf_counter()
        inst = RCSPInstance.from_json_file(instance_name)
        n = inst.n_tasks
        pricer = DutyPricer(inst)
        graph = inst.graph()
        self.runtime.check()
        pool: list[Duty] = []
        cost: dict[Duty, int] = {}
        big = 10.0 * (MAX_DUTY + max(c[7] for c in inst.connections))
        lb, pi, termination, iters = None, None, "lp_converged", 0
        self.runtime.stage_reserve = min(10.0, self.runtime.time_limit / 6)
        try:
            while True:
                iters += 1
                lp = gp.Model("rcsp_rmp")
                lp.Params.OutputFlag = 0
                lam = lp.addVars(len(pool), lb=0.0)
                art = lp.addVars(n, lb=0.0)
                cover = [lp.addConstr(gp.quicksum(lam[k] for k, (_, tasks) in enumerate(pool) if t in tasks) + art[t] >= 1) for t in range(n)]
                lp.setObjective(gp.quicksum(cost[col] * lam[k] for k, col in enumerate(pool)) + big * art.sum(), GRB.MINIMIZE)
                self.runtime.optimize(lp, role="restricted_master", allow_incumbent=False)
                if lp.Status != GRB.OPTIMAL:
                    raise RuntimeError(f"restricted master LP not optimal (status={lp.Status})")
                pi = np.array([max(0.0, cover[t].Pi) for t in range(n)])
                self.runtime.check()
                priced = pricer.price(pi, -self.tol)
                new = [(rc, col) for rc, col in priced if col not in cost][: self.batch]
                if not new:
                    # Exact pricing: every duty has reduced cost >= min(-tol, the most negative one returned).
                    min_rc = min([-self.tol] + [rc for rc, _ in priced[:1]])
                    lb = float(lp.ObjVal) + n * min_rc if all(art[t].X <= 1e-6 for t in range(n)) else None
                    break
                for _, col in new:
                    cost[col] = self.duty_cost(inst, col, graph)
                    pool.append(col)
        except Exception as exc:
            if "budget" not in str(exc).lower() and "time limit" not in str(exc).lower():
                raise
            termination, lb = "time_limit", None
        self.runtime.stage_reserve = 0.0
        chosen, ub = self._integer_master(n, pool, cost, 10.0) if pool else (None, math.inf)
        proved = lb is not None and chosen is not None and ub <= math.ceil(lb - 1e-6) + 1e-9
        if self.enumerate and lb is not None and chosen is not None and not proved:
            for _, col in pricer.price(pi, ub - lb + 1e-6, enumerate_all=True):
                if col not in cost:
                    cost[col] = self.duty_cost(inst, col, graph)
            candidates = sorted(cost)
            enumerated, value = self._integer_master(n, candidates, cost, self.runtime.remaining())
            if enumerated is not None and value < ub - 1e-9:
                chosen, ub = enumerated, value
            proved = self.last_master_optimal
            termination = "enumerated" if proved else "time_limit"
        if chosen is None:
            raise RuntimeError("column generation found no feasible duty plan")
        return {
            "best_objective": ub, "global_lb": ub if proved else (math.ceil(lb - 1e-6) if lb is not None else None), "global_ub": ub,
            "root_lp": lb, "termination_reason": "optimal" if proved else ("root_only" if not self.enumerate else termination), "iters": iters,
            "n_columns": len(pool), "total_time": time.perf_counter() - started,
            "duties": [{"depot": b, "tasks": list(tasks)} for b, tasks in chosen],
        }
