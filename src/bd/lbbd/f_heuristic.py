"""Decomposition-free controls for planning and scheduling (minimum cost; scale-up study).

Both controls start from the shared basic initializer (a_scheduling.greedy_solution), publish every improving
complete schedule, and have the solver interface of the decomposition methods.

- `PSCPSatMonolithic` (exact, no decomposition): the monolithic CP model solved by CP-SAT (OR-Tools, one worker,
  i.e. one thread like every other method): for every eligible (facility i, job j) an optional fixed-size interval
  with presence b_ij and start in [r_j, d_j - p_ij]; exactly one presence per job; one cumulative constraint per
  facility over its optional intervals; minimize sum f_ij b_ij; the initializer is the solution hint. Every solution
  callback publishes the solution; CP-SAT's proved objective bound is published as the lower bound.
- `PSLocalSearch` (heuristic, no bound): iterated local search over assignments. Moves: relocate a job to a cheaper
  facility, swap two jobs between facilities, and a two-job ejection chain (j moves to a cheaper facility v and a job
  k of v moves elsewhere); a move is taken when it lowers the cost and every changed facility stays schedulable. The
  schedulability test is the same exact facility oracle the LBBD methods use (a_scheduling.FacilityOracle: caches,
  relaxation, SGS, then CP-SAT capped at `check_cap` seconds per test; an UNKNOWN answer rejects the move). Kicks move
  2-5 random jobs to random facilities with room; the walk returns to the best solution when a kick does not improve.
  Every new global incumbent is published as soon as a move produces it (not only at the end of a descent).
  There is no master problem and no cut: the search never learns from infeasible sets beyond the oracle's cache.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from src.bd.lbbd.a_data import PSInstance, load_instance, schedule_cost
from src.bd.lbbd.a_scheduling import FEASIBLE, FacilityOracle, Relaxation, bits, greedy_solution, summarize
from src.bd.lbbd.b_mip_xl import integral_costs

INF = float("inf")


class _Publisher:
    def __init__(self, inst: PSInstance, on_incumbent: Callable, deadline: float) -> None:
        self.inst, self.on_incumbent, self.deadline = inst, on_incumbent, deadline
        self.unit = 1.0 if integral_costs(inst) else 1e-6
        self.ub, self.best, self.count, self.stage = INF, None, 0, None

    def offer(self, sol: list[list[int]], stage: str) -> bool:
        value = schedule_cost(self.inst, sol)
        if value is None:
            raise RuntimeError(f"control built an infeasible schedule ({stage})")
        if value < self.ub - 0.5 * self.unit and time.perf_counter() < self.deadline:
            self.ub, self.best, self.stage = value, [list(v) for v in sol], stage
            self.count += 1
            self.on_incumbent(self.best)  # one argument, the runner's control convention
            return True
        return False


# ---------------------------------------------------------------- monolithic CP-SAT

class PSCPSatMonolithic:
    """Monolithic CP-SAT control; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 hint: bool = True) -> None:
        self.time_limit, self.threads, self.seed, self.hint = time_limit, threads, seed, hint
        self.on_incumbent = on_incumbent or (lambda sol: None)
        self.on_bound = on_bound or (lambda bound: None)

    def run(self, data_path: str) -> dict[str, Any]:
        from ortools.sat.python import cp_model
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = load_instance(data_path)
        pub = _Publisher(inst, self.on_incumbent, deadline)
        g = greedy_solution(inst)
        if g is not None:
            pub.offer(g[0], "initializer")
        model = cp_model.CpModel()
        el = inst.eligible
        pres: dict[tuple[int, int], Any] = {}
        start: dict[tuple[int, int], Any] = {}
        ivs: list[list[Any]] = [[] for _ in range(inst.m)]
        dem: list[list[int]] = [[] for _ in range(inst.m)]
        for i in range(inst.m):
            for j in range(inst.n):
                if not el[i, j]:
                    continue
                p = int(inst.proc[i, j])
                b = model.new_bool_var(f"b{i}_{j}")
                s = model.new_int_var(int(inst.release[j]), int(inst.deadline[j]) - p, f"s{i}_{j}")
                ivs[i].append(model.new_optional_fixed_size_interval_var(s, p, b, f"iv{i}_{j}"))
                dem[i].append(int(inst.rate[i, j]))
                pres[i, j], start[i, j] = b, s
        for j in range(inst.n):
            model.add_exactly_one([pres[i, j] for i in range(inst.m) if el[i, j]])
        for i in range(inst.m):
            if ivs[i]:
                model.add_cumulative(ivs[i], dem[i], int(inst.capacity[i]))
        model.minimize(sum(int(inst.cost[i, j]) * b for (i, j), b in pres.items()))
        if self.hint and g is not None:
            for j, (i, s) in enumerate(g[0]):
                for k in range(inst.m):
                    if el[k, j]:
                        model.add_hint(pres[k, j], k == i)
                model.add_hint(start[i, j], int(s))
        state = {"bound": -INF, "solutions": 0}
        outer = self

        class Callback(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self) -> None:
                if time.perf_counter() >= deadline:
                    return
                state["solutions"] += 1
                sol = [None] * inst.n
                for (i, j), b in pres.items():
                    if self.boolean_value(b):
                        sol[j] = [i, int(self.value(start[i, j]))]
                pub.offer(sol, "cpsat")
                bnd = float(self.best_objective_bound)
                if bnd > state["bound"] and time.perf_counter() < deadline:
                    state["bound"] = bnd
                    outer.on_bound(bnd)

        solver = cp_model.CpSolver()
        solver.parameters.num_workers = int(self.threads)
        solver.parameters.random_seed = int(self.seed)
        left = deadline - time.perf_counter()
        status = None
        if left > 0.01:
            solver.parameters.max_time_in_seconds = left
            status = solver.solve(model, Callback())
            bnd = float(solver.best_objective_bound)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE, cp_model.UNKNOWN) and bnd > state["bound"] \
                    and time.perf_counter() < deadline and abs(bnd) < 1e18:
                state["bound"] = bnd
                self.on_bound(bnd)
        out_of_time = time.perf_counter() >= deadline - 0.5
        if status == cp_model.OPTIMAL and pub.best is not None and pub.ub <= float(solver.objective_value) + 0.5 * pub.unit:
            reason = "optimal"  # the retained incumbent (published before the deadline) attains the proved optimum
        elif status == cp_model.INFEASIBLE and pub.best is not None:
            raise RuntimeError("CP-SAT reports infeasible although a verified schedule exists")
        elif status == cp_model.INFEASIBLE and time.perf_counter() < deadline:
            reason = "infeasible"
        elif status is None or out_of_time or status in (cp_model.OPTIMAL, cp_model.INFEASIBLE):
            reason = "time_limit"
        else:
            reason = f"cpsat_status_{solver.status_name(status)}"
        return {"best_objective": pub.ub if pub.best is not None else None,
                "global_lb": state["bound"] if state["bound"] > -INF else None, "termination_reason": reason,
                "cpsat_status": solver.status_name(status) if status is not None else None,
                "budget_exhausted": time.perf_counter() - t0 >= self.time_limit - 1.0, "solution": pub.best,
                "solutions": state["solutions"], "published": pub.count, "best_stage": pub.stage, "branches": int(solver.num_branches),
                "conflicts": int(solver.num_conflicts), "total_seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------- iterated local search

class PSLocalSearch:
    """Iterated local search control; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 check_cap: float = 0.5, ejection: bool = True) -> None:
        self.time_limit, self.seed, self.check_cap, self.ejection = time_limit, seed, check_cap, ejection
        self.on_incumbent = on_incumbent or (lambda sol: None)
        self.on_bound = on_bound or (lambda bound: None)

    # state: assign (list), masks (list of int), sched (list of dict)

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def feasible(self, i: int, mask: int) -> dict[int, int] | None:
        if self.left() <= 0:
            return None
        st, starts = self.oracle.check(i, mask, min(self.check_cap, self.left()))
        return starts if st == FEASIBLE else None

    def solution(self) -> list[list[int]]:
        return [[self.assign[j], self.sched[self.assign[j]][j]] for j in range(self.inst.n)]

    def cost(self) -> int:
        return int(sum(int(self.inst.cost[i, j]) for j, i in enumerate(self.assign)))

    def move(self, j: int, v: int, sv: dict[int, int]) -> None:
        u = self.assign[j]
        self.masks[u] &= ~(1 << j)
        self.sched[u] = {k: s for k, s in self.sched[u].items() if k != j}
        self.masks[v] |= 1 << j
        self.sched[v] = sv
        self.assign[j] = v

    def construct(self) -> bool:
        """Initializer, or (if it fails) insertion with exact checks."""
        inst = self.inst
        g = greedy_solution(inst)
        self.assign = [0] * inst.n
        self.masks = [0] * inst.m
        self.sched = [dict() for _ in range(inst.m)]
        if g is not None:
            for j, (i, s) in enumerate(g[0]):
                self.assign[j] = i
                self.masks[i] |= 1 << j
                self.sched[i][j] = s
            return True
        order = sorted(range(inst.n), key=lambda j: (inst.deadline[j], -int(inst.energy[:, j].min())))
        for j in order:
            for i in sorted(np.nonzero(inst.eligible[:, j])[0], key=lambda i: (int(inst.cost[i, j]), i)):
                st = self.feasible(int(i), self.masks[i] | (1 << j))
                if st is not None:
                    self.assign[j] = int(i)
                    self.masks[i] |= 1 << j
                    self.sched[i] = st
                    break
            else:
                return False
        return True

    def relocate(self) -> bool:
        inst, el = self.inst, self.inst.eligible
        for j in self.rng.permutation(inst.n):
            j = int(j)
            u = self.assign[j]
            for v in sorted(np.nonzero(el[:, j])[0], key=lambda v: (int(inst.cost[v, j]), v)):
                v = int(v)
                if inst.cost[v, j] >= inst.cost[u, j]:
                    break
                sv = self.feasible(v, self.masks[v] | (1 << j))
                if sv is not None:
                    self.move(j, v, sv)
                    return True
        return False

    def swap(self) -> bool:
        inst, el, f = self.inst, self.inst.eligible, self.inst.cost
        cands = []
        for j in range(inst.n):
            u = self.assign[j]
            for k in range(j + 1, inst.n):
                v = self.assign[k]
                if u == v or not el[v, j] or not el[u, k]:
                    continue
                delta = int(f[v, j]) + int(f[u, k]) - int(f[u, j]) - int(f[v, k])
                if delta < 0:
                    cands.append((delta, j, k))
        cands.sort()
        for _, j, k in cands:
            if self.left() <= 0:
                return False
            u, v = self.assign[j], self.assign[k]
            su = self.feasible(u, (self.masks[u] & ~(1 << j)) | (1 << k))
            if su is None:
                continue
            sv = self.feasible(v, (self.masks[v] & ~(1 << k)) | (1 << j))
            if sv is None:
                continue
            self.masks[u] = (self.masks[u] & ~(1 << j)) | (1 << k)
            self.masks[v] = (self.masks[v] & ~(1 << k)) | (1 << j)
            self.sched[u], self.sched[v] = su, sv
            self.assign[j], self.assign[k] = v, u
            return True
        return False

    def eject(self) -> bool:
        inst, el, f = self.inst, self.inst.eligible, self.inst.cost
        for j in self.rng.permutation(inst.n):
            j = int(j)
            u = self.assign[j]
            for v in sorted(np.nonzero(el[:, j])[0], key=lambda v: (int(f[v, j]), v)):
                v = int(v)
                gain = int(f[u, j]) - int(f[v, j])
                if gain <= 0:
                    break
                for k in sorted(bits(self.masks[v]), key=lambda k: -int(inst.energy[v, k])):
                    ws = [int(w) for w in np.nonzero(el[:, k])[0] if w != v and int(f[w, k]) - int(f[v, k]) < gain]
                    if not ws:
                        continue
                    if self.left() <= 0:
                        return False
                    sv = self.feasible(v, (self.masks[v] & ~(1 << k)) | (1 << j))
                    if sv is None:
                        continue  # removing k does not make room for j on v
                    for w in sorted(ws, key=lambda w: (int(f[w, k]), w)):
                        if self.left() <= 0:
                            return False
                        wmask = ((self.masks[u] & ~(1 << j)) if w == u else self.masks[w]) | (1 << k)
                        sw = self.feasible(w, wmask)
                        if sw is None:
                            continue
                        # apply: j u -> v, k v -> w
                        self.masks[u] &= ~(1 << j)
                        self.sched[u] = {a: s for a, s in self.sched[u].items() if a != j}
                        self.masks[v] = (self.masks[v] & ~(1 << k)) | (1 << j)
                        self.sched[v] = sv
                        self.masks[w] = wmask
                        self.sched[w] = sw
                        self.assign[j], self.assign[k] = v, w
                        return True
        return False

    def after_move(self, stage: str) -> None:
        """Publish the current solution at once when it is a new global incumbent (so an improvement found before
        the deadline is never lost to a descent that crosses it)."""
        if self.cost() < self.pub.ub - 0.5 * self.pub.unit:
            self.pub.offer(self.solution(), stage)

    def descend(self) -> None:
        while self.left() > 0:
            if self.relocate() or self.swap() or (self.ejection and self.eject()):
                self.stats["moves"] += 1
                self.after_move("descent")
                continue
            break

    def kick(self) -> None:
        inst, el = self.inst, self.inst.eligible
        q = int(self.rng.integers(2, 6))
        for j in self.rng.choice(inst.n, size=min(q, inst.n), replace=False):
            j = int(j)
            opts = [int(v) for v in np.nonzero(el[:, j])[0] if v != self.assign[j]]
            self.rng.shuffle(opts)
            for v in opts:
                sv = self.feasible(v, self.masks[v] | (1 << j))
                if sv is not None:
                    self.move(j, v, sv)
                    self.after_move("kick")
                    break

    def snapshot(self):
        return list(self.assign), list(self.masks), [dict(s) for s in self.sched]

    def restore(self, snap) -> None:
        self.assign, self.masks, self.sched = list(snap[0]), list(snap[1]), [dict(s) for s in snap[2]]

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        self.inst = inst = load_instance(data_path)
        self.rng = np.random.default_rng(self.seed)
        self.oracle = FacilityOracle(inst, Relaxation(inst), seed=self.seed)
        self.stats = {"moves": 0, "kicks": 0, "improving_kicks": 0}
        self.pub = pub = _Publisher(inst, self.on_incumbent, self.deadline)
        reason = "time_limit"
        if not self.construct():
            reason = "no_solution"
        else:
            pub.offer(self.solution(), "initializer")
            self.descend()  # every new global incumbent is published inside the descent (after_move)
            best, best_cost = self.snapshot(), self.cost()
            while self.left() > 0:
                self.kick()
                self.stats["kicks"] += 1
                self.descend()
                c = self.cost()
                if c < best_cost:
                    best, best_cost = self.snapshot(), c
                    self.stats["improving_kicks"] += 1
                    pub.offer(self.solution(), "ils")  # normally already published by after_move
                elif c > best_cost or self.rng.random() < 0.5:
                    self.restore(best)
        return {"best_objective": pub.ub if pub.best is not None else None, "global_lb": None,
                "termination_reason": reason, "budget_exhausted": time.perf_counter() - self.t0 >= self.time_limit - 1.0,
                "solution": pub.best, "published": pub.count, "best_stage": pub.stage, **self.stats, "oracle": summarize(self.oracle.stats),
                "total_seconds": time.perf_counter() - self.t0}
