"""Facility scheduling for the planning-and-scheduling family: the LBBD subproblem oracle, its relaxation, cores
(minimal infeasible job sets) and the shared basic initializer (scale-up study).

Subproblem. For facility i and a job set S, decide whether the jobs of S can be scheduled on i (integer starts in
[r_j, d_j - p_ij], cumulative resource C_i). The exact oracle (`FacilityOracle.check`) answers in this order, every
step exact:
  1. caches (feasibility is monotone: a subset of a feasible set is feasible with the restricted schedule, a superset
     of an infeasible set is infeasible), per facility, as bit masks;
  2. the interval relaxation (`Relaxation`): a violated row proves infeasibility;
  3. serial schedule generation (`sgs`, six priority rules): a schedule it finds proves feasibility;
  4. sets of at most 8 jobs: exact enumeration of active schedules (`small_exact`, at most 20000 nodes);
  5. CP-SAT (OR-Tools, one worker, timetable edge finding and the overload checker switched on): interval
     variables with start domains [r_j, d_j - p_ij] and one cumulative constraint; FEASIBLE/OPTIMAL gives a schedule (rechecked exactly), INFEASIBLE a proof; UNKNOWN (time limit) is
     reported as such and never used as either.
Naive and engineered LBBD share this oracle; they differ in what they do with it (cuts, relaxation, search).

Relaxation (Hooker 2007, Section 4; Cire, Coban and Hooker 2016): for facility i and every window [a, b] with a a
release date, b a deadline, a < b, let J(a, b) = {j : r_j >= a, d_j <= b, j fits on i}; every feasible schedule
satisfies
    energy:          sum_{j in J(a,b)} p_ij c_ij x_ij <= C_i (b - a)
    k-concurrency:   sum_{j in J(a,b), c_ij > C_i / (k + 1)} p_ij x_ij <= k (b - a)      (k = 1, 2, 3)
(at most k jobs with c_ij > C_i / (k + 1) are in process at any time). Rows that cannot be violated are dropped, and
rows with the same job set keep the tightest right-hand side. The same rows are the engineered master's subproblem
relaxation and a fast infeasibility test in the oracle.

Cores (`FacilityOracle.core`): from an infeasible set (or the jobs of a violated relaxation row), the deletion filter
of Hooker (2005, 2007) removes jobs one at a time (smallest energy first) and keeps a removal when the rest is still
infeasible; when every test resolves, the result is an irreducible infeasible set, and the nogood
sum_{j in core} x_ij <= |core| - 1 is a valid Benders cut. A test that ends UNKNOWN keeps the job (the set stays
infeasible, so the cut stays valid).

Shared basic initializer (`greedy_solution`): jobs in a fixed priority order (three orders tried, the cheapest
complete result kept) are inserted one by one at their earliest feasible start on the cheapest facility with room.
It may fail (no solution) on tight instances.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.bd.lbbd.a_data import PSInstance, machine_load_ok

INFEASIBLE, FEASIBLE, UNKNOWN = "infeasible", "feasible", "unknown"


def bits(mask: int) -> list[int]:
    out = []
    j = 0
    while mask:
        if mask & 1:
            out.append(j)
        mask >>= 1
        j += 1
    return out


def to_mask(jobs) -> int:
    m = 0
    for j in jobs:
        m |= 1 << int(j)
    return m


# ---------------------------------------------------------------- relaxation

class Relaxation:
    """Interval relaxation rows per facility (see the module docstring): rows[i] = (A (k, n) float, rhs (k,), kinds,
    windows)."""

    def __init__(self, inst: PSInstance, concurrency: tuple[int, ...] = (1, 2, 3)) -> None:
        self.inst = inst
        el = inst.eligible
        rel = sorted({int(v) for v in inst.release})
        dl = sorted({int(v) for v in inst.deadline})
        self.rows: list[tuple[np.ndarray, np.ndarray, list[str], list[tuple[int, int]]]] = []
        for i in range(inst.m):
            best: dict[tuple[str, bytes], tuple[float, np.ndarray, tuple[int, int]]] = {}
            C = int(inst.capacity[i])
            for a in rel:
                inside_a = inst.release >= a
                for b in dl:
                    if b <= a:
                        continue
                    J = inside_a & (inst.deadline <= b) & el[i]
                    if not J.any():
                        continue
                    cands = [("energy", np.where(J, inst.proc[i] * inst.rate[i], 0).astype(float), float(C * (b - a)))]
                    for k in concurrency:
                        Jk = J & (inst.rate[i] * (k + 1) > C)
                        if Jk.any():
                            cands.append((f"conc{k}", np.where(Jk, inst.proc[i], 0).astype(float), float(k * (b - a))))
                    for kind, coef, rhs in cands:
                        if coef.sum() <= rhs:
                            continue  # cannot be violated
                        key = (kind, (coef > 0).tobytes())
                        if key not in best or rhs < best[key][0]:
                            best[key] = (rhs, coef, (a, b))
            kinds = [k for k, _ in best]
            if best:
                A = np.array([v[1] for v in best.values()])
                rhs = np.array([v[0] for v in best.values()])
            else:
                A, rhs = np.zeros((0, inst.n)), np.zeros(0)
            self.rows.append((A, rhs, kinds, [v[2] for v in best.values()]))

    def violated(self, i: int, x: np.ndarray) -> int | None:
        """Index of a row of facility i violated by the 0/1 vector x (n,), or None."""
        A, rhs, _, _ = self.rows[i]
        if not len(rhs):
            return None
        lhs = A @ x
        bad = np.nonzero(lhs > rhs + 1e-9)[0]
        return int(bad[np.argmax(lhs[bad] - rhs[bad])]) if len(bad) else None

    def size(self) -> int:
        return sum(len(r[1]) for r in self.rows)


# ---------------------------------------------------------------- serial schedule generation

def earliest_start(profile: np.ndarray, cap: int, lo: int, hi: int, p: int, c: int) -> int | None:
    """Earliest t in [lo, hi] with profile[t:t + p] + c <= cap (profile covers [0, H))."""
    if hi < lo:
        return None
    if c == 0:
        return lo
    bad = (profile + c > cap).astype(np.int64)
    cum = np.concatenate(([0], np.cumsum(bad)))
    t = np.arange(lo, hi + 1)
    ok = cum[t + p] - cum[t] == 0
    k = np.flatnonzero(ok)
    return int(t[k[0]]) if len(k) else None


def sgs(inst: PSInstance, i: int, jobs: list[int], order: list[int] | None = None) -> dict[int, int] | None:
    """Serial schedule generation on facility i: each job (in `order`) at its earliest feasible start."""
    profile = np.zeros(inst.horizon + 1, dtype=np.int64)
    cap = int(inst.capacity[i])
    starts: dict[int, int] = {}
    for j in (order if order is not None else jobs):
        p, c = int(inst.proc[i, j]), int(inst.rate[i, j])
        t = earliest_start(profile, cap, int(inst.release[j]), int(inst.deadline[j]) - p, p, c)
        if t is None:
            return None
        profile[t:t + p] += c
        starts[j] = t
    return starts


def sgs_orders(inst: PSInstance, i: int, jobs: list[int]) -> list[list[int]]:
    p, c, r, d = inst.proc[i], inst.rate[i], inst.release, inst.deadline
    keys = [
        lambda j: (d[j], -p[j] * c[j], j),                 # earliest deadline first
        lambda j: (d[j] - p[j], -p[j] * c[j], j),          # latest start time first
        lambda j: (-p[j] * c[j], d[j], j),                 # largest energy first
        lambda j: (d[j] - r[j] - p[j], d[j], j),           # least slack first
        lambda j: (-c[j], -p[j], j),                       # largest rate first
        lambda j: (r[j], d[j], -p[j], j),                  # earliest release first
    ]
    out, seen = [], set()
    for key in keys:
        o = sorted(jobs, key=key)
        t = tuple(o)
        if t not in seen:
            seen.add(t)
            out.append(o)
    return out


def sgs_any(inst: PSInstance, i: int, jobs: list[int]) -> dict[int, int] | None:
    for order in sgs_orders(inst, i, jobs):
        starts = sgs(inst, i, jobs, order)
        if starts is not None:
            return starts
    return None


# ---------------------------------------------------------------- exact enumeration for small sets

def small_exact(inst: PSInstance, i: int, jobs: list[int], node_limit: int = 20000) -> tuple[str, dict[int, int] | None]:
    """Exact feasibility of a small job set on facility i by depth-first search over serial-SGS job orders.
    Active schedules (every job at its earliest feasible start given the jobs placed before it, over all orders)
    contain a feasible schedule whenever one exists (deadlines = maximum lateness <= 0, a regular objective). A node
    where some remaining job has no feasible start is dead (loads only grow below it); nodes are memoized on (load
    profile, remaining jobs). UNKNOWN when `node_limit` nodes are exceeded."""
    cap = int(inst.capacity[i])
    H = inst.horizon
    P = {j: int(inst.proc[i, j]) for j in jobs}
    Cr = {j: int(inst.rate[i, j]) for j in jobs}
    R = {j: int(inst.release[j]) for j in jobs}
    D = {j: int(inst.deadline[j]) for j in jobs}
    seen: set = set()
    nodes = [0]
    starts: dict[int, int] = {}

    def earliest(prof: list[int], j: int) -> int | None:
        p, c = P[j], Cr[j]
        s, last = R[j], D[j] - p
        while s <= last:
            for t in range(s, s + p):
                if prof[t] + c > cap:
                    s = t + 1
                    break
            else:
                return s
        return None

    def rec(prof: list[int], rest: frozenset) -> bool | None:
        if not rest:
            return True
        nodes[0] += 1
        if nodes[0] > node_limit:
            return None
        key = (tuple(prof), rest)
        if key in seen:
            return False
        seen.add(key)
        place = {}
        for j in rest:
            s = earliest(prof, j)
            if s is None:
                return False
            place[j] = s
        for j in sorted(rest, key=lambda j: (place[j], D[j], j)):
            s = place[j]
            new = list(prof)
            for t in range(s, s + P[j]):
                new[t] += Cr[j]
            starts[j] = s
            res = rec(new, rest - {j})
            if res is None:
                return None
            if res:
                return True
            del starts[j]
        return False

    res = rec([0] * (H + 1), frozenset(jobs))
    if res is None:
        return UNKNOWN, None
    if not res:
        return INFEASIBLE, None
    out = {j: starts[j] for j in jobs}
    if not machine_load_ok(inst, i, list(out), list(out.values())):
        raise RuntimeError("small_exact schedule fails the exact check")
    return FEASIBLE, out


# ---------------------------------------------------------------- CP-SAT

CUMULATIVE_PARAMS = ("use_timetable_edge_finding_in_cumulative", "use_overload_checker_in_cumulative",
                     "use_hard_precedences_in_cumulative")


def cpsat_check(inst: PSInstance, i: int, jobs: list[int], time_limit: float, seed: int = 0,
                hint: dict[int, int] | None = None) -> tuple[str, dict[int, int] | None]:
    """Exact feasibility of jobs on facility i by CP-SAT (one worker)."""
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    starts, intervals, demands = {}, [], []
    for j in jobs:
        p, lo, hi = int(inst.proc[i, j]), int(inst.release[j]), int(inst.deadline[j]) - int(inst.proc[i, j])
        if hi < lo or inst.rate[i, j] > inst.capacity[i]:
            return INFEASIBLE, None
        s = model.new_int_var(lo, hi, f"s{j}")
        starts[j] = s
        intervals.append(model.new_fixed_size_interval_var(s, p, f"iv{j}"))
        demands.append(int(inst.rate[i, j]))
        if hint and j in hint:
            model.add_hint(s, int(hint[j]))
    model.add_cumulative(intervals, demands, int(inst.capacity[i]))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.max_time_in_seconds = max(0.001, float(time_limit))
    solver.parameters.random_seed = int(seed)
    for name in CUMULATIVE_PARAMS:  # stronger cumulative propagation (infeasibility proofs of tight packings)
        try:
            setattr(solver.parameters, name, True)
        except (AttributeError, ValueError):
            pass
    status = solver.solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out = {j: int(solver.value(s)) for j, s in starts.items()}
        if not machine_load_ok(inst, i, list(out), list(out.values())):
            raise RuntimeError("CP-SAT schedule fails the exact check")
        return FEASIBLE, out
    if status == cp_model.INFEASIBLE:
        return INFEASIBLE, None
    return UNKNOWN, None


# ---------------------------------------------------------------- oracle

class FacilityOracle:
    """Exact facility-feasibility oracle with monotone caches (see the module docstring)."""

    def __init__(self, inst: PSInstance, relax: Relaxation | None = None, seed: int = 0, superset_scan: int = 4000,
                 small_size: int = 8, small_nodes: int = 20000) -> None:
        self.inst = inst
        self.relax = relax if relax is not None else Relaxation(inst)
        self.seed = seed
        self.superset_scan = superset_scan
        self.small_size, self.small_nodes = small_size, small_nodes
        self.feas: list[dict[int, dict[int, int]]] = [dict() for _ in range(inst.m)]
        self.feas_list: list[list[int]] = [[] for _ in range(inst.m)]
        self.infeas: list[list[int]] = [[] for _ in range(inst.m)]
        self.infeas_set: list[set[int]] = [set() for _ in range(inst.m)]
        self.stats = {"checks": 0, "cache": 0, "relaxation": 0, "sgs": 0, "small_feasible": 0, "small_infeasible": 0,
                      "small_seconds": 0.0, "cp_calls": 0, "cp_feasible": 0,
                      "cp_infeasible": 0, "cp_unknown": 0, "cp_seconds": 0.0, "core_calls": 0, "core_seconds": 0.0}

    def _store_feasible(self, i: int, mask: int, starts: dict[int, int]) -> None:
        if mask not in self.feas[i]:
            self.feas[i][mask] = starts
            self.feas_list[i].append(mask)

    def _store_infeasible(self, i: int, mask: int) -> None:
        if mask not in self.infeas_set[i]:
            self.infeas_set[i].add(mask)
            self.infeas[i].append(mask)

    def lookup(self, i: int, mask: int) -> tuple[str, dict[int, int] | None]:
        if mask == 0:
            return FEASIBLE, {}
        if mask in self.feas[i]:
            return FEASIBLE, self.feas[i][mask]
        if mask in self.infeas_set[i]:
            return INFEASIBLE, None
        for bad in self.infeas[i]:
            if bad & ~mask == 0:
                return INFEASIBLE, None
        for good in self.feas_list[i][-self.superset_scan:]:
            if mask & ~good == 0:
                st = self.feas[i][good]
                return FEASIBLE, {j: st[j] for j in bits(mask)}
        return UNKNOWN, None

    def check(self, i: int, mask: int, time_limit: float, exact: bool = True) -> tuple[str, dict[int, int] | None]:
        """FEASIBLE with a schedule {job: start}, INFEASIBLE, or UNKNOWN (out of time, or exact=False and the cheap
        tests did not decide)."""
        self.stats["checks"] += 1
        status, starts = self.lookup(i, mask)
        if status != UNKNOWN:
            self.stats["cache"] += 1
            return status, starts
        jobs = bits(mask)
        x = np.zeros(self.inst.n)
        x[jobs] = 1.0
        if not self.inst.eligible[i, jobs].all() or self.relax.violated(i, x) is not None:
            self.stats["relaxation"] += 1
            self._store_infeasible(i, mask)
            return INFEASIBLE, None
        starts = sgs_any(self.inst, i, jobs)
        if starts is not None:
            self.stats["sgs"] += 1
            self._store_feasible(i, mask, starts)
            return FEASIBLE, starts
        if not exact or time_limit <= 0:
            return UNKNOWN, None
        if len(jobs) <= self.small_size:
            t = time.perf_counter()
            status, starts = small_exact(self.inst, i, jobs, self.small_nodes)
            self.stats["small_seconds"] += time.perf_counter() - t
            if status != UNKNOWN:
                self.stats[f"small_{status}"] += 1
                if status == FEASIBLE:
                    self._store_feasible(i, mask, starts)
                else:
                    self._store_infeasible(i, mask)
                return status, starts
        t = time.perf_counter()
        status, starts = cpsat_check(self.inst, i, jobs, time_limit, self.seed)
        self.stats["cp_calls"] += 1
        self.stats["cp_seconds"] += time.perf_counter() - t
        self.stats[f"cp_{status}"] += 1
        if status == FEASIBLE:
            self._store_feasible(i, mask, starts)
        elif status == INFEASIBLE:
            self._store_infeasible(i, mask)
        return status, starts

    def core(self, i: int, mask: int, deadline: float, cap: float = 5.0) -> int:
        """A (when every test resolves, irreducible) infeasible subset of the infeasible set `mask` on facility i."""
        t0 = time.perf_counter()
        self.stats["core_calls"] += 1
        inst = self.inst
        jobs = bits(mask)
        cur = mask
        x = np.zeros(inst.n)
        x[jobs] = 1.0
        bad_el = [j for j in jobs if not inst.eligible[i, j]]
        if bad_el:
            cur = 1 << bad_el[0]
        else:
            row = self.relax.violated(i, x)
            if row is not None:
                A = self.relax.rows[i][0]
                cur = to_mask(j for j in jobs if A[row, j] > 0)
        energy = inst.proc[i] * inst.rate[i]
        for j in sorted(bits(cur), key=lambda j: (energy[j], j)):
            if cur == 1 << j:
                break
            left = deadline - time.perf_counter()
            if left <= 0:
                break
            status, _ = self.check(i, cur & ~(1 << j), min(cap, left))
            if status == INFEASIBLE:
                cur &= ~(1 << j)
        self._store_infeasible(i, cur)
        self.stats["core_seconds"] += time.perf_counter() - t0
        return cur


# ---------------------------------------------------------------- shared basic initializer

def greedy_solution(inst: PSInstance) -> tuple[list[list[int]], int] | None:
    """Shared basic initializer (see the module docstring): [[facility, start] per job] and its cost, or None."""
    el = inst.eligible
    big = np.where(el, inst.energy, np.iinfo(np.int64).max)
    fmin = np.where(el, inst.cost, np.iinfo(np.int64).max)
    second = np.sort(np.where(el, inst.cost, 10 ** 9), axis=0)
    regret = (second[1] if inst.m > 1 else second[0]) - second[0]
    orders = [
        sorted(range(inst.n), key=lambda j: (inst.deadline[j], -int(big[:, j].min()), j)),
        sorted(range(inst.n), key=lambda j: (-int(big[:, j].min()), j)),
        sorted(range(inst.n), key=lambda j: (-int(regret[j]), -int(big[:, j].min()), j)),
    ]
    best = None
    for order in orders:
        profiles = [np.zeros(inst.horizon + 1, dtype=np.int64) for _ in range(inst.m)]
        sol: list[list[int] | None] = [None] * inst.n
        ok = True
        for j in order:
            placed = False
            for i in sorted(np.nonzero(el[:, j])[0], key=lambda i: (fmin[i, j], i)):
                p, c = int(inst.proc[i, j]), int(inst.rate[i, j])
                t = earliest_start(profiles[i], int(inst.capacity[i]), int(inst.release[j]), int(inst.deadline[j]) - p, p, c)
                if t is not None:
                    profiles[i][t:t + p] += c
                    sol[j] = [int(i), int(t)]
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            cost = int(sum(int(inst.cost[i, j]) for j, (i, _) in enumerate(sol)))
            if best is None or cost < best[1]:
                best = ([list(v) for v in sol], cost)
    return best


def solution_from_schedules(inst: PSInstance, assign: list[int], schedules: list[dict[int, int]]) -> list[list[int]]:
    """[[facility, start] per job] from an assignment and per-facility schedules."""
    return [[int(assign[j]), int(schedules[assign[j]][j])] for j in range(inst.n)]


def summarize(stats: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()}
