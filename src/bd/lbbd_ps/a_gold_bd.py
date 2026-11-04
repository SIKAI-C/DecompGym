"""Logic-based Benders reference for planning and scheduling with a cumulative resource (branch-and-check).

Master (one Gurobi search tree): min sum f_ij x_ij, sum_i x_ij = 1, x binary on the eligible pairs, plus the interval
relaxation of every facility's scheduling problem (Hooker 2007; Cire, Coban and Hooker 2016): for every window [a, b]
(a a release date, b a deadline) and J(a, b) = {j : r_j >= a, d_j <= b, j fits on i},
    energy           sum_{j in J(a,b)} p_ij c_ij x_ij <= C_i (b - a)
    k-concurrency    sum_{j in J(a,b), c_ij > C_i / (k + 1)} p_ij x_ij <= k (b - a)      (k = 1, 2, 3).
Subproblem: at every integer master solution each facility's job set is checked exactly (`FacilityOracle`): cached
answers (feasibility is monotone under subsets), the relaxation rows, serial schedule generation, then a time-indexed
Gurobi feasibility model, whose OPTIMAL / INFEASIBLE status is a proof. For an infeasible facility the deletion filter
(Hooker 2005) shrinks the set to an irreducible infeasible core K, and up to `max_cores` disjoint cores are cut with
    sum_{j in K} x_ij <= |K| - 1,
valid because no schedule on i holds all of K. A master solution whose facilities are all feasible is a verified plan.
The master optimum with every cut proves optimality (integer costs: MIPGap = 0, MIPGapAbs = 0.5 and the rounded bound).
`run_outer_loop` is the textbook shortcut (S1, the scale-up naive LBBD): the master re-solved each iteration, one
whole-set nogood per infeasible facility, optionally without the relaxation.
"""
from __future__ import annotations

import math
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.bd.lbbd_ps.a_gen_data import PSInstance
from src.utils.reference_runtime import ReferenceRuntime, relative_gap


FEASIBLE, INFEASIBLE = "feasible", "infeasible"


def bits(mask: int) -> list[int]:
    return [j for j in range(mask.bit_length()) if mask >> j & 1]


def load_ok(inst: PSInstance, i: int, starts: dict[int, int]) -> bool:
    """Windows and the cumulative resource of facility i for integer starts (completions before starts at equal times)."""
    events = []
    for j, s in starts.items():
        if s < inst.release[j] or s + inst.proc[i][j] > inst.deadline[j]:
            return False
        events += [(s, inst.rate[i][j]), (s + inst.proc[i][j], -inst.rate[i][j])]
    load = 0
    for _, delta in sorted(events):
        load += delta
        if load > inst.capacity[i]:
            return False
    return True


def relaxation_rows(inst: PSInstance, i: int, concurrency=(1, 2, 3)) -> list[tuple[dict[int, int], int]]:
    """Energy and k-concurrency rows of facility i that some assignment could violate: ({job: coef}, rhs)."""
    rows: dict[frozenset, tuple[dict[int, int], int]] = {}
    C = inst.capacity[i]
    for a in sorted(set(inst.release)):
        for b in sorted(set(inst.deadline)):
            if b <= a:
                continue
            J = [j for j in range(inst.n_jobs) if inst.release[j] >= a and inst.deadline[j] <= b and inst.eligible(i, j)]
            cands = [({j: inst.proc[i][j] * inst.rate[i][j] for j in J}, C * (b - a), "energy")]
            for k in concurrency:
                cands.append(({j: inst.proc[i][j] for j in J if inst.rate[i][j] * (k + 1) > C}, k * (b - a), f"conc{k}"))
            for coef, rhs, kind in cands:
                if sum(coef.values()) <= rhs:
                    continue
                key = frozenset((kind, j) for j in coef)
                if key not in rows or rhs < rows[key][1]:
                    rows[key] = (coef, rhs)
    return list(rows.values())


class FacilityOracle:
    """Exact cumulative feasibility of a job set on one facility (see the module docstring)."""

    def __init__(self, inst: PSInstance, runtime: ReferenceRuntime) -> None:
        self.inst, self.runtime = inst, runtime
        self.rows = [relaxation_rows(inst, i) for i in range(inst.n_facilities)]
        self.feasible: list[dict[int, dict[int, int]]] = [{} for _ in range(inst.n_facilities)]
        self.infeasible: list[list[int]] = [[] for _ in range(inst.n_facilities)]
        self.exact_solves = 0

    def _sgs(self, i: int, jobs: list[int]) -> dict[int, int] | None:
        """Serial schedule generation: each job at its earliest feasible start, under a few priority orders."""
        inst = self.inst
        orders = [sorted(jobs, key=key) for key in (lambda j: (inst.deadline[j] - inst.proc[i][j], j), lambda j: (-inst.proc[i][j] * inst.rate[i][j], j),
                                                     lambda j: (-inst.rate[i][j], j), lambda j: (-inst.proc[i][j], j))]
        for order in orders:
            profile = [0] * (inst.horizon + 1)
            starts = {}
            for j in order:
                p, c = inst.proc[i][j], inst.rate[i][j]
                s = next((t for t in range(inst.release[j], inst.deadline[j] - p + 1)
                          if all(profile[u] + c <= inst.capacity[i] for u in range(t, t + p))), None)
                if s is None:
                    break
                for u in range(s, s + p):
                    profile[u] += c
                starts[j] = s
            else:
                return starts
        return None

    def _exact(self, i: int, jobs: list[int]) -> dict[int, int] | None:
        inst = self.inst
        self.exact_solves += 1
        model = gp.Model("lbbd_ps_facility")
        model.Params.OutputFlag = 0
        keys = [(j, t) for j in jobs for t in range(inst.release[j], inst.deadline[j] - inst.proc[i][j] + 1)]
        z = model.addVars(keys, vtype=GRB.BINARY)
        model.addConstrs(z.sum(j, "*") == 1 for j in jobs)
        running: dict[int, list] = {}
        for (j, t) in keys:
            for u in range(t, t + inst.proc[i][j]):
                running.setdefault(u, []).append(inst.rate[i][j] * z[j, t])
        model.addConstrs(gp.quicksum(running[u]) <= inst.capacity[i] for u in running)
        self.runtime.optimize(model, role="subproblem", allow_incumbent=False)
        if model.Status == GRB.INFEASIBLE:
            return None
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"facility subproblem neither feasible nor infeasible (status={model.Status})")
        starts = {j: t for (j, t), var in z.items() if var.X > 0.5}
        if not load_ok(inst, i, starts):
            raise RuntimeError("facility subproblem returned a schedule that fails the exact check")
        return starts

    def check(self, i: int, mask: int) -> tuple[str, dict[int, int] | None]:
        inst = self.inst
        jobs = bits(mask)
        if not jobs:
            return FEASIBLE, {}
        if any(not inst.eligible(i, j) for j in jobs):
            return INFEASIBLE, None
        for known, starts in self.feasible[i].items():
            if mask & ~known == 0:
                return FEASIBLE, {j: starts[j] for j in jobs}
        if any(bad & ~mask == 0 for bad in self.infeasible[i]):
            return INFEASIBLE, None
        if any(sum(coef.get(j, 0) for j in jobs) > rhs for coef, rhs in self.rows[i]):
            starts = None
        else:
            starts = self._sgs(i, jobs) or self._exact(i, jobs)
        if starts is None:
            self.infeasible[i].append(mask)
            return INFEASIBLE, None
        self.feasible[i][mask] = starts
        return FEASIBLE, starts

    def core(self, i: int, mask: int) -> int:
        """Deletion filter: an irreducible infeasible subset of the infeasible set `mask` (smallest energy removed first)."""
        inst = self.inst
        for j in sorted(bits(mask), key=lambda j: (inst.proc[i][j] * inst.rate[i][j], j)):
            if self.check(i, mask & ~(1 << j))[0] == INFEASIBLE:
                mask &= ~(1 << j)
        return mask


def greedy_solution(inst: PSInstance) -> dict[int, tuple[int, int]] | None:
    """Jobs by decreasing energy, each at its earliest feasible start on the cheapest facility with room."""
    profile = [[0] * (inst.horizon + 1) for _ in range(inst.n_facilities)]
    plan = {}
    for j in sorted(range(inst.n_jobs), key=lambda j: (-max(inst.proc[i][j] * inst.rate[i][j] for i in range(inst.n_facilities)), j)):
        for i in sorted(range(inst.n_facilities), key=lambda i: (inst.cost[i][j], i)):
            if not inst.eligible(i, j):
                continue
            p, c = inst.proc[i][j], inst.rate[i][j]
            s = next((t for t in range(inst.release[j], inst.deadline[j] - p + 1)
                      if all(profile[i][u] + c <= inst.capacity[i] for u in range(t, t + p))), None)
            if s is not None:
                for u in range(s, s + p):
                    profile[i][u] += c
                plan[j] = (i, s)
                break
        else:
            return None
    return plan


class PSBenders:
    def __init__(self, *, time_limit: float = 60.0, max_cores: int = 3, max_iters: int = 100_000, verbose: bool = False) -> None:
        self.runtime = ReferenceRuntime(time_limit)
        self.max_cores = max_cores
        self.max_iters = max_iters
        self.verbose = verbose

    def load_instance(self, instance_name: str) -> PSInstance:
        return PSInstance.from_json_file(instance_name)

    def _master(self, inst: PSInstance, relaxation: bool) -> tuple[gp.Model, dict]:
        master = gp.Model("lbbd_ps_master")
        master.Params.OutputFlag = 0
        master.Params.MIPGap, master.Params.MIPGapAbs = 0.0, 0.5  # integer costs: exact once the bound is within 0.5
        pairs = [(i, j) for i in range(inst.n_facilities) for j in range(inst.n_jobs) if inst.eligible(i, j)]
        x = master.addVars(pairs, vtype=GRB.BINARY, name="x")
        master.addConstrs(x.sum("*", j) == 1 for j in range(inst.n_jobs))
        if relaxation:
            for i in range(inst.n_facilities):
                for coef, rhs in self.oracle.rows[i]:
                    master.addConstr(gp.quicksum(a * x[i, j] for j, a in coef.items()) <= rhs)
        master.setObjective(gp.quicksum(inst.cost[i][j] * x[i, j] for (i, j) in pairs), GRB.MINIMIZE)
        return master, x

    @staticmethod
    def _masks(inst: PSInstance, values: dict) -> list[int]:
        masks = [0] * inst.n_facilities
        for (i, j), v in values.items():
            if v > 0.5:
                masks[i] |= 1 << j
        return masks

    def _cores(self, i: int, mask: int, strengthen: bool) -> list[int]:
        if not strengthen:
            return [mask]
        out, rest = [], mask
        for _ in range(self.max_cores):
            k = self.oracle.core(i, rest)
            out.append(k)
            rest &= ~k
            if not rest or self.oracle.check(i, rest)[0] != INFEASIBLE:
                break
        return out

    def _plan(self, inst: PSInstance, masks: list[int]) -> tuple[list[dict[str, int]], float] | None:
        """The verified schedule of an assignment whose facilities are all feasible, else None."""
        schedule = []
        for i, mask in enumerate(masks):
            status, starts = self.oracle.check(i, mask)
            if status != FEASIBLE:
                return None
            schedule += [{"job": j, "facility": i, "start": s} for j, s in starts.items()]
        schedule.sort(key=lambda e: e["job"])
        return schedule, float(sum(inst.cost[e["facility"]][e["job"]] for e in schedule))

    def _start(self, instance_name: str, started_at: float | None) -> PSInstance:
        self.runtime.start(started_at)
        inst = self.load_instance(instance_name)
        self.oracle = FacilityOracle(inst, self.runtime)
        self.ub, self.best = math.inf, None
        g = greedy_solution(inst)
        if g is not None:
            self.best = sorted(({"job": j, "facility": i, "start": s} for j, (i, s) in g.items()), key=lambda e: e["job"])
            self.ub = float(sum(inst.cost[i][j] for j, (i, _) in g.items()))
        return inst

    def run(self, instance_name: str, *, started_at: float | None = None) -> dict[str, Any]:
        """Branch-and-check: lazy core nogoods at every integer master solution."""
        started = time.perf_counter()
        inst = self._start(instance_name, started_at)
        master, x = self._master(inst, relaxation=True)
        master.Params.LazyConstraints = 1
        if self.best is not None:
            for (i, j), var in x.items():
                var.Start = 1.0 if self.best[j]["facility"] == i else 0.0
        state: dict[str, Any] = {"checks": 0, "cuts": 0, "error": None}

        def callback(model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                state["checks"] += 1
                masks = self._masks(inst, model.cbGetSolution(x))
                bad = [i for i, mask in enumerate(masks) if self.oracle.check(i, mask)[0] == INFEASIBLE]
                if not bad:
                    schedule, value = self._plan(inst, masks)
                    if value < self.ub - 0.5:
                        self.ub, self.best = value, schedule
                    return
                for i in bad:
                    for k in self._cores(i, masks[i], strengthen=True):
                        model.cbLazy(gp.quicksum(x[i, j] for j in bits(k)) <= len(bits(k)) - 1)
                        state["cuts"] += 1
            except Exception as exc:  # e.g. budget exhausted inside a subproblem: the solution stays unverified, nothing is certified
                state["error"] = f"{type(exc).__name__}: {exc}"
                model.terminate()

        bound = None
        remaining = self.runtime.remaining()
        if remaining > 0:  # never launch the master after the shared budget is spent
            master.Params.TimeLimit = remaining
            master.Params.Threads = 1
            master.optimize(callback)
            try:  # the bound is available without a master incumbent; an interrupted solve may have none at all
                raw = float(master.ObjBound)
            except (gp.GurobiError, AttributeError):
                raw = math.nan
            if master.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED) and math.isfinite(raw) and abs(raw) < GRB.INFINITY:
                bound = math.ceil(raw - 1e-6)
        proved = (master.Status == GRB.OPTIMAL and state["error"] is None and self.best is not None and bound is not None
                  and bound >= self.ub - 1e-9 and self.runtime.remaining() > 0)
        if self.best is None:
            raise RuntimeError("LBBD stopped without a verified schedule")
        lb = self.ub if proved else (min(bound, self.ub) if bound is not None else None)
        budget = state["error"] is None or "budget" in state["error"].lower() or "time limit" in state["error"].lower()
        return {
            "best_objective": self.ub, "global_lb": lb, "global_ub": self.ub, "gap": relative_gap(lb, self.ub),
            "termination_reason": "optimal" if proved else ("time_limit" if budget else f"error: {state['error']}"),
            "iters": state["checks"], "n_cuts": state["cuts"],
            "exact_subproblem_solves": self.oracle.exact_solves, "error": state["error"],
            "total_time": time.perf_counter() - started, "schedule": self.best,
        }

    def run_outer_loop(self, instance_name: str, *, started_at: float | None = None, strengthen: bool = False, relaxation: bool = False) -> dict[str, Any]:
        """Outer-loop LBBD: re-solve the master each iteration (the shortcut screens; see the module docstring)."""
        started = time.perf_counter()
        inst = self._start(instance_name, started_at)
        master, x = self._master(inst, relaxation)
        lb, termination, cuts, it = None, "max_iters", 0, 0
        try:
            for it in range(1, self.max_iters + 1):
                self.runtime.optimize(master, role="master", allow_incumbent=False)
                if master.Status != GRB.OPTIMAL:
                    raise RuntimeError(f"master not optimal (status={master.Status})")
                lb = math.ceil(master.ObjBound - 1e-6)
                masks = self._masks(inst, {k: v.X for k, v in x.items()})
                bad = [i for i, mask in enumerate(masks) if self.oracle.check(i, mask)[0] == INFEASIBLE]
                if self.verbose:
                    print(f"[lbbd_ps {it}] lb={lb} ub={self.ub} infeasible facilities={bad}")
                if not bad:
                    schedule, value = self._plan(inst, masks)
                    if value < self.ub - 0.5:
                        self.ub, self.best = value, schedule
                    termination = "optimal"
                    break
                for i in bad:
                    for k in self._cores(i, masks[i], strengthen):
                        master.addConstr(gp.quicksum(x[i, j] for j in bits(k)) <= len(bits(k)) - 1)
                        cuts += 1
        except Exception as exc:
            if "budget" not in str(exc).lower() and "time limit" not in str(exc).lower():
                raise
            termination = "time_limit"
        if self.best is None:
            raise RuntimeError("LBBD stopped without a verified schedule")
        lb = min(lb, self.ub) if lb is not None else None
        return {
            "best_objective": self.ub, "global_lb": self.ub if termination == "optimal" else lb, "global_ub": self.ub,
            "gap": relative_gap(lb, self.ub), "termination_reason": termination, "iters": it, "n_cuts": cuts,
            "total_time": time.perf_counter() - started, "schedule": self.best,
        }
