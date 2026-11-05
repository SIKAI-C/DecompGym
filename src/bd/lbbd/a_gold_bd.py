"""Logic-based Benders reference: the master assigns tasks, each machine's schedule is the subproblem.

Master:  min sum c_ij x_ij + sum_i theta_i,  sum_i x_ij = 1,  theta_i >= 0,
         plus tardiness relaxations and the Benders cuts collected so far.
Subproblem for machine i and task set J: T*_i(J) = optimal weighted tardiness of 1|r_j|sum w_j T_j.
Cut:     theta_i >= T* - sum_{j in J} min(Delta_ij, T*) (1 - x_ij),
         Delta_ij = w_j * max(0, H_i(J) - d_j),  H_i(J) = max_{k in J} r_k + sum_{k in J} p_ik.
Validity: removing any subset S of J and appending S after an optimal schedule of J \\ S finishes every
appended task by H_i(J), so T*(J) <= T*(J \\ S) + sum_{j in S} Delta_ij; adding tasks never lowers the
optimal tardiness. Capping a coefficient at T* keeps the cut valid because theta_i >= 0.
"""
from __future__ import annotations

import math
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.bd.lbbd.a_gen_data import LBBDInstance
from src.utils.reference_runtime import ReferenceRuntime, relative_gap


def schedule_machine(inst: LBBDInstance, machine: int, tasks: list[int], runtime: ReferenceRuntime) -> tuple[float, dict[int, int]]:
    """Certified optimal weighted tardiness of one machine's task set (time-indexed MIP)."""
    if not tasks:
        return 0.0, {}
    p = {j: inst.proc_time[machine][j] for j in tasks}
    H = max(inst.release[j] for j in tasks) + sum(p.values())
    model = gp.Model("lbbd_machine")
    model.Params.OutputFlag = 0
    keys = [(j, t) for j in tasks for t in range(inst.release[j], H - p[j] + 1)]
    z = model.addVars(keys, vtype=GRB.BINARY)
    model.addConstrs(z.sum(j, "*") == 1 for j in tasks)
    running: dict[int, list] = {}
    for (j, t) in keys:
        for u in range(t, t + p[j]):
            running.setdefault(u, []).append(z[j, t])
    model.addConstrs(gp.quicksum(running[u]) <= 1 for u in running)
    model.setObjective(gp.quicksum(inst.weight[j] * max(0, t + p[j] - inst.due[j]) * z[j, t] for (j, t) in keys), GRB.MINIMIZE)
    model.Params.MIPGap = 0.0
    model.Params.MIPGapAbs = 1e-9
    runtime.optimize(model, role="subproblem", allow_incumbent=False)
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Machine subproblem not optimal (status={model.Status})")
    starts = {j: t for (j, t), var in z.items() if var.X > 0.5}
    return float(round(model.ObjVal)), starts


def _pair_tardiness(inst: LBBDInstance, machine: int, first: int, second: int) -> float:
    p = inst.proc_time[machine]
    c1 = inst.release[first] + p[first]
    c2 = max(inst.release[second], c1) + p[second]
    return inst.weight[first] * max(0, c1 - inst.due[first]) + inst.weight[second] * max(0, c2 - inst.due[second])


class LBBDBenders:
    """Outer-loop logic-based Benders with tardiness relaxations in the master."""

    def __init__(self, *, time_limit: float = 60.0, max_iters: int = 10_000, strong_cuts: bool = True, relaxation: bool = True, verbose: bool = False) -> None:
        self.runtime = ReferenceRuntime(time_limit)
        self.max_iters = max_iters
        self.strong_cuts = strong_cuts  # False: whole-set no-good optimality cuts only (shortcut S1)
        self.relaxation = relaxation
        self.verbose = verbose

    def load_instance(self, instance_name: str) -> LBBDInstance:
        return LBBDInstance.from_json_file(instance_name)

    def _relaxations(self, inst: LBBDInstance, model: gp.Model, x, theta) -> None:
        tasks = range(inst.n_tasks)
        for i in range(inst.n_machines):
            p = inst.proc_time[i]
            # A task alone cannot finish before r_j + p_ij.
            model.addConstr(theta[i] >= gp.quicksum(inst.weight[j] * max(0, inst.release[j] + p[j] - inst.due[j]) * x[i, j] for j in tasks))
            # Two tasks on the same machine: their joint optimal tardiness (best of the two orders, each
            # started as early as possible) is a lower bound for any superset, since adding tasks never helps.
            for j in tasks:
                for k in tasks:
                    if j < k:
                        pair = min(_pair_tardiness(inst, i, j, k), _pair_tardiness(inst, i, k, j))
                        if pair > 0:
                            model.addConstr(theta[i] >= pair * (x[i, j] + x[i, k] - 1))
            # Tasks released no earlier than t1 and due no later than t2: the last of them finishes at or
            # after t1 + their total processing time, and its weight is at least the smallest weight.
            for t1 in sorted(set(inst.release)):
                for t2 in sorted(set(inst.due)):
                    S = [j for j in tasks if inst.release[j] >= t1 and inst.due[j] <= t2]
                    if len(S) < 2 or t1 + sum(p[j] for j in S) <= t2:
                        continue
                    w = min(inst.weight[j] for j in S)
                    if w > 0:
                        model.addConstr(theta[i] >= w * (t1 - t2 + gp.quicksum(p[j] * x[i, j] for j in S)))

    def _strengthen(self, inst: LBBDInstance, machine: int, tasks: list[int], tstar: float) -> tuple[list[int], float]:
        """Greedily drop tasks while the optimal tardiness of the remaining set stays at T*.

        Any assignment that keeps the smaller set J' on this machine has tardiness at least T*(J') = T*,
        and the Delta coefficients of J' use the smaller horizon H_i(J'), so the cut is stronger.
        """
        keep = list(tasks)
        # Try the tasks least likely to matter first: those with the latest due dates.
        for j in sorted(tasks, key=lambda k: -inst.due[k]):
            trial = [k for k in keep if k != j]
            if trial and self._schedule(inst, machine, trial)[0] >= tstar - 1e-9:
                keep = trial
        return keep, tstar

    def _schedule(self, inst: LBBDInstance, machine: int, tasks: list[int]) -> tuple[float, dict[int, int]]:
        key = (machine, frozenset(tasks))
        if key not in self._sched_cache:
            self._sched_cache[key] = schedule_machine(inst, machine, list(tasks), self.runtime)
        return self._sched_cache[key]

    def _cut(self, inst: LBBDInstance, machine: int, tasks: list[int], tstar: float, x, theta):
        """Strengthened optimality cut for one machine, as a linear expression pair (lhs, rhs)."""
        J, tstar = self._strengthen(inst, machine, tasks, tstar)
        H = max(inst.release[j] for j in J) + sum(inst.proc_time[machine][j] for j in J)
        coef = {j: min(tstar, inst.weight[j] * max(0, H - inst.due[j])) for j in J}
        return theta[machine], tstar - gp.quicksum(coef[j] * (1 - x[machine, j]) for j in J)

    def run(self, instance_name: str, *, started_at: float | None = None) -> dict[str, Any]:
        """Reference: one master branch-and-bound; each new master incumbent is checked by the machine
        subproblems, and violated optimality cuts are added as lazy constraints (branch-and-check)."""
        self.runtime.start(started_at)
        self._sched_cache = {}
        inst = self.load_instance(instance_name)
        M, N = range(inst.n_machines), range(inst.n_tasks)
        master = gp.Model("lbbd_branch_and_check")
        master.Params.OutputFlag = 0
        master.Params.LazyConstraints = 1
        master.Params.MIPGap = 0.0
        x = master.addVars(M, N, vtype=GRB.BINARY, name="x")
        theta = master.addVars(M, lb=0.0, name="theta")
        master.addConstrs(x.sum("*", j) == 1 for j in N)
        self._relaxations(inst, master, x, theta)
        master.setObjective(gp.quicksum(inst.proc_cost[i][j] * x[i, j] for i in M for j in N) + theta.sum(), GRB.MINIMIZE)
        state: dict[str, Any] = {"ub": math.inf, "best": None, "checks": 0, "cuts": 0, "error": None}
        started = time.perf_counter()

        def callback(model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                xv = model.cbGetSolution(x)
                tv = model.cbGetSolution(theta)
                sets = {i: [j for j in N if xv[i, j] > 0.5] for i in M}
                solved = {i: self._schedule(inst, i, sets[i]) for i in M}
                state["checks"] += 1
                value = sum(inst.proc_cost[i][j] for i in M for j in sets[i]) + sum(v for v, _ in solved.values())
                if value < state["ub"]:
                    state["ub"] = value
                    state["best"] = [{"task_id": j, "machine_id": i, "start": solved[i][1][j]} for i in M for j in sets[i]]
                for i in M:
                    if solved[i][0] > tv[i] + 1e-6:
                        lhs, rhs = self._cut(inst, i, sets[i], solved[i][0], x, theta)
                        model.cbLazy(lhs >= rhs)
                        state["cuts"] += 1
            except Exception as exc:  # budget exhausted inside a subproblem
                state["error"] = f"{type(exc).__name__}: {exc}"
                model.terminate()

        master.Params.TimeLimit = max(1.0, self.runtime.remaining())
        master.Params.Threads = 1
        master.optimize(callback)
        lb = float(master.ObjBound)
        ub = state["ub"]
        termination = "optimal" if master.Status == GRB.OPTIMAL and state["error"] is None else "time_limit"
        if state["best"] is None:
            raise RuntimeError("LBBD stopped without a complete schedule")
        best = sorted(state["best"], key=lambda a: a["task_id"])
        return {
            "best_objective": ub, "global_lb": min(lb, ub), "global_ub": ub, "gap": relative_gap(min(lb, ub), ub),
            "termination_reason": termination, "iters": state["checks"], "total_time": time.perf_counter() - started,
            "iter_records": [], "assignments": best, "n_cuts": state["cuts"],
        }

    def run_outer_loop(self, instance_name: str, *, started_at: float | None = None) -> dict[str, Any]:
        """Outer-loop LBBD (re-solves the master each iteration); used for the screened shortcuts."""
        self.runtime.start(started_at)
        self._sched_cache = {}
        inst = self.load_instance(instance_name)
        M, N = range(inst.n_machines), range(inst.n_tasks)
        master = gp.Model("lbbd_master")
        master.Params.OutputFlag = 0
        x = master.addVars(M, N, vtype=GRB.BINARY, name="x")
        theta = master.addVars(M, lb=0.0, name="theta")
        master.addConstrs(x.sum("*", j) == 1 for j in N)
        if self.relaxation:
            self._relaxations(inst, master, x, theta)
        master.setObjective(gp.quicksum(inst.proc_cost[i][j] * x[i, j] for i in M for j in N) + theta.sum(), GRB.MINIMIZE)
        master.Params.MIPGap = 0.0
        lb, ub, best, records, cuts = -math.inf, math.inf, None, [], 0
        termination = "max_iters"
        started = time.perf_counter()
        for it in range(1, self.max_iters + 1):
            try:
                self.runtime.optimize(master, role="master", allow_incumbent=False)
            except Exception:
                termination = "time_limit"
                break
            lb = max(lb, float(master.ObjBound))
            sets = {i: [j for j in N if x[i, j].X > 0.5] for i in M}
            try:
                solved = {i: schedule_machine(inst, i, sets[i], self.runtime) for i in M}
            except Exception:
                termination = "time_limit"
                break
            value = sum(inst.proc_cost[i][j] for i in M for j in sets[i]) + sum(v for v, _ in solved.values())
            if value < ub:
                ub = value
                best = [{"task_id": j, "machine_id": i, "start": solved[i][1][j]} for i in M for j in sets[i]]
            records.append({"iteration": it, "elapsed_time": time.perf_counter() - started, "global_lb": lb, "global_ub": ub, "cuts": cuts})
            if self.verbose:
                print(f"[lbbd {it}] lb={lb:.1f} ub={ub:.1f} cuts={cuts}")
            if ub - lb <= 1e-6 * max(1.0, abs(ub)):
                termination = "optimal"
                break
            stop = False
            for i in M:
                tstar = solved[i][0]
                if tstar <= theta[i].X + 1e-6:
                    continue
                J = sets[i]
                if self.strong_cuts:
                    try:
                        J, tstar = self._strengthen(inst, i, J, tstar)
                    except Exception:
                        # Budget exhausted while strengthening: keep the incumbent already stored in `best`.
                        stop = True
                        break
                    H = max(inst.release[j] for j in J) + sum(inst.proc_time[i][j] for j in J)
                    coef = {j: min(tstar, inst.weight[j] * max(0, H - inst.due[j])) for j in J}
                    master.addConstr(theta[i] >= tstar - gp.quicksum(coef[j] * (1 - x[i, j]) for j in J))
                else:
                    # Whole-set no-good: tardiness T* is only known for exactly this set (or a superset).
                    master.addConstr(theta[i] >= tstar * (1 - gp.quicksum(1 - x[i, j] for j in J)))
                cuts += 1
            if stop:
                termination = "time_limit"
                break
        if best is None:
            raise RuntimeError("LBBD stopped without a complete schedule")
        best.sort(key=lambda a: a["task_id"])
        return {
            "best_objective": ub,
            "global_lb": lb if math.isfinite(lb) else None,
            "global_ub": ub,
            "gap": relative_gap(lb, ub),
            "termination_reason": termination,
            "iters": len(records),
            "total_time": time.perf_counter() - started,
            "iter_records": records,
            "assignments": best,
            "n_cuts": cuts,
        }
