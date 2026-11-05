"""Engineered logic-based Benders reference for planning and scheduling (minimum cost): branch-and-check with a
subproblem relaxation, strengthened cuts and master-guided repair (scale-up study).

Master (Gurobi, one search tree; Thorsteinsson's branch-and-check): min sum f_ij x_ij, sum_i x_ij = 1, x binary on
the eligible pairs, plus the interval relaxation of every facility's scheduling problem (a_scheduling.Relaxation:
energy and k-concurrency rows over every release/deadline window; Hooker 2007, Cire, Coban and Hooker 2016 found the
relaxation the most effective single device). The shared basic initializer is the MIP start.
At every integer master solution (MIPSOL; heuristic and injected solutions included) each facility's job set is
checked by the shared exact oracle (cached). For an infeasible facility the lazy cuts are nogoods on cores:
the deletion filter (a_scheduling.FacilityOracle.core) shrinks the set to an irreducible infeasible subset, and up
to `max_cores` further cores are taken from the rest of the set (each found core is removed and the remainder
re-checked), so one infeasible assignment yields several strong cuts
    sum_{j in K} x_ij <= |K| - 1.
A solution whose facilities are all feasible is accepted by the tree and published with its schedules. When some
facilities are infeasible, a repair (`repair`) keeps the feasible facilities, removes from each infeasible facility
the core job that is cheapest to move (smallest cost increase) until the rest is feasible, and reinserts the removed
jobs at their cheapest facility with room (exact oracle checks with a short cap); a complete repaired solution is
published and injected into the tree at the next node (cbSetSolution). Bounds: Gurobi's master bound, valid because
every cut is implied by a proof of infeasibility. Knobs for ablations: relaxation, strengthen (cores vs full-set
nogoods), max_cores, repair, warm_start.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.lbbd.a_data import PSInstance, load_instance, schedule_cost
from src.bd.lbbd.a_scheduling import (FEASIBLE, INFEASIBLE, UNKNOWN, FacilityOracle, Relaxation, bits,
                                      greedy_solution, solution_from_schedules, summarize)
from src.bd.lbbd.b_mip_xl import integral_costs, is_memory_error, make_env, set_gap
from src.bd.lbbd.c_naive_bd import INF, AssignmentMaster, certify, masks_of


class PSBendersXL:
    """Engineered branch-and-check reference; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 relaxation: bool = True, strengthen: bool = True, max_cores: int = 3, core_cap: float = 2.0,
                 repair: bool = True, repair_cap: float = 1.0, warm_start: bool = True, verbose: bool = False) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.relaxation, self.strengthen, self.max_cores, self.core_cap = relaxation, strengthen, max_cores, core_cap
        self.do_repair, self.repair_cap, self.warm_start = repair, repair_cap, warm_start
        self.on_incumbent = on_incumbent or (lambda sol, stage=None: None)
        self.on_bound = on_bound or (lambda bound: None)
        self.verbose = verbose
        self.trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------ bookkeeping

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def log(self, stage: str) -> None:
        self.trace.append({"t": round(time.perf_counter() - self.t0, 3), "incumbent": self.ub if self.ub < INF else None,
                           "bound": self.lb if self.lb > -INF else None, "stage": stage})
        if self.verbose:
            print(self.trace[-1], flush=True)

    def publish_bound(self, value: float, stage: str) -> None:
        value = certify(value)
        if value > self.lb and self.left() > 0:
            self.lb = value
            self.on_bound(value)
            self.log(stage)

    def offer(self, sol: list[list[int]], stage: str) -> bool:
        value = schedule_cost(self.inst, sol)
        if value is None:
            raise RuntimeError(f"engineered LBBD built an infeasible schedule ({stage})")
        if value < self.ub - 0.5 * self.unit and self.left() > 0:
            self.ub, self.best = value, [list(v) for v in sol]
            self.on_incumbent(self.best, stage=stage)
            self.log(stage)
            return True
        return False

    def closed(self) -> bool:
        return self.ub < INF and self.lb > self.ub - self.unit

    # ------------------------------------------------------------ run

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        self.inst = inst = load_instance(data_path)
        self.unit = 1.0 if integral_costs(inst) else 1e-6
        self.ub, self.lb, self.best = INF, -INF, None
        self.stats: dict[str, Any] = {"mipsol_calls": 0, "infeasible_facilities": 0, "lazy_cuts": 0, "core_sizes": 0,
                                      "set_sizes": 0, "repairs": 0, "repair_success": 0, "injected": 0, "node_count": 0.0,
                                      "bc_status": None, "relaxation_rows": 0}
        env = make_env(self.threads, self.soft_mem_gb, self.seed)
        self.oracle = FacilityOracle(inst, Relaxation(inst), seed=self.seed)
        self.master = None
        try:
            reason = self.search(env)
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        finally:
            if self.master is not None:
                self.master.dispose()
            env.dispose()
        ub = self.ub if self.ub < INF else None
        lb = self.lb if self.lb > -INF else None
        st = dict(self.stats)
        if st["lazy_cuts"]:
            st["mean_core_size"] = round(st["core_sizes"] / st["lazy_cuts"], 2)
        return {"best_objective": ub, "global_lb": lb, "global_ub": ub, "termination_reason": reason,
                "budget_exhausted": time.perf_counter() - self.t0 >= self.time_limit - 1.0, "solution": self.best,
                "stats": st, **st, "oracle": summarize(self.oracle.stats), "total_seconds": time.perf_counter() - self.t0,
                "trace": self.trace}

    # ------------------------------------------------------------ cuts and repair

    def cores(self, i: int, mask: int) -> list[int]:
        """Infeasible subsets of the infeasible set `mask` on facility i to cut with (see the module docstring)."""
        if not self.strengthen:
            return [mask]
        out = []
        rest = mask
        for _ in range(max(1, self.max_cores)):
            k = self.oracle.core(i, rest, self.deadline, self.core_cap)
            out.append(k)
            rest &= ~k
            if not rest or self.left() <= 0:
                break
            status, _ = self.oracle.check(i, rest, min(self.core_cap, self.left()))
            if status != INFEASIBLE:
                break
        return out

    def repair(self, assign: list[int], status: list[str], schedules: list[dict[int, int] | None],
               cores: dict[int, list[int]]) -> list[list[int]] | None:
        inst, el = self.inst, self.inst.eligible
        cost = inst.cost
        masks = masks_of(inst, assign)
        pool: list[int] = []

        def extra(i: int, j: int) -> tuple[int, int]:
            """Cost increase of moving job j off facility i to its cheapest other facility (larger energy first)."""
            others = [int(cost[k, j]) for k in range(inst.m) if k != i and el[k, j]]
            return (min(others) - int(cost[i, j]) if others else 10 ** 9, -int(inst.energy[i, j]))

        for i in range(inst.m):
            if status[i] == FEASIBLE:
                continue
            cur = masks[i]
            core = cores[i][0]
            while True:
                if self.left() <= 0:
                    return None
                cand = bits(core & cur) or bits(cur)
                j = min(cand, key=lambda j: extra(i, j))
                cur &= ~(1 << j)
                pool.append(j)
                st, starts = self.oracle.check(i, cur, min(self.repair_cap, self.left()))
                if st == FEASIBLE:
                    schedules[i] = starts
                    break
                if st == UNKNOWN:
                    return None
                core = self.oracle.core(i, cur, self.deadline, min(self.core_cap, self.repair_cap))
            masks[i] = cur
        for j in sorted(pool, key=lambda j: -int(inst.energy[:, j].max())):
            placed = False
            for k in sorted(np.nonzero(el[:, j])[0], key=lambda k: (int(cost[k, j]), k)):
                k = int(k)
                if self.left() <= 0:
                    return None
                st, starts = self.oracle.check(k, masks[k] | (1 << j), min(self.repair_cap, self.left()))
                if st == FEASIBLE:
                    masks[k] |= 1 << j
                    schedules[k] = starts
                    assign[j] = k
                    placed = True
                    break
            if not placed:
                return None
        return solution_from_schedules(inst, assign, schedules)

    # ------------------------------------------------------------ branch-and-check

    def search(self, env: gp.Env) -> str:
        inst = self.inst
        g = greedy_solution(inst)
        if g is not None:
            self.offer(g[0], "initializer")
            per: list[dict[int, int]] = [{} for _ in range(inst.m)]
            for j, (i, s) in enumerate(g[0]):
                per[i][j] = s
            for i in range(inst.m):
                if per[i]:
                    self.oracle._store_feasible(i, sum(1 << j for j in per[i]), per[i])
        self.master = master = AssignmentMaster(inst, env, self.oracle.relax if self.relaxation else None)
        self.stats["relaxation_rows"] = master.relax_rows
        md = master.model
        md.Params.LazyConstraints = 1
        md.Params.Seed = self.seed
        set_gap(md, inst)
        if self.warm_start and self.best is not None:
            master.x.Start = master.start_vector([i for i, _ in self.best])
        pending: list[list[int] | None] = [None]
        errors: list[BaseException] = []

        def callback(cb: gp.Model, where: int) -> None:
            try:
                body(cb, where)
            except BaseException as exc:  # noqa: BLE001 - gurobipy would swallow it and accept an unchecked solution
                errors.append(exc)
                cb.terminate()

        def body(cb: gp.Model, where: int) -> None:
            if self.left() <= 0:
                cb.terminate()
                return
            if where == GRB.Callback.MIPSOL:
                self.stats["mipsol_calls"] += 1
                assign = master.assignment(cb.cbGetSolution(master.xl))
                status: list[str] = []
                schedules: list[dict[int, int] | None] = []
                for i, mask in enumerate(masks_of(inst, assign)):
                    st, starts = self.oracle.check(i, mask, self.left())
                    if st == UNKNOWN:
                        cb.terminate()  # out of time: the solution stays unverified and is not published
                        return
                    status.append(st)
                    schedules.append(starts)
                bad = [i for i, st in enumerate(status) if st == INFEASIBLE]
                if not bad:
                    self.offer(solution_from_schedules(inst, assign, schedules), "branch_and_check")
                    return
                found: dict[int, list[int]] = {}
                for i in bad:
                    self.stats["infeasible_facilities"] += 1
                    mask = masks_of(inst, assign)[i]
                    self.stats["set_sizes"] += len(bits(mask))
                    found[i] = self.cores(i, mask)
                    for k in found[i]:
                        coef, xs, rhs = master.nogood(i, k)
                        cb.cbLazy(gp.LinExpr(coef, xs) <= rhs)
                        self.stats["lazy_cuts"] += 1
                        self.stats["core_sizes"] += len(xs)
                if self.do_repair and self.left() > 0:
                    self.stats["repairs"] += 1
                    sol = self.repair(list(assign), status, schedules, found)
                    if sol is not None:
                        self.stats["repair_success"] += 1
                        if self.offer(sol, "repair"):
                            pending[0] = [i for i, _ in sol]
            elif where == GRB.Callback.MIPNODE:
                if pending[0] is not None and cb.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                    cb.cbSetSolution(master.xl, master.start_vector(pending[0]).tolist())
                    cb.cbUseSolution()
                    self.stats["injected"] += 1
                    pending[0] = None
            elif where == GRB.Callback.MIP:
                bnd = cb.cbGet(GRB.Callback.MIP_OBJBND)
                if bnd < GRB.INFINITY:
                    self.publish_bound(float(bnd), "branch_and_check")

        if self.left() <= 0.01:
            return "time_limit"
        md.Params.TimeLimit = max(0.01, self.left())
        md.optimize(callback)
        if errors:
            raise errors[0]
        self.stats["node_count"] = float(md.NodeCount)
        self.stats["bc_status"] = int(md.Status)
        if md.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.MEM_LIMIT) and md.SolCount:
            self.publish_bound(float(md.ObjBound), "final")
        if md.Status == GRB.OPTIMAL:
            # Gurobi's incumbent passed every facility check, so it is optimal; it certifies optimality only if the
            # retained incumbent (published strictly before the deadline) attains its value
            if self.ub < INF and self.ub <= float(md.ObjVal) + 0.5 * self.unit:
                return "optimal"
            return "time_limit" if self.left() <= 0.5 else "stuck"
        if md.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            if self.ub < INF:
                raise RuntimeError("master infeasible although a verified schedule exists: invalid cut")
            return "infeasible" if self.left() > 0 else "time_limit"
        if md.Status == GRB.MEM_LIMIT:
            return "memory_limit"
        if md.Status == GRB.TIME_LIMIT or (md.Status == GRB.INTERRUPTED and self.left() <= 0.5):
            return "time_limit"
        return "stuck" if md.Status == GRB.INTERRUPTED else f"status_{md.Status}"
