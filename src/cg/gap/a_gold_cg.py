from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.cg.gap.a_gen_data import GAPInstance
from src.utils.base_cg_class import (
    BaseColumnGeneration,
    IntegerMasterSolution,
    MasterLPSolution,
    PricingResult,
)


def _extract(inst: GAPInstance) -> tuple[list[list[int]], list[list[float]], list[int]]:
    machines = sorted(inst.machines, key=lambda m: m.id)
    jobs = sorted(inst.jobs, key=lambda j: j.id)
    b = [m.capacity for m in machines]
    n_jobs = len(jobs)
    n_machines = len(machines)
    a = [[0] * n_machines for _ in range(n_jobs)]
    c = [[0.0] * n_machines for _ in range(n_jobs)]
    for job in jobs:
        for assign in job.assignments:
            j = int(assign.machine_id)
            a[job.id][j] = assign.resource
            c[job.id][j] = assign.cost
    return a, c, b


def _pattern_key(machine: int, jobs_vec: list[int]) -> tuple[int, tuple[int, ...]]:
    return machine, tuple(jobs_vec)


class GAPColumnGeneration(BaseColumnGeneration[GAPInstance, dict[str, Any]]):
    """Column generation for GAP using machine-specific patterns."""

    def __init__(
        self,
        *,
        master_params: dict[str, Any] | None = None,
        pricing_params: dict[str, Any] | None = None,
        max_iters: int = 200,
        rc_tol: float = 1e-6,
        verbose: bool = False,
    ) -> None:
        super().__init__(max_iters=max_iters, verbose=verbose)
        self.master_params = master_params or {}
        self.pricing_params = pricing_params or {}
        self.rc_tolerance = rc_tol
        self._a: list[list[int]] | None = None
        self._c: list[list[float]] | None = None
        self._b: list[int] | None = None

    # Helpers -----------------------------------------------------------------
    def _init_matrices(self, inst: GAPInstance) -> None:
        self._a, self._c, self._b = _extract(inst)

    def _pattern_cost(self, pat: dict[str, Any]) -> float:
        _ = self._a or []
        c = self._c or []
        jobs_vec = pat["jobs"]
        m = pat["machine"]
        return sum(c[i][m] * jobs_vec[i] for i in range(len(jobs_vec)))

    def initial_patterns(self, inst: GAPInstance) -> list[dict[str, Any]]:
        self._init_matrices(inst)
        a = self._a or []
        b = self._b or []
        n_jobs = len(a)
        n_machines = len(b)
        patterns: list[dict[str, Any]] = []
        for j in range(n_machines):
            # empty pattern keeps feasibility of machine selection
            patterns.append({"machine": j, "jobs": [0] * n_jobs})
            # single-job patterns
            for i in range(n_jobs):
                if a[i][j] <= b[j]:
                    vec = [0] * n_jobs
                    vec[i] = 1
                    patterns.append({"machine": j, "jobs": vec})
        return patterns

    def load_instance(self, instance_name: str) -> GAPInstance:
        return GAPInstance.from_json_file(instance_name)

    def solve_restricted_master(
        self,
        inst: GAPInstance,
        patterns: list[dict[str, Any]],
    ) -> MasterLPSolution[dict[str, Any]]:
        a = self._a or []
        b = self._b or []
        c = self._c or []
        n_jobs = len(a)
        n_machines = len(b)
        P = range(len(patterns))
        jobs = range(n_jobs)
        machines = range(n_machines)

        model = gp.Model("gap_master_lp")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.master_params.items():
            model.setParam(k, v)

        lam = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name="lambda")

        job_constrs = []
        for i in jobs:
            cstr = model.addConstr(gp.quicksum(patterns[p]["jobs"][i] * lam[p] for p in P) == 1, name=f"job_{i}")
            job_constrs.append(cstr)

        mach_constrs = []
        for j in machines:
            cstr = model.addConstr(
                gp.quicksum(lam[p] for p in P if patterns[p]["machine"] == j) == 1,
                name=f"machine_{j}",
            )
            mach_constrs.append(cstr)

        def cost(p_idx: int) -> float:
            pat = patterns[p_idx]
            m = pat["machine"]
            return sum(c[i][m] * pat["jobs"][i] for i in jobs)

        model.setObjective(gp.quicksum(cost(p) * lam[p] for p in P), GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"GAP master LP not optimal (status={model.Status})")

        pi = [cstr.Pi for cstr in job_constrs]
        sigma = [cstr.Pi for cstr in mach_constrs]
        return MasterLPSolution(objective=float(model.ObjVal), duals={"pi": pi, "sigma": sigma})

    def solve_pricing(
        self,
        inst: GAPInstance,
        master_solution: MasterLPSolution[dict[str, Any]],
        patterns: list[dict[str, Any]],
    ) -> PricingResult[dict[str, Any]]:
        a = self._a or []
        b = self._b or []
        c = self._c or []
        pi = master_solution.duals["pi"]
        sigma = master_solution.duals["sigma"]
        n_jobs = len(a)
        n_machines = len(b)
        jobs = range(n_jobs)
        best_pat: dict[str, Any] | None = None
        best_rc = 0.0

        for j in range(n_machines):
            model = gp.Model(f"gap_pricing_{j}")
            model.Params.OutputFlag = 0
            for k, v in self.pricing_params.items():
                model.setParam(k, v)

            z = model.addVars(jobs, vtype=GRB.BINARY, name="z")
            model.addConstr(gp.quicksum(a[i][j] * z[i] for i in jobs) <= b[j], name="capacity")
            model.addConstr(gp.quicksum(z[i] for i in jobs) >= 1, name="nonempty")

            model.setObjective(gp.quicksum((pi[i] - c[i][j]) * z[i] for i in jobs) + sigma[j], GRB.MAXIMIZE)
            model.optimize()
            if model.Status != GRB.OPTIMAL:
                continue

            jobs_vec = [round(z[i].X) for i in jobs]
            pattern_cost = sum(c[i][j] * jobs_vec[i] for i in jobs)
            reduced_cost = pattern_cost - sum(pi[i] * jobs_vec[i] for i in jobs) - sigma[j]
            if sum(jobs_vec) == 0:
                reduced_cost = 0.0

            if reduced_cost < best_rc - 1e-12:
                best_rc = reduced_cost
                best_pat = {"machine": j, "jobs": jobs_vec}

        return PricingResult(pattern=best_pat, reduced_cost=best_rc)

    def solve_integer_master(
        self,
        inst: GAPInstance,
        patterns: list[dict[str, Any]],
    ) -> IntegerMasterSolution[dict[str, Any]]:
        a = self._a or []
        b = self._b or []
        c = self._c or []
        n_jobs = len(a)
        n_machines = len(b)
        jobs = range(n_jobs)
        machines = range(n_machines)
        P = range(len(patterns))

        model = gp.Model("gap_master_int")
        model.Params.OutputFlag = 1 if self.verbose else 0

        lam = model.addVars(P, vtype=GRB.BINARY, name="lambda")

        for i in jobs:
            model.addConstr(gp.quicksum(patterns[p]["jobs"][i] * lam[p] for p in P) == 1, name=f"job_{i}")
        for j in machines:
            model.addConstr(gp.quicksum(lam[p] for p in P if patterns[p]["machine"] == j) == 1, name=f"machine_{j}")

        def cost(p_idx: int) -> float:
            pat = patterns[p_idx]
            m = pat["machine"]
            return sum(c[i][m] * pat["jobs"][i] for i in jobs)

        model.setObjective(gp.quicksum(cost(p) * lam[p] for p in P), GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"GAP integer master not optimal (status={model.Status})")

        usage: dict[int, float] = {}
        assignments: list[dict[str, Any]] = []
        for p in P:
            val = lam[p].X
            if val > 0.5:
                usage[p] = float(val)
                pat = patterns[p]
                for i in jobs:
                    if pat["jobs"][i] > 0:
                        assignments.append({"job_id": i, "machine_id": pat["machine"]})

        return IntegerMasterSolution(
            objective=float(model.ObjVal),
            pattern_usages=usage,
            metadata={"patterns": patterns, "assignments": assignments},
        )
