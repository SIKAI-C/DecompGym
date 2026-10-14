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
        time_limit: float = 60.0,
    ) -> None:
        super().__init__(max_iters=max_iters, verbose=verbose, time_limit=time_limit)
        self.master_params = master_params or {}
        self.pricing_params = pricing_params or {}
        self.rc_tolerance = rc_tol
        self._a: list[list[int]] | None = None
        self._c: list[list[float]] | None = None
        self._b: list[int] | None = None
        self._allowed: set[tuple[int, int]] = set()
        self._phase_one = False
        self.phase_one_records: list[dict[str, Any]] = []

    # Helpers -----------------------------------------------------------------
    def _init_matrices(self, inst: GAPInstance) -> None:
        self._a, self._c, self._b = _extract(inst)
        self._allowed = {(job.id, assignment.machine_id) for job in inst.jobs for assignment in job.assignments}

    def _pattern_cost(self, pat: dict[str, Any]) -> float:
        _ = self._a or []
        c = self._c or []
        jobs_vec = pat["jobs"]
        m = pat["machine"]
        return sum(c[i][m] * jobs_vec[i] for i in range(len(jobs_vec)))

    def initial_patterns(self, inst: GAPInstance) -> list[dict[str, Any]]:
        """Find feasible RMP columns through phase-I dual pricing, without an assignment MIP."""
        self._init_matrices(inst)
        a, b = self._a, self._b
        patterns = []
        for machine in range(len(b)):
            patterns.append({"machine": machine, "jobs": [0] * len(a)})
            for job in range(len(a)):
                if (job, machine) in self._allowed and a[job][machine] <= b[machine]:
                    vector = [0] * len(a)
                    vector[job] = 1
                    patterns.append({"machine": machine, "jobs": vector})
        self.phase_one_records = []
        self._phase_one = True
        try:
            for iteration in range(1, self.max_iters + 1):
                self.runtime.check()
                master = self.solve_restricted_master(inst, patterns)
                self.phase_one_records.append({"iteration": iteration, "artificial_objective": master.objective, "n_patterns": len(patterns)})
                if master.objective <= 1e-7:
                    return patterns
                pricing = self.solve_pricing(inst, master, patterns)
                if pricing.reduced_cost >= -self.rc_tolerance or not self.add_pattern(patterns, pricing):
                    raise RuntimeError("GAP phase-I pricing did not eliminate artificial variables")
            raise RuntimeError("GAP phase-I reached its iteration limit without a feasible restricted master")
        finally:
            self._phase_one = False

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

        model = gp.Model("gap_phase1_restricted_master" if self._phase_one else "gap_master_lp")
        model.Params.OutputFlag = 1 if self.verbose else 0
        for k, v in self.master_params.items():
            model.setParam(k, v)

        lam = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name="lambda")

        artificial = model.addVars(jobs, lb=0.0, name="artificial") if self._phase_one else None
        job_constrs = []
        for i in jobs:
            cstr = model.addConstr(gp.quicksum(patterns[p]["jobs"][i] * lam[p] for p in P) + (artificial[i] if self._phase_one else 0) == 1, name=f"job_{i}")
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

        model.setObjective(gp.quicksum(artificial[i] for i in jobs) if self._phase_one else gp.quicksum(cost(p) * lam[p] for p in P), GRB.MINIMIZE)
        self.runtime.optimize(model, role="restricted_master", allow_incumbent=False)
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
        a, b, c = self._a or [], self._b or [], self._c or []
        pi, sigma = master_solution.duals["pi"], master_solution.duals["sigma"]
        jobs = range(len(a))
        best_pat, best_value = None, 0.0
        bounds = []
        generated = []
        for machine in range(len(b)):
            model = gp.Model(f"gap_machine_knapsack_pricing_{machine}")
            model.Params.OutputFlag = 1 if self.verbose else 0
            for key, value in self.pricing_params.items():
                model.setParam(key, value)
            model.Params.PoolSearchMode = 2
            model.Params.PoolSolutions = 64
            z = model.addVars(jobs, vtype=GRB.BINARY, ub={i: int((i, machine) in self._allowed) for i in jobs}, name="z")
            model.addConstr(gp.quicksum(a[i][machine] * z[i] for i in jobs) <= b[machine])
            model.setObjective(gp.quicksum(((0.0 if self._phase_one else c[i][machine]) - pi[i]) * z[i] for i in jobs) - sigma[machine], GRB.MINIMIZE)
            self.runtime.optimize(model, role="pricing")
            if model.Status != GRB.OPTIMAL:
                raise RuntimeError(f"GAP pricing problem not optimal (machine={machine}, status={model.Status})")
            bounds.append(float(model.ObjBound))
            if model.ObjVal < best_value:
                best_value = float(model.ObjVal)
                best_pat = {"machine": machine, "jobs": [round(z[i].X) for i in jobs]}
            for index in range(model.SolCount):
                model.Params.SolutionNumber = index
                generated.append({"machine": machine, "jobs": [round(z[i].Xn) for i in jobs]})
        reduced_cost = min([0.0, *bounds])
        known = {_pattern_key(p["machine"], p["jobs"]) for p in patterns}
        for candidate in generated:
            if candidate == best_pat and reduced_cost < -self.rc_tolerance:
                continue
            key = _pattern_key(candidate["machine"], candidate["jobs"])
            if key not in known:
                patterns.append(candidate)
                known.add(key)
        # Convexity is one per machine; correct the dual certificate for pricing tolerances.
        metadata = {"pricing_bounds": bounds, "phase_one": self._phase_one}
        if not self._phase_one:
            metadata["lp_bound"] = master_solution.objective + sum(min(0.0, bound) for bound in bounds)
        return PricingResult(pattern=best_pat, reduced_cost=reduced_cost, metadata=metadata)

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
        for k, v in self.master_params.items():
            model.setParam(k, v)

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
        self.runtime.optimize(model, role="integer_master", allow_incumbent=True)
        if model.SolCount == 0:
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
