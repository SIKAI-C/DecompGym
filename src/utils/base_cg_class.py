from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import time
from typing import Any, Generic, TypeVar

from src.utils.reference_runtime import ReferenceLimit, ReferenceRuntime


TInst = TypeVar("TInst")
TPattern = TypeVar("TPattern")


@dataclass
class CGIterationRecord:
    """Iteration log for column generation."""

    iteration: int
    elapsed_time: float
    iter_time: float
    master_obj: float
    reduced_cost: float
    n_patterns: int


@dataclass
class MasterLPSolution(Generic[TPattern]):
    """Solution of the restricted master LP."""

    objective: float
    duals: Any
    metadata: dict[str, Any] | None = None


@dataclass
class PricingResult(Generic[TPattern]):
    """Result of the pricing (subproblem)."""

    pattern: TPattern | None
    reduced_cost: float
    metadata: dict[str, Any] | None = None


@dataclass
class IntegerMasterSolution(Generic[TPattern]):
    """Final integer master solution after CG terminates."""

    objective: float
    pattern_usages: dict[int, float]
    metadata: dict[str, Any] | None = None


class BaseColumnGeneration(ABC, Generic[TInst, TPattern]):
    """Lightweight column generation driver.

    Subclasses implement:
      * how to load/parse an instance,
      * how to build the initial pattern set,
      * how to solve the restricted master LP and pricing,
      * how to solve the integer master once no improving pattern exists.
    """

    rc_tolerance: float = 1e-6  # stop when reduced cost >= -tol (minimization)
    is_minimization: bool = True

    def __init__(self, max_iters: int = 200, verbose: bool = False, *, time_limit: float = 60.0) -> None:
        self.max_iters = max_iters
        self.verbose = verbose
        self.runtime = ReferenceRuntime(time_limit)
        self.iter_records: list[CGIterationRecord] = []
        self._run_start_time: float | None = None
        self._run_end_time: float | None = None

    # --- required hooks ----------------------------------------------------
    @abstractmethod
    def load_instance(self, instance_name: str) -> TInst:
        """Load/parse the problem instance referenced by name or path."""

    @abstractmethod
    def initial_patterns(self, inst: TInst) -> list[TPattern]:
        """Return a non-empty list of patterns to seed the master LP."""

    @abstractmethod
    def solve_restricted_master(
        self,
        inst: TInst,
        patterns: list[TPattern],
    ) -> MasterLPSolution[TPattern]:
        """Solve the current restricted master LP and return objective/duals."""

    @abstractmethod
    def solve_pricing(
        self,
        inst: TInst,
        master_solution: MasterLPSolution[TPattern],
        patterns: list[TPattern],
    ) -> PricingResult[TPattern]:
        """Complete pricing across all subproblems; return the best reduced cost.

        Raise if pricing is incomplete. Convergence uses rc_tolerance and
        requires every pricing subproblem to have been checked.
        """

    @abstractmethod
    def solve_integer_master(
        self,
        inst: TInst,
        patterns: list[TPattern],
    ) -> IntegerMasterSolution[TPattern]:
        """Solve the final integer master over the generated pattern set."""

    # --- optional hooks ----------------------------------------------------
    def add_pattern(
        self,
        patterns: list[TPattern],
        pricing_result: PricingResult[TPattern],
    ) -> bool:
        """Insert pricing_result.pattern into the pattern list if useful.

        Returns True if a new pattern was added.
        """
        pat = pricing_result.pattern
        if pat is None:
            return False
        if pat in patterns:
            return False
        patterns.append(pat)
        return True

    def format_iteration_log(
        self,
        record: CGIterationRecord,
        master_solution: MasterLPSolution[TPattern],
        pricing_result: PricingResult[TPattern],
    ) -> str:
        """Pretty string for verbose logging (override for custom content)."""
        return (
            f"[iter {record.iteration:02d}] master_obj={record.master_obj:.6f} "
            f"rc={record.reduced_cost:.3e} patterns={record.n_patterns}"
        )

    def should_terminate(
        self,
        iteration: int,
        pricing_result: PricingResult[TPattern],
    ) -> bool:
        """Default stop rule: non-improving reduced cost or max_iters reached."""
        if iteration >= self.max_iters:
            return True
        return pricing_result.reduced_cost >= -self.rc_tolerance

    def finalize_result(
        self,
        inst: TInst,
        master_solution: MasterLPSolution[TPattern],
        pricing_result: PricingResult[TPattern],
        integer_solution: IntegerMasterSolution[TPattern],
        iterations: int,
    ) -> dict[str, Any]:
        """Assemble a result dictionary similar to other solvers."""
        if self._run_start_time is None or self._run_end_time is None:
            total_time = math.nan
        else:
            total_time = self._run_end_time - self._run_start_time

        # Report the LP bound only when complete pricing meets the convergence tolerance.
        lp_converged = (
            math.isfinite(pricing_result.reduced_cost)
            and pricing_result.reduced_cost >= -self.rc_tolerance
        )
        lp_bound = (pricing_result.metadata or {}).get("lp_bound", master_solution.objective) if lp_converged else None
        global_lb = lp_bound if self.is_minimization else integer_solution.objective
        global_ub = integer_solution.objective if self.is_minimization else lp_bound

        return {
            "best_objective": integer_solution.objective,
            "global_lb": global_lb,
            "global_ub": global_ub,
            "lp_converged": lp_converged,
            "iters": iterations,
            "total_time": total_time,
            "iter_records": self.iter_records,
            "final_master_objective": integer_solution.objective,
            "final_vars": integer_solution.pattern_usages,
            "master_metadata": integer_solution.metadata or {},
            "sub_metadata": pricing_result.metadata or {},
        }

    # --- main driver -------------------------------------------------------
    def run(self, instance_name: str, *, started_at: float | None = None) -> dict[str, Any]:
        self.runtime.start(started_at)
        self._run_start_time = self.runtime.started_at
        self._run_end_time = None
        self.iter_records.clear()
        # Leave time for constructing and solving the final integer restricted master.
        self.runtime.stage_reserve = min(5.0, self.runtime.time_limit / 12.0)
        inst = self.load_instance(instance_name)
        patterns = self.initial_patterns(inst)
        if not patterns:
            raise RuntimeError("initial_patterns returned an empty list")
        last_master_sol = MasterLPSolution(math.nan, None)
        last_pricing = PricingResult(None, -math.inf)
        termination = "iteration_limit"
        try:
            for iteration in range(1, self.max_iters + 1):
                self.runtime.check()
                iter_start = time.perf_counter()
                master_sol = self.solve_restricted_master(inst, patterns)
                pricing = self.solve_pricing(inst, master_sol, patterns)
                # Preserve a certificate only as its complete master/pricing pair.
                last_master_sol, last_pricing = master_sol, pricing
                iter_end = time.perf_counter()
                record = CGIterationRecord(iteration, iter_end - self._run_start_time, iter_end - iter_start, master_sol.objective, pricing.reduced_cost, len(patterns))
                self.iter_records.append(record)
                if self.verbose:
                    print(self.format_iteration_log(record, master_sol, pricing))
                if self.should_terminate(iteration, pricing):
                    termination = "pricing_converged" if pricing.reduced_cost >= -self.rc_tolerance else "iteration_limit"
                    break
                if not self.add_pattern(patterns, pricing):
                    termination = "no_new_columns"
                    break
        except ReferenceLimit:
            termination = "time_limit"
        self.runtime.stage_reserve = 0.0
        integer = self.solve_integer_master(inst, patterns)
        self._run_end_time = time.perf_counter()
        result = self.finalize_result(inst, last_master_sol, last_pricing, integer, len(self.iter_records))
        if getattr(self, "phase_one_records", None):
            result["phase_one_records"] = self.phase_one_records
        return self.runtime.finish(result, target_gap=0.005, termination_reason=termination)
