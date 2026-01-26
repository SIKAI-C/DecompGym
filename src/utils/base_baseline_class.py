from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import time
from typing import Any, Generic, TypeVar


TInst = TypeVar("TInst")
TModel = TypeVar("TModel")


@dataclass
class BaselineSolution(Generic[TModel]):
    """Final solution of the one-shot MILP/MIP baseline.

    - `objective`: final objective value reported by the solver.
    - `vars`: dictionary mapping variable names (or other keys) to values.
    - `best_bound`: solver's best bound on the optimal value, if available.
        For minimization, this is usually a global LOWER bound.
        For maximization, it is usually a global UPPER bound.
    - `model_state`: the underlying solver model, if you want to keep it.
    """

    objective: float
    vars: dict[str, Any]
    best_bound: float | None = None
    model_state: TModel | None = None
    metadata: dict[str, Any] | None = None


class BaseBaseline(ABC, Generic[TInst, TModel]):
    """Base class for baseline models that solve the full MILP/MIP
    directly (e.g., with Gurobi) without Benders.

    It is designed so that its `run()` method returns a result dictionary
    compatible with the Benders base class:

        {
            "best_objective": ...,
            "global_lb": ...,
            "global_ub": ...,
            "iters": 1,
            "total_time": ...,
            "iter_records": [],              # empty for baseline
            "final_master_objective": ...,
            "final_vars": ...,
            "master_metadata": ...,
            "sub_metadata": {},              # empty for baseline
        }
    """

    is_minimization: bool = True

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._run_start_time: float | None = None
        self._run_end_time: float | None = None

    # --- required hooks --------------------------------------------------------
    @abstractmethod
    def load_instance(self, instance_name: str) -> TInst:
        """Return the problem instance referenced by name."""

    @abstractmethod
    def build_model(self, inst: TInst) -> TModel:
        """Build the full MILP/MIP model (e.g., a gurobipy.Model) for the instance."""

    @abstractmethod
    def solve_model(self, inst: TInst, model: TModel) -> BaselineSolution[TModel]:
        """Solve the model and return a BaselineSolution object.

        In a Gurobi-based subclass, you’ll typically:
          * call model.optimize(),
          * read model.objVal, model.ObjBound (or similar),
          * extract variable values into a dict,
          * and wrap them in BaselineSolution.
        """

    # --- optional hooks --------------------------------------------------------
    def on_before_run(self, inst: TInst) -> None:
        """Reset any per-run state (override in subclasses if needed)."""

    def finalize_result(
        self,
        inst: TInst,
        solution: BaselineSolution[TModel],
    ) -> dict[str, Any]:
        """Build a result dictionary compatible with the Benders base class.

        For the baseline:
          * `iters` is set to 1 (one global solve).
          * `iter_records` is an empty list.
          * `global_lb` / `global_ub` are inferred from `best_bound`
            and the objective based on `is_minimization`.
        """
        if self._run_start_time is None or self._run_end_time is None:
            total_time = math.nan
        else:
            total_time = self._run_end_time - self._run_start_time

        # Infer global LB/UB using the solver's best bound if provided.
        if solution.best_bound is not None:
            if self.is_minimization:
                global_lb = solution.best_bound
                global_ub = solution.objective
            else:
                global_lb = solution.objective
                global_ub = solution.best_bound
        else:
            # If no bound reported, treat objective as both LB and UB.
            global_lb = solution.objective
            global_ub = solution.objective

        if self.verbose:
            print(
                f"[baseline] obj={solution.objective:.6f} "
                f"bound={solution.best_bound if solution.best_bound is not None else float('nan'):.6f} "
                f"LB={global_lb:.6f} UB={global_ub:.6f} "
                f"time={total_time:.3f}s"
            )

        return {
            "best_objective": solution.objective,
            "global_lb": global_lb,
            "global_ub": global_ub,
            "iters": 1,
            "total_time": total_time,
            "iter_records": [],  # baseline has no iteration log
            "final_master_objective": solution.objective,
            "final_vars": solution.vars,
            "master_metadata": solution.metadata or {},
            "sub_metadata": {},  # no subproblems in the baseline
        }

    # --- main driver -----------------------------------------------------------
    def run(self, instance_name: str) -> dict[str, Any]:
        """Load an instance, build the full model, solve it, and return results."""
        inst = self.load_instance(instance_name)
        self.on_before_run(inst)

        model = self.build_model(inst)

        self._run_start_time = time.perf_counter()
        solution = self.solve_model(inst, model)
        self._run_end_time = time.perf_counter()

        return self.finalize_result(inst, solution)
