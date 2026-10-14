from __future__ import annotations

from copy import deepcopy

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Generic, TypeVar

from src.utils.reference_runtime import NoReferenceIncumbent, ReferenceLimit, ReferenceRuntime, relative_gap


TInst = TypeVar("TInst")
TMaster = TypeVar("TMaster")
TFeasCut = TypeVar("TFeasCut")
TOptCut = TypeVar("TOptCut")


@dataclass
class MasterSolution(Generic[TMaster]):
    """Solution of the current Benders master problem.

    `vars` typically holds the current values of the complicating variables
    (e.g., y, theta) that the subproblems will see.
    """

    objective: float
    vars: dict[str, Any]
    master_state: TMaster
    metadata: dict[str, Any] | None = None
    bound: float | None = None  # certified master bound, distinct from its incumbent objective


@dataclass
class SubproblemResult(Generic[TFeasCut, TOptCut]):
    """Result of solving all Benders subproblems for a given master solution.

    - `sub_bound` is an estimate of the true recourse contribution for this
        master solution (subclasses must define whether it is a lower or
        upper bound, but be consistent).
    """

    feasibility_cuts: Sequence[TFeasCut]
    optimality_cuts: Sequence[TOptCut]
    all_feasible: bool
    worst_violation: float
    sub_bound: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class IterationRecord:
    """Bookkeeping for convergence plots and logging."""

    iteration: int
    elapsed_time: float  # time from start of run
    iter_time: float  # time spent in this iteration
    master_obj: float  # master objective in this iteration
    sub_bound: float | None  # per-iter subproblem bound, if any
    global_lb: float  # best known global lower bound so far
    global_ub: float  # best known global upper bound so far
    n_feas_cuts: int
    n_opt_cuts: int


class BaseBenders(ABC, Generic[TInst, TMaster, TFeasCut, TOptCut]):
    """Generic Benders decomposition loop.

    Subclasses implement:
      * how to load an instance,
      * how to build the initial master,
      * how to solve the master,
      * how to solve subproblems and generate feasibility/optimality cuts,
      * how to insert those cuts into the master.
    """

    violation_tolerance: float = 1e-6
    is_minimization: bool = True  # subclasses can flip if they solve a max problem

    def __init__(self, max_iters: int = 500, verbose: bool = False, *, time_limit: float = 60.0, target_gap: float = 1e-4) -> None:
        self.max_iters = max_iters
        self.verbose = verbose
        self.runtime = ReferenceRuntime(time_limit)
        self.target_gap = target_gap

        # Bounds and iteration records
        self.global_lb: float = -math.inf
        self.global_ub: float = math.inf
        self.iter_records: list[IterationRecord] = []
        self._incumbent: dict[str, Any] | None = None

        # Total timing
        self._run_start_time: float | None = None
        self._run_end_time: float | None = None

    # --- required hooks --------------------------------------------------------
    @abstractmethod
    def load_instance(self, instance_name: str) -> TInst:
        """Return the problem instance referenced by name."""

    @abstractmethod
    def build_initial_master(self, inst: TInst) -> TMaster:
        """Construct the initial master problem (variables, base constraints,
        optional initial cuts) and return a handle to it.
        """

    @abstractmethod
    def solve_master(self, inst: TInst, master: TMaster) -> MasterSolution[TMaster]:
        """Solve the current master problem and return the objective and the
        current values of the complicating variables that the subproblems see.
        """

    @abstractmethod
    def solve_subproblems(
        self,
        inst: TInst,
        master_solution: MasterSolution[TMaster],
    ) -> SubproblemResult[TFeasCut, TOptCut]:
        """Solve all Benders subproblems for the current master solution and
        return feasibility and optimality cuts.
        """

    @abstractmethod
    def add_feasibility_cuts(
        self,
        inst: TInst,
        master: TMaster,
        cuts: Sequence[TFeasCut],
    ) -> None:
        """Insert feasibility cuts into the master problem."""

    @abstractmethod
    def add_optimality_cuts(
        self,
        inst: TInst,
        master: TMaster,
        cuts: Sequence[TOptCut],
    ) -> None:
        """Insert optimality cuts into the master problem."""

    # --- optional hooks --------------------------------------------------------
    def on_before_run(self, inst: TInst) -> None:
        """Reset per-run state. Subclasses override when needed."""
        self.global_lb = -math.inf
        self.global_ub = math.inf
        self.iter_records.clear()
        self._incumbent = None

    def compute_iteration_bounds(
        self,
        inst: TInst,
        master_solution: MasterSolution[TMaster],
        sub_result: SubproblemResult[TFeasCut, TOptCut],
    ) -> tuple[float, float]:
        """Update global lower/upper bounds given the current iteration.

        Default convention for minimization:
            - master objective is an upper bound,
            - `sub_result.sub_bound` (if provided) is a lower bound.
        Subclasses can override if they use a different convention.
        """
        master_obj = master_solution.objective
        sub_bound = sub_result.sub_bound

        if self.is_minimization:
            # Master is UB, sub_bound is LB (if provided)
            if not math.isinf(self.global_ub):
                self.global_ub = min(self.global_ub, master_obj)
            else:
                self.global_ub = master_obj

            if sub_bound is not None:
                if not math.isinf(self.global_lb):
                    self.global_lb = max(self.global_lb, sub_bound)
                else:
                    self.global_lb = sub_bound
        else:
            # For maximization, reverse roles
            if not math.isinf(self.global_lb):
                self.global_lb = max(self.global_lb, master_obj)
            else:
                self.global_lb = master_obj

            if sub_bound is not None:
                if not math.isinf(self.global_ub):
                    self.global_ub = min(self.global_ub, sub_bound)
                else:
                    self.global_ub = sub_bound

        return self.global_lb, self.global_ub

    def format_iteration_log(
        self,
        iteration: int,
        master_solution: MasterSolution[TMaster],
        sub_result: SubproblemResult[TFeasCut, TOptCut],
        global_lb: float,
        global_ub: float,
    ) -> str:
        """Pretty printer for verbose iteration-level logging."""
        return (
            f"[iter {iteration:02d}] "
            f"master_obj={master_solution.objective:.6f} "
            f"sub_bound={sub_result.sub_bound if sub_result.sub_bound is not None else float('nan'):.6f} "
            f"LB={global_lb:.6f} UB={global_ub:.6f} "
            f"violation={sub_result.worst_violation:.3e} "
            f"n_feas={len(sub_result.feasibility_cuts)} "
            f"n_opt={len(sub_result.optimality_cuts)}"
        )

    def should_terminate(
        self,
        iteration: int,
        master_solution: MasterSolution[TMaster],
        sub_result: SubproblemResult[TFeasCut, TOptCut],
        global_lb: float,
        global_ub: float,
    ) -> bool:
        """Default stopping criterion.
          * all subproblems feasible,
          * violations small,
          * and (if both LB/UB are finite) their gap is small,
          * or iteration limit reached.

        Subclasses can override for a more problem-specific rule.
        """
        gap = relative_gap(global_lb, global_ub)
        return (self._incumbent is not None and gap is not None and gap <= self.target_gap) or iteration >= self.max_iters

    def on_cuts_added(
        self,
        master: TMaster,
        feasibility_cuts: Sequence[TFeasCut],
        optimality_cuts: Sequence[TOptCut],
    ) -> None:
        """Hook for maintaining auxiliary state after inserting new cuts."""

    def finalize_result(
        self,
        inst: TInst,
        master_solution: MasterSolution[TMaster],
        sub_result: SubproblemResult[TFeasCut, TOptCut],
        iterations: int,
    ) -> dict[str, Any]:
        """Build the result dictionary returned by `run`.

        Contains:
            - best objective (according to LB/UB),
            - final LB/UB,
            - number of iterations,
            - total runtime,
            - per-iteration records,
            - final variable values, and metadata.
        """
        if self._run_start_time is None or self._run_end_time is None:
            total_time = math.nan
        else:
            total_time = self._run_end_time - self._run_start_time

        if self._incumbent is None:
            raise NoReferenceIncumbent("Benders stopped without a complete feasible incumbent")
        # The saved plan and its feasible objective were committed together.
        if self.is_minimization:
            best_obj = self.global_ub
        else:
            best_obj = self.global_lb

        return {
            "best_objective": best_obj,
            "global_lb": self.global_lb,
            "global_ub": self.global_ub,
            "iters": iterations,
            "total_time": total_time,
            "iter_records": self.iter_records,  # list[IterationRecord]
            "final_master_objective": master_solution.objective,
            "final_vars": self._incumbent["vars"] if self._incumbent else master_solution.vars,
            "master_metadata": self._incumbent["master_metadata"] if self._incumbent else (master_solution.metadata or {}),
            "sub_metadata": self._incumbent["sub_metadata"] if self._incumbent else (sub_result.metadata or {}),
            "incumbent_iteration": self._incumbent["iteration"] if self._incumbent else None,
        }

    # --- plotting helper -------------------------------------------------------
    def plot_bounds(self, output_dir: str | None = None) -> str | None:
        """Plot convergence diagnostics (bounds, cuts, iteration times).

        If `output_dir` is provided, the figure is saved there as
        `benders_progress.pdf` and the path is returned. Otherwise, the plot is
        only shown.
        """
        import matplotlib.pyplot as plt

        if not self.iter_records:
            raise RuntimeError("No iteration records to plot. Did you call `run`?")

        iters = [rec.iteration for rec in self.iter_records]
        lbs = [rec.global_lb for rec in self.iter_records]
        ubs = [rec.global_ub for rec in self.iter_records]
        feas_counts = [rec.n_feas_cuts for rec in self.iter_records]
        opt_counts = [rec.n_opt_cuts for rec in self.iter_records]
        iter_times = [rec.iter_time for rec in self.iter_records]

        fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
        ax_bounds, ax_cuts, ax_times = axes

        ax_bounds.plot(iters, lbs, marker="o", label="Lower bound")
        ax_bounds.plot(iters, ubs, marker="x", label="Upper bound")
        ax_bounds.set_ylabel("Objective bound")
        ax_bounds.set_title("Benders bounds")
        ax_bounds.legend()
        ax_bounds.grid(True)

        ax_cuts.plot(iters, feas_counts, marker="o", label="Feasibility cuts")
        ax_cuts.plot(iters, opt_counts, marker="x", label="Optimality cuts")
        ax_cuts.set_ylabel("# cuts")
        ax_cuts.set_title("Cuts per iteration")
        ax_cuts.legend()
        ax_cuts.grid(True)

        ax_times.plot(iters, iter_times, marker="o", color="tab:gray")
        ax_times.set_xlabel("Iteration")
        ax_times.set_ylabel("Time (s)")
        ax_times.set_title("Iteration runtime")
        ax_times.grid(True)

        output_path: str | None = None
        if output_dir is not None:
            output_dir_path = Path(output_dir).expanduser().resolve()
            output_dir_path.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir_path / "benders_progress.pdf")
            fig.savefig(output_path, bbox_inches="tight", format="pdf")
            plt.close(fig)
            return output_path

        plt.show()
        plt.close(fig)
        return output_path

    # --- main driver -----------------------------------------------------------
    def run(self, instance_name: str, *, started_at: float | None = None) -> dict[str, Any]:
        """Return the best complete paired plan within one shared runtime budget."""
        self.runtime.start(started_at)
        self._run_start_time = self.runtime.started_at
        self._run_end_time = None
        inst = self.load_instance(instance_name)
        self.on_before_run(inst)
        master = self.build_initial_master(inst)
        last_master_sol = None
        last_sub_result = None
        termination = "iteration_limit"
        try:
            for iteration in range(1, self.max_iters + 1):
                self.runtime.check()
                iter_start = time.perf_counter()
                current_master = self.solve_master(inst, master)
                # This bound is valid even if recourse is infeasible or interrupted.
                if current_master.bound is not None and math.isfinite(current_master.bound):
                    if self.is_minimization:
                        self.global_lb = max(self.global_lb, current_master.bound)
                    else:
                        self.global_ub = min(self.global_ub, current_master.bound)
                current_sub = self.solve_subproblems(inst, current_master)
                previous_bound = self.global_ub if self.is_minimization else self.global_lb
                global_lb, global_ub = self.compute_iteration_bounds(inst, current_master, current_sub)
                incumbent_bound = global_ub if self.is_minimization else global_lb
                improved = incumbent_bound < previous_bound if self.is_minimization else incumbent_bound > previous_bound
                if current_sub.all_feasible and math.isfinite(incumbent_bound) and improved:
                    self._incumbent = {
                        "objective": incumbent_bound,
                        "vars": deepcopy(current_master.vars),
                        "master_metadata": deepcopy(current_master.metadata or {}),
                        "sub_metadata": deepcopy(current_sub.metadata or {}),
                        "iteration": iteration,
                    }
                # Commit only a completed master/recourse pair, never partial scenarios.
                last_master_sol, last_sub_result = current_master, current_sub
                iter_end = time.perf_counter()
                self.iter_records.append(IterationRecord(
                    iteration=iteration, elapsed_time=iter_end - self._run_start_time,
                    iter_time=iter_end - iter_start, master_obj=current_master.objective,
                    sub_bound=current_sub.sub_bound, global_lb=global_lb, global_ub=global_ub,
                    n_feas_cuts=len(current_sub.feasibility_cuts), n_opt_cuts=len(current_sub.optimality_cuts),
                ))
                if self.verbose:
                    print(self.format_iteration_log(iteration, current_master, current_sub, global_lb, global_ub))
                if self.should_terminate(iteration, current_master, current_sub, global_lb, global_ub):
                    gap = relative_gap(global_lb, global_ub)
                    termination = "target_gap" if gap is not None and gap <= self.target_gap else "iteration_limit"
                    break
                if not current_sub.feasibility_cuts and not current_sub.optimality_cuts:
                    termination = "no_new_cuts"
                    break
                if current_sub.feasibility_cuts:
                    self.add_feasibility_cuts(inst, master, current_sub.feasibility_cuts)
                if current_sub.optimality_cuts:
                    self.add_optimality_cuts(inst, master, current_sub.optimality_cuts)
                self.on_cuts_added(master, current_sub.feasibility_cuts, current_sub.optimality_cuts)
        except ReferenceLimit:
            termination = "time_limit"
        if self._incumbent is None:
            raise NoReferenceIncumbent(f"Benders stopped ({termination}) without a complete feasible incumbent")
        self._run_end_time = time.perf_counter()
        result = self.finalize_result(inst, last_master_sol, last_sub_result, len(self.iter_records))
        return self.runtime.finish(result, target_gap=self.target_gap, termination_reason=termination)
