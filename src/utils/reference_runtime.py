"""One wall-clock budget and explicit certificates for decomposition references."""
from __future__ import annotations

import math
import time


REFERENCE_POLICY = "strict-calibrated-references-20261013-v2"


class ReferenceLimit(RuntimeError):
    """A decomposition stage has exhausted its allotted runtime."""


class NoReferenceIncumbent(RuntimeError):
    """No complete original-problem feasible plan is available to publish."""


def relative_gap(lower, upper):
    if lower is None or upper is None or not math.isfinite(lower) or not math.isfinite(upper):
        return None
    # A substantially crossed bound is an inconsistency, not an optimality proof.
    if lower > upper + 1e-8 * max(1.0, abs(upper)):
        return None
    return abs(upper - lower) / max(1.0, abs(upper))


class ReferenceRuntime:
    """Shared by every model in a reference, including initialization and finalization."""

    def __init__(self, time_limit=60.0, output_reserve=6.0):
        if not math.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("time_limit must be positive and finite")
        self.time_limit = float(time_limit)
        self.output_reserve = min(float(output_reserve), self.time_limit * 0.1)
        self.started_at = None
        self.stage_reserve = 0.0
        self.model_solves = []

    def start(self, started_at=None):
        self.started_at = time.perf_counter() if started_at is None else started_at
        self.stage_reserve = 0.0
        self.model_solves.clear()

    def elapsed(self):
        return 0.0 if self.started_at is None else time.perf_counter() - self.started_at

    def remaining(self):
        if self.started_at is None:
            self.start()
        return self.time_limit - self.output_reserve - self.stage_reserve - self.elapsed()

    def check(self):
        remaining = self.remaining()
        if remaining <= 0:
            raise ReferenceLimit("Reference stage exhausted the shared wall-clock budget")
        return remaining

    def optimize(self, model, *, role, allow_incumbent=False):
        from gurobipy import GRB

        remaining = self.check()
        # Apply these after user parameters, to every model and every re-solve.
        model.Params.Threads = 1
        model.Params.TimeLimit = min(float(model.Params.TimeLimit), remaining)
        if role == "pricing":
            model.Params.MIPGap = 0.0
            model.Params.MIPGapAbs = 0.0
        started = time.perf_counter()
        model.optimize()
        self.model_solves.append({"role": role, "name": model.ModelName, "threads": model.Params.Threads, "time_limit": model.Params.TimeLimit, "seconds": time.perf_counter() - started, "status": model.Status, "solution_count": model.SolCount})
        # A just-finished solve is useful even if it crossed the internal cutoff.
        # In particular, a timed-out integer solve may have a feasible incumbent.
        if model.Status == GRB.TIME_LIMIT and not (allow_incumbent and model.SolCount > 0):
            raise ReferenceLimit(f"{role} reached its time limit without an admissible completed result")

    def finish(self, result, *, target_gap, termination_reason):
        for key in ("global_lb", "global_ub"):
            if result.get(key) is not None and not math.isfinite(result[key]):
                result[key] = None
        gap = relative_gap(result.get("global_lb"), result.get("global_ub"))
        optimal = gap is not None and gap <= 1e-8
        result.update(
            reference_policy=REFERENCE_POLICY,
            status="optimal" if optimal else "feasible",
            termination_reason=termination_reason,
            relative_gap=gap,
            target_gap=target_gap,
            target_gap_met=gap is not None and gap <= target_gap,
            proven_optimal=optimal,
            optimality_tolerance=1e-8,
            time_limit=self.time_limit,
            threads=1,
            output_reserve=self.output_reserve,
            model_solves=list(self.model_solves),
        )
        result["total_time"] = self.elapsed()
        return result
