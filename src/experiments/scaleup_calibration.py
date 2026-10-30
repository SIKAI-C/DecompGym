"""Scale-up calibration: one solve (monolithic MIP or a decomposition reference) of one instance.

Every method shares one wall clock started at process start, so model building, loading and imports
count against the same budget T. Each new incumbent is stored as a solution witness whose objective is
recomputed here, independently of the solver; the best witness is written in the family's output
contract and checked with OutputVerifier. Results are written atomically with a fingerprint, and a
complete result with the same fingerprint is not recomputed (safe under Slurm requeue).

Run as: python -m src.experiments.scaleup_calibration --family ufl --data path/data.json --method mip --time-limit 3600 --out result.json
"""
from __future__ import annotations

import time

T0 = time.monotonic()  # before the heavy imports: the budget starts here

import argparse
import hashlib
import json
import math
import os
import platform
import resource
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import numpy as np

SUITE = {"ufl": "bd", "gap": "cg"}
SOURCES = {
    "ufl": ["src/bd/ufl/b_gold_monolithic.py", "src/bd/ufl/b_gold_bd.py", "src/bd/ufl/b_ref_bd_xl.py", "src/bd/ufl/a_gen_data.py"],
    "gap": ["src/cg/gap/a_gold_monolithic.py", "src/cg/gap/a_gold_cg.py", "src/cg/gap/b_ref_cg_xl.py", "src/cg/gap/a_gen_data.py"],
}


def elapsed() -> float:
    return time.monotonic() - T0


def remaining(limit: float) -> float:
    return max(1.0, limit - elapsed())


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- family adapters

class UFLAdapter:
    def __init__(self, data: str) -> None:
        from src.bd.ufl.a_gen_data import UFLInstance
        self.inst = UFLInstance.from_json_file(data)
        self.D, self.f = self.inst.D, self.inst.f

    def objective(self, open_: list[int]) -> float:
        idx = np.asarray(sorted(open_), dtype=int)
        return float(self.f[idx].sum() + self.D[idx].min(axis=0).sum())

    def output(self, open_: list[int], status: str = "feasible") -> dict:
        idx = np.asarray(sorted(open_), dtype=int)
        y = np.zeros(self.inst.m, dtype=int)
        y[idx] = 1
        x = np.zeros((self.inst.m, self.inst.n), dtype=int)
        x[idx[np.argmin(self.D[idx], axis=0)], np.arange(self.inst.n)] = 1
        return {"status": status, "method": "benders_decomposition", "objective_value": self.objective(open_), "y": y.tolist(), "x": x.tolist()}


class GAPAdapter:
    def __init__(self, data: str) -> None:
        from src.cg.gap.a_gen_data import GAPInstance
        from src.cg.gap.b_ref_cg_xl import arrays
        self.cost, self.res, self.cap = arrays(GAPInstance.from_json_file(data))

    def objective(self, assign: list[int]) -> float | None:
        a = np.asarray(assign, dtype=int)
        n = len(a)
        if n != self.cost.shape[1] or a.min() < 0:
            return None
        load = np.bincount(a, weights=self.res[a, np.arange(n)], minlength=len(self.cap))
        return float(self.cost[a, np.arange(n)].sum()) if np.all(load <= self.cap) else None

    def output(self, assign: list[int], status: str = "feasible") -> dict:
        return {"status": status, "method": "column_generation", "objective_value": self.objective(assign),
                "solution": {"assignments": [{"job_id": i, "machine_id": int(k)} for i, k in enumerate(assign)]}}


ADAPTERS = {"ufl": UFLAdapter, "gap": GAPAdapter}


class Recorder:
    """Incumbent witnesses (objective recomputed here) and bound history on the shared clock."""

    def __init__(self, adapter) -> None:
        self.adapter = adapter
        self.witnesses: list[dict] = []
        self.bounds: list[dict] = []
        self.best: float = math.inf

    def solution(self, sol: list[int], source: str) -> None:
        obj = self.adapter.objective(sol)
        if obj is not None and obj < self.best - 1e-9:
            self.best = obj
            self.witnesses.append({"t": elapsed(), "objective": obj, "solution": [int(v) for v in sol], "source": source})

    def bound(self, value: float | None, source: str) -> None:
        if value is None or not math.isfinite(value):
            return
        if not self.bounds or value > self.bounds[-1]["bound"] + 1e-9 * max(1.0, abs(value)):
            self.bounds.append({"t": elapsed(), "bound": float(value), "source": source})


# ---------------------------------------------------------------- methods

def run_mip(family: str, data: str, limit: float, rec: Recorder) -> dict:
    a = rec.adapter
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    if family == "ufl":
        # Strong formulation; assignments may be continuous because, for fixed binary y, the
        # assignment LP is integral, and witnesses are decoded by nearest open facility.
        m, n = a.D.shape
        y = model.addMVar(m, vtype=GRB.BINARY)
        x = model.addMVar((m, n), lb=0.0, ub=1.0)
        model.addConstr(x.sum(axis=0) == 1)
        model.addConstr(x <= y[:, None] * np.ones((1, n)))
        model.setObjective(a.f @ y + (a.D * x).sum(), GRB.MINIMIZE)
        decode = lambda get: list(np.nonzero(np.asarray(get(y)) > 0.5)[0])
    else:
        m, n = a.cost.shape
        x = model.addMVar((m, n), vtype=GRB.BINARY)
        model.addConstr(x.sum(axis=0) == 1)
        model.addConstr((a.res * x).sum(axis=1) <= a.cap)
        model.setObjective((a.cost * x).sum(), GRB.MINIMIZE)
        decode = lambda get: list(np.argmax(np.asarray(get(x)), axis=0))
    build_seconds = elapsed()
    model.Params.TimeLimit = remaining(limit)

    def callback(cb, where):
        if where == GRB.Callback.MIPSOL:
            rec.solution(decode(cb.cbGetSolution), "mip")
        elif where == GRB.Callback.MIP:
            rec.bound(cb.cbGet(GRB.Callback.MIP_OBJBND), "mip")

    model.optimize(callback)
    if model.SolCount:
        rec.solution(decode(lambda v: v.X), "mip_final")
    rec.bound(float(model.ObjBound), "mip_final")
    return {"status": int(model.Status), "proved_optimal": model.Status == GRB.OPTIMAL, "build_seconds": build_seconds,
            "node_count": float(model.NodeCount), "num_vars": int(model.NumVars), "num_constrs": int(model.NumConstrs)}


def run_paper_reference(family: str, data: str, limit: float, rec: Recorder) -> dict:
    """The paper's reference algorithm (naive at this scale), given the remaining budget."""
    from src.bd.ufl.b_gold_bd import UFLBenders
    from src.cg.gap.a_gold_cg import GAPColumnGeneration
    solver = {"ufl": UFLBenders, "gap": GAPColumnGeneration}[family](max_iters=10**6, time_limit=remaining(limit))
    error = None
    try:
        res = solver.run(data)
    except Exception as exc:  # a reference that hits its budget may raise; keep the evidence
        res, error = {}, f"{type(exc).__name__}: {exc}"
    if family == "ufl" and res.get("final_vars"):
        rec.solution([int(k.split("_")[1]) for k, v in res["final_vars"].items() if k.startswith("y_") and v > 0.5], "paper_reference_final")
    records = [{k: getattr(r, k) for k in ("iteration", "elapsed_time", "master_obj", "global_lb", "global_ub") if hasattr(r, k)} for r in getattr(solver, "iter_records", [])]
    # The paper's references report bounds per iteration on their own clock; they are kept as diagnostics only.
    return {"claimed_objective": res.get("best_objective"), "claimed_lb": res.get("global_lb"), "iters": res.get("iters"),
            "error": error, "iteration_records": records,
            "witness_note": None if family == "ufl" else "GAP paper reference exposes pattern usages only; objective unverified"}


def run_engineered_reference(family: str, data: str, limit: float, rec: Recorder) -> dict:
    if family == "ufl":
        from src.bd.ufl.b_ref_bd_xl import UFLBendersXL
        solver = UFLBendersXL(time_limit=remaining(limit), threads=1, on_incumbent=lambda s: rec.solution(s, "ref_xl"), on_bound=lambda b: rec.bound(b, "ref_xl"))
    else:
        from src.cg.gap.b_ref_cg_xl import GAPColumnGenerationXL
        solver = GAPColumnGenerationXL(time_limit=remaining(limit), threads=1, on_incumbent=lambda s: rec.solution(s, "ref_xl"), on_bound=lambda b: rec.bound(b, "ref_xl"))
    res = solver.run(data)
    return {k: v for k, v in res.items() if k not in ("trace", "assignment", "open")}


METHODS = {"mip": run_mip, "ref": run_paper_reference, "ref_xl": run_engineered_reference}


# ---------------------------------------------------------------- driver

def fingerprint(args) -> dict:
    return {"family": args.family, "method": args.method, "time_limit": args.time_limit, "data_sha256": sha256(args.data),
            "source_sha256": {p: sha256(p) for p in SOURCES[args.family] + ["src/experiments/scaleup_calibration.py"]},
            "gurobi": ".".join(map(str, gp.gurobi.version()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    fp = fingerprint(args)
    if out.exists():
        prior = json.loads(out.read_text())
        if prior.get("fingerprint") == fp and prior.get("complete"):
            print(f"complete result exists: {out}")
            return
    rec = Recorder(ADAPTERS[args.family](args.data))
    details = METHODS[args.method](args.family, args.data, args.time_limit, rec)
    total = elapsed()
    result = {"fingerprint": fp, "complete": True, "family": args.family, "method": args.method, "data": args.data,
              "time_limit": args.time_limit, "total_seconds": total, "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
              "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"), "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 if platform.system() == "Linux" else 1024.0 ** 2),
              "best_objective": rec.best if rec.witnesses else None, "best_bound": rec.bounds[-1]["bound"] if rec.bounds else None,
              "details": details, "witness_count": len(rec.witnesses), "bounds": rec.bounds,
              "incumbents": [{k: w[k] for k in ("t", "objective", "source")} for w in rec.witnesses]}
    if rec.witnesses:
        from src.experiments.verification import OutputVerifier
        best = rec.witnesses[-1]
        out.parent.mkdir(parents=True, exist_ok=True)
        sol_path = out.with_suffix(".output.json")
        sol_path.write_text(json.dumps(rec.adapter.output(best["solution"])))
        result["verification"] = OutputVerifier.verify_reference_output(SUITE[args.family], args.family, Path(args.data), sol_path, 1e-6)
        result["witnesses_path"] = str(out.with_suffix(".witnesses.json"))
        out.with_suffix(".witnesses.json").write_text(json.dumps(rec.witnesses))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1, default=str))
    os.replace(tmp, out)
    print(json.dumps({k: result[k] for k in ("family", "method", "total_seconds", "best_objective", "best_bound", "witness_count")} | {"verified": result.get("verification", {}).get("ok")}, default=str))


if __name__ == "__main__":
    main()
