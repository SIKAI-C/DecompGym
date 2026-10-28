"""Scale-up calibration: one monolithic MIP or one decomposition reference on one XL instance.

Records the incumbent/bound trajectory over time so that time-to-quality can be compared.
Run as: python -m src.experiments.scaleup_calibration --family ufl --data path/data.json --method mip --time-limit 3600 --out result.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB



FAMILIES = {
    "ufl": ("bd", "src.bd.ufl.b_gold_monolithic", "UFLBaseline", "src.bd.ufl.b_gold_bd", "UFLBenders"),
    "gap": ("cg", "src.cg.gap.a_gold_monolithic", "GAPBaseline", "src.cg.gap.a_gold_cg", "GAPColumnGeneration"),
}


def _import(module: str, name: str):
    return getattr(__import__(module, fromlist=[name]), name)


def run_mip(family: str, data: str, time_limit: float) -> dict:
    _, mono_mod, mono_cls, _, _ = FAMILIES[family]
    baseline = _import(mono_mod, mono_cls)(params={"Threads": 1, "TimeLimit": time_limit})
    started = time.perf_counter()
    inst = baseline.load_instance(data)
    model = baseline.build_model(inst)
    build_seconds = time.perf_counter() - started
    trace: list[dict] = []
    last = {"inc": math.inf, "bnd": -math.inf}

    def callback(m, where):
        if where == GRB.Callback.MIP:
            inc, bnd = m.cbGet(GRB.Callback.MIP_OBJBST), m.cbGet(GRB.Callback.MIP_OBJBND)
            # Log only changes, so the trace stays small over an hour.
            if inc < last["inc"] - 1e-9 or bnd > last["bnd"] + max(1e-6, 1e-4 * abs(bnd)):
                last.update(inc=inc, bnd=bnd)
                trace.append({"t": time.perf_counter() - started, "incumbent": inc if inc < GRB.INFINITY else None, "bound": bnd})

    model.optimize(callback)
    total = time.perf_counter() - started
    has_sol = model.SolCount > 0
    return {
        "status": int(model.Status),
        "objective": float(model.ObjVal) if has_sol else None,
        "bound": float(model.ObjBound),
        "mip_gap": float(model.MIPGap) if has_sol else None,
        "build_seconds": build_seconds,
        "total_seconds": total,
        "node_count": float(model.NodeCount),
        "num_vars": int(model.NumVars),
        "num_constrs": int(model.NumConstrs),
        "trace": trace,
    }


def run_reference(family: str, data: str, time_limit: float) -> dict:
    _, _, _, ref_mod, ref_cls = FAMILIES[family]
    solver = _import(ref_mod, ref_cls)(max_iters=10**6, time_limit=time_limit)
    started = time.perf_counter()
    error = None
    try:
        res = solver.run(data)
    except Exception as exc:  # A reference that hits its budget may raise; keep the evidence.
        res, error = {}, f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - started
    records = [
        {k: getattr(r, k) for k in ("iteration", "elapsed_time", "master_obj", "global_lb", "global_ub") if hasattr(r, k)}
        for r in getattr(solver, "iter_records", [])
    ]
    return {
        "objective": res.get("best_objective"),
        "global_lb": res.get("global_lb"),
        "global_ub": res.get("global_ub"),
        "iters": res.get("iters"),
        "termination": res.get("termination_reason"),
        "error": error,
        "total_seconds": total,
        "model_solves": len(solver.runtime.model_solves),
        "trace": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--method", choices=["mip", "ref"], required=True)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run_mip(args.family, args.data, args.time_limit) if args.method == "mip" else run_reference(args.family, args.data, args.time_limit)
    result.update({
        "family": args.family, "data": args.data, "method": args.method, "time_limit": args.time_limit,
        "gurobi": ".".join(map(str, gp.gurobi.version())), "host": platform.node(),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, default=str))


if __name__ == "__main__":
    main()
