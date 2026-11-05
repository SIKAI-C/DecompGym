"""Short Phoenix probes of the vertex-coloring methods (scale-up study; development only, not calibration data).

Runs every (instance, method) pair for T seconds in sequence, on a clock started just before the solver object is
built (instance loading counts). Publications are timestamped on receipt; those at or after T are counted as late
and ignored. The best coloring is rechecked with the evaluator and with the independent verifier.

    python scripts/phoenix/scaleup_coloring_probe.py --time-limit 120 --methods mip naive eng \
        --instances outputs/scaleup/raw/coloring/dimacs1993/le450_15c.col ...
(space-separated lists: sbatch --export splits on commas)
Methods: mip (compact MIP, assignment model), mip_pop2 (POP2 model), naive (naive CG), eng (engineered B&P), eng_notabu (without TabuCol), tabu (DSATUR +
TabuCol only: the non-decomposition heuristic control).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cg.coloring.a_data import load_col  # noqa: E402
from src.cg.coloring.b_mip_xl import ColoringCompactMIP, color_count, coloring_output  # noqa: E402
from src.cg.coloring.c_naive_cg import ColoringNaiveCG  # noqa: E402
from src.cg.coloring.d_ref_cg_xl import ColoringBranchAndPrice  # noqa: E402
from src.cg.coloring.e_verifier import verify_solution  # noqa: E402
from src.cg.coloring.heuristic import TabuCol, dsatur  # noqa: E402


class TabuControl:
    """DSATUR, then TabuCol at one color fewer until the deadline (no decomposition)."""

    def __init__(self, *, time_limit, on_incumbent, on_bound, **_):
        self.time_limit, self.on_incumbent = time_limit, on_incumbent

    def run(self, path):
        t0 = time.perf_counter()
        A = load_col(path).adjacency
        col = dsatur(A)
        self.on_incumbent(col.tolist())
        tc = TabuCol(A, seed=0)
        best = tc.reduce(col, t0 + self.time_limit, on_coloring=lambda c: self.on_incumbent(c.tolist()))
        return {"best_objective": int(best.max()) + 1, "global_lb": None, "termination_reason": "time_limit",
                "tabu_iterations": tc.iterations}


METHODS = {"mip": (ColoringCompactMIP, {}), "mip_pop2": (ColoringCompactMIP, {"formulation": "pop2"}), "naive": (ColoringNaiveCG, {}), "eng": (ColoringBranchAndPrice, {}),
           "eng_notabu": (ColoringBranchAndPrice, {"tabu": False}), "tabu": (TabuControl, {})}


def probe(path: str, method: str, T: float) -> dict:
    inst = load_col(path)
    rec = {"inc": [], "bounds": [], "late": 0, "best": None, "coloring": None, "stages": []}
    t0 = time.perf_counter()

    def on_inc(coloring, stage=None):
        t = time.perf_counter() - t0
        if t >= T:
            rec["late"] += 1
            return
        k = color_count(inst, list(coloring))
        if k is None:
            rec["inc"].append((round(t, 2), None))
            return
        if rec["best"] is None or k < rec["best"]:
            rec["best"], rec["coloring"] = k, list(coloring)
            rec["inc"].append((round(t, 2), k, stage))

    def on_bound(b):
        t = time.perf_counter() - t0
        if t < T and (not rec["bounds"] or b > rec["bounds"][-1][1]):
            rec["bounds"].append((round(t, 2), b))

    cls, kw = METHODS[method]
    try:
        res = cls(time_limit=T, threads=1, on_incumbent=on_inc, on_bound=on_bound, soft_mem_gb=12.0, **kw).run(path)
        err = None
    except Exception as exc:  # keep the evidence
        res, err = {}, f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - t0
    ver = None
    if rec["coloring"] is not None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(coloring_output(inst, rec["coloring"]), fh)
        ver = verify_solution(path, fh.name)["ok"]
    t_best = next((t for t, k, *_ in rec["inc"] if k == rec["best"]), None)
    keep = ("termination_reason", "H", "clique_size", "cover_cliques", "build_seconds", "num_vars", "num_constrs",
            "node_count", "cg_iterations", "cg_converged", "cg_stop_reason", "columns", "lp_value", "rmp_status",
            "cg_iters", "exact_rounds", "heuristic_rounds", "mispricings", "early_stops", "exact_seconds", "lp_seconds",
            "pricing_seconds", "mip_runs", "dives", "tabu_seconds", "tabu_iterations", "nodes", "max_depth", "root_status",
            "root_bound", "root_lp", "root_seconds", "clique", "harvested", "formulation")
    return {"instance": Path(path).name, "n": inst.n, "m": inst.m, "method": method, "T": T, "best": rec["best"],
            "t_best": t_best, "bound": rec["bounds"][-1][1] if rec["bounds"] else None, "verified": ver,
            "late": rec["late"], "total_seconds": round(total, 1), "error": err,
            "incumbents": rec["inc"], "bounds": rec["bounds"], "stats": {k: res.get(k) for k in keep if k in res}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["mip", "naive", "eng"])
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for path in a.instances:
        for method in a.methods:
            r = probe(path, method, a.time_limit)
            rows.append(r)
            print("PROBE " + json.dumps(r, default=str), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
