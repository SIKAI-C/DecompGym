"""Short Phoenix probes of the BPPC methods (scale-up study; development only, not calibration data).

Runs every (instance, method) pair for T seconds in sequence, on a clock started just before the solver object is
built (instance loading counts). Publications are timestamped on receipt; those at or after T are counted as late and
ignored. The best packing is rechecked with the evaluator and with the independent verifier.

    python scripts/phoenix/scaleup_bppc_probe.py --time-limit 120 --methods mip naive eng \
        --instances outputs/scaleup/raw/bppc/Istanze/BPPC_8_4_1.txt outputs/scaleup/instances/bppc/da250_r5_k1.txt
(space-separated lists: sbatch --export splits on commas)
Methods: mip (compact MIP), mip_start (the MIP with the shared initializer as start; diagnostic), naive (naive CG),
eng (engineered branch-and-price), eng_nodive (without dives and restricted MIPs), ffd (the shared initializer only),
ls (heuristic-only control: iterated tabu search without decomposition, f_heuristic.BPPCHeuristicControl).

Frozen holdout rule (declared 2026-10-31, before any BPPC run): in every (class, n, density) cell, instances k = 9 and
k = 10 are the holdout (literature files BPPC_<c>_<d>_9/10; generated files <kind><n>_r<rho>_k9/k10); development
probes and Stage 1 panels use k in 1..8 only. Generated instances: python -m src.cg.bppc.a_data --out
outputs/scaleup/instances/bppc --raw outputs/scaleup/raw/bppc/Istanze <names>.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cg.bppc.a_data import load_bppc  # noqa: E402
from src.cg.bppc.b_mip_xl import BPPCCompactMIP, bin_count, bppc_output, public  # noqa: E402
from src.cg.bppc.c_naive_cg import BPPCNaiveCG  # noqa: E402
from src.cg.bppc.d_ref_cg_xl import BPPCBranchAndPrice  # noqa: E402
from src.cg.bppc.e_verifier import verify_solution  # noqa: E402
from src.cg.bppc.f_heuristic import BPPCHeuristicControl  # noqa: E402
from src.cg.bppc.heuristic import basic_initializer  # noqa: E402


class FFDControl:
    def __init__(self, *, time_limit, on_incumbent, **_):
        self.on_incumbent = on_incumbent

    def run(self, path):
        bins = public(basic_initializer(load_bppc(path)))
        self.on_incumbent(bins)
        return {"best_objective": len(bins), "global_lb": None, "termination_reason": "finished_early"}


METHODS = {"mip": (BPPCCompactMIP, {}), "mip_start": (BPPCCompactMIP, {"start": True}), "naive": (BPPCNaiveCG, {}),
           "eng": (BPPCBranchAndPrice, {}), "eng_nodive": (BPPCBranchAndPrice, {"dive": False, "restricted_mip": False}),
           "ffd": (FFDControl, {}), "ls": (BPPCHeuristicControl, {})}


def probe(path: str, method: str, T: float) -> dict:
    inst = load_bppc(path)
    rec = {"inc": [], "bounds": [], "late": 0, "best": None, "bins": None}
    t0 = time.perf_counter()

    def on_inc(bins, stage=None):
        t = time.perf_counter() - t0
        if t >= T:
            rec["late"] += 1
            return
        k = bin_count(inst, bins)
        if k is None:
            rec["inc"].append((round(t, 2), None, stage))
            return
        if rec["best"] is None or k < rec["best"]:
            rec["best"], rec["bins"] = k, [list(b) for b in bins]
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
    if rec["bins"] is not None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(bppc_output(inst, rec["bins"]), fh)
        ver = verify_solution(path, fh.name)["ok"]
    t_best = next((t for t, k, *_ in rec["inc"] if k == rec["best"]), None)
    keep = ("termination_reason", "gurobi_status", "budget_exhausted", "U", "lb0", "clique_K0", "cliques",
            "build_seconds", "num_vars", "num_constrs", "num_nz", "node_count", "cg_iterations", "cg_converged",
            "cg_stop_reason", "columns", "lp_value", "rmp_status", "pricing_calls", "pricing_seconds", "cg_iters",
            "exact_rounds", "heuristic_rounds", "incomplete_exact", "mispricings", "early_terminations",
            "exact_seconds", "lp_seconds", "mip_runs", "mip_seconds", "dives", "dive_seconds", "nodes", "max_depth",
            "stuck_nodes", "root_converged", "root_bound", "root_lp", "root_seconds", "pool_columns", "lns_iterations", "lns_accepted", "tabu_attempts", "tabu_successes", "best_stage",
            "tabu_iterations", "trivial_bound", "best_seconds", "repair_log_tail")
    return {"instance": Path(path).stem, "n": inst.n, "C": inst.capacity, "density": round(inst.density, 3),
            "wbound": inst.weight_bound, "method": method, "T": T, "best": rec["best"], "t_best": t_best,
            "bound": rec["bounds"][-1][1] if rec["bounds"] else None, "verified": ver, "late": rec["late"],
            "total_seconds": round(total, 1), "error": err, "incumbents": rec["inc"], "bounds": rec["bounds"],
            "stats": {k: res.get(k) for k in keep if k in res}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["mip", "naive", "eng"])
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scan", action="store_true", help="no solver: print n, density, trivial bounds and the initializer")
    a = ap.parse_args()
    if a.scan:
        from src.cg.bppc.heuristic import greedy_clique
        for path in a.instances:
            inst = load_bppc(path)
            q = len(greedy_clique(inst.adj, inst.weights))
            init = len(basic_initializer(inst))
            print("SCAN " + json.dumps({"instance": Path(path).stem, "n": inst.n, "density": round(inst.density, 3),
                                        "wbound": inst.weight_bound, "clique": q, "ffd": init,
                                        "gap_bins": init - max(q, inst.weight_bound)}), flush=True)
        return
    rows = []
    for path in a.instances:
        for method in a.methods:
            r = probe(path, method, a.time_limit)
            rows.append(r)
            print("PROBE " + json.dumps(r, default=str), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
