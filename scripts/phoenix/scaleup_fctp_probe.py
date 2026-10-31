"""Short Phoenix probes of the FCTP methods (scale-up study; development only, not calibration data).

Runs every (instance, method) pair for T seconds in sequence, on a clock started just before the solver object is
built (instance loading counts). Publications are timestamped on receipt; those at or after T are counted as late
and ignored. Every published plan is rechecked with flow_cost; the best one also with the independent verifier.

    python scripts/phoenix/scaleup_fctp_probe.py --time-limit 200 --methods mip eng heur naive \
        --instances outputs/scaleup/raw/fctp/glover/N3709.FCTP ...
(space-separated lists: sbatch --export splits on commas)
Methods: mip (compact MIP, Gurobi defaults), mip_f1 (MIPFocus 1), naive (naive Benders), eng (engineered Benders),
eng_nols (engineered without local search: the decomposition-only primal side), eng_nomw (without Magnanti-Wong cuts),
eng_v1 (without cut-set rows), eng_ils (engineered with 5 s ILS polishing: does the decomposition add to the heuristic?),
heur (initializer + slope scaling + basis ILS, no decomposition: the heuristic control).
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bd.fctp.a_data_xl import load_fctp  # noqa: E402
from src.bd.fctp.b_mip_xl import FCTPCompactMIP, fctp_output, flow_cost  # noqa: E402
from src.bd.fctp.c_naive_bd import FCTPNaiveBenders  # noqa: E402
from src.bd.fctp.d_ref_bd_xl import FCTPBendersXL  # noqa: E402
from src.bd.fctp.e_verifier_xl import verify_solution  # noqa: E402
from src.bd.fctp.f_heuristic_xl import FCTPHeuristicControl  # noqa: E402

METHODS = {"mip": FCTPCompactMIP, "mip_f1": functools.partial(FCTPCompactMIP, mip_focus=1), "naive": FCTPNaiveBenders,
           "eng": FCTPBendersXL, "eng_nols": functools.partial(FCTPBendersXL, local_search=False),
           "eng_nomw": functools.partial(FCTPBendersXL, pareto=False), "eng_v1": functools.partial(FCTPBendersXL, cutsets=False),
           "eng_ils": functools.partial(FCTPBendersXL, ils_seconds=5.0), "heur": FCTPHeuristicControl}
REFS = ROOT / "outputs" / "scaleup" / "references" / "fctp_references.json"


def reference(name: str) -> float | None:
    if not REFS.exists():
        return None
    refs = json.loads(REFS.read_text()).get("instances", {})
    return refs.get(name, {}).get("best_known_literature")


def probe(path: str, method: str, T: float) -> dict:
    t0 = time.perf_counter()
    inst = load_fctp(path)
    rec = {"inc": [], "bounds": [], "late": 0, "best": None, "flows": None, "invalid": 0}

    def on_inc(flows, stage=None):
        t = time.perf_counter() - t0
        if t >= T:
            rec["late"] += 1
            return
        cost = flow_cost(inst, flows)
        if cost is None:
            rec["invalid"] += 1
            return
        if rec["best"] is None or cost < rec["best"] - 1e-9:
            rec["best"], rec["flows"] = cost, flows
            rec["inc"].append((round(t, 2), cost, stage))

    def on_bound(b):
        t = time.perf_counter() - t0
        if t < T and (not rec["bounds"] or b > rec["bounds"][-1][1]):
            rec["bounds"].append((round(t, 2), b))

    try:
        res = METHODS[method](time_limit=T - (time.perf_counter() - t0), threads=1, on_incumbent=on_inc, on_bound=on_bound,
                              soft_mem_gb=12.0).run(path)
        err = None
    except Exception as exc:  # keep the evidence
        res, err = {}, f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - t0
    ver = None
    if rec["flows"] is not None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(fctp_output(inst, rec["flows"]), fh)
        ver = verify_solution(path, fh.name)["ok"]
    ref = reference(inst.name)
    gap = (rec["best"] - ref) / ref * 100 if ref and rec["best"] is not None else None
    t_best = rec["inc"][-1][0] if rec["inc"] else None
    stats = {k: v for k, v in res.items() if k not in ("trace", "flows", "stats") and not isinstance(v, (list, dict))}
    return {"instance": inst.name, "m": inst.m, "n": inst.n, "method": method, "T": T, "best": rec["best"],
            "gap_to_lit_pct": round(gap, 3) if gap is not None else None, "t_best": t_best,
            "bound": rec["bounds"][-1][1] if rec["bounds"] else None, "verified": ver, "late": rec["late"],
            "invalid": rec["invalid"], "total_seconds": round(total, 1), "error": err, "stats": stats,
            "incumbents": rec["inc"], "bounds": rec["bounds"][-5:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["mip", "eng", "heur", "naive"])
    ap.add_argument("--time-limit", type=float, default=200.0)
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
