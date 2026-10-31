"""Short Phoenix probes of the SCFL methods (scale-up study; development only, not calibration data).

Every (instance, method) pair runs in its own subprocess (so the peak memory is the method's own) for T seconds, on a
clock started just before the instance is loaded. Publications (designs) are timestamped on receipt; those at or
after T are counted as late and ignored. After the solver returns, every published design is evaluated exactly
(deferred evaluation, not charged to the method) and the best one is also checked with the independent stdlib
verifier.

    python scripts/phoenix/scaleup_scfl_probe.py --time-limit 200 --methods mip eng naive heur \
        --instances outputs/scaleup/instances/scfl/cap111_s500_g1.json ...
(space-separated lists: sbatch --export splits on commas)
Methods: mip (extensive form, strong linking, Gurobi default root), mip_bar (barrier root), mip_weak (no strong
linking rows), mip_f1 (MIPFocus 1), naive (naive L-shaped), eng (engineered branch-and-Benders-cut), eng_nols (without
local search: the decomposition-only primal side), eng_nopd (without partial decomposition), eng_nolift (without
closed-facility lifting), heur (initializer + surrogate-screened local search + ILS, no decomposition), heur_saa
(sample-average EFs on growing scenario subsets of 10, 20, 40, ... scenarios, then the same local search), heur_ev
(the expected-value problem first, then the same SAA rounds).
"""
from __future__ import annotations

import argparse
import functools
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REFS = ROOT / "outputs" / "scaleup" / "references" / "scfl_references.json"


def methods():
    from src.bd.scfl.b_mip_xl import SCFLExtensiveMIP
    from src.bd.scfl.c_naive_bd import SCFLNaiveBenders
    from src.bd.scfl.d_ref_bd_xl import SCFLBendersXL
    from src.bd.scfl.f_heuristic import SCFLHeuristicControl
    mip = functools.partial(SCFLExtensiveMIP, mem_limit_gb=14.0)
    return {"mip": mip, "mip_bar": functools.partial(mip, method=2), "mip_weak": functools.partial(mip, strong=False),
            "mip_f1": functools.partial(mip, mip_focus=1), "naive": SCFLNaiveBenders, "eng": SCFLBendersXL,
            "eng_nols": functools.partial(SCFLBendersXL, local_search=False),
            "eng_nopd": functools.partial(SCFLBendersXL, partial=False),
            "eng_nolift": functools.partial(SCFLBendersXL, lifting=False),
            "heur": functools.partial(SCFLHeuristicControl, mode="ls"),
            "heur_saa": functools.partial(SCFLHeuristicControl, mode="saa"),
            "heur_ev": functools.partial(SCFLHeuristicControl, mode="saa", ev_first=True)}


def reference(name: str) -> float | None:
    if not REFS.exists():
        return None
    return json.loads(REFS.read_text()).get("instances", {}).get(name, {}).get("z_best")


def single(path: str, method: str, T: float, verify: bool) -> dict:
    t0 = time.perf_counter()
    from src.bd.scfl.a_data import load_instance
    from src.bd.scfl.b_mip_xl import DesignEvaluator, scfl_output
    from src.bd.scfl.e_verifier import verify_solution
    inst = load_instance(path)
    pubs, bounds, late = [], [], [0]

    def on_inc(design, stage=None):
        t = time.perf_counter() - t0
        if t >= T:
            late[0] += 1
            return
        pubs.append((t, list(design), stage))

    def on_bound(b):
        t = time.perf_counter() - t0
        if t < T and (not bounds or b > bounds[-1][1]):
            bounds.append((round(t, 2), b))

    try:
        res = methods()[method](time_limit=T - (time.perf_counter() - t0), threads=1, on_incumbent=on_inc,
                                on_bound=on_bound, soft_mem_gb=12.0).run(path)
        err = None
    except Exception as exc:  # keep the evidence
        res, err = {}, f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    t_eval = time.perf_counter()
    ev = DesignEvaluator(inst)
    traj, best, best_design, infeasible = [], None, None, 0
    for t, design, stage in pubs:
        v = ev.objective(design)
        if v is None:
            infeasible += 1
            continue
        if best is None or v < best - 1e-9 * max(1.0, abs(v)):
            best, best_design = v, design
            traj.append((round(t, 2), v, stage))
    ev.close()
    eval_seconds = time.perf_counter() - t_eval
    ver, ver_seconds = None, None
    if verify and best_design is not None:
        t_ver = time.perf_counter()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(scfl_output(inst, best_design, method=method), fh)
        v = verify_solution(path, fh.name)
        ver = v["ok"] and abs(v["objective_value"] - best) <= 1e-6 * max(1.0, abs(best))
        ver_seconds = round(time.perf_counter() - t_ver, 1)
    ref = reference(inst.name)
    stats = {k: v for k, v in res.items() if k not in ("trace", "design", "stats") and not isinstance(v, (list, dict))}
    return {"instance": inst.name, "I": inst.I, "J": inst.J, "S": inst.S, "method": method, "T": T, "best": best,
            "gap_to_ref_pct": round((best - ref) / ref * 100, 4) if ref and best is not None else None,
            "t_best": traj[-1][0] if traj else None, "t_first": traj[0][0] if traj else None,
            "bound": bounds[-1][1] if bounds else None,
            "bound_gap_pct": round((best - bounds[-1][1]) / best * 100, 4) if bounds and best else None,
            "verified": ver, "verify_seconds": ver_seconds, "publications": len(pubs), "infeasible_publications": infeasible,
            "late": late[0], "total_seconds": round(total, 1), "eval_seconds": round(eval_seconds, 1),
            "peak_rss_mb": round(peak_mb, 1), "error": err, "stats": stats, "design": best_design,
            "incumbents": traj, "bounds": bounds[-8:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+")
    ap.add_argument("--methods", nargs="+", default=["mip", "eng", "naive", "heur"])
    ap.add_argument("--time-limit", type=float, default=200.0)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--single", nargs=2, metavar=("INSTANCE", "METHOD"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.single:
        print("RESULT " + json.dumps(single(a.single[0], a.single[1], a.time_limit, not a.no_verify), default=str), flush=True)
        return
    rows = []
    for path in a.instances:
        for method in a.methods:
            cmd = [sys.executable, __file__, "--single", path, method, "--time-limit", str(a.time_limit)]
            if a.no_verify:
                cmd.append("--no-verify")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
            if lines:
                r = json.loads(lines[-1][7:])
            else:
                r = {"instance": Path(path).stem, "method": method, "error": f"exit {proc.returncode}: {proc.stderr[-2000:]}"}
            r["exit_code"] = proc.returncode
            rows.append(r)
            print("PROBE " + json.dumps(r, default=str), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
