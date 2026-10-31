"""Short Phoenix probes of the planning-and-scheduling (LBBD) methods (scale-up study; development only, not
calibration data).

Every (instance, method) pair runs in its own subprocess for T seconds on a clock started just before the instance is
loaded. Publications (complete schedules) are timestamped on receipt and their cost is recomputed exactly
(a_data.schedule_cost); publications at or after T are counted as late and ignored. The best one is checked with the
independent stdlib verifier.

    python scripts/phoenix/scaleup_lbbd_probe.py --time-limit 120 --methods mip eng naive cpsat ls \
        --instances outputs/scaleup/instances/lbbd/c32j4m1.json ...
(space-separated lists: sbatch --export splits on commas)
Methods: mip (time-indexed MIP, Gurobi defaults), mip_f1 (MIPFocus 1), mip_init (the shared initializer as MIP start),
naive (outer-loop LBBD, full-set nogoods, no relaxation), naive_relax (naive + the interval relaxation in the master),
eng (branch-and-check, relaxation, cores, repair), eng_nostrong (full-set nogoods), eng_norelax, eng_norepair,
cpsat (monolithic CP-SAT, one worker: decomposition-free exact control), ls (iterated local search with the exact
facility oracle: decomposition-free heuristic control).
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

REFS = ROOT / "outputs" / "scaleup" / "references" / "lbbd_references.json"


def methods():
    from src.bd.lbbd.b_mip_xl import PSTimeIndexedMIP
    from src.bd.lbbd.c_naive_bd import PSNaiveLBBD
    from src.bd.lbbd.d_ref_bd_xl import PSBendersXL
    from src.bd.lbbd.f_heuristic import PSCPSatMonolithic, PSLocalSearch
    mip = functools.partial(PSTimeIndexedMIP, mem_limit_gb=14.0)
    return {"mip": mip, "mip_f1": functools.partial(mip, mip_focus=1), "mip_init": functools.partial(mip, warm_start=True),
            "naive": PSNaiveLBBD, "naive_relax": functools.partial(PSNaiveLBBD, relaxation=True), "eng": PSBendersXL,
            "eng_nostrong": functools.partial(PSBendersXL, strengthen=False),
            "eng_norelax": functools.partial(PSBendersXL, relaxation=False),
            "eng_norepair": functools.partial(PSBendersXL, repair=False),
            "cpsat": PSCPSatMonolithic, "ls": PSLocalSearch}


def reference(name: str) -> float | None:
    if not REFS.exists():
        return None
    return json.loads(REFS.read_text()).get("instances", {}).get(name, {}).get("z_best")


def single(path: str, method: str, T: float, verify: bool) -> dict:
    t0 = time.perf_counter()
    from src.bd.lbbd.a_data import load_instance, schedule_cost
    from src.bd.lbbd.b_mip_xl import ps_output
    from src.bd.lbbd.e_verifier import verify_solution
    inst = load_instance(path)
    traj, bounds, late, bad = [], [], [0], [0]
    best = {"v": None, "sol": None}

    def on_inc(sol, stage=None):
        t = time.perf_counter() - t0
        if t >= T:
            late[0] += 1
            return
        v = schedule_cost(inst, sol)
        if v is None:
            bad[0] += 1
            return
        if best["v"] is None or v < best["v"] - 0.5:
            best["v"], best["sol"] = v, [list(x) for x in sol]
            traj.append((round(t, 2), v, stage))

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
    ver = None
    if verify and best["sol"] is not None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(ps_output(inst, best["sol"], method=method), fh)
        v = verify_solution(path, fh.name)
        ver = v["ok"] and v["objective_value"] == best["v"]
    ref = reference(inst.name)
    stats = {k: v for k, v in res.items() if k not in ("trace", "solution", "stats") and not isinstance(v, (list, dict))}
    if isinstance(res.get("oracle"), dict):
        stats["oracle"] = res["oracle"]
    bnd = bounds[-1][1] if bounds else None
    return {"instance": inst.name, "n": inst.n, "m": inst.m, "method": method, "T": T, "best": best["v"],
            "gap_to_ref_pct": round((best["v"] - ref) / ref * 100, 4) if ref and best["v"] is not None else None,
            "t_best": traj[-1][0] if traj else None, "t_first": traj[0][0] if traj else None, "bound": bnd,
            "bound_gap_pct": round((best["v"] - bnd) / best["v"] * 100, 4) if bnd is not None and best["v"] else None,
            "verified": ver, "infeasible_publications": bad[0], "late": late[0], "total_seconds": round(total, 1),
            "peak_rss_mb": round(peak_mb, 1), "error": err, "stats": stats, "incumbents": traj, "bounds": bounds[-6:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+")
    ap.add_argument("--methods", nargs="+", default=["mip", "eng", "naive", "cpsat", "ls"])
    ap.add_argument("--time-limit", type=float, default=120.0)
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
            lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
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
