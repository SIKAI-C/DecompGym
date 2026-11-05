"""Phoenix probes of the scale-up TSSO methods (development only, not calibration data).

Three modes, each one Slurm task:
  --single INSTANCE METHOD   run one method for T seconds on a clock started just before the instance is loaded;
                             publications (designs) and bounds are timestamped on receipt, those at or after T are
                             late and ignored; afterwards (outside the clock) every published design is evaluated
                             exactly, the best one is checked by the stdlib verifier, and the method's cut log (if
                             any) is saved for certification. Writes run_<instance>_<method>.json (+ cuts_*.npz).
  --cert INSTANCE            after the runs: the exact LP* interval [lo, hi] (hi = exact value of an exactly feasible
                             LP point at the engineered method's final LP solution and at the LP controls' solutions;
                             lo = the best certified bound), then for every method with a cut log the certified
                             LP-master bound at T and the first certified time within eps of LP* (target
                             (1 - eps) * hi, conservative), searched over the method's own bound-publication times;
                             plus the trivial certificate controls: no cuts, and one round of cuts at the shared
                             initializer's design. Writes cert_<instance>.json.
  --summary                  one table over the run and cert files of --outdir.
Methods: mip, lp_bar (barrier), lp_dual (dual simplex), naive_lp (bound-side naive: LP master, one aggregated cut),
naive_int (integer-master naive, diagnostic), eng (engineered), eng_nostab (engineered without in-out
stabilization), eng_nopool (engineered without the dual pool), ls (per-product local search), saa (SAA extensive
forms + local search).
"""
from __future__ import annotations

import argparse
import functools
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CUT_METHODS = ("lp_bar", "lp_dual", "naive_lp", "naive_int", "eng", "eng_nostab", "eng_nopool")


def methods():
    from src.bd.tsso.b_mip_xl import TSSOExtensiveMIP, TSSOLPControl
    from src.bd.tsso.c_naive_bd import TSSONaiveBenders, TSSONaiveLPBenders
    from src.bd.tsso.d_ref_bd_xl import TSSOBendersXL
    from src.bd.tsso.f_heuristic_xl import TSSOHeuristicControl
    return {"mip": TSSOExtensiveMIP, "lp_bar": functools.partial(TSSOLPControl, method=2),
            "lp_dual": functools.partial(TSSOLPControl, method=1), "naive_lp": TSSONaiveLPBenders,
            "lp_bar_split": functools.partial(TSSOLPControl, method=2, split=True),
            "naive_int": TSSONaiveBenders, "eng": TSSOBendersXL, "eng_nostab": functools.partial(TSSOBendersXL, stabilize=False),
            "eng_nopool": functools.partial(TSSOBendersXL, pool=False),
            "ls": functools.partial(TSSOHeuristicControl, mode="ls"), "saa": functools.partial(TSSOHeuristicControl, mode="saa")}


def single(path: str, method: str, T: float, outdir: Path, opts: dict | None = None, label: str | None = None) -> dict:
    t0 = time.perf_counter()
    import tempfile
    from src.bd.tsso.a_data_xl import load_instance
    from src.bd.tsso.b_mip_xl import DesignEvaluator, tsso_output
    from src.bd.tsso.e_verifier_xl import verify_solution
    inst = load_instance(path)
    pubs, bounds, late = [], [], [0]

    def on_inc(y, stage=None):
        t = time.perf_counter() - t0
        if t >= T:
            late[0] += 1
            return
        pubs.append((t, np.asarray(y, dtype=bool).copy(), stage))

    def on_bound(b):
        t = time.perf_counter() - t0
        if t < T and (not bounds or b > bounds[-1][1]):
            bounds.append((round(t, 3), float(b)))

    kw = {"soft_mem_gb": 13.0, **(opts or {})}
    solver = methods()[method](time_limit=T - (time.perf_counter() - t0), threads=1, on_incumbent=on_inc, on_bound=on_bound, **kw)
    base_method, method = method, label or method
    try:
        res = solver.run(path)
        err = None
    except Exception as exc:  # keep the evidence
        import traceback
        res, err = {}, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}"
    total = time.perf_counter() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    name = inst.name
    cut_file = None
    log = getattr(solver, "cutlog", None)
    # cut times are on the solver's clock, which starts after ours: shift them to our clock
    offset = max(0.0, getattr(solver, "t0", t0) - t0)
    if log is not None and log.n_cuts:
        cut_file = outdir / f"cuts_{name}_{method}.npz"
        log._time = [x + offset for x in log._time]
        log.save(cut_file)
        res["cut_terms"], res["cut_pool_rows"] = log.terms, log.pool_rows
    t_eval = time.perf_counter()
    ev = DesignEvaluator(inst)
    traj, best, best_y = [], None, None
    for t, y, stage in pubs:
        v = ev.value(y)
        if best is None or v < best - 1e-9 * max(1.0, abs(v)):
            best, best_y = v, y
            traj.append((round(t, 2), v, stage))
    ev.close()
    eval_seconds = time.perf_counter() - t_eval
    ver = None
    if best_y is not None:
        t_ver = time.perf_counter()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(tsso_output(inst, best_y, method=method, value=best), fh)
        v = verify_solution(path, fh.name)
        ver = {"ok": v["ok"], "seconds": round(time.perf_counter() - t_ver, 1), "issues": v["issues"][:3]}
    stats = {k: v for k, v in res.items() if k not in ("trace", "design", "y_lp", "int_trace", "simplex_bounds",
                                                        "barrier_trace") and not isinstance(v, (list, dict))}
    out = {"instance": name, "path": path, "I": inst.I, "T": inst.T, "S": inst.S, "method": method, "base_method": base_method,
           "options": opts or {}, "time_limit": T,
           "best": best, "t_best": traj[-1][0] if traj else None, "t_first": traj[0][0] if traj else None,
           "bound": bounds[-1][1] if bounds else None, "t_bound": bounds[-1][0] if bounds else None,
           "verified": ver, "clock_offset": offset, "publications": len(pubs), "late": late[0], "total_seconds": round(total, 1),
           "eval_seconds": round(eval_seconds, 1), "peak_rss_mb": round(peak_mb, 1), "error": err, "stats": stats,
           "incumbents": traj, "bounds": bounds, "cut_file": str(cut_file) if cut_file else None,
           "y_lp": res.get("y_lp"), "int_trace": res.get("int_trace"), "simplex_bounds": res.get("simplex_bounds"),
           "barrier_trace": (res.get("barrier_trace") or [])[-5:], "trace_tail": (res.get("trace") or [])[-5:]}
    (outdir / f"run_{name}_{method}.json").write_text(json.dumps(out, default=str))
    return out


def cert(path: str, outdir: Path, eps: float, budget: float, only: list[str] | None = None) -> dict:
    from src.bd.tsso.a_data_xl import initial_design, load_instance
    from src.bd.tsso.b_mip_xl import DesignEvaluator, make_env
    from src.bd.tsso.c_naive_bd import evaluate_point
    from src.bd.tsso.g_certificate_xl import CutLog, certify, cuts_upto, empty_cuts, load_cuts, lp_upper
    inst = load_instance(path)
    name = inst.name
    runs = {}
    for f in sorted(outdir.glob(f"run_{name}_*.json")):
        r = json.loads(f.read_text())
        runs[r["method"]] = r
    env = make_env()
    out: dict = {"instance": name, "eps": eps, "methods": {}}
    t_all = time.perf_counter()
    # LP* upper end: exactly feasible LP points (computed once per instance by --lpstar when available)
    star = outdir / f"lpstar_{name}.json"
    if star.exists():
        his = json.loads(star.read_text())["lp_upper"]
    else:
        his = {}
        for m in ("eng", "lp_bar", "lp_dual", "lp_bar_split"):
            y = runs.get(m, {}).get("y_lp")
            if y is not None:
                r = lp_upper(inst, np.asarray(y, dtype=float), env)
                his[m] = {"ub": r["ub"], "seconds": round(r["seconds"], 1)}
    out["lp_upper"] = his
    hi = min((v["ub"] for v in his.values()), default=None)
    target = (1.0 - eps) * hi if hi is not None else None
    out["lp_star_hi"] = hi
    out["target"] = target

    def cert_at(cuts, tau):
        c = cuts_upto(cuts, tau)
        r = certify(inst, c, time_limit=budget, env=env)
        return {"tau": round(float(tau), 3), "lb": r["lb"], "lp_value": r.get("lp_value"), "cuts": r["cuts_used"],
                "rejected": r["rejected_cuts"], "max_dual_inf": r.get("max_dual_infeasibility"),
                "master_not_optimal": r["master_not_optimal"], "seconds": round(r["seconds"], 1), "error": r.get("error")}

    los = []
    for m, run in runs.items():
        if only is not None and m not in only:
            continue
        if not run.get("cut_file") or not Path(run["cut_file"]).exists():
            continue
        cuts = load_cuts(run["cut_file"])
        if "clock_offset" not in run:  # older runs: conservative offset = our time of the solver's first (trivial) bound
            b0 = (run.get("bounds") or [[0.0, 1.0]])[0]
            cuts["time"] = cuts["time"] + (b0[0] if b0[1] == 0.0 else 0.0)
        T = run["time_limit"]
        rec = {"final": cert_at(cuts, T), "checks": []}
        los.append(rec["final"]["lb"])
        if target is not None and rec["final"]["lb"] >= target:
            # first certified time within eps: binary search over the method's bound-publication times
            times = sorted({float(t) + 1e-3 for t, _ in run.get("bounds", [])} | {T})
            lo_i, hi_i = 0, len(times) - 1
            while lo_i < hi_i:
                mid = (lo_i + hi_i) // 2
                c = cert_at(cuts, times[mid])
                rec["checks"].append(c)
                if c["lb"] >= target:
                    hi_i = mid
                else:
                    lo_i = mid + 1
            rec["time_to_eps"] = round(times[lo_i], 3)
        else:
            rec["time_to_eps"] = None
        rec["gap_to_hi_pct"] = round(100 * (hi - rec["final"]["lb"]) / hi, 5) if hi else None
        out["methods"][m] = rec
        print(json.dumps({m: rec}, default=str), flush=True)
    if only is not None and "controls" not in only:
        out["lp_star_lo"] = max(los) if los else None
        out["seconds"] = round(time.perf_counter() - t_all, 1)
        env.dispose()
        (outdir / f"cert_{name}_{'_'.join(only)}.json").write_text(json.dumps(out, default=str))
        return out
    # trivial certificate controls
    rec = certify(inst, empty_cuts(inst.T), env=env)
    out["methods"]["empty_certificate"] = {"final": {"lb": rec["lb"]}, "gap_to_hi_pct": round(100 * (hi - rec["lb"]) / hi, 5) if hi else None}
    t1 = time.perf_counter()
    ev = DesignEvaluator(inst, env)
    y0 = initial_design(inst).astype(float)
    res = evaluate_point(inst, ev, y0, time.perf_counter() + 1e9)
    ev.close()
    log = CutLog(inst)
    sub = (np.arange(inst.I)[:, None] * inst.S + np.arange(inst.S)[None, :]).ravel()
    log.add_multi(sub, res["pi"].reshape(-1, inst.T), res["mu"].reshape(-1, inst.T), 0.0)
    r = certify(inst, log.arrays(), time_limit=budget, env=env)
    out["methods"]["oneshot_initializer"] = {"final": {"lb": r["lb"], "seconds": round(time.perf_counter() - t1, 1)},
                                            "gap_to_hi_pct": round(100 * (hi - r["lb"]) / hi, 5) if hi else None}
    los.append(r["lb"])
    lo = max(los) if los else None
    out["lp_star_lo"] = lo
    out["lp_star_width_rel"] = (hi - lo) / hi if hi and lo is not None else None
    out["seconds"] = round(time.perf_counter() - t_all, 1)
    env.dispose()
    tag = "" if only is None else "_" + "_".join(only)
    (outdir / f"cert_{name}{tag}.json").write_text(json.dumps(out, default=str))
    return out


def lpstar(path: str, outdir: Path) -> dict:
    """The exact upper end of LP* from every run that kept an LP solution (the smallest is used)."""
    from src.bd.tsso.a_data_xl import load_instance
    from src.bd.tsso.b_mip_xl import make_env
    from src.bd.tsso.g_certificate_xl import lp_upper
    inst = load_instance(path)
    env = make_env()
    his = {}
    for f in sorted(outdir.glob(f"run_{inst.name}_*.json")):
        run = json.loads(f.read_text())
        converged = run.get("stats", {}).get("lp_seconds") is not None
        if run.get("y_lp") is not None and converged:
            r = lp_upper(inst, np.asarray(run["y_lp"], dtype=float), env)
            his[run["method"]] = {"ub": r["ub"], "seconds": round(r["seconds"], 1)}
    env.dispose()
    out = {"instance": inst.name, "lp_upper": his, "hi": min((v["ub"] for v in his.values()), default=None)}
    (outdir / f"lpstar_{inst.name}.json").write_text(json.dumps(out))
    return out


def summary(outdir: Path) -> None:
    rows = {}
    for f in sorted(outdir.glob("run_*.json")):
        r = json.loads(f.read_text())
        rows.setdefault(r["instance"], {})[r["method"]] = r
    for inst, ms in rows.items():
        ce = {"methods": {}}
        for c in sorted(outdir.glob(f"cert_{inst}*.json")):
            part = json.loads(c.read_text())
            ce["methods"].update(part.get("methods", {}))
            for k in ("lp_star_hi", "target"):
                if part.get(k) is not None:
                    ce[k] = part[k] if ce.get(k) is None else min(ce[k], part[k])
            if part.get("lp_star_lo") is not None:
                ce["lp_star_lo"] = max(ce.get("lp_star_lo") or -1e300, part["lp_star_lo"])
        prim = [r["best"] for r in ms.values() if r.get("best") is not None]
        zb = min(prim) if prim else None
        hi = ce.get("lp_star_hi")
        print(f"\n== {inst}  z_best={zb}  LP*=[{ce.get('lp_star_lo')}, {hi}]  target={ce.get('target')}")
        print(f"{'method':12s} {'primal':>14s} {'gap%':>8s} {'t_best':>7s} {'rep.bound':>14s} {'t_bnd':>7s} "
              f"{'cert@T':>14s} {'certgap%':>9s} {'t_eps':>7s} {'t_eps_rep':>9s} {'int_bound':>12s} {'ver':>4s} {'rss':>6s}")
        tgt = ce.get("target")
        order = ["mip", "lp_bar", "lp_dual", "naive_lp", "naive_int", "eng", "eng_nostab", "eng_nopool", "ls", "saa"]
        for m in order + sorted(set(ms) - set(order)):
            r = ms.get(m)
            if r is None:
                continue
            cm = ce["methods"].get(m, {})
            cb = cm.get("final", {}).get("lb")
            gap = 100 * (r["best"] - zb) / zb if r.get("best") is not None and zb else None
            print(f"{m:12s} {r.get('best') or float('nan'):14.3f} {gap if gap is not None else float('nan'):8.4f} "
                  f"{r.get('t_best') or float('nan'):7.1f} {r.get('bound') or float('nan'):14.3f} "
                  f"{r.get('t_bound') or float('nan'):7.1f} {cb if cb is not None else float('nan'):14.3f} "
                  f"{cm.get('gap_to_hi_pct') if cm.get('gap_to_hi_pct') is not None else float('nan'):9.4f} "
                  f"{cm.get('time_to_eps') if cm.get('time_to_eps') is not None else float('nan'):7.1f} "
                  f"{next((t for t, b in r.get('bounds') or [] if tgt is not None and b >= tgt), float('nan')):9.1f} "
                  f"{(r.get('stats') or {}).get('int_bound') or float('nan'):12.3f} "
                  f"{str((r.get('verified') or {}).get('ok', '-'))[:4]:>4s} {r.get('peak_rss_mb') or 0:6.0f}"
                  + (f"  ERR {r['error'][:80]}" if r.get("error") else "")
                  + f"  [{(r.get('stats') or {}).get('termination_reason')}]")
        for m in ("empty_certificate", "oneshot_initializer"):
            cm = ce["methods"].get(m)
            if cm:
                print(f"{m:20s} cert={cm['final']['lb']:.3f} gap%={cm.get('gap_to_hi_pct')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", nargs=2, metavar=("INSTANCE", "METHOD"))
    ap.add_argument("--cert", metavar="INSTANCE")
    ap.add_argument("--cert-methods", nargs="+", default=None, help="methods to certify ('controls' = trivial controls)")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--lpstar", metavar="INSTANCE")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--cert-budget", type=float, default=300.0)
    ap.add_argument("--outdir", default="outputs/scaleup/probes/tsso")
    ap.add_argument("--opt", nargs="+", default=[], help="solver constructor overrides key=value (Python literals)")
    ap.add_argument("--label", default=None, help="method name used in the output files (default: METHOD)")
    a = ap.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if a.single:
        import ast
        opts = {k: ast.literal_eval(v) for k, v in (o.split("=", 1) for o in a.opt)}
        r = single(a.single[0], a.single[1], a.time_limit, outdir, opts, a.label)
        print("RESULT " + json.dumps({k: v for k, v in r.items() if k not in ("incumbents", "bounds", "y_lp")}, default=str))
    elif a.cert:
        r = cert(a.cert, outdir, a.eps, a.cert_budget, a.cert_methods)
        print("CERT " + json.dumps({k: v for k, v in r.items()}, default=str))
    elif a.lpstar:
        print("LPSTAR " + json.dumps(lpstar(a.lpstar, outdir)))
    elif a.summary:
        summary(outdir)


if __name__ == "__main__":
    main()
