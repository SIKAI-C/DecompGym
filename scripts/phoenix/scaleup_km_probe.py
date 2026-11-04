"""Phoenix probes of the k-median (XL) methods (scale-up study; development only, not calibration data).

SUPERSEDED for Stage 1 by scripts/phoenix/scaleup_km_stage1.py (supervised arms, hard kill, canonical checker at
frozen checkpoints; Codex review 46). Kept for the 11-03 probe record; its retrospective grid times are development
diagnostics only, and its 'bound at T' is the maximum over history (v4 uses the last complete version).

Three modes (each a separate Slurm array task; no optimization on the laptop):
  --run INSTANCE METHOD    one method for T seconds on a clock started before the instance is loaded; publications
                           (designs) and solver-reported bounds are timestamped on receipt, those at or after T are
                           counted as late and ignored. After the solver returns (not charged): exact evaluation of
                           every published design, stdlib verification of the best, and the certificate material
                           (cut logs with timestamps, Lagrangian snapshots, LP duals, LP points) saved as .npz.
  --certify INSTANCE WHAT  exact certified LP-master bounds (g_certificate_xl) of one method's certificate on a time
                           grid (cuts generated before t; T * 2^(-g/2), g = 0..16, plus the method's own key times),
                           or WHAT=controls: the trivial certificates (empty; one-shot cuts at the initializer's
                           design and at the local-search control's final design).
  --summarize INSTANCE...  the rigorous LP* interval [best certified bound, exact master value of the best LP point]
                           and, per method, the first grid time at which its certified bound reaches the conservative
                           target hi - eps * lo (eps = 1e-4), the certified bound at T, primal values and
                           solver-reported (integer-side) bounds as diagnostics.
Methods: mip (radius formulation), mip_classical, lp_bar (compact LP, barrier, crossover off), lp_dual (compact LP,
dual simplex), lagr (subgradient Lagrangian, diagnostic), ls (swap local search + ILS, primal control), naive_lp
(naive LP-master Benders, one aggregated cut per iteration: the bound-side naive control), naive_int (integer-master
naive Benders, diagnostic), eng (engineered Benders), eng_nostab (engineered without in-out stabilization).
"""
from __future__ import annotations

import argparse
import functools
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REFS = ROOT / "outputs" / "scaleup" / "references" / "km_references.json"
EPS = 1e-4


def methods():
    from src.bd.km.b_mip_xl import KMCompactLP, KMCompactMIP
    from src.bd.km.c_naive_bd_xl import KMNaiveBenders, KMNaiveLPBenders
    from src.bd.km.d_ref_bd_xl import KMBendersXL
    from src.bd.km.f_heuristic_xl import KMLagrangian, KMLocalSearch
    mip = functools.partial(KMCompactMIP, mem_limit_gb=14.0)
    lp = functools.partial(KMCompactLP, mem_limit_gb=14.0)
    return {"mip": functools.partial(mip, formulation="radius"), "mip_classical": functools.partial(mip, formulation="classical"),
            "lp_bar": functools.partial(lp, method=2), "lp_dual": functools.partial(lp, method=1),
            "lagr": KMLagrangian, "ls": KMLocalSearch, "naive_lp": KMNaiveLPBenders, "naive_int": KMNaiveBenders,
            "eng": KMBendersXL, "eng_nostab": functools.partial(KMBendersXL, stabilize=False)}


def reference(name: str) -> dict:
    if not REFS.exists():
        return {}
    return json.loads(REFS.read_text()).get("instances", {}).get(name, {})


def stem(path: str) -> str:
    return Path(path).stem


# ---------------------------------------------------------------- run

def run_one(path: str, method: str, T: float, out: Path) -> dict:
    t0 = time.perf_counter()
    pubs, bounds, late = [], [], {"incumbents": 0, "bounds": 0}

    def on_inc(design, stage=None):
        t = time.perf_counter() - t0
        if t >= T:
            late["incumbents"] += 1
            return
        pubs.append((t, list(design), stage))

    def on_bound(b):
        t = time.perf_counter() - t0
        if t >= T:
            late["bounds"] += 1
            return
        if not bounds or b > bounds[-1][1]:
            bounds.append((round(t, 3), float(b)))

    offset = time.perf_counter() - t0
    try:
        res = methods()[method](time_limit=T - offset, threads=1, on_incumbent=on_inc, on_bound=on_bound,
                                soft_mem_gb=12.0).run(path)
        err = None
    except Exception as exc:  # keep the evidence
        import traceback
        res, err = {}, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}"
    total = time.perf_counter() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    from src.bd.km.a_data_xl import load_instance
    from src.bd.km.b_mip_xl import DesignEvaluator, km_output
    from src.bd.km.e_verifier_xl import verify_solution
    inst = load_instance(path)
    ev = DesignEvaluator(inst)
    traj, best, best_design, invalid = [], None, None, 0
    for t, design, stage in pubs:
        v = ev.objective(design)
        if v is None:
            invalid += 1
            continue
        if best is None or v < best - 1e-9 * max(1.0, abs(v)):
            best, best_design = v, design
            traj.append((round(t, 3), v, stage))
    ver = None
    if best_design is not None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(km_output(inst, best_design, ev.D, method=method), fh)
        vr = verify_solution(path, fh.name)
        ver = bool(vr["ok"] and abs(vr["objective_value"] - best) <= 1e-6 * max(1.0, abs(best)))
    # certificate material (times on the probe clock; only material generated before T)
    arrays: dict[str, np.ndarray] = {"offset": np.array([offset])}
    log = res.get("cut_log")
    if log is not None and log.get("kind") == "single":
        tt = np.asarray(log["t"], dtype=float) + offset
        keep = tt < T
        arrays.update(cut_t=tt[keep], cut_j=np.asarray(log["j"], dtype=np.int32)[keep], cut_v=np.asarray(log["v"], dtype=float)[keep])
    elif log is not None and log.get("kind") == "aggregated":
        tt = np.asarray(log["t"], dtype=float) + offset
        keep = tt < T
        V = np.asarray(log["v"], dtype=float).reshape(len(tt), -1) if len(tt) else np.zeros((0, inst.n))
        arrays.update(agg_t=tt[keep], agg_v=V[keep])
    if res.get("u_snapshots"):
        st = np.array([s["t"] for s in res["u_snapshots"]]) + offset
        keep = st < T
        arrays.update(snap_t=st[keep], snap_v=np.array([s["v"] for s in res["u_snapshots"]])[keep],
                      snap_L=np.array([s["L"] for s in res["u_snapshots"]])[keep])
    if res.get("certificate_v") is not None and res.get("certificate_seconds") is not None:
        tc = res["certificate_seconds"] + offset
        if tc < T:
            arrays.update(dual_t=np.array([tc]), dual_v=np.asarray(res["certificate_v"], dtype=float))
    for key in ("final_lp_point", "lp_point"):
        if res.get(key) is not None:
            arrays["lp_point"] = np.asarray(res[key], dtype=float)
    np.savez_compressed(out / f"{stem(path)}__{method}.npz", **arrays)
    skip = {"trace", "cut_log", "stats", "u_snapshots", "certificate_v", "final_lp_point", "lp_point", "design"}
    stats = {k: v for k, v in res.items() if k not in skip and not isinstance(v, (list, dict, np.ndarray))}
    if "phase1_end" in res:
        stats["phase1_end_probe"] = res["phase1_end"] + offset
    row = {"instance": stem(path), "m": inst.m, "n": inst.n, "k": inst.k, "method": method, "T": T, "best": best,
           "t_first": traj[0][0] if traj else None, "t_best": traj[-1][0] if traj else None, "verified": ver,
           "solver_bound": bounds[-1][1] if bounds else None, "solver_bound_t": bounds[-1][0] if bounds else None,
           "publications": len(pubs), "invalid_publications": invalid, "late": late, "offset": offset,
           "total_seconds": round(total, 2), "peak_rss_mb": round(peak_mb, 1), "error": err, "stats": stats,
           "design": best_design, "incumbents": traj[-60:], "bounds": bounds[-60:],
           "simplex_bounds": res.get("simplex_bounds", [])[-60:] if isinstance(res.get("simplex_bounds"), list) else None,
           "bound_trace": res.get("bound_trace")}
    (out / f"{stem(path)}__{method}.json").write_text(json.dumps(row, default=str))
    return row


# ---------------------------------------------------------------- certify

def grid_times(T: float, extra: list[float]) -> list[float]:
    g = [T * 2 ** (-i / 2) for i in range(17)] + [0.5] + [t for t in extra if 0 < t <= T]
    return sorted(set(round(t, 4) for t in g))


def certify_one(path: str, what: str, T: float, out: Path) -> dict:
    from src.bd.km.a_data_xl import load_instance
    from src.bd.km.c_naive_bd_xl import greedy_design
    from src.bd.km.g_certificate_xl import Certificate, Checker, aggregated, lp_interval, one_shot, singles
    from src.bd.scfl.b_mip_xl import make_env
    inst = load_instance(path)
    t_setup = time.perf_counter()
    chk = Checker(inst)
    env = make_env(1, 12.0)
    rows = []
    result = {"instance": stem(path), "what": what, "T": T, "points": rows}
    if what == "controls":
        D = chk.D
        g = np.nonzero(greedy_design(D, inst.k))[0]
        cands = [("empty", Certificate()), ("one_shot_initializer", one_shot(D, g))]
        ls = out / f"{stem(path)}__ls.json"
        if ls.exists() and json.loads(ls.read_text()).get("design"):
            cands.append(("one_shot_ls_final", one_shot(D, json.loads(ls.read_text())["design"])))
        for name, cert in cands:
            r = chk.certify(cert, env=env)
            rows.append({"control": name, "bound": r["bound"], "lp_status": r["lp_status"], "lp_seconds": r["lp_seconds"]})
    else:
        f = out / f"{stem(path)}__{what}.npz"
        if not f.exists():
            result["error"] = "no run output"
        else:
            z = np.load(f)
            row_json = json.loads((out / f"{stem(path)}__{what}.json").read_text())
            # Exact master value at the method's own LP point (>= LP*): once a certified bound reaches it, later grid
            # points (more cuts) can gain at most the remaining difference, so they are copied, not re-solved.
            own_hi = lp_interval(chk, [], [("own LP point", z["lp_point"])])["hi_float"] if "lp_point" in z else None
            result["own_lp_point_value"] = own_hi
            capped = (lambda b: own_hi is not None and b is not None and b >= own_hi - 1e-9 * max(1.0, abs(own_hi)))
            budget_end = time.perf_counter() + 600.0
            extra = []
            if "phase1_end_probe" in row_json.get("stats", {}):
                extra.append(row_json["stats"]["phase1_end_probe"])
            if "cut_t" in z:
                ct, cj, cv = z["cut_t"], z["cut_j"], z["cut_v"]
                o = np.argsort(ct, kind="stable")
                ct, cj, cv = ct[o], cj[o], cv[o]
                last = -1
                for t in grid_times(T, extra + [float(ct.max()) + 1e-3] if len(ct) else extra):
                    c = int(np.searchsorted(ct, t, side="left"))
                    if c == last or (rows and capped(rows[-1]["bound"])):
                        rows.append(dict(rows[-1], t=t, copied=True))
                        continue
                    if time.perf_counter() > budget_end:
                        result["budget_exhausted_at"] = t
                        break
                    last = c
                    r = chk.certify(singles(cj[:c], cv[:c]), env=env)
                    rows.append({"t": t, "cuts": c, "bound": r["bound"], "lp_value": r["lp_value"], "lp_status": r["lp_status"],
                                 "lp_seconds": r["lp_seconds"], "exact_seconds": r["exact_seconds"], "rejected": len(r["rejected"])})
            elif "agg_t" in z:
                at, V = z["agg_t"], z["agg_v"]
                last = -1
                for t in grid_times(T, extra + [float(at.max()) + 1e-3] if len(at) else extra):
                    c = int(np.searchsorted(at, t, side="left"))
                    if c == last or (rows and capped(rows[-1]["bound"])):
                        rows.append(dict(rows[-1], t=t, copied=True))
                        continue
                    if time.perf_counter() > budget_end:
                        result["budget_exhausted_at"] = t
                        break
                    last = c
                    r = chk.certify(aggregated(list(V[:c]), inst.n), env=env)
                    rows.append({"t": t, "cuts": c, "bound": r["bound"], "lp_value": r["lp_value"], "lp_status": r["lp_status"],
                                 "lp_seconds": r["lp_seconds"], "exact_seconds": r["exact_seconds"]})
            elif "snap_t" in z:
                for t, v, L in zip(z["snap_t"], z["snap_v"], z["snap_L"]):
                    r = chk.certify(chk.dual_vector_certificate(v), env=env)
                    rows.append({"t": float(t), "bound": r["bound"], "lagrangian_value": float(L), "lp_status": r["lp_status"],
                                 "lp_seconds": r["lp_seconds"]})
            elif "dual_t" in z:
                r = chk.certify(chk.dual_vector_certificate(z["dual_v"]), env=env)
                rows.append({"t": float(z["dual_t"][0]), "bound": r["bound"], "lp_status": r["lp_status"], "lp_seconds": r["lp_seconds"]})
    env.dispose()
    result["seconds"] = round(time.perf_counter() - t_setup, 1)
    (out / f"{stem(path)}__cert_{what}.json").write_text(json.dumps(result, default=str))
    return result


# ---------------------------------------------------------------- summarize

CERT_METHODS = ["eng", "eng_nostab", "naive_lp", "naive_int", "lp_bar", "lp_dual", "lagr"]


def summarize(paths: list[str], out: Path, T: float) -> list[dict]:
    from src.bd.km.a_data_xl import load_instance
    from src.bd.km.g_certificate_xl import Checker, lp_interval
    rows = []
    for path in paths:
        name = stem(path)
        inst = load_instance(path)
        runs = {f.name.split("__")[1][:-5]: json.loads(f.read_text()) for f in out.glob(f"{name}__*.json")
                if not f.name.split("__")[1].startswith("cert_") and f.name.split("__")[1] != "summary.json"}
        certs = {f.name.split("__cert_")[1][:-5]: json.loads(f.read_text()) for f in out.glob(f"{name}__cert_*.json")}
        best_cert = max((p["bound"] for c in certs.values() for p in c.get("points", []) if p.get("bound") is not None),
                        default=None)
        chk = Checker(inst)
        points = []
        for meth in ("eng", "eng_nostab", "naive_lp", "lp_bar", "lp_dual"):
            f = out / f"{name}__{meth}.npz"
            if f.exists():
                z = np.load(f)
                if "lp_point" in z:
                    points.append((meth, z["lp_point"]))
        iv = lp_interval(chk, [], points)
        iv["lo"], iv["lo_float"] = best_cert, best_cert   # 11-03 probe record: float lower end (not rigorous)
        iv["rel_width"] = ((iv["hi_float"] - best_cert) / max(1.0, abs(iv["hi_float"]))) if (best_cert is not None and iv["hi_float"]) else None
        ref = reference(name)
        target = None
        if iv["hi_float"] is not None and best_cert is not None:
            target = iv["hi_float"] - EPS * abs(best_cert)
        summ = {"instance": name, "m": inst.m, "n": inst.n, "k": inst.k, "lp_interval": iv, "target": target,
                "opt_ref": ref.get("opt_or_bks"), "opt_proven": ref.get("optimal"), "lb1_ref": ref.get("lb1"), "methods": {}}
        z_best = min((r["best"] for r in runs.values() if r.get("best") is not None), default=None)
        summ["z_best_run"] = z_best
        for meth, r in sorted(runs.items()):
            e = {"best": r.get("best"), "t_best": r.get("t_best"), "verified": r.get("verified"),
                 "solver_bound": r.get("solver_bound"), "solver_bound_t": r.get("solver_bound_t"),
                 "termination": r.get("stats", {}).get("termination_reason"), "error": (r.get("error") or "")[:300],
                 "peak_rss_mb": r.get("peak_rss_mb"), "late": r.get("late")}
            if z_best is not None and r.get("best") is not None:
                e["gap_to_zbest_pct"] = round(100 * (r["best"] - z_best) / max(1.0, abs(z_best)), 4)
            if summ["opt_ref"] and r.get("best") is not None:
                e["gap_to_ref_pct"] = round(100 * (r["best"] - summ["opt_ref"]) / summ["opt_ref"], 4)
            if r.get("simplex_bounds"):
                sb = r["simplex_bounds"]
                e["simplex_bound_final"] = sb[-1][1]
                e["simplex_bound_t_eps"] = next((t for t, b in sb if target is not None and b >= target), None)
            c = certs.get(meth)
            if c and c.get("points"):
                pts = [p for p in c["points"] if p.get("bound") is not None]
                e["cert_bound_T"] = max((p["bound"] for p in pts), default=None)
                e["cert_t_eps"] = next((p["t"] for p in sorted(pts, key=lambda p: p["t"]) if target is not None and p["bound"] >= target), None)
                if iv["hi_float"]:
                    e["cert_gap_to_lp_pct"] = round(100 * (iv["hi_float"] - e["cert_bound_T"]) / iv["hi_float"], 5) if e["cert_bound_T"] is not None else None
                if summ["opt_ref"] and e["cert_bound_T"] is not None:
                    e["cert_gap_to_opt_pct"] = round(100 * (summ["opt_ref"] - e["cert_bound_T"]) / summ["opt_ref"], 5)
            summ["methods"][meth] = e
        ctrl = certs.get("controls")
        if ctrl:
            summ["controls"] = {p["control"]: p["bound"] for p in ctrl.get("points", [])}
        (out / f"{name}__summary.json").write_text(json.dumps(summ, default=str, indent=1))
        rows.append(summ)
        print("SUMMARY " + json.dumps(summ, default=str), flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, metavar=("INSTANCE", "METHOD"))
    ap.add_argument("--certify", nargs=2, metavar=("INSTANCE", "WHAT"))
    ap.add_argument("--summarize", nargs="+", metavar="INSTANCE")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.run:
        r = run_one(a.run[0], a.run[1], a.time_limit, out)
        print("PROBE " + json.dumps({k: v for k, v in r.items() if k not in ("design", "incumbents", "bounds")}, default=str), flush=True)
    if a.certify:
        r = certify_one(a.certify[0], a.certify[1], a.time_limit, out)
        print("CERT " + json.dumps(r, default=str), flush=True)
    if a.summarize:
        summarize(a.summarize, out, a.time_limit)


if __name__ == "__main__":
    main()
