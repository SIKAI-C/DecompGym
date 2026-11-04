"""Timing of the exact certificate checker on ~10^6 multiplier nonzeros (one core; run on Phoenix through
scaleup_devrun.sbatch: DEV_CMD='python scripts/phoenix/xl_certificate_bench.py').

Scenarios (synthetic data, decimal multipliers with float-like precision):
  ufl_multi   UFL 500 x 2000, 2000 single-customer cuts x 501 multipliers (dense link rows)
  ufl_agg     UFL 500 x 2000, 250 aggregated cuts over all 2000 customers, ~2 multipliers per term
  fctp_dense  FCTP 50 x 100, 200 cuts with all 5150 row multipliers (float duals, 17 significant digits)
  recert      exact master re-certification of the ufl_multi rows (~10^6 nonzeros) with every multiplier nonzero
  lp_interval rigorous LP* interval of a 500 x 500 UFL (Koerkel-Ghosh-like costs): Gurobi DE solve, exact dual bound,
              exact repaired primal
"""
from __future__ import annotations

import json
import random
import resource
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments import xl_certificate as xc  # noqa: E402
from src.experiments.xl_certificate import CheckerLimits, check_certificate, fctp_view, lp_interval, ufl_view  # noqa: E402


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run(name: str, view, cert: dict) -> dict:
    t0 = time.perf_counter()
    data = json.dumps(cert).encode()
    t1 = time.perf_counter()
    res = check_certificate(view, data, limits=CheckerLimits(max_seconds=3600.0))
    t2 = time.perf_counter()
    st = res.stats
    out = {"scenario": name, "bytes": len(data), "nonzeros": st.get("nonzeros"), "cuts": st.get("cuts"), "accepted": st.get("accepted"),
           "rejected": st.get("rejected"), "cert_ok": res.cert_ok, "reason": res.reason, "lb": res.lb_float,
           "parse_s": st.get("parse_seconds"), "expand_s": st.get("expand_seconds"), "master_s": st.get("master_seconds"),
           "recert_s": st.get("recert_seconds"), "check_s": round(t2 - t1, 3), "serialize_s": round(t1 - t0, 3),
           "master_status": st.get("master_status"), "master_obj": st.get("master_objective"),
           "master_nonzeros": st.get("master_nonzeros"), "peak_rss_mb": round(rss_mb(), 1)}
    print(json.dumps(out), flush=True)
    return out


def ufl(rng: random.Random, m: int = 500, n: int = 2000):
    D = [[rng.randint(1000, 2000) for _ in range(n)] for _ in range(m)]
    f = [rng.randint(1000, 2000) for _ in range(m)]
    t = time.perf_counter()
    view = ufl_view({"f": f, "D": D})
    print(json.dumps({"build_view": "ufl", "m": m, "n": n, "seconds": round(time.perf_counter() - t, 3), "peak_rss_mb": round(rss_mb(), 1)}), flush=True)
    return view, D, f


def ufl_multi(rng, view, D, m, n, cuts=2000):
    out = []
    for k in range(cuts):
        j = k % n
        ck = max(D[i][j] for i in range(m)) + rng.randint(0, 999999) / 1e6
        pi = {"d": f"{ck:.6f}"}
        pi.update({f"l{i}": f"{-(ck - D[i][j]):.6f}" for i in range(m)})
        out.append({"kind": "opt", "terms": [{"s": f"c{j}", "pi": pi}]})
    return {"view": "ufl-v1", "version": 1, "cuts": out}


def ufl_agg(rng, view, D, m, n, cuts=250):
    out = []
    for _ in range(cuts):
        terms = []
        for j in range(n):
            order = sorted(range(m), key=lambda i: D[i][j])
            ck = D[order[1]][j] + rng.randint(0, 10**6) / 10**rng.randint(3, 9)
            pi = {"d": repr(ck)}
            for i in order[:2]:
                if ck > D[i][j]:
                    pi[f"l{i}"] = repr(-(ck - D[i][j]))
            terms.append({"s": f"c{j}", "pi": pi})
        out.append({"kind": "opt", "terms": terms})
    return {"view": "ufl-v1", "version": 1, "cuts": out}


def fctp(rng, tmp: Path, m: int = 50, n: int = 100, cuts: int = 200):
    total = 20 * m * n
    cs = sorted(rng.sample(range(1, total), m - 1))
    cd = sorted(rng.sample(range(1, total), n - 1))
    s = [b - a for a, b in zip([0] + cs, cs + [total])]
    d = [b - a for a, b in zip([0] + cd, cd + [total])]
    c = [rng.randint(3, 8) for _ in range(m * n)]
    fx = [rng.randint(50, 200) for _ in range(m * n)]
    path = tmp / "bench.FCTP"
    path.write_text(f"{m} {n}\n" + " ".join(map(str, s)) + "\n" + " ".join(map(str, d)) + "\n" + " ".join(map(str, c)) + "\n"
                    + " ".join(map(str, fx)) + "\n")
    t = time.perf_counter()
    view = fctp_view(path)
    print(json.dumps({"build_view": "fctp", "m": m, "n": n, "seconds": round(time.perf_counter() - t, 3)}), flush=True)
    out = []
    for k in range(cuts):
        u = [rng.uniform(-5, 5) for _ in range(m)]
        v = [rng.uniform(0, 10) for _ in range(n)]
        pi = {f"s{i}": repr(u[i]) for i in range(m)}
        pi.update({f"d{j}": repr(v[j]) for j in range(n)})
        pi.update({f"l{i}_{j}": repr(-max(0.0, u[i] + v[j] - c[i * n + j])) for i in range(m) for j in range(n)})
        out.append({"kind": "opt" if k % 4 else "feas", "terms": [{"s": "t", "pi": pi}]})
    return view, {"view": "fctp-v1", "version": 1, "cuts": out}


def recert_stress(rng, view, cert) -> None:
    st = xc._Stats()
    rows = [xc.build_cut(view, c, st, CheckerLimits()) for c in cert["cuts"]]
    obj, lo, hi = xc._master_columns(view)
    duals = [rng.uniform(0.001, 1.0) for _ in rows]
    t = time.perf_counter()
    b = xc.recertify(obj, lo, hi, rows, duals)
    print(json.dumps({"scenario": "recert", "rows": len(rows), "nonzeros": sum(len(r.cols) for r in rows), "nonzero_duals": len(duals),
                      "seconds": round(time.perf_counter() - t, 3), "bound": xc.float_down(b)}), flush=True)


def interval(rng, m: int = 500, n: int = 500) -> None:
    D = [[rng.randint(1000, 2000) for _ in range(n)] for _ in range(m)]
    f = [rng.randint(1000, 2000) for _ in range(m)]
    view = ufl_view({"f": f, "D": D})
    t = time.perf_counter()
    out = lp_interval(view, time_limit=1200.0)
    out.pop("repair_notes", None)
    print(json.dumps({"scenario": "lp_interval", "m": m, "n": n, "seconds": round(time.perf_counter() - t, 3), **{k: out.get(k) for k in (
        "status", "solve_seconds", "dual_seconds", "primal_seconds", "repair", "rigorous", "lp_lo_float", "lp_hi_float", "width",
        "relative_width", "gurobi_objective")}, "peak_rss_mb": round(rss_mb(), 1)}), flush=True)


def main() -> None:
    rng = random.Random(0)
    m, n = 500, 2000
    view, D, f = ufl(rng, m, n)
    multi = ufl_multi(rng, view, D, m, n)
    run("ufl_multi", view, multi)
    recert_stress(rng, view, multi)
    del multi
    run("ufl_agg", view, ufl_agg(rng, view, D, m, n))
    with tempfile.TemporaryDirectory() as tmp:
        fview, cert = fctp(rng, Path(tmp))
        run("fctp_dense", fview, cert)
    del view, D
    interval(rng)


if __name__ == "__main__":
    main()
