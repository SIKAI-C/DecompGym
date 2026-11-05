"""Independent verifier for scale-up FCTP solutions (Klose ".FCTP" instances; standard library only).

Output contract: {"status": "optimal" | "feasible", "method": str, "objective_value": number,
                  "y": [[0/1 per customer] per supplier] (m x n),
                  and the flows either as "x": [[flow per customer] per supplier] (m x n)
                  or as "flows": [[supplier, customer, value], ...] (0-based; repeats are summed)}.
Checks: y binary of shape m x n; flows finite and >= -tol; every supplier ships exactly its supply and every customer
receives exactly its demand (|residual| <= tol * max(1, amount)); an arc with flow > tol must be open (y = 1); the
objective is recomputed as sum f_ij y_ij + sum c_ij x_ij and must match the claim within tol * max(1, |objective|);
non-finite claims, a missing method or an unknown status are rejected. The instance file is parsed here,
independently of the solver code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def read_fctp(path: str | Path) -> dict[str, Any]:
    tok = Path(path).read_text().split()
    m, n = int(tok[0]), int(tok[1])
    if len(tok) != 2 + m + n + 2 * m * n:
        raise ValueError("token count does not match the header")
    vals = [float(t) for t in tok[2:]]
    s, d = vals[:m], vals[m:m + n]
    c = [vals[m + n + i * n:m + n + (i + 1) * n] for i in range(m)]
    off = m + n + m * n
    f = [vals[off + i * n:off + (i + 1) * n] for i in range(m)]
    return {"m": m, "n": n, "s": s, "d": d, "c": c, "f": f}


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _flows(out: dict[str, Any], m: int, n: int) -> tuple[list[list[float]] | None, str | None]:
    if out.get("x") is not None:
        x = out["x"]
        if not isinstance(x, list) or len(x) != m or any(not isinstance(r, list) or len(r) != n for r in x):
            return None, f"x must be an {m} x {n} matrix"
        if not all(_is_num(v) for r in x for v in r):
            return None, "x is not numeric"
        return [[float(v) for v in r] for r in x], None
    if out.get("flows") is not None:
        if not isinstance(out["flows"], list):
            return None, "flows must be a list"
        x = [[0.0] * n for _ in range(m)]
        for e in out["flows"]:
            if not isinstance(e, (list, tuple)) or len(e) != 3:
                return None, "flows entries must be [supplier, customer, value]"
            i, j, v = e
            if isinstance(i, bool) or isinstance(j, bool) or not isinstance(i, int) or not isinstance(j, int) or not (0 <= i < m and 0 <= j < n):
                return None, f"invalid flow index {e}"
            if not _is_num(v):
                return None, f"invalid flow value {e}"
            x[i][j] += float(v)
        return x, None
    return None, "flows missing (need 'x' or 'flows')"


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> dict[str, Any]:
    d = read_fctp(data_path)
    m, n = d["m"], d["n"]
    try:
        out = json.loads(Path(output_path).read_text())
    except (OSError, ValueError) as exc:
        return {"ok": False, "issues": [f"unreadable output: {exc}"]}
    if not isinstance(out, dict):
        return {"ok": False, "issues": ["output must be a JSON object"]}
    issues: list[str] = []
    y = out.get("y")
    if (not isinstance(y, list) or len(y) != m or any(not isinstance(r, list) or len(r) != n for r in y)
            or not all(_is_num(v) and v in (0, 1) for r in y for v in r)):
        return {"ok": False, "issues": [f"y must be an {m} x {n} matrix of values in {{0, 1}}"]}
    x, err = _flows(out, m, n)
    if err:
        return {"ok": False, "issues": [err]}
    if not all(math.isfinite(v) for r in x for v in r):
        return {"ok": False, "issues": ["non-finite flow"]}
    neg = [(i, j) for i in range(m) for j in range(n) if x[i][j] < -tol]
    if neg:
        issues.append(f"negative flow on {len(neg)} arcs, e.g. {neg[0]}")
    bad_s = [i for i in range(m) if abs(sum(x[i]) - d["s"][i]) > tol * max(1.0, d["s"][i])]
    if bad_s:
        i = bad_s[0]
        issues.append(f"supply not shipped exactly at {len(bad_s)} suppliers, e.g. {i}: {sum(x[i])} vs {d['s'][i]}")
    bad_d = [j for j in range(n) if abs(sum(x[i][j] for i in range(m)) - d["d"][j]) > tol * max(1.0, d["d"][j])]
    if bad_d:
        j = bad_d[0]
        issues.append(f"demand not met exactly at {len(bad_d)} customers, e.g. {j}: {sum(x[i][j] for i in range(m))} vs {d['d'][j]}")
    closed = [(i, j) for i in range(m) for j in range(n) if x[i][j] > tol and y[i][j] != 1]
    if closed:
        issues.append(f"flow on {len(closed)} closed arcs, e.g. {closed[0]}")
    total = sum(d["f"][i][j] * y[i][j] + d["c"][i][j] * x[i][j] for i in range(m) for j in range(n))
    claimed = out.get("objective_value")
    if not _is_num(claimed) or not math.isfinite(claimed) or abs(claimed - total) > tol * max(1.0, abs(total)):
        issues.append(f"objective mismatch or non-finite: claimed {claimed}, recomputed {total}")
    if out.get("status") not in ("optimal", "feasible"):
        issues.append("status must be 'optimal' or 'feasible' for a returned plan")
    if not isinstance(out.get("method"), str) or not out.get("method"):
        issues.append("method missing")
    return {"ok": not issues, "issues": issues, "objective_value": total}
