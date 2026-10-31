"""Independent verifier for MCND solutions (Canad ".dow" instances, scale-up study).

Output contract: {"status": "optimal" | "feasible", "method": str, "objective_value": number,
                  "y": [0/1 per arc, in file order],
                  and the flows either as "x": [[flow per arc] per commodity] (K x m, file order)
                  or as "flows": [[commodity, arc, value], ...] (0-based, nonzero entries only; repeats are summed)}.
Checks: y binary of length m; flows finite and >= -tol; flow conservation for every commodity and node
(|outflow - inflow - b| <= tol * max(1, d_k)); capacities sum_k x_ak <= u_a y_a + tol * max(1, u_a), so closed arcs
carry no flow; the objective is recomputed as sum_a f_a y_a + sum_{a,k} c_a x_ak and must match the claim within
tol * max(1, |objective|); non-finite claims, a missing method or an unknown status are rejected.
The instance file is parsed here, independently of the solver code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_dow(path: str | Path) -> dict[str, Any]:
    lines = Path(path).read_text().splitlines()
    if not lines or lines[0].strip() != "MULTIGEN.DAT:":
        raise ValueError("not a Canad .dow file")
    tok = [int(t) for t in " ".join(lines[1:]).split()]
    n, m, K = tok[:3]
    if len(tok) != 3 + 7 * m + 3 * K:
        raise ValueError("token count does not match the header")
    arcs = np.array(tok[3:3 + 7 * m], dtype=np.int64).reshape(m, 7)
    com = np.array(tok[3 + 7 * m:], dtype=np.int64).reshape(K, 3)
    return {"n": n, "m": m, "K": K, "tail": arcs[:, 0] - 1, "head": arcs[:, 1] - 1, "c": arcs[:, 2].astype(float),
            "u": arcs[:, 3].astype(float), "f": arcs[:, 4].astype(float), "o": com[:, 0] - 1, "t": com[:, 1] - 1,
            "d": com[:, 2].astype(float)}


def _flows(out: dict[str, Any], K: int, m: int) -> tuple[np.ndarray | None, str | None]:
    if "x" in out and out["x"] is not None:
        try:
            x = np.array(out["x"], dtype=float)
        except (TypeError, ValueError):
            return None, "x is not numeric"
        if x.shape != (K, m):
            return None, f"x has shape {x.shape}, expected ({K}, {m})"
        return x, None
    if "flows" in out and out["flows"] is not None:
        x = np.zeros((K, m))
        for entry in out["flows"]:
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                return None, "flows entries must be [commodity, arc, value]"
            k, a, v = entry
            if isinstance(k, bool) or isinstance(a, bool) or not isinstance(k, int) or not isinstance(a, int) or not (0 <= k < K and 0 <= a < m):
                return None, f"invalid flow index {entry}"
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None, f"invalid flow value {entry}"
            x[k, a] += float(v)
        return x, None
    return None, "flows missing (need 'x' or 'flows')"


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> dict[str, Any]:
    d = read_dow(data_path)
    out = json.loads(Path(output_path).read_text())
    issues: list[str] = []
    n, m, K = d["n"], d["m"], d["K"]
    y_raw = out.get("y")
    if not isinstance(y_raw, list) or len(y_raw) != m or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v in (0, 1) for v in y_raw):
        return {"ok": False, "issues": [f"y must be a list of {m} values in {{0, 1}}"]}
    y = np.array(y_raw, dtype=float)
    x, err = _flows(out, K, m)
    if err:
        return {"ok": False, "issues": [err]}
    if not np.all(np.isfinite(x)):
        return {"ok": False, "issues": ["non-finite flow"]}
    if np.any(x < -tol):
        issues.append(f"negative flow (min {x.min():.3g})")
    # Flow conservation: outflow - inflow = +d at the origin, -d at the destination, 0 elsewhere.
    net = np.zeros((K, n))
    np.add.at(net.T, d["tail"], x.T)
    np.subtract.at(net.T, d["head"], x.T)
    b = np.zeros((K, n))
    b[np.arange(K), d["o"]] += d["d"]
    b[np.arange(K), d["t"]] -= d["d"]
    resid = np.abs(net - b)
    bad = resid > tol * np.maximum(1.0, d["d"])[:, None]
    if bad.any():
        k, i = map(int, np.argwhere(bad)[0])
        issues.append(f"flow conservation violated at {int(bad.sum())} (commodity, node) pairs, e.g. commodity {k} node {i}: residual {resid[k, i]:.3g}")
    load = x.sum(axis=0)
    over = load > d["u"] * y + tol * np.maximum(1.0, d["u"])
    if over.any():
        a = int(np.nonzero(over)[0][0])
        issues.append(f"capacity violated on {int(over.sum())} arcs, e.g. arc {a}: load {load[a]:.6g}, open {int(y[a])}, capacity {d['u'][a]:.6g}")
    total = float(d["f"] @ y + (d["c"][None, :] * x).sum())
    claimed = out.get("objective_value")
    if isinstance(claimed, bool) or not isinstance(claimed, (int, float)) or not math.isfinite(claimed) or abs(claimed - total) > tol * max(1.0, abs(total)):
        issues.append(f"objective mismatch or non-finite: claimed {claimed}, recomputed {total}")
    if out.get("status") not in ("optimal", "feasible"):
        issues.append("status must be 'optimal' or 'feasible' for a returned design")
    if not isinstance(out.get("method"), str) or not out.get("method"):
        issues.append("method missing")
    return {"ok": not issues, "issues": issues, "objective_value": total}
