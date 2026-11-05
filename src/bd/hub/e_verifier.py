"""Independent verifier for UMAHLP solutions (scale-up study).

Output contract: {"status": "optimal" | "feasible", "method": str, "objective_value": number, "hubs": [hub indices]}
with 0-based node indices of the open hubs. Checks: hubs is a nonempty list of distinct integers in [0, n); the
objective is recomputed as
    sum_{k in hubs} f_k + sum_{i,j} w_ij * min_{k,m in hubs} (chi d_ik + alpha d_km + delta d_mj)
(every ordered pair, including i = j, each on its cheapest path through one or two open hubs; multiple allocation
and uncapacitated hubs make this routing optimal for the hub set) and must match the claim within
tol * max(1, |objective|); non-finite claims, a missing method or an unknown status are rejected.
The data file (a_data's self-contained JSON: n, w, d, f, chi, alpha, delta) is parsed here, independently of the
solver code; the routing is computed first hub by first hub (a different order than b_mip_xl.route_costs).
A spec file ({"source": "AP", ...}) has no data of its own and is materialized with a_data first (then only the
objective recomputation is independent of the solver code).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_data(path: str | Path) -> dict[str, Any]:
    obj = json.loads(Path(path).read_text())
    if "w" not in obj:
        from src.bd.hub.a_data import from_json  # spec file: materialize the instance
        inst = from_json(obj)
        return {"n": inst.n, "w": inst.w, "d": inst.d, "f": inst.f, "chi": inst.chi, "alpha": inst.alpha, "delta": inst.delta}
    n = int(obj["n"])
    out = {"n": n, "w": np.array(obj["w"], dtype=float), "d": np.array(obj["d"], dtype=float), "f": np.array(obj["f"], dtype=float),
           "chi": float(obj["chi"]), "alpha": float(obj["alpha"]), "delta": float(obj["delta"])}
    if out["w"].shape != (n, n) or out["d"].shape != (n, n) or out["f"].shape != (n,):
        raise ValueError("data file: w and d must be n x n and f of length n")
    for key in ("w", "d", "f"):
        if not np.all(np.isfinite(out[key])):
            raise ValueError(f"data file: non-finite {key}")
    return out


def routing_cost(data: dict[str, Any], hubs: list[int]) -> float:
    """sum_ij w_ij * cheapest path cost through the open hubs, enumerated first hub by first hub."""
    d, w = data["d"], data["w"]
    H = np.asarray(hubs, dtype=np.int64)
    n = data["n"]
    best = np.full((n, n), np.inf)
    tail = data["alpha"] * d[H][:, H]           # (k, m)
    last = data["delta"] * d[H, :]              # (m, j)
    for a, k in enumerate(H):
        rest = (tail[a][:, None] + last).min(axis=0)          # min over second hub m of alpha d_km + delta d_mj
        best = np.minimum(best, data["chi"] * d[:, k][:, None] + rest[None, :])
    return float((w * best).sum())


def objective(data: dict[str, Any], hubs: list[int]) -> float:
    return float(data["f"][np.asarray(hubs, dtype=np.int64)].sum()) + routing_cost(data, hubs)


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> dict[str, Any]:
    data = read_data(data_path)
    out = json.loads(Path(output_path).read_text())
    n = data["n"]
    hubs = out.get("hubs")
    if (not isinstance(hubs, list) or not hubs
            or not all(isinstance(h, int) and not isinstance(h, bool) and 0 <= h < n for h in hubs)
            or len(set(hubs)) != len(hubs)):
        return {"ok": False, "issues": [f"hubs must be a nonempty list of distinct integers in [0, {n})"]}
    issues: list[str] = []
    total = objective(data, hubs)
    claimed = out.get("objective_value")
    if (isinstance(claimed, bool) or not isinstance(claimed, (int, float)) or not math.isfinite(claimed)
            or abs(claimed - total) > tol * max(1.0, abs(total))):
        issues.append(f"objective mismatch or non-finite: claimed {claimed}, recomputed {total}")
    if out.get("status") not in ("optimal", "feasible"):
        issues.append("status must be 'optimal' or 'feasible' for a returned hub set")
    if not isinstance(out.get("method"), str) or not out.get("method"):
        issues.append("method missing")
    return {"ok": not issues, "issues": issues, "objective_value": total}
