"""Independent verifier for planning-and-scheduling solutions (ps-v1 JSON instances, scale-up study; standard library
only).

Output contract: {"status": "optimal" | "feasible", "method": str, "objective_value": number,
                  "schedule": [{"job": j, "facility": i, "start": s}, ...]}   (0-based indices).
Checks, all exact on integers: every job appears exactly once; facility indices in range; starts are integers (JSON
integers, or floats with an integral value); r_j <= s_j and s_j + p_ij <= d_j on the chosen facility; on every
facility the summed resource rate of the jobs in process (job j occupies [s_j, s_j + p_ij)) never exceeds C_i (an
event sweep in which completions at time t are processed before starts at t); the objective sum_j f_{i(j) j} is
recomputed and must match the claim within tol * max(1, |objective|); a missing method, an unknown status or a
non-finite claim is rejected. The instance file is parsed here, independently of the solver code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def read_instance(path: str | Path) -> dict[str, Any]:
    obj = json.loads(Path(path).read_text())
    if obj.get("format") != "ps-v1":
        raise ValueError("not a ps-v1 instance")

    def ints(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) != int(v):
            raise ValueError("instance data must be integers")
        return int(v)

    C = [ints(v) for v in obj["capacity"]]
    r = [ints(v) for v in obj["release"]]
    d = [ints(v) for v in obj["deadline"]]
    P = [[ints(v) for v in row] for row in obj["proc"]]
    R = [[ints(v) for v in row] for row in obj["rate"]]
    F = [[ints(v) for v in row] for row in obj["cost"]]
    m, n = len(C), len(r)
    if len(d) != n or any(len(M) != m or any(len(row) != n for row in M) for M in (P, R, F)):
        raise ValueError("inconsistent instance dimensions")
    return {"m": m, "n": n, "C": C, "r": r, "d": d, "p": P, "c": R, "f": F}


def _is_int(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    return isinstance(v, float) and math.isfinite(v) and v.is_integer()


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> dict[str, Any]:
    d = read_instance(data_path)
    m, n = d["m"], d["n"]
    try:
        out = json.loads(Path(output_path).read_text())
    except (OSError, ValueError) as exc:
        return {"ok": False, "issues": [f"unreadable output: {exc}"]}
    if not isinstance(out, dict):
        return {"ok": False, "issues": ["output must be a JSON object"]}
    sched = out.get("schedule")
    if not isinstance(sched, list):
        return {"ok": False, "issues": ["schedule must be a list"]}
    issues: list[str] = []
    where: dict[int, tuple[int, int]] = {}
    for e in sched:
        if not isinstance(e, dict) or not all(_is_int(e.get(k)) for k in ("job", "facility", "start")):
            return {"ok": False, "issues": [f"invalid schedule entry {e!r}: integer job, facility and start required"]}
        j, i, s = int(e["job"]), int(e["facility"]), int(e["start"])
        if not 0 <= j < n or not 0 <= i < m:
            return {"ok": False, "issues": [f"job or facility index out of range in {e!r}"]}
        if j in where:
            return {"ok": False, "issues": [f"job {j} scheduled more than once"]}
        where[j] = (i, s)
    if len(where) != n:
        missing = sorted(set(range(n)) - set(where))
        return {"ok": False, "issues": [f"{len(missing)} jobs unscheduled, e.g. {missing[:5]}"]}
    events: list[list[tuple[int, int]]] = [[] for _ in range(m)]
    total = 0
    for j in range(n):
        i, s = where[j]
        p, c = d["p"][i][j], d["c"][i][j]
        if s < d["r"][j]:
            issues.append(f"job {j} starts at {s} before its release date {d['r'][j]}")
        if s + p > d["d"][j]:
            issues.append(f"job {j} ends at {s + p} after its deadline {d['d'][j]}")
        events[i].append((s, c))
        events[i].append((s + p, -c))
        total += d["f"][i][j]
    for i in range(m):
        load = 0
        for t, delta in sorted(events[i]):
            load += delta
            if load > d["C"][i]:
                issues.append(f"facility {i}: resource load {load} > capacity {d['C'][i]} at time {t}")
                break
    claimed = out.get("objective_value")
    if (not isinstance(claimed, (int, float)) or isinstance(claimed, bool) or not math.isfinite(claimed)
            or abs(claimed - total) > tol * max(1.0, abs(total))):
        issues.append(f"objective mismatch or non-finite: claimed {claimed}, recomputed {total}")
    if out.get("status") not in ("optimal", "feasible"):
        issues.append("status must be 'optimal' or 'feasible' for a returned schedule")
    if not isinstance(out.get("method"), str) or not out.get("method"):
        issues.append("method missing")
    return {"ok": not issues, "issues": issues[:20], "objective_value": total}
