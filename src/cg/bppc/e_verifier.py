"""Independent verifier for bin packing with conflicts plans (scale-up study). Standard library only.

Output contract:
    {"status": "optimal" | "feasible", "method": str, "objective_value": number, "bins": [[item, ...], ...]}
with items numbered 1..N as in the instance file (first line "N C"; then "i w a1 ... ak": item i, its weight w and
items in conflict with it; conflicts are undirected, a pair listed on either line counts). Checks, from the instance
text only: every bin is a nonempty list of item numbers (integers, not booleans) in 1..N; every item is packed exactly
once; every bin's total weight is at most C; no bin holds two conflicting items; the objective (the number of bins)
matches the claim within tol * max(1, objective); the claim is a finite number; status is "optimal" or "feasible" and
method is a string.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def read_instance(path: str | Path) -> dict[str, Any]:
    rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    n, cap = int(rows[0][0]), int(rows[0][1])
    if len(rows) - 1 != n:
        raise ValueError(f"expected {n} item lines, got {len(rows) - 1}")
    w = [0] * (n + 1)
    conflicts: set[tuple[int, int]] = set()
    seen = set()
    for r in rows[1:]:
        i = int(r[0])
        if not 1 <= i <= n or i in seen:
            raise ValueError(f"item number {i} out of range or repeated")
        seen.add(i)
        w[i] = int(r[1])
        for t in r[2:]:
            j = int(t)
            if not 1 <= j <= n or j == i:
                raise ValueError(f"item {i} lists an invalid conflict {j}")
            conflicts.add((min(i, j), max(i, j)))
    return {"n": n, "C": cap, "w": w, "conflicts": conflicts}


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> dict[str, Any]:
    d = read_instance(data_path)
    try:
        out = json.loads(Path(output_path).read_text())
    except (OSError, json.JSONDecodeError) as err:
        return {"ok": False, "issues": [f"unreadable output: {err}"]}
    if not isinstance(out, dict):
        return {"ok": False, "issues": ["output must be a JSON object"]}
    bins = out.get("bins")
    if not isinstance(bins, list):
        return {"ok": False, "issues": ["bins missing"]}
    n, cap, w, conflicts = d["n"], d["C"], d["w"], d["conflicts"]
    issues: list[str] = []
    count = [0] * (n + 1)
    for k, b in enumerate(bins):
        if not isinstance(b, list) or not b or not all(isinstance(i, int) and not isinstance(i, bool) and 1 <= i <= n for i in b):
            issues.append(f"bin {k}: must be a nonempty list of item numbers in 1..{n}")
            continue
        load = sum(w[i] for i in b)
        if load > cap:
            issues.append(f"bin {k}: weight {load} exceeds capacity {cap}")
        for a in range(len(b)):
            for c in range(a + 1, len(b)):
                if (min(b[a], b[c]), max(b[a], b[c])) in conflicts:
                    issues.append(f"bin {k}: items {b[a]} and {b[c]} conflict")
        for i in b:
            count[i] += 1
    wrong = [i for i in range(1, n + 1) if count[i] != 1]
    if wrong:
        issues.append(f"{len(wrong)} items not packed exactly once (e.g. {wrong[:5]})")
    total = len(bins)
    claimed = out.get("objective_value")
    if isinstance(claimed, bool) or not isinstance(claimed, (int, float)) or not math.isfinite(claimed) or abs(claimed - total) > tol * max(1.0, abs(total)):
        issues.append(f"objective mismatch or non-finite: claimed {claimed}, recomputed {total}")
    if out.get("status") not in ("optimal", "feasible"):
        issues.append("status must be 'optimal' or 'feasible' for a returned plan")
    if not isinstance(out.get("method"), str):
        issues.append("method missing")
    return {"ok": not issues, "issues": issues, "objective_value": float(total), "bins": total}
