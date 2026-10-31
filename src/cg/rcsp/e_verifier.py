"""Independent verifier for RCSP duty plans (van Rossum 2026 benchmark, scale-up study).

Output contract:
    {"status": "optimal" | "feasible", "method": str, "objective_value": number,
     "duties": [{"depot": <crew base station>, "tasks": [<task index>, ...]}, ...]}
with each duty's tasks in performing order. The instance is given as the RCSP spec JSON
{"dir": <instance directory>, "variant": "single" | "multi"} (a relative dir is resolved against the spec's folder).

Checks, from the raw text files only (no solver code is imported):
- every duty: the depot is a crew base of the variant; a SOURCE connection depot -> first task, a CONNECTION for
  every consecutive pair and a SINK last task -> depot exist in connections_<variant>_depot.txt;
- duty length (sum of the connection durations) <= 34 200 s (9.5 h);
- meal break: some break connection a in the duty (timeBeforeBreak >= 0) has (durations of the connections before a)
  + timeBeforeBreak(a) <= 19 800 s and timeAfterBreak(a) + (durations of the connections after a) <= 19 800 s;
- every task of tasks.txt is in at least one duty;
- the objective (sum of the connection costs of all duties) matches the claim within tol * max(1, |objective|).
`verify_connection_file` checks a solution in the benchmark's own format (one duty per line, connection indices).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MAX_DUTY = 34_200
MAX_WORK = 19_800


def read_raw(spec_path: str | Path) -> dict[str, Any]:
    p = Path(spec_path)
    spec = json.loads(p.read_text())
    variant = spec.get("variant")
    if variant not in ("single", "multi") or "dir" not in spec:
        raise ValueError('spec must be {"dir": ..., "variant": "single" | "multi"}')
    d = Path(spec["dir"])
    if not d.is_absolute():
        d = (p.parent / d).resolve()
    rows = lambda f: [line.split() for line in (d / f).read_text().splitlines() if line.strip()]
    tasks = {int(r[0]) for r in rows("tasks.txt")}
    bases = {int(r[0]) for r in rows(f"crewbases_{variant}_depot.txt")}
    source, sink, conn, by_index = {}, {}, {}, {}
    for r in rows(f"connections_{variant}_depot.txt"):
        rec = {"index": int(r[0]), "type": r[1], "a": int(r[2]), "b": int(r[3]), "tbb": int(r[4]), "tab": int(r[5]),
               "dur": int(r[6]), "cost": int(r[7])}
        by_index[rec["index"]] = rec
        table = {"SOURCE": source, "SINK": sink, "CONNECTION": conn}[rec["type"]]
        table[(rec["a"], rec["b"])] = rec
    return {"tasks": tasks, "bases": bases, "source": source, "sink": sink, "conn": conn, "by_index": by_index}


def check_duty(raw: dict[str, Any], depot: Any, tasks: Any) -> tuple[int | None, str]:
    if not isinstance(depot, int) or isinstance(depot, bool) or depot not in raw["bases"]:
        return None, f"depot {depot!r} is not a crew base"
    if not isinstance(tasks, list) or not tasks or not all(isinstance(t, int) and not isinstance(t, bool) and t in raw["tasks"] for t in tasks):
        return None, "tasks must be a nonempty list of task indices"
    chain = [raw["source"].get((depot, tasks[0]))] + [raw["conn"].get((a, b)) for a, b in zip(tasks, tasks[1:])] + [raw["sink"].get((tasks[-1], depot))]
    if any(c is None for c in chain):
        k = chain.index(None)
        return None, "missing SOURCE" if k == 0 else "missing SINK" if k == len(chain) - 1 else f"no connection {tasks[k - 1]} -> {tasks[k]}"
    return evaluate_chain(chain)


def evaluate_chain(chain: list[dict[str, Any]]) -> tuple[int | None, str]:
    length = sum(c["dur"] for c in chain)
    if length > MAX_DUTY:
        return None, f"duty length {length} s exceeds {MAX_DUTY} s"
    elapsed = 0
    for c in chain:
        if c["tbb"] >= 0 and elapsed + c["tbb"] <= MAX_WORK and c["tab"] + (length - elapsed - c["dur"]) <= MAX_WORK:
            return sum(c["cost"] for c in chain), ""
        elapsed += c["dur"]
    return None, "no meal break with at most 5.5 h of work before and after it"


def verify_solution(spec_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> dict[str, Any]:
    raw = read_raw(spec_path)
    try:
        out = json.loads(Path(output_path).read_text())
    except (OSError, json.JSONDecodeError) as err:
        return {"ok": False, "issues": [f"unreadable output: {err}"]}
    if not isinstance(out, dict):
        return {"ok": False, "issues": ["output must be a JSON object"]}
    duties = out.get("duties")
    if not isinstance(duties, list):
        return {"ok": False, "issues": ["duties missing"]}
    issues: list[str] = []
    covered: set[int] = set()
    total = 0
    for k, duty in enumerate(duties):
        if not isinstance(duty, dict):
            issues.append(f"duty {k}: not an object")
            continue
        cost, why = check_duty(raw, duty.get("depot"), duty.get("tasks"))
        if cost is None:
            issues.append(f"duty {k}: {why}")
            continue
        total += cost
        covered.update(duty["tasks"])
    missing = raw["tasks"] - covered
    if missing:
        issues.append(f"{len(missing)} tasks uncovered (e.g. {sorted(missing)[:5]})")
    claimed = out.get("objective_value")
    if isinstance(claimed, bool) or not isinstance(claimed, (int, float)) or not math.isfinite(claimed) or abs(claimed - total) > tol * max(1.0, abs(total)):
        issues.append(f"objective mismatch or non-finite: claimed {claimed}, recomputed {total}")
    if out.get("status") not in ("optimal", "feasible"):
        issues.append("status must be 'optimal' or 'feasible' for a returned plan")
    if not isinstance(out.get("method"), str):
        issues.append("method missing")
    return {"ok": not issues, "issues": issues, "objective_value": float(total), "duties": len(duties)}


def verify_connection_file(spec_path: str | Path, solution_path: str | Path) -> dict[str, Any]:
    """A solution in the benchmark's format (one duty per line, connection indices SOURCE ... SINK)."""
    raw = read_raw(spec_path)
    issues: list[str] = []
    covered: set[int] = set()
    total, n = 0, 0
    for k, line in enumerate(x for x in Path(solution_path).read_text().splitlines() if x.strip()):
        n += 1
        chain = [raw["by_index"].get(int(c)) for c in line.split()]
        if len(chain) < 2 or any(c is None for c in chain):
            issues.append(f"duty {k}: unknown connection")
            continue
        types = [c["type"] for c in chain]
        if types[0] != "SOURCE" or types[-1] != "SINK" or any(t != "CONNECTION" for t in types[1:-1]):
            issues.append(f"duty {k}: not SOURCE -> CONNECTION* -> SINK")
            continue
        if any(a["b"] != b["a"] for a, b in zip(chain, chain[1:])) or chain[0]["a"] != chain[-1]["b"]:
            issues.append(f"duty {k}: connections not consecutive or crew bases differ")
            continue
        cost, why = evaluate_chain(chain)
        if cost is None:
            issues.append(f"duty {k}: {why}")
            continue
        total += cost
        covered.update(c["b"] for c in chain[:-1])
    missing = raw["tasks"] - covered
    if missing:
        issues.append(f"{len(missing)} tasks uncovered")
    return {"ok": not issues, "issues": issues, "objective_value": float(total), "duties": n}
