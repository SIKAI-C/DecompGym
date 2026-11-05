"""Independent checker for planning-and-scheduling outputs: solution.schedule = [{"job", "facility", "start"}, ...].

All checks are exact on integers: every job appears exactly once; indices are in range; starts are integers;
r_j <= s_j and s_j + p_ij <= d_j on the chosen facility; on every facility the summed rate of the jobs in process
(job j occupies [s_j, s_j + p_ij)) never exceeds C_i (an event sweep in which completions at time t come before
starts at t). The objective sum_j f_{i(j) j} is recomputed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.utils.output_contract import integer_value


def _load(obj: Any) -> dict[str, Any]:
    if isinstance(obj, (str, Path)):
        return json.loads(Path(obj).read_text())
    if isinstance(obj, dict):
        return obj
    raise TypeError("expected dict or path-like input")


def _result(ok: bool, message: str, objective_value: float | None = None) -> dict[str, Any]:
    return {"ok": ok, "issues": [] if ok else [message], "objective_value": objective_value}


def verify_solution(data: Any, output: Any, tol: float | None = None) -> dict[str, Any]:
    _ = tol
    try:
        d, output = _load(data), _load(output)
        n, m = int(d["n_jobs"]), int(d["n_facilities"])
    except Exception as exc:
        return _result(False, f"data load error: {exc}")
    solution = output.get("solution")
    if not isinstance(solution, dict) or not isinstance(solution.get("schedule"), list):
        return _result(False, "solution.schedule missing or not a list")
    where: dict[int, tuple[int, int]] = {}
    for idx, entry in enumerate(solution["schedule"]):
        if not isinstance(entry, dict) or not all(integer_value(entry.get(k)) for k in ("job", "facility", "start")):
            return _result(False, f"schedule[{idx}] needs integer job, facility and start")
        j, i, s = int(entry["job"]), int(entry["facility"]), int(entry["start"])
        if not 0 <= j < n or not 0 <= i < m:
            return _result(False, f"schedule[{idx}] has a job or facility index out of range")
        if j in where:
            return _result(False, f"job {j} is scheduled more than once")
        where[j] = (i, s)
    if len(where) != n:
        return _result(False, f"{n - len(where)} job(s) not scheduled")
    events: list[list[tuple[int, int]]] = [[] for _ in range(m)]
    total = 0
    for j, (i, s) in where.items():
        p, c = d["proc"][i][j], d["rate"][i][j]
        if s < d["release"][j]:
            return _result(False, f"job {j} starts at {s}, before its release date {d['release'][j]}")
        if s + p > d["deadline"][j]:
            return _result(False, f"job {j} ends at {s + p}, after its deadline {d['deadline'][j]}")
        events[i] += [(s, c), (s + p, -c)]
        total += d["cost"][i][j]
    for i in range(m):
        load = 0
        for t, delta in sorted(events[i]):
            load += delta
            if load > d["capacity"][i]:
                return _result(False, f"facility {i}: resource load {load} exceeds capacity {d['capacity'][i]} at time {t}")
    return _result(True, "ok", float(total))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a planning-and-scheduling solution.")
    parser.add_argument("--dir", type=Path, default=Path("."))
    parser.add_argument("--output", default="output.json")
    args = parser.parse_args()
    print(json.dumps(verify_solution(args.dir / "data.json", args.dir / args.output), indent=2))


if __name__ == "__main__":
    main()
