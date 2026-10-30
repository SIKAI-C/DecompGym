"""Audit a candidate by enumerating depot patterns, without a Benders master.

This shares the exact-cover engine; tiny tests provide a separate exhaustive
oracle. All certificates are numerical and apply only to the supplied routes.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

from .data import validate_data, verify
from .exact_cover import solve_exact_cover
from .solvers import BudgetExpired, Solver, relative_gap


def certify(data, candidate, seconds=30.0):
    start = time.monotonic()
    validate_data(data)
    checked = verify(data, candidate)
    if not checked["ok"]:
        raise ValueError(f"Invalid candidate: {checked['issues']}")
    if data["m"] > 16:
        raise ValueError("Enumeration audit limited to 16 depots")
    upper = checked["recomputed_objective"]
    tolerance = 1e-7 * max(1.0, abs(upper))
    worker = Solver(data, "lp_benders", max(seconds, 1e-6))
    worker.deadline = start + max(0, seconds)
    patterns = []
    k = max(map(len, data["routes"]))

    def answer(status, lower=None, **extra):
        return {
            "status": status,
            "candidate_objective": upper,
            "lower_bound": lower,
            "relative_gap": relative_gap(lower, upper),
            "patterns_checked": len(patterns),
            "patterns_total": 2 ** data["m"],
            "patterns": patterns,
            "seconds": time.monotonic() - start,
            "scope": "supplied route pool; numerical tolerance 1e-7; no Benders master or saved cuts used",
            **extra,
        }

    try:
        for values in product((0, 1), repeat=data["m"]):
            worker.remaining()
            y = np.asarray(values)
            fixed = float(np.dot(data["depot_cost"], y))
            record = {"y": list(values), "fixed_cost": fixed}
            fleet = np.asarray(data["max_vehicles"]) * y
            if sum(fleet) * k < data["n"]:
                patterns.append({**record, "reason": "insufficient_fleet", "lower_bound": None})
                continue
            if fixed >= upper:
                patterns.append({**record, "reason": "opening_cost_bound", "lower_bound": fixed})
                continue
            lp = worker.recourse_lp(y)
            if lp.status == 2:
                patterns.append({**record, "reason": "lp_infeasible", "lower_bound": None})
                continue
            if lp.status != 0:
                return answer("unresolved", interrupted_pattern=record)
            # Direct route-wise dual validation/repair avoids relying on cuts or
            # the master model. Every route satisfies sum(alpha)+beta <= cost.
            alpha = np.asarray(lp.eqlin.marginals).copy()
            beta = np.minimum(lp.ineqlin.marginals, 0.0)
            if not np.isfinite(alpha).all() or not np.isfinite(beta).all():
                raise RuntimeError("Nonfinite LP dual")
            violation = max(
                0.0,
                max(
                    sum(alpha[i] for i in route) + beta[j] - c
                    for route, j, c in zip(data["routes"], data["route_depot"], data["route_cost"], strict=True)
                ),
            )
            alpha -= violation + 1e-10
            guard = 1e-9 * max(1.0, float(abs(alpha).sum() + abs(beta) @ fleet), data["n"] * max(data["route_cost"]))
            bound = fixed + max(0.0, float(alpha.sum() + beta @ fleet) - guard)
            record.update(lp_lower_bound=bound, alpha=alpha.tolist(), beta=beta.tolist(), numerical_guard=guard)
            if bound >= upper - tolerance:
                patterns.append({**record, "reason": "lp_bound", "lower_bound": bound})
                continue
            exact = solve_exact_cover(data, y, worker.remaining(), alpha, beta)
            if exact["status"] == "infeasible":
                patterns.append({**record, "reason": "integer_infeasible", "lower_bound": None, "exact_cover": exact})
                continue
            if exact["routes"] is not None and fixed + exact["objective"] < upper - tolerance:
                output = {
                    "y": list(values),
                    "lambda": [int(r in exact["routes"]) for r in range(len(data["routes"]))],
                    "objective_value": fixed + exact["objective"],
                }
                if not verify(data, output)["ok"]:
                    raise RuntimeError("Invalid counterexample")
                return answer("counterexample", better_solution=output, interrupted_pattern=record)
            patterns.append(
                {**record, "reason": "exact_cover", "lower_bound": fixed + exact["lower_bound"], "exact_cover": exact}
            )
            if exact["status"] != "optimal":
                return answer("unresolved")
    except BudgetExpired:
        return answer("unresolved")
    bounds = [r["lower_bound"] for r in patterns if r["lower_bound"] is not None]
    if not bounds:
        raise RuntimeError("All patterns excluded despite verified candidate")
    lower = min(bounds)
    gap = relative_gap(lower, upper)
    if gap is None:
        raise RuntimeError("Enumeration bound contradicts candidate")
    return answer("certified" if gap <= 1e-7 else "unresolved_gap", lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not 0 < args.seconds <= 3600:
        parser.error("seconds must be in (0, 3600]")
    args.output.mkdir(parents=True, exist_ok=False)
    results = json.loads((args.campaign / "results.json").read_text())
    evidence = []
    for row in results:
        if row["method"] != "integer_benders_exact_cover":
            continue
        data_path = args.campaign / row["instance"] / "data.json"
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        if digest != row["data_sha256"]:
            raise ValueError("Input hash changed")
        result = certify(json.loads(data_path.read_text()), row["incumbent"], args.seconds)
        result.update(instance=row["instance"], data_sha256=digest, candidate=row["incumbent"])
        evidence.append(result)
        (args.output / "certificates.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
        print(row["instance"], result["status"], result["lower_bound"], result["candidate_objective"], flush=True)
    if not evidence:
        raise ValueError("No exact-cover results in campaign")
    # Keep the auditing sources alongside the evidence, distinct from the run snapshot.
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    hashes = {}
    for path in Path(__file__).parent.glob("*.py"):
        payload = path.read_bytes()
        (snapshot / path.name).write_bytes(payload)
        hashes[path.name] = hashlib.sha256(payload).hexdigest()
    (args.output / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return int(any(r["status"] != "certified" for r in evidence))


if __name__ == "__main__":
    raise SystemExit(main())
