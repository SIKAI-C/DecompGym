"""Strict generation of literature-recipe small-scale instances (CG families bp, csp, gap, bppc).

An instance is published (meta.json written) only if
  1. the monolithic MIP proves optimality within its Gurobi TimeLimit (1800 s; the paper sets no MIP limit) (optimizer runtime; Threads = 1, MIPGap = 0) and the
     harness verifier accepts its plan;
  2. the decomposition reference returns a plan the verifier accepts, with total_time <= REF_TIME_LIMIT (60 s); and
  3. its objective lies within the 1 % rule: -1e-6 <= (z_ref - z_mip) / max(1, |z_mip|) <= 0.01 (a materially
     better reference than a proved optimum is an inconsistency).
Otherwise the instance is rejected: meta.json is removed, a rejection.json records the reason, and RuntimeError is
raised. Failures are never replaced by other seeds.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Callable

from src.utils.api import solve, verify

MIP_TIME_LIMIT = 1800.0
REF_TIME_LIMIT = 60.0
GAP_RULE = 0.01


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def generate_strict_suite(problem: str, *, gen: Callable[[int, str, Path], tuple[dict, dict]], render_prompt: Callable[[dict], str],
                          sizes, num_instances: int, out_root: str | Path, seed_offset: int = 0,
                          reference_options: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """gen(seed, size, inst_dir) writes data.json and returns (data, recipe); the recipe goes to meta.json only."""
    out_root = Path(out_root)
    metas = []
    for size in sizes:
        for seed in range(seed_offset, seed_offset + num_instances):
            folder = f"{problem}_{size}_seed{seed:03d}"
            inst_dir = out_root / folder
            inst_dir.mkdir(parents=True, exist_ok=True)
            for stale in ("meta.json", "rejection.json", "gold_monolithic.json", "gold_cg.json", "prompt.txt"):
                (inst_dir / stale).unlink(missing_ok=True)
            try:
                data, recipe = gen(seed, size, inst_dir)
                (inst_dir / "prompt.txt").write_text(render_prompt(data).rstrip() + "\n")
            except Exception as exc:
                _write(inst_dir / "rejection.json", {"folder": folder, "reason": f"generation or prompt failed: {type(exc).__name__}: {exc}"})
                raise RuntimeError(f"{folder}: rejected: generation or prompt failed") from exc
            path = inst_dir / "data.json"
            results: dict[str, dict[str, Any]] = {}
            for method, filename, options in (("monolithic", "gold_monolithic.json", {"params": {"TimeLimit": MIP_TIME_LIMIT, "Threads": 1, "MIPGap": 0.0}}),
                                              ("decomposition", "gold_cg.json", {"time_limit": REF_TIME_LIMIT, **(reference_options or {})})):
                started = time.perf_counter()
                try:
                    result = solve("cg", problem, path, method=method, **options)
                    result["verification"] = verify("cg", problem, path, result["output"])
                except Exception as exc:  # the MIP wrappers raise unless Gurobi reports OPTIMAL
                    result = {"best_objective": None, "error": f"{type(exc).__name__}: {exc}"}
                result["wall_seconds"] = time.perf_counter() - started
                _write(inst_dir / filename, result)
                results[method] = result
            mono, ref = results["monolithic"], results["decomposition"]
            reason, gap = None, None
            if mono.get("error") or not (mono.get("verification") or {}).get("ok"):
                reason = f"monolithic MIP not proved optimal or not verified: {mono.get('error') or mono.get('verification')}"
            elif ref.get("error") or not (ref.get("verification") or {}).get("ok"):
                reason = f"decomposition reference failed or not verified: {ref.get('error') or ref.get('verification')}"
            elif not 0.0 <= float(ref.get("total_time", math.inf)) <= REF_TIME_LIMIT:
                reason = f"decomposition reference time {ref.get('total_time')} s outside [0, {REF_TIME_LIMIT}] s"
            else:
                z_mip, z_ref = float(mono["best_objective"]), float(ref["best_objective"])
                gap = (z_ref - z_mip) / max(1.0, abs(z_mip))
                if not math.isfinite(gap) or gap < -1e-6 or gap > GAP_RULE:
                    reason = f"decomposition gap {gap} outside [-1e-6, {GAP_RULE}]"
            if reason is not None:
                _write(inst_dir / "rejection.json", {"folder": folder, "reason": reason})
                raise RuntimeError(f"{folder}: rejected: {reason}")
            meta = {"seed": seed, "size": size, "folder": folder, "ground_truth_objective": float(mono["best_objective"]),
                    "ground_truth_source": "monolithic_mip", "mip_time_limit": MIP_TIME_LIMIT, "mip_threads": 1, "mip_gap": 0.0,
                    "mip_seconds": mono.get("total_time", mono.get("runtime")), "mip_bound": mono.get("global_lb"),
                    "reference_objective": float(ref["best_objective"]), "reference_gap": gap,
                    "reference_bound": ref.get("global_lb"), "reference_seconds": ref.get("total_time"), **recipe}
            _write(inst_dir / "meta.json", meta)
            metas.append(meta)
    return metas
