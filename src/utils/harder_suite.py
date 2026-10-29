"""Instance generation for the harder-family cohort, in the same artifact format as the paper's families.

Each instance folder gets data.json, prompt.txt, gold_monolithic.json, gold_<suite>.json and meta.json.
The ground truth is the monolithic MIP optimum when the MIP proves it; otherwise the decomposition
reference's own optimality certificate (user decision, 2026-10-28, for MIPs that cannot close the gap).
Both plans are re-verified with the harness checks; an instance without a certified optimum gets no meta.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from src.utils.api import solve, verify


def _certified(result: dict[str, Any], tol: float = 1e-6) -> bool:
    lb, ub = result.get("global_lb"), result.get("global_ub")
    return lb is not None and ub is not None and ub - lb <= tol * max(1.0, abs(ub))


def generate_suite(suite: str, problem: str, *, gen: Callable, render_prompt: Callable, sizes, num_instances: int, out_root: str | Path,
                   seed_offset: int = 0, mono_time_limit: float = 300.0, ref_time_limit: float = 60.0, tol: float = 1e-6) -> list[dict[str, Any]]:
    out_root = Path(out_root)
    metas = []
    for size in sizes:
        for seed in range(seed_offset, seed_offset + num_instances):
            folder = f"{problem}_{size}_seed{seed:03d}"
            inst_dir = out_root / folder
            data = gen(seed, size, inst_dir)
            (inst_dir / "prompt.txt").write_text(render_prompt(data).rstrip() + "\n")
            path = inst_dir / "data.json"
            gold = {}
            for method, filename, options in (("monolithic", "gold_monolithic.json", {"params": {"TimeLimit": mono_time_limit}}),
                                              ("decomposition", f"gold_{suite}.json", {"time_limit": ref_time_limit})):
                try:
                    result = solve(suite, problem, path, method=method, **options)
                    check = verify(suite, problem, path, result["output"])
                    result["verification"] = check
                    gold[method] = result if check["ok"] else None
                except Exception as exc:  # e.g. a MIP that cannot prove optimality within its limit
                    result, gold[method] = {"best_objective": None, "error": f"{type(exc).__name__}: {exc}"}, None
                (inst_dir / filename).write_text(json.dumps(result, indent=2, default=str))
            mono, ref = gold["monolithic"], gold["decomposition"]
            if mono is not None and ref is not None and _certified(ref) and abs(mono["best_objective"] - ref["best_objective"]) > tol * max(1.0, abs(mono["best_objective"])):
                raise RuntimeError(f"{folder}: certified reference {ref['best_objective']} differs from the MIP optimum {mono['best_objective']}")
            if mono is not None:
                truth, source = mono["best_objective"], "monolithic_mip"
            elif ref is not None and _certified(ref):
                truth, source = ref["best_objective"], "decomposition_certificate"
            else:
                (inst_dir / "meta.json").unlink(missing_ok=True)
                continue
            meta = {"seed": seed, "size": size, "folder": folder, "ground_truth_objective": truth, "ground_truth_source": source,
                    "reference_proved": bool(ref is not None and _certified(ref)), "reference_seconds": (ref or {}).get("total_time")}
            (inst_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            metas.append(meta)
    return metas
