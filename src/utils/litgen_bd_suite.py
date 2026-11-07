"""Instance generation for the literature-recipe small-scale BD families (UFL Koerkel-Ghosh, TSPLIB-derived p-median,
Agarwal-Aneja / Roberti FCTP), in the paper's artifact format.

Each instance folder gets data.json, prompt.txt, gold_monolithic.json, gold_bd.json, meta.json and the generator's
source.json (recipe provenance, copied to meta.json["instance_info"]; data.json holds only the model data). The ground truth is
the monolithic MIP optimum, proved on one thread with MIPGap = 0 within `mono_time_limit` (the MIP wrapper raises
otherwise) and its measured solve time must not exceed the limit. Both gold plans are re-verified with the harness
checks. The decomposition reference must finish within `ref_time_limit` (measured) and reach the MIP optimum within `max_reference_gap` (relative); its exact gap is
recorded. Each instance is built in a staging folder (`<out_root>.staging/<folder>`, outside the dataset) and replaces
the published folder only after every check passed; an instance that fails any check raises instead of being skipped
(no seed is ever dropped by solver outcome) and leaves the previously published folder untouched.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from src.utils.api import solve, verify


MONO_PARAMS = {"Threads": 1, "MIPGap": 0.0}


def generate_suite(problem: str, *, gen: Callable, render_prompt: Callable, sizes, num_instances: int, out_root: str | Path,
                   seed_offset: int = 0, mono_time_limit: float = 1800.0, ref_time_limit: float = 60.0,
                   max_reference_gap: float = 0.01) -> list[dict[str, Any]]:
    out_root = Path(out_root)
    metas = []
    for size in sizes:
        for seed in range(seed_offset, seed_offset + num_instances):
            folder = f"{problem}_{size}_seed{seed:03d}"
            inst_dir = Path(f"{out_root}.staging") / folder
            shutil.rmtree(inst_dir, ignore_errors=True)
            data = gen(seed, size, inst_dir)
            (inst_dir / "prompt.txt").write_text(render_prompt(data).rstrip() + "\n")
            path = inst_dir / "data.json"
            mono = solve("bd", problem, path, method="monolithic", params={**MONO_PARAMS, "TimeLimit": mono_time_limit})
            mono["verification"] = verify("bd", problem, path, mono["output"])
            (inst_dir / "gold_monolithic.json").write_text(json.dumps(mono, indent=2, default=str))
            ref = solve("bd", problem, path, method="decomposition", time_limit=ref_time_limit)
            ref["verification"] = verify("bd", problem, path, ref["output"])
            (inst_dir / "gold_bd.json").write_text(json.dumps(ref, indent=2, default=str))
            truth, value = float(mono["best_objective"]), float(ref["best_objective"])
            gap = (value - truth) / max(1.0, abs(truth))
            problems = [name for name, ok in (("monolithic plan rejected", mono["verification"]["ok"]),
                                              ("reference plan rejected", ref["verification"]["ok"]),
                                              (f"MIP time above {mono_time_limit} s", float(mono["total_time"]) <= mono_time_limit),
                                              ("MIP bound not closed", abs(mono["global_ub"] - mono["global_lb"]) <= 1e-9 * max(1.0, abs(truth))),
                                              (f"reference time above {ref_time_limit} s", 0.0 <= float(ref["total_time"]) <= ref_time_limit),
                                              ("reference below the proved optimum", gap >= -1e-9),
                                              (f"reference gap above {max_reference_gap}", gap <= max_reference_gap)) if not ok]
            if problems:
                raise RuntimeError(f"{folder}: " + "; ".join(problems))
            meta = {"seed": seed, "size": size, "folder": folder, "m": data.get("m"), "n": data.get("n"),
                    "ground_truth_objective": truth, "ground_truth_source": "monolithic_mip",
                    "mip_seconds": mono.get("total_time"), "mip_params": {**MONO_PARAMS, "TimeLimit": mono_time_limit},
                    "mip_seconds_definition": "wall time of the Gurobi optimize() call (BaseBaseline total_time)", "reference_objective": value, "reference_gap": gap,
                    "reference_seconds": ref.get("total_time"), "reference_proved": bool(ref.get("proven_optimal")),
                    "reference_termination": ref.get("termination_reason"),
                    # Recipe provenance comes from the source.json sidecar (never part of data.json or the prompt).
                    "instance_info": json.loads(source.read_text()) if (source := inst_dir / "source.json").is_file() else None}
            (inst_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            final = out_root / folder
            out_root.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(final, ignore_errors=True)
            shutil.move(str(inst_dir), str(final))
            try:
                inst_dir.parent.rmdir()  # the staging root disappears once empty
            except OSError:
                pass
            metas.append(meta)
            print(json.dumps({k: meta[k] for k in ("folder", "ground_truth_objective", "mip_seconds", "reference_objective", "reference_gap", "reference_seconds")}), flush=True)
    return metas
