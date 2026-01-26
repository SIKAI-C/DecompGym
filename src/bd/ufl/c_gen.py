from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.bd.ufl.a_gen_data import generate_ufl_instance
from src.bd.ufl.b_gold_bd import UFLBenders
from src.bd.ufl.b_gold_monolithic import UFLBaseline
from src.bd.ufl.b_problem import render_prompt
from src.utils.json_utils import save_json, serialize_iter_records


def generate_suite(
    difficulties: Iterable[str],
    num_instances: int,
    out_root: str | Path,
    *,
    tol: float = 1e-4,
    seed_offset: int = 0,
) -> list[dict]:
    """
    Generate a small UFL benchmark suite with ground truths.

    For each difficulty and seed in range(num_instances):
      - build instance folder: ufl_{difficulty}_seedXXX
      - write data.json via generate_ufl_instance
      - write prompt.txt via render_prompt
      - run gold monolithic and gold Benders solvers
      - if objectives match within tol, write meta.json and keep instance
      - otherwise, skip (and optionally clean up)
    Returns a list of meta dicts for successful instances.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    metas: list[dict] = []

    for diff in difficulties:
        for seed in range(seed_offset, seed_offset + num_instances):
            folder_name = f"ufl_{diff}_seed{seed:03d}"
            inst_dir = out_root / folder_name
            inst_dir.mkdir(parents=True, exist_ok=True)

            # 1) Generate instance (writes data.json)
            data = generate_ufl_instance(seed=seed, difficulty=diff, output_dir=inst_dir)

            # 2) Write prompt.txt
            prompt_text = render_prompt(data)
            (inst_dir / "prompt.txt").write_text(prompt_text)

            data_path = str(inst_dir / "data.json")

            # 3) Gold monolithic
            mono_solver = UFLBaseline(verbose=False)
            mono_res = _make_serializable(mono_solver.run(data_path))
            save_json(mono_res, inst_dir / "gold_monolithic.json")
            mono_obj = mono_res.get("best_objective")

            # 4) Gold Benders
            bd_solver = UFLBenders(verbose=False)
            bd_res = _make_serializable(bd_solver.run(data_path))
            save_json(bd_res, inst_dir / "gold_bd.json")
            bd_obj = bd_res.get("best_objective")

            # 5) Validate objectives
            if mono_obj is None or bd_obj is None:
                # keep files for debugging, but do not record meta
                continue
            if abs(mono_obj - bd_obj) > tol:
                # mismatch; skip meta
                continue

            meta = {
                "seed": seed,
                "difficulty": diff,
                "m": data.get("m"),
                "n": data.get("n"),
                "ground_truth_objective": mono_obj,
            }
            save_json(meta, inst_dir / "meta.json")
            metas.append(meta)

    return metas


def _make_serializable(res: dict) -> dict:
    """Convert fields that are not JSON-serializable (e.g., IterationRecord)."""
    out = dict(res)
    if "iter_records" in out:
        out["iter_records"] = serialize_iter_records(out["iter_records"])
    return out


if __name__ == "__main__":
    generate_suite(difficulties=["easy", "medium", "hard"], num_instances=3, out_root="z_artifacts/benchmarks/ufl")
