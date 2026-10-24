from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.cg.shift.a_gen_data import generate_shift_instance
from src.cg.shift.a_gold_cg import ShiftColumnGeneration
from src.cg.shift.a_gold_monolithic import ShiftBaseline
from src.cg.shift.b_problem import render_prompt
from src.utils.json_utils import save_json


def _serialize_iter_records(records: list) -> list[dict]:
    out: list[dict] = []
    for rec in records:
        out.append(
            {
                "iteration": rec.iteration,
                "elapsed_time": rec.elapsed_time,
                "iter_time": rec.iter_time,
                "master_obj": rec.master_obj,
                "reduced_cost": rec.reduced_cost,
                "n_patterns": rec.n_patterns,
            }
        )
    return out


def _serialize(res: dict) -> dict:
    out = dict(res)
    if "iter_records" in out:
        out["iter_records"] = _serialize_iter_records(out["iter_records"])
    return out


def generate_suite(
    sizes: Iterable[str],
    num_instances: int,
    out_root: str | Path,
    *,
    tol: float = 1e-4,
    seed_offset: int = 0,
) -> list[dict]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    metas: list[dict] = []

    for size in sizes:
        for seed in range(seed_offset, seed_offset + num_instances):
            folder_name = f"shift_{size}_seed{seed:03d}"
            inst_dir = out_root / folder_name
            inst_dir.mkdir(parents=True, exist_ok=True)

            data = generate_shift_instance(seed=seed, size=size, output_dir=inst_dir)
            (inst_dir / "prompt.txt").write_text(render_prompt(data).rstrip() + "\n")
            data_path = str(inst_dir / "data.json")

            mono_solver = ShiftBaseline(verbose=False)
            mono_res = _serialize(mono_solver.run(data_path))
            save_json(mono_res, inst_dir / "gold_monolithic.json")
            mono_obj = mono_res.get("best_objective")

            cg_solver = ShiftColumnGeneration(verbose=False)
            cg_res = _serialize(cg_solver.run(data_path))
            save_json(cg_res, inst_dir / "gold_cg.json")
            cg_obj = cg_res.get("best_objective")

            if mono_obj is None or cg_obj is None:
                continue
            if abs(mono_obj - cg_obj) > tol:
                continue

            meta = {
                "seed": seed,
                "size": size,
                "T": len(data.get("time_slots", [])),
                "ground_truth_objective": mono_obj,
                "folder": folder_name,
            }
            save_json(meta, inst_dir / "meta.json")
            metas.append(meta)
    return metas


if __name__ == "__main__":
    generate_suite(
        sizes=["small", "medium", "large"],
        num_instances=3,
        out_root="z_artifacts/benchmarks/cg/shift",
    )
