from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.bd.lrp.a_gen_data import generate_lrp_instance
from src.bd.lrp.a_gold_bd import LRPBenders
from src.bd.lrp.a_gold_monolithic import LRPBaseline
from src.bd.lrp.b_problem import render_prompt
from src.utils.json_utils import save_json, serialize_iter_records


def generate_suite(
    sizes: Iterable[str],
    num_instances: int,
    out_root: str | Path,
    *,
    seed_offset: int = 0,
    tol: float = 1e-4,
) -> list[dict]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    metas: list[dict] = []

    for size in sizes:
        for seed in range(seed_offset, seed_offset + num_instances):
            folder_name = f"lrp_{size}_seed{seed:03d}"
            inst_dir = out_root / folder_name
            inst_dir.mkdir(parents=True, exist_ok=True)

            data = generate_lrp_instance(seed=seed, size=size, output_dir=inst_dir)
            (inst_dir / "prompt.txt").write_text(render_prompt(data))

            data_path = str(inst_dir / "data.json")
            mono_solver = LRPBaseline(verbose=False)
            mono_res = _serialize(mono_solver.run(data_path))
            save_json(mono_res, inst_dir / "gold_monolithic.json")
            mono_obj = mono_res.get("best_objective")

            bd_solver = LRPBenders(verbose=False)
            bd_res = _serialize(bd_solver.run(data_path))
            save_json(bd_res, inst_dir / "gold_bd.json")
            bd_obj = bd_res.get("best_objective")

            if mono_obj is None or bd_obj is None or abs(mono_obj - bd_obj) > tol:
                continue

            meta = {
                "seed": seed,
                "size": size,
                "m": data.get("m"),
                "n": data.get("n"),
                "routes": len(data.get("routes", [])),
                "ground_truth_objective": mono_obj,
                "folder": folder_name,
            }
            save_json(meta, inst_dir / "meta.json")
            metas.append(meta)

    return metas


def _serialize(res: dict) -> dict:
    out = dict(res)
    if "iter_records" in out:
        out["iter_records"] = serialize_iter_records(out["iter_records"])
    return out


if __name__ == "__main__":
    generate_suite(["small", "medium", "large"], num_instances=3, out_root="z_artifacts/benchmarks/bd/lrp")
