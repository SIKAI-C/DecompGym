from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.bd.tsso.a_gen_data import TSSOGenerator, TSSOInstance, generate_tsso_instance
from src.bd.tsso.a_gold_bd import TSSOBenders
from src.bd.tsso.a_gold_monolithic import TSSOBaseline
from src.bd.tsso.b_problem import render_prompt
from src.utils.json_utils import save_json, serialize_iter_records


def generate_suite(
    sizes: Iterable[str],
    num_instances: int,
    out_root: str | Path,
    *,
    tol: float = 1e-4,
    seed_offset: int = 0,
) -> list[dict]:
    """
    Generate a small TSSO benchmark suite with ground truths and prompts.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    metas: list[dict] = []

    for size in sizes:
        for seed in range(seed_offset, seed_offset + num_instances):
            folder_name = f"tsso_{size}_seed{seed:03d}"
            inst_dir = out_root / folder_name
            inst_dir.mkdir(parents=True, exist_ok=True)

            data = generate_tsso_instance(seed=seed, size=size, output_dir=inst_dir)

            prompt_text = render_prompt(data)
            (inst_dir / "prompt.txt").write_text(prompt_text)

            data_path = str(inst_dir / "data.json")

            mono_solver = TSSOBaseline(verbose=False)
            mono_res = _make_serializable(mono_solver.run(data_path))
            save_json(mono_res, inst_dir / "gold_monolithic.json")
            mono_obj = mono_res.get("best_objective")

            bd_solver = TSSOBenders(verbose=False)
            bd_res = _make_serializable(bd_solver.run(data_path))
            save_json(bd_res, inst_dir / "gold_bd.json")
            bd_obj = bd_res.get("best_objective")

            if mono_obj is None or bd_obj is None:
                continue

            meta = {
                "seed": seed,
                "size": size,
                "n_products": data.get("n_products"),
                "n_periods": data.get("n_periods"),
                "n_scenarios": data.get("n_scenarios"),
                "ground_truth_objective": mono_obj,
                "folder": folder_name,
            }
            save_json(meta, inst_dir / "meta.json")
            metas.append(meta)
            _plot_artifacts(inst_dir, mono_res, bd_res, bd_solver)

    return metas


def _make_serializable(res: dict) -> dict:
    out = dict(res)
    if "iter_records" in out:
        out["iter_records"] = serialize_iter_records(out["iter_records"])
    return out


def _plot_artifacts(inst_dir: Path, mono_res: dict, bd_res: dict, bd_solver: TSSOBenders) -> None:
    inst = TSSOInstance.from_json_file(inst_dir / "data.json")
    gen = TSSOGenerator()
    plots_dir = inst_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    gen.plot_instance(inst, output_path=plots_dir / "instance.pdf")
    gen.plot_solution(inst, mono_res, output_path=plots_dir / "solution_monolithic.pdf")
    gen.plot_solution(inst, bd_res, output_path=plots_dir / "solution_bd.pdf")
    bd_solver.plot_bounds(output_dir=str(plots_dir))


if __name__ == "__main__":
    generate_suite(["small", "medium", "large"], num_instances=1, out_root="z_artifacts/benchmarks/bd/tsso")
