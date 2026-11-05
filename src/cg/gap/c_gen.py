"""Literature-recipe small-scale suite: data, prompt, verified references and meta.json (strict policy, src.utils.litgen_suite)."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.cg.gap.a_gen_data import generate_gap_instance_with_recipe
from src.cg.gap.b_problem import render_prompt
from src.utils.litgen_suite import generate_strict_suite


def _gen(seed: int, size: str, inst_dir: Path) -> tuple[dict, dict]:
    return generate_gap_instance_with_recipe(seed, size, inst_dir)


def generate_suite(sizes: Iterable[str], num_instances: int, out_root: str | Path, *, seed_offset: int = 0) -> list[dict]:
    return generate_strict_suite("gap", gen=_gen, render_prompt=render_prompt, sizes=list(sizes), num_instances=num_instances,
                                 out_root=out_root, seed_offset=seed_offset)
