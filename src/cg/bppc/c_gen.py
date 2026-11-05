"""Literature-recipe small-scale suite: data, prompt, verified references and meta.json (strict policy, src.utils.litgen_suite)."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.cg.bppc.a_gen_data import generate_bppc_instance, recipe_record
from src.cg.bppc.b_problem import render_prompt
from src.utils.litgen_suite import generate_strict_suite


def _gen(seed: int, size: str, inst_dir: Path) -> tuple[dict, dict]:
    return (generate_bppc_instance(seed, size, inst_dir), recipe_record(seed, size))


def generate_suite(sizes: Iterable[str], num_instances: int, out_root: str | Path, *, seed_offset: int = 0) -> list[dict]:
    return generate_strict_suite("bppc", gen=_gen, render_prompt=render_prompt, sizes=list(sizes), num_instances=num_instances,
                                 out_root=out_root, seed_offset=seed_offset)
