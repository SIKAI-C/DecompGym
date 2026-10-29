from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.cg.bppc.a_gen_data import generate_bppc_instance
from src.cg.bppc.b_problem import render_prompt
from src.utils.harder_suite import generate_suite as _generate_suite


def generate_suite(sizes: Iterable[str], num_instances: int, out_root: str | Path, *, seed_offset: int = 0) -> list[dict]:
    return _generate_suite("cg", "bppc", gen=generate_bppc_instance, render_prompt=render_prompt, sizes=list(sizes),
                           num_instances=num_instances, out_root=out_root, seed_offset=seed_offset)
