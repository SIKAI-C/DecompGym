from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.bd.ufl.a_gen_data import generate_ufl_instance
from src.bd.ufl.b_problem import render_prompt
from src.utils.litgen_bd_suite import generate_suite as _generate_suite


def generate_suite(sizes: Iterable[str], num_instances: int, out_root: str | Path, *, seed_offset: int = 0) -> list[dict]:
    """Koerkel-Ghosh UFL instances (a_gen_data.generate_ufl_instance) with a proved MIP optimum and a matching Benders reference."""
    return _generate_suite("ufl", gen=generate_ufl_instance, render_prompt=render_prompt, sizes=list(sizes),
                           num_instances=num_instances, out_root=out_root, seed_offset=seed_offset)


if __name__ == "__main__":
    generate_suite(sizes=["small", "medium", "large"], num_instances=3, out_root="outputs/litgen_data/bd/ufl")
