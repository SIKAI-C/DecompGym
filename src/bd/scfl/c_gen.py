from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.bd.scfl.a_gen_data import generate_scfl_instance
from src.bd.scfl.b_problem import render_prompt
from src.utils.harder_suite import generate_suite as _generate_suite


def generate_suite(sizes: Iterable[str], num_instances: int, out_root: str | Path, *, seed_offset: int = 0) -> list[dict]:
    return _generate_suite("bd", "scfl", gen=generate_scfl_instance, render_prompt=render_prompt, sizes=list(sizes),
                           num_instances=num_instances, out_root=out_root, seed_offset=seed_offset, mono_time_limit=60.0, strict=True,
                           tol=1e-9)  # the SCFL certificate tolerance (a_gold_monolithic.CERT_TOL)
