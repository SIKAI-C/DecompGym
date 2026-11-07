"""Instance suite of the scfl family: data, prompt, verified references and meta.json (src.utils.instance_suite)."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.utils.instance_suite import generate_suite as _generate_suite


def generate_suite(sizes: Iterable[str], num_instances: int, out_root: str | Path, *, seed_offset: int = 0) -> list[dict]:
    """Bodur et al. (2017) two-stage stochastic CFL instances (a_gen_data.generate_scfl_instance) with a proved MIP optimum and a matching certified Benders reference."""
    return _generate_suite("scfl", sizes=list(sizes), num_instances=num_instances, out_root=out_root, seed_offset=seed_offset)
