"""Instance suite of the lbbd_ps family: data, prompt, verified references and meta.json (src.utils.instance_suite)."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.utils.instance_suite import generate_suite as _generate_suite


def generate_suite(sizes: Iterable[str], num_instances: int, out_root: str | Path, *, seed_offset: int = 0) -> list[dict]:
    """Planning and cumulative scheduling instances (Hooker; Cire, Coban and Hooker) (a_gen_data.generate_lbbd_ps_instance) with a proved MIP optimum and a matching certified logic-based Benders reference."""
    return _generate_suite("lbbd_ps", sizes=list(sizes), num_instances=num_instances, out_root=out_root, seed_offset=seed_offset)
