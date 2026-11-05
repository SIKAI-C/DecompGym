from __future__ import annotations

from typing import Any

from src.cg.gcol.a_gen_data import GraphColoringInstance
from src.utils.partition_bnp import PartitionBranchAndPrice, solve_partition_mip


class GraphColoringBaseline:
    """Assignment MIP with symmetry breaking; the color count is capped by a greedy coloring."""

    def __init__(self, *, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def run(self, path: str) -> dict[str, Any]:
        inst = GraphColoringInstance.from_json_file(path)
        conflicts = [tuple(e) for e in inst.edges]
        ub = len(PartitionBranchAndPrice(inst.n_vertices, conflicts).greedy())
        return solve_partition_mip(inst.n_vertices, conflicts, upper_bound=ub, params=self.params)
