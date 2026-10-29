from __future__ import annotations

from typing import Any

from src.cg.bppc.a_gen_data import BPPCInstance
from src.utils.partition_bnp import PartitionBranchAndPrice, solve_partition_mip


class BPPCBaseline:
    """Assignment MIP with symmetry breaking; the bin count is capped by first-fit decreasing with conflicts."""

    def __init__(self, *, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def run(self, path: str) -> dict[str, Any]:
        inst = BPPCInstance.from_json_file(path)
        conflicts = [tuple(e) for e in inst.conflicts]
        ub = len(PartitionBranchAndPrice(inst.n_items, conflicts, weights=inst.weights, capacity=inst.capacity).greedy())
        return solve_partition_mip(inst.n_items, conflicts, weights=inst.weights, capacity=inst.capacity, upper_bound=ub, params=self.params)
