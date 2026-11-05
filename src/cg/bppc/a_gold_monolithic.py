from __future__ import annotations

from typing import Any

from src.cg.bppc.a_gen_data import BPPCInstance
from src.utils.partition_bnp import PartitionBranchAndPrice, solve_partition_mip


class BPPCBaseline:
    """Assignment MIP with symmetry breaking (items relabeled by non-increasing weight); the bin count is capped by first-fit decreasing with conflicts."""

    def __init__(self, *, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def run(self, path: str) -> dict[str, Any]:
        inst = BPPCInstance.from_json_file(path)
        conflicts = [tuple(e) for e in inst.conflicts]
        ub = len(PartitionBranchAndPrice(inst.n_items, conflicts, weights=inst.weights, capacity=inst.capacity).greedy())
        # Relabel items by non-increasing weight so the "item i uses bins 0..i" symmetry breaking starts with the
        # largest items (standard for assignment BPP models); groups are mapped back to the original item ids.
        order = sorted(range(inst.n_items), key=lambda i: (-inst.weights[i], i))
        new_id = {old: new for new, old in enumerate(order)}
        result = solve_partition_mip(inst.n_items, [(new_id[a], new_id[b]) for a, b in conflicts],
                                     weights=[inst.weights[i] for i in order], capacity=inst.capacity, upper_bound=ub,
                                     params=self.params)
        result["groups"] = [sorted(order[i] for i in group) for group in result["groups"]]
        return result
