from __future__ import annotations

from typing import Any

from src.cg.bppc.a_gen_data import BPPCInstance
from src.utils.partition_bnp import PartitionBranchAndPrice


class BPPCBranchAndPrice:
    """Bin columns, knapsack-with-conflicts pricing, Ryan-Foster same/differ branching."""

    def __init__(self, *, time_limit: float = 60.0, root_only: bool = False) -> None:
        self.time_limit, self.root_only = time_limit, root_only

    def run(self, path: str) -> dict[str, Any]:
        inst = BPPCInstance.from_json_file(path)
        res = PartitionBranchAndPrice(inst.n_items, [tuple(e) for e in inst.conflicts], weights=inst.weights, capacity=inst.capacity,
                                      time_limit=self.time_limit, root_only=self.root_only).run()
        res["bins"] = res.pop("groups")
        return res
