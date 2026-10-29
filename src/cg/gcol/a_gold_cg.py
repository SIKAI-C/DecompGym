from __future__ import annotations

from typing import Any

from src.cg.gcol.a_gen_data import GraphColoringInstance
from src.utils.partition_bnp import PartitionBranchAndPrice


class GraphColoringBranchAndPrice:
    """Mehrotra-Trick: independent-set columns, maximum-weight independent set pricing, same/differ branching."""

    def __init__(self, *, time_limit: float = 60.0, root_only: bool = False) -> None:
        self.time_limit, self.root_only = time_limit, root_only

    def run(self, path: str) -> dict[str, Any]:
        inst = GraphColoringInstance.from_json_file(path)
        res = PartitionBranchAndPrice(inst.n_vertices, [tuple(e) for e in inst.edges], time_limit=self.time_limit, root_only=self.root_only).run()
        res["color_classes"] = res.pop("groups")
        return res
