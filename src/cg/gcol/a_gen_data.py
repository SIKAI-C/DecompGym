"""Graph coloring instances: color every vertex so adjacent vertices differ, with the fewest colors."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


SIZE_CONFIG = {
    "small": {"n_range": (15, 18), "density_range": (0.3, 0.5)},
    "medium": {"n_range": (22, 26), "density_range": (0.3, 0.5)},
    "large": {"n_range": (30, 35), "density_range": (0.3, 0.5)},
}


@dataclass
class GraphColoringInstance(JsonDataMixin):
    instance_size: str
    n_vertices: int
    edges: list[list[int]]

    def __post_init__(self) -> None:
        for u, v in self.edges:
            if not (0 <= u < self.n_vertices and 0 <= v < self.n_vertices) or u == v:
                raise ValueError("invalid edge")

    @classmethod
    def from_json_file(cls, path: str | Path) -> GraphColoringInstance:
        return cast("GraphColoringInstance", super().from_json_file(path))


class GraphColoringGenerator(BaseInstanceGenerator[GraphColoringInstance]):
    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> GraphColoringInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        cfg = SIZE_CONFIG[size_label]
        rng = random.Random(seed)
        n = rng.randint(*cfg["n_range"])
        p = rng.uniform(*cfg["density_range"])
        edges = [[u, v] for u in range(n) for v in range(u + 1, n) if rng.random() < p]
        return GraphColoringInstance(size_label, n, edges)


def generate_gcol_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst = GraphColoringGenerator().generate_instance(seed=seed, size_label=size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return inst.to_dict()
