"""Bin packing with conflicts: pack items into the fewest bins of one capacity; conflicting items may not share a bin.

Small-scale instances follow the arbitrary-conflict-graph classes of Sadykov and Vanderbeck (2013):
R. Sadykov, F. Vanderbeck, "Bin packing with conflicts: a generic branch-and-price algorithm", INFORMS Journal on
Computing 25(2):244-255 (classes "ua" and "da"; described in Capua, Frota, Ochi, Vidal 2017, arXiv:1705.08495, Sec. 4:
"(ta) and (ua) have the same characteristics as the instances of sets (t) and (u), but with an arbitrary conflict
graph ... edge densities from 10% to 90%"; "(d) and (da) were generated with item weights uniformly distributed in
[500, 2500], and a bin capacity of 10000 ... 120, 250 or 500 items").
  - UA items: the (u) class of Muritiba, Iori, Malaguti, Toth (2010), INFORMS JOC 22(3):401-415, i.e. Falkenauer (1996)
    "uniform" items: integer weights U{20, ..., 100}, capacity 150 (n in {120, 250, 500, 1000} in the literature).
  - DA items: integer weights U{500, ..., 2500}, capacity 10000 (n in {120, 250, 500}).
  - Arbitrary conflict graph with edge density rho: uniformly random item pairs, added if new, until
    |E| = ceil(rho n (n - 1) / 2) (the same convention as the scale-up files, src.cg.bppc.a_data.random_conflicts).
The ua/ta sets of the literature reuse the Muritiba item files and only replace the graph; here the UA items are drawn
directly from the Falkenauer-U distribution, since the published files have n >= 120.

Adaptations (named): (1) size: n is shrunk to 35/40/45 (SIZE_CONFIG) so that the monolithic MIP proves optimality
(MIPGap = 0, Threads = 1) within the 1800 s generation limit (litgen_suite rejects any instance where it does not).
Round-2 pilots (seeds 100-103, n = 45): MIP proved in 41.0/0.6/90.8/0.2 s, reference CG (full process) <= 1.5 s, gap 0;
n = 50 (seed 100, DA) did not prove within 1800 s, so n = 45 is the largest tested size whose pilot MIPs all proved. The binding limit is
the MIP proof, not the reference CG time; (2) regime mix: even benchmark seeds draw DA, odd seeds UA (seeds 0-2 give DA, UA, DA); (3) density:
one value of the published grid {0.1, ..., 0.9} per regime (DENSITY), fixed for every size and seed;
(4) randomness: Python's random.Random(f"bppc-litgen:{regime}:{size}:{seed}"), weights first, then the graph.
The recipe (regime, density, edge count) is recorded in meta.json, not in data.json.
The previous harder-cohort generator (G(n, p) with p ~ U[0.1, 0.4]) is kept in the harder-family branch history.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, cast

from src.cg.bppc.a_data import random_conflicts
from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


REGIMES = {
    "UA": {"weight_range": (20, 100), "capacity": 150},
    "DA": {"weight_range": (500, 2500), "capacity": 10000},
}

# One published density per regime (Sadykov-Vanderbeck grid 0.1-0.9).
DENSITY = {"UA": 0.5, "DA": 0.5}

SIZE_CONFIG = {
    "small": {"UA": 35, "DA": 35},
    "medium": {"UA": 40, "DA": 40},
    "large": {"UA": 45, "DA": 45},
}


def regime_for_seed(seed: int) -> str:
    return "DA" if seed % 2 == 0 else "UA"


@dataclass
class BPPCInstance(JsonDataMixin):
    instance_size: str
    n_items: int
    capacity: int
    weights: list[int]
    conflicts: list[list[int]]

    def __post_init__(self) -> None:
        if len(self.weights) != self.n_items or any(w <= 0 or w > self.capacity for w in self.weights):
            raise ValueError("weights must be positive and fit in a bin")
        for a, b in self.conflicts:
            if not (0 <= a < self.n_items and 0 <= b < self.n_items) or a == b:
                raise ValueError("invalid conflict")

    @classmethod
    def from_json_file(cls, path: str | Path) -> BPPCInstance:
        return cast("BPPCInstance", super().from_json_file(path))


def sv_instance(size_label: str, n: int, regime: str, rho: float, rnd: random.Random) -> BPPCInstance:
    """One Sadykov-Vanderbeck arbitrary-graph instance (weights first, then the graph, from one stream)."""
    cfg = REGIMES[regime]
    lo, hi = cfg["weight_range"]
    weights = [rnd.randint(lo, hi) for _ in range(n)]
    conflicts = sorted([a, b] for a, b in random_conflicts(n, rho, rnd))
    return BPPCInstance(size_label, n, cfg["capacity"], weights, conflicts)


def recipe_record(seed: int, size: str) -> dict[str, Any]:
    regime = regime_for_seed(seed)
    n = SIZE_CONFIG[size][regime]
    return {"recipe": f"sadykov-vanderbeck-2013-{regime.lower()}", "regime": regime, "density": DENSITY[regime],
            "n_items": n, "n_conflicts": math.ceil(DENSITY[regime] * n * (n - 1) / 2 - 1e-9)}


class BPPCGenerator(BaseInstanceGenerator[BPPCInstance]):
    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> BPPCInstance:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        seed = 0 if seed is None else seed
        regime = regime_for_seed(seed)
        rnd = random.Random(f"bppc-litgen:{regime}:{size_label}:{seed}")
        return sv_instance(size_label, SIZE_CONFIG[size_label][regime], regime, DENSITY[regime], rnd)


def generate_bppc_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst = BPPCGenerator().generate_instance(seed=seed, size_label=size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return inst.to_dict()
