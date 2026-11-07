"""1D cutting stock instances generated with CUTGEN1 (Gau and Waescher 1995).

Source: T. Gau, G. Waescher (1995), "CUTGEN1: A problem generator for the standard one-dimensional cutting stock
problem", European Journal of Operational Research 84(3):572-579. CUTGEN1 is listed among the BPPLIB generators
(Delorme, Iori, Martello 2018, https://github.com/mdelorme2/BPPLIB). The procedure is pinned to the Java port of the
original FORTRAN code, https://github.com/akavrt/cutgen at commit bb082a9a9c1d045bea7837caafab8e722a728a7a
(ProblemGenerator.java, PseudoRandom.java), whose unit test
reproduces the two sample problems printed in the EJOR paper (p. 576; seed 1994, m = 10, L = 1000, v1 = 0.375,
v2 = 0.625, d = 50). tests/test_litgen_cg4.py checks this port against both of them.

CUTGEN1 procedure (descriptors m, L, v1, v2, d; one pseudo-random stream R in (0, 1)):
  1. lengths: for i = 1..m, l_i = int((v1 + (v2 - v1) R) L + R) with one R per item; sort non-increasingly;
  2. demands: draw R_1..R_m, D = m d; for i < m, d_i = max(1, int(D R_i / sum R + 0.5)); d_m = max(1, D - sum_{i<m} d_i);
     demand i belongs to the i-th longest length;
  3. merge equal lengths (their demands are added), so the instance may have fewer than m item types.
The stream is the Park-Miller "minimal standard" generator (IA = 16807, IM = 2^31 - 1, Schrage's method, R = seed / IM).

Classes: the 18 standard CUTGEN1 classes (Foerster and Waescher 2000, IJPR 38(7):1657-1676; tabulated e.g. by
Araujo, Poldi and Smith 2014, Pesquisa Operacional 34(2), Table 4): L = 1000, (v1, v2) in {(0.01, 0.2), (0.01, 0.8),
(0.2, 0.8)}, m in {10, 20, 40}, d in {10, 100}. This generator uses the cell (v1, v2) = (0.2, 0.8), d = 10 (classes
13, 15, 17 differ only in m). Only m, the size descriptor that CUTGEN1 itself varies, is set per label:
small/medium/large = m = 250/400/700 (round-2 recalibration, 2026-11-05; the published classes use m in {10, 20, 40},
CUTGEN1 is parametric in m, so these values go beyond the published grid; the round-1 table was m = 10/20/40). Because
lengths are integers in [200, 800], merging leaves at most 601 distinct types. The arc-flow MIP of src/cg/bp/arcflow.py
proved optimality in a few seconds at every pilot (litgen_suite rejects any instance where it does not, Threads = 1); the
sizes are set so that the full-process reference CG (time limit 60 s) takes a few seconds at small, about 8-18 s at
medium and about 25-36 s at large on the pilot seeds. The CUTGEN1 seed of an
instance is derived from the benchmark seed by cutgen1_seed (one Python random draw in [1, 2^31 - 2]); one problem is
drawn per seed (CUTGEN1 draws problems from one continuing stream; here every instance restarts the stream).
The paper-era generator (factor-scaled widths of 10-60 % of a random stock size) is preserved in the paper code base.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


# Problem config (CUTGEN1, see the module docstring)
CUTGEN1_CELL = {"name": "cutgen1_v0.2-0.8_d10", "L": 1000, "v1": 0.2, "v2": 0.8, "dbar": 10}

DEFAULT_CONFIG = {
    "stock_unit": "mm",
    "item_name_prefix": "PaperProduct_",
}

# Number of CUTGEN1 item types m per size label (round 2; beyond the published m = 10/20/40, see the module docstring).
SIZE_SCALING = {
    "small": {"m": 250},
    "medium": {"m": 400},
    "large": {"m": 700},
}

_IA, _IM, _IQ, _IR = 16807, 2147483647, 127773, 2836


class CutgenRandom:
    """Park-Miller minimal-standard generator exactly as in CUTGEN1 (Schrage's method)."""

    def __init__(self, seed: int) -> None:
        if not 0 < seed < _IM:
            raise ValueError("CUTGEN1 seed must lie in [1, 2^31 - 2]")
        self.seed = int(seed)

    def next(self) -> float:
        k = self.seed // _IQ
        self.seed = _IA * (self.seed - k * _IQ) - _IR * k
        if self.seed < 0:
            self.seed += _IM
        return self.seed / _IM


def cutgen1_problem(seed_or_rng: int | CutgenRandom, *, m: int, L: int, v1: float, v2: float, dbar: int) -> tuple[list[int], list[int]]:
    """One CUTGEN1 problem: (lengths, demands), lengths non-increasing and distinct."""
    rng = seed_or_rng if isinstance(seed_or_rng, CutgenRandom) else CutgenRandom(seed_or_rng)
    lengths = []
    for _ in range(m):
        r = rng.next()
        lengths.append(int((v1 + (v2 - v1) * r) * L + r))
    lengths.sort(reverse=True)
    rands = [rng.next() for _ in range(m)]
    total, rest = m * dbar, m * dbar
    demands = []
    for i in range(m - 1):
        demands.append(max(1, int(total * rands[i] / sum(rands) + 0.5)))
        rest -= demands[-1]
    demands.append(max(1, rest))
    out_l: list[int] = []
    out_d: list[int] = []
    for i in range(m):
        if i < m - 1 and lengths[i] == lengths[i + 1]:
            demands[i + 1] += demands[i]
        else:
            out_l.append(lengths[i])
            out_d.append(demands[i])
    return out_l, out_d


def cutgen1_seed(seed: int) -> int:
    """CUTGEN1 stream seed of a benchmark seed."""
    return random.Random(seed).randint(1, _IM - 1)


@dataclass
class CSPItemType(JsonDataMixin):
    id: int
    name: str
    size: int
    demand: int

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.size = int(self.size)
        self.demand = int(self.demand)
        if self.size <= 0:
            raise ValueError("size must be positive")
        if self.demand <= 0:
            raise ValueError("demand must be positive")


@dataclass
class CuttingStockInstance(JsonDataMixin):
    stock_size: int
    stock_unit: str
    item_types: list[CSPItemType]

    def __post_init__(self) -> None:
        self.stock_size = int(self.stock_size)
        if self.item_types and isinstance(self.item_types[0], dict):
            self.item_types = [CSPItemType(**it) for it in self.item_types]
        if self.stock_size <= 0:
            raise ValueError("stock_size must be positive")
        for it in self.item_types:
            if it.size > self.stock_size:
                raise ValueError(f"item {it.id} width {it.size} exceeds stock_size {self.stock_size}")

    @classmethod
    def from_json_file(cls, path: str | Path) -> CuttingStockInstance:
        return cast("CuttingStockInstance", super().from_json_file(path))


class CuttingStockGenerator(BaseInstanceGenerator[CuttingStockInstance]):
    def __init__(self, *, size_label: str = "medium", seed: int = 0) -> None:
        self.default_size_label = size_label
        self.default_seed = seed

    def generate_instance(
        self,
        *,
        seed: int | None = None,
        size_label: str | None = None,
    ) -> CuttingStockInstance:
        size = (size_label or self.default_size_label).lower()
        if size not in SIZE_SCALING:
            raise ValueError(f"Unknown size_label '{size}'. Expected one of {list(SIZE_SCALING)}.")

        cfg = DEFAULT_CONFIG
        cell = CUTGEN1_CELL
        bench_seed = seed if seed is not None else self.default_seed
        lengths, demands = cutgen1_problem(cutgen1_seed(bench_seed), m=SIZE_SCALING[size]["m"], L=cell["L"],
                                           v1=cell["v1"], v2=cell["v2"], dbar=cell["dbar"])
        stock_size = cell["L"]
        items = [
            CSPItemType(id=i, name=f"{cfg['item_name_prefix']}{i}", size=lengths[i], demand=demands[i])
            for i in range(len(lengths))
        ]

        return CuttingStockInstance(
            stock_size=stock_size,
            stock_unit=cfg["stock_unit"],
            item_types=items,
        )


def recipe_record(seed: int, size: str) -> dict[str, Any]:
    """Recipe metadata of a benchmark instance (for meta.json)."""
    return {"recipe": CUTGEN1_CELL["name"], "cutgen1_seed": cutgen1_seed(seed), "m": SIZE_SCALING[size]["m"],
            "L": CUTGEN1_CELL["L"], "v1": CUTGEN1_CELL["v1"], "v2": CUTGEN1_CELL["v2"], "dbar": CUTGEN1_CELL["dbar"]}


def generate_csp_instance(
    seed: int,
    size: str = "medium",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    gen = CuttingStockGenerator(size_label=size, seed=seed)
    inst = gen.generate_instance(seed=seed, size_label=size)
    data = inst.to_dict()
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return data
