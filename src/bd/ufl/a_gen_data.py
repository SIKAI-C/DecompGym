from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.utils.base_data import JsonDataMixin


# -----------------------
# Instance generation
# -----------------------

@dataclass
class UFLInstance(JsonDataMixin):
    """Container for an Uncapacitated Facility Location instance."""

    m: int  # number of facilities
    n: int  # number of customers
    Xc: np.ndarray | None  # (n, 2) customer coords; None for non-geometric (Koerkel-Ghosh) instances
    Xf: np.ndarray | None  # (m, 2) facility coords; None for non-geometric instances
    f: np.ndarray  # (m,) fixed opening costs
    D: np.ndarray  # (m, n) shipping costs

    def __post_init__(self) -> None:
        self.Xc = None if self.Xc is None else np.asarray(self.Xc, dtype=float)
        self.Xf = None if self.Xf is None else np.asarray(self.Xf, dtype=float)
        self.f = np.asarray(self.f, dtype=float)
        self.D = np.asarray(self.D, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        # Absent optional fields are omitted, so paper-format files keep exactly their original keys.
        return {key: value for key, value in super().to_dict().items() if value is not None}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> UFLInstance:
        return cast("UFLInstance", super().from_dict({"Xc": None, "Xf": None, **payload}))

    @classmethod
    def from_json_file(cls, path: str | Path) -> UFLInstance:
        # Narrow return type for convenience
        return cast("UFLInstance", super().from_json_file(path))


# -----------------------
# Koerkel-Ghosh recipe (small-scale benchmark generator since 2026-11-05)
# -----------------------

KG_SIZE_CONFIG = {"small": {"n": 60}, "medium": {"n": 75}, "large": {"n": 100}}  # n = m; calibrated 2026-11-06 (round 2) on pilot seeds >= 100
KG_SIZES = ("small", "medium", "large")
KG_CLASSES = {"a": (100, 200), "b": (1000, 2000), "c": (10000, 20000)}  # opening-cost ranges (F_LO, F_HI)
KG_COST_RANGE = (1000, 2000)  # connection-cost range (C_LO, C_HI)
KG_SRAND = 123456  # the published generator calls srand(123456) once and writes instances 1, 2, ... in sequence
MSVC_RAND_MAX = 32767


class MsvcRand:
    """The Microsoft C runtime rand(): holdrand = holdrand * 214013 + 2531011 (mod 2^32); rand = (holdrand >> 16) & 0x7fff.

    The published Koerkel-Ghosh generator calls srand/rand and itoa (a Microsoft C runtime function); with this
    generator our port reproduces the UflLib files ga/gs{250,500,750}{a,b,c}-k bit for bit (checked on 2026-11-05).
    """

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def rand(self) -> int:
        self.state = (self.state * 214013 + 2531011) & 0xFFFFFFFF
        return (self.state >> 16) & 0x7FFF


def _kg_draw(rng: MsvcRand, lo: int, hi: int) -> int:
    # C: (int)(rand()*1.0/RAND_MAX * (HI - LO) + LO), evaluated in double; truncation = floor for positive values.
    return int(rng.rand() * 1.0 / MSVC_RAND_MAX * (hi - lo) + lo)


def kggen(m: int, n: int, f_range: tuple[int, int], c_range: tuple[int, int], instance: int, *, symmetric: bool,
          srand: int = KG_SRAND) -> tuple[list[int], list[list[int]]]:
    """Port of the published Koerkel-Ghosh generator (UflLib, https://algo.rwth-aachen.de/ufllib/data/kggenasym.c and
    kggensym.c; both downloads are the same file, the symmetric loop is the commented-out alternative in it).

    srand(srand) once; then for instance = 1, 2, ...: for each facility i: F[i] = draw(F_LO, F_HI), followed by
    C[i][j] = draw(C_LO, C_HI) for j = 0..n-1 (asymmetric) or C[i][j] = C[j][i] = draw for j = i..n-1 (symmetric).
    Returns instance number `instance` (1-based) of that stream as (F, C)."""
    if instance < 1:
        raise ValueError("instance numbers start at 1")
    if symmetric and m != n:
        raise ValueError("the symmetric generator requires m = n")
    rng = MsvcRand(srand)
    for _ in range(instance):
        F = [0] * m
        C = [[0] * n for _ in range(m)]
        for i in range(m):
            F[i] = _kg_draw(rng, *f_range)
            for j in (range(i, n) if symmetric else range(n)):
                C[i][j] = _kg_draw(rng, *c_range)
                if symmetric:
                    C[j][i] = C[i][j]
    return F, C


def kg_schedule(seed: int, size: str) -> tuple[str, bool]:
    """Declared class/symmetry coverage: class = (a, b, c)[seed mod 3]; symmetric iff (size index + seed) is even.
    Evaluation seeds 0-2 then cover every class in every size, and every class in both variants across the sizes
    (small: sym a, asym b, sym c; medium: asym a, sym b, asym c; large: sym a, asym b, sym c)."""
    return "abc"[seed % 3], (KG_SIZES.index(size) + seed) % 2 == 0


def generate_ufl_instance(
    seed: int,
    size: str = "small",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Koerkel-Ghosh UFL instance (the small-scale benchmark recipe since 2026-11-05).

    Source: Koerkel (1989) EJOR 39:157-173; Ghosh (2003) EJOR 150:150-162; UflLib "Koerkel-Ghosh" instances and
    generator (https://algo.rwth-aachen.de/ufllib/KoerkelGhosh.shtml): n = m, connection costs uniform in
    [1000, 2000], opening costs uniform in [100, 200] (class A), [1000, 2000] (B) or [10000, 20000] (C), symmetric
    and asymmetric connection matrices. The draws follow the published C generator exactly (`kggen`: Microsoft C
    rand(), srand(123456), integer truncation, so values are integers in [LO, HI - 1] and HI only when rand() = 32767).

    Adaptations (named):
    - size: the library has n = m = 250/500/750; we keep every distribution and shrink n = m to KG_SIZE_CONFIG[size]
      with the calibration target that the reference Benders (60 s budget, one thread, full process) takes at most
      30 s (small, medium) and 45 s (large) on pilot seeds >= 100 (in-process pilot times plus full-process checks); the monolithic MIP only proves optimality (MIPGap 0,
      one thread, limit 1800 s);
    - instances: our seed s is instance number s + 1 of the generator stream for (n, class, symmetry), i.e. the file
      `<prefix><s + 1>` that the published program writes with `num_prob >= s + 1`;
    - class and symmetry per instance follow the declared schedule `kg_schedule` (not chosen by solver outcome);
    - the facility index and customer index of D[i][j] are the library's (row = facility i, column = customer j).
    """
    if size not in KG_SIZE_CONFIG:
        raise ValueError(f"Unknown size '{size}'. Expected one of {list(KG_SIZE_CONFIG)}.")
    n = int(KG_SIZE_CONFIG[size]["n"])
    fclass, symmetric = kg_schedule(seed, size)
    F, C = kggen(n, n, KG_CLASSES[fclass], KG_COST_RANGE, seed + 1, symmetric=symmetric)
    info = {"recipe": "koerkel-ghosh", "opening_cost_class": fclass.upper(), "symmetric": symmetric,
            "opening_cost_range": list(KG_CLASSES[fclass]), "connection_cost_range": list(KG_COST_RANGE),
            "generator": "UflLib kggen port (MSVC rand, srand 123456)", "generator_instance_number": seed + 1}
    inst = UFLInstance(m=n, n=n, Xc=None, Xf=None, f=np.asarray(F, dtype=float), D=np.asarray(C, dtype=float))
    data = inst.to_dict()
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
        # Recipe provenance stays outside data.json (agents receive only data.json and prompt.txt).
        (out_dir / "source.json").write_text(json.dumps(info, indent=2))
    return data


# -----------------------
# Extra-large scale-up variants (exploration)
# -----------------------

XL_CONFIG = {
    # Koerkel-Ghosh type B: non-metric uniform costs, known to be hard for monolithic MIP. (The former "xl" variant,
    # the paper-era grid recipe scaled up, was removed with that recipe; see the Git history.)
    "xl_kg": {"mn_range": (250, 400), "cost_range": (1000, 2000), "fixed_range": (1000, 2000)},
}


def generate_ufl_xl_instance(seed: int, variant: str = "xl_kg", output_dir: str | Path | None = None) -> dict[str, Any]:
    """Generate an extra-large UFL instance for the scale-up study."""
    if variant not in XL_CONFIG:
        raise ValueError(f"Unknown variant '{variant}'. Expected one of {list(XL_CONFIG)}.")
    cfg = XL_CONFIG[variant]
    rng = np.random.default_rng(seed)
    m = n = int(rng.integers(cfg["mn_range"][0], cfg["mn_range"][1] + 1))
    D = rng.integers(cfg["cost_range"][0], cfg["cost_range"][1] + 1, size=(m, n)).astype(float)
    f = rng.integers(cfg["fixed_range"][0], cfg["fixed_range"][1] + 1, size=m).astype(float)
    # Coordinates are placeholders for plotting only; costs are not geometric.
    Xc = rng.random((n, 2))
    Xf = rng.random((m, 2))
    inst = UFLInstance(m=m, n=n, Xc=Xc, Xf=Xf, f=f, D=D)
    data = inst.to_dict()
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return data


def load_ufllib_instance(path: str | Path) -> UFLInstance:
    """Read a UflLib "simple format" file (e.g. Koerkel-Ghosh): `FILE:` line, `m n 0`, then per facility `index opening_cost c_1 ... c_n`."""
    lines = [line.split() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("FILE")]
    m, n = int(lines[0][0]), int(lines[0][1])
    rows = lines[1:1 + m]
    if len(rows) != m or any(len(r) != n + 2 for r in rows):
        raise ValueError(f"{path}: expected {m} facility rows with {n + 2} fields")
    f = np.array([float(r[1]) for r in rows])
    D = np.array([[float(v) for v in r[2:]] for r in rows])
    # Coordinates are placeholders; Koerkel-Ghosh costs are not geometric.
    return UFLInstance(m=m, n=n, Xc=np.zeros((n, 2)), Xf=np.zeros((m, 2)), f=f, D=D)


def read_ufllib_reference(path: str | Path) -> float:
    """The objective value at the end of a UflLib `.opt` / `.bub` solution file."""
    return float(Path(path).read_text().split()[-1])
