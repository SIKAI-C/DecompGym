from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


# -----------------------
# Instance generation
# -----------------------

# The paper's generator (grid points, Euclidean costs, f in [10, 20]); kept for provenance, superseded on 2026-11-05
# by the Koerkel-Ghosh recipe below (`generate_ufl_instance`).
PAPER_SIZE_CONFIG = {
    "small": {
        "m_range": (5, 10),  # facilities
        "n_range": (15, 30),  # customers
    },
    "medium": {
        "m_range": (10, 20),
        "n_range": (30, 60),
    },
    "large": {
        "m_range": (20, 40),
        "n_range": (60, 120),
    },
}


@dataclass
class UFLInstance(JsonDataMixin):
    """Container for an Uncapacitated Facility Location instance."""

    m: int  # number of facilities
    n: int  # number of customers
    Xc: np.ndarray | None  # (n, 2) customer coords; None for non-geometric (Koerkel-Ghosh) instances
    Xf: np.ndarray | None  # (m, 2) facility coords; None for non-geometric instances
    f: np.ndarray  # (m,) fixed opening costs
    D: np.ndarray  # (m, n) shipping costs
    instance_info: dict[str, Any] | None = None  # recipe provenance (Koerkel-Ghosh instances only)

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
        return cast("UFLInstance", super().from_dict({"Xc": None, "Xf": None, "instance_info": None, **payload}))

    @classmethod
    def from_json_file(cls, path: str | Path) -> UFLInstance:
        # Narrow return type for convenience
        return cast("UFLInstance", super().from_json_file(path))


class UFLGenerator(BaseInstanceGenerator[UFLInstance]):
    """Synthetic UFL generator mirroring the tutorial recipe."""

    def __init__(self, *, seed: int = 1804) -> None:
        self.default_seed = seed

    def generate_instance(
        self,
        m: int = 20,
        n: int = 100,
        *,
        seed: int | None = None,
        grid_param: int = 1024,
    ) -> UFLInstance:
        rng = np.random.default_rng(seed if seed is not None else self.default_seed)

        Xc = rng.integers(0, grid_param + 1, size=(n, 2)) / float(grid_param)
        Xf = rng.integers(-1, 22, size=(m, 2)) / 20.0
        f = rng.integers(10, 21, size=m).astype(float)

        diff = Xf[:, None, :] - Xc[None, :, :]
        D = np.linalg.norm(diff, axis=2) * 10.0

        return UFLInstance(m=m, n=n, Xc=Xc, Xf=Xf, f=f, D=D)

    def _ensure_ax(self, ax: plt.Axes | None) -> plt.Axes:
        if ax is not None:
            return ax
        _, ax_new = plt.subplots()
        return ax_new

    def plot_instance(self, inst: UFLInstance, output_path: str | Path | None = None) -> None:
        ax = self._ensure_ax(None)
        ax.set_title("UFL instance")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.33)

        ax.scatter(inst.Xf[:, 0], inst.Xf[:, 1], marker="s", s=70, label="Facilities", alpha=0.6)
        ax.scatter(inst.Xc[:, 0], inst.Xc[:, 1], marker="o", s=40, label="Customers", color="tab:red")
        ax.legend(loc="upper left")

        if output_path is None:
            plt.show()
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
            plt.close(ax.figure)

    def plot_solution(
        self,
        inst: UFLInstance,
        result: dict[str, Any],
        *,
        output_path: str | Path | None = None,
        open_tol: float = 0.5,
    ) -> None:
        """Plot using the finalized result dict from baseline or Benders."""
        vars_dict = result.get("final_vars", {}) or {}
        master_meta = result.get("master_metadata", {}) or {}
        sub_meta = result.get("sub_metadata", {}) or {}

        if "x_assignment" in sub_meta:
            x_arr = np.asarray(sub_meta["x_assignment"], dtype=float).reshape(inst.m, inst.n)
        elif "x_assignment" in master_meta:
            x_arr = np.asarray(master_meta["x_assignment"], dtype=float).reshape(inst.m, inst.n)
        else:
            x_arr = np.zeros((inst.m, inst.n), dtype=float)
            for i in range(inst.m):
                for j in range(inst.n):
                    x_arr[i, j] = float(vars_dict.get(f"x_{i}_{j}", 0.0) or 0.0)

        y_arr = np.asarray([float(vars_dict.get(f"y_{i}", 0.0) or 0.0) for i in range(inst.m)], dtype=float)

        ax = self._ensure_ax(None)
        ax.set_title("UFL solution")
        ax.set_ylim(-0.1, 1.33)

        for i in range(inst.m):
            for j in range(inst.n):
                if x_arr[i, j] > 1e-6:
                    ax.plot(
                        [inst.Xc[j, 0], inst.Xf[i, 0]],
                        [inst.Xc[j, 1], inst.Xf[i, 1]],
                        color="black",
                        linewidth=0.5,
                        alpha=min(0.95, 0.25 + x_arr[i, j]),
                    )

        open_mask = y_arr >= open_tol
        ax.scatter(
            inst.Xf[~open_mask, 0],
            inst.Xf[~open_mask, 1],
            marker="s",
            s=60,
            color="lightblue",
            label="Facilities (closed)",
        )
        ax.scatter(
            inst.Xf[open_mask, 0],
            inst.Xf[open_mask, 1],
            marker="s",
            s=60,
            color="darkblue",
            label="Facilities (open)",
        )
        ax.scatter(
            inst.Xc[:, 0],
            inst.Xc[:, 1],
            marker="o",
            s=35,
            color="tab:red",
            label="Customers",
        )
        ax.legend(loc="upper left")

        if output_path is None:
            plt.show()
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
            plt.close(ax.figure)


def generate_ufl_paper_instance(
    seed: int,
    size: str = "small",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    The paper's recipe: sample (m, n) from size-specific ranges, generate a UFL instance,
    optionally write it to output_dir/data.json, and return the data dict.
    """
    if size not in PAPER_SIZE_CONFIG:
        raise ValueError(f"Unknown size '{size}'. Expected one of {list(PAPER_SIZE_CONFIG)}.")

    cfg = PAPER_SIZE_CONFIG[size]
    rng = np.random.default_rng(seed)

    m = int(rng.integers(cfg["m_range"][0], cfg["m_range"][1] + 1))
    n = int(rng.integers(cfg["n_range"][0], cfg["n_range"][1] + 1))

    gen = UFLGenerator(seed=seed)
    inst = gen.generate_instance(m=m, n=n, seed=seed)
    data = inst.to_dict()

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")

    return data


# -----------------------
# Koerkel-Ghosh recipe (small-scale benchmark generator since 2026-11-05)
# -----------------------

KG_SIZE_CONFIG = {"small": {"n": 50}, "medium": {"n": 70}, "large": {"n": 90}}  # n = m; calibrated 2026-11-05 on pilot seeds >= 100
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
      so that the monolithic MIP proves optimality within 60 s on one thread;
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
    inst = UFLInstance(m=n, n=n, Xc=None, Xf=None, f=np.asarray(F, dtype=float), D=np.asarray(C, dtype=float), instance_info=info)
    data = inst.to_dict()
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
    return data


# -----------------------
# Extra-large scale-up variants (exploration; the paper sizes above are unchanged)
# -----------------------

XL_CONFIG = {
    # The paper recipe, scaled up; fixed costs grow with n/m so that a similar share of sites opens.
    "xl": {"m_range": (150, 200), "n_range": (600, 1000)},
    # Koerkel-Ghosh type B: non-metric uniform costs, known to be hard for monolithic MIP.
    "xl_kg": {"mn_range": (250, 400), "cost_range": (1000, 2000), "fixed_range": (1000, 2000)},
}


def generate_ufl_xl_instance(seed: int, variant: str = "xl", output_dir: str | Path | None = None) -> dict[str, Any]:
    """Generate an extra-large UFL instance for the scale-up study."""
    if variant not in XL_CONFIG:
        raise ValueError(f"Unknown variant '{variant}'. Expected one of {list(XL_CONFIG)}.")
    cfg = XL_CONFIG[variant]
    rng = np.random.default_rng(seed)
    if variant == "xl":
        m = int(rng.integers(cfg["m_range"][0], cfg["m_range"][1] + 1))
        n = int(rng.integers(cfg["n_range"][0], cfg["n_range"][1] + 1))
        inst = UFLGenerator(seed=seed).generate_instance(m=m, n=n, seed=seed)
        # The paper's large tier has about 30 sites and 90 customers.
        inst.f = np.round(inst.f * (n / 90.0) * (30.0 / m), 2)
    else:
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
