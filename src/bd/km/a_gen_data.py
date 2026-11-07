from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


# The paper's generator (separate uniform facility/customer sets, 10 x Euclidean costs); kept for provenance,
# superseded on 2026-11-05 by the TSPLIB-derived p-median recipe below (`generate_km_instance`).
PAPER_SIZE_CONFIG = {
    "small": {"m_range": (5, 10), "n_range": (10, 20), "k_frac": (0.2, 0.3)},
    "medium": {"m_range": (10, 15), "n_range": (20, 30), "k_frac": (0.2, 0.35)},
    "large": {"m_range": (15, 20), "n_range": (30, 40), "k_frac": (0.15, 0.35)},
}


@dataclass
class KMedianInstance(JsonDataMixin):
    """k-median instance data."""

    m: int  # facilities
    n: int  # customers
    k: int  # facilities to open
    Xc: np.ndarray  # (n, 2)
    Xf: np.ndarray  # (m, 2)
    D: np.ndarray  # (m, n) assignment cost

    def __post_init__(self) -> None:
        self.Xc = np.asarray(self.Xc, dtype=float)
        self.Xf = np.asarray(self.Xf, dtype=float)
        self.D = np.asarray(self.D, dtype=float)
        if not (0 < self.k <= self.m):
            raise ValueError("Require 0 < k <= m.")
        if self.Xc.shape != (self.n, 2):
            raise ValueError("Xc shape mismatch")
        if self.Xf.shape != (self.m, 2):
            raise ValueError("Xf shape mismatch")
        if self.D.shape != (self.m, self.n):
            raise ValueError("D shape mismatch")

    @classmethod
    def from_json_file(cls, path: str | Path) -> KMedianInstance:
        return cast("KMedianInstance", super().from_json_file(path))


class KMedianGenerator(BaseInstanceGenerator[KMedianInstance]):
    """Synthetic k-median generator matching the tutorial recipe."""

    def __init__(self, *, seed: int = 1804, grid_param: int = 1024) -> None:
        self.default_seed = seed
        self.grid_param = grid_param

    def generate_instance(
        self,
        m: int = 20,
        n: int = 100,
        k: int = 5,
        *,
        seed: int | None = None,
    ) -> KMedianInstance:
        if k <= 0 or k > m:
            raise ValueError("Require 0 < k <= m.")

        rng = np.random.default_rng(seed if seed is not None else self.default_seed)

        Xc = rng.integers(0, self.grid_param + 1, size=(n, 2)) / float(self.grid_param)
        Xf = rng.integers(-1, 22, size=(m, 2)) / 20.0

        diff = Xf[:, None, :] - Xc[None, :, :]
        D = np.linalg.norm(diff, axis=2) * 10.0

        return KMedianInstance(m=m, n=n, k=k, Xc=Xc, Xf=Xf, D=D)

    def _ensure_fig_ax(self) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots()
        return fig, ax

    def plot_instance(self, inst: KMedianInstance, output_path: str | Path | None = None) -> None:
        fig, ax = self._ensure_fig_ax()
        ax.set_title("k-median instance")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.33)
        ax.scatter(inst.Xf[:, 0], inst.Xf[:, 1], marker="s", s=70, label="Facilities", alpha=0.6)
        ax.scatter(inst.Xc[:, 0], inst.Xc[:, 1], marker="o", s=40, color="tab:red", label="Customers")
        ax.legend(loc="upper left")

        if output_path is None:
            plt.show()
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
            plt.close(fig)

    def plot_solution(
        self,
        inst: KMedianInstance,
        result: dict[str, Any],
        *,
        output_path: str | Path | None = None,
        open_tol: float = 0.5,
    ) -> None:
        """Plot using finalized result dict from baseline or Benders."""
        vars_dict = result.get("final_vars", {}) or {}
        sub_meta = result.get("sub_metadata", {}) or {}
        master_meta = result.get("master_metadata", {}) or {}

        if "x_assignment" in sub_meta:
            x_arr = np.asarray(sub_meta["x_assignment"], dtype=float).reshape(inst.m, inst.n)
        elif "x_assignment" in master_meta:
            x_arr = np.asarray(master_meta["x_assignment"], dtype=float).reshape(inst.m, inst.n)
        else:
            x_arr = np.zeros((inst.m, inst.n), dtype=float)
            for i in range(inst.m):
                for j in range(inst.n):
                    x_arr[i, j] = float(vars_dict.get(f"x_{i}_{j}", 0.0) or 0.0)

        if "y_assignment" in sub_meta:
            y_arr = np.asarray(sub_meta["y_assignment"], dtype=float).reshape(inst.m)
        elif "y_assignment" in master_meta:
            y_arr = np.asarray(master_meta["y_assignment"], dtype=float).reshape(inst.m)
        else:
            y_arr = np.array([vars_dict.get(f"y_{i}", 0.0) or 0.0 for i in range(inst.m)], dtype=float)

        fig, ax = self._ensure_fig_ax()
        ax.set_title(f"k-median solution (k={inst.k})")
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
        ax.scatter(inst.Xf[~open_mask, 0], inst.Xf[~open_mask, 1], marker="s", s=60, color="lightblue", label="Closed")
        ax.scatter(inst.Xf[open_mask, 0], inst.Xf[open_mask, 1], marker="s", s=60, color="darkblue", label="Open")
        ax.scatter(inst.Xc[:, 0], inst.Xc[:, 1], marker="o", s=35, color="tab:red", label="Customers")
        ax.legend(loc="upper left")

        if output_path is None:
            plt.show()
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
            plt.close(fig)


def generate_km_paper_instance(
    seed: int,
    size: str = "small",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """The paper's recipe (superseded on 2026-11-05)."""
    if size not in PAPER_SIZE_CONFIG:
        raise ValueError(f"Unknown size '{size}'. Expected one of {list(PAPER_SIZE_CONFIG)}.")
    cfg = PAPER_SIZE_CONFIG[size]
    rng = np.random.default_rng(seed)
    m = int(rng.integers(cfg["m_range"][0], cfg["m_range"][1] + 1))
    n = int(rng.integers(cfg["n_range"][0], cfg["n_range"][1] + 1))
    frac = rng.uniform(cfg["k_frac"][0], cfg["k_frac"][1])
    k = max(1, min(m - 1, round(frac * m)))

    gen = KMedianGenerator(seed=seed)
    inst = gen.generate_instance(m=m, n=n, k=k, seed=seed)
    data = inst.to_dict()

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")

    return data


# -----------------------
# TSPLIB-derived p-median (small-scale benchmark recipe since 2026-11-05)
# -----------------------

TSP_SIZES = ("small", "medium", "large")
TSP_SIZE_CONFIG = {"small": {"n": 250}, "medium": {"n": 400}, "large": {"n": 525}}  # nodes; calibrated 2026-11-06 (round 2) on pilot seeds >= 100
TSP_PARENTS = ("rl1304", "fl1400", "u1432", "vm1748")  # Duran-Mateluna, Ales and Elloumi (2023), Table 3 ("medium")
TSP_PARENT_SHA256 = {  # of the uncompressed TSPLIB95 files shipped in src/bd/km/tsplib/*.tsp.gz
    "rl1304": "5ff734892f9500f46491340b0cd58e844a322a373ec6a07e026687a253e568be",
    "fl1400": "7afc62a19f3c691f13193fbc55865938097071370882174c926aff722db42a7c",
    "u1432": "3762b5ec208d8ba303863f6a1a0d2edbb4930edd9e4742b2046df07699199c2f",
    "vm1748": "7f7ee48f6af398c0a95cfa985f638a185268789bb191d6d31161ebcd465c7ca0",
}
TSP_P_VALUES = (5, 10, 20)  # the first three entries of the literature's p grid (5, 10, 20, 50, 100, ...)
TSPLIB_DIR = Path(__file__).resolve().parent / "tsplib"


def read_tsplib_euc2d(name: str, directory: str | Path = TSPLIB_DIR) -> np.ndarray:
    """Coordinates (DIMENSION x 2, file order) of a TSPLIB EUC_2D file `<name>.tsp.gz`; the sha256 must match."""
    import gzip
    import hashlib

    raw = gzip.decompress((Path(directory) / f"{name}.tsp.gz").read_bytes())
    if name in TSP_PARENT_SHA256 and hashlib.sha256(raw).hexdigest() != TSP_PARENT_SHA256[name]:
        raise ValueError(f"{name}.tsp does not match its recorded sha256")
    header, coords, in_coords = {}, [], False
    for line in raw.decode("ascii").splitlines():
        line = line.strip()
        if not line or line == "EOF":
            continue
        if line.startswith("NODE_COORD_SECTION"):
            in_coords = True
        elif not in_coords and ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
        elif in_coords:
            parts = line.split()
            coords.append((float(parts[1]), float(parts[2])))
    if header.get("EDGE_WEIGHT_TYPE") != "EUC_2D" or len(coords) != int(header["DIMENSION"]):
        raise ValueError(f"{name}: expected {header.get('DIMENSION')} EUC_2D coordinates")
    return np.asarray(coords, dtype=float)


def floor_euclidean(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """D[i][j] = floor(sqrt(dx * dx + dy * dy)) in IEEE double, dx = P_i,x - Q_j,x (the p-median convention)."""
    dx = P[:, None, 0] - Q[None, :, 0]
    dy = P[:, None, 1] - Q[None, :, 1]
    return np.floor(np.sqrt(dx * dx + dy * dy))


def tsp_schedule(seed: int, size: str) -> tuple[str, int]:
    """Declared parent/p coverage: parent = TSP_PARENTS[(3 * size index + seed) mod 4], p = TSP_P_VALUES[seed mod 3].
    Evaluation seeds 0-2 then use every p value in every size and every parent at least twice over the 9 instances."""
    return TSP_PARENTS[(3 * TSP_SIZES.index(size) + seed) % len(TSP_PARENTS)], TSP_P_VALUES[seed % len(TSP_P_VALUES)]


def generate_km_instance(
    seed: int,
    size: str = "small",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """TSPLIB-derived p-median instance (the small-scale benchmark recipe since 2026-11-05).

    Source: Duran-Mateluna, Ales and Elloumi (2023), "An efficient Benders decomposition for the p-median problem",
    European Journal of Operational Research (2023), arXiv:2111.13405, Section 4.1: TSPLIB (Reinelt 1991) instances, "the sites are at the same
    location as the clients and thus N = M", "the Euclidean distance rounded down to the nearest integer is used as
    distance"; p grid 5, 10, 20, 50, 100, ... (Tables 3-4). Same convention as Garcia, Labbe and Marin (2011).

    Adaptations (named):
    - TSPLIB is a collection, not a generator: an instance is a uniform random subset (without replacement,
      numpy default_rng(seed).choice, kept in file order) of n nodes of one parent EUC_2D file; parents are the
      four "medium" files of Duran-Mateluna et al. Table 3 (rl1304, fl1400, u1432, vm1748), disjoint from the
      scale-up evaluation files (d2103, pcb3038, fl3795, rl5934);
    - n is shrunk to TSP_SIZE_CONFIG[size] (below the parents' 1304-1748 nodes)
      with the calibration target that the reference Benders (60 s budget, one thread, full process) takes at most
      30 s (small, medium) and 45 s (large) on pilot seeds >= 100 (in-process pilot times plus full-process checks); the monolithic MIP only proves optimality (MIPGap 0,
      one thread, limit 1800 s);
    - p takes the first three values of the literature's p grid, (5, 10, 20), verbatim (larger grid values exceed
      what is meaningful at our n); parent and p follow the declared schedule `tsp_schedule`;
    - distances: floor of the Euclidean distance (the p-median adaptation); TSPLIB's own EUC_2D rule rounds to the
      nearest integer (nint), which is NOT used here. Duplicate coordinates in a parent (drilling files) are kept.
    Sites = customers = the sampled nodes (m = n, Xf = Xc, D symmetric with zero diagonal).
    """
    if size not in TSP_SIZE_CONFIG:
        raise ValueError(f"Unknown size '{size}'. Expected one of {list(TSP_SIZE_CONFIG)}.")
    n = int(TSP_SIZE_CONFIG[size]["n"])
    parent, p = tsp_schedule(seed, size)
    coords = read_tsplib_euc2d(parent)
    rng = np.random.default_rng(seed)
    nodes = np.sort(rng.choice(len(coords), size=n, replace=False))
    X = coords[nodes]
    info = {"recipe": "tsplib-derived-p-median", "parent": parent, "parent_sha256": TSP_PARENT_SHA256[parent],
            "parent_nodes_1based": [int(v) + 1 for v in nodes], "p": p, "distance": "floor(euclidean)",
            "sites_equal_customers": True}
    inst = KMedianInstance(m=n, n=n, k=p, Xc=X, Xf=X.copy(), D=floor_euclidean(X, X))
    data = inst.to_dict()
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
        # Recipe provenance (parent file and node list) stays outside data.json.
        (out_dir / "source.json").write_text(json.dumps(info, indent=2))
    return data
