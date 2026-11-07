from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.utils.base_data import JsonDataMixin


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
