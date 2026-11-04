"""k-median (p-median) instances for the scale-up study (the paper's k-median family in this directory,
a_gen_data.py and friends, is unchanged; these `_xl` files are separate).

Problem (the paper's k-median family): choose exactly k of m candidate facilities and serve every customer j from
its cheapest open facility; minimize sum_j min_{i open} D_ij. (The paper's verifier also admits fractional
assignments x_ij <= y_i with sum_i x_ij = 1; for a 0/1 design the cheapest open facility is an optimal assignment,
so the value of a design is the expression above.)

Distances. An instance stores coordinates, not the m x n matrix:
    D_ij = r(s * sqrt(dx * dx + dy * dy)),   dx = xf_i - xc_j,  dy = yf_i - yc_j,
evaluated in IEEE double in exactly this order (numpy elementwise and the stdlib verifier give bit-identical values;
s = 1 skips the multiplication), with r = floor ("floor": the TSPLIB p-median convention of Garcia, Labbe and Marin
2011 and Duran-Mateluna, Ales and Elloumi 2023, Euclidean distance rounded down to an integer) or the identity
("none": the paper's generator, s = 10).

Relation to the paper's family (Codex review 46 #6): same optimization semantics (exactly k open, cheapest-open
service, total cost), different instance distribution and interface: the paper's generator uses separate facility
and customer sets with unrounded 10 x Euclidean costs and ships D, Xf, Xc in data.json; the XL tiers are TSPLIB point
sets with facilities = customers and floor(Euclidean) costs, shipped as coordinates plus the exact distance rule
(b_problem_xl.py). Published LB1 values (Duran-Mateluna et al.) are integer-reported Phase-1 bounds, not exact LP*
references; the frozen references are rigorous LP* intervals (scripts/phoenix/scaleup_km_stage1.py --reference).

Sources.
- TSPLIB (Reinelt 1991) EUC_2D coordinate files; facilities = customers = all nodes (m = n), as in the p-median
  literature. Optimal / best-known values (and the LP-relaxation bounds LB1 of the Benders Phase 1) for the probed
  (instance, p) pairs are transcribed from Duran-Mateluna, Ales and Elloumi (2023), Tables 3-4, in
  outputs/scaleup/references/km_references.json.
- `make_uniform`: the paper's generator recipe (src/bd/km/a_gen_data.py: customers on a 1025 x 1025 grid of the
  unit square, D = 10 x Euclidean distance, no rounding) at larger sizes with separate facility and customer sets
  (m != n). The paper's facility grid ({-1..21} / 20, 529 points) cannot hold hundreds of distinct facilities, so
  facilities are drawn from the same coordinate range [-0.05, 1.05] on the customer grid's resolution
  ({-51..1075} / 1024): our documented choice.

File format (JSON, `write_instance`):
    {"format": "kmedian-xl-v1", "name": str, "k": int, "rounding": "floor" | "none", "scale": number,
     "facilities": [[x, y], ...] (m), "customers": [[x, y], ...] (n), "meta": {...}}.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT = "kmedian-xl-v1"


@dataclass
class KMInstance:
    fac: np.ndarray      # (m, 2) facility coordinates
    cus: np.ndarray      # (n, 2) customer coordinates
    k: int               # number of facilities to open
    rounding: str = "floor"
    scale: float = 1.0
    name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def m(self) -> int:
        return len(self.fac)

    @property
    def n(self) -> int:
        return len(self.cus)

    def validate(self) -> None:
        if self.fac.ndim != 2 or self.fac.shape[1] != 2 or self.cus.ndim != 2 or self.cus.shape[1] != 2:
            raise ValueError("coordinates must be (count, 2) arrays")
        if self.m < 1 or self.n < 1:
            raise ValueError("empty instance")
        if not (1 <= int(self.k) <= self.m):
            raise ValueError(f"require 1 <= k <= m (k={self.k}, m={self.m})")
        if self.rounding not in ("floor", "none"):
            raise ValueError(f"unknown rounding {self.rounding!r}")
        if not (math.isfinite(self.scale) and self.scale > 0):
            raise ValueError("scale must be positive and finite")
        if not (np.all(np.isfinite(self.fac)) and np.all(np.isfinite(self.cus))):
            raise ValueError("non-finite coordinates")

    def distances(self) -> np.ndarray:
        """The (m, n) matrix D (float64; integral values with rounding='floor')."""
        return distance_block(self.fac, self.cus, self.rounding, self.scale)


def distance_block(fac: np.ndarray, cus: np.ndarray, rounding: str, scale: float) -> np.ndarray:
    dx = fac[:, 0][:, None] - cus[:, 0][None, :]
    dy = fac[:, 1][:, None] - cus[:, 1][None, :]
    D = np.sqrt(dx * dx + dy * dy)
    if scale != 1.0:
        D = scale * D
    if rounding == "floor":
        D = np.floor(D)
    return D


def integral_costs(D: np.ndarray) -> bool:
    """True if every distance is an integer below 2**52 (then all sums of distances are exact in float64)."""
    return bool(np.all(D == np.floor(D)) and float(D.max(initial=0.0)) < 2.0 ** 40)


class SortedCosts:
    """Per customer, the facilities in increasing order of distance (ties by index), restricted to the first
    L = m - k + 1: any design with k open facilities opens one of them, so the cheapest open facility of an integer
    design lies there; and for every y in [0, 1]^m with sum y = k the cumulative sum of y over them reaches 1, so the
    LP subproblem's greedy fill stops there too.
        order (n, L) int32, ds (n, L) float64 (ds[j, t] = D[order[j, t], j])."""

    def __init__(self, D: np.ndarray, k: int) -> None:
        m, n = D.shape
        self.m, self.n, self.k = m, n, k
        self.L = L = m - k + 1
        DT = np.ascontiguousarray(D.T)
        if L < m:
            part = np.argpartition(DT, L - 1, axis=1)[:, :L]
            vals = np.take_along_axis(DT, part, axis=1)
            o = np.lexsort((part, vals), axis=1)  # by distance, ties by facility index
            order = np.take_along_axis(part, o, axis=1)
        else:
            order = np.argsort(DT, axis=1, kind="stable")
        self.order = order.astype(np.int32)
        self.ds = np.take_along_axis(DT, self.order, axis=1)
        self.d1 = self.ds[:, 0].copy()                 # min_i D_ij
        self.theta_max = DT.max(axis=1)                # max_i D_ij


# ---------------------------------------------------------------- TSPLIB

def read_tsplib(path: str | Path) -> tuple[str, np.ndarray]:
    """EUC_2D TSPLIB file (plain or .gz): returns (name, (N, 2) coordinates in node order)."""
    p = Path(path)
    raw = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    lines = raw.decode("ascii", errors="strict").splitlines()
    name, dim, ewt, coords, in_coords = p.name.split(".")[0], None, None, [], False
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if in_coords:
            if s == "EOF" or not (s[0].isdigit() or s[0] in "+-"):
                break
            parts = s.split()
            if len(parts) != 3:
                raise ValueError(f"{path}: bad coordinate line {s!r}")
            coords.append((float(parts[1]), float(parts[2])))
            continue
        if ":" in s:
            key, val = (t.strip() for t in s.split(":", 1))
            if key == "NAME":
                name = val
            elif key == "DIMENSION":
                dim = int(val)
            elif key == "EDGE_WEIGHT_TYPE":
                ewt = val
        elif s == "NODE_COORD_SECTION":
            in_coords = True
    if ewt != "EUC_2D":
        raise ValueError(f"{path}: EDGE_WEIGHT_TYPE {ewt}, expected EUC_2D")
    if dim is None or len(coords) != dim:
        raise ValueError(f"{path}: {len(coords)} coordinates, DIMENSION {dim}")
    return name, np.array(coords, dtype=float)


def tsplib_instance(path: str | Path, k: int) -> KMInstance:
    name, xy = read_tsplib(path)
    inst = KMInstance(fac=xy.copy(), cus=xy.copy(), k=int(k), rounding="floor", scale=1.0, name=f"{name}_p{k}",
                      meta={"source": "TSPLIB (Reinelt 1991)", "tsplib": name, "sites_equal_customers": True,
                            "distance": "floor of the Euclidean distance (Garcia et al. 2011; Duran-Mateluna et al. 2023)"})
    inst.validate()
    return inst


# ---------------------------------------------------------------- generator (paper recipe, larger sizes)

def make_uniform(m: int, n: int, k: int, seed: int, grid: int = 1024) -> KMInstance:
    rng = np.random.default_rng(seed)
    cus = rng.integers(0, grid + 1, size=(n, 2)) / float(grid)
    lo, hi = -round(0.05 * grid), grid + round(0.05 * grid)
    fac = rng.integers(lo, hi + 1, size=(m, 2)) / float(grid)
    inst = KMInstance(fac=fac, cus=cus, k=int(k), rounding="none", scale=10.0, name=f"kmu{m}x{n}_k{k}_g{seed}",
                      meta={"source": "paper recipe (src/bd/km/a_gen_data.py) at larger sizes, m != n",
                            "generator": "customers integers(0, grid+1)/grid, facilities integers(-0.05 grid, 1.05 grid + 1)/grid; "
                                         "D = 10 x Euclidean; numpy default_rng(seed), customers first", "seed": seed})
    inst.validate()
    return inst


def random_instance(rng: np.random.Generator, m: int, n: int, k: int, integer: bool = True, same_sites: bool = False,
                    span: int = 12) -> KMInstance:
    """Small instances for tests: integer coordinates in [0, span] (many distance ties) with floor rounding, or real
    coordinates without rounding."""
    if same_sites and m != n:
        raise ValueError("same_sites requires m == n")
    if integer:
        cus = rng.integers(0, span + 1, size=(n, 2)).astype(float)
        fac = cus.copy() if same_sites else rng.integers(0, span + 1, size=(m, 2)).astype(float)
        rounding, scale = "floor", 1.0
    else:
        cus = rng.random((n, 2))
        fac = cus.copy() if same_sites else rng.random((m, 2))
        rounding, scale = "none", 10.0
    inst = KMInstance(fac=fac, cus=cus, k=int(k), rounding=rounding, scale=scale,
                      name=f"rand{m}x{n}_k{k}", meta={"generator": "random_instance (tests)"})
    inst.validate()
    return inst


# ---------------------------------------------------------------- JSON

def _num(v: float):
    v = float(v)
    return int(v) if v.is_integer() and abs(v) < 2 ** 53 else v


def instance_dict(inst: KMInstance) -> dict:
    return {"format": FORMAT, "name": inst.name, "k": int(inst.k), "rounding": inst.rounding, "scale": _num(inst.scale),
            "facilities": [[_num(a), _num(b)] for a, b in inst.fac], "customers": [[_num(a), _num(b)] for a, b in inst.cus],
            "meta": inst.meta}


def write_instance(inst: KMInstance, path: str | Path) -> str:
    """Write the JSON file (floats via repr: exact round trip); returns its sha256."""
    text = json.dumps(instance_dict(inst), separators=(",", ":"))
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def parse_instance(obj: dict, name: str = "") -> KMInstance:
    if obj.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} instance")
    inst = KMInstance(fac=np.array(obj["facilities"], dtype=float).reshape(-1, 2),
                      cus=np.array(obj["customers"], dtype=float).reshape(-1, 2), k=int(obj["k"]),
                      rounding=obj.get("rounding", "floor"), scale=float(obj.get("scale", 1.0)),
                      name=obj.get("name") or name, meta=obj.get("meta", {}))
    inst.validate()
    return inst


def load_instance(path: str | Path) -> KMInstance:
    return parse_instance(json.loads(Path(path).read_text()), name=Path(path).stem)


def main() -> None:
    """python -m src.bd.km.a_data_xl --tsp <file.tsp> --p 200 300 --out <dir>   (TSPLIB instances), or
    python -m src.bd.km.a_data_xl --uniform M N K --seeds 1 2 --out <dir>      (paper recipe, m != n)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tsp", nargs="*", default=[])
    ap.add_argument("--p", nargs="*", type=int, default=[])
    ap.add_argument("--uniform", nargs=3, type=int, action="append", default=[], metavar=("M", "N", "K"))
    ap.add_argument("--seeds", nargs="*", type=int, default=[1])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in a.tsp:
        for p in a.p:
            inst = tsplib_instance(f, p)
            print(json.dumps({"name": inst.name, "sha256": write_instance(inst, out / f"{inst.name}.json"), "m": inst.m}))
    for m, n, k in a.uniform:
        for s in a.seeds:
            inst = make_uniform(m, n, k, s)
            print(json.dumps({"name": inst.name, "sha256": write_instance(inst, out / f"{inst.name}.json")}))


if __name__ == "__main__":
    main()
