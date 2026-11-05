"""UFL at scale: the Koerkel-Ghosh instance generator used by the scale-up (protocol v4, Stage 1 and holdout).

Published recipe (UflLib, "Koerkel-Ghosh" instances; https://algo.rwth-aachen.de/ufllib/KoerkelGhosh.shtml, after
Koerkel 1989 and Ghosh 2003, EJOR 150:150-162):
  - n = m (every customer is also a potential facility site);
  - connection costs c_ij drawn uniformly at random from the integers in [1000, 2000];
  - opening costs drawn uniformly from [100, 200] (class A), [1000, 2000] (class B) or [10000, 20000] (class C);
  - symmetric and asymmetric connection matrices (UflLib "gs" and "ga" sets).
UflLib ships n = m = 250, 500 and 750 only; the scale-up uses the SAME recipe at n = m = 3000 and 4000 (a size
extension, named as such in the paper). This module is the single pinned implementation for the scale-up and must
keep matching the published recipe above; the small-scale UFL generator (src/bd/ufl/a_gen_data.py, maintained
separately) is not used here.

Our conventions where the page is silent (frozen with the Stage 1 panel; any change changes every instance hash):
  - integers drawn with numpy.random.default_rng(seed).integers (inclusive bounds), the full N x N matrix D first,
    then the N opening costs;
  - symmetric: D = upper triangle (including the diagonal) of the drawn matrix mirrored to the lower triangle
    (c_ij = c_ji); asymmetric: the drawn matrix as is. The diagonal c_ii is a regular draw (not 0), as in the
    recipe's "connection costs drawn uniformly" (no self-service rule is stated);
  - lower-case class labels a / b / c for A / B / C;
  - name kg{s|a}{N}{class}_g{seed}; the file stores D[i][j] = cost of serving customer j from facility i.

File format (JSON): {"name", "m", "n", "f": [m], "D": [[n] * m], "Xc", "Xf" (placeholders), "meta": {...}}; read by
src/bd/ufl/g_bound_xl.load_instance and src/experiments/xl_certificate.ufl_view.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

KG_COST = (1000, 2000)
KG_FIXED = {"a": (100, 200), "b": (1000, 2000), "c": (10000, 20000)}
RECIPE_URL = "https://algo.rwth-aachen.de/ufllib/KoerkelGhosh.shtml"


def kg_name(N: int, fclass: str, seed: int, symmetric: bool = True) -> str:
    return f"kg{'s' if symmetric else 'a'}{N}{fclass}_g{seed}"


def make_kg(N: int, fclass: str, seed: int, symmetric: bool = True) -> dict:
    """Koerkel-Ghosh instance at n = m = N (module docstring): integer costs D_ij ~ U{1000..2000} (symmetric: D = D^T,
    upper triangle drawn), fixed costs ~ U{f_lo..f_hi} by class; numpy default_rng(seed), D first, then f. The output is
    byte-identical to the 11-03 probe generator (src/bd/ufl/g_bound_xl.make_kg before it moved here)."""
    if fclass not in KG_FIXED:
        raise ValueError(f"unknown Koerkel-Ghosh class {fclass!r} (a, b or c)")
    if N < 1:
        raise ValueError("N must be positive")
    rng = np.random.default_rng(seed)
    D = rng.integers(KG_COST[0], KG_COST[1] + 1, size=(N, N), dtype=np.int64)
    if symmetric:
        D = np.triu(D) + np.triu(D, 1).T
    lo, hi = KG_FIXED[fclass]
    f = rng.integers(lo, hi + 1, size=N, dtype=np.int64)
    return {"name": kg_name(N, fclass, seed, symmetric), "m": N, "n": N, "f": f.tolist(), "D": D.tolist(),
            "Xc": [[0, 0]] * N, "Xf": [[0, 0]] * N,
            "meta": {"generator": "Koerkel-Ghosh style (make_kg): D ~ U{1000..2000}, f ~ U by class; numpy default_rng(seed)",
                     "class": fclass, "symmetric": symmetric, "seed": seed, "N": N}}


def write_instance(obj: dict, path: str | Path) -> str:
    """Compact JSON (no spaces); returns the sha256 of the written bytes."""
    text = json.dumps(obj, separators=(",", ":"))
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def check_recipe(obj: dict) -> list[str]:
    """Recipe conformance of an instance object (used by the panel generation step): sizes, integer cost ranges, the
    class range of the opening costs, and symmetry when declared. Returns the problems (empty = conforming)."""
    p = []
    meta = obj.get("meta") or {}
    N = int(obj.get("m", -1))
    D = np.asarray(obj["D"], dtype=np.int64)
    f = np.asarray(obj["f"], dtype=np.int64)
    if obj.get("n") != N or D.shape != (N, N) or f.shape != (N,):
        p.append("shape: n = m = N and D is N x N")
    if D.size and (D.min() < KG_COST[0] or D.max() > KG_COST[1]):
        p.append("connection costs outside [1000, 2000]")
    lo, hi = KG_FIXED.get(meta.get("class"), (None, None))
    if lo is None:
        p.append("unknown class")
    elif f.size and (f.min() < lo or f.max() > hi):
        p.append(f"opening costs outside [{lo}, {hi}]")
    if meta.get("symmetric") and not np.array_equal(D, D.T):
        p.append("declared symmetric but D != D^T")
    return p
