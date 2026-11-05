"""Uncapacitated multiple-allocation hub location (UMAHLP) instances (scale-up study).

Problem (Campbell 1994; Hamacher, Labbe, Nickel, Sonneborn 2004; Contreras, Cordeau, Laporte 2011): choose a nonempty
set H of hubs, hub k at fixed cost f_k, and route the flow W_ij of every ordered pair (i, j) along a path
i -> k -> m -> j through one or two open hubs (k = m allowed) at unit cost
    F_ijkm = chi * d_ik + alpha * d_km + delta * d_mj      (collection, transfer, distribution).
Hubs are uncapacitated and every pair may use its own path (multiple allocation), so for a fixed H the routing is
closed form: each pair takes its cheapest path over open hubs. Objective:
    sum_{k in H} f_k + sum_{i,j} W_ij * min_{k,m in H} F_ijkm.
Every ordered pair with W_ij > 0 counts, including i = j (the AP data has intra-node flows, routed i -> k -> k -> i at
cost (chi + delta) d_ik, or 0 when i is itself a hub). This convention reproduces every UMApHMP optimum of OR-Library
phub2 (tests/test_scaleup_hub.py).

Instance files. `load_hub(path)` reads a JSON file of one of two kinds:
- a self-contained data file (what solvers, agents and the verifier read), written by `write_instance`:
    {"family": "umahlp", "name", "n", "chi", "alpha", "delta", "f": [n], "w": [[n] * n], "d": [[n] * n],
     "coords": [[x, y] * n] or null, "source": {...provenance...}}
  with d the distance matrix already scaled (AP: Euclidean / 1000) and w[i][j] the flow from i to j;
- a small spec {"source": "AP", "n": 100, "fixed": "L50"} (or "CAB"), materialized from the OR-Library files in
  outputs/scaleup/raw/hub (key "raw_dir" overrides the folder). `python -m src.bd.hub.a_data --n 100 200 --fixed L50 T50
  --out-dir outputs/scaleup/instances/hub` writes <name>/data.json + meta.json per instance.

AP data (OR-Library phub1-3, Ernst and Krishnamoorthy): APdata200 holds 200 postal districts (coordinates, a 200 x 200
flow matrix with intra-district flows on the diagonal) and the unit costs chi = 3, alpha = 0.75, delta = 2;
n-node instances come from the authors' generate.c (ported exactly in `ap_aggregate`: 5 rows of boxes with equal node
counts by y, n/5 boxes per row by x, flows summed per box, coordinates the flow-weighted (in + out) average, values
written with printf "%f", i.e. rounded to 6 decimals). Distances are Euclidean / 1000 (OR-Library note).

Fixed costs (AP). For n in {10, 20, 25, 40, 50} the OR-Library files FcostL.NN / FcostT.NN (phub1, Ernst and
Krishnamoorthy's CSAHLP data; "T" tends to be harder) are used as they are (to our knowledge the "nL" / "nT" AP
convention of the UMAHLP literature with chi = 3, alpha = 0.75, delta = 2; not verified against a paper here).
Contreras, Cordeau and Laporte (2011, Sec. 5, read from the CIRRELT-2010-26 working paper) use "set-up costs of the
type loose (L)" for n = 25, 50, ..., 200 and refer to Contreras, Diaz, Fernandez (2011, INFORMS JoC) for them; that
file set is not in OR-Library (not available here), and their AP experiments use other unit costs (chi = delta = 1,
tau in {0.2, 0.5, 0.8}, distances scaled by TC in {2, 5, 10}). For other n this module offers our own documented rule
(kinds "L50" / "T50", `inherited_fixed_costs`); the literature convention for n > 50 remains an open question.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RAW_DIR = REPO / "outputs" / "scaleup" / "raw" / "hub"
N_ROWS = 5  # generate.c: number of y subdivisions
AP_FCOST_SIZES = (10, 20, 25, 40, 50)


@dataclass
class HubInstance:
    n: int
    w: np.ndarray        # (n, n) flow from i to j (diagonal = intra-node flow)
    d: np.ndarray        # (n, n) distances (already scaled)
    f: np.ndarray        # (n,) fixed hub opening costs
    chi: float = 3.0     # collection cost per unit distance and flow
    alpha: float = 0.75  # transfer (inter-hub) cost
    delta: float = 2.0   # distribution cost
    coords: np.ndarray | None = None
    name: str = ""
    source: dict = field(default_factory=dict)

    def validate(self) -> None:
        n = self.n
        if self.w.shape != (n, n) or self.d.shape != (n, n) or self.f.shape != (n,):
            raise ValueError("w and d must be n x n and f of length n")
        for name, arr in (("w", self.w), ("d", self.d), ("f", self.f)):
            if not np.all(np.isfinite(arr)) or np.any(arr < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not all(math.isfinite(v) and v >= 0 for v in (self.chi, self.alpha, self.delta)):
            raise ValueError("unit costs must be finite and nonnegative")
        if np.any(np.diag(self.d) != 0):
            raise ValueError("d must have a zero diagonal")

    def to_json(self) -> dict[str, Any]:
        return {"family": "umahlp", "name": self.name, "n": self.n, "chi": self.chi, "alpha": self.alpha, "delta": self.delta,
                "f": self.f.tolist(), "w": self.w.tolist(), "d": self.d.tolist(),
                "coords": None if self.coords is None else self.coords.tolist(), "source": self.source}


# ---------------------------------------------------------------- OR-Library files

def bundled_sections(path: str | Path) -> dict[str, str]:
    """Sections of an OR-Library bundle: a line 'FILE <name>' between star lines, then the content up to the next star line."""
    lines = Path(path).read_text().splitlines()
    out: dict[str, str] = {}
    i = 0
    while i < len(lines):
        m = re.match(r"^FILE (\S+)\s*$", lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not lines[j].startswith("*****"):
            j += 1
        k = j + 1
        while k < len(lines) and not lines[k].startswith("*****"):
            k += 1
        out[m.group(1)] = "\n".join(lines[j + 1:k]).strip()
        i = k
    return out


def parse_ap_nodes(text: str) -> dict[str, Any]:
    """The AP nodes-file format: n, n coordinate pairs, the n x n flow matrix, p, collection, transfer, distribution."""
    tok = text.split()
    n = int(tok[0])
    need = 1 + 2 * n + n * n + 4
    if len(tok) != need:
        raise ValueError(f"AP nodes file: {len(tok)} numbers, expected {need} for n = {n}")
    xy = np.array(tok[1:1 + 2 * n], dtype=float).reshape(n, 2)
    w = np.array(tok[1 + 2 * n:1 + 2 * n + n * n], dtype=float).reshape(n, n)
    p, c, t, d = tok[1 + 2 * n + n * n:]
    return {"n": n, "xy": xy, "w": w, "p": int(p), "chi": float(c), "alpha": float(t), "delta": float(d)}


def _f6(a: np.ndarray) -> np.ndarray:
    """printf("%f") rounding (6 decimals, correctly rounded like C's printf)."""
    return np.array([float(f"{v:.6f}") for v in np.asarray(a, dtype=float).ravel()]).reshape(np.shape(a))


def ap_aggregate(base: dict[str, Any], nn: int) -> dict[str, Any]:
    """Exact port of OR-Library generate.c (Ernst and Krishnamoorthy): aggregate the 200-node AP data to nn nodes."""
    NN = base["n"]
    if nn <= 0 or nn % N_ROWS:
        raise ValueError(f"n must be a positive multiple of {N_ROWS}")
    xy, w = base["xy"], base["w"]
    OD = w.sum(axis=1) + w.sum(axis=0)  # OD[p] += w[p][q]; OD[q] += w[p][q]
    # qsort by (y, x); generate.c aborts on two nodes at the same location, so the order is total.
    keys = {(float(x), float(y)) for x, y in xy}
    if len(keys) != NN:
        raise ValueError("two nodes at the same location")
    order = sorted(range(NN), key=lambda p: (xy[p, 1], xy[p, 0]))
    n_cols = nn // N_ROWS
    row_step = NN // N_ROWS
    if row_step // n_cols < 1:
        raise ValueError(f"n = {nn} exceeds the number of nodes per row times {N_ROWS}")
    new_i = np.empty(NN, dtype=np.int64)  # by original index
    row_start = 0
    for p in range(N_ROWS):
        row_end = row_start + row_step + (1 if p < NN % N_ROWS else 0)
        row = sorted(order[row_start:row_end], key=lambda q: (xy[q, 0], xy[q, 1]))
        order[row_start:row_end] = row
        size = row_end - row_start
        col_step = size // n_cols
        col_start = row_start
        for q in range(n_cols):
            col_end = col_start + col_step + (1 if q < size % n_cols else 0)
            for r in range(col_start, col_end):
                new_i[order[r]] = p * n_cols + q
            col_start = col_end
        row_start = row_end
    new_OD = np.zeros(nn)
    new_x = np.zeros(nn)
    new_y = np.zeros(nn)
    # Same accumulation order as generate.c (over the sorted points) for the coordinate sums.
    for p in order:
        b = new_i[p]
        new_x[b] += OD[p] * xy[p, 0]
        new_y[b] += OD[p] * xy[p, 1]
        new_OD[b] += OD[p]
    new_w = np.zeros((nn, nn))
    np.add.at(new_w, (new_i[:, None], new_i[None, :]), w)
    new_x /= new_OD
    new_y /= new_OD
    return {"n": nn, "xy": _f6(np.stack([new_x, new_y], axis=1)), "w": _f6(new_w), "chi": base["chi"], "alpha": base["alpha"],
            "delta": base["delta"], "members": new_i}


def euclidean(xy: np.ndarray, scale: float = 1e-3) -> np.ndarray:
    diff = xy[:, None, :] - xy[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2)) * scale
    np.fill_diagonal(d, 0.0)
    return d


def read_raw(raw_dir: str | Path = RAW_DIR) -> dict[str, str]:
    raw = Path(raw_dir)
    sec = bundled_sections(raw / "phub1.txt")
    for other in ("phub2.txt", "phub3.txt"):
        p = raw / other
        if p.exists():
            for k, v in bundled_sections(p).items():
                if k == "solutions":
                    k = f"solutions.{other.split('.')[0]}"
                sec.setdefault(k, v)
    sec["solutions.phub1"] = sec.pop("solutions")
    return sec


def ap_fixed_costs(sec: dict[str, str], base: dict[str, Any], agg: dict[str, Any], kind: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Fixed costs of an aggregated AP instance.

    kind "L" / "T": the OR-Library file FcostL.NN / FcostT.NN as it is (only n in {10, 20, 25, 40, 50}).
    kind "L50" / "T50" (any n): inherited from the 50-node file (`inherited_fixed_costs`); equal to "L" / "T" at n = 50.
    """
    n = agg["n"]
    if kind in ("L", "T"):
        key = f"Fcost{kind}.{n}"
        if key not in sec:
            raise ValueError(f"no OR-Library fixed-cost file {key}; available sizes {AP_FCOST_SIZES} (kind '{kind}50' "
                             "inherits the 50-node costs)")
        f = np.array(sec[key].split(), dtype=float)
        if len(f) != n:
            raise ValueError(f"{key}: {len(f)} values, expected {n}")
        return f, {"fixed": kind, "fixed_source": f"OR-Library phub1 {key}"}
    if kind in ("L50", "T50"):
        key = f"Fcost{kind[0]}.50"
        f50 = np.array(sec[key].split(), dtype=float)
        f, nested = inherited_fixed_costs(base, agg, ap_aggregate(base, 50), f50)
        return f, {"fixed": kind, "fixed_source": f"inherited from OR-Library phub1 {key} (ours; see a_data.inherited_fixed_costs)",
                   "nested_in_50": nested}
    raise ValueError(f"unknown fixed-cost kind {kind!r}")


def inherited_fixed_costs(base: dict[str, Any], agg: dict[str, Any], agg50: dict[str, Any], f50: np.ndarray) -> tuple[np.ndarray, bool]:
    """Our convention for AP sizes without an OR-Library fixed-cost file (no published file set is available here).

    Every aggregated node is a box of original districts; the 50-node file gives a cost per 50-node box. A node's
    cost is the flow-weighted (in + out flow of the original districts) mean of the 50-node costs of its districts:
        f_b = sum_{p in b} OD_p f50[box50(p)] / sum_{p in b} OD_p.
    For n in {100, 150, 200} every n-node box lies inside one 50-node box (generate.c splits each row of 40 districts
    in x order), so f_b is simply the cost of the enclosing 50-node box: a hub costs what a hub in that region costs at
    n = 50, and the n-node instance refines only the location choice. Returns (f, nested)."""
    w = base["w"]
    OD = w.sum(axis=1) + w.sum(axis=0)
    b, b50 = agg["members"], agg50["members"]
    num = np.zeros(agg["n"])
    den = np.zeros(agg["n"])
    np.add.at(num, b, OD * f50[b50])
    np.add.at(den, b, OD)
    parents = [np.unique(b50[b == k]) for k in range(agg["n"])]
    f = num / den
    for k, par in enumerate(parents):
        if len(par) == 1:
            f[k] = f50[par[0]]  # exactly the enclosing box's cost (no rounding from the weighted mean)
    return f, all(len(par) == 1 for par in parents)


def ap_instance(n: int, fixed: str = "L", raw_dir: str | Path = RAW_DIR) -> HubInstance:
    sec = read_raw(raw_dir)
    base = parse_ap_nodes(sec["APdata200"])
    agg = ap_aggregate(base, n)
    f, info = ap_fixed_costs(sec, base, agg, fixed)
    inst = HubInstance(n=n, w=agg["w"], d=euclidean(agg["xy"]), f=f, chi=agg["chi"], alpha=agg["alpha"], delta=agg["delta"],
                       coords=agg["xy"], name=f"AP{n}{fixed}",
                       source={"source": "AP", "n": n, "generator": "OR-Library generate.c (ported)", "distance": "euclidean/1000", **info})
    inst.validate()
    return inst


def cab_instance(n: int, alpha: float, fixed: float, raw_dir: str | Path = RAW_DIR) -> HubInstance:
    """CAB (O'Kelly, OR-Library phub4): the top-left n x n block (n in {10, 15, 20, 25}), chi = delta = 1, a given
    inter-hub factor alpha and a uniform fixed cost (the CAB convention of the UMAHLP literature). Flows and distances
    are taken as they are in the file (no normalization)."""
    rows = [ln.split() for ln in (Path(raw_dir) / "phub4.txt").read_text().splitlines()]
    rows = [r for r in rows if len(r) == 4 and r[0].isdigit() and r[1].isdigit()]
    N = 25
    if len(rows) != N * N:
        raise ValueError("phub4: expected 625 (i, j, W, C) rows")
    W = np.zeros((N, N))
    C = np.zeros((N, N))
    for i, j, wv, cv in rows:
        W[int(i) - 1, int(j) - 1] = float(wv)
        C[int(i) - 1, int(j) - 1] = float(cv.rstrip(";"))
    if n not in (10, 15, 20, 25):
        raise ValueError("CAB sizes are 10, 15, 20, 25")
    inst = HubInstance(n=n, w=W[:n, :n].copy(), d=C[:n, :n].copy(), f=np.full(n, float(fixed)), chi=1.0, alpha=float(alpha),
                       delta=1.0, name=f"CAB{n}", source={"source": "CAB", "n": n, "alpha": alpha, "fixed": fixed})
    inst.validate()
    return inst


# ---------------------------------------------------------------- JSON files

def from_json(obj: dict[str, Any], raw_dir: str | Path | None = None) -> HubInstance:
    if "w" in obj:
        n = int(obj["n"])
        inst = HubInstance(n=n, w=np.array(obj["w"], dtype=float), d=np.array(obj["d"], dtype=float),
                           f=np.array(obj["f"], dtype=float), chi=float(obj["chi"]), alpha=float(obj["alpha"]),
                           delta=float(obj["delta"]), coords=None if obj.get("coords") is None else np.array(obj["coords"], dtype=float),
                           name=str(obj.get("name", "")), source=obj.get("source") or {})
        inst.validate()
        return inst
    src = str(obj.get("source", "")).upper()
    raw = obj.get("raw_dir", raw_dir or RAW_DIR)
    if src == "AP":
        return ap_instance(int(obj["n"]), str(obj.get("fixed", "L")), raw)
    if src == "CAB":
        return cab_instance(int(obj["n"]), float(obj["alpha"]), float(obj["fixed"]), raw)
    raise ValueError("not a hub instance: need a data file (with 'w') or a spec with source AP or CAB")


def load_hub(path: str | Path) -> HubInstance:
    inst = from_json(json.loads(Path(path).read_text()))
    if not inst.name:
        inst.name = Path(path).stem
    return inst


def write_instance(inst: HubInstance, path: str | Path) -> None:
    Path(path).write_text(json.dumps(inst.to_json()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Write self-contained UMAHLP data files from the OR-Library AP data.")
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument("--fixed", nargs="+", default=["L"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for n in args.n:
        for kind in args.fixed:
            inst = ap_instance(n, kind, args.raw_dir)
            folder = out / inst.name  # same layout as the other scale-up families: <name>/data.json + meta.json
            folder.mkdir(parents=True, exist_ok=True)
            write_instance(inst, folder / "data.json")
            (folder / "meta.json").write_text(json.dumps({"name": inst.name, "family": "umahlp", **inst.source}, indent=1))
            print(folder / "data.json")


if __name__ == "__main__":
    main()
