"""Bin packing with conflicts (BPPC) instances (scale-up study).

File format (Fernandes Muritiba, Iori, Malaguti and Toth 2010; the BPPC set of the Bologna OR library, whose
0_format_description.txt reads "N C", then "i w a1 a2 ... ak" per item): the first line holds the number of items N
and the bin capacity C; then one line per item: its number i (1..N), its integer weight w and the items a1..ak that
conflict with it. In the library files every list holds only items numbered above i (each conflict is listed once);
the loader accepts any listing and takes the union (the conflict graph is undirected). Items are numbered 1..N in
solutions (output contract, e_verifier); internally item k is file item k + 1.

Literature set ("t", "u"; 800 files BPPC_<c>_<d>_<k>.txt): class c = 1..8 is u120, u250, u500, u1000 (weights
U[20, 100], capacity 150) or t60, t120, t249, t501 (Falkenauer triplets, weights in [250, 500], capacity 1000; without
conflicts the optimum is n / 3 full bins); the conflict graph follows Gendreau, Laporte and Semet (2004): item values
v_i ~ U[0, 1] and a conflict iff (v_i + v_j) / 2 <= d / 10 for the file label d = 0..9, so the graphs are threshold
graphs (Bacci and Nicoloso 2017) with edge densities of about 0, 2, 8, 18, 33, 50, 68, 82, 92 and 98%; 10 instances
k per class and label; instance k of a class has the same weights for every d. Sadykov and Vanderbeck (2013) solved
all of them to optimality within an hour; per-instance optima are not published (group averages: Capua, Frota, Ochi
and Vidal 2017, arXiv:1705.08495, Table 1).

Arbitrary conflict graphs (Sadykov and Vanderbeck 2013, Sections 4.2-4.3; their files are not public, so the recipe
is re-run here with fixed seeds, see arbitrary_instance): "ua"/"ta" keep the items of BPPC_<c>_0_<k>; "da" draws new
items with weights U{500..2500} and capacity 10000 (n in {120, 250, 500}). Conflicts: uniformly random item pairs,
added if new, until the edge density reaches rho (|E| = ceil(rho n (n - 1) / 2)). Only random.Random(seed).random()
is used, whose stream Python guarantees across versions, so the files are reproducible anywhere.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LIT_CLASSES = {1: ("u", 120), 2: ("u", 250), 3: ("u", 500), 4: ("u", 1000),
               5: ("t", 60), 6: ("t", 120), 7: ("t", 249), 8: ("t", 501)}


@dataclass
class BPPCInstance:
    name: str
    capacity: int
    weights: np.ndarray            # (n,) int64, internal item k = file item k + 1
    adj: np.ndarray                # (n, n) bool, symmetric, False diagonal
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.weights)

    def edges(self) -> np.ndarray:
        """(m, 2) conflicting pairs i < j (internal numbering)."""
        I, J = np.nonzero(np.triu(self.adj, 1))
        return np.stack([I, J], axis=1)

    @property
    def num_conflicts(self) -> int:
        return int(np.triu(self.adj, 1).sum())

    @property
    def density(self) -> float:
        n = self.n
        return self.num_conflicts / (n * (n - 1) / 2) if n > 1 else 0.0

    @property
    def weight_bound(self) -> int:
        """ceil(total weight / capacity)."""
        return int(-(-int(self.weights.sum()) // self.capacity))


def parse_bppc(text: str, name: str = "bppc") -> BPPCInstance:
    """Parses the Muritiba et al. text format (see the module docstring)."""
    rows = [ln.split() for ln in text.splitlines() if ln.strip()]
    if not rows or len(rows[0]) != 2:
        raise ValueError(f"{name}: first line must be 'N C'")
    N, C = int(rows[0][0]), int(rows[0][1])
    if N <= 0 or C <= 0:
        raise ValueError(f"{name}: N and C must be positive")
    if len(rows) - 1 != N:
        raise ValueError(f"{name}: expected {N} item lines, got {len(rows) - 1}")
    w = np.zeros(N, dtype=np.int64)
    adj = np.zeros((N, N), dtype=bool)
    seen = set()
    for r in rows[1:]:
        if len(r) < 2:
            raise ValueError(f"{name}: item line {r} needs a number and a weight")
        i, wi = int(r[0]), int(r[1])
        if not 1 <= i <= N or i in seen:
            raise ValueError(f"{name}: item number {i} out of range or repeated")
        if not 0 < wi <= C:
            raise ValueError(f"{name}: item {i} has weight {wi} outside (0, C = {C}]")
        seen.add(i)
        w[i - 1] = wi
        for t in r[2:]:
            j = int(t)
            if not 1 <= j <= N or j == i:
                raise ValueError(f"{name}: item {i} lists an invalid conflict {j}")
            adj[i - 1, j - 1] = adj[j - 1, i - 1] = True
    return BPPCInstance(name, C, w, adj)


def load_bppc(path: str | Path) -> BPPCInstance:
    p = Path(path)
    return parse_bppc(p.read_text(), p.stem)


def format_bppc(inst: BPPCInstance) -> str:
    """The library format, each conflict listed once (with the higher-numbered item)."""
    lines = [f"{inst.n} {inst.capacity}"]
    for i in range(inst.n):
        nb = (np.flatnonzero(inst.adj[i, i + 1:]) + i + 2).tolist()
        lines.append(" ".join(str(v) for v in [i + 1, int(inst.weights[i])] + nb))
    return "\n".join(lines) + "\n"


def write_bppc(path: str | Path, inst: BPPCInstance) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(format_bppc(inst))
    return p


# ---------------------------------------------------------------- Sadykov-Vanderbeck arbitrary conflict graphs

def _rng(text: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big"))


def random_conflicts(n: int, rho: float, rnd: random.Random) -> list[tuple[int, int]]:
    """Uniformly random pairs (i, j), i != j, added if new, until |E| = ceil(rho n (n - 1) / 2) (Sadykov and
    Vanderbeck 2013, Section 4.2). Only rnd.random() is used."""
    target = math.ceil(rho * n * (n - 1) / 2 - 1e-9)
    have: set[tuple[int, int]] = set()
    order: list[tuple[int, int]] = []
    while len(order) < target:
        i, j = int(rnd.random() * n), int(rnd.random() * n)
        if i == j:
            continue
        e = (min(i, j), max(i, j))
        if e in have:
            continue
        have.add(e)
        order.append(e)
    return order


def arbitrary_instance(kind: str, n: int, rho10: int, k: int, raw_dir: str | Path | None = None) -> BPPCInstance:
    """Instance k of class kind ("ua", "ta" or "da") with n items and edge density rho10 / 10.
    "ua"/"ta": the weights of the literature file BPPC_<c>_0_<k>.txt (read from raw_dir); "da": weights U{500..2500},
    capacity 10000, drawn per (n, k) (the same items for every density)."""
    name = f"{kind}{n}_r{rho10}_k{k}"
    if kind in ("ua", "ta"):
        c = {v: c for c, v in LIT_CLASSES.items()}.get((kind[0], n))
        if c is None or raw_dir is None:
            raise ValueError(f"{name}: needs a literature class ({kind[0]}, {n}) and raw_dir")
        base = load_bppc(Path(raw_dir) / f"BPPC_{c}_0_{k}.txt")
        weights, C = base.weights.copy(), base.capacity
    elif kind == "da":
        rw = _rng(f"bppc-sv-da-weights:{n}:{k}")
        weights = np.array([500 + int(rw.random() * 2001) for _ in range(n)], dtype=np.int64)
        C = 10000
    else:
        raise ValueError(f"unknown kind {kind!r}")
    adj = np.zeros((n, n), dtype=bool)
    for i, j in random_conflicts(n, rho10 / 10, _rng(f"bppc-sv-arbitrary:{name}")):
        adj[i, j] = adj[j, i] = True
    return BPPCInstance(name, int(C), weights, adj, {"kind": kind, "rho": rho10 / 10, "k": k})


def literature_name(cls: int, d: int, k: int) -> str:
    return f"BPPC_{cls}_{d}_{k}"


def main() -> None:
    """Writes Sadykov-Vanderbeck-recipe instances: python -m src.cg.bppc.a_data --out DIR --raw DIR ua500_r7_k1 ..."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default=None, help="directory of the literature BPPC_*.txt files (for ua / ta)")
    ap.add_argument("names", nargs="+", help="<kind><n>_r<rho10>_k<k>, e.g. da250_r5_k3")
    args = ap.parse_args()
    for nm in args.names:
        head, r, kk = nm.split("_")
        kind, n = head[:2], int(head[2:])
        inst = arbitrary_instance(kind, n, int(r[1:]), int(kk[1:]), args.raw)
        p = write_bppc(Path(args.out) / f"{nm}.txt", inst)
        print(p, inst.n, inst.capacity, inst.num_conflicts, round(inst.density, 4))


if __name__ == "__main__":
    main()
