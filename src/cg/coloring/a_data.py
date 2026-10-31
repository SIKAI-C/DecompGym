"""Vertex coloring instances in the DIMACS formats (scale-up study; minimize the number of colors).

Two file formats of the Second DIMACS Implementation Challenge (Johnson and Trick 1996), as distributed on M. Trick's
coloring pages (mat.tepper.cmu.edu/COLOR and COLOR02):
- ASCII ".col": comment lines "c ...", one problem line "p edge N M" (also "p col N M"), edge lines "e u v" with
  vertices 1..N; lines "n v w" (vertex weights) are ignored. M is the declared number of edge lines; several files
  list every edge in both directions (the DSJ graphs declare twice their edges), so duplicates are merged.
- binary ".col.b": a first line with the length L of the preamble, then L characters of ASCII preamble (comments and
  the "p" line), then the lower triangle of the adjacency matrix row by row: row i (0-based) occupies (i + 8) // 8
  bytes and bit j of the row (most significant bit first) is edge {i, j}, j <= i (the reader of the challenge's
  bin2asc.c). tests/test_scaleup_coloring.py checks that the binary files of the 1993 archive equal the ASCII files
  of the COLOR02 page edge for edge.
Vertices are numbered 0..N-1 internally (file vertex v is vertex v - 1); a coloring is a list of N integer labels and
its objective is the number of distinct labels. A self-loop makes a graph uncolorable and is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ColoringInstance:
    name: str
    n: int
    edges: np.ndarray  # (m, 2) int64, 0-based, u < v, unique, lexicographically sorted
    declared_edges: int | None = None
    _adj: np.ndarray | None = field(default=None, repr=False)
    _nbrs: list[np.ndarray] | None = field(default=None, repr=False)

    @property
    def m(self) -> int:
        return len(self.edges)

    @property
    def adjacency(self) -> np.ndarray:
        """(n, n) boolean adjacency matrix (symmetric, zero diagonal)."""
        if self._adj is None:
            A = np.zeros((self.n, self.n), dtype=bool)
            if self.m:
                A[self.edges[:, 0], self.edges[:, 1]] = True
                A[self.edges[:, 1], self.edges[:, 0]] = True
            self._adj = A
        return self._adj

    @property
    def neighbors(self) -> list[np.ndarray]:
        if self._nbrs is None:
            A = self.adjacency
            self._nbrs = [np.nonzero(A[v])[0] for v in range(self.n)]
        return self._nbrs

    @property
    def degree(self) -> np.ndarray:
        return self.adjacency.sum(axis=1).astype(np.int64)

    @property
    def density(self) -> float:
        return 2.0 * self.m / (self.n * (self.n - 1)) if self.n > 1 else 0.0


def from_edges(name: str, n: int, pairs, declared: int | None = None) -> ColoringInstance:
    """An instance from 0-based vertex pairs (duplicates and orientation merged; self-loops rejected)."""
    E = np.asarray(list(pairs) if not isinstance(pairs, np.ndarray) else pairs, dtype=np.int64).reshape(-1, 2)
    if len(E):
        if E.min() < 0 or E.max() >= n:
            raise ValueError(f"{name}: edge endpoint outside 0..{n - 1}")
        if (E[:, 0] == E[:, 1]).any():
            raise ValueError(f"{name}: self-loop (a graph with a loop has no proper coloring)")
        E = np.sort(E, axis=1)
        E = np.unique(E, axis=0)
    return ColoringInstance(name, int(n), E, declared)


def _problem_line(line: str, path) -> tuple[int, int]:
    parts = line.split()
    if len(parts) < 4 or parts[0] != "p":
        raise ValueError(f"{path}: bad problem line {line!r}")
    return int(parts[2]), int(parts[3])


def load_col_ascii(path: str | Path) -> ColoringInstance:
    n = declared = None
    pairs: list[tuple[int, int]] = []
    for raw in Path(path).read_text(encoding="latin-1").splitlines():
        line = raw.strip()
        if not line or line[0] == "c":
            continue
        tag = line[0]
        if tag == "p":
            if n is not None:
                raise ValueError(f"{path}: second problem line")
            n, declared = _problem_line(line, path)
        elif tag == "e":
            if n is None:
                raise ValueError(f"{path}: edge before the problem line")
            parts = line.split()
            u, v = int(parts[1]), int(parts[2])
            if not (1 <= u <= n and 1 <= v <= n):
                raise ValueError(f"{path}: edge {u} {v} outside 1..{n}")
            pairs.append((u - 1, v - 1))
        elif tag in ("n", "x", "d", "v"):
            continue  # vertex weights or other optional descriptors: not part of the coloring problem
        else:
            raise ValueError(f"{path}: unexpected line {line!r}")
    if n is None:
        raise ValueError(f"{path}: no problem line")
    return from_edges(Path(path).name.removesuffix(".col"), n, np.array(pairs, dtype=np.int64).reshape(-1, 2), declared)


def load_col_binary(path: str | Path) -> ColoringInstance:
    data = Path(path).read_bytes()
    nl = data.index(b"\n")
    length = int(data[:nl].decode("ascii").strip())
    preamble = data[nl + 1:nl + 1 + length].decode("latin-1")
    body = data[nl + 1 + length:]
    n = declared = None
    for line in preamble.splitlines():
        line = line.strip()
        if line.startswith("p"):
            n, declared = _problem_line(line, path)
    if n is None:
        raise ValueError(f"{path}: no problem line in the preamble")
    need = sum((i + 8) // 8 for i in range(n))
    if len(body) < need:
        raise ValueError(f"{path}: binary block has {len(body)} bytes, {need} expected")
    pairs = []
    pos = 0
    for i in range(n):
        k = (i + 8) // 8
        bits = np.unpackbits(np.frombuffer(body, dtype=np.uint8, count=k, offset=pos))[:i + 1]  # MSB first
        pos += k
        js = np.nonzero(bits)[0]
        if len(js):
            pairs.append(np.stack([np.full(len(js), i), js], axis=1))
    E = np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)
    return from_edges(Path(path).name.removesuffix(".col.b"), n, E, declared)


def load_col(path: str | Path) -> ColoringInstance:
    """Reads a DIMACS coloring instance (".col" ASCII or ".col.b" binary)."""
    return load_col_binary(path) if str(path).endswith(".b") else load_col_ascii(path)


def write_col(path: str | Path, inst: ColoringInstance, comment: str = "") -> Path:
    """Writes an instance as an ASCII .col file (each edge once, u < v, 1-based)."""
    lines = [f"c {comment}"] if comment else []
    lines.append(f"p edge {inst.n} {inst.m}")
    lines += [f"e {u + 1} {v + 1}" for u, v in inst.edges.tolist()]
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)
