"""Independent verifier for vertex colorings (DIMACS instances, scale-up study). Standard library only.

Output contract:
    {"status": "optimal" | "feasible", "method": str, "objective_value": number, "coloring": [label, ...]}
with one integer label per vertex, in the vertex order of the instance file (entry k is vertex k + 1). Checks, from
the instance file only (ASCII ".col": "p edge N M" and "e u v" lines; binary ".col.b": a preamble-length line, the
ASCII preamble, then the lower triangle of the adjacency matrix, row i in (i + 8) // 8 bytes, most significant bit
first): the coloring has exactly N integer labels (booleans rejected); no edge joins two vertices with the same label;
the objective (the number of distinct labels) matches the claim within tol * max(1, |objective|); the claim is a
finite number; status is "optimal" or "feasible" and method is a string.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def read_graph(path: str | Path) -> tuple[int, set[tuple[int, int]]]:
    """(N, edges as 0-based pairs (u, v) with u < v)."""
    edges: set[tuple[int, int]] = set()
    if str(path).endswith(".b"):
        data = Path(path).read_bytes()
        nl = data.index(b"\n")
        length = int(data[:nl].decode("ascii").strip())
        preamble = data[nl + 1:nl + 1 + length].decode("latin-1")
        body = data[nl + 1 + length:]
        n = None
        for line in preamble.splitlines():
            parts = line.split()
            if parts and parts[0] == "p":
                n = int(parts[2])
        if n is None:
            raise ValueError("no problem line")
        pos = 0
        for i in range(n):
            k = (i + 8) // 8
            row = body[pos:pos + k]
            if len(row) < k:
                raise ValueError("binary block too short")
            pos += k
            for j in range(i + 1):
                if row[j >> 3] & (0x80 >> (j & 7)):
                    if i == j:
                        raise ValueError("self-loop")
                    edges.add((j, i))
        return n, edges
    n = None
    for raw in Path(path).read_text(encoding="latin-1").splitlines():
        parts = raw.split()
        if not parts or parts[0] == "c":
            continue
        if parts[0] == "p":
            n = int(parts[2])
        elif parts[0] == "e":
            if n is None:
                raise ValueError("edge before the problem line")
            u, v = int(parts[1]) - 1, int(parts[2]) - 1
            if not (0 <= u < n and 0 <= v < n) or u == v:
                raise ValueError(f"bad edge {raw!r}")
            edges.add((min(u, v), max(u, v)))
    if n is None:
        raise ValueError("no problem line")
    return n, edges


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> dict[str, Any]:
    n, edges = read_graph(data_path)
    try:
        out = json.loads(Path(output_path).read_text())
    except (OSError, json.JSONDecodeError) as err:
        return {"ok": False, "issues": [f"unreadable output: {err}"]}
    if not isinstance(out, dict):
        return {"ok": False, "issues": ["output must be a JSON object"]}
    col = out.get("coloring")
    if not isinstance(col, list):
        return {"ok": False, "issues": ["coloring missing"]}
    issues: list[str] = []
    if len(col) != n:
        issues.append(f"coloring has {len(col)} labels, the graph has {n} vertices")
    if not all(isinstance(c, int) and not isinstance(c, bool) for c in col):
        issues.append("labels must be integers")
    if issues:
        return {"ok": False, "issues": issues}
    bad = [(u, v) for u, v in edges if col[u] == col[v]]
    if bad:
        issues.append(f"{len(bad)} edges join vertices with the same label (e.g. {bad[0][0] + 1} and {bad[0][1] + 1})")
    total = len(set(col))
    claimed = out.get("objective_value")
    if isinstance(claimed, bool) or not isinstance(claimed, (int, float)) or not math.isfinite(claimed) or abs(claimed - total) > tol * max(1.0, abs(total)):
        issues.append(f"objective mismatch or non-finite: claimed {claimed}, recomputed {total}")
    if out.get("status") not in ("optimal", "feasible"):
        issues.append("status must be 'optimal' or 'feasible' for a returned coloring")
    if not isinstance(out.get("method"), str):
        issues.append("method missing")
    return {"ok": not issues, "issues": issues, "objective_value": float(total), "vertices": n, "edges": len(edges)}
