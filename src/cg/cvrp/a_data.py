"""Capacitated vehicle routing (CVRP) instances in the CVRPLIB / TSPLIB format (Uchoa et al. 2017 "X" set; scale-up
study).

File layout: header lines "KEY : value" (NAME, COMMENT, TYPE = CVRP, DIMENSION, EDGE_WEIGHT_TYPE = EUC_2D, CAPACITY),
then NODE_COORD_SECTION (node id, x, y), DEMAND_SECTION (node id, demand) and DEPOT_SECTION (depot node id, closed by
-1), and EOF. Conventions (CVRPLIB; reproduced exactly on the bundled best-known solutions by tests/test_scaleup_cvrp.py):
- one depot (node 1 in every X instance); the other nodes are the customers, numbered 1..n in file order, so customer i
  is node i + 1 of the file; the .sol files use this numbering ("Route #k: i j ...", then "Cost c");
- distances are Euclidean distances rounded to the nearest integer (TSPLIB nint, floor(d + 0.5); with integer
  coordinates a distance is never exactly halfway); the objective is the total distance, without vehicle costs;
- the fleet is unlimited: the "k" of an instance name is the minimum number of vehicles, not a fleet size (several
  best-known solutions use more routes, e.g. X-n101-k25 has 26 and X-n195-k51 has 53).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CVRPInstance:
    name: str
    coords: np.ndarray   # (n + 1, 2); row 0 is the depot, row i is customer i
    demand: np.ndarray   # (n + 1,) integers; demand[0] = 0
    capacity: int

    @property
    def n(self) -> int:
        return len(self.coords) - 1

    def distances(self) -> np.ndarray:
        """(n + 1) x (n + 1) integer distances (TSPLIB nint of the Euclidean distance), depot first."""
        p = self.coords.astype(float)
        d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(axis=2))
        return np.floor(d + 0.5).astype(np.int64)

    @property
    def min_vehicles(self) -> int:
        """ceil(total demand / capacity), a lower bound on the number of routes."""
        return int(math.ceil(int(self.demand.sum()) / self.capacity))


def _sections(path: str | Path) -> tuple[dict[str, str], dict[str, list[list[str]]]]:
    header: dict[str, str] = {}
    sections: dict[str, list[list[str]]] = {}
    current = None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "EOF":
            break
        key = line.split(":")[0].strip().upper() if ":" in line else line.split()[0].upper()
        if key.endswith("_SECTION"):
            current = key
            sections[current] = []
            continue
        if ":" in line and current is None:
            k, v = line.split(":", 1)
            header[k.strip().upper()] = v.strip().strip('"')
            continue
        if current is None:
            raise ValueError(f"{path}: unexpected line {line!r}")
        sections[current].append(line.split())
    return header, sections


def load_vrp(path: str | Path) -> CVRPInstance:
    """Reads a CVRPLIB .vrp file (EUC_2D, one depot); see the module docstring for the numbering."""
    header, sec = _sections(path)
    if header.get("TYPE", "CVRP").upper() != "CVRP":
        raise ValueError(f"{path}: TYPE {header.get('TYPE')} is not CVRP")
    if header.get("EDGE_WEIGHT_TYPE", "").upper() != "EUC_2D":
        raise ValueError(f"{path}: only EDGE_WEIGHT_TYPE EUC_2D is supported")
    dim = int(header["DIMENSION"])
    Q = int(header["CAPACITY"])
    coords = {int(r[0]): (float(r[1]), float(r[2])) for r in sec["NODE_COORD_SECTION"]}
    dem = {int(r[0]): int(r[1]) for r in sec["DEMAND_SECTION"]}
    depots = [int(r[0]) for r in sec.get("DEPOT_SECTION", [["1"]]) if int(r[0]) != -1]
    if len(depots) != 1:
        raise ValueError(f"{path}: exactly one depot is supported, got {depots}")
    if len(coords) != dim or len(dem) != dim or sorted(coords) != list(range(1, dim + 1)):
        raise ValueError(f"{path}: DIMENSION {dim} does not match the coordinate / demand sections")
    dep = depots[0]
    order = [dep] + [v for v in range(1, dim + 1) if v != dep]
    q = np.array([dem[v] for v in order], dtype=np.int64)
    if q[0] != 0:
        raise ValueError(f"{path}: depot demand {q[0]} is not 0")
    if (q[1:] < 0).any():
        raise ValueError(f"{path}: negative demand")
    return CVRPInstance(header.get("NAME", Path(path).stem), np.array([coords[v] for v in order]), q, Q)


def load_sol(path: str | Path) -> tuple[list[list[int]], int]:
    """Routes (customer numbers as in the module docstring) and the stated cost of a CVRPLIB .sol file."""
    routes: list[list[int]] = []
    cost = None
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.lower().startswith("route"):
            routes.append([int(t) for t in s.split(":", 1)[1].split()])
        elif s.lower().startswith("cost"):
            cost = int(round(float(s.split()[1])))
    if cost is None:
        raise ValueError(f"{path}: no Cost line")
    return routes, cost
