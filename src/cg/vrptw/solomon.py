"""Solomon's (1987) 56 VRPTW instances with 100 customers, vendored as the original text files (data only).

Source: M. M. Solomon, "Algorithms for the vehicle routing and scheduling problems with time window constraints",
Operations Research 35(2):254-265, 1987; files as distributed on the SINTEF TOP page
(https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/, `solomon-100.zip`, folder `In/`). The copies in
`solomon/` are byte-identical to that archive; `solomon/SHA256SUMS` records their hashes, checked on every load.

File layout: name line, "VEHICLE" block (NUMBER = 25, CAPACITY), "CUSTOMER" block with one row per node
"CUST NO. XCOORD. YCOORD. DEMAND READY TIME DUE DATE SERVICE TIME", node 0 = the depot. All values are integers.
Classes: R1 (r101-r112), R2 (r201-r211), C1 (c101-c109), C2 (c201-c208), RC1 (rc101-rc108), RC2 (rc201-rc208);
type-1 classes have a short horizon and capacity 200; type-2 classes have a long horizon and capacity 700 (C2) or
1000 (R2, RC2). R = uniformly random geography, C = clustered, RC = mixed.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path


SOLOMON_DIR = Path(__file__).resolve().parent / "solomon"
CLASSES = ("R1", "R2", "C1", "C2", "RC1", "RC2")
CLASS_FILES = {
    "R1": [f"r1{k:02d}" for k in range(1, 13)],
    "R2": [f"r2{k:02d}" for k in range(1, 12)],
    "C1": [f"c1{k:02d}" for k in range(1, 10)],
    "C2": [f"c2{k:02d}" for k in range(1, 9)],
    "RC1": [f"rc1{k:02d}" for k in range(1, 9)],
    "RC2": [f"rc2{k:02d}" for k in range(1, 9)],
}


@dataclass(frozen=True)
class SolomonFile:
    name: str
    vehicles: int
    capacity: int
    rows: tuple[tuple[int, int, int, int, int, int], ...]  # (x, y, demand, ready, due, service) of nodes 0..100


def _hashes() -> dict[str, str]:
    lines = (SOLOMON_DIR / "SHA256SUMS").read_text().split("\n")
    return {name: digest for digest, name in (line.split() for line in lines if line.strip())}


@lru_cache(maxsize=None)
def load_solomon(name: str) -> SolomonFile:
    path = SOLOMON_DIR / f"{name}.txt"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _hashes()[path.name]:
        raise ValueError(f"{path}: hash differs from solomon/SHA256SUMS")
    lines = [line.split() for line in raw.decode("ascii").splitlines()]
    numeric = [row for row in lines if row and all(tok.lstrip("-").isdigit() for tok in row)]
    vehicles, capacity = (int(v) for v in numeric[0])
    rows = [tuple(int(v) for v in row[1:]) for row in numeric[1:]]
    if [int(row[0]) for row in numeric[1:]] != list(range(101)) or any(len(r) != 6 for r in rows):
        raise ValueError(f"{path}: expected nodes 0..100 with six values each")
    return SolomonFile(name, vehicles, capacity, tuple(rows))  # type: ignore[arg-type]


def class_of(name: str) -> str:
    return next(c for c, files in CLASS_FILES.items() if name in files)
