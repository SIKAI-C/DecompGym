"""Railway crew scheduling (RCSP) instances of the van Rossum (2026) benchmark (scale-up study).

Problem (B.T.C. van Rossum, "Branch and Price for Railway Crew Scheduling: Benchmark Instances and Computational
Study", arXiv:2606.10081): choose a minimum-cost set of duties covering every task at least once. A duty is a path
SOURCE -> CONNECTION* -> SINK in the connection graph whose SOURCE and SINK use the same crew base, and it must
- last at most 9.5 h (MAX_DUTY = 34 200 s; the duty length is the sum of the connection durations), and
- contain a break connection a (a meal break of 30 min) with at most 5.5 h (MAX_WORK = 19 800 s) of work between the
  duty start and the break start, before(a) = (durations of the connections before a) + timeBeforeBreak(a), and at
  most 5.5 h between the break end and the duty end, after(a) = timeAfterBreak(a) + (durations of the connections
  after a, the SINK included).
Cost: the sum of the connection costs (1 per second of duty time plus 28 800 = 8 h on every SOURCE connection).
The rules are those of the benchmark README and of the paper's labelling algorithm (Appendix A: the break is taken on
every break arc that starts within 5.5 h of the duty start, so the latest such arc decides; for the "after" time this
is equivalent to "some break arc satisfies both limits", because after(a) does not increase along the duty).

Instance directory s{S}_t{T}/ (format: the benchmark's instances/README.md):
    tasks.txt                       index startTime endTime startStation endStation duration     (seconds)
    connections_{single,multi}_depot.txt
                                    index type startBlock endBlock timeBeforeBreak timeAfterBreak duration cost
                                    (type SOURCE: startBlock = depot station, endBlock = task; SINK: task -> depot
                                    station; CONNECTION: task -> task; -1 = no break on this connection)
    crewbases_{single,multi}_depot.txt, stations.txt   index name
    tracks.txt, geoStations.txt     not needed
Variants: "single" (one crew base) and "multi" (every station is a crew base; a duty still returns to its own base).

Spec. Solvers receive a path. `load_rcsp(path)` accepts
- a JSON spec file {"dir": "<instance directory>", "variant": "single" | "multi"} (a relative "dir" is resolved
  against the spec file's folder), which is what the calibration runner, the verifier and agents receive; or
- an instance directory, with the variant given as `load_rcsp(dir, variant="multi")`.
`write_spec(dir, variant, out)` writes such a spec.

Data conventions, checked on all 100 instances and both variants (2.2 M connections; see the module tests):
task ids are 0..n-1 and stations 0..S-1; duration = end - start for every task; SOURCE duration = the first task's
duration and SOURCE cost = duration + 28 800; SINK duration = cost = 0; CONNECTION (i, j) duration = cost =
end(j) - end(i) with start(j) >= end(i) and the same station in between; every break connection has
timeBeforeBreak = 300 and timeBeforeBreak + 1 800 + timeAfterBreak = duration; there is at most one connection per
(type, startBlock, endBlock), so a duty is fully described by its crew base and its task sequence; the multi-depot
file contains every single-depot connection with the same index. The duty length is therefore
end(last task) - start(first task). The loader checks only what the algorithms rely on: acyclicity (tasks in a
topological order), well-formed indices, no duplicate connection per pair, and `time_consistent` (the duty-length
offset of every task relative to the first task depends only on the two tasks), which the compact MIP needs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


MAX_DUTY = 34_200  # 9.5 h
MAX_WORK = 19_800  # 5.5 h of continuous work before and after the meal break
DUTY_FIXED_COST = 28_800  # 8 h, on every SOURCE connection
SOURCE, SINK, CONNECTION = 0, 1, 2
KINDS = {"SOURCE": SOURCE, "SINK": SINK, "CONNECTION": CONNECTION}


@dataclass
class RCSPInstance:
    name: str
    variant: str
    start: np.ndarray       # (n,) task start time [s]
    end: np.ndarray         # (n,) task end time [s]
    from_station: np.ndarray
    to_station: np.ndarray
    depots: list[int]       # crew base stations of the variant
    # Connections by original index (arrays of length max index + 1; missing indices have kind -1).
    c_kind: np.ndarray
    c_tail: np.ndarray      # depot station (SOURCE) or task
    c_head: np.ndarray      # task, or depot station (SINK)
    c_tbb: np.ndarray       # time before the break, -1 if none
    c_tab: np.ndarray       # time after the break, -1 if none
    c_dur: np.ndarray
    c_cost: np.ndarray
    stations: dict[int, str] = field(default_factory=dict)
    source_dir: str = ""
    max_duty: int = MAX_DUTY
    max_work: int = MAX_WORK

    def __post_init__(self) -> None:
        n = self.n
        kind = self.c_kind
        ok = kind >= 0
        if np.any(ok & ~np.isin(kind, (SOURCE, SINK, CONNECTION))):
            raise ValueError("unknown connection type")
        idx = np.nonzero(ok)[0]
        k, t, h = kind[idx], self.c_tail[idx], self.c_head[idx]
        dep = set(self.depots)
        for name, cond in (("SOURCE task", (k == SOURCE) & ((h < 0) | (h >= n))), ("SINK task", (k == SINK) & ((t < 0) | (t >= n))),
                           ("CONNECTION task", (k == CONNECTION) & ((t < 0) | (t >= n) | (h < 0) | (h >= n)))):
            if cond.any():
                raise ValueError(f"{name} index out of range")
        if any(int(s) not in dep for s in t[k == SOURCE]) or any(int(s) not in dep for s in h[k == SINK]):
            raise ValueError("SOURCE/SINK at a station that is not a crew base of the variant")
        brk = self.c_tbb[idx] >= 0
        if np.any(brk & (self.c_tab[idx] < 0)) or np.any(brk & (k != CONNECTION)):
            raise ValueError("break times must come in pairs and only on CONNECTION arcs")
        if np.any(self.c_dur[idx] < 0):
            raise ValueError("negative connection duration")
        # Arc lists.
        src = idx[k == SOURCE]
        snk = idx[k == SINK]
        arc = idx[k == CONNECTION]
        self.src_conn, self.src_depot, self.src_task = src, self.c_tail[src], self.c_head[src]
        self.snk_conn, self.snk_task, self.snk_depot = snk, self.c_tail[snk], self.c_head[snk]
        self.arc_conn, self.arc_tail, self.arc_head = arc, self.c_tail[arc], self.c_head[arc]
        self.arc_dur, self.arc_cost = self.c_dur[arc], self.c_cost[arc]
        self.arc_tbb, self.arc_tab = self.c_tbb[arc], self.c_tab[arc]
        self.arc_break = self.arc_tbb >= 0
        self.source_of = {(int(d), int(j)): int(c) for c, d, j in zip(src, self.src_depot, self.src_task)}
        self.sink_of = {(int(i), int(d)): int(c) for c, i, d in zip(snk, self.snk_task, self.snk_depot)}
        self.arc_of = {(int(i), int(j)): int(c) for c, i, j in zip(arc, self.arc_tail, self.arc_head)}
        if len(self.source_of) != len(src) or len(self.sink_of) != len(snk) or len(self.arc_of) != len(arc):
            raise ValueError("more than one connection for the same (type, startBlock, endBlock)")
        # Topological order of the task graph (Kahn); the benchmark's connections all go forward in time.
        indeg = np.bincount(self.arc_head, minlength=n)
        succ = [[] for _ in range(n)]
        for i, j in zip(self.arc_tail.tolist(), self.arc_head.tolist()):
            succ[i].append(j)
        tie = np.lexsort((np.arange(n), self.end, self.start))
        rank = np.empty(n, dtype=np.int64)
        rank[tie] = np.arange(n)
        import heapq
        heap = [(int(rank[v]), v) for v in range(n) if indeg[v] == 0]
        heapq.heapify(heap)
        order = []
        indeg = indeg.copy()
        while heap:
            _, v = heapq.heappop(heap)
            order.append(v)
            for w in succ[v]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    heapq.heappush(heap, (int(rank[w]), w))
        if len(order) != n:
            raise ValueError("the connection graph has a cycle")
        self.order = np.array(order, dtype=np.int64)
        self.pos = np.empty(n, dtype=np.int64)
        self.pos[self.order] = np.arange(n)
        # Time consistency: SOURCE duration = end(task) - start(task), CONNECTION duration = end(j) - end(i).
        # Then the elapsed duty time at task k of a duty starting with task s is end(k) - start(s) on every path.
        self.time_consistent = bool(
            np.all(self.c_dur[src] == self.end[self.src_task] - self.start[self.src_task])
            and np.all(self.arc_dur == self.end[self.arc_head] - self.end[self.arc_tail]))

    @property
    def n(self) -> int:
        return len(self.start)

    @property
    def n_arcs(self) -> int:
        return len(self.arc_conn)

    def summary(self) -> dict[str, Any]:
        return {"name": self.name, "variant": self.variant, "tasks": self.n, "depots": len(self.depots),
                "sources": len(self.src_conn), "sinks": len(self.snk_conn), "connections": self.n_arcs,
                "break_connections": int(self.arc_break.sum()), "time_consistent": self.time_consistent}


def _rows(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text().splitlines() if line.strip()]


def read_instance(directory: str | Path, variant: str) -> RCSPInstance:
    if variant not in ("single", "multi"):
        raise ValueError("variant must be 'single' or 'multi'")
    d = Path(directory)
    tasks = sorted((tuple(int(x) for x in r[:6]) for r in _rows(d / "tasks.txt")), key=lambda r: r[0])
    n = len(tasks)
    if [r[0] for r in tasks] != list(range(n)):
        raise ValueError("task indices must be 0..n-1")
    t = np.array(tasks, dtype=np.int64).reshape(n, 6)
    if np.any(t[:, 2] - t[:, 1] != t[:, 5]) or np.any(t[:, 5] < 0):
        raise ValueError("task duration must equal end - start and be nonnegative")
    rows = _rows(d / f"connections_{variant}_depot.txt")
    m = 1 + max(int(r[0]) for r in rows) if rows else 0
    arr = {k: np.full(m, -1, dtype=np.int64) for k in ("kind", "tail", "head", "tbb", "tab", "dur", "cost")}
    for r in rows:
        c = int(r[0])
        if arr["kind"][c] != -1:
            raise ValueError(f"duplicate connection index {c}")
        arr["kind"][c] = KINDS[r[1]]
        for key, val in zip(("tail", "head", "tbb", "tab", "dur", "cost"), r[2:8]):
            arr[key][c] = int(val)
    stations = {int(r[0]): r[1] if len(r) > 1 else "" for r in _rows(d / "stations.txt")} if (d / "stations.txt").exists() else {}
    depots = sorted(int(r[0]) for r in _rows(d / f"crewbases_{variant}_depot.txt"))
    return RCSPInstance(d.name, variant, t[:, 1].copy(), t[:, 2].copy(), t[:, 3].copy(), t[:, 4].copy(), depots,
                        arr["kind"], arr["tail"], arr["head"], arr["tbb"], arr["tab"], arr["dur"], arr["cost"],
                        stations=stations, source_dir=str(d))


def read_spec(path: str | Path) -> tuple[Path, str]:
    p = Path(path)
    spec = json.loads(p.read_text())
    if not isinstance(spec, dict) or "dir" not in spec or spec.get("variant") not in ("single", "multi"):
        raise ValueError('RCSP spec must be {"dir": ..., "variant": "single" | "multi"}')
    d = Path(spec["dir"])
    if not d.is_absolute():
        d = (p.parent / d).resolve()
    return d, spec["variant"]


def load_rcsp(path: str | Path, variant: str | None = None) -> RCSPInstance:
    p = Path(path)
    if p.is_dir():
        if variant is None:
            raise ValueError("an instance directory needs an explicit variant ('single' or 'multi')")
        return read_instance(p, variant)
    d, v = read_spec(p)
    if variant is not None and variant != v:
        raise ValueError(f"spec variant {v} differs from the requested variant {variant}")
    return read_instance(d, v)


def write_spec(directory: str | Path, variant: str, out: str | Path, relative: bool = False) -> Path:
    out = Path(out)
    d = Path(directory).resolve()
    if relative:
        import os
        d = Path(os.path.relpath(d, out.parent.resolve()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dir": str(d), "variant": variant}) + "\n")
    return out


def read_solution_file(path: str | Path) -> list[list[int]]:
    """The benchmark's solution format: one duty per line, as connection indices."""
    return [[int(x) for x in line.split()] for line in Path(path).read_text().splitlines() if line.strip()]


def duties_from_connections(inst: RCSPInstance, lines: list[list[int]]) -> list[dict[str, Any]]:
    """Connection-index duties -> [{"depot", "tasks"}]; raises ValueError on a malformed chain."""
    out = []
    for k, conns in enumerate(lines):
        if len(conns) < 2 or any(not 0 <= c < len(inst.c_kind) or inst.c_kind[c] < 0 for c in conns):
            raise ValueError(f"duty {k}: unknown connection index")
        kinds = [int(inst.c_kind[c]) for c in conns]
        if kinds[0] != SOURCE or kinds[-1] != SINK or any(x != CONNECTION for x in kinds[1:-1]):
            raise ValueError(f"duty {k}: not SOURCE -> CONNECTION* -> SINK")
        tasks = [int(inst.c_head[conns[0]])] + [int(inst.c_head[c]) for c in conns[1:-1]]
        for a, b in zip(conns, conns[1:]):
            if inst.c_head[a] != inst.c_tail[b]:
                raise ValueError(f"duty {k}: connections {a} and {b} are not consecutive")
        if inst.c_tail[conns[0]] != inst.c_head[conns[-1]]:
            raise ValueError(f"duty {k}: SOURCE and SINK use different crew bases")
        out.append({"depot": int(inst.c_tail[conns[0]]), "tasks": tasks})
    return out
