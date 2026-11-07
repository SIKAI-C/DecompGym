"""Railway crew scheduling (RCSP): small instances cut from the van Rossum (2026) benchmark by time windows.

Problem (B.T.C. van Rossum, "Branch and Price for Railway Crew Scheduling: Benchmark Instances and Computational
Study", arXiv:2606.10081, Section 2; semantics as in the scale-up RCSP): choose a minimum-cost set of duties covering
every task at least once. A duty is a path SOURCE -> CONNECTION* -> SINK in the connection graph whose SOURCE and
SINK use the same crew base, and it must
- last at most 9.5 h (34 200 s; the duty length is the sum of the connection durations, SOURCE and SINK included);
- contain a break connection a (timeBeforeBreak(a) >= 0; a 30-min meal break) with
  before(a) = (durations of the connections before a) + timeBeforeBreak(a) <= 5.5 h (19 800 s) and
  after(a) = timeAfterBreak(a) + (durations of the connections after a) <= 5.5 h.
Cost: the sum of the connection costs (1 per second of duty time plus 28 800 on every SOURCE connection).

Why a cut and not the paper's generator (option A). The benchmark generator (paper Section 4.1) chains a gravity
demand model, a line-planning MIP, a PESP timetable over 06:00-22:00 and greedy rolling-stock scheduling before it
builds tasks and connections; the line costs, line capacities, frequencies, PESP weights and the rolling-stock rules
are not specified well enough to reimplement, and even its smallest output (2 large stations, a full day) has 90+
tasks. Only its last two steps are fully specified, and the benchmark files already are their output. We therefore
cut sub-instances from published benchmark instances (option B) by a rule that ignores every published solution:
1. parent instance and variant: seeds < 100 (evaluation) use the evaluation parents EVAL_PARENTS[seed % 3], seeds
   >= 100 (calibration and development) use the disjoint DEV_PARENTS[seed % 3]; the variant is "multi" for odd seeds
   and "single" for even seeds (the paper's two variants, same tasks and task-task connections);
2. time window: a window of SIZE_CONFIG[size]["window_hours"] hours (8 / 10 / 16 h; longer windows let the 5.5 h break
   rules and, above 9.5 h, the duty-length rule exclude paths, though the size rule below may shorten them) whose start t0 is a uniformly random whole minute in [first task start, last task
   end - window] of the parent; keep the tasks that start and end inside the window. Structural size rule
   (declared; it reads no cost and no solution): while the kept tasks admit more than SIZE_CONFIG[size]
   ["max_duties"] feasible duties, the window end moves 15 minutes earlier. This bounds the duty enumeration of the
   60 s exact references, since the duty count varies by two orders of magnitude between parents;
3. keep every connection of the variant between two kept tasks and every SOURCE / SINK connection of a kept task,
   with all their fields (type, break times, duration, cost); nothing is recomputed;
4. postprocessing, as the paper's own postprocessing step (Section 4.1, "Postprocessing"): remove every task that is
   in no feasible duty of the sub-instance, rebuild, and repeat until every task is coverable (this also removes
   isolated tasks). A window that keeps no task is redrawn (declared rule; never triggered by the size table).
The published best-known duties are never read. The benchmark timetable is hourly periodic, so windows of one parent
differ little; the parents and variants supply the variety (a requested 16 h window spans almost the whole parent day,
about 06:00-22:45, before the duty cap shrinks it, so two development seeds with the same parent and variant can give the
same task set: 2 pairs among the 12 pilot seeds; the evaluation seeds 0, 1, 2 of one size use three different parents).
The requested 16 h window is shorter than every parent day except s3_t172 (15.95 h), whose whole day exceeds the
200 000-duty cap, so the window shrinks; provenance.json records the effective window, and the final instances were
checked to keep a proper subset of the parent's tasks.
Size calibration (round 2, 2026-11-05; every size setting was chosen from pilots on the development parents, seeds
100-111; full-process exact column generation with the 60 s budget, Threads = 1). Chronology and calibration exposure
(outputs/litgen_dev/round2_evidence/localrun/status_r2_clrp_rcsp.md, steps [16]-[23]):
- the LARGE setting, 16 h / 200 000 duties, was selected first ([16]-[18]), from development-parent pilots only;
- next, a diagnostic ran six pilot seeds (100-105) of 16 h / 200 000 on the EVALUATION parents
  (outputs/litgen_dev/rcsp/r2/pilots.jsonl, names containing "evalparents"; step [19], in the same batch as the
  development pilots of the then-candidate small 9.5 h / 20 000 and medium 12 h / 80 000). Its seeds 103 and 104
  produce the same task sets as the final rcsp_large_seed001 and rcsp_large_seed002, and its s2_t90 run (48.8 s)
  previewed rcsp_large_seed000. The diagnostic did not change the large setting;
- the final small (8 h / 8 000) and medium (10 h / 30 000) settings were selected AFTER the diagnostic ([20]-[23]),
  from development-parent evidence (the 54 s unproved development pilots of 9.5 h / 20 000 and 12 h / 80 000).
The evaluation instances are therefore not strictly held out from the calibration process: the evaluation parents
were exposed to calibration before the small and medium settings were final. The reference time is not monotone in the window: it is set by the hardness of the integer
set-covering master over the duties within the LP gap, which spikes on single windows. Maximum over 12 pilot seeds:
8 h / 8 000 duties 4.2 s; 10 h / 30 000 6.0 s; 16 h / 200 000 3.1 s. Rejected: 9.5 h / 20 000 (one seed unproved at
60 s), 12 h / 80 000 (one seed unproved at 60 s; its MIP needed 208 s), the whole parent day (s2_t120, 232 000 and
408 000 duties, unproved at 60 s). The set-covering MIP proved the 4 tested 16 h / 200 000 pilots within 3 s. Round 1
used 7 / 9.5 / 12 h with caps 2 000 / 20 000 / 40 000.
Tasks are renumbered 0..n-1 by (start, end, original index) and connections 0..m-1 in their original order. The
agent-visible data carry no parent name, window or original indices (they go to `provenance.json` next to
data.json, written by `generate_rcsp_instance`). Stations and crew bases keep their benchmark station indices.
Vendored parents (`benchmark/`, MIT licence): evaluation s2_t90, s2_t122, s3_t172; development s2_t118, s2_t120,
s3_t168 (all from the benchmark's tiny group; s{S}_t{T} = S large stations, T tasks).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, cast

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


MAX_DUTY = 34_200  # 9.5 h
MAX_WORK = 19_800  # 5.5 h
BENCHMARK = Path(__file__).resolve().parent / "benchmark"
EVAL_PARENTS = ("s2_t90", "s2_t122", "s3_t172")
DEV_PARENTS = ("s2_t118", "s2_t120", "s3_t168")
FIRST_DEV_SEED = 100

SIZE_CONFIG = {
    "small": {"window_hours": 8.0, "max_duties": 8_000},
    "medium": {"window_hours": 10.0, "max_duties": 30_000},
    "large": {"window_hours": 16.0, "max_duties": 200_000},
}

DEFAULT_CONFIG = {"max_draws": 100, "shrink_seconds": 900}
RECIPE_VERSION = "rcsp-window-v3-20261106"
RECIPE = "time-window sub-instance of a van Rossum (2026) benchmark instance"


@dataclass
class RCSPInstance(JsonDataMixin):
    n_tasks: int
    tasks: list[list[int]]  # [start, end, start_station, end_station, duration] per task
    stations: list[int]
    crew_bases: list[int]
    connections: list[list[Any]]  # [index, type, startBlock, endBlock, timeBeforeBreak, timeAfterBreak, duration, cost]

    def __post_init__(self) -> None:
        if len(self.tasks) != self.n_tasks:
            raise ValueError("tasks must have n_tasks entries")
        if [c[0] for c in self.connections] != list(range(len(self.connections))):
            raise ValueError("connection indices must be 0..m-1")

    @classmethod
    def from_json_file(cls, path: str | Path) -> RCSPInstance:
        return cast("RCSPInstance", super().from_json_file(path))

    def graph(self) -> dict[str, dict]:
        """Connection lookups: source[(base, task)], conn[(task, task)], sink[(task, base)] -> connection row."""
        out: dict[str, dict] = {"SOURCE": {}, "CONNECTION": {}, "SINK": {}}
        for c in self.connections:
            out[c[1]][(c[2], c[3])] = c
        return out


def duty_cost(chain: list[list[Any]]) -> int | None:
    """Cost of a duty given as its connection rows (SOURCE ... SINK), or None if it breaks a labour rule."""
    length = sum(c[6] for c in chain)
    if length > MAX_DUTY:
        return None
    elapsed = 0
    for c in chain:
        if c[4] >= 0 and elapsed + c[4] <= MAX_WORK and c[5] + (length - elapsed - c[6]) <= MAX_WORK:
            return sum(c[7] for c in chain)
        elapsed += c[6]
    return None


def enumerate_duties(inst: RCSPInstance, limit: int = 5_000_000) -> list[tuple[int, tuple[int, ...], int]]:
    """Every feasible duty (base, task sequence, cost), by depth-first search from every SOURCE connection.

    The search only prunes on duty length (connection durations are nonnegative, so a too-long prefix stays too
    long) and on a prefix that is past 5.5 h without any break arc that could still start in time.
    """
    g = inst.graph()
    succ: dict[int, list[list[Any]]] = {}
    for (i, _), c in g["CONNECTION"].items():
        succ.setdefault(i, []).append(c)
    sinks: dict[int, list[list[Any]]] = {}
    for (i, _), c in g["SINK"].items():
        sinks.setdefault(i, []).append(c)
    out = []
    for (base, first), src in g["SOURCE"].items():
        stack = [([src], (first,), src[6], False)]
        while stack:
            chain, path, length, had_break = stack.pop()
            for snk in sinks.get(path[-1], []):
                if snk[3] == base:
                    cost = duty_cost(chain + [snk])
                    if cost is not None:
                        out.append((base, path, cost))
                        if len(out) > limit:
                            raise RuntimeError(f"more than {limit} feasible duties")
            for c in succ.get(path[-1], []):
                nl = length + c[6]
                brk = had_break or (c[4] >= 0 and length + c[4] <= MAX_WORK)
                if nl > MAX_DUTY or (not brk and nl > MAX_WORK):
                    continue
                stack.append((chain + [c], path + (c[3],), nl, brk))
    return out


def _read(name: str, variant: str) -> dict[str, Any]:
    d = BENCHMARK / name
    rows = lambda f: [line.split() for line in (d / f).read_text().splitlines() if line.strip()]
    tasks = {int(r[0]): [int(x) for x in r[1:6]] for r in rows("tasks.txt")}
    conns = [[int(r[0]), r[1], *(int(x) for x in r[2:8])] for r in rows(f"connections_{variant}_depot.txt")]
    conns.sort(key=lambda c: c[0])
    return {"tasks": tasks, "connections": conns, "stations": sorted(int(r[0]) for r in rows("stations.txt")),
            "bases": sorted(int(r[0]) for r in rows(f"crewbases_{variant}_depot.txt"))}


def cut(raw: dict[str, Any], keep: list[int]) -> tuple[RCSPInstance, list[int], list[int]]:
    """The induced sub-instance on the kept task ids (renumbered by start, end, original id), with the original task
    and connection ids of its tasks and connections."""
    keep = sorted(keep, key=lambda t: (raw["tasks"][t][0], raw["tasks"][t][1], t))
    new = {t: k for k, t in enumerate(keep)}
    conns, origin = [], []
    for c in raw["connections"]:
        kind, a, b = c[1], c[2], c[3]
        if kind == "SOURCE" and b in new:
            row = [kind, a, new[b]]
        elif kind == "SINK" and a in new:
            row = [kind, new[a], b]
        elif kind == "CONNECTION" and a in new and b in new:
            row = [kind, new[a], new[b]]
        else:
            continue
        conns.append([len(conns), *row, *c[4:8]])
        origin.append(c[0])
    inst = RCSPInstance(len(keep), [raw["tasks"][t] for t in keep], raw["stations"], raw["bases"], conns)
    return inst, keep, origin


def window_cut(raw: dict[str, Any], t0: int, t1: int) -> tuple[RCSPInstance, list[int], list[int], int]:
    """Steps 2-4 of the module docstring for the window [t0, t1]: the tasks inside it, then the paper's
    postprocessing. Returns the instance, its original task and connection ids, and its number of feasible duties."""
    keep = [t for t, row in raw["tasks"].items() if row[0] >= t0 and row[1] <= t1]
    while True:
        inst, ids, origin = cut(raw, keep)
        duties = enumerate_duties(inst)
        covered = {t for _, path, _ in duties for t in path}
        if len(covered) == inst.n_tasks:
            return inst, ids, origin, len(duties)
        keep = [ids[k] for k in sorted(covered)]


def _too_many_duties(raw: dict[str, Any], t0: int, t1: int, cap: int) -> bool:
    """True if the tasks inside [t0, t1] admit more than `cap` feasible duties (before postprocessing, which only
    removes tasks that lie in no duty and so leaves the count unchanged)."""
    keep = [t for t, row in raw["tasks"].items() if row[0] >= t0 and row[1] <= t1]
    try:
        enumerate_duties(cut(raw, keep)[0], limit=cap)
    except RuntimeError:
        return True
    return False


class RCSPGenerator(BaseInstanceGenerator[RCSPInstance]):
    def draw(self, seed: int, size_label: str) -> tuple[RCSPInstance, dict[str, Any]]:
        if size_label not in SIZE_CONFIG:
            raise ValueError(f"Unknown size_label '{size_label}'. Expected one of {list(SIZE_CONFIG)}.")
        seed = int(seed)
        parents = DEV_PARENTS if seed >= FIRST_DEV_SEED else EVAL_PARENTS
        name = parents[seed % len(parents)]
        variant = "multi" if seed % 2 == 1 else "single"
        raw = _read(name, variant)
        hours, cap = SIZE_CONFIG[size_label]["window_hours"], SIZE_CONFIG[size_label]["max_duties"]
        span = int(round(hours * 3600))
        first = min(row[0] for row in raw["tasks"].values())
        last = max(row[1] for row in raw["tasks"].values()) - span
        rng = random.Random(f"rcsp-{size_label}-{seed}")
        for draw in range(DEFAULT_CONFIG["max_draws"]):
            t0 = first + 60 * rng.randint(0, max(0, (last - first) // 60))
            t1 = t0 + span
            # Structural size rule: shorten the window from the right until it admits at most `cap` feasible duties.
            while t1 > t0 and _too_many_duties(raw, t0, t1, cap):
                t1 -= DEFAULT_CONFIG["shrink_seconds"]
            inst, ids, origin, n_duties = window_cut(raw, t0, t1)
            if inst.n_tasks:
                files = [BENCHMARK / name / f for f in ("tasks.txt", f"connections_{variant}_depot.txt", f"crewbases_{variant}_depot.txt", "stations.txt")]
                provenance = {"recipe": RECIPE, "recipe_version": RECIPE_VERSION, "seed": seed, "size": size_label, "benchmark": name, "variant": variant,
                              "upstream": "https://github.com/BartvanRossum/rcsp-benchmark (MIT), files vendored in src/cg/rcsp/benchmark",
                              "file_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                              "window": [t0, t1], "window_shortened_seconds": t0 + span - t1, "draw": draw, "n_duties": n_duties,
                              "task_ids": ids, "connection_ids": origin}
                return inst, provenance
        raise RuntimeError(f"no nonempty {size_label} window for seed {seed}")

    def generate_instance(self, *, seed: int | None = None, size_label: str = "medium") -> RCSPInstance:
        return self.draw(0 if seed is None else int(seed), size_label)[0]


def generate_rcsp_instance(seed: int, size: str = "medium", output_dir: str | Path | None = None) -> dict[str, Any]:
    inst, provenance = RCSPGenerator().draw(seed, size)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")
        # Provenance stays outside data.json: agents receive only data.json and prompt.txt.
        (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return inst.to_dict()
