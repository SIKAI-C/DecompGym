"""Scale-up calibration: one solve (monolithic MIP or a decomposition reference) of one instance.

Every method shares one wall clock started at process start, so model building, loading and imports
count against the same budget T; publications at or after T are not eligible. Each new incumbent is a
solution witness whose objective is recomputed here, independently of the solver, and is journaled to
disk (fsync) on receipt, so a crash or OOM kill keeps every incumbent found before it. The best witness
is written in the family's output contract and checked with OutputVerifier; a failed check withdraws
the score. Results are written atomically with a fingerprint over the dependency closure; a complete,
matching result is not recomputed, and an interrupted attempt (e.g. a Slurm requeue) is archived and
rerun with a fresh clock.

Run as: python -m src.experiments.scaleup_calibration --family ufl --data path/data.json --method mip --time-limit 3600 --out result.json
"""
from __future__ import annotations

import time

T0 = time.monotonic()  # before the heavy imports: the budget starts here

import argparse
import hashlib
import json
import math
import os
import platform
import resource
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import numpy as np

SUITE = {"ufl": "bd", "gap": "cg", "clrp": "bd", "mcnd": "bd", "hub": "bd", "rcsp": "cg", "cvrp": "cg", "pdptw": "cg"}
COMMON_SOURCES = ["src/experiments/scaleup_calibration.py", "src/experiments/verification.py", "src/experiments/solution_checks.py",
                  "src/utils/output_contract.py", "src/utils/base_bd_class.py", "src/utils/base_cg_class.py", "src/utils/reference_runtime.py",
                  "src/utils/base_data.py"]
SOURCES = {
    "ufl": ["src/bd/ufl/b_gold_bd.py", "src/bd/ufl/b_ref_bd_xl.py", "src/bd/ufl/a_gen_data.py", "src/bd/ufl/d_verifier.py"],
    "gap": ["src/cg/gap/a_gold_cg.py", "src/cg/gap/b_ref_cg_xl.py", "src/cg/gap/a_gen_data.py", "src/cg/gap/d_verifier.py"],
    "cvrp": ["src/cg/cvrp/a_data.py", "src/cg/cvrp/heuristic.py", "src/cg/cvrp/b_mip_xl.py", "src/cg/cvrp/c_naive_cg.py", "src/cg/cvrp/d_ref_cg_xl.py", "src/cg/cvrp/e_verifier.py",
             "src/bd/clrp/d_ref_cg_xl.py", "src/bd/clrp/c_heuristic.py"],
    "pdptw": ["src/cg/pdptw/a_data.py", "src/cg/pdptw/pricing.py", "src/cg/pdptw/b_mip_xl.py", "src/cg/pdptw/c_naive_cg.py", "src/cg/pdptw/d_ref_cg_xl.py", "src/cg/pdptw/e_verifier.py"],
    "rcsp": ["src/cg/rcsp/a_data.py", "src/cg/rcsp/pricing.py", "src/cg/rcsp/b_mip_xl.py", "src/cg/rcsp/c_naive_cg.py", "src/cg/rcsp/d_ref_cg_xl.py", "src/cg/rcsp/e_verifier.py", "src/cg/rcsp/f_mcf_mip.py"],
    "hub": ["src/bd/hub/a_data.py", "src/bd/hub/b_mip_xl.py", "src/bd/hub/c_naive_bd.py", "src/bd/hub/d_ref_bd_xl.py", "src/bd/hub/e_verifier.py"],
    "mcnd": ["src/bd/mcnd/a_data.py", "src/bd/mcnd/b_mip_xl.py", "src/bd/mcnd/c_naive_bd.py", "src/bd/mcnd/d_ref_bd_xl.py", "src/bd/mcnd/e_verifier.py"],
    "clrp": ["src/bd/clrp/a_data.py", "src/bd/clrp/b_mip_xl.py", "src/bd/clrp/c_heuristic.py", "src/bd/clrp/d_verifier.py", "src/bd/clrp/d_ref_cg_xl.py", "src/bd/clrp/e_naive_cg.py"],
}
SOFT_MEM_GB = 12.0  # Gurobi SoftMemLimit, below the 16 GB job allocation with headroom for Python
HARD_MEM_GB = 14.0  # hard Gurobi MemLimit for model construction (below the 16 GB job allocation)


def elapsed() -> float:
    return time.monotonic() - T0


def remaining(limit: float) -> float:
    """Seconds left of the budget (may be <= 0; callers must not start new work then)."""
    return limit - elapsed()


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- family adapters

class UFLAdapter:
    def __init__(self, data: str) -> None:
        from src.bd.ufl.a_gen_data import UFLInstance
        self.inst = UFLInstance.from_json_file(data)
        self.D, self.f = self.inst.D, self.inst.f

    def objective(self, open_: list[int]) -> float:
        idx = np.asarray(sorted(open_), dtype=int)
        return float(self.f[idx].sum() + self.D[idx].min(axis=0).sum())

    def output(self, open_: list[int], status: str = "feasible") -> dict:
        idx = np.asarray(sorted(open_), dtype=int)
        y = np.zeros(self.inst.m, dtype=int)
        y[idx] = 1
        x = np.zeros((self.inst.m, self.inst.n), dtype=int)
        x[idx[np.argmin(self.D[idx], axis=0)], np.arange(self.inst.n)] = 1
        return {"status": status, "method": "benders_decomposition", "objective_value": self.objective(open_), "y": y.tolist(), "x": x.tolist()}


class GAPAdapter:
    def __init__(self, data: str) -> None:
        from src.cg.gap.a_gen_data import GAPInstance
        from src.cg.gap.b_ref_cg_xl import arrays
        self.cost, self.res, self.cap = arrays(GAPInstance.from_json_file(data))

    def objective(self, assign: list[int]) -> float | None:
        a = np.asarray(assign, dtype=int)
        n = len(a)
        if n != self.cost.shape[1] or a.min() < 0:
            return None
        load = np.bincount(a, weights=self.res[a, np.arange(n)], minlength=len(self.cap))
        return float(self.cost[a, np.arange(n)].sum()) if np.all(load <= self.cap) else None

    def output(self, assign: list[int], status: str = "feasible") -> dict:
        return {"status": status, "method": "column_generation", "objective_value": self.objective(assign),
                "solution": {"assignments": [{"job_id": i, "machine_id": int(k)} for i, k in enumerate(assign)]}}


class CLRPAdapter:
    """Prodhon CLRP instance (a .dat file); solutions are route lists [{"depot", "customers"}]."""

    def __init__(self, data: str) -> None:
        from src.bd.clrp.a_data import load_prodhon
        self.inst = load_prodhon(data)
        self.cost = self.inst.travel("ceil")

    @staticmethod
    def normalize(routes: list) -> list:
        return [{"depot": int(r["depot"]), "customers": [int(c) for c in r["customers"]]} for r in routes]

    def objective(self, routes: list) -> float | None:
        from src.bd.clrp.b_mip_xl import route_cost
        return route_cost(self.inst, self.cost, self.normalize(routes))

    def output(self, routes: list, status: str = "feasible") -> dict:
        return {"status": status, "method": "decomposition", "objective_value": self.objective(routes), "routes": self.normalize(routes)}


class MCNDAdapter:
    """MCND (.dow file); solutions are designs (open arc indices). Evaluating a design solves a multicommodity flow
    LP, so evaluation is deferred: publications are timestamped and journaled on receipt and evaluated after the
    run (the solver is not charged for the evaluation)."""

    deferred = True

    def __init__(self, data: str) -> None:
        from src.bd.mcnd.a_data import load_dow
        self.inst = load_dow(data)
        self._cache: dict[tuple[int, ...], float | None] = {}

    @staticmethod
    def normalize(design: list) -> list:
        return sorted({int(a) for a in design})

    def objective(self, design: list) -> float | None:
        from src.bd.mcnd.b_mip_xl import design_cost
        key = tuple(self.normalize(design))
        if key not in self._cache:
            self._cache[key] = design_cost(self.inst, list(key))
        return self._cache[key]

    def output(self, design: list, status: str = "feasible") -> dict:
        from src.bd.mcnd.b_mip_xl import design_output
        return design_output(self.inst, self.normalize(design), method="benders_decomposition", status=status)


class HubAdapter:
    """Uncapacitated multiple-allocation hub location (data.json); solutions are hub sets. Evaluating a hub set
    is a closed-form O(n^3) routing, deferred for the same reason as MCND (n = 200: ~1-2 s per evaluation)."""

    deferred = True

    def __init__(self, data: str) -> None:
        from src.bd.hub.a_data import load_hub
        self.inst = load_hub(data)
        self._cache: dict[tuple[int, ...], float | None] = {}

    @staticmethod
    def normalize(hubs: list) -> list:
        return sorted({int(h) for h in hubs})

    def objective(self, hubs: list) -> float | None:
        from src.bd.hub.b_mip_xl import hub_cost
        key = tuple(self.normalize(hubs))
        if not key:
            return None
        if key not in self._cache:
            self._cache[key] = hub_cost(self.inst, list(key))
        return self._cache[key]

    def output(self, hubs: list, status: str = "feasible") -> dict:
        from src.bd.hub.b_mip_xl import hub_output
        return hub_output(self.inst, self.normalize(hubs), status=status)


class RCSPAdapter:
    """Railway crew scheduling (JSON spec {"dir", "variant"}); solutions are duty lists [{"depot", "tasks"}]."""

    def __init__(self, data: str) -> None:
        from src.cg.rcsp.a_data import load_rcsp
        self.inst = load_rcsp(data)

    @staticmethod
    def normalize(duties: list) -> list:
        from src.cg.rcsp.b_mip_xl import normalize_plan
        return normalize_plan(duties)

    def objective(self, duties: list) -> float | None:
        from src.cg.rcsp.b_mip_xl import plan_cost
        return plan_cost(self.inst, self.normalize(duties))

    def output(self, duties: list, status: str = "feasible") -> dict:
        from src.cg.rcsp.b_mip_xl import rcsp_output
        return rcsp_output(self.inst, self.normalize(duties), status=status)


class CVRPAdapter:
    """CVRP (.vrp file, X set); solutions are route lists of customer ids."""

    def __init__(self, data: str) -> None:
        from src.cg.cvrp.a_data import load_vrp
        self.inst = load_vrp(data)

    @staticmethod
    def normalize(routes: list) -> list:
        from src.cg.cvrp.b_mip_xl import normalize_routes
        return normalize_routes(routes)

    def objective(self, routes: list) -> float | None:
        from src.cg.cvrp.b_mip_xl import route_cost
        cost = route_cost(self.inst, self.normalize(routes))
        return None if cost is None else float(cost)

    def output(self, routes: list, status: str = "feasible") -> dict:
        from src.cg.cvrp.b_mip_xl import cvrp_output
        return cvrp_output(self.inst, self.normalize(routes), status=status)


class PDPTWAdapter:
    """PDPTW (VRPLIB format, Li & Lim via PyVRP); solutions are route lists of node ids."""

    def __init__(self, data: str) -> None:
        from src.cg.pdptw.a_data import load_pdptw
        self.inst = load_pdptw(data)
        self.dist = self.inst.distances()

    @staticmethod
    def normalize(routes: list) -> list:
        from src.cg.pdptw.b_mip_xl import normalize_plan
        return normalize_plan(routes)

    def objective(self, routes: list) -> float | None:
        from src.cg.pdptw.b_mip_xl import route_cost
        cost = route_cost(self.inst, self.dist, self.normalize(routes))
        return None if cost is None else float(cost)

    def output(self, routes: list, status: str = "feasible") -> dict:
        from src.cg.pdptw.b_mip_xl import pdptw_output
        return pdptw_output(self.inst, self.dist, self.normalize(routes), status=status)


UFLAdapter.normalize = GAPAdapter.normalize = staticmethod(lambda sol: [int(v) for v in sol])
ADAPTERS = {"ufl": UFLAdapter, "gap": GAPAdapter, "clrp": CLRPAdapter, "mcnd": MCNDAdapter, "hub": HubAdapter, "rcsp": RCSPAdapter, "cvrp": CVRPAdapter, "pdptw": PDPTWAdapter}


class Recorder:
    """Incumbent witnesses (objective recomputed here) and bound history on the shared clock.

    A publication is timestamped on receipt; at or after the deadline it is counted but not eligible.
    Eligible witnesses and bounds are appended to a journal and fsynced before the call returns.
    """

    def __init__(self, adapter, limit: float, journal: Path) -> None:
        self.adapter, self.limit = adapter, limit
        self.witnesses: list[dict] = []
        self.bounds: list[dict] = []
        self.best: float = math.inf
        self.late = 0
        self.infeasible = 0
        self.pending: list[dict] = []  # deferred publications (adapter.deferred), evaluated by evaluate_pending()
        self.journal = journal.open("a")

    def _write(self, record: dict) -> None:
        self.journal.write(json.dumps(record) + "\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())

    def solution(self, sol: list[int], source: str) -> None:
        t = elapsed()
        if t >= self.limit:
            self.late += 1
            return
        if getattr(self.adapter, "deferred", False):
            pub = {"t": t, "solution": self.adapter.normalize(sol), "source": source}
            self.pending.append(pub)
            self._write({"kind": "publication", **pub})
            return
        obj = self.adapter.objective(sol)
        if obj is None:
            self.infeasible += 1
        elif obj < self.best - 1e-9:
            self.best = obj
            w = {"t": t, "objective": obj, "solution": self.adapter.normalize(sol), "source": source}
            self.witnesses.append(w)
            self._write({"kind": "incumbent", **w})

    def evaluate_pending(self) -> None:
        """Evaluate deferred publications in arrival order; each keeps its publication time."""
        for pub in self.pending:
            obj = self.adapter.objective(pub["solution"])
            if obj is None:
                self.infeasible += 1
            elif obj < self.best - 1e-9:
                self.best = obj
                self.witnesses.append({"t": pub["t"], "objective": obj, "solution": pub["solution"], "source": pub["source"]})
        self.pending = []

    def bound(self, value: float | None, source: str) -> None:
        t = elapsed()
        if value is None or not math.isfinite(value) or t >= self.limit:
            return
        if not self.bounds or value > self.bounds[-1]["bound"] + 1e-9 * max(1.0, abs(value)):
            b = {"t": t, "bound": float(value), "source": source}
            self.bounds.append(b)
            self._write({"kind": "bound", **b})


# ---------------------------------------------------------------- methods

def run_rcsp_mip(data: str, limit: float, rec: Recorder, family: str = "rcsp", flow: bool = False, initializer: bool = False) -> dict:
    """Compact MIPs that need their own callbacks; every integer solution is decoded and rechecked before it is
    published. RCSP: the exact crew-indexed MCF of van Rossum (2026, Appendix B, with VIs and symmetry breaking,
    barrier root) is the baseline; the layered flow model (a relaxation control) stays available as mip_flow.
    CVRP: the flow model with rounded capacity cuts. PDPTW: the two-index model of Furtado, Munari and Morabito
    (2017); with the initializer (mip_init, a diagnostic arm) the shared insertion plan is the MIP start."""
    kwargs = {}
    if family == "cvrp":
        from src.cg.cvrp.b_mip_xl import CVRPCompactMIP as CompactMIP
    elif family == "pdptw":
        from src.cg.pdptw.b_mip_xl import PDPTWCompactMIP as CompactMIP
        kwargs = {"warm_start": initializer}
    elif flow:
        from src.cg.rcsp.b_mip_xl import RCSPCompactMIP as CompactMIP
    else:
        from src.cg.rcsp.f_mcf_mip import RCSPCrewMCF as CompactMIP
        kwargs = {"method": 2}
    if remaining(limit) <= 0.01:
        return {"termination": "deadline_before_start"}
    res = CompactMIP(time_limit=remaining(limit), threads=1, on_incumbent=lambda s: rec.solution(s, "mip"),
                     on_bound=lambda b: rec.bound(b, "mip"), soft_mem_gb=SOFT_MEM_GB, **kwargs).run(data)
    reason = res.get("termination_reason", "")
    termination = "finished" if reason in ("optimal", "finished") else "memory_limit" if "mem" in reason else "deadline"
    return {"termination": termination} | {k: v for k, v in res.items() if k not in ("trace", "duties")}


def run_mip(family: str, data: str, limit: float, rec: Recorder, initializer: bool = False) -> dict:
    if family in ("rcsp", "cvrp", "pdptw"):
        return run_rcsp_mip(data, limit, rec, family, initializer=initializer)
    a = rec.adapter
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.SoftMemLimit = SOFT_MEM_GB
    if family == "ufl":
        # Strong formulation; assignments may be continuous because, for fixed binary y, the
        # assignment LP is integral, and witnesses are decoded by nearest open facility.
        m, n = a.D.shape
        y = model.addMVar(m, vtype=GRB.BINARY)
        x = model.addMVar((m, n), lb=0.0, ub=1.0)
        model.addConstr(x.sum(axis=0) == 1)
        model.addConstr(x <= y[:, None] * np.ones((1, n)))
        model.setObjective(a.f @ y + (a.D * x).sum(), GRB.MINIMIZE)
        decode = lambda get: list(np.nonzero(np.asarray(get(y)) > 0.5)[0])
        if initializer:
            # The engineered reference's own initial solution, to separate initializer from method effects.
            from src.bd.ufl.b_ref_bd_xl import local_search
            y0 = local_search(a.D, a.f, time.perf_counter() + min(60.0, 0.05 * limit))
            rec.solution(list(np.nonzero(y0 > 0.5)[0]), "mip_initializer")
            y.Start = y0
            x.Start = (np.arange(m)[:, None] == np.argmin(np.where(y0[:, None] > 0.5, a.D, np.inf), axis=0)[None, :]).astype(float)
    elif family == "mcnd":
        from src.bd.mcnd.b_mip_xl import build_mcnd_mip
        model.dispose()
        model, v = build_mcnd_mip(a.inst)
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.SoftMemLimit = SOFT_MEM_GB
        yv = v["y"]
        # Only the design is published; the runner recomputes optimal flows for it (the MIP's own flows need not be).
        decode = lambda get: list(np.nonzero(np.asarray(get(yv)) > 0.5)[0])
    elif family == "hub":
        from src.bd.hub.b_mip_xl import build_hub_mip
        model.dispose()
        # The flow model is large (n = 200: ~16M variables); memory limits are set on the environment *before*
        # construction, and a hard MemLimit makes an oversized build fail cleanly instead of being killed.
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", 1)
        env.setParam("SoftMemLimit", SOFT_MEM_GB)
        env.setParam("MemLimit", HARD_MEM_GB)
        env.start()
        try:
            model, v = build_hub_mip(a.inst, env=env)
        except gp.GurobiError as exc:
            return {"termination": "memory_limit_during_build" if exc.errno == GRB.Error.OUT_OF_MEMORY else "crash",
                    "error": f"GurobiError: {exc}", "build_seconds": elapsed()}
        zv = v["z"]
        decode = lambda get: list(np.nonzero(np.asarray(get(zv)) > 0.5)[0])
    elif family == "clrp":
        from src.bd.clrp.b_mip_xl import build_clrp_mip, decode_routes
        model.dispose()
        model, v = build_clrp_mip(a.inst, a.cost)
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.SoftMemLimit = SOFT_MEM_GB
        arcs, xv = v["arcs"], [v["x"][arc] for arc in v["arcs"]]
        decode = lambda get: decode_routes(a.inst, dict(zip(arcs, get(xv))))
    else:
        m, n = a.cost.shape
        x = model.addMVar((m, n), vtype=GRB.BINARY)
        model.addConstr(x.sum(axis=0) == 1)
        model.addConstr((a.res * x).sum(axis=1) <= a.cap)
        model.setObjective((a.cost * x).sum(), GRB.MINIMIZE)
        decode = lambda get: list(np.argmax(np.asarray(get(x)), axis=0))
    build_seconds = elapsed()
    if remaining(limit) <= 0.01:
        return {"termination": "deadline_during_build", "build_seconds": build_seconds}
    model.Params.TimeLimit = remaining(limit)

    def callback(cb, where):
        if where == GRB.Callback.MIPSOL:
            rec.solution(decode(cb.cbGetSolution), "mip")
        elif where == GRB.Callback.MIP:
            rec.bound(cb.cbGet(GRB.Callback.MIP_OBJBND), "mip")

    model.optimize(callback)
    if model.SolCount:
        rec.solution(decode(lambda v: model.getAttr("X", v) if isinstance(v, list) else v.X), "mip_final")
    rec.bound(float(model.ObjBound), "mip_final")
    termination = {GRB.OPTIMAL: "finished", GRB.TIME_LIMIT: "deadline", GRB.MEM_LIMIT: "memory_limit"}.get(model.Status, f"status_{model.Status}")
    return {"termination": termination, "status": int(model.Status), "proved_optimal": model.Status == GRB.OPTIMAL, "build_seconds": build_seconds,
            "node_count": float(model.NodeCount), "num_vars": int(model.NumVars), "num_constrs": int(model.NumConstrs)}


def run_paper_reference(family: str, data: str, limit: float, rec: Recorder) -> dict:
    """The paper's reference algorithm (naive at this scale), given the remaining budget."""
    from src.bd.ufl.b_gold_bd import UFLBenders
    from src.cg.gap.a_gold_cg import GAPColumnGeneration

    class PublishingUFLBenders(UFLBenders):
        """The paper's UFL Benders, unchanged, publishing each completed master/recourse pair's plan."""

        def compute_iteration_bounds(self, inst, master, sub):
            bounds = super().compute_iteration_bounds(inst, master, sub)
            if sub.all_feasible:
                rec.solution([int(k.split("_")[1]) for k, v in master.vars.items() if k.startswith("y_") and v > 0.5], "paper_reference")
            return bounds

    class PublishingGAPColumnGeneration(GAPColumnGeneration):
        """The paper's GAP CG, unchanged, publishing its integer restricted master's assignment.
        That master runs once, after column generation, so this is when the paper's method has a plan."""

        def solve_integer_master(self, inst, patterns):
            solution = super().solve_integer_master(inst, patterns)
            assign = [-1] * len(inst.jobs)
            for a in solution.metadata.get("assignments", []):
                assign[a["job_id"]] = a["machine_id"]
            if min(assign) >= 0:
                rec.solution(assign, "paper_reference")
            return solution

    from src.utils.reference_runtime import ReferenceLimit

    if remaining(limit) <= 0.01:
        return {"termination": "deadline_before_start"}
    memory = {"SoftMemLimit": SOFT_MEM_GB}  # same memory guard as every other method
    params = {"master_params": memory, "subproblem_params": memory} if family == "ufl" else {"master_params": memory}
    solver = {"ufl": PublishingUFLBenders, "gap": PublishingGAPColumnGeneration}[family](max_iters=10**6, time_limit=remaining(limit), **params)
    error, termination = None, None
    try:
        res = solver.run(data)
    except ReferenceLimit as exc:  # the budget ran out before a completed result
        res, error, termination = {}, f"ReferenceLimit: {exc}", "deadline"
    except MemoryError as exc:
        res, error, termination = {}, f"MemoryError: {exc}", "oom"
    except gp.GurobiError as exc:
        res, error = {}, f"GurobiError: {exc}"
        termination = "memory_limit" if exc.errno == GRB.Error.OUT_OF_MEMORY or "memory" in str(exc).lower() else "crash"
    except Exception as exc:  # keep the evidence
        res, error, termination = {}, f"{type(exc).__name__}: {exc}", "crash"
    if termination is None:
        reason = str(res.get("termination_reason", ""))
        termination = "deadline" if "time" in reason else "memory_limit" if "mem" in reason else "finished"
    if family == "ufl" and res.get("final_vars"):
        rec.solution([int(k.split("_")[1]) for k, v in res["final_vars"].items() if k.startswith("y_") and v > 0.5], "paper_reference_final")
    records = [{k: getattr(r, k) for k in ("iteration", "elapsed_time", "master_obj", "global_lb", "global_ub") if hasattr(r, k)} for r in getattr(solver, "iter_records", [])]
    # The paper's references report bounds per iteration on their own clock; they are kept as diagnostics only.
    return {"termination": termination, "reference_termination_reason": res.get("termination_reason"), "claimed_objective": res.get("best_objective"), "claimed_lb": res.get("global_lb"), "iters": res.get("iters"),
            "error": error, "iteration_records": records,
            }


def run_engineered_reference(family: str, data: str, limit: float, rec: Recorder) -> dict:
    if remaining(limit) <= 0.01:
        return {"termination": "deadline_before_start"}
    # References may pass the stage that produced an incumbent; it is kept as provenance in the journal and witnesses.
    hooks = {"on_incumbent": lambda s, stage=None: rec.solution(s, f"ref_xl:{stage}" if stage else "ref_xl"),
             "on_bound": lambda b: rec.bound(b, "ref_xl"), "soft_mem_gb": SOFT_MEM_GB}
    if family == "ufl":
        from src.bd.ufl.b_ref_bd_xl import UFLBendersXL
        solver = UFLBendersXL(time_limit=remaining(limit), threads=1, **hooks)
    elif family == "clrp":
        from src.bd.clrp.d_ref_cg_xl import CLRPColumnGenerationXL
        solver = CLRPColumnGenerationXL(time_limit=remaining(limit), threads=1, **hooks)
    elif family == "mcnd":
        from src.bd.mcnd.d_ref_bd_xl import MCNDBendersXL
        solver = MCNDBendersXL(time_limit=remaining(limit), threads=1, **hooks)
    elif family == "hub":
        from src.bd.hub.d_ref_bd_xl import HubBendersXL
        solver = HubBendersXL(time_limit=remaining(limit), threads=1, **hooks)
    elif family == "rcsp":
        from src.cg.rcsp.d_ref_cg_xl import RCSPColumnGenerationXL
        solver = RCSPColumnGenerationXL(time_limit=remaining(limit), threads=1, **hooks)
    elif family == "cvrp":
        from src.cg.cvrp.d_ref_cg_xl import CVRPColumnGenerationXL
        solver = CVRPColumnGenerationXL(time_limit=remaining(limit), threads=1, **hooks)
    elif family == "pdptw":
        from src.cg.pdptw.d_ref_cg_xl import PDPTWColumnGenerationXL
        solver = PDPTWColumnGenerationXL(time_limit=remaining(limit), threads=1, **hooks)
    else:
        from src.cg.gap.b_ref_cg_xl import GAPColumnGenerationXL
        solver = GAPColumnGenerationXL(time_limit=remaining(limit), threads=1, **hooks)
    res = solver.run(data)
    reason = res.get("termination_reason", "")
    termination = "memory_limit" if reason == "memory_limit" else "finished" if reason == "optimal" else "deadline"
    return {"termination": termination} | {k: v for k, v in res.items() if k not in ("trace", "assignment", "open")}


def run_heuristic_control(family: str, data: str, limit: float, rec: Recorder) -> dict:
    """Heuristic control without decomposition (fixed seed), same budget and verification."""
    a = rec.adapter
    started = time.perf_counter()
    if family == "clrp":
        from src.bd.clrp.c_heuristic import depot_search
        best, _ = depot_search(a.inst, a.cost, started + remaining(limit), 0, lambda plan: rec.solution(plan, "heuristic_control"))
        return {"termination": "deadline", "heuristic_best": best, "seed": 0}
    if family != "ufl":
        raise ValueError("no heuristic control for this family")
    from src.bd.ufl.b_ref_bd_xl import iterated_local_search
    best, _ = iterated_local_search(a.D, a.f, started + remaining(limit), seed=0, on_incumbent=lambda s: rec.solution(s, "heuristic_control"))
    return {"termination": "deadline", "heuristic_best": best, "seed": 0}


def run_root_cg_control(family: str, data: str, limit: float, rec: Recorder) -> dict:
    """Naive price-and-branch control: root column generation, then integer restricted masters (CLRP)."""
    if family != "clrp":
        raise ValueError("the root-CG control is defined for CLRP only")
    from src.bd.clrp.d_ref_cg_xl import CLRPColumnGenerationXL
    solver = CLRPColumnGenerationXL(time_limit=remaining(limit), threads=1, root_only=True, on_incumbent=lambda s: rec.solution(s, "root_cg"),
                                    on_bound=lambda b: rec.bound(b, "root_cg"), soft_mem_gb=SOFT_MEM_GB)
    res = solver.run(data)
    reason = res.get("termination_reason", "")
    return {"termination": "memory_limit" if reason == "memory_limit" else "finished" if reason == "optimal" else "deadline"} | {k: v for k, v in res.items() if k not in ("trace", "routes")}


def run_naive_bd_control(family: str, data: str, limit: float, rec: Recorder) -> dict:
    """Naive textbook Benders control (protocol v3.1): outer-loop integer master, one aggregate cut per iteration (MCND)."""
    if family not in ("mcnd", "hub"):
        raise ValueError("the naive BD control is defined for MCND and hub location only")
    if family == "hub":
        from src.bd.hub.c_naive_bd import HubNaiveBenders as NaiveBenders
    else:
        from src.bd.mcnd.c_naive_bd import MCNDNaiveBenders as NaiveBenders
    res = NaiveBenders(time_limit=remaining(limit), threads=1, on_incumbent=lambda s: rec.solution(s, "naive_bd"),
                           on_bound=lambda b: rec.bound(b, "naive_bd"), soft_mem_gb=SOFT_MEM_GB).run(data)
    reason = res.get("termination_reason", "")
    return {"termination": "finished" if reason == "optimal" else "deadline"} | {k: v for k, v in res.items() if k not in ("trace", "open")}


def run_naive_cg_control(family: str, data: str, limit: float, rec: Recorder) -> dict:
    """Naive textbook column generation control (protocol v3.1): root CG with one best column per depot, then an
    integer restricted master in the last 20% of the budget (CLRP)."""
    if family not in ("clrp", "rcsp", "cvrp", "pdptw"):
        raise ValueError("the naive CG control is defined for CLRP, RCSP, CVRP and PDPTW only")
    if family == "rcsp":
        from src.cg.rcsp.c_naive_cg import RCSPNaiveCG as NaiveCG
    elif family == "cvrp":
        from src.cg.cvrp.c_naive_cg import CVRPNaiveCG as NaiveCG
    elif family == "pdptw":
        from src.cg.pdptw.c_naive_cg import PDPTWNaiveCG as NaiveCG
    else:
        from src.bd.clrp.e_naive_cg import CLRPNaiveCG as NaiveCG
    res = NaiveCG(time_limit=remaining(limit), threads=1, on_incumbent=lambda s: rec.solution(s, "naive_cg"),
                      on_bound=lambda b: rec.bound(b, "naive_cg"), soft_mem_gb=SOFT_MEM_GB).run(data)
    return {"termination": "deadline"} | {k: v for k, v in res.items() if k != "routes"}


METHODS = {"mip": run_mip, "mip_flow": lambda family, data, limit, rec: run_rcsp_mip(data, limit, rec, family, flow=True), "mip_init": lambda *args: run_mip(*args, initializer=True), "root_cg": run_root_cg_control, "naive_cg": run_naive_cg_control, "naive_bd": run_naive_bd_control, "ref": run_paper_reference, "ref_xl": run_engineered_reference, "ls": run_heuristic_control}


# ---------------------------------------------------------------- driver

def referenced_inputs(args) -> dict:
    """Hashes of input files referenced by a spec (RCSP specs point to an instance directory)."""
    if args.family != "rcsp":
        return {}
    from src.cg.rcsp.a_data import read_spec
    directory, _ = read_spec(args.data)
    return {str(f.relative_to(directory)): sha256(f) for f in sorted(Path(directory).iterdir()) if f.is_file()}


def fingerprint(args) -> dict:
    return {"family": args.family, "method": args.method, "time_limit": args.time_limit, "data_sha256": sha256(args.data),
            "referenced_inputs_sha256": referenced_inputs(args),
            "source_sha256": {p: sha256(p) for p in COMMON_SOURCES + SOURCES[args.family]},
            "gurobi": ".".join(map(str, gp.gurobi.version()))}


def write_result(args, fp: dict, out: Path, rec: Recorder, details: dict, previous: list[Path], total: float | None, extra: dict | None = None) -> dict:
    """Verify the best eligible witness and write the result atomically (score withdrawn unless verified)."""
    journal, witnesses_path, solution_path = out.with_suffix(".journal.jsonl"), out.with_suffix(".witnesses.json"), out.with_suffix(".output.json")
    result = {"fingerprint": fp, "complete": True, "family": args.family, "method": args.method, "data": args.data,
              "time_limit": args.time_limit, "total_seconds": total, "termination": details.get("termination"),
              "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
              "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"), "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
              "previous_attempts": [str(p) for p in previous],
              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "best_bound": rec.bounds[-1]["bound"] if rec.bounds else None, "late_publications": rec.late, "infeasible_publications": rec.infeasible,
              "details": details, "witness_count": len(rec.witnesses), "bounds": rec.bounds,
              "incumbents": [{k: w[k] for k in ("t", "objective", "source")} for w in rec.witnesses]} | (extra or {})
    verified = None
    if rec.witnesses:
        from src.experiments.verification import OutputVerifier
        solution_path.write_text(json.dumps(rec.adapter.output(rec.witnesses[-1]["solution"])))
        if args.family == "clrp":  # not a paper family: its own independent verifier on the Prodhon file
            from src.bd.clrp.d_verifier import verify_solution
            result["verification"] = verify_solution(args.data, solution_path, 1e-6)
        elif args.family == "mcnd":
            from src.bd.mcnd.e_verifier import verify_solution
            result["verification"] = verify_solution(args.data, solution_path, 1e-6)
        elif args.family == "hub":
            from src.bd.hub.e_verifier import verify_solution
            result["verification"] = verify_solution(args.data, solution_path, 1e-6)
        elif args.family == "rcsp":
            from src.cg.rcsp.e_verifier import verify_solution
            result["verification"] = verify_solution(args.data, solution_path, 1e-6)
        elif args.family == "cvrp":
            from src.cg.cvrp.e_verifier import verify_solution
            result["verification"] = verify_solution(args.data, solution_path, 1e-6)
        elif args.family == "pdptw":
            from src.cg.pdptw.e_verifier import verify_solution
            result["verification"] = verify_solution(args.data, solution_path)
        else:
            result["verification"] = OutputVerifier.verify_reference_output(SUITE[args.family], args.family, Path(args.data), solution_path, 1e-6)
        verified = bool(result["verification"].get("ok"))
        witnesses_path.write_text(json.dumps(rec.witnesses))
    result["scored_objective"] = rec.best if verified else None
    result["artifact_sha256"] = {str(p): sha256(p) for p in (journal, witnesses_path, solution_path) if p.exists()}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1, default=str))
    os.replace(tmp, out)
    return result


def replay(journal: Path, adapter, limit: float) -> Recorder:
    """Rebuild a Recorder from a journal; incomplete trailing lines (a kill mid-write) are ignored."""
    rec = Recorder(adapter, limit, Path(os.devnull))
    for line in journal.read_text().splitlines() if journal.exists() else []:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") == "incumbent" and r["t"] < limit and adapter.objective(r["solution"]) == r["objective"] and r["objective"] < rec.best:
            rec.best = r["objective"]
            rec.witnesses.append({k: r[k] for k in ("t", "objective", "solution", "source")})
        elif r.get("kind") == "publication" and r["t"] < limit:
            rec.pending.append({k: r[k] for k in ("t", "solution", "source")})
        elif r.get("kind") == "bound" and r["t"] < limit:
            rec.bounds.append({k: r[k] for k in ("t", "bound", "source")})
    rec.evaluate_pending()
    return rec


def complete(out: Path, fp: dict) -> bool:
    if not out.exists():
        return False
    prior = json.loads(out.read_text())
    intact = all(Path(p).exists() and sha256(p) == h for p, h in prior.get("artifact_sha256", {}).items())
    return prior.get("fingerprint") == fp and bool(prior.get("complete")) and intact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--finalize", action="store_true", help="Postmortem: score a killed attempt from its journal (run after the solver process exits)")
    parser.add_argument("--exit-code", type=int, default=None, help="Exit code of the solver process (with --finalize)")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(args)
    journal = out.with_suffix(".journal.jsonl")
    if complete(out, fp):
        print(f"complete result exists: {out}")
        return
    if args.finalize:
        # A hard kill (cgroup OOM, SIGKILL) skips the solver's own result; its journal still holds every
        # eligible incumbent. SIGTERM (exit 143) is a Slurm preemption: left for the requeued run.
        if args.exit_code in (0, 143, None) or not journal.exists():
            return
        rec = replay(journal, ADAPTERS[args.family](args.data), args.time_limit)
        termination = "oom_or_killed" if args.exit_code == 137 else "crash"
        result = write_result(args, fp, out, rec, {"termination": termination, "exit_code": args.exit_code}, [], None, {"postmortem": True})
        print(json.dumps({k: result[k] for k in ("family", "method", "termination", "scored_objective", "witness_count")}, default=str))
        return
    # An earlier attempt that did not finish (requeue/preemption) is archived, never resumed; it is
    # an infrastructure interruption and this attempt starts with a fresh clock.
    previous = sorted(out.parent.glob(out.stem + ".attempt*.journal.jsonl"))
    if journal.exists():
        journal.rename(out.with_name(f"{out.stem}.attempt{len(previous) + 1}.journal.jsonl"))
        previous = sorted(out.parent.glob(out.stem + ".attempt*.journal.jsonl"))
    rec = Recorder(ADAPTERS[args.family](args.data), args.time_limit, journal)
    try:
        details = METHODS[args.method](args.family, args.data, args.time_limit, rec)
    except MemoryError:
        details = {"termination": "oom"}
    except Exception as exc:  # incumbents already journaled stay eligible
        details = {"termination": "crash", "error": f"{type(exc).__name__}: {exc}"}
    total = elapsed()
    rec.evaluate_pending()  # after the clock stops: evaluation is not charged to the solver
    rec.journal.close()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 if platform.system() == "Linux" else 1024.0 ** 2)
    result = write_result(args, fp, out, rec, details, previous, total, {"peak_rss_mb": peak})
    print(json.dumps({k: result[k] for k in ("family", "method", "termination", "total_seconds", "scored_objective", "best_bound", "witness_count", "late_publications")}, default=str))


if __name__ == "__main__":
    main()
