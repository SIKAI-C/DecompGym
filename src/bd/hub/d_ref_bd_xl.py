"""Engineered Benders reference for the UMAHLP (scale-up study; the paper's references are unchanged), in the spirit
of Contreras, Cordeau, Laporte (2011, Operations Research 59(6)): multi-cut Benders on the reduced path formulation
with closed-form duals, strong (Pareto-type) optimality cuts, reduction tests and a local-search heuristic, here run
as one branch-and-cut tree.

Master: min sum_h f_h z_h + sum_i theta_i over z in {0, 1}^n, sum_h z_h >= 1, one recourse variable per ORIGIN i
(theta_i = sum_j W_ij Q_ij(z), CCL's |H|-cut reformulation), theta_i >= the routing cost of origin i with every
candidate hub open. Cuts are per-origin aggregates of closed-form duals of c_naive_bd.HubCuts:
- at integer points (heuristic solutions, lazy cuts) the CCL Algorithm 2 dual: a = C_ij(H1), u = 0 on open hubs,
  each closed hub covers its loop and its edges to open hubs, and closed-closed edges whose requirement exceeds the
  two lower bounds get half of the deficit on each end (`HubCuts.half_split`, all closed hubs by default). On AP this
  dual choice matters most: an outer loop with these cuts closes 50L in 7 and 100L50 in 6 iterations, against 27+ and
  17+ iterations (still 8% / 20% open) with the plain ownership duals of the naive control. The value a is then raised
  within [C_ij(H1), c_(2)] to maximize the cut at a core point (a Magnanti-Wong selection inside the family, in the
  role of CCL's approximate Pareto-optimal cuts, Algorithm 4);
- at fractional points (root LP loop, root user cuts): per origin the most violated of the ownership cut with the
  critical dual value (UFL-like closed form, leading group = largest values) and the integer-point cuts of a few
  roundings with a maximizing the cut at the point; when the support is small (<= max_support hubs) the exact LP
  dual on the support (`HubCuts.lp_cut`, Gurobi, pairs batched) is used instead.

Stages, all within one wall-clock budget:
1. The shared basic initializer (greedy add, identical to the naive control), then local search with CCL's
   neighborhoods (close, open, swap; CCL Algorithm 6) and random multi-starts (CCL Algorithm 7, shortened); the best
   solutions seed the master with their cuts.
2. Root cutting-plane loop on the master LP with in-out stabilization (separation at lam z_lp + (1 - lam) core,
   lam -> 1 when the stabilized point gives no violated cut or the bound stalls, then exact LP separation on small
   supports), cuts for every origin violated at the LP point; LP-rounding plus local search for incumbents;
   non-binding cuts dropped.
3. Reduction test (CCL Proposition 4): a hub whose LP reduced cost r_h satisfies LP + r_h > UB is closed for good;
   the cut machinery is rebuilt on the remaining candidates (smaller subproblems, stronger cuts).
4. Gurobi branch-and-cut on the master (one tree): at every integer solution the routing is evaluated in closed
   form, published, and the violated per-origin cuts are added as lazy constraints; fractional cuts as user cuts at
   the root node; new tree incumbents get a local search whose result is injected back. Bounds come from the master
   LP during the root loop and from the MIP callback afterwards.
"""
from __future__ import annotations

import inspect
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.hub.a_data import HubInstance, load_hub
from src.bd.hub.b_mip_xl import route_costs
from src.bd.hub.c_naive_bd import (INF, HubCuts, add_savings, certify, greedy_add, is_memory_error, make_env,
                                   objective)


# ---------------------------------------------------------------- local search (CCL Algorithm 6)

def local_search(inst: HubInstance, z: np.ndarray, cand: np.ndarray, deadline: float, swaps: bool = True) -> tuple[np.ndarray, float]:
    """Best-improvement descent over close / open / swap moves (a neighborhood is explored only when the previous ones
    fail). Returns (hub vector, cost)."""
    z = z.copy()
    cost, _ = objective(inst, z)
    tol = 1e-9 * max(1.0, cost)
    while time.perf_counter() < deadline:
        H = np.nonzero(z)[0]
        moved = False
        # Close.
        if len(H) > 1:
            best, arg = cost, None
            for k in H:
                z2 = z.copy()
                z2[k] = False
                c2, _ = objective(inst, z2)
                if c2 < best - tol:
                    best, arg = c2, k
            if arg is not None:
                z[arg], cost, moved = False, best, True
        if moved:
            continue
        if time.perf_counter() >= deadline:
            break
        # Open.
        rest = np.nonzero(cand & ~z)[0]
        if len(rest):
            C = route_costs(inst, z)
            net = add_savings(inst, H, C, rest) - inst.f[rest]
            k = int(np.argmax(net))
            if net[k] > tol:
                z[rest[k]] = True
                cost, _ = objective(inst, z)
                continue
        if not swaps or len(H) < 1 or not len(rest):
            break
        # Swap (close k, open h).
        best, arg = cost, None
        for k in H:
            if time.perf_counter() >= deadline:
                break
            z2 = z.copy()
            z2[k] = False
            if not z2.any():
                c_base = INF
                h_costs = inst.f[rest] + (inst.chi * (inst.w.sum(axis=1) @ inst.d[:, rest]) + inst.delta * (inst.d[rest] @ inst.w.sum(axis=0)))
                j = int(np.argmin(h_costs))
                cand_cost, h = float(h_costs[j]), rest[j]
            else:
                c_base, _ = objective(inst, z2)
                C = route_costs(inst, z2)
                net = add_savings(inst, np.nonzero(z2)[0], C, rest) - inst.f[rest]
                j = int(np.argmax(net))
                cand_cost, h = c_base - float(net[j]), rest[j]
            if cand_cost < best - tol:
                best, arg = cand_cost, (k, h)
        if arg is None:
            break
        z[arg[0]], z[arg[1]] = False, True
        cost, _ = objective(inst, z)
    return z, cost


class HubBendersXL:
    """Engineered Benders reference; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 heuristic_fraction: float = 0.05, multistarts: int = 10, root_fraction: float = 0.3, root_rounds: int = 8,
                 inout_lambda: float = 0.5, max_group: int = 40, root_user_rounds: int = 10, reduction: bool = True,
                 pareto: bool = True, exact_lp: bool = True, max_support: int = 25, lp_lambda: float = 1.0,
                 closed_progress: float = 2e-3, lp_progress: float = 1e-5, half_size: int | None = None,
                 verbose: bool = False) -> None:
        self.time_limit = time_limit
        self.threads = threads
        self.on_incumbent = on_incumbent or (lambda hubs, stage=None: None)  # receives the sorted list of open hub indices
        try:  # hooks that accept a `stage` keyword also receive the incumbent's provenance
            params = inspect.signature(self.on_incumbent).parameters
            self.hook_takes_stage = "stage" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            self.hook_takes_stage = False
        self.on_bound = on_bound or (lambda bound: None)  # receives every new valid global lower bound
        self.soft_mem_gb = soft_mem_gb
        self.seed = seed
        self.heuristic_fraction, self.multistarts = heuristic_fraction, multistarts
        self.root_fraction, self.root_rounds = root_fraction, root_rounds
        self.inout_lambda = inout_lambda
        self.max_group = max_group
        self.root_user_rounds = root_user_rounds
        self.reduction = reduction
        self.pareto = pareto
        self.exact_lp, self.max_support, self.lp_lambda = exact_lp, max_support, lp_lambda
        self.closed_progress, self.lp_progress = closed_progress, lp_progress
        self.half_size = half_size
        self.verbose = verbose
        self.trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------ bookkeeping

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def log(self, stage: str) -> None:
        self.trace.append({"t": round(self.elapsed(), 3), "incumbent": self.ub if self.ub < INF else None,
                           "bound": self.lb if self.lb > -INF else None, "stage": stage})
        if self.verbose:
            print(self.trace[-1], flush=True)

    def publish_bound(self, value: float, stage: str) -> None:
        value = certify(value)
        if value > self.lb:
            self.lb = value
            self.on_bound(value)
            self.log(stage)

    def offer(self, z: np.ndarray, cost: float, stage: str) -> bool:
        if cost < self.ub - 1e-9 * max(1.0, abs(cost)):
            self.ub, self.best = cost, z.copy()
            self.pending = True
            hubs = np.nonzero(z)[0].tolist()
            self.on_incumbent(hubs, stage=stage) if self.hook_takes_stage else self.on_incumbent(hubs)
            self.log(stage)
            return True
        return False

    def closed(self) -> bool:
        return self.ub < INF and self.lb >= self.ub - 1e-6 * max(1.0, abs(self.ub))

    # ------------------------------------------------------------ run

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        return self.solve(load_hub(data_path))

    def solve(self, inst: HubInstance) -> dict[str, Any]:
        if not hasattr(self, "t0"):
            self.t0 = time.perf_counter()
            self.deadline = self.t0 + self.time_limit
        self.inst = inst
        self.rng = np.random.default_rng(self.seed)
        self.ub, self.lb = INF, -INF
        self.best: np.ndarray | None = None
        self.pending = False
        self.pool: dict[bytes, float] = {}
        self.stats = {"iterations": 0, "root_rounds": 0, "lp_rounds": 0, "cuts": 0, "root_cuts": 0, "seed_cuts": 0, "lazy_cuts": 0, "user_cuts": 0,
                      "mipsol_calls": 0, "injected": 0, "local_searches": 0, "fixed_hubs": 0, "candidates": inst.n,
                      "master_seconds": 0.0, "sub_seconds": 0.0, "heuristic_seconds": 0.0, "setup_seconds": 0.0,
                      "root_seconds": 0.0, "root_bound": None, "node_count": 0.0}
        reason = "time_limit"
        env = make_env(self.threads, self.soft_mem_gb)
        self.env = env
        try:
            reason = self.search()
        except gp.GurobiError as err:
            if not is_memory_error(err):
                raise
            reason = "memory_limit"
        env.dispose()
        self.stats["iterations"] = self.stats["root_rounds"] + self.stats["mipsol_calls"]
        ub = self.ub if self.ub < INF else None
        lb = self.lb if self.lb > -INF else None
        return {"best_objective": ub, "global_lb": lb, "global_ub": ub,
                "gap": (ub - lb) / abs(ub) if ub is not None and lb is not None and ub else None,
                "termination_reason": reason, "hubs": np.nonzero(self.best)[0].tolist() if self.best is not None else None,
                "stats": self.stats, **self.stats, "total_seconds": self.elapsed(), "trace": self.trace}

    def search(self) -> str:
        inst = self.inst
        self.publish_bound(0.0, "trivial")
        self.cand = np.ones(inst.n, dtype=bool)
        # 1. Shared initializer, local search, multi-start.
        t = time.perf_counter()
        z0 = greedy_add(inst, self.deadline)
        c0, _ = objective(inst, z0)
        self.offer(z0, c0, "initializer")
        self.pool[z0.tobytes()] = c0
        if self.left() <= 0:
            return "time_limit"
        self.heuristics(time.perf_counter() + self.heuristic_fraction * max(0.0, self.left()))
        self.stats["heuristic_seconds"] += time.perf_counter() - t
        if self.left() <= 0:
            return "time_limit"
        # Cut machinery: open hubs of the incumbent first, then the others by decreasing net saving.
        t = time.perf_counter()
        try:
            self.cuts = HubCuts(inst, self.cand, self.hub_rank(), half_size=self.half_size or inst.n, deadline=self.deadline)
        except TimeoutError:
            return "time_limit"
        self.stats["setup_seconds"] += time.perf_counter() - t
        self.publish_bound(float(inst.f.min() + self.cuts.lb_origin.sum()), "trivial")
        if self.left() <= 0:
            return "time_limit"
        self.build_master()
        # 2. Root loop; 3. reduction test.
        self.root_loop()
        if self.closed():
            return "optimal"
        if self.left() <= 0.01:
            return "time_limit"
        # 4. Branch-and-cut.
        return self.branch_and_cut()

    # ------------------------------------------------------------ heuristics

    def improve(self, z: np.ndarray, deadline: float, stage: str) -> tuple[np.ndarray, float]:
        self.stats["local_searches"] += 1
        z2, c2 = local_search(self.inst, z, self.cand, deadline)
        self.pool[z2.tobytes()] = c2
        self.offer(z2, c2, stage)
        return z2, c2

    def heuristics(self, deadline: float) -> None:
        inst = self.inst
        self.improve(self.best, deadline, "local_search")
        p = int(self.best.sum())
        cand_idx = np.nonzero(self.cand)[0]
        for r in range(self.multistarts):
            if time.perf_counter() >= deadline or self.left() <= 0:
                break
            q = int(np.clip(p + self.rng.integers(-1, 2), 1, len(cand_idx)))
            z = np.zeros(inst.n, dtype=bool)
            z[self.rng.choice(cand_idx, size=q, replace=False)] = True
            self.improve(z, deadline, "multistart")

    def hub_rank(self) -> np.ndarray:
        """Static hub order: incumbent hubs, then hubs by decreasing net saving w.r.t. the incumbent."""
        inst = self.inst
        H = np.nonzero(self.best)[0]
        others = np.nonzero(~self.best)[0]
        net = add_savings(inst, H, route_costs(inst, self.best), others) - inst.f[others] if len(others) else np.zeros(0)
        order = np.concatenate([H, others[np.argsort(-net, kind="stable")]])
        rank = np.empty(inst.n, dtype=np.int64)
        rank[order] = np.arange(inst.n)
        return rank

    def core_point(self) -> np.ndarray:
        n = self.inst.n
        core = np.where(self.cand, 1.0 / max(1, self.cand.sum()), 0.0)
        if self.best is not None:
            core = 0.5 * self.best + 0.5 * core
        if getattr(self, "lp_point", None) is not None:
            core = 0.5 * core + 0.5 * self.lp_point
        return core

    # ------------------------------------------------------------ master

    def build_master(self) -> None:
        inst = self.inst
        md = gp.Model(env=self.env)
        md.Params.OutputFlag = 0
        md.Params.Seed = self.seed
        self.master = md
        self.z = md.addMVar(inst.n, lb=0.0, ub=1.0, name="z")
        self.theta = md.addMVar(inst.n, lb=self.cuts.lb_origin, name="theta")
        md.addConstr(self.z.sum() >= 1)
        md.setObjective(inst.f @ self.z + self.theta.sum(), GRB.MINIMIZE)
        self.zv = self.z.tolist()
        self.tv = self.theta.tolist()
        self.rows: list[gp.Constr] = []
        self.lp_point = None
        # Seed cuts from the best distinct heuristic solutions.
        for key, _ in sorted(self.pool.items(), key=lambda kv: kv[1])[:5]:
            if self.left() <= 0:
                break
            zz = np.frombuffer(key, dtype=bool).copy()
            if np.any(zz & ~self.cand):
                continue
            cut = self.binary_cut(zz)
            for i in range(inst.n):
                self.rows.append(md.addConstr(self.cut_expr(cut, i) >= cut["const"][i]))
            self.stats["seed_cuts"] += inst.n
            self.stats["cuts"] += inst.n

    def binary_cut(self, z: np.ndarray) -> dict[str, Any]:
        t = time.perf_counter()
        cut = self.cuts.binary_cut(z, self.core_point() if self.pareto else None)
        self.stats["sub_seconds"] += time.perf_counter() - t
        return cut

    def cut_expr(self, cut: dict[str, Any], i: int) -> gp.LinExpr:
        coef = cut["coef"][i]
        nz = np.nonzero(coef > 1e-12)[0]
        expr = gp.LinExpr(coef[nz].tolist(), [self.zv[h] for h in nz])
        expr.add(self.tv[i])
        return expr

    # ------------------------------------------------------------ root

    def root_loop(self) -> None:
        inst, md = self.inst, self.master
        t_root = time.perf_counter()
        root_deadline = t_root + self.root_fraction * max(0.0, self.left())
        lam = self.inout_lambda
        core = self.core_point()
        best_bound, stalled = -INF, 0
        z_lp = None
        mode = "closed"
        history: list[float] = []
        while self.stats["root_rounds"] < self.root_rounds and time.perf_counter() < root_deadline and self.left() > 0:
            t = time.perf_counter()
            md.Params.TimeLimit = max(0.01, self.left())
            md.optimize()
            self.stats["master_seconds"] += time.perf_counter() - t
            if md.Status != GRB.OPTIMAL:
                break
            self.stats["root_rounds"] += 1
            bound = float(md.ObjVal)
            self.publish_bound(bound, "root_lp")
            history.append(bound)
            # Stalled: less than `progress` relative improvement over the last 5 rounds of this mode.
            progress = self.closed_progress if mode == "closed" else self.lp_progress
            stalled = stalled + 1 if len(history) > 5 and bound - history[-6] <= progress * max(1.0, abs(bound)) else 0
            best_bound = max(best_bound, bound)
            if self.closed():
                break
            z_lp = np.clip(np.asarray(self.z.X), 0.0, 1.0)
            th = np.asarray(self.theta.X)
            self.lp_point = z_lp
            if self.reduction and self.ub < INF:
                self.reduce(bound)  # reduced costs of this LP; cheap unless enough hubs close to rebuild the cuts
            sep = lam * z_lp + (1.0 - lam) * core
            t = time.perf_counter()
            cut = None
            if mode == "lp" and np.count_nonzero(sep > 1e-9) <= self.max_support:
                cut = self.cuts.lp_cut(sep, self.env, self.max_support, deadline=min(root_deadline, self.deadline))
                self.stats["lp_rounds"] += cut is not None
            if cut is None:
                cut = self.separate(sep, z_lp, th)
            self.stats["sub_seconds"] += time.perf_counter() - t
            viol = cut["const"] - cut["coef"] @ z_lp - th
            add = np.nonzero(viol > 1e-6 * np.maximum(1.0, np.abs(cut["const"])))[0]
            for i in add:
                self.rows.append(md.addConstr(self.cut_expr(cut, int(i)) >= cut["const"][i]))
            self.stats["root_cuts"] += len(add)
            self.stats["cuts"] += len(add)
            core = 0.5 * core + 0.5 * sep
            # LP-guided incumbent: open the hubs with z_lp >= 0.5 (or the largest one), then local search.
            if self.stats["root_rounds"] % 5 == 1:
                zr = z_lp >= 0.5
                if not zr.any():
                    zr[int(np.argmax(z_lp))] = True
                if zr.tobytes() not in self.pool:
                    t = time.perf_counter()
                    self.improve(zr, min(self.deadline, time.perf_counter() + 0.02 * max(0.0, self.left())), "lp_rounding")
                    self.stats["heuristic_seconds"] += time.perf_counter() - t
            if len(add) == 0 or stalled >= 3:
                if lam < 1.0:
                    lam, stalled, history = 1.0, 0, []  # finish this mode with cuts at the LP point
                    continue
                if mode == "closed" and self.exact_lp:
                    mode, lam, stalled, history = "lp", self.lp_lambda, 0, []  # exact LP separation on the support
                    continue
                break
        # Final LP: bound, reduction test, drop loose cuts.
        if self.left() > 0:
            md.Params.TimeLimit = max(0.01, self.left())
            md.optimize()
            if md.Status == GRB.OPTIMAL:
                self.publish_bound(float(md.ObjVal), "root_lp")
                self.lp_point = np.clip(np.asarray(self.z.X), 0.0, 1.0)
                if self.reduction and self.ub < INF:
                    self.reduce(float(md.ObjVal))
                loose = [c for c in self.rows if c.Slack < -1e-6 * max(1.0, abs(c.RHS))]
                md.remove(loose)
                gone = set(id(c) for c in loose)
                self.rows = [c for c in self.rows if id(c) not in gone]
        self.stats["root_bound"] = self.lb
        self.stats["root_seconds"] = time.perf_counter() - t_root

    def separate(self, sep: np.ndarray, z_lp: np.ndarray, th: np.ndarray) -> dict[str, Any]:
        """Closed-form separation at a fractional point: the ownership cut with the critical dual value, and the
        integer-point cuts (half-split duals) of the roundings {sep >= 1/2} and the p largest sep values
        (p = |incumbent|), each with the dual value chosen to maximize the cut at sep; per origin the cut most
        violated at the LP point is kept."""
        n = self.inst.n
        cands = [self.cuts.fractional_cut(sep, self.max_group)]
        order = np.argsort(-np.where(self.cand, sep, -1.0), kind="stable")
        p = int(self.best.sum()) if self.best is not None else 1
        sets = [sep >= 0.5]
        for q in (p,):
            zz = np.zeros(n, dtype=bool)
            zz[order[:min(q, int(self.cand.sum()))]] = True
            sets.append(zz)
        seen: set[bytes] = set()
        for zz in sets:
            zz = zz & self.cand
            if not zz.any() or zz.tobytes() in seen:
                continue
            seen.add(zz.tobytes())
            cands.append(self.cuts.binary_cut(zz, sep))
        viol = np.stack([cd["const"] - cd["coef"] @ z_lp - th for cd in cands])  # (k, n)
        pick = np.argmax(viol, axis=0)
        const = np.array([cands[k]["const"][i] for i, k in enumerate(pick)])
        coef = np.stack([cands[k]["coef"][i] for i, k in enumerate(pick)])
        return {"const": const, "coef": coef}

    def reduce(self, lp_value: float) -> None:
        """CCL Proposition 4: close hub h for good if z_h is nonbasic at 0 and LP + reduced cost > UB."""
        rc = np.asarray(self.z.RC)
        vb = np.asarray(self.master.getAttr("VBasis", self.zv))
        margin = 1e-6 * max(1.0, abs(self.ub))
        fix = self.cand & (vb == -1) & (lp_value + rc > self.ub + margin)
        if self.best is not None:
            fix &= ~self.best
        if not fix.any():
            return
        idx = np.nonzero(fix)[0]
        for h in idx:
            self.zv[h].UB = 0.0
        self.cand &= ~fix
        self.stats["fixed_hubs"] += len(idx)
        self.stats["candidates"] = int(self.cand.sum())
        # Rebuild the cut machinery on the remaining candidates when it pays off (paths via closed hubs vanish).
        if int((self.cuts.cand & ~self.cand).sum()) >= 0.1 * len(self.cuts.Hc) and self.left() > 0:
            t = time.perf_counter()
            try:
                self.cuts = HubCuts(self.inst, self.cand, self.hub_rank(), half_size=self.half_size or self.inst.n,
                                    deadline=self.deadline)
                self.theta.LB = self.cuts.lb_origin
            except TimeoutError:
                pass  # keep the current machinery: its cuts stay valid with the extra hubs closed
            self.stats["setup_seconds"] += time.perf_counter() - t

    # ------------------------------------------------------------ branch-and-cut

    def branch_and_cut(self) -> str:
        inst, md = self.inst, self.master
        n = inst.n
        self.z.VType = GRB.BINARY
        md.Params.LazyConstraints = 1
        md.Params.PreCrush = 1
        md.Params.MIPGap = 0.0
        if self.best is not None:
            _, per = objective(inst, self.best)
            self.z.Start = self.best.astype(float)
            self.theta.Start = per
            self.pending = False
        user_rounds = [0]
        errors: list[BaseException] = []

        def callback(cb: gp.Model, where: int) -> None:
            # gurobipy prints and ignores exceptions raised in a callback, which would accept an unchecked solution;
            # record the error, stop the search and re-raise after optimize.
            try:
                body(cb, where)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                cb.terminate()

        def body(cb: gp.Model, where: int) -> None:
            if where == GRB.Callback.MIPSOL:
                self.stats["mipsol_calls"] += 1
                if self.left() <= 0:
                    cb.terminate()
                    return
                zz = np.asarray(cb.cbGetSolution(self.zv)) > 0.5
                th = np.asarray(cb.cbGetSolution(self.tv))
                if not zz.any() or np.any(zz & ~self.cand):
                    raise RuntimeError("master returned an empty or non-candidate hub set")
                cut = self.binary_cut(zz)
                cost = float(inst.f[zz].sum() + cut["Q"].sum())
                if self.offer(zz, cost, "branch_and_cut") and self.left() > 0:
                    t = time.perf_counter()
                    self.improve(zz, min(self.deadline, time.perf_counter() + 0.02 * max(0.0, self.left())), "bc_local_search")
                    self.stats["heuristic_seconds"] += time.perf_counter() - t
                viol = cut["const"] - cut["coef"] @ zz - th
                for i in np.nonzero(viol > 1e-6 * np.maximum(1.0, np.abs(cut["Q"])))[0]:
                    cb.cbLazy(self.cut_expr(cut, int(i)) >= cut["const"][i])
                    self.stats["lazy_cuts"] += 1
                    self.stats["cuts"] += 1
            elif where == GRB.Callback.MIPNODE:
                if cb.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                    return
                if self.pending and self.best is not None:
                    _, per = objective(inst, self.best)
                    cb.cbSetSolution(self.zv, self.best.astype(float).tolist())
                    cb.cbSetSolution(self.tv, per.tolist())
                    cb.cbUseSolution()
                    self.pending = False
                    self.stats["injected"] += 1
                if cb.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0 and user_rounds[0] < self.root_user_rounds and self.left() > 0:
                    user_rounds[0] += 1
                    zn = np.clip(np.asarray(cb.cbGetNodeRel(self.zv)), 0.0, 1.0)
                    th = np.asarray(cb.cbGetNodeRel(self.tv))
                    t = time.perf_counter()
                    cut = self.cuts.fractional_cut(zn, self.max_group)
                    self.stats["sub_seconds"] += time.perf_counter() - t
                    viol = cut["const"] - cut["coef"] @ zn - th
                    for i in np.nonzero(viol > 1e-6 * np.maximum(1.0, np.abs(cut["const"])))[0]:
                        cb.cbCut(self.cut_expr(cut, int(i)) >= cut["const"][i])
                        self.stats["user_cuts"] += 1
                        self.stats["cuts"] += 1
            elif where == GRB.Callback.MIP:
                bnd = cb.cbGet(GRB.Callback.MIP_OBJBND)
                if bnd < GRB.INFINITY:
                    self.publish_bound(float(bnd), "branch_and_cut")
                if self.left() <= 0:
                    cb.terminate()

        t = time.perf_counter()
        md.Params.TimeLimit = max(0.01, self.left())
        md.optimize(callback)
        self.stats["master_seconds"] += time.perf_counter() - t
        if errors:
            raise errors[0]
        self.stats["node_count"] = float(md.NodeCount)
        try:
            self.publish_bound(float(md.ObjBound), "final")
        except (gp.GurobiError, AttributeError):
            pass
        if md.Status == GRB.OPTIMAL:
            return "optimal"
        if md.Status == GRB.MEM_LIMIT:
            return "memory_limit"
        return "time_limit"
