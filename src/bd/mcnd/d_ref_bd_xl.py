"""Engineered Benders reference for the MCND (scale-up study; the paper's references are unchanged).

Master: min sum_a f_a y_a + sum_k theta_k over y in {0, 1}^m with one recourse variable per commodity
(theta_tot = sum_k theta_k carries the aggregate cuts). Cuts come from node potentials pi of the STRONG recourse
(linking bounds x_ak <= min(d_k, u_a) y_a, flows into o_k and out of t_k fixed at 0), with the arc coefficients
h_a(pi) = min_t u_a t + sum_k L_ak (g_ak - t)^+ chosen as the cheapest dual-feasible completion (c_naive_bd), so
closed arcs never get the arbitrary degenerate duals of the LP. Every pi gives
    aggregate cut       theta_tot >= sum_k d_k (pi_k[o_k] - pi_k[t_k]) - sum_a h_a(pi) y_a
    commodity cuts      theta_k   >= d_k (pi_k[o_k] - pi_k[t_k]) - sum_a L_ak g_ak y_a    (g with costs c)
    feasibility cut     sum_a h_a(pi) y_a >= sum_k d_k (pi_k[o_k] - pi_k[t_k])          (Farkas ray, costs 0).
Commodity cuts are valid because each commodity's cost in any feasible flow is at least its single-commodity
min-cost flow with capacities L_ak y_a, and (pi_k, g_k) is dual feasible for it; they disaggregate the bound.
With the strong recourse, the converged root master LP equals the LP relaxation of the strong compact formulation.

Master-only valid inequalities (valid for every feasible design; no flow variables needed):
- commodity cut-sets  sum_{a in out(o_k)} L_ak y_a >= d_k  and  sum_{a in in(t_k)} L_ak y_a >= d_k;
- cut-set inequalities for node sets S with crossing demand D(S) (commodities with o_k in S, t_k not in S):
      sum_{a in delta+(S)} min(u_a, D) y_a >= D,   its cardinality version sum y_a >= r(S) (fewest arcs whose
      min(u_a, D) reach D) and its mixed-integer rounding (MIR) with the most violated divisor;
  singletons and their complements are added up front; general S are separated at fractional master points
  (greedy node-by-node growth from every node, both directions, and connected components of the arcs with
  y_a >= threshold), in the root loop as rows and at the branch-and-cut root node as user cuts; when an integer
  design disconnects a commodity, the reachable set gives lazy cut-set cuts;
- theta_k >= d_k * (shortest o_k-t_k path with costs c over all arcs).

Stages, all within one wall-clock budget:
1. The shared basic initializer (one slope-scaling step, identical to the naive control), then slope scaling
   (Crainic, Gendron, Hernu 2004); every design is evaluated exactly, cleaned of arcs without flow, published,
   and its cuts seed the master.
2. Root cutting-plane loop on the master LP with in-out stabilization (separation at lam*y_lp + (1-lam)*core,
   core moved toward feasible separation points; lam -> 1 when the stabilized point gives no violated cut or the
   bound stalls), aggregate + commodity cuts from the persistent strong recourse LP, cut-set separation;
   non-binding Benders cuts are dropped at the end.
3. LP-guided primal heuristics: designs read off the root LP point (arcs carrying recourse flow, thresholds on y),
   slope scaling restarted from the LP slopes c_a + f_a y_a / x_a, and an arc-drop local search on the incumbent
   (close the open arc with the largest fixed cost per unit of flow, re-solve the recourse, keep improvements).
4. Gurobi branch-and-cut on the master (one tree): at every integer solution the strong recourse is solved on the
   open arcs only; the design is evaluated, cleaned and published; violated aggregate/commodity cuts, or
   connectivity cut-set and Farkas cuts, are added as lazy constraints. A new tree incumbent gets a short arc-drop
   search; improved designs are injected back into the tree. The master bound is published through the MIP callback.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.mcnd.a_data import MCNDInstance, load_dow
from src.bd.mcnd.b_mip_xl import useless_arcs
from src.bd.mcnd.c_naive_bd import (FLOW_EPS, INF, RecourseLP, SlopeScaling, arc_excess, certify, cut_constant, gather,
                                    initial_design, is_memory_error, make_env, strong_arc_coefficients)


# ---------------------------------------------------------------- helpers

def shortest_paths(inst: MCNDInstance) -> np.ndarray:
    """All-pairs shortest path costs with unit costs c over all arcs (Floyd-Warshall)."""
    D = np.full((inst.n, inst.n), INF)
    np.fill_diagonal(D, 0.0)
    np.minimum.at(D, (inst.tail, inst.head), inst.cost)
    for k in range(inst.n):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    return D


def cardinality(caps: np.ndarray, demand: float) -> int:
    """Fewest values of caps (largest first) that sum to at least demand (len + 1 if impossible)."""
    s = np.cumsum(np.sort(caps)[::-1])
    hit = np.nonzero(s >= demand - 1e-9)[0]
    return int(hit[0]) + 1 if len(hit) else len(caps) + 1


def mir_row(coef: np.ndarray, rhs: float, y: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """Most violated mixed-integer rounding of sum coef_a y_a >= rhs (coef >= 0, y integer >= 0) at the point y:
    for a divisor lam with beta = rhs / lam fractional (f = frac(beta)),
        sum_a (floor(coef_a / lam) + min(1, frac(coef_a / lam) / f)) y_a >= ceil(beta).
    Divisors: the coefficients of the most fractional y_a and the largest coefficient. Returns (row, rhs, violation)."""
    frac = (y > 1e-6) & (y < 1.0 - 1e-6)
    # Coefficients of the (at most 8) most fractional arcs, plus the largest coefficient.
    most = np.nonzero(frac)[0][np.argsort(np.abs(y[frac] - 0.5), kind="stable")[:8]]
    lams = np.unique(np.concatenate([coef[most], [coef.max()]])) if len(coef) else np.zeros(0)
    best = None
    for lam in lams:
        if lam <= 1e-9:
            continue
        beta = rhs / lam
        fl = math.floor(beta + 1e-9)
        f = beta - fl
        if f < 1e-6 or f > 1.0 - 1e-6:
            continue  # integral (up to rounding): the rounded right-hand side would not be safe
        a = coef / lam
        fa = np.clip(a - np.floor(a + 1e-12), 0.0, 1.0)
        g = np.floor(a + 1e-12) + np.minimum(1.0, fa / f)
        r = float(fl + 1)
        viol = (r - g @ y) / max(1.0, r)
        if best is None or viol > best[2]:
            best = (g, r, viol)
    return best


def reachable(inst: MCNDInstance, y: np.ndarray, source: int) -> np.ndarray:
    seen = np.zeros(inst.n, dtype=bool)
    seen[source] = True
    frontier = np.array([source])
    t, h = inst.tail[y], inst.head[y]
    while len(frontier):
        nxt = np.unique(h[np.isin(t, frontier)])
        nxt = nxt[~seen[nxt]]
        seen[nxt] = True
        frontier = nxt
    return seen


def components(n: int, tail: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Connected-component label of every node in the undirected graph of the given arcs."""
    label = np.arange(n)
    for _ in range(n):
        new = label.copy()
        np.minimum.at(new, tail, label[head])
        np.minimum.at(new, head, label[tail])
        new = new[new]
        if np.array_equal(new, label):
            break
        label = new
    return label


def crossing_demand(inst: MCNDInstance, S: np.ndarray) -> float:
    return float(inst.demand[S[inst.orig] & ~S[inst.dest]].sum())


class ReducedRecourse:
    """Strong recourse of an integer design, solved as a fresh LP over the open arcs only."""

    def __init__(self, inst: MCNDInstance, env: gp.Env) -> None:
        self.inst, self.env = inst, env
        self.L = np.where(useless_arcs(inst), 0.0, inst.link_bound())
        self.b = inst.supply()
        self.solves = 0
        self.seconds = 0.0

    def solve(self, y: np.ndarray, time_limit: float) -> dict[str, Any]:
        t0 = time.perf_counter()
        inst, K = self.inst, self.inst.K
        arcs = np.nonzero(y)[0]
        model = gp.Model(env=self.env)
        model.Params.OutputFlag = 0
        model.Params.InfUnbdInfo = 1
        model.Params.TimeLimit = max(0.01, time_limit)
        x = model.addMVar((K, len(arcs)), lb=0.0, ub=self.L[:, arcs])
        tail, head = inst.tail[arcs], inst.head[arcs]
        rows: list = []
        for i in range(inst.n):
            o, d = np.nonzero(tail == i)[0], np.nonzero(head == i)[0]
            if not len(o) and not len(d):
                rows.append(None)
                if np.any(self.b[:, i] != 0):
                    model.dispose()
                    return {"status": "disconnected"}
                continue
            expr = 0
            if len(o):
                expr = x[:, o].sum(axis=1)
            if len(d):
                expr = expr - x[:, d].sum(axis=1)
            rows.append(model.addConstr(expr == self.b[:, i]))
        model.addConstr(x.sum(axis=0) <= inst.cap[arcs])
        model.setObjective((inst.cost[arcs][None, :] * x).sum(), GRB.MINIMIZE)
        model.optimize()
        st = model.Status
        if st == GRB.INF_OR_UNBD:  # bounded variables: infeasible; re-solve without dual reductions for the ray
            model.Params.DualReductions = 0
            model.optimize()
            st = model.Status
        if st == GRB.OPTIMAL:
            flows = np.zeros((K, inst.m))
            flows[:, arcs] = np.where(np.asarray(x.X) < 1e-11, 0.0, np.asarray(x.X))
            out = {"status": "optimal", "obj": float(model.ObjVal), "pi": gather(rows, "Pi", K), "flows": flows}
        elif st == GRB.INFEASIBLE:
            # Gurobi's Farkas certificate: the dual ray is pi = -FarkasDual (see RecourseLP).
            out = {"status": "infeasible", "pi": -gather(rows, "FarkasDual", K)}
        else:
            out = {"status": "time_limit" if st == GRB.TIME_LIMIT else f"status_{st}"}
        model.dispose()
        self.solves += 1
        self.seconds += time.perf_counter() - t0
        return out


# ---------------------------------------------------------------- reference

class MCNDBendersXL:
    """Engineered Benders branch-and-cut reference for the MCND; see the module docstring."""

    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, soft_mem_gb: float | None = None, seed: int = 0,
                 ss_fraction: float = 0.08, ss_steps: int = 40, root_fraction: float = 0.3, root_rounds: int = 400,
                 heuristic_fraction: float = 0.1, inout_lambda: float = 0.3, commodity_cuts: bool = True,
                 max_commodity_cuts: int | None = None, cutsets: bool = True, max_cutsets: int = 60,
                 bc_drop_evals: int = 10, verbose: bool = False) -> None:
        self.time_limit = time_limit
        self.threads = threads
        self.on_incumbent = on_incumbent or (lambda design: None)  # receives the list of open arc indices
        self.on_bound = on_bound or (lambda bound: None)  # receives every new valid global lower bound
        self.soft_mem_gb = soft_mem_gb
        self.seed = seed
        self.ss_fraction, self.ss_steps = ss_fraction, ss_steps
        self.root_fraction, self.root_rounds = root_fraction, root_rounds
        self.heuristic_fraction = heuristic_fraction
        self.inout_lambda = inout_lambda
        self.commodity_cuts = commodity_cuts
        self.max_commodity_cuts = max_commodity_cuts
        self.cutsets, self.max_cutsets = cutsets, max_cutsets
        self.bc_drop_evals = bc_drop_evals  # arc-drop evaluations after each new branch-and-cut incumbent
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

    def offer(self, y: np.ndarray, res: dict[str, Any], stage: str) -> bool:
        """Clean a feasible design (close arcs without flow) and publish it if it improves the incumbent."""
        inst = self.inst
        flows = res["flows"]
        used = y & (flows.sum(axis=0) > FLOW_EPS)
        cost = float(inst.fixed[used].sum() + (inst.cost[None, :] * flows).sum())
        if cost < self.ub - 1e-9:
            self.ub, self.best, self.best_flows = cost, used.copy(), flows
            self.pending = True
            self.on_incumbent(np.nonzero(used)[0].tolist())
            self.log(stage)
            return True
        return False

    def closed(self) -> bool:
        return self.ub < INF and self.lb >= self.ub - 1e-6 * max(1.0, abs(self.ub))

    # ------------------------------------------------------------ cut data

    def cut_data(self, pi: np.ndarray, feasibility: bool) -> dict[str, Any]:
        inst = self.inst
        G = arc_excess(inst, pi, 0.0 if feasibility else inst.cost, self.skip)
        consts = cut_constant(inst, pi)
        out = {"const": float(consts.sum()), "h": strong_arc_coefficients(G, inst.cap, self.L)}
        if not feasibility and self.commodity_cuts:
            out["consts"] = consts
            out["P"] = self.L * G
        return out

    def violated_commodities(self, cd: dict[str, Any], y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        if "P" not in cd:
            return np.zeros(0, dtype=int)
        viol = cd["consts"] - cd["P"] @ y - theta
        ks = np.nonzero(viol > 1e-6 * np.maximum(1.0, np.abs(cd["consts"])))[0]
        if self.max_commodity_cuts is not None and len(ks) > self.max_commodity_cuts:
            ks = ks[np.argsort(-viol[ks])[:self.max_commodity_cuts]]
        return ks

    # ------------------------------------------------------------ run

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        return self.solve(load_dow(data_path))

    def solve(self, inst: MCNDInstance) -> dict[str, Any]:
        if not hasattr(self, "t0"):
            self.t0 = time.perf_counter()
            self.deadline = self.t0 + self.time_limit
        self.inst = inst
        self.skip = useless_arcs(inst)
        self.L = np.where(self.skip, 0.0, inst.link_bound())
        self.ub, self.lb = INF, -INF
        self.best: np.ndarray | None = None
        self.best_flows: np.ndarray | None = None
        self.pending = False
        self.stats = {"iterations": 0, "root_rounds": 0, "ss_steps": 0, "cuts": 0, "aggregate_cuts": 0, "commodity_cuts": 0,
                      "feasibility_cuts": 0, "cutset_cuts": 0, "lazy_cuts": 0, "user_cuts": 0, "mipsol_calls": 0, "injected": 0,
                      "evaluations": 0, "drop_moves": 0, "master_seconds": 0.0, "sub_seconds": 0.0, "heuristic_seconds": 0.0,
                      "root_seconds": 0.0, "root_bound": None, "root_cuts_total": 0, "root_cuts_kept": 0, "node_count": 0.0}
        reason = "time_limit"
        env = make_env(self.threads, self.soft_mem_gb)
        self.env = env
        self.ss: SlopeScaling | None = None
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
                "termination_reason": reason, "open": np.nonzero(self.best)[0].tolist() if self.best is not None else None,
                "stats": self.stats, **self.stats, "total_seconds": self.elapsed(), "trace": self.trace}

    def search(self) -> str:
        inst = self.inst
        self.publish_bound(0.0, "trivial")
        self.reduced = ReducedRecourse(inst, self.env)
        self.build_master()
        # 1. Shared initializer, then slope scaling.
        t = time.perf_counter()
        self.ss = SlopeScaling(inst, self.env)
        y0 = initial_design(self.ss, self.left())
        if y0 is None:
            return "time_limit"
        self.evaluate(y0, "initializer", seed_cuts="all")
        self.slope_scaling(time.perf_counter() + self.ss_fraction * max(0.0, self.left()), {y0.tobytes()})
        self.stats["heuristic_seconds"] += time.perf_counter() - t
        if self.closed():
            return "optimal"
        # 2. Root cutting-plane loop; 3. LP-guided heuristics.
        if self.left() > 0:
            y_lp, flows_lp = self.root_loop()
            if self.closed():
                return "optimal"
            if y_lp is not None and self.left() > 0:
                t = time.perf_counter()
                self.lp_heuristics(y_lp, flows_lp, time.perf_counter() + self.heuristic_fraction * max(0.0, self.left()))
                self.stats["heuristic_seconds"] += time.perf_counter() - t
        self.ss.dispose()
        if self.closed():
            return "optimal"
        # 4. Branch-and-cut.
        if self.left() <= 0.01:
            return "time_limit"
        return self.branch_and_cut()

    # ------------------------------------------------------------ master

    def build_master(self) -> None:
        inst = self.inst
        K, m = inst.K, inst.m
        md = gp.Model(env=self.env)
        md.Params.OutputFlag = 0
        md.Params.Seed = self.seed
        self.master = md
        self.y = md.addMVar(m, lb=0.0, ub=1.0, name="y")
        sp = shortest_paths(inst)[inst.orig, inst.dest]
        self.theta = md.addMVar(K, lb=inst.demand * np.where(np.isfinite(sp), sp, 0.0), name="theta")
        self.theta_tot = md.addVar(lb=0.0, name="theta_tot")
        md.addConstr(self.theta_tot == self.theta.sum())
        md.setObjective(inst.fixed @ self.y + self.theta_tot, GRB.MINIMIZE)
        self.yv = self.y.tolist()
        self.tv = self.theta.tolist()
        self.cuts: list[gp.Constr] = []
        self.cutset_keys: set[tuple] = set()
        # Commodity cut-sets at the origin and destination.
        for k in range(K):
            for arcs in (np.nonzero(inst.tail == inst.orig[k])[0], np.nonzero(inst.head == inst.dest[k])[0]):
                md.addConstr(gp.LinExpr(self.L[k, arcs].tolist(), [self.yv[a] for a in arcs]) >= inst.demand[k])
        # Node cut-sets out of every singleton and into every singleton, with their cardinality version.
        for i in range(inst.n):
            S = np.zeros(inst.n, dtype=bool)
            S[i] = True
            for T in (S, ~S):
                for arcs, coef, rhs, _ in self.cutset_rows(T, None):
                    md.addConstr(gp.LinExpr(coef.tolist(), [self.yv[a] for a in arcs]) >= rhs)
                    self.stats["cutset_cuts"] += 1
        # Arc-commodity data for cut-set separation.
        self.Dm = np.zeros((inst.n, inst.n))
        np.add.at(self.Dm, (inst.orig, inst.dest), inst.demand)

    def cutset_rows(self, S: np.ndarray, y: np.ndarray | None) -> list[tuple[np.ndarray, np.ndarray, float, str]]:
        """Cut-set rows of delta+(S): capacity, cardinality and (with a point y) the most violated MIR."""
        inst = self.inst
        D = crossing_demand(inst, S)
        if D <= 0:
            return []
        arcs = np.nonzero(S[inst.tail] & ~S[inst.head])[0]
        caps = np.minimum(inst.cap[arcs], D)
        rows = [(arcs, caps, D, "cap")]
        r = cardinality(caps, D)
        if r > 1:
            rows.append((arcs, np.ones(len(arcs)), float(r), "card"))
        if y is not None and len(arcs):
            mir = mir_row(caps, D, y[arcs])
            if mir is not None and mir[2] > 1e-4:
                rows.append((arcs, mir[0], mir[1], "mir"))
        return rows

    def separate_cutsets(self, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Violated cut-set rows at a fractional point y, most violated first (at most max_cutsets)."""
        inst = self.inst
        n = inst.n
        W = np.zeros((n, n))
        np.add.at(W, (inst.tail, inst.head), inst.cap * y)
        cands: dict[bytes, np.ndarray] = {}
        for Wd, Dd, flip in ((W, self.Dm, False), (W.T, self.Dm.T, True)):
            # Greedy growth: add the node that minimizes capacity minus crossing demand of delta+(S).
            out_w, out_d = Wd.sum(axis=1), Dd.sum(axis=1)
            for i in range(n):
                S = np.zeros(n, dtype=bool)
                S[i] = True
                cap_s = out_w[i] - Wd[i, i]
                dem_s = out_d[i] - Dd[i, i]
                for _ in range(max(1, n // 2) - 1):
                    # Effect of adding v: arcs S->v leave the cut, arcs v->(not S) enter it.
                    into_v_w = Wd[S].sum(axis=0)
                    into_v_d = Dd[S].sum(axis=0)
                    from_v_w = out_w - Wd[:, S].sum(axis=1)
                    from_v_d = out_d - Dd[:, S].sum(axis=1)
                    slack = (cap_s - into_v_w + from_v_w - np.diag(Wd)) - (dem_s - into_v_d + from_v_d - np.diag(Dd))
                    slack[S] = INF
                    v = int(np.argmin(slack))
                    if not np.isfinite(slack[v]):
                        break
                    cap_s = cap_s - into_v_w[v] + from_v_w[v] - Wd[v, v]
                    dem_s = dem_s - into_v_d[v] + from_v_d[v] - Dd[v, v]
                    S[v] = True
                    T = ~S if flip else S
                    cands[T.tobytes()] = T.copy()
        # Connected components of the arcs with y above a threshold, and their complements.
        for tau in (0.999, 0.7, 0.4, 0.15, 0.02):
            sel = y >= tau
            label = components(n, inst.tail[sel], inst.head[sel])
            for c in np.unique(label):
                S = label == c
                if 1 < S.sum() < n:
                    for T in (S, ~S):
                        cands[T.tobytes()] = T.copy()
        found = []
        inst_cap, orig, dest, dem = inst.cap, inst.orig, inst.dest, inst.demand
        for key, S in cands.items():
            D = float(dem[S[orig] & ~S[dest]].sum())
            if D <= 0:
                continue
            arcs = np.nonzero(S[inst.tail] & ~S[inst.head])[0]
            # Only near-tight sets get the cardinality and MIR rows (the capacity row with slack < 50% of D).
            if np.minimum(inst_cap[arcs], D) @ y[arcs] > 1.5 * D:
                continue
            for arcs, coef, rhs, kind in self.cutset_rows(S, y):
                if (key, kind) in self.cutset_keys:
                    continue
                viol = (rhs - coef @ y[arcs]) / max(1.0, rhs)
                if viol > 1e-3:
                    found.append((viol, key, kind, arcs, coef, rhs))
        found.sort(key=lambda r: -r[0])
        out = []
        for viol, key, kind, arcs, coef, rhs in found[:self.max_cutsets]:
            self.cutset_keys.add((key, kind))
            out.append((arcs, coef, rhs))
        return out

    def add_rows(self, rows: list[tuple[np.ndarray, np.ndarray, float]], cb_model: gp.Model | None = None, lazy: bool = False) -> int:
        for arcs, coef, rhs in rows:
            expr = gp.LinExpr(coef.tolist(), [self.yv[a] for a in arcs])
            if cb_model is None:
                self.master.addConstr(expr >= rhs)
            elif lazy:
                cb_model.cbLazy(expr >= rhs)
                self.stats["lazy_cuts"] += 1
            else:
                cb_model.cbCut(expr >= rhs)
                self.stats["user_cuts"] += 1
        self.stats["cutset_cuts"] += len(rows)
        return len(rows)

    def add_cut(self, cd: dict[str, Any], feasibility: bool, ks: np.ndarray | None = None,
                cb_model: gp.Model | None = None, aggregate: bool = True) -> int:
        """Add the aggregate (or feasibility) cut and the commodity cuts ks as rows, or as lazy constraints in a callback."""
        rows = []
        if aggregate:
            h = cd["h"]
            nz = np.nonzero(h > 1e-12)[0]
            expr = gp.LinExpr(h[nz].tolist(), [self.yv[a] for a in nz])
            if not feasibility:
                expr.add(self.theta_tot)
            rows.append(expr >= cd["const"])
            self.stats["feasibility_cuts" if feasibility else "aggregate_cuts"] += 1
        if ks is not None:
            for k in ks:
                p = cd["P"][k]
                nz = np.nonzero(p > 1e-12)[0]
                expr = gp.LinExpr(p[nz].tolist(), [self.yv[a] for a in nz])
                expr.add(self.tv[k])
                rows.append(expr >= float(cd["consts"][k]))
            self.stats["commodity_cuts"] += len(ks)
        for r in rows:
            if cb_model is not None:
                cb_model.cbLazy(r)
                self.stats["lazy_cuts"] += 1
            else:
                self.cuts.append(self.master.addConstr(r))
        self.stats["cuts"] += len(rows)
        return len(rows)

    # ------------------------------------------------------------ primal side

    def evaluate(self, y: np.ndarray, stage: str, seed_cuts: str | None) -> dict[str, Any]:
        """Exact evaluation of an integer design: publish it (cleaned); seed the master with its cuts
        (seed_cuts "all": aggregate and every commodity cut, "aggregate": the aggregate cut only, None: no cuts).
        Never called with seed_cuts inside a Gurobi callback (rows cannot be added during optimize)."""
        t = time.perf_counter()
        res = self.reduced.solve(y, self.left())
        self.stats["sub_seconds"] += time.perf_counter() - t
        self.stats["evaluations"] += 1
        if res["status"] == "optimal":
            self.offer(y, res, stage)
            if seed_cuts:
                cd = self.cut_data(res["pi"], feasibility=False)
                self.add_cut(cd, False, np.arange(self.inst.K) if "P" in cd and seed_cuts == "all" else None)
        elif res["status"] == "infeasible" and seed_cuts:
            self.add_cut(self.cut_data(res["pi"], feasibility=True), True)
        return res

    def slope_scaling(self, deadline: float, seen: set[bytes], max_steps: int | None = None, seed_cuts: str = "all") -> None:
        steps = 0
        while steps < (max_steps or self.ss_steps) and time.perf_counter() < deadline and self.left() > 0:
            flow = self.ss.step(min(self.left(), deadline - time.perf_counter()))
            if flow is None:
                break
            steps += 1
            self.stats["ss_steps"] += 1
            y = flow > FLOW_EPS
            key = y.tobytes()
            if key in seen:
                break  # slope scaling has cycled
            seen.add(key)
            self.evaluate(y, "slope_scaling", seed_cuts=seed_cuts)

    def lp_heuristics(self, y_lp: np.ndarray, flows_lp: np.ndarray | None, deadline: float) -> None:
        inst = self.inst
        xbar = flows_lp.sum(axis=0) if flows_lp is not None else None
        seen: set[bytes] = set()
        cands = ([xbar > FLOW_EPS] if xbar is not None else []) + [y_lp >= tau for tau in (0.5, 0.2, 0.05)]
        for y in cands:
            if time.perf_counter() >= deadline or self.left() <= 0:
                return
            if y.tobytes() not in seen:
                seen.add(y.tobytes())
                self.evaluate(y, "lp_rounding", seed_cuts=None)
        if xbar is not None:
            # Slope scaling from the LP: unit costs that reproduce the LP cost f_a y_a + c_a x_a on arcs with flow.
            self.ss.rho = np.where(xbar > FLOW_EPS, inst.cost + inst.fixed * y_lp / np.maximum(xbar, FLOW_EPS),
                                   inst.cost + inst.fixed / inst.cap)
            # After the root clean-up only aggregate cuts are seeded, so the branch-and-cut master stays small.
            self.slope_scaling(min(deadline, time.perf_counter() + 0.5 * max(0.0, deadline - time.perf_counter())), seen,
                               seed_cuts="aggregate")
        self.drop_search(deadline)

    def drop_search(self, deadline: float, batch: int = 25, max_evals: int | None = None) -> None:
        """Close the open arc with the largest fixed cost per unit of flow; keep the first improving move."""
        tried: set[tuple[bytes, int]] = set()
        evals = 0
        while self.best is not None and time.perf_counter() < deadline and self.left() > 0:
            y = self.best.copy()
            xbar = self.best_flows.sum(axis=0)
            open_ = np.nonzero(y)[0]
            order = open_[np.argsort(-self.inst.fixed[open_] / np.maximum(xbar[open_], FLOW_EPS), kind="stable")]
            key = y.tobytes()
            improved = False
            for a in [a for a in order if (key, int(a)) not in tried][:batch]:
                if time.perf_counter() >= deadline or self.left() <= 0 or (max_evals is not None and evals >= max_evals):
                    return
                evals += 1
                tried.add((key, int(a)))
                y2 = y.copy()
                y2[a] = False
                res = self.evaluate(y2, "drop", seed_cuts=None)
                if res["status"] == "optimal" and self.ub < INF and self.best is not None and not np.array_equal(self.best, y):
                    self.stats["drop_moves"] += 1
                    improved = True
                    break
            if not improved:
                return

    # ------------------------------------------------------------ root

    def root_loop(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        inst, md = self.inst, self.master
        t_root = time.perf_counter()
        root_deadline = t_root + self.root_fraction * max(0.0, self.left())
        rec = RecourseLP(inst, self.env, strong=True)
        core = 0.5 * (self.best.astype(float) + 1.0) if self.best is not None else np.ones(inst.m)
        lam = self.inout_lambda
        best_bound, stalled = -INF, 0
        n_before = len(self.cuts)
        y_lp = None
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
            stalled = stalled + 1 if bound <= best_bound + 1e-5 * max(1.0, abs(bound)) else 0
            best_bound = max(best_bound, bound)
            if self.closed():
                break
            y_lp = np.clip(np.asarray(self.y.X), 0.0, 1.0)
            th = np.asarray(self.theta.X)
            th_tot = float(self.theta_tot.X)
            added = self.add_rows(self.separate_cutsets(y_lp)) if self.cutsets else 0
            sep = lam * y_lp + (1.0 - lam) * core
            t = time.perf_counter()
            res = rec.solve(sep, min(self.left(), max(0.01, root_deadline - time.perf_counter())))
            self.stats["sub_seconds"] += time.perf_counter() - t
            if res["status"] == "optimal":
                cd = self.cut_data(res["pi"], feasibility=False)
                if cd["const"] - cd["h"] @ y_lp - th_tot > 1e-6 * max(1.0, abs(cd["const"])):
                    added += self.add_cut(cd, False)
                ks = self.violated_commodities(cd, y_lp, th)
                if len(ks):
                    added += self.add_cut(cd, False, ks, aggregate=False)
                core = 0.5 * core + 0.5 * sep
            elif res["status"] == "infeasible":
                cd = self.cut_data(res["pi"], feasibility=True)
                if cd["const"] - cd["h"] @ y_lp > 1e-6 * max(1.0, abs(cd["const"])):
                    added += self.add_cut(cd, True)
            else:
                break  # out of time
            if added == 0 or stalled >= 5:
                if lam < 1.0:
                    lam, stalled = 1.0, 0  # finish with Kelley cuts at the LP point
                    continue
                break
        self.stats["root_cuts_total"] = len(self.cuts) - n_before
        flows_lp = None
        # Drop non-binding Benders cuts so the branch-and-cut master stays small; keep the final LP point.
        if self.left() > 0:
            md.Params.TimeLimit = max(0.01, self.left())
            md.optimize()
            if md.Status == GRB.OPTIMAL:
                self.publish_bound(float(md.ObjVal), "root_lp")
                y_lp = np.clip(np.asarray(self.y.X), 0.0, 1.0)
                loose = [c for c in self.cuts if c.Slack < -1e-6 * max(1.0, abs(c.RHS))]
                md.remove(loose)
                gone = set(id(c) for c in loose)
                self.cuts = [c for c in self.cuts if id(c) not in gone]
                res = rec.solve(y_lp, self.left(), flows=True)
                if res["status"] == "optimal":
                    flows_lp = res["flows"]
        rec.dispose()
        self.stats["root_cuts_kept"] = len(self.cuts)
        self.stats["root_bound"] = self.lb
        self.stats["root_seconds"] = time.perf_counter() - t_root
        return y_lp, flows_lp

    # ------------------------------------------------------------ branch-and-cut

    def branch_and_cut(self) -> str:
        inst, md = self.inst, self.master
        self.y.VType = GRB.BINARY
        md.Params.LazyConstraints = 1
        md.Params.PreCrush = 1
        md.Params.MIPGap = 0.0
        if self.best is not None:
            self.y.Start = self.best.astype(float)
            per = (inst.cost[None, :] * self.best_flows).sum(axis=1)
            self.theta.Start = per
            self.theta_tot.Start = float(per.sum())
            self.pending = False
        cache: OrderedDict[bytes, dict[str, Any]] = OrderedDict()

        def evaluate(y: np.ndarray) -> dict[str, Any]:
            key = y.tobytes()
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            t = time.perf_counter()
            res = self.reduced.solve(y, self.left())
            self.stats["sub_seconds"] += time.perf_counter() - t
            self.stats["evaluations"] += 1
            if res["status"] in ("optimal", "infeasible"):
                # Potentials only (flows are K x m): a cached design was already offered when first evaluated.
                cache[key] = {k: v for k, v in res.items() if k != "flows"}
                while len(cache) > 300:
                    cache.popitem(last=False)
            return res

        errors: list[BaseException] = []

        def callback(cb: gp.Model, where: int) -> None:
            # gurobipy prints and ignores exceptions raised in a callback, which would accept an unchecked design;
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
                yy = np.asarray(cb.cbGetSolution(self.yv)) > 0.5
                th = np.asarray(cb.cbGetSolution(self.tv))
                th_tot = float(cb.cbGetSolution(self.theta_tot))
                # Disconnected commodities: cut-set cuts on the reachable sets.
                rows, done = [], set()
                for k in range(inst.K):
                    if int(inst.orig[k]) in done:
                        continue
                    done.add(int(inst.orig[k]))
                    S = reachable(inst, yy, int(inst.orig[k]))
                    if np.any(~S[inst.dest[inst.orig == inst.orig[k]]]):
                        rows += [(a, c, r) for a, c, r, kind in self.cutset_rows(S, None)]
                if rows:
                    self.add_rows(rows, cb_model=cb, lazy=True)
                    return
                res = evaluate(yy)
                if res["status"] == "optimal":
                    if "flows" in res and self.offer(yy, res, "branch_and_cut") and self.bc_drop_evals:
                        # Local improvement of a new tree incumbent; the result is injected at the next node.
                        self.drop_search(self.deadline, batch=self.bc_drop_evals, max_evals=self.bc_drop_evals)
                    cd = self.cut_data(res["pi"], feasibility=False)
                    if cd["const"] - cd["h"] @ yy - th_tot > 1e-6 * max(1.0, abs(res["obj"])):
                        self.add_cut(cd, False, cb_model=cb)
                    ks = self.violated_commodities(cd, yy.astype(float), th)
                    if len(ks):
                        self.add_cut(cd, False, ks, cb_model=cb, aggregate=False)
                elif res["status"] == "infeasible":
                    self.add_cut(self.cut_data(res["pi"], feasibility=True), True, cb_model=cb)
                elif res["status"] == "time_limit":
                    cb.terminate()
                else:  # a disconnected design is caught above; anything else would accept an unchecked design
                    raise RuntimeError(f"recourse status {res['status']} at an integer master solution")
            elif where == GRB.Callback.MIPNODE:
                if cb.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                    return
                if self.pending and self.best is not None:
                    per = (inst.cost[None, :] * self.best_flows).sum(axis=1)
                    cb.cbSetSolution(self.yv, self.best.astype(float).tolist())
                    cb.cbSetSolution(self.tv, per.tolist())
                    cb.cbSetSolution(self.theta_tot, float(per.sum()))
                    cb.cbUseSolution()
                    self.pending = False
                    self.stats["injected"] += 1
                if self.cutsets and cb.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
                    y_node = np.clip(np.asarray(cb.cbGetNodeRel(self.yv)), 0.0, 1.0)
                    self.add_rows(self.separate_cutsets(y_node), cb_model=cb, lazy=False)
            elif where == GRB.Callback.MIP:
                bnd = cb.cbGet(GRB.Callback.MIP_OBJBND)
                if bnd < GRB.INFINITY:
                    self.publish_bound(float(bnd), "branch_and_cut")

        t = time.perf_counter()
        md.Params.TimeLimit = max(0.01, self.left())
        md.optimize(callback)
        self.stats["master_seconds"] += time.perf_counter() - t
        if errors:
            raise errors[0]
        self.stats["node_count"] = float(md.NodeCount)
        if md.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.MEM_LIMIT):
            self.publish_bound(float(md.ObjBound), "final")
        if md.Status == GRB.OPTIMAL:
            return "optimal"
        if md.Status == GRB.MEM_LIMIT:
            return "memory_limit"
        if md.Status == GRB.INFEASIBLE and self.best is None:
            raise RuntimeError("Benders master infeasible: the instance has no feasible design")
        return "time_limit"
