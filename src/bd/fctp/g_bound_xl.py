"""FCTP at scale, bound side (protocol v4, Sections 3 and 6): generator of larger Glover-like instances, the compact
LP control and the Balinski LP-phase arm, with the compressed certificate "fctp-cert-c1". Phoenix only.

View: the canonical "fctp-v1" of src/experiments/xl_certificate.fctp_view (read-only use): one coupled
transportation subproblem 't' with rows 's<i>' (sum_j x_ij = s_i), 'd<j>' (sum_i x_ij = d_j) and 'l<i>_<j>'
(x_ij <= m_ij y_ij, m_ij = min(s_i, d_j)), boxes 0 <= x_ij <= m_ij, no first-stage rows. Its deterministic
equivalent is the compact LP  min c.x + f.y  s.t. the rows above, y in [0, 1]^{m x n}.

Why FCTP's bound side cannot separate (checked here empirically): at the LP optimum y_ij = x_ij / m_ij, so the
compact LP equals Balinski's transportation LP with unit costs c_ij + f_ij / m_ij (decomposition-free, m + n rows).
Its duals (u, v) give the single Benders cut  theta >= s.u + d.v - sum_ij m_ij (u_i + v_j - c_ij)^+ y_ij  whose
master value is that LP's value (f_ij >= m_ij (u_i + v_j - c_ij) by dual feasibility), i.e. LP*. The engineered
Benders reference (d_ref_bd_xl) and, under v4's rule "same initializer", the naive LP-master Benders both start from
this cut, and the single view subproblem is itself a transportation LP of the compact LP's size.

Compressed certificate: {"format": "fctp-cert-c1", "view": "fctp-v1", "version": k, "u": [...], "v": [...]}: one
opt cut with the term pi_s<i> = u_i, pi_d<j> = v_j, pi_l<i>_<j> = -(u_i + v_j - c_ij)^+ (cheapest completion).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.bd.fctp.a_data_xl import load_fctp
from src.bd.scfl.b_mip_xl import is_memory_error, make_env

INF = float("inf")
VIEW_ID = "fctp-v1"
FORMAT = "fctp-cert-c1"


def make_glover_like(m: int, n: int, seed: int, base_path: str | Path) -> str:
    """Klose-format text of a balanced m x n instance whose supplies, demands, unit costs and fixed costs are
    bootstrapped (numpy default_rng(seed), i.i.d. with replacement) from a Glover-Amini-Kochenberger base instance;
    demands are rescaled to the total supply (rounded; the residual goes to the largest demand)."""
    base = load_fctp(base_path)
    rng = np.random.default_rng(seed)
    s = rng.choice(base.supply.astype(np.int64), size=m, replace=True)
    d = rng.choice(base.demand.astype(np.int64), size=n, replace=True).astype(float)
    d = np.maximum(1, np.round(d * s.sum() / d.sum())).astype(np.int64)
    d[int(np.argmax(d))] += int(s.sum() - d.sum())
    c = rng.choice(base.tcost.ravel().astype(np.int64), size=m * n, replace=True)
    f = rng.choice(base.fcost.ravel().astype(np.int64), size=m * n, replace=True)
    assert s.sum() == d.sum() and d.min() > 0
    parts = [f"{m} {n}", " ".join(map(str, s.tolist())), " ".join(map(str, d.tolist())),
             " ".join(map(str, c.tolist())), " ".join(map(str, f.tolist()))]
    return "\n".join(parts) + "\n"


def cert_bytes(u, v, version: Any = None) -> bytes:
    return json.dumps({"format": FORMAT, "view": VIEW_ID, "version": version, "u": [float(x) for x in u],
                       "v": [float(x) for x in v]}, separators=(",", ":")).encode()


def to_generic(data: bytes, tcost: np.ndarray, version: Any = None) -> bytes:
    obj = json.loads(data)
    if obj.get("format") != FORMAT:
        raise ValueError("format_mismatch")
    u, v = np.asarray(obj["u"], dtype=float), np.asarray(obj["v"], dtype=float)
    pi: dict[str, float] = {}
    for i in np.nonzero(u)[0].tolist():
        pi[f"s{i}"] = float(u[i])
    for j in np.nonzero(v)[0].tolist():
        pi[f"d{j}"] = float(v[j])
    G = u[:, None] + v[None, :] - tcost
    ii, jj = np.nonzero(G > 0)
    for i, j, g in zip(ii.tolist(), jj.tolist(), G[ii, jj].tolist()):
        pi[f"l{i}_{j}"] = -g
    cut = {"kind": "opt", "terms": [{"s": "t", "pi": pi}]}
    return json.dumps({"view": VIEW_ID, "version": obj.get("version") if version is None else version, "cuts": [cut]},
                      separators=(",", ":")).encode()


def cert_size(u, v, tcost) -> int:
    u, v = np.asarray(u), np.asarray(v)
    return int((u != 0).sum() + (v != 0).sum() + ((u[:, None] + v[None, :] - tcost) > 0).sum())


def exact_balinski_value(inst, x_float: np.ndarray) -> tuple[Fraction | None, str]:
    """Exact objective of the compact-LP point (x, y = x / m) from integer-rounded transportation flows, if they are
    exactly feasible (supplies, demands, 0 <= x <= m): a rigorous LP_hi."""
    X = np.rint(np.asarray(x_float, dtype=float).reshape(inst.m, inst.n)).astype(np.int64)
    cap = inst.cap.astype(np.int64)
    if (X < 0).any() or (X > cap).any() or (X.sum(axis=1) != inst.supply.astype(np.int64)).any() \
            or (X.sum(axis=0) != inst.demand.astype(np.int64)).any():
        return None, "rounded flows infeasible"
    c = inst.tcost.astype(np.int64)
    f = inst.fcost.astype(np.int64)
    val = Fraction(int((c * X).sum()))
    nz = np.nonzero(X)
    for i, j in zip(*nz):
        val += Fraction(int(f[i, j]) * int(X[i, j]), int(cap[i, j]))
    return val, "ok"


class _Base:
    def __init__(self, *, time_limit: float = 3600.0, threads: int = 1, on_incumbent: Callable | None = None,
                 on_bound: Callable | None = None, on_certificate: Callable | None = None, soft_mem_gb: float | None = None,
                 seed: int = 0, mem_limit_gb: float | None = None, method: int = 2, crossover: int = 0,
                 publish_margin: float = 30.0, **kw) -> None:
        self.time_limit, self.threads, self.soft_mem_gb, self.seed = time_limit, threads, soft_mem_gb, seed
        self.mem_limit_gb, self.method, self.crossover, self.publish_margin = mem_limit_gb, method, crossover, publish_margin
        self.on_bound = on_bound or (lambda b: None)
        self.on_certificate = on_certificate or (lambda c: None)

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


class FCTPCompactLP(_Base):
    """Compact LP control (barrier with crossover off, or dual simplex), explicit y and linking rows; certificate:
    the supply/demand duals (Pi, or BarPi of an incomplete barrier) as one cut. `balinski=True`: the transportation
    LP with costs c + f/m instead (the engineered Benders LP phase, decomposition-free)."""

    def __init__(self, *, balinski: bool = False, **kw) -> None:
        super().__init__(**kw)
        self.balinski = balinski

    def run(self, data_path: str) -> dict[str, Any]:
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + self.time_limit
        inst = load_fctp(data_path)
        m, n = inst.m, inst.n
        N = m * n
        cap = inst.cap.ravel().astype(float)
        env = make_env(self.threads, self.soft_mem_gb, self.seed, self.mem_limit_gb)
        out: dict[str, Any] = {"method": self.method, "balinski": self.balinski, "termination_reason": "deadline_during_build"}
        md = None
        try:
            md = gp.Model(env=env)
            if self.balinski:
                x = md.addMVar(N, lb=0.0, ub=cap, obj=inst.tcost.ravel() + inst.fcost.ravel() / cap)
            else:
                x = md.addMVar(N, lb=0.0, ub=cap, obj=inst.tcost.ravel().astype(float))
                y = md.addMVar(N, lb=0.0, ub=1.0, obj=inst.fcost.ravel().astype(float))
            k = np.arange(N)
            Rs = sp.csr_matrix((np.ones(N), (k // n, k)), shape=(m, N))
            Rd = sp.csr_matrix((np.ones(N), (k % n, k)), shape=(n, N))
            srows = md.addConstr(Rs @ x == inst.supply.astype(float))
            drows = md.addConstr(Rd @ x == inst.demand.astype(float))
            if not self.balinski:
                md.addConstr(x - sp.diags(cap) @ y <= 0.0)
            md.ModelSense = GRB.MINIMIZE
            md.update()
            out.update(build_seconds=self.elapsed(), num_vars=int(md.NumVars), num_constrs=int(md.NumConstrs))
            margin = min(self.publish_margin if self.method == 2 else 5.0, 0.1 * self.time_limit)
            if self.left() - margin > 0.01:
                md.Params.TimeLimit = self.left() - margin
                md.Params.Method = self.method
                md.Params.Crossover = self.crossover if self.method == 2 else -1
                md.optimize()
                st = int(md.Status)
                out["gurobi_status"] = st
                u = v = None
                if st in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                    u, v = np.asarray(srows.Pi, dtype=float), np.asarray(drows.Pi, dtype=float)
                    out["certificate_source"] = "Pi"
                    out["lp_x"] = np.asarray(x.X, dtype=float)
                elif self.method == 2:
                    try:
                        u, v = np.asarray(srows.BarPi, dtype=float), np.asarray(drows.BarPi, dtype=float)
                        out["certificate_source"] = "BarPi"
                    except (gp.GurobiError, AttributeError):
                        u = None
                if u is not None and self.left() > 0:
                    out["certificate_size"] = cert_size(u, v, inst.tcost)
                    self.on_certificate(cert_bytes(u, v, version=1))
                    out["certificate_seconds"] = self.elapsed()
                if st == GRB.OPTIMAL and self.left() > 0:
                    out["termination_reason"] = "optimal"
                    out["lp_value"] = float(md.ObjVal)
                    self.on_bound(float(md.ObjVal) - 1e-9 * abs(float(md.ObjVal)))
                else:
                    out["termination_reason"] = {GRB.TIME_LIMIT: "time_limit", GRB.MEM_LIMIT: "memory_limit"}.get(st, f"status_{st}")
                out["barrier_iterations"] = int(md.BarIterCount) if self.method == 2 else None
                out["simplex_iterations"] = float(md.IterCount)
        except (gp.GurobiError, MemoryError) as err:
            if isinstance(err, gp.GurobiError) and not is_memory_error(err):
                raise
            out["termination_reason"] = "memory_limit"
            out["error"] = str(err)[:300]
        finally:
            if md is not None:
                md.dispose()
            env.dispose()
        if "lp_x" in out:
            val, why = exact_balinski_value(inst, out["lp_x"])
            out["lp_hi_exact"] = str(val) if val is not None else None
            out["lp_hi_note"] = why
            del out["lp_x"]
        out["total_seconds"] = self.elapsed()
        return out
