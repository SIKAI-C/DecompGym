"""Compact UMAHLP MIPs and the independent hub-set evaluator (scale-up study).

Two compact formulations; `build_hub_mip(..., formulation="auto")` takes the flow formulation (disaggregated linking),
which Gurobi handles best on AP: its LP was as tight as the path LP on every AP instance probed (both equal to the
optimum on 10L/T, 25T, 50L, 50T), and it is 2.6x smaller at n = 50 (0.25M vs 0.66M variables) and much smaller beyond.
Laptop probes (Gurobi 11.0.3, one thread): AP 50T solved at the root in 7.0 s (flow) vs 17.7 s (path), 50L in 9.2 s vs
23.6 s; at n = 100 (L50) the flow model (2.0M variables, 6.0M nonzeros, built in 11 s) had not finished its root LP after
57 s; the path model would have 9.6M variables / 28M nonzeros.

1. Path formulation (Hamacher, Labbe, Nickel, Sonneborn 2004), reduced as in Contreras, Cordeau, Laporte (2011, Sec. 2.1):
       min  sum_h f_h z_h + sum_{(i,j)} W_ij ( sum_h F_ij(h,h) x_ijh + sum_{e in E_ij} F_ij(e) x_ije )
       s.t. sum_h x_ijh + sum_{e in E_ij} x_ije = 1                     (every pair)
            x_ijh + sum_{e in E_ij, h in e} x_ije <= z_h                 (every pair and hub: strong linking)
            x >= 0, z binary.
   A two-hub edge e = {k, m} carries the cheaper direction F_ij(e) = min(F_ijkm, F_ijmk) and is kept only if it is
   strictly cheaper than both single-hub paths (Property 1; for i = j no edge survives, Property 2). The dropped paths
   never improve a routing over an open hub set, so the integer optimum is unchanged and the LP is at least as tight.
   This is the strongest compact formulation of the literature (its LP bound is often the optimum), but it has
   n^2 (n + |E_ij|) variables and n^3 linking rows: on AP (chi = 3, alpha = 0.75, delta = 2) about 17% of the
   n(n-1)/2 edges survive per pair, i.e. ~0.67M variables / 2.0M nonzeros at n = 50, ~9.6M / 28M at n = 100,
   ~47M / 140M at n = 150 and ~145M / 420M at n = 200 (`path_size`). CCL report that CPLEX could not even load the
   path model for n > 50 in 8 GB.
2. Flow formulation (Ernst and Krishnamoorthy 1998; Boland, Krishnamoorthy, Ernst, Ebery 2004), one single-source
   flow per origin i through the hub layers:
       min  sum_h f_h z_h + sum_{i,k} chi d_ik Z_ik + sum_{i,k,l} alpha d_kl Y_ikl + sum_{i,l,j} delta d_lj X_ilj
       s.t. sum_k Z_ik = O_i;  sum_l X_ilj = W_ij;
            Z_ik + sum_l Y_ilk - sum_l Y_ikl - sum_j X_ikj = 0            (conservation at hub k)
            Z_ik <= O_i z_k;  X_ilj <= W_ij z_l  (disaggregated; `flow_link="aggregated"`: sum_i X_ilj <= D_j z_l)
            Z, Y, X >= 0, z binary  (Y_ikk fixed to 0),
   with O_i = sum_j W_ij and D_j = sum_i W_ij: ~2 n^3 variables (2M at n = 100, 6.8M at n = 150, 16M at n = 200,
   `flow_size`) and a weaker LP in theory (equal to the path LP on the AP probes above). CCL (2011, Table 5) report
   that this flow model loads beyond n = 50 but solves no instance with more than 75 nodes within 2 h (CPLEX 10.1).

Evaluator: for a hub set H the optimal routing is closed form, C_ij(H) = min_{k,m in H} F_ijkm, computed as
A[i, m] = min_{k in H} (chi d_ik + alpha d_km), C[i, j] = min_{m in H} (A[i, m] + delta d_mj); `hub_cost` adds the
fixed costs. The verifier (e_verifier) recomputes the same quantity in a different order.
"""
from __future__ import annotations

from typing import Any, Iterable

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.bd.hub.a_data import HubInstance

CHUNK = 4_000_000          # elements per temporary array in the vectorized loops


# ---------------------------------------------------------------- hub sets and routing

def hub_vector(inst: HubInstance, hubs: Iterable[int] | np.ndarray) -> np.ndarray:
    """Boolean mask of open hubs from a list of indices (a boolean array of length n is taken as the mask)."""
    if isinstance(hubs, np.ndarray) and hubs.dtype == bool:
        if len(hubs) != inst.n:
            raise ValueError("hub mask has the wrong length")
        return hubs.copy()
    idx = np.asarray(list(hubs), dtype=np.int64)
    if len(idx) and (idx.min() < 0 or idx.max() >= inst.n):
        raise ValueError("hub index out of range")
    z = np.zeros(inst.n, dtype=bool)
    z[idx] = True
    return z


def first_leg(inst: HubInstance, H: np.ndarray) -> np.ndarray:
    """A[i, m] = min_{k in H} (chi d_ik + alpha d_km) for m in H (shape (n, |H|))."""
    d = inst.d
    T = inst.alpha * d[np.ix_(H, H)]  # (k, m)
    A = np.empty((inst.n, len(H)))
    step = max(1, CHUNK // max(1, len(H) * len(H)))
    for s in range(0, inst.n, step):
        A[s:s + step] = (inst.chi * d[s:s + step, H][:, :, None] + T[None, :, :]).min(axis=1)
    return A


def route_costs(inst: HubInstance, hubs: Iterable[int] | np.ndarray) -> np.ndarray:
    """C[i, j] = min over paths i -> k -> m -> j with k, m open of chi d_ik + alpha d_km + delta d_mj."""
    H = np.nonzero(hub_vector(inst, hubs))[0]
    if not len(H):
        raise ValueError("at least one hub must be open")
    A = first_leg(inst, H)
    D = inst.delta * inst.d[H, :]  # (m, j)
    C = np.empty((inst.n, inst.n))
    step = max(1, CHUNK // max(1, len(H) * inst.n))
    for s in range(0, inst.n, step):
        C[s:s + step] = (A[s:s + step, :, None] + D[None, :, :]).min(axis=1)
    return C


def hub_cost(inst: HubInstance, hubs: Iterable[int] | np.ndarray) -> float:
    """True objective of a hub set: fixed costs plus the optimal (cheapest-path) routing cost."""
    z = hub_vector(inst, hubs)
    return float(inst.f[z].sum() + (inst.w * route_costs(inst, z)).sum())


def hub_output(inst: HubInstance, hubs: Iterable[int] | np.ndarray, method: str = "benders_decomposition",
               status: str = "feasible") -> dict[str, Any]:
    """A hub set in the output contract of e_verifier."""
    z = hub_vector(inst, hubs)
    return {"status": status, "method": method, "objective_value": hub_cost(inst, z), "hubs": np.nonzero(z)[0].tolist()}


# ---------------------------------------------------------------- path structure

def loop_costs(inst: HubInstance, i: int | slice | np.ndarray) -> np.ndarray:
    """F_ij(h, h) = chi d_ih + delta d_hj for origins i: shape (|i|, n_j, n_h)."""
    d = inst.d
    di = np.atleast_2d(d[i])
    return inst.chi * di[:, None, :] + inst.delta * d.T[None, :, :]


def edge_costs(inst: HubInstance, i: int) -> np.ndarray:
    """F_ij{k, m} = min(F_ijkm, F_ijmk) for one origin: shape (n_j, n_k, n_m) (symmetric in k, m)."""
    d = inst.d
    F = inst.chi * d[i][None, :, None] + inst.alpha * d[None, :, :] + inst.delta * d.T[:, None, :]  # (j, k, m)
    return np.minimum(F, F.transpose(0, 2, 1))


def surviving_edges(inst: HubInstance, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Edges kept by Properties 1-2 for origin i: (j, k, m, F) with k < m, F strictly below both single-hub paths."""
    n = inst.n
    Fu = edge_costs(inst, i)
    L = loop_costs(inst, i)[0]  # (j, h)
    keep = Fu < np.minimum(L[:, :, None], L[:, None, :])
    keep &= np.triu(np.ones((n, n), dtype=bool), 1)[None, :, :]
    keep[i] = False  # i == j: one hub only
    if inst.w is not None:
        keep &= (inst.w[i] > 0)[:, None, None]
    j, k, m = np.nonzero(keep)
    return j, k, m, Fu[j, k, m]


def path_size(inst: HubInstance) -> dict[str, int]:
    """Size of the reduced path formulation (counted origin by origin, nothing stored)."""
    n = inst.n
    pairs = int((inst.w > 0).sum())
    edges = 0
    for i in range(n):
        edges += len(surviving_edges(inst, i)[0])
    loops = pairs * n
    return {"pairs": pairs, "loop_vars": loops, "edge_vars": edges, "variables": n + loops + edges,
            "rows": pairs + pairs * n, "nonzeros": 2 * loops + 3 * edges + pairs * n}


def flow_size(inst: HubInstance, flow_link: str = "disaggregated") -> dict[str, int]:
    n = inst.n
    link = n ** 3 if flow_link == "disaggregated" else n * n
    return {"variables": n + n * n + 2 * n ** 3, "rows": n + 3 * n * n + link,
            "nonzeros": n * n + n ** 3 + 3 * n ** 3 + n * n + 2 * n * n + 2 * link}


# ---------------------------------------------------------------- compact MIPs

def build_path_mip(inst: HubInstance, env: gp.Env | None = None) -> tuple[gp.Model, dict[str, Any]]:
    n = inst.n
    model = gp.Model(env=env) if env else gp.Model()
    z = model.addMVar(n, vtype=GRB.BINARY, name="z")
    Z = z.tolist()
    pairs = np.argwhere(inst.w > 0)
    model.addConstr(z.sum() >= 1, name="one_hub")
    obj = gp.LinExpr(inst.f.tolist(), Z)
    n_loop = n_edge = 0
    for i in range(n):
        js = pairs[pairs[:, 0] == i, 1]
        if not len(js):
            continue
        Lc = loop_costs(inst, i)[0][js]  # (|js|, h)
        xl = model.addMVar((len(js), n), lb=0.0, name=f"xl{i}")
        ej, ek, em, eF = surviving_edges(inst, i)
        pos = np.full(n, -1)
        pos[js] = np.arange(len(js))
        ep = pos[ej]
        xe = model.addMVar(len(ej), lb=0.0, name=f"xe{i}") if len(ej) else None
        wi = inst.w[i, js]
        obj.add(gp.LinExpr((wi[:, None] * Lc).ravel().tolist(), xl.reshape(-1).tolist()))
        if xe is not None:
            obj.add(gp.LinExpr((wi[ep] * eF).tolist(), xe.tolist()))
        XL = xl.tolist()
        XE = xe.tolist() if xe is not None else []
        # Assignment rows.
        order = np.argsort(ep, kind="stable")
        st = np.searchsorted(ep[order], np.arange(len(js) + 1))
        for p in range(len(js)):
            ks = order[st[p]:st[p + 1]].tolist()
            model.addLConstr(gp.LinExpr([1.0] * (n + len(ks)), XL[p] + [XE[k] for k in ks]), GRB.EQUAL, 1.0)
        # Strong linking rows (pair, hub).
        rows = np.concatenate([ep * n + ek, ep * n + em])
        evar = np.concatenate([np.arange(len(ej))] * 2)
        o2 = np.argsort(rows, kind="stable")
        rs, ev = rows[o2], evar[o2]
        st2 = np.searchsorted(rs, np.arange(len(js) * n + 1))
        for p in range(len(js)):
            XLp = XL[p]
            for h in range(n):
                a, b = st2[p * n + h], st2[p * n + h + 1]
                if a == b:
                    model.addLConstr(XLp[h] - Z[h], GRB.LESS_EQUAL, 0.0)
                else:
                    ks = ev[a:b].tolist()
                    model.addLConstr(gp.LinExpr([1.0] * (len(ks) + 1) + [-1.0], [XLp[h]] + [XE[k] for k in ks] + [Z[h]]),
                                     GRB.LESS_EQUAL, 0.0)
        n_loop += len(js) * n
        n_edge += len(ej)
    model.setObjective(obj, GRB.MINIMIZE)
    return model, {"z": z, "formulation": "path", "loop_vars": n_loop, "edge_vars": n_edge}


def build_flow_mip(inst: HubInstance, env: gp.Env | None = None, flow_link: str = "disaggregated") -> tuple[gp.Model, dict[str, Any]]:
    n = inst.n
    d, W = inst.d, inst.w
    O, D = W.sum(axis=1), W.sum(axis=0)
    model = gp.Model(env=env) if env else gp.Model()
    z = model.addMVar(n, vtype=GRB.BINARY, name="z")
    Zc = model.addMVar((n, n), lb=0.0, name="Z")       # Z[i, k] collection
    ub = np.full((n, n, n), GRB.INFINITY)
    ub[:, np.arange(n), np.arange(n)] = 0.0
    Y = model.addMVar((n, n, n), lb=0.0, ub=ub, name="Y")  # Y[i, k, l] transfer
    X = model.addMVar((n, n, n), lb=0.0, name="X")         # X[i, l, j] distribution
    model.addConstr(z.sum() >= 1, name="one_hub")
    model.addConstr(Zc.sum(axis=1) == O, name="supply")
    model.addConstr(X.sum(axis=1) == W, name="demand")
    model.addConstr(Zc + Y.sum(axis=1) - Y.sum(axis=2) - X.sum(axis=2) == 0, name="conserve")
    zk = z[np.broadcast_to(np.arange(n)[None, :], (n, n))]
    model.addConstr(Zc - O[:, None] * zk <= 0, name="link_collect")
    if flow_link == "disaggregated":
        zl = z[np.broadcast_to(np.arange(n)[None, :, None], (n, n, n))]
        model.addConstr(X - W[:, None, :] * zl <= 0, name="link_distribute")
    else:
        zl = z[np.broadcast_to(np.arange(n)[:, None], (n, n))]  # (l, j) -> z_l
        model.addConstr(X.sum(axis=0) - D[None, :] * zl <= 0, name="link_distribute")
    cost = (inst.f @ z + (inst.chi * d * Zc).sum() + (inst.alpha * d[None, :, :] * Y).sum()
            + (inst.delta * d[None, :, :] * X).sum())
    model.setObjective(cost, GRB.MINIMIZE)
    return model, {"z": z, "formulation": f"flow_{flow_link}"}


def build_hub_mip(inst: HubInstance, env: gp.Env | None = None, formulation: str = "auto",
                  flow_link: str = "disaggregated") -> tuple[gp.Model, dict[str, Any]]:
    if formulation == "auto":
        formulation = "flow"
    if formulation == "path":
        return build_path_mip(inst, env)
    if formulation == "flow":
        return build_flow_mip(inst, env, flow_link)
    raise ValueError(f"unknown formulation {formulation!r}")
