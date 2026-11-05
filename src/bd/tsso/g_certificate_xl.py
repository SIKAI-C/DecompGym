"""Certified lower bounds for TSSO Benders certificates (scale-up study, protocol v4 Section 3 after Codex reviews
42-43): an exact-arithmetic LP-master bound computed only from subproblem dual vectors.

View (a_data_xl.VIEW, declared): subproblems k = (i, s), ID k = i * S + s, trusted weight p_s (taken from the
instance, never from the certificate). Subproblem LP (columns xr, xo, inv, back per period, all in [0, D_k] with
D_k = sum_t d_its; rows bal_t equalities; cap_t: xr_t <= K_it y_it). The boxes are implied for EVERY y in [0, 1]^T,
fractional or binary (proof in a_data_xl), so Q_k(y) is unchanged by them on the whole first-stage box.

Certificate = a list of cuts; a cut is a set of terms (k, pi_k in R^T for the balance rows, mu_k in R^T for the
capacity rows, mu_k >= 0 meaning the multiplier of xr_t <= K_it y_it, i.e. mu = -Pi in Gurobi's sign convention).
A cut with one term is a multi-cut row, a cut with several terms an aggregated row; within a cut every ID must be
unique and in range, and every value finite, otherwise the cut is rejected (reported, never used).

Safe cut (Neumaier-Shcherbina) of a term, exact: mu is projected to mu >= 0; with the reduced costs
    r_xr,t = c_reg_t - pi_t + mu_t,   r_xo,t = c_ot_t - pi_t,
    r_inv,t = h_t + pi_t - pi_t+1,    r_back,t = b_t - pi_t + pi_t+1   (pi_T := 0 for the last period)
weak duality over the boxes gives, for every y in [0, 1]^T and every (pi, mu),
    Q_k(y) >= pi . d_k - sum_t mu_t K_it y_it + D_k * sum_j min(0, r_j)            (l_j = 0, u_j = D_k),
so a cut is valid whatever the agent's duals are (an infeasible dual vector only weakens it).

Master (identical first-stage rows to the compact LP relaxation: none besides 0 <= y <= 1):
    min F.y + sum_k p_k theta_k,  y in [0, 1]^(I x T),  theta_k in [L_k, U_k],
    L_k = sum_j min(c_j l_j, c_j u_j) = 0,  U_k = sum_j max(c_j l_j, c_j u_j) = D_k sum_t (c_reg + c_ot + h + b)_it,
    single-term cut:  theta_k + sum_t mu_t K_it y_it >= pi . d_k + penalty_k,
    aggregated cut:   sum_k p_k theta_k + sum_k p_k sum_t mu_kt K_it y_it >= sum_k p_k (pi_k . d_k + penalty_k).
U_k bounds the cost of every feasible point of the boxed subproblem, so the theta box keeps every (y, Q(y)).

Certification: the master LP is solved in floating point by Gurobi (only to obtain multipliers lambda >= 0 of the
cut rows); lambda is rounded to the dyadic grid 2^-64 and clipped at 0, and the bound
    LB = sum_c lambda_c b_c + sum_columns min(r_j lo_j, r_j hi_j),   r = obj - A^T lambda,
is evaluated EXACTLY with the rows A, b rebuilt from the instance data (exact float64 values as rationals) and the
dual vectors (also rounded to the 2^-64 grid, which a safe cut allows: any vector gives a valid cut). Weak duality
over the finite master box makes LB valid for any lambda >= 0, so a failed or infeasible master solve degrades to
lambda = 0 (LB = sum_k p_k L_k = 0), never to an invalid or infinite value. All exact quantities are Python
integers on a common power-of-two scale (exactly the rational values; `exact="fraction"` re-does the same
computation with fractions.Fraction for tests). The returned float is rounded down.

LP* reference (`lp_upper`): an exactly feasible point of the LP relaxation at a given y (dual-simplex primal
repaired: y and production on the 2^-64 grid, xr clipped to [0, K y], xo >= 0, inventory/backlog recomputed from
the net position), evaluated exactly -- an exact upper bound on LP*. With a certified lower bound it brackets LP*.
"""
from __future__ import annotations

import json
import math
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.bd.tsso.a_data_xl import VIEW, TSSOXLInstance

GRID = 64
TWO_G = float(2 ** GRID)


# ---------------------------------------------------------------- cut log (written by the methods)

class CutLog:
    """Cuts with their creation times (seconds on the method's clock). Every term: subproblem ID, pi (T), mu (T).
    Dual vectors are stored once per distinct (pi, mu) of a batch (scenarios often share a dual vertex)."""

    def __init__(self, inst: TSSOXLInstance) -> None:
        self.T, self.I, self.S = inst.T, inst.I, inst.S
        self._sub: list[np.ndarray] = []
        self._cut: list[np.ndarray] = []
        self._pool: list[np.ndarray] = []   # distinct [pi | mu] rows
        self._idx: list[np.ndarray] = []    # term -> row of the global pool
        self._time: list[float] = []
        self._pool_rows = 0
        self.n_cuts = 0

    def _store(self, pi: np.ndarray, mu: np.ndarray, n: int) -> np.ndarray:
        rows = np.concatenate([np.asarray(pi, dtype=float).reshape(n, self.T), np.asarray(mu, dtype=float).reshape(n, self.T)], axis=1)
        uniq, inv = np.unique(rows, axis=0, return_inverse=True)
        self._pool.append(uniq)
        idx = inv.reshape(-1).astype(np.int64) + self._pool_rows
        self._pool_rows += len(uniq)
        return idx

    def add_multi(self, sub: np.ndarray, pi: np.ndarray, mu: np.ndarray, t: float) -> None:
        """One single-term cut per row."""
        n = len(sub)
        if not n:
            return
        self._sub.append(np.asarray(sub, dtype=np.int64))
        self._cut.append(np.arange(self.n_cuts, self.n_cuts + n, dtype=np.int64))
        self._idx.append(self._store(pi, mu, n))
        self._time.extend([t] * n)
        self.n_cuts += n

    def add_aggregate(self, sub: np.ndarray, pi: np.ndarray, mu: np.ndarray, t: float) -> None:
        """One cut over all given terms."""
        n = len(sub)
        if not n:
            return
        self._sub.append(np.asarray(sub, dtype=np.int64))
        self._cut.append(np.full(n, self.n_cuts, dtype=np.int64))
        self._idx.append(self._store(pi, mu, n))
        self._time.append(t)
        self.n_cuts += 1

    def pooled(self) -> dict[str, np.ndarray]:
        T = self.T
        if not self._sub:
            z = np.zeros(0, dtype=np.int64)
            return {"sub": z, "cut": z, "idx": z, "pool": np.zeros((0, 2 * T)), "time": np.zeros(0)}
        return {"sub": np.concatenate(self._sub), "cut": np.concatenate(self._cut), "idx": np.concatenate(self._idx),
                "pool": np.concatenate(self._pool), "time": np.asarray(self._time, dtype=float)}

    def arrays(self) -> dict[str, np.ndarray]:
        return expand(self.pooled(), self.T)

    def save(self, path: str | Path, **extra: Any) -> None:
        np.savez(path, **self.pooled(), **{k: np.asarray(v) for k, v in extra.items()})

    @property
    def terms(self) -> int:
        return int(sum(len(s) for s in self._sub))

    @property
    def pool_rows(self) -> int:
        return self._pool_rows


def expand(pooled: dict[str, np.ndarray], T: int) -> dict[str, np.ndarray]:
    pool, idx = pooled["pool"], pooled["idx"]
    return {"sub": pooled["sub"], "cut": pooled["cut"], "pi": pool[idx, :T], "mu": pool[idx, T:], "time": pooled["time"]}


def load_cuts(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if "pool" in d:
        return expand(d, d["pool"].shape[1] // 2)
    return d


def cuts_upto(cuts: dict[str, np.ndarray], tau: float) -> dict[str, np.ndarray]:
    """The cuts created strictly before tau (renumbered)."""
    keep_cut = np.nonzero(cuts["time"] < tau)[0]
    mask = np.isin(cuts["cut"], keep_cut)
    remap = np.full(len(cuts["time"]), -1, dtype=np.int64)
    remap[keep_cut] = np.arange(len(keep_cut))
    return {"sub": cuts["sub"][mask], "cut": remap[cuts["cut"][mask]], "pi": cuts["pi"][mask], "mu": cuts["mu"][mask],
            "time": cuts["time"][keep_cut]}


def empty_cuts(T: int) -> dict[str, np.ndarray]:
    z = np.zeros(0, dtype=np.int64)
    return {"sub": z, "cut": z, "pi": np.zeros((0, T)), "mu": np.zeros((0, T)), "time": np.zeros(0)}


# ---------------------------------------------------------------- JSON certificates (agent side)

def read_certificate(inst: TSSOXLInstance, path: str | Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """certificate.json: {"view": VIEW id, "cuts": [{"terms": [{"k": id, "pi": [T], "mu": [T]}, ...]}, ...]}
    ("pi"/"mu" dense lists, or sparse {"t": value} dicts; mu defaults to zeros). Malformed cuts are dropped and
    reported; the result is in CutLog array form (time 0)."""
    T = inst.T
    issues: list[str] = []
    try:
        obj = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return empty_cuts(T), [f"unreadable certificate: {exc}"]
    if not isinstance(obj, dict) or obj.get("view") != VIEW["id"]:
        return empty_cuts(T), [f"view must be {VIEW['id']!r}"]

    def vec(v: Any) -> np.ndarray | None:
        out = np.zeros(T)
        if v is None:
            return out
        try:
            if isinstance(v, dict):
                for key, val in v.items():
                    t = int(key)
                    if not 0 <= t < T:
                        return None
                    out[t] = float(val)
            else:
                if len(v) != T:
                    return None
                out[:] = [float(x) for x in v]
        except (TypeError, ValueError):
            return None
        return out

    subs, cutids, pis, mus = [], [], [], []
    n_ok = 0
    for c, cut in enumerate(obj.get("cuts", []) if isinstance(obj.get("cuts"), list) else []):
        terms = cut.get("terms") if isinstance(cut, dict) else None
        if not isinstance(terms, list) or not terms:
            issues.append(f"cut {c}: no terms")
            continue
        rows = []
        for term in terms:
            if not isinstance(term, dict):
                rows = None
                break
            try:
                k = int(term.get("k"))
            except (TypeError, ValueError):
                rows = None
                break
            pi, mu = vec(term.get("pi")), vec(term.get("mu"))
            if pi is None or mu is None:
                rows = None
                break
            rows.append((k, pi, mu))
        if rows is None:
            issues.append(f"cut {c}: malformed term")
            continue
        for k, pi, mu in rows:
            subs.append(k)
            cutids.append(n_ok)
            pis.append(pi)
            mus.append(mu)
        n_ok += 1
    if not subs:
        return empty_cuts(T), issues
    return {"sub": np.asarray(subs, dtype=np.int64), "cut": np.asarray(cutids, dtype=np.int64), "pi": np.asarray(pis),
            "mu": np.asarray(mus), "time": np.zeros(n_ok)}, issues


def write_certificate(cuts: dict[str, np.ndarray], path: str | Path) -> None:
    out = []
    order = np.argsort(cuts["cut"], kind="stable")
    for c in np.unique(cuts["cut"]):
        idx = order[cuts["cut"][order] == c]
        out.append({"terms": [{"k": int(cuts["sub"][j]), "pi": cuts["pi"][j].tolist(), "mu": cuts["mu"][j].tolist()} for j in idx]})
    Path(path).write_text(json.dumps({"view": VIEW["id"], "cuts": out}))


# ---------------------------------------------------------------- exact data

def _ratio_scale(values: np.ndarray) -> int:
    """Smallest s such that every value * 2^s is an integer (values are float64, hence dyadic)."""
    s = 0
    for v in np.unique(np.asarray(values, dtype=float).ravel()):
        d = float(v).as_integer_ratio()[1]
        s = max(s, d.bit_length() - 1)
    return s


def _ints(values: np.ndarray, s: int) -> list:
    out = []
    for v in np.asarray(values, dtype=float).ravel():
        n, d = float(v).as_integer_ratio()
        out.append(n << (s - (d.bit_length() - 1)))
    return out


def grid_ints(a: np.ndarray) -> list:
    """Values rounded to the 2^-GRID grid, as integers n meaning n / 2^GRID (exact)."""
    return [int(v) for v in np.rint(np.asarray(a, dtype=float).ravel() * TWO_G)]


class ExactData:
    """The instance's float64 data as exact integers on power-of-two scales (see the module docstring)."""

    def __init__(self, inst: TSSOXLInstance) -> None:
        self.inst = inst
        I = inst.I
        costs = np.stack([inst.c_reg, inst.c_ot, inst.h, inst.b])
        self.SC = _ratio_scale(costs)
        self.SK = _ratio_scale(inst.K)
        self.SF = _ratio_scale(inst.F)
        self.SP = _ratio_scale(inst.prob)
        self.d_integral = bool(np.all(inst.demand == np.round(inst.demand)) and np.all(np.abs(inst.demand) < 2 ** 52))
        self.SD = 0 if self.d_integral else _ratio_scale(inst.demand)
        self.S1 = max(GRID, self.SC)
        self.ST = max(self.S1 + self.SD, GRID + self.SK)  # scale of a single-term row
        sh = self.S1 - self.SC
        self.cr = [[v << sh for v in _ints(inst.c_reg[i], self.SC)] for i in range(I)]
        self.co = [[v << sh for v in _ints(inst.c_ot[i], self.SC)] for i in range(I)]
        self.hh = [[v << sh for v in _ints(inst.h[i], self.SC)] for i in range(I)]
        self.bb = [[v << sh for v in _ints(inst.b[i], self.SC)] for i in range(I)]
        self.Kint = [_ints(inst.K[i], self.SK) for i in range(I)]
        self.Fint = [_ints(inst.F[i], self.SF) for i in range(I)]
        self.pint = _ints(inst.prob, self.SP)
        # sum_t (c_reg + c_ot + h + b)_it at scale SC, for U_k
        self.ccol = [sum(_ints(costs[:, i, :], self.SC)) for i in range(I)]
        self._dcache: dict[int, tuple[list, int]] = {}

    def demand(self, k: int) -> tuple[list, int]:
        """(d_k as ints at scale SD, D_k = sum_t d_k at scale SD)."""
        if k not in self._dcache:
            S = self.inst.S
            i, s = divmod(k, S)
            col = self.inst.demand[i, :, s]
            d = [int(v) for v in col] if self.d_integral else _ints(col, self.SD)
            self._dcache[k] = (d, sum(d))
            if len(self._dcache) > 200_000:
                self._dcache.clear()
        return self._dcache[k]

    def upper(self, k: int) -> int:
        """U_k at scale SC + SD."""
        return self.demand(k)[1] * self.ccol[k // self.inst.S]

    def safe_term(self, k: int, pi_g: list, mu_g: list) -> tuple[int, list]:
        """Exact safe cut of one term: (const, coef) at scale ST with theta_k + coef . y_i >= const."""
        i = k // self.inst.S
        T = self.inst.T
        d, D = self.demand(k)
        gs = self.S1 - GRID
        cr, co, hh, bb = self.cr[i], self.co[i], self.hh[i], self.bb[i]
        p1 = [x << gs for x in pi_g]
        mu_g = [m if m > 0 else 0 for m in mu_g]
        neg = 0
        for t in range(T):
            pt = p1[t]
            r = cr[t] - pt + (mu_g[t] << gs)
            if r < 0:
                neg += r
            r = co[t] - pt
            if r < 0:
                neg += r
            nxt = p1[t + 1] if t + 1 < T else 0
            r = hh[t] + pt - nxt
            if r < 0:
                neg += r
            r = bb[t] - pt + nxt
            if r < 0:
                neg += r
        pid = 0
        for t in range(T):
            pid += pi_g[t] * d[t]
        const = (pid << (self.ST - GRID - self.SD)) + ((neg * D) << (self.ST - self.S1 - self.SD))
        Ks = self.Kint[i]
        shk = self.ST - GRID - self.SK
        coef = [(mu_g[t] * Ks[t]) << shk for t in range(T)]
        return const, coef


def safe_term_fraction(inst: TSSOXLInstance, k: int, pi: np.ndarray, mu: np.ndarray) -> tuple[Fraction, list]:
    """Reference implementation of ExactData.safe_term with fractions (pi, mu already on the grid)."""
    i, s = divmod(k, inst.S)
    T = inst.T
    F_ = Fraction
    pi = [F_(int(v), 2 ** GRID) for v in np.rint(np.asarray(pi) * TWO_G)]
    mu = [max(F_(0), F_(int(v), 2 ** GRID)) for v in np.rint(np.asarray(mu) * TWO_G)]
    d = [F_(float(v)) for v in inst.demand[i, :, s]]
    D = sum(d)
    neg = F_(0)
    for t in range(T):
        nxt = pi[t + 1] if t + 1 < T else F_(0)
        for r in (F_(float(inst.c_reg[i, t])) - pi[t] + mu[t], F_(float(inst.c_ot[i, t])) - pi[t],
                  F_(float(inst.h[i, t])) + pi[t] - nxt, F_(float(inst.b[i, t])) - pi[t] + nxt):
            if r < 0:
                neg += r
    const = sum(pi[t] * d[t] for t in range(T)) + D * neg
    return const, [mu[t] * F_(float(inst.K[i, t])) for t in range(T)]


# ---------------------------------------------------------------- float safe cuts and validation

def validate_cuts(inst: TSSOXLInstance, cuts: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, int]]:
    """Boolean mask of the cuts kept: IDs in range and unique within the cut, all values finite."""
    n_cuts = len(cuts["time"])
    ok = np.ones(n_cuts, dtype=bool)
    if not len(cuts["sub"]):
        return ok, {"rejected_cuts": 0}
    cut = cuts["cut"]
    bad_term = (cuts["sub"] < 0) | (cuts["sub"] >= inst.I * inst.S) | ~np.isfinite(cuts["pi"]).all(axis=1) | ~np.isfinite(cuts["mu"]).all(axis=1)
    ok[np.unique(cut[bad_term])] = False
    order = np.lexsort((cuts["sub"], cut))
    dup = (cut[order][1:] == cut[order][:-1]) & (cuts["sub"][order][1:] == cuts["sub"][order][:-1])
    ok[np.unique(cut[order][1:][dup])] = False
    return ok, {"rejected_cuts": int((~ok).sum()), "bad_terms": int(bad_term.sum()), "duplicate_terms": int(dup.sum())}


def float_safe_cuts(inst: TSSOXLInstance, sub: np.ndarray, pi: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized float version of the safe cut (for building the master LP only): const (n,), coef (n, T) and the
    dual infeasibility (n,) = max_j max(0, -r_j)."""
    S, T = inst.S, inst.T
    n = len(sub)
    const, coef, dinf = np.zeros(n), np.zeros((n, T)), np.zeros(n)
    for a in range(0, n, 100_000):  # chunks keep the (n, 4, T) temporaries small
        b = min(n, a + 100_000)
        i, s = sub[a:b] // S, sub[a:b] % S
        p = np.rint(pi[a:b] * TWO_G) / TWO_G
        m = np.maximum(np.rint(mu[a:b] * TWO_G) / TWO_G, 0.0)
        nxt = np.concatenate([p[:, 1:], np.zeros((b - a, 1))], axis=1)
        R = np.stack([inst.c_reg[i] - p + m, inst.c_ot[i] - p, inst.h[i] + p - nxt, inst.b[i] - p + nxt], axis=1)
        d = inst.demand[i, :, s]
        const[a:b] = (p * d).sum(axis=1) + d.sum(axis=1) * np.minimum(R, 0.0).sum(axis=(1, 2))
        coef[a:b] = m * inst.K[i]
        dinf[a:b] = np.maximum(-R.min(axis=(1, 2)), 0.0)
    return const, coef, dinf


# ---------------------------------------------------------------- the checker

def certify(inst: TSSOXLInstance, cuts: dict[str, np.ndarray], *, time_limit: float = 600.0, exact: str = "int",
            env: gp.Env | None = None) -> dict[str, Any]:
    """Certified LP-master bound of a cut set (see the module docstring). Returns {"lb": float rounded down,
    "lb_exact": Fraction, "lp_value": float master LP value (solver-reported), diagnostics...}."""
    t0 = time.perf_counter()
    S = inst.S
    out: dict[str, Any] = {"cuts_in": int(len(cuts["time"])), "terms_in": int(len(cuts["sub"]))}
    ok, val = validate_cuts(inst, cuts)
    out.update(val)
    keep = ok[cuts["cut"]] if len(cuts["sub"]) else np.zeros(0, dtype=bool)
    sub, cut, pi, mu = cuts["sub"][keep], cuts["cut"][keep], cuts["pi"][keep], cuts["mu"][keep]
    kept_ids = np.nonzero(ok)[0]
    remap = np.full(len(ok), -1, dtype=np.int64)
    remap[kept_ids] = np.arange(len(kept_ids))
    cut = remap[cut]
    n_cuts = len(kept_ids)
    out["cuts_used"] = int(n_cuts)
    own = env is None
    env = env or _env()
    ed = ExactData(inst)
    total = Fraction(0)
    lp_value = 0.0
    status: list[int] = []
    try:
        if n_cuts:
            const, coef, dinf = float_safe_cuts(inst, sub, pi, mu)
            out["max_dual_infeasibility"] = float(dinf.max()) if len(dinf) else 0.0
            size = np.bincount(cut, minlength=n_cuts)
            prod_of_term = sub // S
            # cuts touching a single product can be solved product by product (the master separates)
            pmin = np.full(n_cuts, np.iinfo(np.int64).max)
            pmax = np.full(n_cuts, -1)
            np.minimum.at(pmin, cut, prod_of_term)
            np.maximum.at(pmax, cut, prod_of_term)
            single_product = bool(np.all(pmin == pmax))
            if single_product:
                comps = [(int(i), np.nonzero(pmin == i)[0]) for i in np.unique(pmin)]
            else:
                comps = [(None, np.arange(n_cuts))]
            deadline = t0 + time_limit
            term_order = np.argsort(cut, kind="stable")
            starts = np.searchsorted(cut[term_order], np.arange(n_cuts + 1))
            for prod, cids in comps:
                tids = np.concatenate([term_order[starts[c]:starts[c + 1]] for c in cids]) if len(cids) else np.zeros(0, int)
                lam, v, st = _solve_master(inst, env, prod, cids, tids, sub, cut, size, const, coef, max(1.0, deadline - time.perf_counter()))
                status.append(st)
                lp_value += v
                part = _exact_bound(inst, ed, cids, tids, sub, cut, size, pi, mu, lam, exact)
                total += part
        out["lp_value"] = lp_value
    except Exception as exc:  # noqa: BLE001 - a checker failure must degrade to the trivial bound
        out["error"] = f"{type(exc).__name__}: {exc}"
        total = Fraction(0)  # lambda = 0: sum_k p_k L_k = 0
    finally:
        if own:
            env.dispose()
    out["lb_exact"] = total
    out["lb"] = fraction_floor(total)
    out["master_status"] = status[:8]
    out["master_not_optimal"] = int(sum(1 for s in status if s != GRB.OPTIMAL))
    out["seconds"] = time.perf_counter() - t0
    return out


def fraction_floor(x: Fraction) -> float:
    """The largest float <= x."""
    f = float(x)
    if Fraction(f) > x:
        f = math.nextafter(f, -math.inf)
    return f


def _env() -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()
    return env


def _solve_master(inst, env, prod, cids, tids, sub, cut, size, const, coef, time_limit) -> tuple[dict[int, float], float, int]:
    """Float master LP over the cuts cids (terms tids); returns the cut-row multipliers, the LP value and the
    status. Columns: y of the products involved, theta of the subproblems involved."""
    S, T = inst.S, inst.T
    ks = np.unique(sub[tids])
    prods = np.unique(ks // S)
    pcol = {int(p): j for j, p in enumerate(prods)}
    kcol = np.searchsorted(ks, sub[tids])
    nk, ny = len(ks), len(prods) * T
    row_local = np.searchsorted(cids, cut[tids])
    agg = size[cut[tids]] > 1
    w = np.where(agg, inst.prob[sub[tids] % S], 1.0)
    # theta columns
    A_th = sp.csr_matrix((w, (row_local, kcol)), shape=(len(cids), nk))
    # y columns: w * coef_term spread over the term's product
    yc = np.array([pcol[int(p)] for p in sub[tids] // S]) * T
    rows = np.repeat(row_local, T)
    cols = (yc[:, None] + np.arange(T)[None, :]).ravel()
    A_y = sp.csr_matrix(((w[:, None] * coef[tids]).ravel(), (rows, cols)), shape=(len(cids), ny))
    rhs = np.bincount(row_local, weights=w * const[tids], minlength=len(cids))
    md = gp.Model(env=env)
    md.Params.OutputFlag = 0
    md.Params.FeasibilityTol = 1e-9
    md.Params.OptimalityTol = 1e-9
    md.Params.TimeLimit = time_limit
    U = inst.total_demand.ravel()[ks] * (inst.c_reg + inst.c_ot + inst.h + inst.b).sum(axis=1)[ks // S]
    y = md.addMVar(ny, lb=0.0, ub=1.0, obj=inst.F[prods].ravel())
    th = md.addMVar(nk, lb=0.0, ub=U, obj=inst.prob[ks % S])
    cons = md.addConstr(A_th @ th + A_y @ y >= rhs)
    md.optimize()
    st = int(md.Status)
    if st == GRB.OPTIMAL:
        lam_all = np.asarray(cons.Pi, dtype=float)
        val = float(md.ObjVal)
    else:
        lam_all = np.zeros(len(cids))
        val = 0.0
    md.dispose()
    lam = {int(cids[j]): float(lam_all[j]) for j in np.nonzero(np.isfinite(lam_all) & (lam_all > 0))[0]}
    return lam, val, st


def _exact_bound(inst, ed: ExactData, cids, tids, sub, cut, size, pi, mu, lam: dict[int, float], exact: str) -> Fraction:
    """Exact N-S bound of one master component for the multipliers lam (rows with lam > 0 only)."""
    S, T = inst.S, inst.T
    G = GRID
    ST, SP, SF = ed.ST, ed.SP, ed.SF
    SU = ed.SC + ed.SD
    FS = max(G + ST + SP, SF)
    lam_g = {c: int(round(v * TWO_G)) for c, v in lam.items()}
    lam_g = {c: v for c, v in lam_g.items() if v > 0}
    if exact == "fraction":
        return _exact_bound_fraction(inst, cids, tids, sub, cut, size, pi, mu, lam_g)
    rhs_sum = 0                       # sum_c lambda_c b_c at scale FS
    r_theta: dict[int, int] = {}      # sum_c lambda_c A_c,theta_k at scale FS
    r_y: dict[tuple[int, int], int] = {}
    tid_of_cut: dict[int, list] = {}
    for j in tids:
        c = int(cut[j])
        if c in lam_g:
            tid_of_cut.setdefault(c, []).append(int(j))
    for c, js in tid_of_cut.items():
        L = lam_g[c]
        if size[c] == 1:  # theta_k + coef.y >= const, row scale ST
            j = js[0]
            k = int(sub[j])
            const, coef = ed.safe_term(k, grid_ints(pi[j]), grid_ints(mu[j]))
            sh = FS - G - ST
            rhs_sum += (L * const) << sh
            r_theta[k] = r_theta.get(k, 0) + (L << (FS - G))
            i = k // S
            for t in range(T):
                if coef[t]:
                    r_y[(i, t)] = r_y.get((i, t), 0) + ((L * coef[t]) << sh)
        else:  # sum_k p_k theta_k + sum_k p_k coef_k.y >= sum_k p_k const_k, row scale ST + SP
            sh = FS - G - ST - SP
            acc_rhs = 0
            acc_y: dict[tuple[int, int], int] = {}
            for j in js:
                k = int(sub[j])
                pk = ed.pint[k % S]
                const, coef = ed.safe_term(k, grid_ints(pi[j]), grid_ints(mu[j]))
                acc_rhs += pk * const
                r_theta[k] = r_theta.get(k, 0) + ((L * pk) << (FS - G - SP))
                i = k // S
                for t in range(T):
                    if coef[t]:
                        acc_y[(i, t)] = acc_y.get((i, t), 0) + pk * coef[t]
            rhs_sum += (L * acc_rhs) << sh
            for key, v in acc_y.items():
                r_y[key] = r_y.get(key, 0) + ((L * v) << sh)
    total = rhs_sum << SU
    for k, a in r_theta.items():
        r = (ed.pint[k % S] << (FS - SP)) - a
        if r < 0:
            total += r * ed.upper(k)
    for (i, t), a in r_y.items():
        r = (ed.Fint[i][t] << (FS - SF)) - a
        if r < 0:
            total += r << SU
    return Fraction(total, 1 << (FS + SU))


def _exact_bound_fraction(inst, cids, tids, sub, cut, size, pi, mu, lam_g: dict[int, int]) -> Fraction:
    """The same bound with fractions.Fraction throughout (reference for tests; slow)."""
    S, T = inst.S, inst.T
    F_ = Fraction
    rhs = F_(0)
    r_theta: dict[int, Fraction] = {}
    r_y: dict[tuple[int, int], Fraction] = {}
    for j in tids:
        c = int(cut[j])
        if c not in lam_g:
            continue
        L = F_(lam_g[c], 2 ** GRID)
        k = int(sub[j])
        w = F_(float(inst.prob[k % S])) if size[c] > 1 else F_(1)
        const, coef = safe_term_fraction(inst, k, pi[j], mu[j])
        rhs += L * w * const
        r_theta[k] = r_theta.get(k, F_(0)) + L * w
        i = k // S
        for t in range(T):
            r_y[(i, t)] = r_y.get((i, t), F_(0)) + L * w * coef[t]
    total = rhs
    for k, a in r_theta.items():
        i, s = divmod(k, S)
        r = F_(float(inst.prob[s])) - a
        if r < 0:
            D = sum(F_(float(v)) for v in inst.demand[i, :, s])
            U = D * sum(F_(float(x)) for a_ in (inst.c_reg, inst.c_ot, inst.h, inst.b) for x in a_[i])
            total += r * U
    for (i, t), a in r_y.items():
        r = F_(float(inst.F[i, t])) - a
        if r < 0:
            total += r
    return total


# ---------------------------------------------------------------- LP* upper end

def lp_upper(inst: TSSOXLInstance, y: np.ndarray, env: gp.Env | None = None, time_limit: float = 1e9) -> dict[str, Any]:
    """Exact upper bound on LP* (the compact LP relaxation's optimum): an exactly feasible LP point at y (rounded to
    the 2^-64 grid and clipped to [0, 1]) with the recourse repaired as in the module docstring, evaluated exactly."""
    from src.bd.tsso.b_mip_xl import ProductRecourse
    t0 = time.perf_counter()
    I, T, S = inst.I, inst.T, inst.S
    own = env is None
    env = env or _env()
    ed = ExactData(inst)
    yg = [[min(max(v, 0), 1 << GRID) for v in grid_ints(y[i])] for i in range(I)]  # y at scale G
    yf = np.array([[v / TWO_G for v in row] for row in yg])
    G, SC, SP, SF, SK = GRID, ed.SC, ed.SP, ed.SF, ed.SK
    SD = ed.SD
    # recourse cost of product i, scenario s at scale SC + G + SD... we keep every primal value at scale G + SD + SK
    # (production is clipped to K y: K at SK, y at G) -> costs at scale SC_raw + G + SD + SK.
    PS = G + SD + SK
    total_rec = 0  # sum_k p_k cost_k at scale SP + SCr + PS
    craw = [[_ints(a[i], SC) for a in (inst.c_reg, inst.c_ot, inst.h, inst.b)] for i in range(I)]
    try:
        for i in range(I):
            lp = ProductRecourse(inst, i, env)
            res = lp.solve(yf[i], time_limit, duals=False)
            if res["status"] != "optimal":
                lp.dispose()
                raise RuntimeError(f"recourse LP at y: {res['status']}")
            X = [np.asarray(v.X).reshape(S, T) for v in (lp.xr, lp.xo)]
            lp.dispose()
            cap = [ed.Kint[i][t] * yg[i][t] for t in range(T)]  # scale SK + G
            cr, co, hh, bb = craw[i]
            for s in range(S):
                d, _ = ed.demand(i * S + s)
                xr_g = grid_ints(X[0][s])
                xo_g = grid_ints(X[1][s])
                net = 0
                cost = 0
                for t in range(T):
                    xr = min(max(xr_g[t], 0) << (SD + SK), cap[t] << SD)
                    xo = max(xo_g[t], 0) << (SD + SK)
                    net += xr + xo - (d[t] << (G + SK))
                    cost += cr[t] * xr + co[t] * xo + (hh[t] * net if net > 0 else -bb[t] * net)
                total_rec += ed.pint[s] * cost
    finally:
        if own:
            env.dispose()
    fy = 0
    for i in range(I):
        for t in range(T):
            fy += ed.Fint[i][t] * yg[i][t]  # scale SF + G
    scale = SP + SC + PS
    num = (total_rec << max(0, SF + G - scale)) + (fy << max(0, scale - SF - G))
    ub = Fraction(num, 1 << max(scale, SF + G))
    f = float(ub)
    if Fraction(f) < ub:
        f = math.nextafter(f, math.inf)
    return {"ub_exact": ub, "ub": f, "seconds": time.perf_counter() - t0}
