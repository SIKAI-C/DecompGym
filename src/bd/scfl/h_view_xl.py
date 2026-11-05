"""Two-stage stochastic CFL (scale-up, bound side): the two-stage view "scfl-v1" for the canonical certificate evaluator
src/experiments/xl_certificate.py (read-only use of its BDView / RowBlock / Subproblem API), the compressed
certificate format "scfl-cert-c1" and its expansion to the canonical format.

View scfl-v1 (equal to the strong-linking extensive form of src/bd/scfl/b_mip_xl.build_extensive_form relaxed, up to
redundant rows x_ij <= 0 where d_sj = 0):
- first stage: y_i in [0, 1] (ids 'y<i>'), cost f_i; one row 'cover': sum_i u_i y_i >= max_s sum_j d_sj ((15d));
- subproblems: one per scenario s (id 's<s>', trusted weight p_s, uniform 1/S unless the instance lists
  probabilities), variables x_ij (index i*J + j) = flow from facility i to customer j, cost c_ij (the instance's
  unit cost, parsed exactly from its decimal text), rows
      'd<j>':   sum_i x_ij  = d_sj                       (multiplier v_j, free)
      'c<i>':   sum_j x_ij <= 0 - (-u_i) y_i             (multiplier -alpha_i <= 0)
      'l<i>_<j>': x_ij     <= 0 - (-d_sj) y_i            (strong linking; multiplier -beta_ij <= 0)
  boxes 0 <= x_ij <= d_sj (implied by 'd<j>' and x >= 0, so Q_s(y) is preserved at every y; natural bound none);
- theta boxes: the default formula [0, sum_ij c_ij d_sj];
- recourse NOT complete at fractional y (the strong-linking rows): opt cuts from any dual-feasible (v, alpha, beta)
  are valid everywhere (finite boxes), so a penalized or relaxed oracle needs no feasibility cuts; the certified
  master is a relaxation of the compact LP wherever the cuts do not describe the domain.
Data must be integral for capacities and demands (OR-Library CAP and CST generators: yes); unit costs are exact
rationals of their decimal text.

Compressed format scfl-cert-c1 (transport only; sizes are counted on the expanded form):
    {"format": "scfl-cert-c1", "view": "scfl-v1", "version": k,
     "cuts": [{"terms": [[s, [v_1..v_J], [alpha_1..alpha_I]], ...]}, ...]}
A term (s, v, alpha) is the dual (pi_d<j> = v_j, pi_c<i> = -alpha_i, pi_l<i>_<j> = -(v_j - alpha_i - c_ij)^+) of
scenario s (the cheapest completion: r_ij = 0 or c_ij - v_j + alpha_i > 0); its expanded size is
#{j: v_j != 0} + #{i: alpha_i > 0} + #{(i, j): v_j - alpha_i - c_ij > 0}. A cut with several terms is the sum of
its terms over distinct scenarios (weights from the view).
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

VIEW_ID = "scfl-v1"
FORMAT = "scfl-cert-c1"


def _xc():
    import src.experiments.xl_certificate as xc  # read-only use (maintained by another session)
    return xc


# ---------------------------------------------------------------- view

def _fast_rowblock_class():
    xc = _xc()

    class FastRowBlock(xc.RowBlock):
        """A RowBlock built directly from integer data over denominator 1 (no per-value Fraction conversion); the
        row ids, index, senses and A-structure are shared between scenarios (read-only)."""
        __slots__ = ()

        def __init__(self, ids, index, sense, H, A, T, ncols: int, ny: int) -> None:  # noqa: D401 - see class doc
            self.ids, self.index, self.sense = ids, index, sense
            self.H, self.A, self.T = H, A, T
            self.den = 1
            self.min_col, self.max_col = 0, ncols - 1
            self.min_ycol, self.max_ycol = 0, ny - 1
            self._float = None
            self._uniform = {}

    return FastRowBlock


def _fast_subproblem_class():
    xc = _xc()

    class FastSubproblem(xc.Subproblem):
        """A Subproblem with shared integer cost numerators C over `den`, zero lower bounds and upper bounds
        U = demand * den (integers), set directly."""
        __slots__ = ()

        def __init__(self, id: str, weight: Fraction, rows, C: list, den: int, zeros: list, U: list, theta_hi: Fraction) -> None:
            self.id = id
            self.weight = weight
            self.pw = (weight.numerator, weight.denominator)
            self.rows = rows
            self.nvars = len(C)
            self.C, self.L, self.U, self.den = C, zeros, U, den
            self.Cbox = zeros                       # C >= 0 with L = 0: C_j * L_j = 0
            self.K0 = 0
            self.uniform = rows.uniform_rows(self.nvars)
            self._sorted = None
            self.theta_lo = Fraction(0)
            self.theta_hi = theta_hi
            self.theta_override = False
            self.var_ids = None
            self.natural_lb = None
            self.natural_ub = None

    return FastSubproblem


def load_exact(path: str | Path) -> dict:
    """Instance numbers parsed exactly: ints, and Fractions of the decimal text of non-integers."""
    xc = _xc()
    obj = xc.exact_json(Path(path).read_text())
    if obj.get("format") != "scfl-v1":
        raise ValueError("not an scfl-v1 instance")
    return obj


def scfl_view(path: str | Path):
    """The canonical BDView of an instance file (see the module docstring)."""
    xc = _xc()
    obj = load_exact(path)
    cap = [xc.exact(v) for v in obj["capacity"]]
    fx = [xc.exact(v) for v in obj["fixed_cost"]]
    cost = [[xc.exact(v) for v in row] for row in obj["unit_cost"]]
    dem = [[xc.exact(v) for v in row] for row in obj["demand"]]
    I, J, S = len(cap), len(cost[0]), len(dem)
    if any(q.denominator != 1 for q in cap) or any(q.denominator != 1 for row in dem for q in row):
        raise ValueError("scfl-v1 needs integral capacities and demands")
    if any(c < 0 for row in cost for c in row):
        raise ValueError("negative unit cost")
    capi = [int(q) for q in cap]
    demi = [[int(q) for q in row] for row in dem]
    if obj.get("probability") is not None:
        prob = [xc.exact(v) for v in obj["probability"]]
    else:
        prob = [Fraction(1, S)] * S
    FastRowBlock, FastSubproblem = _fast_rowblock_class(), _fast_subproblem_class()
    ids = [f"d{j}" for j in range(J)] + [f"c{i}" for i in range(I)] + [f"l{i}_{j}" for i in range(I) for j in range(J)]
    index = {rid: k for k, rid in enumerate(ids)}
    sense = [xc.EQ] * J + [xc.LE] * (I + I * J)
    A = ([(tuple(i * J + j for i in range(I)), (1,) * I) for j in range(J)]
         + [(tuple(i * J + j for j in range(J)), (1,) * J) for i in range(I)]
         + [((i * J + j,), (1,)) for i in range(I) for j in range(J)])
    empty = ((), ())
    t_cap = [((i,), (-capi[i],)) for i in range(I)]
    zH = [0] * (I + I * J)
    tcache: dict[tuple[int, int], tuple] = {}
    flat_c = [cost[i][j] for i in range(I) for j in range(J)]
    den = math.lcm(*{q.denominator for q in flat_c})
    C = [q.numerator * (den // q.denominator) for q in flat_c]
    Csum_j = [sum(C[i * J + j] for i in range(I)) for j in range(J)]   # sum_i C_ij (for the theta box)
    zeros = [0] * (I * J)
    subs = []
    for s in range(S):
        d = demi[s]
        T = [empty] * J + t_cap
        for i in range(I):
            for j in range(J):
                dj = d[j]
                if dj:
                    key = (i, dj)
                    t = tcache.get(key)
                    if t is None:
                        t = ((i,), (-dj,))
                        tcache[key] = t
                    T.append(t)
                else:
                    T.append(empty)
        rb = FastRowBlock(ids, index, sense, list(d) + zH, A, T, I * J, I)
        U = [d[j] * den for i in range(I) for j in range(J)]
        hi = Fraction(sum(Csum_j[j] * d[j] for j in range(J)), den)
        subs.append(FastSubproblem(f"s{s}", prob[s], rb, C, den, zeros, U, hi))
    dmax = max(sum(row) for row in demi)
    rows = [{"id": "cover", "sense": ">=", "rhs": dmax, "coefs": {i: capi[i] for i in range(I)}}]
    return xc.BDView(VIEW_ID, [f"y{i}" for i in range(I)], fx, subs, rows,
                     meta={"family": "scfl", "I": I, "J": J, "S": S, "name": obj.get("name"), "dmax": dmax})


# ---------------------------------------------------------------- compressed certificates

def cert_bytes(cuts: list, version: Any = None) -> bytes:
    """cuts: list of lists of terms (s, v (J,), alpha (I,))."""
    out = {"format": FORMAT, "view": VIEW_ID, "version": version,
           "cuts": [{"terms": [[int(s), np.asarray(v, dtype=float).tolist(), np.asarray(a, dtype=float).tolist()]
                               for s, v, a in cut]} for cut in cuts]}
    return json.dumps(out, separators=(",", ":")).encode()


def term_sizes(cost: np.ndarray, V: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Expanded size of the terms (V[k], A[k]) (see the module docstring)."""
    V = np.atleast_2d(V)
    A = np.atleast_2d(A)
    out = (V != 0).sum(axis=1) + (A > 0).sum(axis=1)
    for a in range(0, len(V), 256):
        b = min(len(V), a + 256)
        G = V[a:b, None, :] - A[a:b, :, None] - cost[None, :, :]
        out[a:b] += (G > 0).sum(axis=(1, 2))
    return out.astype(np.int64)


def to_generic(data: bytes, cost: np.ndarray, version: Any = None) -> bytes:
    """Canonical certificate (xl_certificate format) of an scfl-cert-c1 version; cost is the float unit-cost
    matrix (the completions are computed in floating point: any multiplier gives a valid safe cut)."""
    obj = json.loads(data)
    if obj.get("format") != FORMAT:
        raise ValueError("format_mismatch")
    I, J = cost.shape
    cuts = []
    for c in obj.get("cuts", []):
        terms = []
        for s, v, a in c["terms"]:
            v = np.asarray(v, dtype=float)
            a = np.asarray(a, dtype=float)
            pi: dict[str, float] = {}
            for j in np.nonzero(v)[0].tolist():
                pi[f"d{j}"] = float(v[j])
            for i in np.nonzero(a > 0)[0].tolist():
                pi[f"c{i}"] = -float(a[i])
            G = v[None, :] - a[:, None] - cost
            ii, jj = np.nonzero(G > 0)
            for i, j, g in zip(ii.tolist(), jj.tolist(), G[ii, jj].tolist()):
                pi[f"l{i}_{j}"] = -g
            terms.append({"s": f"s{int(s)}", "pi": pi})
        cuts.append({"kind": "opt", "terms": terms})
    return json.dumps({"view": VIEW_ID, "version": obj.get("version") if version is None else version, "cuts": cuts},
                      separators=(",", ":")).encode()


def expanded_size(data: bytes, cost: np.ndarray) -> int:
    obj = json.loads(data)
    tot = 0
    for c in obj.get("cuts", []):
        if c["terms"]:
            V = np.asarray([t[1] for t in c["terms"]], dtype=float)
            A = np.asarray([t[2] for t in c["terms"]], dtype=float)
            tot += int(term_sizes(cost, V, A).sum())
    return tot
