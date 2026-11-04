"""k-median (scale-up): the frozen two-stage view "km-v1" for the canonical certificate evaluator
`src/experiments/xl_certificate.py` (protocol v4, Section 3.1; Codex review 46 #3). Read-only use of that module.

View km-v1 (classical y-space; equal to the compact LP of b_mip_xl over the same first-stage rows):
- first stage: y_i in [0, 1] (facility i open; ids 'y<i>', 0-based), cost 0; one row 'card': sum_i y_i = k;
- subproblems: one per customer j, id 'c<j>' (0-based), trusted weight 1, variables x_ij (ids 'x<i>'), cost D_ij;
  row 'd': sum_i x_ij = 1 (multiplier free); rows 'l<i>': x_ij <= y_i, written x_ij <= 0 - (-1) y_i (multiplier
  <= 0); boxes 0 <= x_ij <= 1 (implied by 'd' and x >= 0, so Q_j(y) is preserved at every fractional y with
  sum y >= 1; natural upper bound: none);
- theta boxes: the default formula [sum_i min(0, D_ij), sum_i max(0, D_ij)] = [0, sum_i D_ij];
- recourse complete on the master domain (sum y = k >= 1).
Distances are the instance's exact values (integers for TSPLIB floor distances; otherwise the float64 values of the
frozen formula, taken exactly). The k-median radius formulation has extra master variables and is not part of this
view: its cuts are expressed in y-space (c_naive_bd_xl, b_mip_xl.KMCompactLP converts radius-LP duals to v_j).

Compressed family format -> canonical format (`to_generic`): a km-cert-v1 term (j, v) with implied completion becomes
{"s": "c<j>", "pi": {"d": v, "l<i>": -(v - D_ij) for D_ij < v}}; an explicit term lists its w. The frozen size rule
counts exactly these entries, so both formats have the same size. Numbers are written as JSON floats; the canonical
checker reads their decimal text exactly (for integral distances and integral v every entry is an integer).
"""
from __future__ import annotations

import json
from fractions import Fraction
from typing import Any, Sequence

import numpy as np

from src.bd.km.a_data_xl import KMInstance
from src.bd.km.g_certificate_xl import VIEW_ID, Certificate, Checker, Limits, repair_point


def _xc():
    import src.experiments.xl_certificate as xc  # read-only use (maintained by another session)
    return xc


def km_view(inst: KMInstance, D: np.ndarray | None = None, natural: bool = True):
    """The canonical BDView of an instance (heavy at N >= 3000: n subproblems with m variables each; natural=False
    omits the natural-bound lists, which only the fractional box check uses)."""
    xc = _xc()
    D = inst.distances() if D is None else D
    m, n = D.shape
    integral = bool(np.all(D == np.floor(D)))
    block = xc.RowBlock(["d"] + [f"l{i}" for i in range(m)], [xc.EQ] + [xc.LE] * m, [1] + [0] * m,
                        [{i: 1 for i in range(m)}] + [{i: 1} for i in range(m)], [{}] + [{i: -1} for i in range(m)])
    zeros, ones, nat_ub = [0] * m, [1] * m, ([None] * m if natural else None)
    subs = []
    for j in range(n):
        col = D[:, j]
        cost = [int(x) for x in col.tolist()] if integral else [Fraction(float(x)) for x in col.tolist()]
        subs.append(xc.Subproblem(f"c{j}", 1, block, cost, zeros, ones, natural_ub=nat_ub))
    return xc.BDView(VIEW_ID, [f"y{i}" for i in range(m)], [0] * m, subs,
                     [{"id": "card", "sense": "=", "rhs": int(inst.k), "coefs": {i: 1 for i in range(m)}}],
                     repair=km_repair, meta={"family": "km", "instance": inst.name, "m": m, "n": n, "k": int(inst.k)})


def km_repair(view, y_float: Sequence[float], xs_float: Sequence[Sequence[float]]):
    """Exactly feasible point of the view near a float LP point: y repaired exactly to [0, 1]^m with sum y = k
    (g_certificate_xl.repair_point), then each customer's exact greedy optimum for that y."""
    k = int(view.meta["k"])
    y = repair_point(np.asarray(y_float, dtype=float), k)
    m = view.n
    xs = []
    for sp in view.subproblems:
        x = [Fraction(0)] * m
        rem = Fraction(1)
        for i in sorted(range(m), key=lambda i: sp.C[i]):
            if rem <= 0:
                break
            take = min(y[i], rem)
            if take:
                x[i] = take
                rem -= take
        xs.append(x)
    return y, xs


def to_generic(cert: Certificate, chk: Checker, version: Any = None) -> dict[str, Any]:
    """The canonical certificate equivalent to a km-cert-v1 certificate (see the module docstring)."""
    def term(j: int, v: float) -> dict[str, Any]:
        ix, dd = chk.support(j, v)
        pi = {"d": float(v)}
        for i, d in zip(ix.tolist(), dd.tolist()):
            pi[f"l{i}"] = -(float(v) - d)
        return {"s": f"c{j}", "pi": pi}

    cuts = [{"kind": "opt", "terms": [term(int(j), float(v))]} for j, v in zip(cert.single_j, cert.single_v)]
    for ids, v in cert.multi:
        cuts.append({"kind": "opt", "terms": [term(int(j), float(x)) for j, x in zip(ids, v)]})
    for ids, v, ws in cert.explicit:
        terms = []
        for j, x, (ix, wv) in zip(ids, v, ws):
            pi = {"d": float(x)}
            for i, w in zip(ix, wv):
                pi[f"l{int(i)}"] = -float(w)
            terms.append({"s": f"c{int(j)}", "pi": pi})
        cuts.append({"kind": "opt", "terms": terms})
    return {"view": VIEW_ID, "version": cert.version if version is None else version, "cuts": cuts}


def generic_limits(limits: Limits):
    xc = _xc()
    return xc.CheckerLimits(max_bytes=limits.max_bytes, max_nonzeros=limits.max_nonzeros,
                            max_master_nonzeros=limits.max_master_nonzeros, max_seconds=limits.max_seconds,
                            max_abs_coef=limits.max_abs_coef, max_token_chars=limits.max_token_chars,
                            max_exponent=limits.max_exponent, gurobi_mem_gb=limits.gurobi_mem_gb)


def check_canonical(view, cert: Certificate, chk: Checker, limits: Limits | None = None, deadline: float | None = None):
    """Canonical result (xl_certificate.check_certificate) of a km-cert-v1 certificate, through its canonical bytes."""
    xc = _xc()
    limits = limits or Limits()
    data = json.dumps(to_generic(cert, chk), separators=(",", ":")).encode()
    return xc.check_certificate(view, data, limits=generic_limits(limits), deadline=deadline)


def cross_check(view, chk: Checker, cert: Certificate, limits: Limits | None = None, rel_tol: float = 1e-9) -> dict[str, Any]:
    """Both evaluators on the same certificate: the canonical one (exact LB) and the optimized family backend. They
    solve the master independently, so their exact bounds may differ at the level of Gurobi's tolerances; `agree`
    requires the same cert_ok and |LB_family - LB_canonical| <= rel_tol * max(1, |LB_canonical|)."""
    limits = limits or Limits()
    can = check_canonical(view, cert, chk, limits)
    fam = chk.certify(cert, limits=limits)
    lb_c = can.lb
    lb_f = Fraction(fam["lb"])
    agree = can.cert_ok == fam["cert_ok"] and abs(lb_f - lb_c) <= Fraction(rel_tol) * max(Fraction(1), abs(lb_c))
    return {"agree": bool(agree), "canonical": {"cert_ok": can.cert_ok, "lb": str(lb_c), "lb_float": can.lb_float,
                                                 "reason": can.reason, "seconds": can.stats.get("checker_seconds")},
            "family": {"cert_ok": fam["cert_ok"], "lb": fam["lb"], "lb_float": fam["bound"], "reason": fam["reason"],
                       "seconds": fam.get("checker_seconds")},
            "abs_diff": float(abs(lb_f - lb_c))}
