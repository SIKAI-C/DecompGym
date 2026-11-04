"""Exact checker for Benders LP-master certificates (DecompGym-XL protocol v4, Section 3; Codex reviews 42-43).

The certified quantity is a lower bound on the LP-master value of a two-stage view (`BDView`)
    min f.y + sum_s p_s theta_s  over  y in [lb, ub] (finite; default [0, 1]) with the first-stage rows G y (>=|=|<=) g,
    theta_s in [L_s, U_s],  theta_s >= Q_s(y),
    Q_s(y) = min { c_s.x : A_s x (>=|=|<=) h_s - T_s y,  l_s <= x <= u_s }   (finite boxes, proved per family),
computed only from multipliers published by the solver (the certificate). Nothing the solver claims is trusted:

1. Every multiplier is converted exactly to a rational (`ratio`: decimal text, JSON number token or float); a
   non-finite, unparsable, too long (> 64 characters) or out-of-range (decimal exponent beyond +-400) value rejects
   its cut. Duplicate subproblem ids in a cut, duplicate row ids in a term, unknown ids and malformed terms reject
   the cut too (rejected, never merged or repaired).
2. Sign projection to the dual sign of each row (>=: pi >= 0, <=: pi <= 0, =: free), exactly; the clipped mass is
   reported.
3. Safe cut (Neumaier-Shcherbina) in exact arithmetic with r = c_s - A_s^T pi (feasibility cuts: r = -A_s^T pi):
       Q_s(y) >= pi.(h_s - T_s y) + sum_j min(r_j l_j, r_j u_j)          (opt),
       0      >= pi.(h_s - T_s y) + sum_j min(r_j l_j, r_j u_j)          (feas; pi = -FarkasDual, then projected).
   It holds for every multiplier vector because of the finite boxes, so a forged certificate only weakens the cut.
   An opt cut sums its terms with the view's weights p_s (never weights from the certificate):
       sum_t p_t theta_{s_t} >= sum_t p_t (pi_t.(h - T y) + kappa_t);  a feas cut sums p_t (pi_t.(h - T y) + kappa_t) <= 0.
   The instance data are rationals too (decimal text parsed exactly, never through float).
4. Master LP: Gurobi (one thread) solves the master with the accepted cuts in floating point, only to obtain row
   multipliers lambda. The bound is then re-certified exactly with the ORIGINAL rational rows (never coefficients read
   back from Gurobi): lambda is sign-projected and truncated toward zero onto a dyadic grid (2^-62 relative to its
   largest entry; any multiplier gives a valid bound), and
       LB = sum_r lambda_r b_r + sum_columns min(d_j lo_j, d_j hi_j),   d = obj - M^T lambda,
   over the finite master boxes (y in [lb, ub], theta_s in [L_s, U_s]). LB_cert = max(LB, trivial bound).
5. Failure semantics: a parse failure, a size, nonzero or time limit, a view mismatch, a numerical failure or an
   infeasible master gives cert_ok = False and LB_cert = the trivial bound sum_i min(f_i lb_i, f_i ub_i) + sum_s p_s L_s
   (= sum_i min(0, f_i) + sum_s p_s L_s for y in [0, 1]); never +inf.
6. Checker budget (protocol 3.3.6): `check_versions_supervised` runs all checkpoints of an attempt in a forked child
   under ONE hard wall-clock deadline (reading, hashing, parsing, expansion, master LP, re-certification) and kills it
   at the deadline (unfinished checkpoints: trivial bound, reason "checker_budget_exhausted"); `check_versions` is the
   same logic in process with a cooperative budget. Every distinct stored version is checked against its receipt
   hash and size before its result is computed or reused.

Exactness and speed: all exact quantities are Python integers on explicit common denominators (rows of a block
share one denominator, the cost/box data of a subproblem another, the multipliers of a term a third); a term is
expanded in O(touched coefficients) big-integer operations, untouched columns enter through a precomputed sum. The
master re-certification is one big-integer multiply-add per master nonzero.

LP* reference interval (`lp_interval`, Section 3.4): the deterministic equivalent of the view (= the compact LP
relaxation over the same first-stage rows) is solved by Gurobi; LP_lo = its dual re-certified exactly
(`de_dual_bound`); LP_hi = the exactly evaluated objective of an exactly feasible primal (`de_primal_value`), obtained
from Gurobi's primal by the view's family repair hook or, generically, by exact/rationalized rounding verified in
exact arithmetic. If no primal can be verified, LP_hi is None (the interval is reported as not rigorous).

Family views (frozen; Section 3.1): `ufl_view` (one subproblem per customer, x_ij <= 1 implied, sum y >= 1 as a
first-stage row) and `fctp_view` (one coupled transportation LP, flows <= min(s_i, d_j) implied, no first-stage
rows: exactly the rows of src/bd/fctp/b_mip_xl.build_fctp_mip). Porting another family (TSSO, k-median): build a
`RowBlock` per distinct row structure (shared by subproblems with identical rows) and one `Subproblem` per scenario
or customer, then `BDView(view_id, y_ids, f, subproblems, rows=first-stage rows, repair=...)`; or write the explicit
`bd_view.json` format read by `BDView.from_dict` (see `BDView.to_dict`).
"""
from __future__ import annotations

import bisect
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

GE, LE, EQ = 1, -1, 0
_SENSE = {">=": GE, ">": GE, "G": GE, "<=": LE, "<": LE, "L": LE, "=": EQ, "==": EQ, "E": EQ}
_SENSE_STR = {GE: ">=", LE: "<=", EQ: "="}
_GRB_SENSE = {GE: ">", LE: "<", EQ: "="}

MAX_TOKEN_CHARS = 64
MAX_DECIMAL_EXPONENT = 400
DUAL_GRID_BITS = 62


# ---------------------------------------------------------------- exact numbers

class NumberError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def ratio(value: Any, max_chars: int = MAX_TOKEN_CHARS, max_exp: int = MAX_DECIMAL_EXPONENT) -> tuple[int, int]:
    """Exact (numerator, denominator > 0) of a certificate value: decimal text (a JSON number token or a string),
    an int or a finite float (its exact binary value). Raises NumberError (non_finite, bad_number, token_too_long,
    out_of_range)."""
    t = type(value)
    if t is str:
        if len(value) > max_chars:
            raise NumberError("token_too_long")
        try:
            d = Decimal(value)
        except (InvalidOperation, ValueError):   # (a) certificate value: the cut is rejected with this reason
            raise NumberError("bad_number") from None
        if not d.is_finite():
            raise NumberError("non_finite")
        if not d:
            return 0, 1
        e = d.adjusted()
        if e > max_exp or e < -max_exp:
            raise NumberError("out_of_range")
        return d.as_integer_ratio()
    if t is float:
        if not math.isfinite(value):
            raise NumberError("non_finite")
        return value.as_integer_ratio()
    if t is int:
        if value.bit_length() > 4 * max_exp:
            raise NumberError("out_of_range")
        return value, 1
    if isinstance(value, Fraction):
        return value.numerator, value.denominator
    if isinstance(value, float):  # float subclasses (numpy.float64)
        return ratio(float(value), max_chars, max_exp)
    raise NumberError("bad_number")


def exact(value: Any, allow_float: bool = False) -> Fraction:
    """Exact rational of a MODEL number (view data): int, Fraction, Decimal, decimal text or 'p/q' text; numpy
    integers through .item(). Python floats are rejected (NumberError "float_not_canonical") unless allow_float: a
    float is a binary rounding of the instance's decimal text (json.loads turns 0.1 into 3602879701896397/2^55, which
    would change the certified problem), so frozen views are built only from canonical tokens (`exact_json`, text
    files). Certificate multipliers are parsed by `ratio` instead, where either representation is safe."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool) or value is None:
        raise NumberError("bad_number")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not allow_float:
            raise NumberError("float_not_canonical")
        if not math.isfinite(value):
            raise NumberError("non_finite")
        return Fraction(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise NumberError("non_finite")
        return Fraction(value)
    if isinstance(value, str):
        s = value.strip()
        try:
            d = Decimal(s)
        except (InvalidOperation, ValueError):   # model data: 'p/q' text is tried next
            try:
                return Fraction(s)
            except (ValueError, ZeroDivisionError):   # (b) bad model data: raised, the view is never built
                raise NumberError("bad_number") from None
        if not d.is_finite():
            raise NumberError("non_finite")
        return Fraction(*d.as_integer_ratio())
    if hasattr(value, "item"):
        return exact(value.item(), allow_float)
    raise NumberError("bad_number")


def exact_json(text: str | bytes) -> Any:
    """json.loads with every non-integer number parsed exactly as a Fraction of its decimal text."""
    return json.loads(text, parse_float=lambda s: Fraction(*Decimal(s).as_integer_ratio()))


def float_down(q: Fraction) -> float:
    """Largest float <= q."""
    try:
        f = q.numerator / q.denominator
    except OverflowError:   # reporting only (scored comparisons use the exact Fraction): safe direction
        return sys.float_info.max if q > 0 else -math.inf
    if Fraction(f) > q:
        f = math.nextafter(f, -math.inf)
    return f


def safe_float(q: Fraction) -> float | None:
    """Nearest float of q, or None if q is outside the float range (JSON-safe reporting)."""
    try:
        return q.numerator / q.denominator
    except OverflowError:   # reporting only: outside the float range is reported as None (exact value kept as text)
        return None


def float_up(q: Fraction) -> float:
    """Smallest float >= q."""
    try:
        f = q.numerator / q.denominator
    except OverflowError:   # reporting only (scored comparisons use the exact Fraction): safe direction
        return math.inf if q > 0 else -sys.float_info.max
    if Fraction(f) < q:
        f = math.nextafter(f, math.inf)
    return f


def _lcm(values: Iterable[int]) -> int:
    s = set(values)
    s.discard(1)
    return math.lcm(*s) if s else 1


def _scaled(values: Sequence[Fraction]) -> tuple[list[int], int]:
    den = _lcm(v.denominator for v in values)
    return [v.numerator * (den // v.denominator) for v in values], den


def _sense(s: Any) -> int:
    if s in (GE, LE, EQ) and isinstance(s, int) and not isinstance(s, bool):
        return s
    if s not in _SENSE:
        raise ValueError(f"unknown row sense {s!r}")
    return _SENSE[s]


# ---------------------------------------------------------------- view

class RowBlock:
    """Rows of one subproblem structure (shared by subproblems with identical rows): ids, senses, rhs h, the
    x-coefficients A (by variable index) and the y-coefficients T (by first-stage index) of A x (sense) h - T y.
    All values are stored as integers over the block denominator `den`."""

    __slots__ = ("ids", "index", "sense", "H", "A", "T", "den", "min_col", "max_col", "min_ycol", "max_ycol", "_float", "_uniform")

    def __init__(self, ids: Sequence[str], sense: Sequence[Any], rhs: Sequence[Any],
                 a: Sequence[Mapping[int, Any]], t: Sequence[Mapping[int, Any]] | None = None) -> None:
        n = len(ids)
        t = t if t is not None else [{}] * n
        if not (len(sense) == len(rhs) == len(a) == len(t) == n):
            raise ValueError("row block: inconsistent lengths")
        self.ids = [str(i) for i in ids]
        self.index = {rid: k for k, rid in enumerate(self.ids)}
        if len(self.index) != n:
            raise ValueError("row block: duplicate row ids")
        self.sense = [_sense(s) for s in sense]
        h = [exact(v) for v in rhs]
        a_rows = [(tuple(int(j) for j in row), [exact(v) for v in row.values()]) for row in a]
        t_rows = [(tuple(int(k) for k in row), [exact(v) for v in row.values()]) for row in t]
        dens = [v.denominator for v in h]
        for _, vals in a_rows:
            dens.extend(v.denominator for v in vals)
        for _, vals in t_rows:
            dens.extend(v.denominator for v in vals)
        den = _lcm(dens)
        self.den = den
        self.H = [v.numerator * (den // v.denominator) for v in h]
        self.A = [(cols, tuple(v.numerator * (den // v.denominator) for v in vals)) for cols, vals in a_rows]
        self.T = [(cols, tuple(v.numerator * (den // v.denominator) for v in vals)) for cols, vals in t_rows]
        acols = [j for cols, _ in a_rows for j in cols]
        tcols = [k for cols, _ in t_rows for k in cols]
        self.min_col, self.max_col = (min(acols), max(acols)) if acols else (0, -1)
        self.min_ycol, self.max_ycol = (min(tcols), max(tcols)) if tcols else (0, -1)
        self._float = None
        self._uniform: dict[int, dict[int, int]] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def uniform_rows(self, nvars: int) -> dict[int, int]:
        """Rows with the same coefficient on every one of the nvars variables (e.g. the UFL/k-median assignment row
        sum_i x_ij = 1): {row index: coefficient numerator}. Their multipliers shift all reduced costs equally, which
        `expand_term` evaluates in O(log nvars) instead of O(nvars)."""
        if nvars not in self._uniform:
            out = {}
            for i, (cols, vals) in enumerate(self.A):
                if nvars and len(cols) == nvars and len(set(vals)) == 1 and len(set(cols)) == nvars:
                    out[i] = vals[0]
            self._uniform[nvars] = out
        return self._uniform[nvars]

    def floats(self) -> tuple[list[float], list[tuple[float, ...]], list[tuple[float, ...]]]:
        """Float images (h, A values, T values) for the floating-point solves only."""
        if self._float is None:
            d = self.den
            self._float = ([h / d for h in self.H], [tuple(v / d for v in vals) for _, vals in self.A],
                           [tuple(v / d for v in vals) for _, vals in self.T])
        return self._float


class Subproblem:
    """One recourse LP: weight p_s > 0 (trusted, from the view), its rows (a RowBlock, possibly shared), costs c,
    finite boxes [l, u] (integers over `den`), the theta box [L_s, U_s] and, for the fractional box check only, the
    natural bounds the boxes replace (None = infinite)."""

    __slots__ = ("id", "weight", "pw", "rows", "nvars", "C", "L", "U", "den", "Cbox", "K0", "theta_lo", "theta_hi",
                 "theta_override", "var_ids", "natural_lb", "natural_ub", "uniform", "_sorted")

    def __init__(self, id: str, weight: Any, rows: RowBlock, cost: Sequence[Any], lb: Sequence[Any], ub: Sequence[Any],
                 *, theta_lo: Any = None, theta_hi: Any = None, var_ids: Sequence[str] | None = None,
                 natural_lb: Sequence[Any] | None = None, natural_ub: Sequence[Any] | None = None) -> None:
        self.id = str(id)
        self.weight = exact(weight)
        if self.weight <= 0:
            raise ValueError(f"subproblem {id}: weight must be positive")
        self.pw = (self.weight.numerator, self.weight.denominator)
        self.rows = rows
        nv = len(cost)
        if len(lb) != nv or len(ub) != nv:
            raise ValueError(f"subproblem {id}: cost and box lengths differ")
        self.nvars = nv
        c = [exact(v) for v in cost]
        lo = [exact(v) for v in lb]
        hi = [exact(v) for v in ub]
        if any(a > b for a, b in zip(lo, hi)):
            raise ValueError(f"subproblem {id}: lb > ub")
        if rows.min_col < 0 or rows.max_col >= nv:
            raise ValueError(f"subproblem {id}: row coefficient on an unknown variable")
        den = _lcm([v.denominator for v in c] + [v.denominator for v in lo] + [v.denominator for v in hi])
        self.den = den
        self.C = [v.numerator * (den // v.denominator) for v in c]
        self.L = [v.numerator * (den // v.denominator) for v in lo]
        self.U = [v.numerator * (den // v.denominator) for v in hi]
        self.Cbox = [cj * (lj if cj >= 0 else uj) for cj, lj, uj in zip(self.C, self.L, self.U)]
        self.K0 = sum(self.Cbox)
        self.uniform = rows.uniform_rows(nv)
        self._sorted = None
        lo_default = Fraction(self.K0, den * den)
        hi_default = Fraction(sum(cj * (uj if cj >= 0 else lj) for cj, lj, uj in zip(self.C, self.L, self.U)), den * den)
        self.theta_lo = lo_default if theta_lo is None else exact(theta_lo)
        self.theta_hi = hi_default if theta_hi is None else exact(theta_hi)
        if self.theta_lo > self.theta_hi:
            raise ValueError(f"subproblem {id}: theta_lo > theta_hi")
        # An explicit theta box tighter than the default formula needs a family proof (protocol 3.1).
        self.theta_override = self.theta_lo > lo_default or self.theta_hi < hi_default
        self.var_ids = list(var_ids) if var_ids is not None else None
        self.natural_lb = list(natural_lb) if natural_lb is not None else None
        self.natural_ub = list(natural_ub) if natural_ub is not None else None

    def cost(self, j: int) -> Fraction:
        return Fraction(self.C[j], self.den)

    def box(self, j: int) -> tuple[Fraction, Fraction]:
        return Fraction(self.L[j], self.den), Fraction(self.U[j], self.den)

    def sorted_sums(self) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
        """Costs in ascending order with prefix sums of C*U and U and suffix sums of C*L and L (lazily, for the
        uniform-row shift): for a shift t, the variables with C_j < t are the first k = bisect_left(Cs, t)."""
        if self._sorted is None:
            order = sorted(range(self.nvars), key=self.C.__getitem__)
            cs = [self.C[j] for j in order]
            pcu, pu, scl, sl = [0], [0], [0], [0]
            for j in order:
                pcu.append(pcu[-1] + self.C[j] * self.U[j])
                pu.append(pu[-1] + self.U[j])
            for j in reversed(order):
                scl.append(scl[-1] + self.C[j] * self.L[j])
                sl.append(sl[-1] + self.L[j])
            self._sorted = (cs, pcu, pu, scl[::-1], sl[::-1])
        return self._sorted


class _Row:
    """A master or first-stage row sum nums_j / den * z_cols_j (sense) rhs / den, with its float image."""

    __slots__ = ("id", "sense", "cols", "nums", "rhs", "den", "fvals", "frhs")

    def __init__(self, id: str, sense: int, cols: list[int], nums: list[int], rhs: int, den: int) -> None:
        self.id, self.sense, self.cols, self.nums, self.rhs, self.den = id, sense, cols, nums, rhs, den
        self.fvals = [v / den for v in nums]
        self.frhs = rhs / den


class BDView:
    """Frozen two-stage view of a BD-LP family instance (protocol v4, Section 3.1)."""

    def __init__(self, view_id: str, y_ids: Sequence[str], f: Sequence[Any], subproblems: Sequence[Subproblem],
                 rows: Sequence[Mapping[str, Any]] = (), *, y_lb: Sequence[Any] | None = None, y_ub: Sequence[Any] | None = None,
                 repair: Callable | None = None, meta: Mapping[str, Any] | None = None, theta_proof: str | None = None) -> None:
        """`theta_proof`: reference to the frozen family proof (e.g. a note section) that makes this a trusted view.
        A theta box tighter than the default formula is refused unless it is given (protocol 3.1)."""
        self.view_id = str(view_id)
        self.y_ids = [str(v) for v in y_ids]
        n = len(self.y_ids)
        self.n = n
        self.y_index = {v: k for k, v in enumerate(self.y_ids)}
        if len(self.y_index) != n:
            raise ValueError("duplicate first-stage variable ids")
        self.f = [exact(v) for v in f]
        self.y_lo = [exact(v) for v in y_lb] if y_lb is not None else [Fraction(0)] * n
        self.y_hi = [exact(v) for v in y_ub] if y_ub is not None else [Fraction(1)] * n
        if len(self.f) != n or len(self.y_lo) != n or len(self.y_hi) != n:
            raise ValueError("first-stage cost/box lengths differ from the number of variables")
        if any(a > b for a, b in zip(self.y_lo, self.y_hi)):
            raise ValueError("first-stage lb > ub")
        self.subproblems = list(subproblems)
        if not self.subproblems:
            raise ValueError("a view needs at least one subproblem")
        self.index = {sp.id: k for k, sp in enumerate(self.subproblems)}
        if len(self.index) != len(self.subproblems):
            raise ValueError("duplicate subproblem ids")
        self.theta_proof = theta_proof
        overrides = [sp.id for sp in self.subproblems if sp.theta_override]
        if overrides and not theta_proof:
            raise ValueError(f"theta boxes tighter than the default formula ({overrides[:5]}) need a trusted view with a proof "
                             "reference (theta_proof)")
        for sp in self.subproblems:
            if sp.rows.min_ycol < 0 or sp.rows.max_ycol >= n:
                raise ValueError(f"subproblem {sp.id}: T coefficient on an unknown first-stage variable")
        self.fs_rows: list[_Row] = []
        for r in rows:
            coefs = r.get("coefs", {})
            cols, vals = [], []
            for k, v in coefs.items():
                kk = self.y_index[k] if isinstance(k, str) else int(k)
                if not 0 <= kk < n:
                    raise ValueError(f"first-stage row {r.get('id')}: unknown variable {k}")
                cols.append(kk)
                vals.append(exact(v))
            rhs = exact(r["rhs"])
            nums, den = _scaled(vals + [rhs])
            self.fs_rows.append(_Row(str(r.get("id", f"r{len(self.fs_rows)}")), _sense(r["sense"]), cols, nums[:-1], nums[-1], den))
        self.repair = repair
        self.meta = dict(meta or {})
        if theta_proof:
            self.meta["theta_proof"] = theta_proof

    @property
    def S(self) -> int:
        return len(self.subproblems)

    def trivial_bound(self) -> Fraction:
        """sum_i min(f_i lb_i, f_i ub_i) + sum_s p_s L_s: valid whatever the certificate (failure value)."""
        b = sum((min(fi * lo, fi * hi) for fi, lo, hi in zip(self.f, self.y_lo, self.y_hi)), Fraction(0))
        return b + sum((sp.weight * sp.theta_lo for sp in self.subproblems), Fraction(0))

    def theta_overrides(self) -> list[str]:
        """Subproblems whose theta box is tighter than the default formula (allowed only with `theta_proof`)."""
        return [sp.id for sp in self.subproblems if sp.theta_override]

    def stats(self) -> dict[str, Any]:
        blocks = {id(sp.rows) for sp in self.subproblems}
        return {"view": self.view_id, "first_stage_vars": self.n, "first_stage_rows": len(self.fs_rows),
                "subproblems": self.S, "row_blocks": len(blocks),
                "subproblem_rows": sum(len(sp.rows) for sp in self.subproblems),
                "subproblem_vars": sum(sp.nvars for sp in self.subproblems)}

    # ------------------------------------------------------------ explicit JSON form

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any], repair: Callable | None = None) -> "BDView":
        """Explicit view (`bd_view.json`):
        {"view": id,
         "first_stage": {"vars": [{"id", "cost", "lb"?, "ub"?}], "rows": [{"id", "sense", "rhs", "coefs": {y_id: v}}]},
         "blocks": {name: [row, ...]}?,
         "subproblems": [{"id", "weight", "vars": [{"id", "cost", "lb", "ub", "natural_lb"?, "natural_ub"?}],
                          "rows": name | [{"id", "sense", "rhs", "a": {x_id: v}, "t": {y_id: v}}], "theta_lo"?, "theta_hi"?}]}
        Numbers may be ints, decimal strings, 'p/q' strings or Fractions. A missing natural bound equals the box; null
        means infinite."""
        fs = obj.get("first_stage", {})
        yv = fs.get("vars", [])
        y_ids = [v["id"] for v in yv]
        y_index = {v: k for k, v in enumerate(y_ids)}
        blocks_src = obj.get("blocks", {}) or {}
        cache: dict[tuple, RowBlock] = {}

        def block(rows_src: list, x_ids: list[str], key: Any) -> RowBlock:
            ck = (key, tuple(x_ids)) if key is not None else None
            if ck is not None and ck in cache:
                return cache[ck]
            x_index = {v: k for k, v in enumerate(x_ids)}
            ids, sense, rhs, a, t = [], [], [], [], []
            for r in rows_src:
                ids.append(r["id"])
                sense.append(r["sense"])
                rhs.append(r.get("rhs", 0))
                a.append({x_index[k]: v for k, v in (r.get("a") or {}).items()})
                t.append({y_index[k]: v for k, v in (r.get("t") or {}).items()})
            rb = RowBlock(ids, sense, rhs, a, t)
            if ck is not None:
                cache[ck] = rb
            return rb

        subs = []
        for s in obj["subproblems"]:
            xv = s["vars"]
            x_ids = [v["id"] for v in xv]
            rows = s["rows"]
            rb = block(blocks_src[rows], x_ids, rows) if isinstance(rows, str) else block(rows, x_ids, None)
            nat_lb = [v.get("natural_lb", v.get("lb", 0)) for v in xv]
            nat_ub = [v.get("natural_ub", v.get("ub")) for v in xv]
            subs.append(Subproblem(s["id"], s.get("weight", 1), rb, [v.get("cost", 0) for v in xv], [v["lb"] for v in xv],
                                   [v["ub"] for v in xv], theta_lo=s.get("theta_lo"), theta_hi=s.get("theta_hi"), var_ids=x_ids,
                                   natural_lb=[None if b is None else exact(b) for b in nat_lb],
                                   natural_ub=[None if b is None else exact(b) for b in nat_ub]))
        rows = [{"id": r.get("id"), "sense": r["sense"], "rhs": r["rhs"], "coefs": r.get("coefs", {})} for r in fs.get("rows", [])]
        return cls(obj["view"], y_ids, [v.get("cost", 0) for v in yv], subs, rows,
                   y_lb=[v.get("lb", 0) for v in yv], y_ub=[v.get("ub", 1) for v in yv], repair=repair, meta=obj.get("meta"),
                   theta_proof=obj.get("theta_proof"))

    @classmethod
    def from_json(cls, path: str | Path, repair: Callable | None = None) -> "BDView":
        return cls.from_dict(exact_json(Path(path).read_text()), repair=repair)

    def to_dict(self) -> dict[str, Any]:
        """Explicit JSON form (numbers as exact strings); shared row blocks are written once."""
        q = lambda v: str(v)
        y_vars = [{"id": y, "cost": q(c), "lb": q(lo), "ub": q(hi)} for y, c, lo, hi in zip(self.y_ids, self.f, self.y_lo, self.y_hi)]
        fs_rows = [{"id": r.id, "sense": _SENSE_STR[r.sense], "rhs": q(Fraction(r.rhs, r.den)),
                    "coefs": {self.y_ids[c]: q(Fraction(v, r.den)) for c, v in zip(r.cols, r.nums)}} for r in self.fs_rows]
        blocks: dict[str, list] = {}
        names: dict[int, str] = {}
        subs = []
        for sp in self.subproblems:
            x_ids = sp.var_ids or [f"x{j}" for j in range(sp.nvars)]
            rb = sp.rows
            key = id(rb)
            if key not in names:
                names[key] = f"b{len(names)}"
                d = rb.den
                blocks[names[key]] = [{"id": rid, "sense": _SENSE_STR[s], "rhs": q(Fraction(h, d)),
                                       "a": {x_ids[j]: q(Fraction(v, d)) for j, v in zip(*rb.A[i])},
                                       "t": {self.y_ids[k]: q(Fraction(v, d)) for k, v in zip(*rb.T[i])}}
                                      for i, (rid, s, h) in enumerate(zip(rb.ids, rb.sense, rb.H))]
            xv = []
            for j in range(sp.nvars):
                lo, hi = sp.box(j)
                v = {"id": x_ids[j], "cost": q(sp.cost(j)), "lb": q(lo), "ub": q(hi)}
                if sp.natural_lb is not None and sp.natural_lb[j] != lo:
                    v["natural_lb"] = None if sp.natural_lb[j] is None else q(sp.natural_lb[j])
                if sp.natural_ub is not None and sp.natural_ub[j] != hi:
                    v["natural_ub"] = None if sp.natural_ub[j] is None else q(sp.natural_ub[j])
                xv.append(v)
            subs.append({"id": sp.id, "weight": q(sp.weight), "vars": xv, "rows": names[key],
                         "theta_lo": q(sp.theta_lo), "theta_hi": q(sp.theta_hi)})
        out = {"view": self.view_id, "first_stage": {"vars": y_vars, "rows": fs_rows}, "blocks": blocks,
               "subproblems": subs, "meta": self.meta}
        if self.theta_proof:
            out["theta_proof"] = self.theta_proof
        return out


# ---------------------------------------------------------------- certificate parsing

@dataclass
class CheckerLimits:
    """Per-check limits (protocol 3.2 initial values; frozen per family before calibration)."""
    max_bytes: int = 256 * 2**20
    max_nonzeros: int = 2_000_000          # multiplier entries in one certificate
    max_expanded: int = 200_000_000        # coefficient operations while expanding cuts (~1 min; time is the real cap)
    max_master_nonzeros: int = 20_000_000
    max_seconds: float = 600.0
    max_abs_coef: float = 1e12             # a cut with a larger float coefficient or rhs is rejected (numerics)
    max_token_chars: int = MAX_TOKEN_CHARS
    max_exponent: int = MAX_DECIMAL_EXPONENT
    gurobi_mem_gb: float | None = 8.0
    threads: int = 1
    method: int = -1
    feasibility_tol: float = 1e-9          # Gurobi tolerances only affect how tight LB is, never its validity
    optimality_tol: float = 1e-9


class _Dup(dict):
    """A JSON object that had duplicate keys (its terms or multipliers are ambiguous: the cut is rejected)."""


def _pairs(pairs: list) -> dict:
    d = dict(pairs)
    return d if len(d) == len(pairs) else _Dup(d)


def _structure_tokens(text: str) -> int:
    return text.count(",") + text.count(":") + text.count("[") + text.count("{")


def parse_certificate_bytes(data: bytes, limits: CheckerLimits) -> Any:
    """Parse with every number kept as its token text (converted exactly later) and duplicate keys flagged. The bytes
    must be strict UTF-8 without a byte-order mark (any other encoding is malformed: "not_utf8" / "bom"); the
    structural pre-scan runs on the same decoded text that is parsed and bounds the number of JSON values."""
    if len(data) > limits.max_bytes:
        raise _CertFailure("too_many_bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:   # (a) certificate rejected: not UTF-8
        raise _CertFailure("not_utf8") from None
    if text.startswith("\ufeff"):
        raise _CertFailure("bom")
    if _structure_tokens(text) > 8 * limits.max_nonzeros + 1000:
        raise _CertFailure("too_many_values")
    try:
        return json.loads(text, parse_float=str, parse_int=str, parse_constant=str, object_pairs_hook=_pairs)
    except (ValueError, RecursionError, UnicodeDecodeError) as exc:   # (a) certificate rejected: not valid JSON
        raise _CertFailure("parse_error", str(exc)[:200]) from None


class _CertFailure(Exception):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason, self.detail = reason, detail


class CertificateRejected(Exception):
    """Raised by a `convert` hook (e.g. a family certificate format -> the generic format) for malformed input: the
    certificate is rejected with this reason and charged to the agent. Any other exception of the hook is an
    evaluator failure (infrastructure); a MemoryError goes through the memory attribution gate."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CutError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class _Stats:
    cuts: int = 0
    accepted: int = 0
    opt_cuts: int = 0
    feas_cuts: int = 0
    trivial: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    terms: int = 0
    nonzeros: int = 0
    expanded: int = 0
    clipped_count: int = 0
    clipped_mass: Fraction = Fraction(0)
    infeasible_cut: bool = False

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


# ---------------------------------------------------------------- safe cut construction (exact)

def expand_term(sp: Subproblem, pi: Mapping[str, Any], zero_cost: bool, stats: _Stats | None = None,
                limits: CheckerLimits | None = None) -> tuple[int, int, dict[int, int]]:
    """Exact safe cut of one term: returns (den, const, ycoef) with
        Q_s(y) >= const/den + sum_k ycoef_k/den * y_k        (opt; zero_cost=False)
        0      >= const/den + sum_k ycoef_k/den * y_k        (feas; zero_cost=True: r = -A^T pi),
    after exact sign projection of pi. Raises CutError for unknown rows or unusable numbers."""
    limits = limits or _DEFAULT_LIMITS
    stats = stats if stats is not None else _Stats()
    rows = sp.rows
    index, sense, H, A, T = rows.index, rows.sense, rows.H, rows.A, rows.T
    entries = []
    dens = set()
    mc, me = limits.max_token_chars, limits.max_exponent
    for rid, val in pi.items():
        i = index.get(rid)
        if i is None:
            raise CutError("unknown_row")
        try:
            num, den = ratio(val, mc, me)
        except NumberError as exc:   # (a) the cut is rejected with this reason
            raise CutError(exc.reason) from None
        if not num:
            continue
        s = sense[i]
        if (s == GE and num < 0) or (s == LE and num > 0):
            stats.clipped_count += 1
            stats.clipped_mass += Fraction(abs(num), den)
            continue
        entries.append((i, num, den))
        dens.add(den)
    dens.discard(1)
    dpi = math.lcm(*dens) if dens else 1
    R: dict[int, int] = {}
    Y: dict[int, int] = {}
    hs = 0
    ru = 0          # sum of a_i p_i over uniform rows: the same shift of every reduced cost
    work = 0
    rget, yget = R.get, Y.get
    uni = sp.uniform
    budget = limits.max_expanded - stats.expanded
    for count, (i, num, den) in enumerate(entries):
        if not count % 64 and work > budget:
            stats.expanded += work
            raise _CertFailure("too_much_expansion")
        p = num if den == dpi else num * (dpi // den)
        h = H[i]
        if h:
            hs += p * h
        au = uni.get(i) if uni else None
        if au is not None:
            ru += au * p
            work += 1
        else:
            cols, vals = A[i]
            for j, a in zip(cols, vals):
                R[j] = rget(j, 0) + a * p
            work += len(cols)
        tc, tv = T[i]
        for k, t in zip(tc, tv):
            Y[k] = yget(k, 0) + t * p
        work += len(tc) + 1
    stats.expanded += work
    if stats.expanded > limits.max_expanded:   # before the cut can be accepted or rejected
        raise _CertFailure("too_much_expansion")
    dr, dv = rows.den, sp.den
    dv2 = dv * dv
    scale = dr * dpi
    L, U = sp.L, sp.U
    # rho_j = (r_j) * dv * dr * dpi with r_j = c_j - (A^T pi)_j; untouched columns (no non-uniform entry) have
    # rho0_j = C_j scale - ru dv and are summed in closed form; touched columns are corrected one by one.
    shift = ru * dv
    if zero_cost:
        rho0 = -shift
        if rho0:
            _, _, pu, _, sl = sp.sorted_sums()
            kap = rho0 * (sl[0] if rho0 >= 0 else pu[-1])
        else:
            kap = 0
        for j, rj in R.items():
            if rj:
                rho = rho0 - rj * dv
                kap += rho * (L[j] if rho >= 0 else U[j]) - (rho0 * (L[j] if rho0 >= 0 else U[j]) if rho0 else 0)
    elif not shift:
        C, cbox = sp.C, sp.Cbox
        kap = sp.K0 * scale
        for j, rj in R.items():
            rho = C[j] * scale - rj * dv
            kap += rho * (L[j] if rho >= 0 else U[j]) - cbox[j] * scale
    else:
        C = sp.C
        cs, pcu, pu, scl, sl = sp.sorted_sums()
        k = bisect.bisect_left(cs, Fraction(shift, scale))   # columns with C_j scale < shift have rho0 < 0 (use U)
        kap = scale * (pcu[k] + scl[k]) - shift * (pu[k] + sl[k])
        for j, rj in R.items():
            r0 = C[j] * scale - shift
            rho = r0 - rj * dv
            kap += rho * (L[j] if rho >= 0 else U[j]) - r0 * (L[j] if r0 >= 0 else U[j])
    const = hs * dv2 + kap
    ycoef = {k: -v * dv2 for k, v in Y.items() if v}
    return dv2 * scale, const, ycoef


def build_cut(view: BDView, cut: Any, stats: _Stats, limits: CheckerLimits, stop: float | None = None,
              clock: Callable[[], float] = time.monotonic) -> _Row | None:
    """One certificate cut as a master row  sum_s theta_coef_s theta_s + sum_k a_k y_k >= rhs  (integers over den).
    Returns None for a trivially satisfied row; raises CutError for a rejected cut (and _CertFailure at `stop`)."""
    if not isinstance(cut, dict) or type(cut) is _Dup:
        raise CutError("malformed")
    kind = cut.get("kind")
    if kind not in ("opt", "feas"):
        raise CutError("bad_kind")
    terms = cut.get("terms")
    if not isinstance(terms, list) or not terms:
        raise CutError("no_terms")
    seen: set[str] = set()
    parts = []
    feas = kind == "feas"
    for count, term in enumerate(terms):
        if stop is not None and count and not count % 512 and clock() > stop:
            raise _CertFailure("time_limit")
        if not isinstance(term, dict) or type(term) is _Dup:
            raise CutError("malformed_term")
        sid = term.get("s")
        if not isinstance(sid, str):
            raise CutError("malformed_term")
        if sid in seen:
            raise CutError("duplicate_id")
        seen.add(sid)
        si = view.index.get(sid)
        if si is None:
            raise CutError("unknown_id")
        pi = term.get("pi")
        if type(pi) is _Dup:
            raise CutError("duplicate_row")
        if not isinstance(pi, dict):
            raise CutError("malformed_term")
        sp = view.subproblems[si]
        parts.append((si, sp.pw) + expand_term(sp, pi, feas, stats, limits))
        stats.terms += 1
    drow = _lcm(den * pw[1] for _, pw, den, _, _ in parts)
    theta: dict[int, int] = {}
    ycoef: dict[int, int] = {}
    rhs = 0
    yget = ycoef.get
    for si, (pn, pd), den, const, yc in parts:
        fct = pn * (drow // (den * pd))
        rhs += fct * const
        for k, g in yc.items():
            ycoef[k] = yget(k, 0) - fct * g
        if not feas:
            theta[si] = pn * (drow // pd)
    cols, nums = [], []
    for k, v in ycoef.items():
        if v:
            cols.append(k)
            nums.append(v)
    n = view.n
    for si, v in theta.items():
        cols.append(n + si)
        nums.append(v)
    if not cols:
        if rhs > 0:
            stats.infeasible_cut = True  # 0 >= positive: no first-stage point has feasible recourse
        return None
    g = math.gcd(drow, rhs, *nums)
    if g > 1:
        drow //= g
        rhs //= g
        nums = [v // g for v in nums]
    try:
        row = _Row(f"cut{stats.cuts}", GE, cols, nums, rhs, drow)
    except OverflowError:   # (a) the cut is rejected: coefficients outside the float range
        raise CutError("magnitude") from None
    m = limits.max_abs_coef
    if abs(row.frhs) > m or any(abs(v) > m for v in row.fvals):
        raise CutError("magnitude")
    if feas:
        stats.feas_cuts += 1
    else:
        stats.opt_cuts += 1
    return row


_DEFAULT_LIMITS = CheckerLimits()


# ---------------------------------------------------------------- exact re-certification

def snap_duals(duals: Sequence[float], senses: Sequence[int]) -> tuple[list[int], int]:
    """Sign-projected multipliers truncated toward zero onto the grid 2^-E (E so that the largest is below 2^62):
    returns integers Lam with lambda_r = Lam_r / 2^E. Any multiplier vector gives a valid bound."""
    lam = []
    for v, s in zip(duals, senses):
        v = float(v)
        if not math.isfinite(v) or (s == GE and v < 0) or (s == LE and v > 0):
            v = 0.0
        lam.append(v)
    big = max((abs(v) for v in lam), default=0.0)
    if big == 0.0:
        return [0] * len(lam), 0
    e = max(0, DUAL_GRID_BITS - math.frexp(big)[1])
    return [int(math.ldexp(v, e)) for v in lam], e


def recertify(obj: Sequence[Fraction], lo: Sequence[Fraction], hi: Sequence[Fraction], rows: Sequence[_Row],
              duals: Sequence[float], deadline: float | None = None, clock: Callable[[], float] = time.monotonic) -> Fraction:
    """Exact bound sum_r lambda_r b_r + sum_j min(d_j lo_j, d_j hi_j), d = obj - M^T lambda, over the finite boxes,
    with lambda = snap_duals(duals) and the rows' exact integer data."""
    lam, e = snap_duals(duals, [r.sense for r in rows])
    lc = _lcm(r.den for r, l in zip(rows, lam) if l)
    g = (1 << e) * lc
    nc = len(obj)
    acc = [0] * nc
    rs = 0
    for count, (r, l) in enumerate(zip(rows, lam)):
        if deadline is not None and not count % 256 and clock() > deadline:
            raise _CertFailure("time_limit")
        if not l:
            continue
        fct = l * (lc // r.den)
        if r.rhs:
            rs += fct * r.rhs
        for c, v in zip(r.cols, r.nums):
            acc[c] += fct * v
    bound = Fraction(rs, g)
    for c in range(nc):
        if deadline is not None and not c % 1024 and clock() > deadline:
            raise _CertFailure("time_limit")
        d = obj[c] - Fraction(acc[c], g) if acc[c] else obj[c]
        bound += d * lo[c] if d >= 0 else d * hi[c]
    if deadline is not None and clock() > deadline:
        raise _CertFailure("time_limit")
    return bound


# ---------------------------------------------------------------- master LP (floating point, multipliers only)

@dataclass
class MasterSolve:
    status: str                      # optimal | infeasible | time_limit | numerical | memory_limit | error
    duals: list[float] | None = None
    objective: float | None = None
    seconds: float = 0.0
    detail: str = ""


_TRUSTED_ENV: Any = None   # the Gurobi environment started by the supervised checker before setup is acknowledged


def trusted_init(limits: CheckerLimits) -> Any:
    """Trusted checker initialization: the NumPy/SciPy/gurobipy imports and a started Gurobi environment (licensing,
    one thread, MemLimit); reused by every master solve of the process."""
    global _TRUSTED_ENV
    import numpy  # noqa: F401
    import scipy.sparse  # noqa: F401
    import gurobipy as gp

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", limits.threads)
    if limits.gurobi_mem_gb:
        env.setParam("MemLimit", limits.gurobi_mem_gb)
    env.start()
    _TRUSTED_ENV = env
    return env


def gurobi_master(lo: Sequence[float], hi: Sequence[float], obj: Sequence[float], rows: Sequence[_Row], time_limit: float,
                  limits: CheckerLimits) -> MasterSolve:
    """Solve min obj.z over the box and the rows (floats) with Gurobi, one thread; return the row multipliers. Uses the
    trusted environment started by `trusted_init` when there is one, else starts (and disposes) its own."""
    t0 = time.monotonic()
    try:
        import gurobipy as gp
        from gurobipy import GRB
        import numpy as np
        import scipy.sparse as sps
    except ImportError as exc:  # pragma: no cover - (b) environment problem -> master_solver_error (infrastructure)
        return MasterSolve("error", detail=f"import: {exc}")
    env = model = None
    own_env = _TRUSTED_ENV is None
    try:
        if own_env:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.setParam("Threads", limits.threads)
            if limits.gurobi_mem_gb:
                env.setParam("MemLimit", limits.gurobi_mem_gb)
            env.start()
        else:
            env = _TRUSTED_ENV
        model = gp.Model(env=env)
        model.Params.TimeLimit = max(0.01, time_limit)
        model.Params.Method = limits.method
        model.Params.FeasibilityTol = limits.feasibility_tol
        model.Params.OptimalityTol = limits.optimality_tol
        z = model.addMVar(len(obj), lb=np.asarray(lo, dtype=float), ub=np.asarray(hi, dtype=float), obj=np.asarray(obj, dtype=float))
        if rows:
            indptr = [0]
            indices: list[int] = []
            data: list[float] = []
            for r in rows:
                indices.extend(r.cols)
                data.extend(r.fvals)
                indptr.append(len(indices))
            A = sps.csr_matrix((np.asarray(data, dtype=float), np.asarray(indices, dtype=np.int64), np.asarray(indptr, dtype=np.int64)),
                               shape=(len(rows), len(obj)))
            model.addMConstr(A, z, np.array([_GRB_SENSE[r.sense] for r in rows]), np.asarray([r.frhs for r in rows], dtype=float))
        model.optimize()
        st = model.Status
        if st == GRB.OPTIMAL:
            duals = model.getAttr("Pi", model.getConstrs()) if rows else []
            return MasterSolve("optimal", list(duals), float(model.ObjVal), time.monotonic() - t0)
        status = {GRB.INFEASIBLE: "infeasible", GRB.INF_OR_UNBD: "infeasible", GRB.TIME_LIMIT: "time_limit",
                  GRB.MEM_LIMIT: "memory_limit", GRB.NUMERIC: "numerical", GRB.SUBOPTIMAL: "numerical"}.get(st, f"status_{st}")
        return MasterSolve(status, seconds=time.monotonic() - t0)
    except gp.GurobiError as exc:   # out of memory -> the memory gate; any other solver error -> infrastructure
        status = "memory_limit" if getattr(exc, "errno", None) == 10001 else "error"
        return MasterSolve(status, seconds=time.monotonic() - t0, detail=str(exc)[:200])
    finally:
        if model is not None:
            model.dispose()
        if env is not None and own_env:
            env.dispose()


# ---------------------------------------------------------------- certificate check

@dataclass
class CertResult:
    cert_ok: bool
    lb: Fraction                      # LB_cert (exact): the trivial bound on failure
    trivial: Fraction
    reason: str | None = None
    version: str | None = None        # compact scalar metadata only (<= 64 characters), else None
    stats: dict[str, Any] = field(default_factory=dict)
    infrastructure: str | None = None  # an evaluator-side failure (never charged to the certificate)

    @property
    def lb_float(self) -> float:
        return float_down(self.lb)

    def to_dict(self) -> dict[str, Any]:
        return {"cert_ok": self.cert_ok, "lb": str(self.lb), "lb_float": self.lb_float, "trivial": str(self.trivial),
                "reason": self.reason, "version": self.version, "stats": self.stats, "infrastructure": self.infrastructure}


MAX_VERSION_CHARS = 64


def _compact_version(v: Any) -> str | None:
    """The certificate's "version" as compact scalar metadata: a JSON number or string of at most 64 characters
    (numbers are kept as their token text); anything else is dropped (None)."""
    return v if isinstance(v, str) and len(v) <= MAX_VERSION_CHARS else None


def _master_columns(view: BDView) -> tuple[list[Fraction], list[Fraction], list[Fraction]]:
    obj = list(view.f) + [sp.weight for sp in view.subproblems]
    lo = list(view.y_lo) + [sp.theta_lo for sp in view.subproblems]
    hi = list(view.y_hi) + [sp.theta_hi for sp in view.subproblems]
    return obj, lo, hi


def check_certificate(view: BDView, certificate: bytes | str | Path | Mapping[str, Any] | None, *,
                      limits: CheckerLimits | None = None, deadline: float | None = None,
                      solve_master: Callable[..., MasterSolve] | None = None,
                      clock: Callable[[], float] = time.monotonic,
                      on_stage: Callable[[str], None] | None = None) -> CertResult:
    """Certified lower bound of one certificate version (protocol 3.3). `certificate` is the published bytes (or a
    path to them, or an already parsed dict for in-process use). `deadline` is on `clock`; the check also stops at
    limits.max_seconds. `solve_master` replaces the Gurobi master solve (tests); `on_stage(name)` is called when the
    check enters "parse", "expand", "master" and "recert" (failure attribution in the supervised checker). A master
    solver error (license, unknown status, wrong multiplier count) is an infrastructure failure, not a certificate
    failure."""
    stage = on_stage or (lambda name: None)
    limits = limits or _DEFAULT_LIMITS
    t_start = clock()
    stop = t_start + limits.max_seconds
    if deadline is not None:
        stop = min(stop, deadline)
    trivial = view.trivial_bound()
    st = _Stats()
    timings: dict[str, float] = {}
    version = None

    def fail(reason: str, detail: str = "", infra: str | None = None) -> CertResult:
        out = _stats_dict(st, timings, clock() - t_start)
        if detail:
            out["detail"] = detail
        return CertResult(False, trivial, trivial, reason, version, out, infra)

    if certificate is None:
        return fail("absent")
    try:
        if isinstance(certificate, Path):
            try:
                if certificate.stat().st_size > limits.max_bytes:
                    raise _CertFailure("too_many_bytes")
                certificate = certificate.read_bytes()
            except OSError as exc:   # (b) the evaluator could not read the file: infrastructure, never the agent's
                return fail("infrastructure", str(exc)[:200], infra="certificate_unreadable")
        if isinstance(certificate, str):
            certificate = certificate.encode("utf-8")
        stage("parse")
        if isinstance(certificate, (bytes, bytearray)):
            obj = parse_certificate_bytes(bytes(certificate), limits)
            st_bytes = len(certificate)
        else:
            obj, st_bytes = certificate, None
        timings["parse_seconds"] = clock() - t_start
        if clock() > stop:
            raise _CertFailure("time_limit")
        if not isinstance(obj, dict) or type(obj) is _Dup:
            raise _CertFailure("malformed")
        version = _compact_version(obj.get("version"))
        if obj.get("version") is not None and version is None:
            timings["version_dropped"] = 1.0
        if obj.get("view") != view.view_id:
            raise _CertFailure("view_mismatch")
        cuts = obj.get("cuts", [])
        if not isinstance(cuts, list):
            raise _CertFailure("malformed")
        nnz = 0
        for cut in cuts:
            terms = cut.get("terms") if isinstance(cut, dict) else None
            if isinstance(terms, list):
                for term in terms:
                    pi = term.get("pi") if isinstance(term, dict) else None
                    if isinstance(pi, dict):
                        nnz += len(pi)
        st.nonzeros = nnz
        if nnz > limits.max_nonzeros:
            raise _CertFailure("too_many_nonzeros")
        stage("expand")
        t1 = clock()
        rows: list[_Row] = list(view.fs_rows)
        master_nnz = sum(len(r.cols) for r in rows)
        for cut in cuts:
            st.cuts += 1
            if clock() > stop:
                raise _CertFailure("time_limit")
            try:
                row = build_cut(view, cut, st, limits, stop, clock)
            except CutError as exc:   # (a) this cut is rejected (counted); the certificate continues
                st.reject(exc.reason)
                if st.expanded > limits.max_expanded:
                    raise _CertFailure("too_much_expansion") from None
                continue
            st.accepted += 1
            if st.expanded > limits.max_expanded:
                raise _CertFailure("too_much_expansion")
            if row is None:
                st.trivial += 1
                continue
            rows.append(row)
            master_nnz += len(row.cols)
            if master_nnz > limits.max_master_nonzeros:
                raise _CertFailure("too_many_master_nonzeros")
        timings["expand_seconds"] = clock() - t1
        if st.infeasible_cut:
            raise _CertFailure("master_infeasible", "a cut without variables has a positive right-hand side")
        obj_q, lo_q, hi_q = _master_columns(view)
        stage("master")
        t2 = clock()
        remaining = stop - t2
        if remaining <= 0:
            raise _CertFailure("time_limit")
        ms = (solve_master or gurobi_master)([float(v) for v in lo_q], [float(v) for v in hi_q], [float(v) for v in obj_q],
                                             rows, remaining, limits)
        timings["master_seconds"] = clock() - t2
        st_master = {"master_status": ms.status, "master_objective": ms.objective, "master_rows": len(rows),
                     "master_nonzeros": master_nnz, "master_detail": ms.detail}
        # The returned status is classified BEFORE the elapsed-time fallback: a memory outcome becomes reason "memory"
        # (the caller's frozen-limit/baseline gate decides attribution) and a solver error is infrastructure, even when
        # the solve also overran the time limit.
        if ms.status == "memory_limit":
            res = fail("memory")
            res.stats.update(st_master)
            return res
        known = ("optimal", "infeasible", "time_limit", "numerical")
        if ms.status not in known or (ms.status == "optimal" and (ms.duals is None or len(ms.duals) != len(rows))):
            res = fail("infrastructure", infra="master_solver_error")
            res.stats.update(st_master)
            return res
        if clock() > stop:
            res = fail("time_limit")
            res.stats.update(st_master)
            return res
        if ms.status != "optimal":
            # certificate-driven outcomes are charged to the certificate
            res = fail({"infeasible": "master_infeasible", "time_limit": "time_limit", "numerical": "numerical"}[ms.status])
            res.stats.update(st_master)
            return res
        stage("recert")
        t3 = clock()
        lb = recertify(obj_q, lo_q, hi_q, rows, ms.duals, deadline=stop, clock=clock)
        timings["recert_seconds"] = clock() - t3
    except _CertFailure as exc:   # (a) whole-certificate rejection or an explicit frozen limit (reason given)
        return fail(exc.reason, exc.detail)
    except MemoryError:   # reason "memory": the caller's frozen-limit/baseline gate decides (default infrastructure)
        return fail("memory")
    if clock() > stop:  # a result finished after the limit is never accepted
        res = fail("time_limit")
        res.stats.update(st_master)
        return res
    out = _stats_dict(st, timings, clock() - t_start)
    out.update(st_master)
    out["bytes"] = st_bytes
    out["lb_recertified"] = str(lb)
    out["lb_recertified_float"] = safe_float(lb)
    return CertResult(True, max(lb, trivial), trivial, None, version, out)


def _stats_dict(st: _Stats, timings: Mapping[str, float], total: float) -> dict[str, Any]:
    return {"cuts": st.cuts, "accepted": st.accepted, "opt_cuts": st.opt_cuts, "feas_cuts": st.feas_cuts,
            "trivial_cuts": st.trivial, "rejected": dict(st.rejected), "terms": st.terms, "nonzeros": st.nonzeros,
            "expanded": st.expanded, "clipped_count": st.clipped_count, "clipped_mass": str(st.clipped_mass),
            "clipped_mass_float": safe_float(st.clipped_mass),
            **{k: round(v, 6) for k, v in timings.items()}, "checker_seconds": round(total, 6)}


RESOURCE_REASONS = {"time_limit", "checker_budget_exhausted", "certificate_memory_limit",
                    "too_many_bytes", "too_many_values", "too_many_nonzeros", "too_much_expansion", "too_many_master_nonzeros"}


def failure_category(cert_ok: bool, reason: str | None, infrastructure: str | None) -> str | None:
    """None (ok), "absent" (nothing published), "infrastructure" (evaluator side: rerun), "certificate_resource"
    (an explicit frozen time/size/memory limit exceeded while processing the certificate: charged to the agent) or
    "certificate" (the certificate's content was rejected: charged to the agent)."""
    if cert_ok:
        return None
    if infrastructure:
        return "infrastructure"
    if reason == "absent":
        return "absent"
    return "certificate_resource" if reason in RESOURCE_REASONS else "certificate"


def _iter_versions(view: BDView, checkpoints: Sequence[Mapping[str, Any]], store_dir: str | Path | None,
                   limits: CheckerLimits, budget_seconds: float, solve_master: Callable[..., MasterSolve] | None,
                   clock: Callable[[], float], on_stage: Callable[[str, int], None] | None = None,
                   memory_attributable: Callable[[], bool] | None = None, convert: Callable[[bytes], bytes] | None = None):
    """Yield (index, result) in checking order (final version first). The budget covers every phase: reading and
    hashing the stored file, parsing, expansion, the master LP and the re-certification. Every distinct stored file
    is read and checked against its receipt hash (and recorded size) before a cached result for that hash is reused.
    A MemoryError is charged to the certificate ("certificate_memory_limit") only if `memory_attributable()` says an
    explicit frozen memory limit applies and the trusted view fitted well within it; otherwise it is infrastructure.
    Every result carries "failure": {"stage", "checkpoint", "category"} (None when ok)."""
    import hashlib

    trivial = view.trivial_bound()
    order = list(range(len(checkpoints)))
    if order:
        order = [order[-1]] + order[:-1]
    cache: dict[str, CertResult] = {}
    file_sha: dict[str, str | None] = {}   # stored file -> sha256 of its bytes (None: unreadable)
    used = 0.0
    stage_now = ["start"]

    def stage(name: str, idx: int) -> None:
        stage_now[0] = name
        if on_stage is not None:
            on_stage(name, idx)

    def memory_result(stats: dict[str, Any] | None = None) -> CertResult:
        """Every memory outcome (MemoryError, Gurobi out-of-memory) passes this gate: charged to the certificate only
        with an explicit frozen memory limit that the trusted setup fitted well within, else infrastructure."""
        if memory_attributable is not None and memory_attributable():
            return CertResult(False, trivial, trivial, "certificate_memory_limit", stats=dict(stats or {}))
        return CertResult(False, trivial, trivial, "infrastructure", stats=dict(stats or {}), infrastructure="memory_error_unattributed")

    def entry(idx, cp, res, *, cached=False, infra=None, reason=None):
        infra = infra or res.infrastructure
        reason = reason or res.reason
        cat = failure_category(res.cert_ok, reason, infra)
        return {"checkpoint": cp.get("checkpoint"), "sha256": cp.get("sha256"), "receipt_t": cp.get("t"), "seq": cp.get("seq"),
                "cert_ok": res.cert_ok, "lb": res.lb, "lb_float": res.lb_float, "reason": reason,
                "version": res.version, "cached": cached, "infrastructure": infra, "stats": res.stats,
                "failure": None if cat is None else {"stage": stage_now[0], "checkpoint": idx, "category": cat},
                "checker_seconds_used": round(used, 6)}

    def fail(reason: str, infra: str | None = None) -> CertResult:
        return CertResult(False, trivial, trivial, reason, infrastructure=infra)

    for idx in order:
        cp = checkpoints[idx]
        sha, fname = cp.get("sha256"), cp.get("file")
        stage("lookup", idx)
        if not fname and not sha and cp.get("seq") is None:
            yield idx, entry(idx, cp, fail("absent"))                       # nothing published before this checkpoint
            continue
        if not fname or not sha:
            yield idx, entry(idx, cp, fail("infrastructure", "stored_version_missing"))
            continue
        if used >= budget_seconds:
            yield idx, entry(idx, cp, fail("checker_budget_exhausted"))
            continue
        t0 = clock()
        deadline = t0 + (budget_seconds - used)
        data = None
        infra = None
        path = Path(store_dir) / fname if store_dir is not None else Path(fname)
        if fname not in file_sha or (file_sha[fname] == sha and sha not in cache):
            stage("read", idx)
            try:
                size = path.stat().st_size
                if cp.get("bytes") is not None and size != cp["bytes"]:
                    file_sha[fname] = None
                    infra = "size_mismatch"
                elif size > limits.max_bytes:
                    file_sha[fname] = None
                    infra = "size_above_limit"
                else:
                    data = path.read_bytes()
                    stage("hash", idx)
                    file_sha[fname] = hashlib.sha256(data).hexdigest()
            except OSError:   # (b) stored version unreadable: infrastructure
                file_sha[fname] = None
                infra = "stored_version_unreadable"
            except MemoryError:   # the memory gate (default infrastructure)
                used += clock() - t0
                yield idx, entry(idx, cp, memory_result())
                continue
        if infra is None and file_sha.get(fname) != sha:
            infra = "hash_mismatch" if file_sha.get(fname) is not None else "stored_version_unreadable"
        if infra is not None:
            used += clock() - t0
            yield idx, entry(idx, cp, fail("infrastructure", infra))
            continue
        if sha in cache:
            used += clock() - t0
            yield idx, entry(idx, cp, cache[sha], cached=True)
            continue
        if clock() > deadline:
            used += clock() - t0
            yield idx, entry(idx, cp, fail("checker_budget_exhausted"))
            continue
        if convert is not None:   # inside the budget (and, supervised, inside the child's deadline and memory cap)
            stage("convert", idx)
            try:
                data = convert(data)
            except CertificateRejected as exc:   # (a) malformed input, rejected with its reason
                used += clock() - t0
                res = fail(exc.reason if exc.reason in RESOURCE_REASONS else f"rejected:{exc.reason}")
                cache[sha] = res
                yield idx, entry(idx, cp, res)
                continue
            except MemoryError:   # the memory gate (default infrastructure); other exceptions propagate (b)
                used += clock() - t0
                yield idx, entry(idx, cp, memory_result())
                continue
            if clock() > deadline:
                used += clock() - t0
                yield idx, entry(idx, cp, fail("checker_budget_exhausted"))
                continue
        res = check_certificate(view, data, limits=limits, deadline=deadline, solve_master=solve_master, clock=clock,
                                on_stage=lambda name, idx=idx: stage(name, idx))
        if not res.cert_ok and res.reason == "memory":
            res = memory_result(res.stats)
        used += clock() - t0
        if not res.cert_ok and res.reason == "time_limit" and used >= budget_seconds:
            yield idx, entry(idx, cp, res, reason="checker_budget_exhausted")
            continue
        if res.infrastructure is None:
            cache[sha] = res
        yield idx, entry(idx, cp, res)


def check_versions(view: BDView, checkpoints: Sequence[Mapping[str, Any]], *, store_dir: str | Path | None = None,
                   limits: CheckerLimits | None = None, budget_seconds: float = 600.0,
                   solve_master: Callable[..., MasterSolve] | None = None,
                   clock: Callable[[], float] = time.monotonic,
                   convert: Callable[[bytes], bytes] | None = None) -> list[dict[str, Any]]:
    """Check the certificate version current at each supervisor checkpoint, in process, under one cumulative
    cooperative budget per attempt (protocol 3.3.6). `checkpoints` are the supervisor's records {"checkpoint", "file",
    "sha256", "bytes", "t", "seq"} in schedule order (the last one is T_f, the final version). The final version is
    checked first so that the scored bound at T_f always gets the budget. A stored version that is missing, unreadable,
    of the wrong size or failing its receipt hash is an infrastructure error (field "infrastructure", trivial bound).
    Versions left when the budget is spent get the trivial bound with reason "checker_budget_exhausted". In process
    a MemoryError is never attributable (infrastructure). For the hard wall-clock cap and memory attribution use
    `check_versions_supervised`."""
    out: list[dict[str, Any] | None] = [None] * len(checkpoints)
    for idx, d in _iter_versions(view, checkpoints, store_dir, limits or _DEFAULT_LIMITS, budget_seconds, solve_master, clock,
                                 convert=convert):
        out[idx] = d
    return [d for d in out if d is not None]


IPC_MAX_FRAME = 1 << 20      # bytes of one child -> parent message (results carry counts and short strings only)
_CERT_STAGES = ("read", "hash", "convert", "parse", "expand", "master", "recert", "lookup")


def _vm_bytes() -> int | None:
    """Current virtual memory size of this process (what RLIMIT_AS limits), from /proc/self/status (Linux)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmSize:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):   # unmeasurable: None -> view_fits False -> memory never attributable
        pass
    return None


def _ipc_entry(d: dict[str, Any]) -> dict[str, Any]:
    out = dict(d)
    out["lb"] = str(d["lb"])
    return out


def _child_main(wfd: int, view: BDView, checkpoints: list, store_dir, limits: CheckerLimits, budget_seconds: float,
                solve_master, memory_bytes: int | None, fit_fraction: float, init_solver: bool, convert=None) -> None:
    """Forked checker: framed JSON messages (4-byte big-endian length + UTF-8 JSON, at most IPC_MAX_FRAME bytes).
    Setup (memory cap, trusted imports and Gurobi licensing, baseline measurement) completes BEFORE "setup" is sent."""
    import os

    state = {"stage": "setup", "idx": None}
    fits = False

    def send(msg: dict[str, Any]) -> None:
        data = json.dumps(msg, default=str).encode("utf-8")
        if len(data) > IPC_MAX_FRAME:
            data = json.dumps({"type": "error", "stage": state["stage"], "idx": state["idx"],
                               "error": f"message of {len(data)} bytes exceeds the IPC frame limit"}).encode("utf-8")
        view_ = memoryview(len(data).to_bytes(4, "big") + data)
        while view_:
            n = os.write(wfd, view_)
            view_ = view_[n:]

    try:
        if memory_bytes:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        if init_solver:
            trusted_init(limits)
        baseline = _vm_bytes()      # after the trusted imports and licensing, before any certificate is read
        fits = memory_bytes is not None and baseline is not None and baseline <= fit_fraction * memory_bytes
        send({"type": "setup", "baseline_vm_bytes": baseline, "memory_bytes": memory_bytes, "view_fits": fits,
              "solver_initialized": bool(init_solver)})

        def on_stage(name: str, idx: int) -> None:
            state["stage"], state["idx"] = name, idx
            send({"type": "stage", "stage": name, "idx": idx})

        for idx, d in _iter_versions(view, checkpoints, store_dir, limits, budget_seconds, solve_master, time.monotonic,
                                     on_stage=on_stage, memory_attributable=lambda: fits, convert=convert):
            send({"type": "result", "idx": idx, "entry": _ipc_entry(d)})
        send({"type": "done"})
    except BaseException as exc:  # noqa: BLE001 - reported with its stage; the parent classifies it
        try:
            send({"type": "error", "stage": state["stage"], "idx": state["idx"], "error": f"{type(exc).__name__}: {exc}"[:300],
                  "memory_error": isinstance(exc, MemoryError), "view_fits": fits})
        except BaseException:  # noqa: BLE001
            pass
    finally:
        try:
            os.close(wfd)
        finally:
            os._exit(0)


def _unchecked(cp: Mapping[str, Any], idx: int, trivial: Fraction, reason: str, infra: str | None, stage: str,
               extra: dict[str, Any]) -> dict[str, Any]:
    return {"checkpoint": cp.get("checkpoint"), "sha256": cp.get("sha256"), "receipt_t": cp.get("t"), "seq": cp.get("seq"),
            "cert_ok": False, "lb": trivial, "lb_float": float_down(trivial), "reason": reason, "version": None,
            "cached": False, "infrastructure": infra, "stats": extra,
            "failure": {"stage": stage, "checkpoint": idx, "category": failure_category(False, reason, infra)}}


def check_versions_supervised(view: BDView, checkpoints: Sequence[Mapping[str, Any]], *, store_dir: str | Path | None = None,
                              limits: CheckerLimits | None = None, budget_seconds: float = 600.0,
                              solve_master: Callable[..., MasterSolve] | None = None,
                              memory_bytes: int | None = None, fit_fraction: float = 0.5,
                              init_solver: bool = True, convert: Callable[[bytes], bytes] | None = None) -> list[dict[str, Any]]:
    """`check_versions` in a forked child under ONE hard wall-clock deadline (budget_seconds from the fork) covering
    every phase of every checkpoint: reading, hashing, an optional `convert` hook (e.g. a family format -> the generic
    format; CertificateRejected = malformed input charged to the agent, MemoryError = memory gate, anything else =
    infrastructure), parsing, expansion, master LP and re-certification.

    Setup is trusted initialization and completes before the child acknowledges it: the address-space cap, the
    NumPy/SciPy/gurobipy imports and a started Gurobi environment (licensing; reused by every master solve), then the
    baseline memory measurement. IPC is a raw pipe watched with poll() against the deadline; messages are
    length-framed and bounded (IPC_MAX_FRAME), and a result is accepted only if its frame completes before the
    deadline. At the deadline the child is killed; kill, reap and close always run (try/finally). Unfinished
    checkpoints are classified by the last stage the child reported:
      - the frozen time budget ran out after setup was acknowledged -> "checker_budget_exhausted",
        category "certificate_resource" (charged to the agent);
      - a memory outcome (MemoryError, Gurobi out-of-memory) while processing a certificate passes the gate only with
        the explicit frozen limit `memory_bytes` (RLIMIT_AS) and a baseline at most `fit_fraction` of it ->
        "certificate_memory_limit" (charged to the agent); otherwise infrastructure;
      - anything else (fork/pipe failure, setup timeout or failure, including licensing and OOM during
        initialization, an unexpected exception, a dead child, a malformed or oversized frame, a parent-side IPC
        error) -> infrastructure.
    Every entry carries "failure": {"stage", "checkpoint", "category"} (None when ok). Linux (fork) only."""
    import os
    import select
    import signal

    limits = limits or _DEFAULT_LIMITS
    trivial = view.trivial_bound()
    checkpoints = list(checkpoints)
    t0 = time.monotonic()
    deadline = t0 + budget_seconds

    def all_infra(infra: str, detail: str) -> list[dict[str, Any]]:
        return [_unchecked(cp, idx, trivial, "infrastructure", infra, "startup", {"detail": detail[:300]}) |
                {"checker_wall_seconds": round(time.monotonic() - t0, 6), "checker_process": "startup_error", "checker_setup": None}
                for idx, cp in enumerate(checkpoints)]

    try:
        rfd, wfd = os.pipe()
    except OSError as exc:   # (b) infrastructure: structured results, no check
        return all_infra("checker_startup_error", f"pipe: {exc}")
    try:
        pid = os.fork()
    except OSError as exc:   # (b) infrastructure: structured results, no check
        os.close(rfd)
        os.close(wfd)
        return all_infra("checker_startup_error", f"fork: {exc}")
    if pid == 0:  # child
        os.close(rfd)
        _child_main(wfd, view, checkpoints, store_dir, limits, budget_seconds, solve_master, memory_bytes, fit_fraction,
                    init_solver, convert)
    results: dict[int, dict[str, Any]] = {}
    setup: dict[str, Any] | None = None
    last: dict[str, Any] = {"stage": "setup", "idx": None}
    status, detail, err_msg = "running", "", None
    exit_info = "unknown"
    try:
        os.close(wfd)
        os.set_blocking(rfd, False)
        poller = select.poll()
        poller.register(rfd, select.POLLIN | select.POLLHUP | select.POLLERR)
        buf = bytearray()
        while status == "running":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "killed"
                break
            if not poller.poll(max(1, min(int(remaining * 1000) + 1, 250))):
                continue
            try:
                chunk = os.read(rfd, 1 << 16)
            except BlockingIOError:   # benign: no data yet (other OSErrors reach the IPC handler: infrastructure)
                continue
            if not chunk:
                status = "eof"
                break
            buf += chunk
            while len(buf) >= 4 and status == "running":
                n = int.from_bytes(buf[:4], "big")
                if n > IPC_MAX_FRAME:
                    status, detail = "ipc_error", f"frame of {n} bytes"
                    break
                if len(buf) < 4 + n:
                    break
                frame = bytes(buf[4:4 + n])
                del buf[:4 + n]
                if time.monotonic() > deadline:   # a frame completed after the deadline is never accepted
                    status = "killed"
                    break
                try:
                    msg = json.loads(frame)
                    kind = msg["type"]
                except (ValueError, KeyError, TypeError):   # (b) infrastructure
                    status, detail = "ipc_error", "malformed frame"
                    break
                if kind == "setup":
                    setup = msg
                elif kind == "stage":
                    last = {"stage": msg.get("stage"), "idx": msg.get("idx")}
                elif kind == "result":
                    idx = msg.get("idx")
                    e = msg.get("entry")
                    if not isinstance(idx, int) or not 0 <= idx < len(checkpoints) or not isinstance(e, dict):
                        status, detail = "ipc_error", "malformed result"
                        break
                    try:
                        e["lb"] = Fraction(e["lb"])
                    except (KeyError, TypeError, ValueError, ZeroDivisionError):   # (b) infrastructure
                        status, detail = "ipc_error", "malformed result bound"
                        break
                    results[idx] = e
                elif kind == "done":
                    status = "done"
                elif kind == "error":
                    status, err_msg = "error", msg
                    last = {"stage": msg.get("stage"), "idx": msg.get("idx")}
                else:
                    status, detail = "ipc_error", f"unknown message {kind!r}"
    except Exception as exc:  # noqa: BLE001 - parent-side IPC failure: infrastructure, never a lost deadline
        status, detail = "ipc_error", f"{type(exc).__name__}: {exc}"[:300]
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError) as exc:   # the child already exited (reaped next); noted in the detail
            detail = detail or f"kill: {exc}"
        try:
            _, wstatus = os.waitpid(pid, 0)
            exit_info = f"signal {os.WTERMSIG(wstatus)}" if os.WIFSIGNALED(wstatus) else f"exit {os.WEXITSTATUS(wstatus)}"
        except (ChildProcessError, OSError):   # exit status unknown: kept as "unknown" (classification uses the IPC state)
            pass
        try:
            os.close(rfd)
        except OSError:   # benign: the parent's read end is released either way
            pass
    elapsed = time.monotonic() - t0
    if status == "done" and len(results) < len(checkpoints):
        status, detail = "ipc_error", "child finished without every result"

    in_cert_stage = setup is not None and last.get("stage") in _CERT_STAGES
    if status == "killed" and setup is not None:
        reason, infra = "checker_budget_exhausted", None
    elif status == "error" and err_msg is not None and err_msg.get("memory_error") and err_msg.get("view_fits") and in_cert_stage:
        reason, infra = "certificate_memory_limit", None
    else:
        reason = "infrastructure"
        infra = {"killed": "checker_setup_timeout", "error": "checker_setup_error" if setup is None else "checker_error",
                 "eof": "checker_died", "ipc_error": "checker_ipc_error"}.get(status, "checker_failed")
    out = []
    for idx, cp in enumerate(checkpoints):
        if idx in results:
            d = results[idx]
        else:
            stage = last.get("stage") if last.get("idx") == idx or setup is None else "pending"
            d = _unchecked(cp, idx, trivial, reason, infra, stage or "setup",
                           {"child": exit_info, "child_error": (err_msg or {}).get("error"), "ipc_detail": detail})
        d["checker_wall_seconds"] = round(elapsed, 6)
        d["checker_process"] = status
        d["checker_setup"] = setup
        out.append(d)
    return out


# ---------------------------------------------------------------- deterministic equivalent and LP* interval

def _de_layout(view: BDView) -> list[int]:
    offs, k = [], view.n
    for sp in view.subproblems:
        offs.append(k)
        k += sp.nvars
    return offs + [k]


def de_dual_bound(view: BDView, duals: Sequence[float]) -> Fraction:
    """Exact lower bound on LP* from multipliers of the deterministic equivalent's rows (first-stage rows, then each
    subproblem's rows in view order), re-certified over the finite boxes with the view's rational data."""
    senses = [r.sense for r in view.fs_rows]
    for sp in view.subproblems:
        senses.extend(sp.rows.sense)
    if len(duals) != len(senses):
        raise ValueError(f"{len(duals)} multipliers for {len(senses)} rows")
    lam, e = snap_duals(duals, senses)
    dens = {r.den for r, l in zip(view.fs_rows, lam) if l}
    pos = len(view.fs_rows)
    for sp in view.subproblems:
        nr = len(sp.rows)
        if any(lam[pos:pos + nr]):
            dens.add(sp.rows.den)
        pos += nr
    lc = _lcm(dens)
    g = (1 << e) * lc
    acc_y = [0] * view.n
    rs = 0
    for r, l in zip(view.fs_rows, lam):
        if l:
            fct = l * (lc // r.den)
            rs += fct * r.rhs
            for c, v in zip(r.cols, r.nums):
                acc_y[c] += fct * v
    bound = Fraction(0)
    pos = len(view.fs_rows)
    for sp in view.subproblems:
        rb = sp.rows
        nr = len(rb)
        lam_s = lam[pos:pos + nr]
        pos += nr
        acc_x: dict[int, int] = {}
        xget = acc_x.get
        rmul = lc // rb.den
        for i, l in enumerate(lam_s):
            if not l:
                continue
            fct = l * rmul
            if rb.H[i]:
                rs += fct * rb.H[i]
            cols, vals = rb.A[i]
            for j, a in zip(cols, vals):
                acc_x[j] = xget(j, 0) + fct * a
            tc, tv = rb.T[i]
            for k, t in zip(tc, tv):
                acc_y[k] += fct * t
        # x columns: d_j = p c_j - acc_j / g;  d_j * (g dv pd) = pn C_j g - acc_j dv pd
        pn, pd = sp.pw
        dv = sp.den
        base = pn * g
        s = sp.K0 * base
        C, L, U, cbox = sp.C, sp.L, sp.U, sp.Cbox
        mul = dv * pd
        for j, a in acc_x.items():
            delta = C[j] * base - a * mul
            s += delta * (L[j] if delta >= 0 else U[j]) - cbox[j] * base
        bound += Fraction(s, g * dv * dv * pd)
    bound += Fraction(rs, g)
    for k in range(view.n):
        d = view.f[k] - Fraction(acc_y[k], g) if acc_y[k] else view.f[k]
        bound += d * view.y_lo[k] if d >= 0 else d * view.y_hi[k]
    return bound


def de_primal_value(view: BDView, y: Sequence[Fraction], xs: Sequence[Sequence[Fraction]]) -> tuple[Fraction | None, str]:
    """Exact objective f.y + sum_s p_s c_s.x_s of a primal point if it satisfies every box and row exactly, else
    (None, reason)."""
    n = view.n
    if len(y) != n or len(xs) != view.S:
        return None, "shape"
    for k in range(n):
        if not view.y_lo[k] <= y[k] <= view.y_hi[k]:
            return None, f"y box {view.y_ids[k]}"
    for r in view.fs_rows:
        lhs = sum((Fraction(v) * y[c] for c, v in zip(r.cols, r.nums)), Fraction(0))
        rhs = Fraction(r.rhs)
        if (r.sense == GE and lhs < rhs) or (r.sense == LE and lhs > rhs) or (r.sense == EQ and lhs != rhs):
            return None, f"first-stage row {r.id}"
    obj = sum((fk * yk for fk, yk in zip(view.f, y)), Fraction(0))
    for sp, x in zip(view.subproblems, xs):
        if len(x) != sp.nvars:
            return None, f"subproblem {sp.id}: shape"
        dv = sp.den
        for j in range(sp.nvars):
            if not (Fraction(sp.L[j], dv) <= x[j] <= Fraction(sp.U[j], dv)):
                return None, f"subproblem {sp.id}: box {j}"
        rb = sp.rows
        for i in range(len(rb)):
            cols, vals = rb.A[i]
            tc, tv = rb.T[i]
            lhs = sum((a * x[j] for j, a in zip(cols, vals)), Fraction(0)) + sum((t * y[k] for k, t in zip(tc, tv)), Fraction(0))
            h = rb.H[i]
            s = rb.sense[i]
            if (s == GE and lhs < h) or (s == LE and lhs > h) or (s == EQ and lhs != h):
                return None, f"subproblem {sp.id}: row {rb.ids[i]}"
        cx = sum((Fraction(cj, dv) * xj for cj, xj in zip(sp.C, x) if cj and xj), Fraction(0))
        obj += sp.weight * cx
    return obj, "ok"


def _rationalize(values: Sequence[float], lo: Sequence[Fraction], hi: Sequence[Fraction], max_den: int | None) -> list[Fraction]:
    out = []
    for v, a, b in zip(values, lo, hi):
        q = Fraction(float(v))
        if max_den is not None:
            q = q.limit_denominator(max_den)
        out.append(min(max(q, a), b))
    return out


def repair_primal(view: BDView, y_float: Sequence[float], xs_float: Sequence[Sequence[float]],
                  repair: Callable | None = None) -> tuple[Fraction | None, str, dict[str, Any]]:
    """Best exactly verified primal objective from a floating-point primal: the family repair hook first, then the
    generic ladder (exact float values clipped to the boxes, then rationalized values with denominators <= 10^3,
    10^6, 10^9), each verified by de_primal_value. Returns (objective or None, method, notes)."""
    notes: dict[str, Any] = {}
    hook = repair or view.repair
    if hook is not None:
        try:
            cand = hook(view, y_float, xs_float)
        except Exception as exc:  # noqa: BLE001 - reference only: recorded in notes; the generic repair is verified
            # exactly, and without any verified primal LP_hi is None (the bound target is then undefined)
            cand = None
            notes["hook_error"] = f"{type(exc).__name__}: {exc}"[:200]
        if cand is not None:
            val, why = de_primal_value(view, *cand)
            notes["hook"] = why
            if val is not None:
                return val, "family", notes
    best: Fraction | None = None
    how = "failed"
    for max_den in (None, 10**3, 10**6, 10**9):
        y = _rationalize(y_float, view.y_lo, view.y_hi, max_den)
        xs = [_rationalize(x, [Fraction(v, sp.den) for v in sp.L], [Fraction(v, sp.den) for v in sp.U], max_den)
              for sp, x in zip(view.subproblems, xs_float)]
        val, why = de_primal_value(view, y, xs)
        notes[f"generic_{max_den or 'float'}"] = why
        if val is not None and (best is None or val < best):
            best, how = val, f"generic_{max_den or 'float'}"
            break
    return best, how, notes


@dataclass
class DESolve:
    status: str
    duals: list[float] | None = None
    y: list[float] | None = None
    xs: list[list[float]] | None = None
    objective: float | None = None
    seconds: float = 0.0
    detail: str = ""


def gurobi_de(view: BDView, *, time_limit: float = 3600.0, method: int = -1, threads: int = 1,
              mem_gb: float | None = None, crossover: int = -1, tol: float = 1e-9) -> DESolve:
    """Solve the deterministic equivalent (the compact LP relaxation over the view's rows) with Gurobi."""
    import gurobipy as gp
    from gurobipy import GRB
    import numpy as np
    import scipy.sparse as sps

    t0 = time.monotonic()
    offs = _de_layout(view)
    ncols = offs[-1]
    lb = [float(v) for v in view.y_lo]
    ub = [float(v) for v in view.y_hi]
    obj = [float(v) for v in view.f]
    for sp in view.subproblems:
        dv = sp.den
        w = sp.weight
        lb.extend(v / dv for v in sp.L)
        ub.extend(v / dv for v in sp.U)
        obj.extend(float(w * Fraction(c, dv)) if c else 0.0 for c in sp.C)
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    senses: list[str] = []
    rhs: list[float] = []
    for r in view.fs_rows:
        indices.extend(r.cols)
        data.extend(r.fvals)
        indptr.append(len(indices))
        senses.append(_GRB_SENSE[r.sense])
        rhs.append(r.frhs)
    for sp, off in zip(view.subproblems, offs):
        rb = sp.rows
        fh, fa, ft = rb.floats()
        for i in range(len(rb)):
            cols, _ = rb.A[i]
            tc, _ = rb.T[i]
            indices.extend(off + j for j in cols)
            data.extend(fa[i])
            indices.extend(tc)
            data.extend(ft[i])
            indptr.append(len(indices))
            senses.append(_GRB_SENSE[rb.sense[i]])
            rhs.append(fh[i])
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", threads)
    if mem_gb:
        env.setParam("MemLimit", mem_gb)
    env.start()
    model = gp.Model(env=env)
    try:
        model.Params.TimeLimit = max(0.01, time_limit)
        model.Params.Method = method
        model.Params.Crossover = crossover
        model.Params.FeasibilityTol = tol
        model.Params.OptimalityTol = tol
        z = model.addMVar(ncols, lb=np.asarray(lb), ub=np.asarray(ub), obj=np.asarray(obj))
        if rhs:
            A = sps.csr_matrix((np.asarray(data, dtype=float), np.asarray(indices, dtype=np.int64), np.asarray(indptr, dtype=np.int64)),
                               shape=(len(rhs), ncols))
            model.addMConstr(A, z, np.array(senses), np.asarray(rhs, dtype=float))
        model.optimize()
        st = model.Status
        if st != GRB.OPTIMAL:
            return DESolve({GRB.INFEASIBLE: "infeasible", GRB.INF_OR_UNBD: "infeasible", GRB.TIME_LIMIT: "time_limit",
                            GRB.MEM_LIMIT: "memory_limit"}.get(st, f"status_{st}"), seconds=time.monotonic() - t0)
        duals = model.getAttr("Pi", model.getConstrs()) if rhs else []
        xv = np.asarray(z.X, dtype=float)
        return DESolve("optimal", list(duals), xv[:view.n].tolist(), [xv[offs[s]:offs[s + 1]].tolist() for s in range(view.S)],
                       float(model.ObjVal), time.monotonic() - t0)
    finally:
        model.dispose()
        env.dispose()


def lp_interval(view: BDView, *, time_limit: float = 3600.0, method: int = -1, threads: int = 1, mem_gb: float | None = None,
                repair: Callable | None = None, solution: DESolve | None = None) -> dict[str, Any]:
    """Rigorous LP* interval [LP_lo, LP_hi] (protocol 3.4) from a floating-point solve of the deterministic
    equivalent: LP_lo = exact re-certified dual bound, LP_hi = exact objective of a verified primal (None if the
    repair fails: then the interval is not rigorous and `width` is the gap to the float objective)."""
    t0 = time.monotonic()
    sol = solution or gurobi_de(view, time_limit=time_limit, method=method, threads=threads, mem_gb=mem_gb)
    out: dict[str, Any] = {"view": view.view_id, "status": sol.status, "gurobi_objective": sol.objective, "solve_seconds": sol.seconds}
    if sol.status != "optimal":
        out.update(lp_lo=None, lp_hi=None, rigorous=False)
        return out
    t1 = time.monotonic()
    lo = max(de_dual_bound(view, sol.duals), view.trivial_bound())
    t2 = time.monotonic()
    hi, how, notes = repair_primal(view, sol.y, sol.xs, repair)
    t3 = time.monotonic()
    out.update(lp_lo=str(lo), lp_lo_float=float_down(lo), dual_seconds=t2 - t1, primal_seconds=t3 - t2, repair=how, repair_notes=notes)
    if hi is None:
        out.update(lp_hi=None, lp_hi_float=None, rigorous=False, width=float(Fraction(sol.objective) - lo) if sol.objective is not None else None)
    else:
        out.update(lp_hi=str(hi), lp_hi_float=float_up(hi), rigorous=lo <= hi, width=float(hi - lo),
                   relative_width=float((hi - lo) / max(abs(hi), Fraction(1))))
    out["total_seconds"] = time.monotonic() - t0
    return out


# ---------------------------------------------------------------- fractional-domain box check (acceptance suite)

def fractional_box_check(view: BDView, ys: Sequence[Sequence[float]] | None = None, *, samples: int = 4, seed: int = 0,
                         subproblems: Sequence[int] | None = None, tol: float = 1e-7, time_limit: float = 60.0) -> dict[str, Any]:
    """Check (in floating point, Gurobi) that the declared boxes do not change Q_s(y) at fractional first-stage
    points: for each sample y in X_LP and each selected subproblem, the LP with the boxes and the LP with the natural
    bounds (None = infinite) must agree in status and value. Samples: the given ys, else all-1/2, all-ones and random
    uniform points satisfying the first-stage rows."""
    import gurobipy as gp
    from gurobipy import GRB

    rng = random.Random(seed)
    pts = [list(map(float, y)) for y in ys] if ys is not None else []
    if ys is None:
        cand = [[0.5] * view.n, [1.0] * view.n] + [[rng.random() for _ in range(view.n)] for _ in range(samples)]
        for y in cand:
            ok = True
            for r in view.fs_rows:
                lhs = sum(v * y[c] for c, v in zip(r.cols, r.fvals))
                if (r.sense == GE and lhs < r.frhs - 1e-9) or (r.sense == LE and lhs > r.frhs + 1e-9) or (r.sense == EQ and abs(lhs - r.frhs) > 1e-9):
                    ok = False
            if ok:
                pts.append(y)
    sel = list(subproblems) if subproblems is not None else list(range(view.S))
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()
    mismatches = []
    checked = 0

    def solve(sp: Subproblem, y: list[float], natural: bool) -> tuple[str, float | None]:
        rb = sp.rows
        fh, fa, ft = rb.floats()
        m = gp.Model(env=env)
        m.Params.TimeLimit = time_limit
        m.Params.FeasibilityTol = 1e-9
        m.Params.OptimalityTol = 1e-9
        lo = [v / sp.den for v in sp.L]
        hi = [v / sp.den for v in sp.U]
        if natural:
            nl, nu = sp.natural_lb, sp.natural_ub
            lo = [-GRB.INFINITY if (nl is not None and nl[j] is None) else (float(nl[j]) if nl is not None else lo[j]) for j in range(sp.nvars)]
            hi = [GRB.INFINITY if (nu is not None and nu[j] is None) else (float(nu[j]) if nu is not None else hi[j]) for j in range(sp.nvars)]
        x = m.addVars(sp.nvars, lb=lo, ub=hi, obj=[c / sp.den for c in sp.C])
        for i in range(len(rb)):
            cols, _ = rb.A[i]
            tc, _ = rb.T[i]
            expr = gp.quicksum(a * x[j] for j, a in zip(cols, fa[i]))
            r = fh[i] - sum(t * y[k] for k, t in zip(tc, ft[i]))
            m.addLConstr(expr, _GRB_SENSE[rb.sense[i]], r)
        m.Params.InfUnbdInfo = 1
        m.Params.DualReductions = 0
        m.optimize()
        st = m.Status
        val = float(m.ObjVal) if st == GRB.OPTIMAL else None
        name = {GRB.OPTIMAL: "optimal", GRB.INFEASIBLE: "infeasible", GRB.UNBOUNDED: "unbounded",
                GRB.INF_OR_UNBD: "inf_or_unbd"}.get(st, f"status_{st}")
        m.dispose()
        return name, val

    try:
        for p, y in enumerate(pts):
            for s in sel:
                sp = view.subproblems[s]
                a = solve(sp, y, False)
                b = solve(sp, y, True)
                checked += 1
                same = a[0] == b[0] and (a[1] is None or abs(a[1] - b[1]) <= tol * max(1.0, abs(b[1])))
                theta_ok = a[1] is None or float(sp.theta_lo) - tol <= a[1] <= float(sp.theta_hi) + tol
                if not same or not theta_ok:
                    mismatches.append({"sample": p, "subproblem": sp.id, "boxed": a, "natural": b, "theta_box_ok": theta_ok})
    finally:
        env.dispose()
    return {"ok": not mismatches, "samples": len(pts), "checked": checked, "mismatches": mismatches[:20]}


# ---------------------------------------------------------------- family views (frozen, protocol 3.1)

FAMILY_VIEWS: dict[str, dict[str, Any]] = {
    "ufl": {
        "view": "ufl-v1",
        "first_stage": "y_i in [0, 1] (facility i open, ids 'y<i>', 0-based), cost f_i; one row 'cover': sum_i y_i >= 1",
        "subproblems": "one per customer j, id 'c<j>' (0-based, data order), weight 1",
        "subproblem": "Q_j(y) = min sum_i D_ij x_ij  s.t.  row 'd': sum_i x_ij = 1 (multiplier free);  "
                      "row 'l<i>': x_ij <= y_i (multiplier <= 0);  0 <= x_ij <= 1 (box, implied by 'd' and x >= 0)",
        "theta_box": "[sum_i min(0, D_ij), sum_i max(0, D_ij)] (default formula)",
        "compact_lp": "identical to the strong UFL LP (x_ij <= y_i, sum_i x_ij = 1); 'cover' is implied there",
    },
    "fctp": {
        "view": "fctp-v1",
        "first_stage": "y_ij in [0, 1] (arc i->j open, ids 'y<i>_<j>'), cost f_ij; no first-stage rows",
        "subproblems": "one coupled transportation LP, id 't', weight 1 (no finer split)",
        "subproblem": "Q(y) = min sum c_ij x_ij  s.t.  row 's<i>': sum_j x_ij = s_i (free);  row 'd<j>': sum_i x_ij = d_j "
                      "(free);  row 'l<i>_<j>': x_ij <= m_ij y_ij (multiplier <= 0), m_ij = min(s_i, d_j);  "
                      "0 <= x_ij <= m_ij (box, implied by the equalities and x >= 0)",
        "theta_box": "[0, sum c_ij m_ij] (default formula)",
        "feasibility": "not complete: designs that cannot carry the demand need 'feas' cuts (pi = -FarkasDual of the "
                       "zero-objective feasibility LP over the same rows)",
        "compact_lp": "identical to src/bd/fctp/b_mip_xl.build_fctp_mip relaxed (no coverage or cut-set rows on either side)",
    },
}


def family_view_json(family: str) -> dict[str, Any]:
    """Machine-readable family view description (bd_view.json of the task prompt) with the certificate format."""
    spec = dict(FAMILY_VIEWS[family])
    spec["certificate"] = {
        "format": '{"view": "<view id>", "version": <int>, "cuts": [{"kind": "opt" | "feas", '
                  '"terms": [{"s": "<subproblem id>", "pi": {"<row id>": "<decimal or number>", ...}}, ...]}, ...]}',
        "rules": ["terms of a cut have distinct subproblem ids; weights come from the view",
                  "multipliers are projected to the row's dual sign (>=: pi >= 0, <=: pi <= 0, =: free)",
                  "missing rows have multiplier 0; non-finite or unparsable values reject the cut",
                  "feas cuts: pi = -FarkasDual of the zero-objective feasibility LP over the original rows"],
    }
    return spec


def _read_numbers_text(path: str | Path) -> list[Fraction]:
    return [exact(tok) for tok in Path(path).read_text().split()]


def ufl_view(data: str | Path | Mapping[str, Any]) -> BDView:
    """UFL view from data.json (keys m, n, f, D; numbers parsed exactly from their decimal text) or a UflLib simple
    format file (`FILE:` line, `m n 0`, then `index f_i c_i1 .. c_in` per facility)."""
    if isinstance(data, Mapping):
        obj = data
    else:
        text = Path(data).read_text()
        if text.lstrip().startswith("{"):
            obj = exact_json(text)
        else:
            rows = [line.split() for line in text.splitlines() if line.strip() and not line.startswith("FILE")]
            m, n = int(rows[0][0]), int(rows[0][1])
            fac = rows[1:1 + m]
            obj = {"m": m, "n": n, "f": [exact(r[1]) for r in fac], "D": [[exact(v) for v in r[2:2 + n]] for r in fac]}
    f = [exact(v) for v in obj["f"]]
    D = [[exact(v) for v in row] for row in obj["D"]]
    m, n = len(f), len(D[0]) if D else 0
    if len(D) != m or any(len(r) != n for r in D):
        raise ValueError("UFL data: D must be m x n")
    block = RowBlock(["d"] + [f"l{i}" for i in range(m)], [EQ] + [LE] * m, [1] + [0] * m,
                     [{i: 1 for i in range(m)}] + [{i: 1} for i in range(m)], [{}] + [{i: -1} for i in range(m)])
    zeros, ones = [0] * m, [1] * m
    nat_ub = [None] * m
    subs = [Subproblem(f"c{j}", 1, block, [D[i][j] for i in range(m)], zeros, ones, natural_ub=nat_ub) for j in range(n)]
    return BDView("ufl-v1", [f"y{i}" for i in range(m)], f, subs,
                  [{"id": "cover", "sense": ">=", "rhs": 1, "coefs": {i: 1 for i in range(m)}}],
                  repair=ufl_repair, meta={"family": "ufl", "m": m, "n": n})


def ufl_repair(view: BDView, y_float: Sequence[float], xs_float: Sequence[Sequence[float]]) -> tuple[list[Fraction], list[list[Fraction]]]:
    """Exactly feasible UFL point near a float LP point: y rationalized (denominators <= 10^6; kept if not worse) and
    clipped to [0, 1], topped up to sum y >= 1, then each customer's exact greedy optimum for that y (fill the
    cheapest facilities up to y-mass 1)."""
    m = view.n
    best = None
    for max_den in (10**6, None):
        y = _rationalize(y_float, view.y_lo, view.y_hi, max_den)
        deficit = 1 - sum(y, Fraction(0))
        if deficit > 0:
            for i in sorted(range(m), key=lambda i: -y[i]):
                add = min(deficit, 1 - y[i])
                y[i] += add
                deficit -= add
                if deficit <= 0:
                    break
        if deficit > 0:
            continue
        xs = []
        val = sum((fi * yi for fi, yi in zip(view.f, y) if yi), Fraction(0))
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
                    val += sp.weight * Fraction(sp.C[i], sp.den) * take
            xs.append(x)
        if best is None or val < best[0]:
            best = (val, y, xs)
    if best is None:
        raise ValueError("UFL repair failed: sum y < 1")
    return best[1], best[2]


def fctp_view(data: str | Path) -> BDView:
    """FCTP view from a Klose .FCTP text file (m n, supplies, demands, unit costs, fixed costs; arc order i * n + j),
    numbers parsed exactly."""
    tok = Path(data).read_text().split()
    m, n = int(tok[0]), int(tok[1])
    vals = [exact(t) for t in tok[2:]]
    if len(vals) != m + n + 2 * m * n:
        raise ValueError("FCTP data: token count does not match the header")
    s, d = vals[:m], vals[m:m + n]
    c = vals[m + n:m + n + m * n]
    fx = vals[m + n + m * n:]
    cap = [min(s[i], d[j]) for i in range(m) for j in range(n)]
    ids = [f"s{i}" for i in range(m)] + [f"d{j}" for j in range(n)] + [f"l{i}_{j}" for i in range(m) for j in range(n)]
    sense = [EQ] * (m + n) + [LE] * (m * n)
    rhs = list(s) + list(d) + [0] * (m * n)
    a = [{i * n + j: 1 for j in range(n)} for i in range(m)] + [{i * n + j: 1 for i in range(m)} for j in range(n)] \
        + [{k: 1} for k in range(m * n)]
    t = [{}] * (m + n) + [{k: -cap[k]} for k in range(m * n)]
    block = RowBlock(ids, sense, rhs, a, t)
    sub = Subproblem("t", 1, block, c, [0] * (m * n), cap, var_ids=[f"x{i}_{j}" for i in range(m) for j in range(n)],
                     natural_ub=[None] * (m * n))
    return BDView("fctp-v1", [f"y{i}_{j}" for i in range(m) for j in range(n)], fx, [sub], [],
                  repair=fctp_repair, meta={"family": "fctp", "m": m, "n": n, "supply": [str(v) for v in s], "demand": [str(v) for v in d]})


def fctp_repair(view: BDView, y_float: Sequence[float], xs_float: Sequence[Sequence[float]]) -> tuple[list[Fraction], list[list[Fraction]]]:
    """Exactly feasible FCTP point: the flows rounded (integers for integral data: the relaxation's optimal flows are
    transportation vertices with capacities m_ij; otherwise rationalized), checked by the caller, and y_ij = x_ij / m_ij
    (the cheapest linking y for those flows)."""
    sp = view.subproblems[0]
    dv = sp.den
    caps = [Fraction(u, dv) for u in sp.U]
    integral = all(c.denominator == 1 for c in caps) and all(Fraction(h, sp.rows.den).denominator == 1 for h in sp.rows.H)
    x = []
    for v, cap in zip(xs_float[0], caps):
        q = Fraction(round(v)) if integral else Fraction(float(v)).limit_denominator(10**6)
        x.append(min(max(q, Fraction(0)), cap))
    y = [xj / cap if cap > 0 else Fraction(0) for xj, cap in zip(x, caps)]
    return y, [x]


VIEW_BUILDERS: dict[str, Callable[[Any], BDView]] = {"ufl": ufl_view, "fctp": fctp_view}


def build_view(family: str, data: str | Path) -> BDView:
    return VIEW_BUILDERS[family](data)
