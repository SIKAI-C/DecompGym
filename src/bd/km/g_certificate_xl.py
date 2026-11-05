"""k-median (scale-up): optimized family backend of the exact LP-master certificate checker.

Canonical evaluator (protocol v4, Section 3; Codex review 46 #3): `src/experiments/xl_certificate.py` with the
k-median view `h_view_xl.km_view` ("km-v1"). This module is the optimized backend for the COMPRESSED family format
km-cert-v1 (implied cheapest completions), used for development, the controls and the references; it is cross-checked
against the canonical evaluator (h_view_xl.cross_check) and shares its view, limits and failure semantics.

View "km-v1" (classical y-space): min sum_j theta_j s.t. theta_j >= Q_j(y), sum_i y_i = k, y in [0, 1]^m,
    Q_j(y) = min {sum_i D_ij x_ij : row 'd' sum_i x_ij = 1;  row 'l<i>' x_ij <= y_i;  0 <= x_ij <= 1},
one subproblem per customer (id 'c<j>', trusted weight 1), recourse complete for sum y = k with 1 <= k <= m. The
first-stage rows (sum y = k and the box) are exactly those of the compact LP (b_mip_xl), so every certified bound
is at most LP*. Theta boxes (default formula, valid for every fractional y in [0, 1]^m where Q_j is finite):
L_j = sum_i min(0, D_ij) = 0, U_j = sum_i max(0, D_ij).

km-cert-v1: {"format": "km-cert-v1", "view": "km-v1", "version": <any>, "cuts": [{"ids": [j, ...], "v": [v_j, ...],
"w": [[[i, w_ji], ...], ...] (optional)}]}. A term (j, v_j, w_j) is the dual vector (pi_d = v, pi_l<i> = -w_i) of
customer j's subproblem; without "w" the cheapest completion w_i = (v - D_ij)^+ is implied. A cut with several ids
is the sum of its terms over distinct ids (each subproblem weight applied once). Each term gives the
Neumaier-Shcherbina safe cut (w+ = max(w, 0), the sign projection of pi_l <= 0)
    theta_j >= v + sum_i min(0, D_ij - v + w+_i) - sum_i w+_i y_i,                                         (*)
valid for every finite (v, w) and every y in [0, 1]^m with sum y >= 1 (the correction is exactly 0 for the implied
completion).

Frozen rules (Codex review 46 #3):
- Parsing is bounded: <= max_bytes, a structural token pre-scan, number tokens of <= 64 characters with a decimal
  exponent within +-400, duplicate JSON keys and non-finite values reject; a value of magnitude > max_abs_coef (1e12)
  rejects its cut (the canonical 'magnitude' rule). Published values are rounded to the nearest float64 before use:
  any finite (v, w) gives a valid cut (*), so rounding changes which valid cut is certified, never validity.
- Size limit, counted on the EXPANDED (canonical) form: a term with implied completion counts 1 + #{i: D_ij < v}
  multiplier entries (pi_d and one pi_l<i> per positive w_i, exactly its size in the canonical format), an explicit
  term 1 + len(w). The total must be <= max_nonzeros (2e6); km-cert-v1 is only a transport compression.
- Whole-check deadline: min(start + max_seconds, deadline) covers parsing, expansion, the Gurobi master solve and the
  exact re-certification; a result finished after it is never accepted.
- Failure semantics: a parse failure, a view mismatch, a size or time limit, a numerical failure or a non-optimal or
  infeasible master gives cert_ok = False and LB = the trivial bound (0), never +inf. Rejected cuts are counted
  (`rejected`) and do not fail the certificate.
- Re-certification: the master LP's duals (lambda_c >= 0 after projection, mu free) are converted exactly, and
      LB = mu k + sum_c lambda_c b_c + sum_j min(r_j L_j, r_j U_j) + sum_i min(0, r_i),
      r_j = 1 - sum_{c: j in ids(c)} lambda_c,  r_i = -mu - sum_c lambda_c W_ci,
  with b_c, W_ci the exact constant and coefficients of cut c (*) from the instance's exact distances and the
  certificate's values; never coefficients read back from Gurobi.
- LP* interval (`lp_interval`): exact endpoints with provenance: lo = the largest certified LB among the supplied
  checker results (cert_ok only; its source recorded), hi = the smallest exactly evaluated master objective
  sum_j Q_j(y) over the supplied LP points, each repaired exactly to y in [0, 1]^m with sum y = k (source recorded);
  `rigorous` = (lo <= hi).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from src.bd.km.a_data_xl import KMInstance, integral_costs
from src.bd.scfl.b_mip_xl import make_env

FORMAT = "km-cert-v1"
VIEW_ID = "km-v1"


@dataclass
class Limits:
    """Per-check limits (protocol 3.2 initial values, shared with the canonical checker; frozen for Stage 1)."""
    max_bytes: int = 256 * 2**20
    max_nonzeros: int = 2_000_000          # expanded multiplier entries (see the module docstring)
    max_master_nonzeros: int = 20_000_000
    max_seconds: float = 600.0
    max_abs_coef: float = 1e12
    max_token_chars: int = 64
    max_exponent: int = 400
    gurobi_mem_gb: float | None = 8.0
    reject_tol: float | None = None        # optional rejection of explicit duals (None: Neumaier-Shcherbina only)


DEFAULT_LIMITS = Limits()


class CertFailure(Exception):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason, self.detail = reason, detail


# ---------------------------------------------------------------- certificate container

@dataclass
class Certificate:
    single_j: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))   # one-ID cuts, implied w
    single_v: np.ndarray = field(default_factory=lambda: np.zeros(0))
    multi: list = field(default_factory=list)       # [(ids (int array, unique), v (float array))], implied w
    explicit: list = field(default_factory=list)    # [(ids, v, [(facility idx int array, w float array) per id])]
    version: Any = None
    rejected: list = field(default_factory=list)    # parse-time rejections (reasons)

    def num_cuts(self) -> int:
        return len(self.single_j) + len(self.multi) + len(self.explicit)


def singles(j, v, version: Any = None) -> Certificate:
    return Certificate(single_j=np.asarray(j, dtype=np.int64), single_v=np.asarray(v, dtype=float), version=version)


def aggregated(V: list, n: int, version: Any = None) -> Certificate:
    ids = np.arange(n)
    return Certificate(multi=[(ids, np.asarray(v, dtype=float)) for v in V], version=version)


def one_shot(D: np.ndarray, design, version: Any = None) -> Certificate:
    """The trivial control: one cut per customer at a single design (v_j = distance to its nearest open facility)."""
    idx = np.asarray(sorted({int(i) for i in design}), dtype=np.int64)
    return singles(np.arange(D.shape[1]), D[idx].min(axis=0), version)


def to_json(cert: Certificate, version: Any = None) -> dict:
    cuts = [{"ids": [int(j)], "v": [float(v)]} for j, v in zip(cert.single_j, cert.single_v)]
    cuts += [{"ids": [int(j) for j in ids], "v": [float(x) for x in v]} for ids, v in cert.multi]
    cuts += [{"ids": [int(j) for j in ids], "v": [float(x) for x in v],
              "w": [[[int(i), float(w)] for i, w in zip(ix, wv)] for ix, wv in ws]} for ids, v, ws in cert.explicit]
    return {"format": FORMAT, "view": VIEW_ID, "version": cert.version if version is None else version, "cuts": cuts}


def to_bytes(cert: Certificate, version: Any = None) -> bytes:
    return json.dumps(to_json(cert, version), separators=(",", ":")).encode()


# ---------------------------------------------------------------- bounded parsing

class _Dup(dict):
    """A JSON object with duplicate keys (ambiguous: its cut is rejected)."""


def _pairs(pairs: list) -> dict:
    d = dict(pairs)
    return d if len(d) == len(pairs) else _Dup(d)


def number(value: Any, limits: Limits = DEFAULT_LIMITS) -> float:
    """A published number (token text, int or float) as the nearest float64; raises ValueError(reason) for
    malformed, non-finite, too long, out-of-range or too large values."""
    if isinstance(value, bool) or value is None:
        raise ValueError("bad_number")
    if isinstance(value, str):
        if len(value) > limits.max_token_chars:
            raise ValueError("token_too_long")
        try:
            d = Decimal(value)
        except (InvalidOperation, ValueError):
            raise ValueError("bad_number") from None
        if not d.is_finite():
            raise ValueError("non_finite")
        if d and abs(d.adjusted()) > limits.max_exponent:
            raise ValueError("out_of_range")
        q = Fraction(*d.as_integer_ratio())
    elif isinstance(value, int):
        if value.bit_length() > 4 * limits.max_exponent:
            raise ValueError("out_of_range")
        q = Fraction(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non_finite")
        q = Fraction(value)
    elif isinstance(value, Fraction):
        q = value
    else:
        raise ValueError("bad_number")
    if abs(q) > limits.max_abs_coef:
        raise ValueError("magnitude")
    return float(q)


def _index(value: Any, bound: int) -> int:
    if isinstance(value, bool):
        raise ValueError("bad_index")
    if isinstance(value, str):
        if not value.isdigit() or len(value) > 12:
            raise ValueError("bad_index")
        value = int(value)
    if not isinstance(value, int) or not 0 <= value < bound:
        raise ValueError("bad_index")
    return value


def parse_certificate_bytes(data: bytes, m: int, n: int, limits: Limits = DEFAULT_LIMITS) -> Certificate:
    """Bounded parse of a km-cert-v1 version (raises CertFailure for whole-certificate failures)."""
    if len(data) > limits.max_bytes:
        raise CertFailure("too_many_bytes")
    tokens = data.count(b",") + data.count(b":") + data.count(b"[") + data.count(b"{")
    if tokens > 8 * limits.max_nonzeros + 1000:
        raise CertFailure("too_many_values")
    try:
        obj = json.loads(data, parse_float=str, parse_int=str, parse_constant=str, object_pairs_hook=_pairs)
    except (ValueError, RecursionError, UnicodeDecodeError) as exc:
        raise CertFailure("parse_error", str(exc)[:200]) from None
    return parse_certificate_obj(obj, m, n, limits)


def parse_certificate_obj(obj: Any, m: int, n: int, limits: Limits = DEFAULT_LIMITS) -> Certificate:
    if not isinstance(obj, dict) or type(obj) is _Dup:
        raise CertFailure("malformed")
    if obj.get("format") != FORMAT:
        raise CertFailure("format_mismatch")
    if obj.get("view") not in (None, VIEW_ID):
        raise CertFailure("view_mismatch")
    cuts = obj.get("cuts")
    if not isinstance(cuts, list):
        raise CertFailure("malformed")
    sj, sv, multi, explicit, rejected = [], [], [], [], []
    for c, cut in enumerate(cuts):
        try:
            if not isinstance(cut, dict) or type(cut) is _Dup:
                raise ValueError("malformed")
            ids_raw, v_raw = cut.get("ids"), cut.get("v")
            if not isinstance(ids_raw, list) or not isinstance(v_raw, list) or not ids_raw or len(ids_raw) != len(v_raw):
                raise ValueError("malformed")
            ids = [_index(j, n) for j in ids_raw]
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate_id")
            v = [number(x, limits) for x in v_raw]
            w = cut.get("w")
            if w is None:
                if len(ids) == 1:
                    sj.append(ids[0])
                    sv.append(v[0])
                else:
                    multi.append((np.asarray(ids, dtype=np.int64), np.asarray(v, dtype=float)))
                continue
            if not isinstance(w, list) or len(w) != len(ids):
                raise ValueError("malformed_w")
            ws = []
            for entries in w:
                if not isinstance(entries, list):
                    raise ValueError("malformed_w")
                ix, wv = [], []
                for e in entries:
                    if not isinstance(e, list) or len(e) != 2:
                        raise ValueError("malformed_w")
                    ix.append(_index(e[0], m))
                    wv.append(number(e[1], limits))
                if len(set(ix)) != len(ix):
                    raise ValueError("duplicate_row")
                ws.append((np.asarray(ix, dtype=np.int64), np.asarray(wv, dtype=float)))
            explicit.append((np.asarray(ids, dtype=np.int64), np.asarray(v, dtype=float), ws))
        except ValueError as exc:
            rejected.append(f"cut {c}: {exc}")
    return Certificate(single_j=np.asarray(sj, dtype=np.int64), single_v=np.asarray(sv, dtype=float), multi=multi,
                       explicit=explicit, version=obj.get("version"), rejected=rejected)


def parse_certificate(obj: Any, m: int, n: int, limits: Limits = DEFAULT_LIMITS) -> tuple[Certificate, list[str]]:
    """Parse an in-memory JSON object (compatibility helper): (certificate, rejected reasons)."""
    try:
        cert = parse_certificate_obj(obj, m, n, limits)
    except CertFailure as exc:
        return Certificate(), [exc.reason]
    return cert, list(cert.rejected)


# ---------------------------------------------------------------- exact helpers

def _fr(x: float) -> Fraction:
    return Fraction(float(x))


def float_down(q: Fraction) -> float:
    try:
        f = q.numerator / q.denominator
    except OverflowError:
        return 1.7976931348623157e308 if q > 0 else -math.inf
    if Fraction(f) > q:
        f = math.nextafter(f, -math.inf)
    return f


def float_up(q: Fraction) -> float:
    try:
        f = q.numerator / q.denominator
    except OverflowError:
        return math.inf if q > 0 else -1.7976931348623157e308
    if Fraction(f) < q:
        f = math.nextafter(f, math.inf)
    return f


def _is_integer_float(x: float) -> bool:
    return math.isfinite(x) and float(x).is_integer() and abs(x) < 2.0 ** 52


# ---------------------------------------------------------------- checker

class Checker:
    def __init__(self, inst: KMInstance, D: np.ndarray | None = None) -> None:
        self.inst = inst
        self.D = inst.distances() if D is None else D
        self.m, self.n = self.D.shape
        self.k = inst.k
        DT = np.ascontiguousarray(self.D.T)
        self.order = np.argsort(DT, axis=1, kind="stable").astype(np.int32)
        self.ds = np.take_along_axis(DT, self.order, axis=1)
        self.integral = integral_costs(self.D)
        self.U_float = DT.sum(axis=1)
        self._U_exact: dict[int, Fraction] = {}

    def U_exact(self, j: int) -> Fraction:
        if j not in self._U_exact:
            col = self.D[:, j]
            if self.integral:
                self._U_exact[j] = Fraction(int(sum(int(x) for x in col)))
            else:
                self._U_exact[j] = sum((_fr(x) for x in col), Fraction(0))
        return self._U_exact[j]

    def dual_vector_certificate(self, v, version: Any = None) -> Certificate:
        """Certificate of a vector of assignment-row duals or Lagrangian multipliers v (one per customer): the cuts
        (j, v_j) and (j, min(v_j, delta_j)), delta_j = the (m - k + 1)-th smallest distance of customer j (a dual of
        the reduced compact LP may exceed delta_j, where its cheapest completion over all facilities is weaker;
        both cuts are valid for any v)."""
        v = np.asarray(v, dtype=float)
        delta = self.ds[:, self.m - self.k]
        return singles(np.r_[np.arange(self.n), np.arange(self.n)], np.r_[v, np.minimum(v, delta)], version)

    def support(self, j: int, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Facilities with D_ij < v (ascending distance) and their distances."""
        c = int(np.searchsorted(self.ds[j], v, side="left"))
        return self.order[j, :c], self.ds[j, :c]

    def counts(self, j: np.ndarray, v: np.ndarray) -> np.ndarray:
        """#{i: D_ij < v} for paired arrays (vectorized in chunks)."""
        j = np.asarray(j, dtype=np.int64)
        v = np.asarray(v, dtype=float)
        out = np.zeros(len(j), dtype=np.int64)
        for a in range(0, len(j), 2048):
            b = min(len(j), a + 2048)
            out[a:b] = (self.ds[j[a:b]] < v[a:b, None]).sum(axis=1)
        return out

    def expanded_size(self, cert: Certificate) -> int:
        """Size of the certificate in the canonical format (the frozen counting rule)."""
        size = int(len(cert.single_j) + self.counts(cert.single_j, cert.single_v).sum()) if len(cert.single_j) else 0
        for ids, v in cert.multi:
            size += int(len(ids) + self.counts(ids, v).sum())
        for ids, v, ws in cert.explicit:
            size += len(ids) + sum(len(ix) for ix, _ in ws)
        return size

    # ------------------------------------------------------------ explicit duals

    def _explicit_cut(self, j: int, v: float, ix: np.ndarray, w: np.ndarray, reject_tol: float | None):
        """(accepted, reason, float coefficient map, exact constant b, exact coefficient map) of one ID's dual."""
        col = self.D[:, j]
        wp = np.maximum(w, 0.0)
        infeas = max(0.0, float(-w.min())) if len(w) else 0.0
        listed = set(int(i) for i in ix)
        sup, _ = self.support(j, v)
        unl = [int(i) for i in sup if int(i) not in listed]
        if len(ix):
            infeas = max(infeas, float(np.max(v - col[ix] - wp)))
        if unl:
            infeas = max(infeas, float(np.max(v - col[unl])))
        if reject_tol is not None and infeas > reject_tol * max(1.0, abs(v)):
            return False, f"dual infeasibility {infeas:.3g}", None, None, None
        V = _fr(v)
        b = V
        exact: dict[int, Fraction] = {}
        for i, wi in zip(ix.tolist(), wp.tolist()):
            W = _fr(wi)
            corr = _fr(col[i]) - V + W
            if corr < 0:
                b += corr
            if W:
                exact[i] = W
        for i in unl:
            b += _fr(col[i]) - V
        fl = {i: float(x) for i, x in exact.items()}
        return True, "", fl, b, exact

    # ------------------------------------------------------------ certify

    def certify(self, cert: Certificate | bytes | dict, *, limits: Limits | None = None, deadline: float | None = None,
                reject_tol: Any = "default", time_limit: float | None = None, env: gp.Env | None = None, method: int = 1,
                lp_coef_scale: float | None = None, clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
        """Exact certified LP-master bound of one certificate version (see the module docstring). Returns a dict with
        cert_ok, lb (exact, as text), bound (float rounded down; the trivial bound 0 on failure), reason, version,
        rejected, lp_value (Gurobi's master objective), lp_status and timings. `lp_coef_scale` (tests only) multiplies
        the y-coefficients handed to Gurobi, to show that re-certification never trusts the LP model."""
        limits = limits or DEFAULT_LIMITS
        if time_limit is not None:
            limits = Limits(**{**limits.__dict__, "max_seconds": time_limit})
        reject_tol = limits.reject_tol if reject_tol == "default" else reject_tol
        t0 = clock()
        stop = t0 + limits.max_seconds
        if deadline is not None:
            stop = min(stop, deadline)
        out: dict[str, Any] = {"cert_ok": False, "lb": "0", "bound": 0.0, "trivial": "0", "reason": None, "version": None,
                               "view": VIEW_ID, "rejected": [], "lp_value": None, "lp_status": None, "cuts_used": 0,
                               "cuts_with_positive_dual": 0, "nonzeros_expanded": None}

        def fail(reason: str, detail: str = "") -> dict[str, Any]:
            out.update(cert_ok=False, lb="0", bound=0.0, reason=reason, checker_seconds=round(clock() - t0, 3))
            if detail:
                out["detail"] = detail
            return out

        def check_time() -> None:
            if clock() > stop:
                raise CertFailure("time_limit")

        try:
            if cert is None:
                return fail("absent")
            if isinstance(cert, (bytes, bytearray)):
                cert = parse_certificate_bytes(bytes(cert), self.m, self.n, limits)
            elif isinstance(cert, dict):
                cert = parse_certificate_obj(cert, self.m, self.n, limits)
            out["version"] = cert.version
            rejected = list(cert.rejected)
            check_time()
            res = self._certify(cert, limits, stop, reject_tol, env, method, lp_coef_scale, rejected, out, check_time, clock)
        except CertFailure as exc:
            return fail(exc.reason, exc.detail)
        except MemoryError:
            return fail("memory")
        if clock() > stop:
            return fail("time_limit")
        out["checker_seconds"] = round(clock() - t0, 3)
        return res

    def _certify(self, cert, limits, stop, reject_tol, env, method, lp_coef_scale, rejected, out, check_time, clock):
        m, n, k = self.m, self.n, self.k
        t0 = clock()
        # singles: validate, dedupe
        sj = np.asarray(cert.single_j, dtype=np.int64)
        sv = np.asarray(cert.single_v, dtype=float)
        ok = (sj >= 0) & (sj < n) & np.isfinite(sv) & (np.abs(sv) <= limits.max_abs_coef) if len(sj) else np.zeros(0, dtype=bool)
        if len(sj) and not ok.all():
            rejected.append(f"{int((~ok).sum())} single cuts with invalid ID or value")
        sj, sv = sj[ok], sv[ok]
        if len(sj):
            key = np.unique(np.stack([sj.astype(float), sv], axis=1), axis=0)
            sj, sv = key[:, 0].astype(np.int64), key[:, 1]
        multi = []
        for c, (ids, v) in enumerate(cert.multi):
            ids, v = np.asarray(ids, dtype=np.int64), np.asarray(v, dtype=float)
            if (len(ids) == 0 or len(ids) != len(v) or ids.min() < 0 or ids.max() >= n or len(np.unique(ids)) != len(ids)
                    or not np.all(np.isfinite(v)) or np.any(np.abs(v) > limits.max_abs_coef)):
                rejected.append(f"multi cut {c}: invalid IDs or values")
                continue
            multi.append((ids, v))
        # frozen size rule (expanded form), checked before any expansion work
        s_cnt = self.counts(sj, sv) if len(sj) else np.zeros(0, dtype=np.int64)
        size = int(len(sj) + s_cnt.sum())
        m_cnt = []
        for ids, v in multi:
            cnt = self.counts(ids, v)
            m_cnt.append(cnt)
            size += int(len(ids) + cnt.sum())
        for ids, v, ws in cert.explicit:
            size += len(ids) + sum(len(ix) for ix, _ in ws)
        out["nonzeros_expanded"] = size
        if size > limits.max_nonzeros:
            raise CertFailure("too_many_nonzeros")
        check_time()
        expl = []  # (ids, [(fl map, exact b, exact map) per id])
        for c, (ids, v, ws) in enumerate(cert.explicit):
            ids, v = np.asarray(ids, dtype=np.int64), np.asarray(v, dtype=float)
            if (len(ids) == 0 or len(ids) != len(v) or len(ws) != len(ids) or ids.min() < 0 or ids.max() >= n
                    or len(np.unique(ids)) != len(ids) or not np.all(np.isfinite(v))):
                rejected.append(f"explicit cut {c}: invalid IDs or non-finite v")
                continue
            parts, reason = [], ""
            for j, vj, (ix, w) in zip(ids.tolist(), v.tolist(), ws):
                ix, w = np.asarray(ix, dtype=np.int64), np.asarray(w, dtype=float)
                if len(ix) != len(w) or (len(ix) and (ix.min() < 0 or ix.max() >= m)) or len(np.unique(ix)) != len(ix) \
                        or not np.all(np.isfinite(w)):
                    reason = "invalid w"
                    break
                acc, why, fl, b, ex = self._explicit_cut(j, vj, ix, w, reject_tol)
                if not acc:
                    reason = why
                    break
                parts.append((fl, b, ex))
            if reason:
                rejected.append(f"explicit cut {c}: {reason}")
                continue
            expl.append((ids, parts))
            if not c % 64:
                check_time()
        # ---- master LP (floats; multipliers only)
        rows_r, rows_c, rows_v, rhs, th_r, th_c = [], [], [], [], [], []
        r = 0
        if len(sj):
            rr = np.repeat(np.arange(len(sj)), s_cnt)
            tt = np.arange(int(s_cnt.sum())) - np.repeat(np.cumsum(s_cnt) - s_cnt, s_cnt)
            rows_r.append(rr)
            rows_c.append(self.order[sj[rr], tt].astype(np.int64))
            rows_v.append(sv[rr] - self.ds[sj[rr], tt])
            rhs.append(sv)
            th_r.append(np.arange(len(sj)))
            th_c.append(sj)
            r = len(sj)
        for (ids, v), cnt in zip(multi, m_cnt):
            Wc = np.zeros(m)
            for j, vj, cj in zip(ids.tolist(), v.tolist(), cnt.tolist()):
                Wc[self.order[j, :cj]] += vj - self.ds[j, :cj]
            nz = np.nonzero(Wc)[0]
            rows_r.append(np.full(len(nz), r))
            rows_c.append(nz)
            rows_v.append(Wc[nz])
            rhs.append(np.array([math.fsum(v.tolist())]))
            th_r.append(np.full(len(ids), r))
            th_c.append(ids)
            r += 1
            check_time()
        for ids, parts in expl:
            Wc: dict[int, float] = {}
            for fl, _, _ in parts:
                for i, x in fl.items():
                    Wc[i] = Wc.get(i, 0.0) + x
            nz = np.array(sorted(Wc), dtype=np.int64)
            rows_r.append(np.full(len(nz), r))
            rows_c.append(nz)
            rows_v.append(np.array([Wc[i] for i in nz.tolist()]))
            rhs.append(np.array([float(sum((p[1] for p in parts), Fraction(0)))]))
            th_r.append(np.full(len(ids), r))
            th_c.append(ids)
            r += 1
        R = r
        master_nnz = int(sum(len(x) for x in rows_c)) + int(sum(len(x) for x in th_c))
        if master_nnz > limits.max_master_nonzeros:
            raise CertFailure("too_many_master_nonzeros")
        out["expand_seconds"] = round(clock() - t0, 3)
        check_time()
        status, lp_value, lam, mu = "no_cuts", None, np.zeros(R), 0.0
        own = env is None
        t1 = clock()
        if own:
            env = make_env(1, mem_limit_gb=limits.gurobi_mem_gb)
        md = gp.Model(env=env)
        try:
            md.Params.OutputFlag = 0
            md.Params.Method = method
            md.Params.TimeLimit = max(0.01, stop - clock())
            md.Params.FeasibilityTol = 1e-9
            md.Params.OptimalityTol = 1e-9
            y = md.addMVar(m, lb=0.0, ub=1.0, name="y")
            th = md.addMVar(n, lb=0.0, ub=self.U_float, name="theta")
            card = md.addConstr(y.sum() == k)
            md.setObjective(th.sum(), GRB.MINIMIZE)
            cons = None
            if R:
                vals = np.concatenate(rows_v)
                if lp_coef_scale is not None:
                    vals = vals * lp_coef_scale
                Ay = sp.csr_matrix((vals, (np.concatenate(rows_r), np.concatenate(rows_c))), shape=(R, m))
                tr = np.concatenate(th_r)
                At = sp.csr_matrix((np.ones(len(tr)), (tr, np.concatenate(th_c))), shape=(R, n))
                cons = md.addConstr(Ay @ y + At @ th >= np.concatenate(rhs))
            md.optimize()
            if md.Status == GRB.OPTIMAL:
                status = "optimal"
                lp_value = float(md.ObjVal)
                mu = float(card.Pi)
                if cons is not None:
                    lam = np.maximum(np.asarray(cons.Pi, dtype=float), 0.0)
                if not (math.isfinite(mu) and np.all(np.isfinite(lam))):
                    status = "non_finite_duals"
            else:
                status = {GRB.INFEASIBLE: "infeasible", GRB.INF_OR_UNBD: "infeasible", GRB.TIME_LIMIT: "time_limit",
                          GRB.MEM_LIMIT: "memory_limit"}.get(md.Status, f"lp_status_{md.Status}")
        except gp.GurobiError as err:
            status = "memory_limit" if getattr(err, "errno", None) == GRB.Error.OUT_OF_MEMORY else f"gurobi_error: {err}"
        finally:
            md.dispose()
            if own:
                env.dispose()
        out.update(lp_value=lp_value, lp_status=status, cuts_used=R, rejected=rejected,
                   cuts_with_positive_dual=int(np.count_nonzero(lam)), master_seconds=round(clock() - t1, 3))
        if status != "optimal":
            raise CertFailure({"infeasible": "master_infeasible", "time_limit": "time_limit"}.get(status, "numerical"), status)
        check_time()
        # ---- exact Neumaier-Shcherbina bound from the original data
        t2 = clock()
        LB = self._exact_bound(sj, sv, s_cnt, multi, m_cnt, expl, lam, mu, check_time)
        out["exact_seconds"] = round(clock() - t2, 3)
        LB = max(LB, Fraction(0))  # LB_cert = max(LB, trivial bound)
        out.update(cert_ok=True, lb=str(LB), bound=float_down(LB), reason=None)
        return out

    def _exact_bound(self, sj, sv, s_cnt, multi, m_cnt, expl, lam, mu, check_time) -> Fraction:
        m, n, k = self.m, self.n, self.k
        Mu = _fr(mu)
        sel = np.nonzero(lam > 0)[0]
        lam_fr = {int(c): _fr(lam[c]) for c in sel}
        # common power-of-two scale for the multipliers: lambda_c = A_c / 2^S
        S = max((x.denominator.bit_length() - 1 for x in lam_fr.values()), default=0)
        A = {c: x.numerator << (S - (x.denominator.bit_length() - 1)) for c, x in lam_fr.items()}
        G_int = [0] * m                   # sum_c A_c W_ci for integral W (scale 2^S)
        G_fr: dict[int, Fraction] = {}    # non-integral contributions (exact)
        Lam_int = [0] * n                 # sum_{c: j in ids} A_c (scale 2^S)
        B = Fraction(0)
        ns = len(sj)
        for count, c in enumerate(sel.tolist()):
            if not count % 256:
                check_time()
            a = A[c]
            if c < ns:
                j, v = int(sj[c]), float(sv[c])
                Lam_int[j] += a
                B += lam_fr[c] * _fr(v)
                cnt = int(s_cnt[c])
                ix, dd = self.order[j, :cnt], self.ds[j, :cnt]
                if self.integral and _is_integer_float(v):
                    iv = int(v)
                    for i, d in zip(ix.tolist(), dd.tolist()):
                        G_int[i] += a * (iv - int(d))
                else:
                    V = _fr(v)
                    for i, d in zip(ix.tolist(), dd.tolist()):
                        G_fr[i] = G_fr.get(i, Fraction(0)) + lam_fr[c] * (V - _fr(d))
            elif c < ns + len(multi):
                ids, v = multi[c - ns]
                cnt = m_cnt[c - ns]
                for j in ids.tolist():
                    Lam_int[j] += a
                if self.integral and all(_is_integer_float(x) for x in v.tolist()):
                    B += lam_fr[c] * sum(int(x) for x in v.tolist())
                    Wc: dict[int, int] = {}
                    for j, vj, cj in zip(ids.tolist(), v.tolist(), cnt.tolist()):
                        iv = int(vj)
                        for i, d in zip(self.order[j, :cj].tolist(), self.ds[j, :cj].tolist()):
                            Wc[i] = Wc.get(i, 0) + (iv - int(d))
                    for i, w in Wc.items():
                        G_int[i] += a * w
                else:
                    B += lam_fr[c] * sum((_fr(x) for x in v.tolist()), Fraction(0))
                    for j, vj, cj in zip(ids.tolist(), v.tolist(), cnt.tolist()):
                        V = _fr(vj)
                        for i, d in zip(self.order[j, :cj].tolist(), self.ds[j, :cj].tolist()):
                            G_fr[i] = G_fr.get(i, Fraction(0)) + lam_fr[c] * (V - _fr(d))
            else:
                ids, parts = expl[c - ns - len(multi)]
                for j in ids.tolist():
                    Lam_int[j] += a
                for _, b, ex in parts:
                    B += lam_fr[c] * b
                    for i, W in ex.items():
                        G_fr[i] = G_fr.get(i, Fraction(0)) + lam_fr[c] * W
        check_time()
        den = 1 << S
        LB = Mu * k + B
        for j in range(n):
            if Lam_int[j] > den:  # r_j = 1 - Lambda_j < 0: the theta box's upper end
                LB += Fraction(den - Lam_int[j], den) * self.U_exact(j)
            # r_j >= 0: r_j * L_j = 0 (L_j = 0)
        for i in range(m):
            g = Fraction(G_int[i], den) + G_fr.get(i, Fraction(0))
            ri = -Mu - g
            if ri < 0:
                LB += ri
        return LB


# ---------------------------------------------------------------- budget helpers for publishing arms

def fit_budget(chk_counts: Callable[[np.ndarray, np.ndarray], np.ndarray], j: np.ndarray, v: np.ndarray,
               limit: int) -> np.ndarray:
    """Indices of the longest prefix of the single cuts (j, v) (in the given priority order) whose expanded size
    fits the limit."""
    if not len(j):
        return np.zeros(0, dtype=np.int64)
    size = 1 + chk_counts(np.asarray(j), np.asarray(v))
    keep = np.cumsum(size) <= limit
    return np.nonzero(keep)[0]


def sorted_counts(D: np.ndarray, sc) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """#{i: D_ij < v} using the L-prefix of SortedCosts and a full-column fallback beyond it."""
    def counts(j: np.ndarray, v: np.ndarray) -> np.ndarray:
        j = np.asarray(j, dtype=np.int64)
        v = np.asarray(v, dtype=float)
        out = np.zeros(len(j), dtype=np.int64)
        for a in range(0, len(j), 4096):
            b = min(len(j), a + 4096)
            out[a:b] = (sc.ds[j[a:b]] < v[a:b, None]).sum(axis=1)
        far = np.nonzero(v >= sc.ds[j, -1])[0] if len(j) else np.zeros(0, dtype=np.int64)
        for t in far.tolist():
            out[t] = int((D[:, j[t]] < v[t]).sum())
        return out
    return counts


# ---------------------------------------------------------------- LP* interval (exact endpoints with provenance)

def repair_point(y: np.ndarray, k: int) -> list[Fraction]:
    """Exact rational y in [0, 1]^m with sum y = k, close to the float point y."""
    Y = [min(max(_fr(x), Fraction(0)), Fraction(1)) if math.isfinite(x) else Fraction(0) for x in np.asarray(y, float).tolist()]
    s = sum(Y, Fraction(0))
    d = Fraction(k) - s
    order = np.argsort(-np.asarray(y, dtype=float), kind="stable").tolist()
    if d > 0:
        for i in order:
            add = min(Fraction(1) - Y[i], d)
            Y[i] += add
            d -= add
            if d == 0:
                break
    elif d < 0:
        for i in reversed(order):
            sub_ = min(Y[i], -d)
            Y[i] -= sub_
            d += sub_
            if d == 0:
                break
    if d != 0:
        raise ValueError("cannot repair the point (k > m?)")
    return Y


def exact_master_value(chk: Checker, Y: list[Fraction]) -> Fraction:
    """sum_j Q_j(Y) exactly (greedy fill in increasing distance), for a rational Y in [0, 1]^m with sum Y = k."""
    total = Fraction(0)
    for j in range(chk.n):
        acc, val = Fraction(0), Fraction(0)
        for i, d in zip(chk.order[j].tolist(), chk.ds[j].tolist()):
            yi = Y[i]
            if yi == 0:
                continue
            take = min(yi, 1 - acc)
            val += take * (int(d) if chk.integral else _fr(d))
            acc += take
            if acc == 1:
                break
        if acc != 1:
            raise ValueError("point does not cover a customer (sum y < 1?)")
        total += val
    return total


def lp_interval(chk: Checker, results: list[tuple[str, dict]], points: list[tuple[str, np.ndarray]]) -> dict[str, Any]:
    """Rigorous LP* bracket with provenance: lo = the largest exact certified LB among `results` (pairs (source,
    checker result); only cert_ok results count), hi = the smallest exact master objective among `points` (pairs
    (source, float LP point), each repaired exactly)."""
    lo, lo_src = None, None
    for src, res in results:
        if res and res.get("cert_ok"):
            q = Fraction(res["lb"])
            if lo is None or q > lo:
                lo, lo_src = q, src
    hi, hi_src = None, None
    for src, p in points:
        if p is None:
            continue
        q = exact_master_value(chk, repair_point(np.asarray(p, dtype=float), chk.k))
        if hi is None or q < hi:
            hi, hi_src = q, src
    out = {"lo": str(lo) if lo is not None else None, "lo_float": float_down(lo) if lo is not None else None,
           "lo_provenance": lo_src, "hi": str(hi) if hi is not None else None,
           "hi_float": float_up(hi) if hi is not None else None, "hi_provenance": hi_src,
           "rigorous": lo is not None and hi is not None and lo <= hi}
    if lo is not None and hi is not None:
        out["width"] = float(hi - lo)
        out["rel_width"] = float((hi - lo) / max(abs(hi), Fraction(1)))
    return out


def target(interval: dict[str, Any], eps: float) -> Fraction | None:
    """Conservative bound target hi - eps * max(|hi|, 1) (bgap_LP = (LP_hi - LB)/max(|LP_hi|, 1) <= eps)."""
    if interval.get("hi") is None:
        return None
    hi = Fraction(interval["hi"])
    return hi - Fraction(str(eps)) * max(abs(hi), Fraction(1))
