"""DecompGym-XL scorer, protocol v4 (Sections 3.4 and 4; v3.1 Sections 4.3 and 20.1-20.6 where v4 is silent).

Inputs per attempt: the supervisor's timestamped, independently verified incumbents (receipt times on the
supervisor clock, relative to the solver start), the solver's bound/status claims, the method audit (`method_ok`,
passed through), infrastructure errors (supervisor errors, unverifiable stored payloads) and, for BD-LP families, the
certificate checkpoints checked by src.experiments.xl_certificate.check_versions.
Per instance: the frozen reference manifest entry (`Reference`): T_f, z_best, z_MIP and the MIP's time to reach it,
the objective sense, whether objectives are integral, eps_p (near-best) and eps_b (bound target), the rigorous LP*
interval [LP_lo, LP_hi] (LP_hi = the exact objective of a verified primal; without it the bound target is undefined),
the compact MIP's dual bound at T_f, the number of certificate checkpoints and the tier's headline target.

Primal metrics (v3.1, minimization coordinates y = s z with s = +1 for min, -1 for max):
    feasible            a verified incumbent received strictly before T_f (timeouts and crashes are scored by the best
                        eligible incumbent; unlike the legacy src/experiments/verification.py)
    matches_or_beats_MIP  y <= y_MIP + tau, tau = 0 for integral objectives else 1e-6 + 1e-9 |z_MIP|; None without z_MIP
    gap_best            (y - y_best) / max(|z_best|, 1) at T_f (signed; +inf without an incumbent)
    primal_integral     (1/T_f) int_0^T_f p(t) dt, p = 1 before the first incumbent, else clip(gap_best(t), 0, 1)
    speedup             max(t_MIP, 1 ms) / max(t, 1 ms), t = first incumbent meeting the MIP target; censored ->
                        speedup_upper = max(t_MIP, 1 ms) / max(T_f, 1 ms) (the same floor in both branches)
    near_best           gap_best <= eps_p
Bound claims (integrity filter): a claim is an object with optional "bound"/"lower_bound" (native sense: lower bound for
min, upper bound for max), "status" and "objective" (finite numbers, compared exactly). A claimed bound L is refuted if
s L > U + tau(U) for any independently verified objective U (this attempt's incumbents, late ones included, z_best,
z_MIP, a proven optimum, or extra verified values); an 'optimal' claim is the claim L = objective and is also refuted if
it differs from a proven optimum by more than tau. A malformed claim (not an object, duplicate keys, non-finite or
non-numeric values, an 'optimal' claim without a valid objective) is recorded as a failed claim. A refuted or
malformed claim fails every target of the attempt.
Bound metrics (BD-LP, minimization):
    LB_cert             the certified bound of the final version (the T_f checkpoint); trivial bound when cert_ok fails
    bgap_LP             (LP_hi - LB_cert) / max(|LP_hi|, 1), exact (conservative: LP_hi >= LP*)
    bound_target        cert_ok and method_ok and no refuted claim and bgap_LP <= eps_b (no incumbent required)
    t_bound             first checkpoint whose certified bound meets the target (gated by method_ok and no refuted claim;
                        an earlier valid attainment survives a failed final version); censored (None) otherwise;
                        t_bound_raw ignores the gates; t_bound_receipt is that version's receipt time (diagnostic)
    beats_MIP_bound     LB_cert >= the compact MIP's dual bound at T_f (reported only)
Infrastructure errors are never scores. An inconsistent reference (`Reference.validate`), a malformed incumbent
record, a supervisor error or an unverifiable stored payload makes every target of the attempt undefined. An incomplete
checkpoint record (not exactly the frozen schedule ending at T_f, or no checked certificate on a BD-LP instance), a
checkpoint with a non-finite time, a missing or non-finite bound or a non-boolean cert_ok, a stored certificate that
failed its receipt check, or a certified bound above LP_hi or z_best (an inconsistent view or reference) makes the
bound target undefined. Such attempts are listed in `infrastructure_errors` and must be rerun or their reference fixed.
Headlines: a target headline uses the final success; a speed headline "time_to_<target>" succeeds iff the gated
attainment time exists (bound Q@k keeps using the final-version success).
All target comparisons are exact (fractions.Fraction of the float objectives, the decimal epsilons and the exact
bounds); reported gaps are floats.

Q@k (`q_at_k`, per pair): Q@k = 1 if one of the first k attempts meets the target (with feasibility, method_ok and
no refuted claim), separately for matches_or_beats_MIP, near_best and bound_target; None when the target is not
defined for the instance (e.g. no MIP incumbent: report a predeclared feasibility cohort instead) or when an attempt
among the first k has an infrastructure error and no earlier attempt succeeded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Iterable, Mapping, Sequence

TARGETS = ("matches_or_beats_MIP", "near_best", "bound_target")
TAU_ABS = Fraction("1e-6")
TAU_REL = Fraction("1e-9")
TIME_FLOOR = 1e-3   # seconds; a receipt time below it counts as TIME_FLOOR in speed ratios


def q(value: Any) -> Fraction | None:
    """Exact rational of a manifest/log value (decimal text, int, float, Fraction); None stays None."""
    if value is None:
        return None
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value")
        return Fraction(value)
    return Fraction(str(value).strip())


def _f(x: Fraction | None) -> float | None:
    return None if x is None else float(x)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _exact_or_none(x: Any) -> Fraction | None:
    """An exact rational of a finite log/manifest number, or None if it is missing or not a finite number."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = q(x)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None   # (b) callers turn a None here into an infrastructure error (malformed log/manifest record)
    return v


def claim_number(value: Any) -> Fraction | None:
    """A finite claim value as an exact rational (int, finite float, Fraction or decimal text of at most 64
    characters with |exponent| <= 400); None for anything else (booleans, NaN, infinities, other text)."""
    from decimal import Decimal, InvalidOperation

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, Fraction)):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(value) if math.isfinite(value) else None
    if isinstance(value, str) and len(value) <= 64:
        try:
            d = Decimal(value.strip())
        except (InvalidOperation, ValueError):
            return None   # (a) agent: a non-numeric claim value -> malformed claim, fails the attempt
        if d.is_finite() and (not d or abs(d.adjusted()) <= 400):
            return Fraction(*d.as_integer_ratio())
    return None


def claim_values(c: Any) -> tuple[list[tuple[str, Fraction]], str | None]:
    """Native-sense bound values of one claim [(name, value)] or (, reason) for a malformed claim."""
    if not isinstance(c, Mapping):
        return [], "not_an_object"
    if c.get("_error"):
        return [], str(c["_error"])
    out = []
    for key in ("bound", "lower_bound"):
        if c.get(key) is not None:
            v = claim_number(c[key])
            if v is None:
                return [], f"bad_{key}"
            out.append((key, v))
    status = c.get("status")
    if status is not None and not isinstance(status, str):
        return [], "bad_status"
    obj = c.get("objective")
    if obj is not None and claim_number(obj) is None:
        return [], "bad_objective"
    if isinstance(status, str) and status.lower() == "optimal":
        if obj is None:
            return [], "optimal_without_objective"
        out.append(("optimal", claim_number(obj)))
    return out, None


@dataclass(frozen=True)
class Reference:
    """Frozen per-instance reference (protocol v4 Section 7 manifest entry)."""
    T_f: float
    z_best: Fraction
    sense: int = 1                         # +1 minimization, -1 maximization
    z_mip: Fraction | None = None          # MIP's best verified objective at T_f (None: no MIP incumbent)
    t_mip: float | None = None             # supervisor time at which the MIP first reached z_MIP
    integral: bool = False
    eps_p: Fraction | None = None
    eps_b: Fraction | None = None
    lp_lo: Fraction | None = None
    lp_hi: Fraction | None = None
    lb_mip: Fraction | None = None         # compact MIP dual bound at T_f (native sense)
    z_opt: Fraction | None = None          # proven optimum, if any
    headline: str | None = None            # one of TARGETS or "time_to_<target>"
    instance: str | None = None
    checkpoints: int = 4                   # frozen checkpoint schedule T_f * i / checkpoints, i = 1..checkpoints

    @classmethod
    def from_manifest(cls, e: Mapping[str, Any]) -> "Reference":
        sense = e.get("sense", "min")
        lp = e.get("lp_interval") or (None, None)
        if isinstance(lp, Mapping):
            lp = (lp.get("lp_lo"), lp.get("lp_hi"))
        return cls(T_f=float(e["T_f"]), z_best=q(e["z_best"]), sense=1 if sense in ("min", 1, "+1") else -1,
                   z_mip=q(e.get("z_mip")), t_mip=None if e.get("t_mip") is None else float(e["t_mip"]),
                   integral=bool(e.get("integral", False)), eps_p=q(e.get("eps_p")), eps_b=q(e.get("eps_b")),
                   lp_lo=q(lp[0]), lp_hi=q(lp[1]), lb_mip=q(e.get("lb_mip")), z_opt=q(e.get("z_opt")),
                   headline=e.get("headline"), instance=e.get("instance"), checkpoints=int(e.get("checkpoints", 4)))

    def validate(self) -> list[str]:
        """Consistency of the frozen reference, checked before any arithmetic (violations are infrastructure errors:
        the manifest must be fixed): finite T_f > 0 and t_MIP >= 0, exact finite numbers, eps >= 0, and the orderings
        y_best <= y_MIP, y_opt <= y_best, LP_lo <= LP_hi, LP_lo <= z_best and LB_MIP <= z_best (minimization)."""
        errs = []
        if not _finite(self.T_f) or self.T_f <= 0:
            errs.append("T_f_not_finite_positive")
        if self.sense not in (1, -1):
            errs.append("bad_sense")
        if not isinstance(self.checkpoints, int) or isinstance(self.checkpoints, bool) or self.checkpoints < 1:
            errs.append("bad_checkpoints")
        if self.z_best is None:
            errs.append("z_best_missing")
        for name in ("z_best", "z_mip", "eps_p", "eps_b", "lp_lo", "lp_hi", "lb_mip", "z_opt"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, Fraction))):
                errs.append(f"{name}_not_exact")
        if self.t_mip is not None and (not _finite(self.t_mip) or self.t_mip < 0):
            errs.append("t_mip_not_finite_nonnegative")
        if errs:
            return errs
        s, yb, tau = self.sense, self.sense * self.z_best, self.tau(self.z_best)
        for name in ("eps_p", "eps_b"):
            if getattr(self, name) is not None and getattr(self, name) < 0:
                errs.append(f"{name}_negative")
        if self.z_mip is not None and yb > s * self.z_mip + self.tau(self.z_mip):
            errs.append("z_best_worse_than_z_mip")
        if self.z_opt is not None and s * self.z_opt > yb + tau:
            errs.append("z_opt_worse_than_z_best")
        if self.lp_lo is not None and self.lp_hi is not None and self.lp_lo > self.lp_hi:
            errs.append("lp_lo_above_lp_hi")
        if s == 1 and self.lp_lo is not None and self.lp_lo > self.z_best + tau:
            errs.append("lp_lo_above_z_best")
        if s == 1 and self.lb_mip is not None and self.lb_mip > self.z_best + tau:
            errs.append("lb_mip_above_z_best")
        return errs

    def tau(self, z: Fraction | None) -> Fraction:
        if self.integral or z is None:
            return Fraction(0)
        return TAU_ABS + TAU_REL * abs(z)

    def norm(self) -> Fraction:
        return max(abs(self.z_best), Fraction(1))


@dataclass
class Attempt:
    """One scored execution: verified incumbents [(t, objective)] (receipt time on the supervisor clock; objective
    recomputed by the family verifier), claims, method audit and certificate checkpoints."""
    incumbents: Sequence[tuple[float, Any]] = ()
    method_ok: bool | None = None
    claims: Sequence[Mapping[str, Any]] = ()
    certificate: Sequence[Mapping[str, Any]] | None = None
    termination: str | None = None
    attempt: int | None = None
    late_incumbents: Sequence[tuple[float, Any]] = ()   # verified but received at or after T_f (refutation only)
    infrastructure: Sequence[str] = ()                  # supervisor errors, unverifiable stored payloads, ...
    extra: dict[str, Any] = field(default_factory=dict)


def _first_time(traj: Sequence[tuple[float, Fraction]], pred: Callable[[Fraction], bool]) -> float | None:
    for t, y in traj:
        if pred(y):
            return t
    return None


def score_attempt(att: Attempt, ref: Reference, *, known_feasible: Iterable[Any] = ()) -> dict[str, Any]:
    """Per-attempt v4 record (see the module docstring)."""
    known_feasible = list(known_feasible)   # materialized once: validated and used for refutation
    pre = [f"reference: {e}" for e in ref.validate()]
    for name, rows in (("incumbent", att.incumbents), ("late_incumbent", att.late_incumbents)):
        for row in rows:
            if (not isinstance(row, (tuple, list)) or len(row) != 2 or not _finite(row[0]) or row[0] < 0
                    or _exact_or_none(row[1]) is None):
                pre.append(f"malformed_{name}_record")
                break
    if any(_exact_or_none(z) is None for z in known_feasible):
        pre.append("malformed_known_feasible_value")
    if pre:
        return _undefined_record(att, ref, pre + list(att.infrastructure))
    s, T = ref.sense, ref.T_f
    norm = ref.norm()
    y_best = s * ref.z_best
    y_mip = None if ref.z_mip is None else s * ref.z_mip
    tau_mip = ref.tau(ref.z_mip)
    method_ok = att.method_ok is True

    # eligible trajectory: receipt strictly before T_f, running best in y coordinates
    eligible = sorted((float(t), s * q(z)) for t, z in att.incumbents if float(t) < T)
    traj: list[tuple[float, Fraction]] = []
    for t, y in eligible:
        if not traj or y < traj[-1][1]:
            traj.append((t, y))
    feasible = bool(traj)
    y_final = traj[-1][1] if traj else None
    gap = (y_final - y_best) / norm if feasible else None

    # primal integral over [0, T_f]
    area = Fraction(0)
    prev_t, prev_p = 0.0, Fraction(1)
    for t, y in traj:
        area += prev_p * Fraction(t - prev_t)
        prev_t, prev_p = t, min(max((y - y_best) / norm, Fraction(0)), Fraction(1))
    area += prev_p * Fraction(T - prev_t)
    pi = area / Fraction(T)

    # bound and status claims
    verified = [s * q(z) for _, z in list(att.incumbents) + list(att.late_incumbents)]
    verified += [y_best] + ([y_mip] if y_mip is not None else []) + ([s * ref.z_opt] if ref.z_opt is not None else [])
    verified += [s * q(z) for z in known_feasible]
    u_min = min(verified)
    refuted = []
    for c in att.claims:
        raw = dict(c) if isinstance(c, Mapping) else {"raw": repr(c)[:200]}
        values, err = claim_values(c)
        if err is not None:
            refuted.append({"claim": raw, "why": f"malformed_claim:{err}"})
            continue
        for name, v in values:
            yl = s * v
            if name == "optimal" and ref.z_opt is not None and abs(yl - s * ref.z_opt) > ref.tau(ref.z_opt):
                refuted.append({"claim": raw, "why": "optimal_differs_from_proven_optimum"})
            if yl > u_min + ref.tau(s * u_min):
                refuted.append({"claim": raw, "why": f"{name}_above_verified_solution"})
    refuted_claim = bool(refuted)
    gate = method_ok and not refuted_claim

    # MIP target and speed
    if y_mip is None:
        matches = None
        t_match = None
    else:
        matches = feasible and y_final <= y_mip + tau_mip
        t_match = _first_time(traj, lambda y: y <= y_mip + tau_mip)
    speedup = speedup_upper = None
    censored = None
    if y_mip is not None and ref.t_mip is not None:
        if t_match is not None:
            speedup, censored = max(ref.t_mip, TIME_FLOOR) / max(t_match, TIME_FLOOR), False
        else:
            speedup_upper, censored = max(ref.t_mip, TIME_FLOOR) / max(T, TIME_FLOOR), True

    near = None if ref.eps_p is None else (feasible and gap <= ref.eps_p)
    t_near = None if ref.eps_p is None else _first_time(traj, lambda y: (y - y_best) / norm <= ref.eps_p)

    rec: dict[str, Any] = {
        "instance": ref.instance, "attempt": att.attempt, "termination": att.termination, "method_ok": att.method_ok,
        "feasible": feasible, "objective": None if y_final is None else float(s * y_final),
        "incumbents_eligible": len(eligible), "incumbents_late": len(att.late_incumbents),
        "first_incumbent_t": traj[0][0] if traj else None,
        "gap_best": float(gap) if gap is not None else math.inf, "gap_best_exact": None if gap is None else str(gap),
        "primal_integral": float(pi),
        "matches_or_beats_MIP": matches, "t_matches_MIP": t_match,
        "speedup": speedup, "speedup_censored": censored, "speedup_upper": speedup_upper,
        "near_best": near, "t_near_best": t_near,
        "refuted_claim": refuted_claim, "refutations": refuted, "claims": len(att.claims),
    }

    # bound side
    bound, bound_infra = _bound_metrics(att, ref, gate)
    rec.update(bound)

    success = {
        "matches_or_beats_MIP": None if matches is None else bool(matches and gate),
        "near_best": None if near is None else bool(near and gate),
        "bound_target": rec["bound_target"],
    }
    time_to = {   # censored (None) when never reached or when the attempt fails the gates
        "matches_or_beats_MIP": t_match if gate and matches is not None else None,
        "near_best": t_near if gate else None,
        "bound_target": rec.get("t_bound"),
    }
    infra = list(att.infrastructure) + bound_infra
    undefined = list(TARGETS) if att.infrastructure else (["bound_target"] if bound_infra else [])
    for tgt in undefined:
        success[tgt] = None
        time_to[tgt] = None
    if att.infrastructure:
        rec["bound_target"] = None
    rec["success"] = success
    rec["time_to_target"] = time_to
    rec["infrastructure_errors"] = infra
    rec["undefined_by_infrastructure"] = undefined
    if ref.headline:
        speed = ref.headline.startswith("time_to_")
        name = ref.headline[len("time_to_"):] if speed else ref.headline
        if speed:
            defined = success.get(name) is not None
            t_head = time_to.get(name)
            rec["headline"] = {"target": ref.headline, "success": (t_head is not None) if defined else None, "time": t_head}
        else:
            rec["headline"] = {"target": ref.headline, "success": success.get(name), "time": None}
    return rec


def _undefined_record(att: Attempt, ref: Reference, infra: list[str]) -> dict[str, Any]:
    """Record of an attempt whose inputs are invalid (reference or logs): no metric is computed, every target is
    undefined (infrastructure)."""
    rec: dict[str, Any] = {"instance": ref.instance, "attempt": att.attempt, "termination": att.termination,
                           "method_ok": att.method_ok, "infrastructure_errors": infra,
                           "undefined_by_infrastructure": list(TARGETS), "success": {t: None for t in TARGETS},
                           "time_to_target": {t: None for t in TARGETS}, "bound_target": None, "refuted_claim": None}
    if ref.headline:
        rec["headline"] = {"target": ref.headline, "success": None, "time": None}
    return rec


def _bound_metrics(att: Attempt, ref: Reference, gate: bool) -> tuple[dict[str, Any], list[str]]:
    """Bound side and its infrastructure errors. Defined (bound_target not None) only for minimization instances
    with a frozen LP_hi (a verified primal) and eps_b. The checkpoint record must be exactly the frozen schedule
    (ending at T_f) and consistent with the reference; otherwise the bound target is undefined (infrastructure)."""
    out: dict[str, Any] = {"cert_ok": None, "lb_cert": None, "lb_cert_float": None, "lb_cert_max_checkpoints": None,
                           "bgap_LP": None, "bgap_LP_exact": None, "bound_target": None, "t_bound": None, "t_bound_raw": None,
                           "t_bound_receipt": None, "beats_MIP_bound": None}
    infra: list[str] = []
    defined = ref.lp_hi is not None and ref.eps_b is not None and ref.sense == 1
    raw = list(att.certificate or [])
    if not raw:
        if defined:
            infra.append("no_checked_certificate_checkpoints")
        return out, infra
    # validate every field before any arithmetic: finite checkpoint times, non-null finite bounds, boolean cert_ok
    if any(not isinstance(c, Mapping) or not _finite(c.get("checkpoint")) or _exact_or_none(c.get("lb")) is None
           or not isinstance(c.get("cert_ok"), bool) for c in raw):
        return out, ["malformed_checkpoint_record"]
    cps = sorted(raw, key=lambda c: float(c["checkpoint"]))
    lbs = [_exact_or_none(c["lb"]) for c in cps]
    oks = [c["cert_ok"] for c in cps]
    k = ref.checkpoints
    expected = [ref.T_f * i / k for i in range(1, k + 1)]
    if len(cps) != k or any(abs(float(c["checkpoint"]) - e) > 1e-9 * max(1.0, ref.T_f) for c, e in zip(cps, expected)):
        infra.append("incomplete_checkpoints")
    for c in cps:
        if c.get("infrastructure"):
            infra.append(f"checkpoint_{c['checkpoint']}:{c['infrastructure']}")
    lb, cert_ok = lbs[-1], oks[-1]
    out.update(cert_ok=cert_ok, lb_cert=str(lb), lb_cert_float=float(lb), lb_cert_max_checkpoints=float(max(lbs)))
    if ref.lb_mip is not None and ref.sense == 1:
        out["beats_MIP_bound"] = bool(lb >= ref.lb_mip)
    if ref.sense == 1 and any(v > ref.z_best + ref.tau(ref.z_best) for v in lbs):
        infra.append("lb_cert_above_z_best")
    if ref.lp_hi is None:
        return out, infra
    hi = ref.lp_hi
    hnorm = max(abs(hi), Fraction(1))
    bgap = (hi - lb) / hnorm
    out.update(bgap_LP=float(bgap), bgap_LP_exact=str(bgap))
    if any(v > hi for v in lbs):
        infra.append("lb_cert_above_lp_hi")
    if defined and not infra:
        out["bound_target"] = bool(cert_ok and gate and bgap <= ref.eps_b)
        for c, v, o in zip(cps, lbs, oks):
            if o and (hi - v) / hnorm <= ref.eps_b:
                out["t_bound_raw"] = float(c["checkpoint"])
                rt = c.get("receipt_t", c.get("t"))
                out["t_bound_receipt"] = float(rt) if _finite(rt) else None
                break
        out["t_bound"] = out["t_bound_raw"] if gate else None
    return out, infra


# ---------------------------------------------------------------- Q@k

def q_at_k(records: Sequence[Mapping[str, Any]], k_max: int | None = None, targets: Sequence[str] = TARGETS) -> dict[str, Any]:
    """Per-pair Q@k for each target from the attempt records in attempt order (attempt 1 first). Q@k = 1 if a known
    success lies among the first k attempts; otherwise None if one of them has an infrastructure error for the target
    (or fewer than k attempts exist), else 0. "first" is the first success index, None if not certain. A target that is
    undefined for the instance (no MIP incumbent, no bound side) has defined = False."""
    k_max = k_max or len(records)
    out: dict[str, Any] = {}
    for tgt in targets:
        recs = list(records[:k_max])
        infra = [tgt in r.get("undefined_by_infrastructure", ()) for r in recs]
        preds = [r["success"].get(tgt) for r in recs]
        if not recs or all(p is None and not i for p, i in zip(preds, infra)):
            out[tgt] = {"q": [None] * k_max, "first": None, "defined": False, "infrastructure": any(infra)}
            continue
        qs: list[int | None] = []
        for k in range(1, k_max + 1):
            window = range(min(k, len(recs)))
            if any(preds[i] is True and not infra[i] for i in window):
                qs.append(1)
            elif any(infra[i] for i in window) or k > len(recs):
                qs.append(None)
            else:
                qs.append(0)
        first = next((i + 1 for i in range(len(recs)) if preds[i] is True and not infra[i]), None)
        if first is not None and any(infra[:first - 1]):
            first = None
        out[tgt] = {"q": qs, "first": first, "defined": True, "infrastructure": any(infra)}
    return out


def aggregate_q(pairs: Mapping[Any, Mapping[str, Any]], k_max: int, targets: Sequence[str] = TARGETS) -> dict[str, Any]:
    """Mean Q@k over pairs where the target is defined (pairs: key -> q_at_k output); pairs whose Q@k is unknown
    because of an infrastructure error are counted separately, never as failures."""
    out = {}
    for tgt in targets:
        rows = [p[tgt]["q"] for p in pairs.values() if p[tgt]["defined"]]
        q_k, unknown = [], []
        for k in range(k_max):
            vals = [r[k] for r in rows if r[k] is not None]
            q_k.append(sum(vals) / len(vals) if vals else None)
            unknown.append(sum(1 for r in rows if r[k] is None))
        out[tgt] = {"pairs": len(rows), "q": q_k, "unknown_pairs": unknown}
    return out


# ---------------------------------------------------------------- supervisor record -> Attempt

class InvalidSolution(Exception):
    """Raised by an incumbent verifier (`objective`) for a solution that is invalid (infeasible, malformed, wrong
    objective): the publication counts as infeasible, charged to the solver. Returning None means the same. Any other
    exception of the verifier is an evaluator failure: infrastructure, never an infeasible incumbent."""

def attempt_from_supervisor(record: Mapping[str, Any], objective: Callable[[bytes], Any], *, method_ok: bool | None,
                            certificate: Sequence[Mapping[str, Any]] | None = None, termination: str | None = None,
                            attempt: int | None = None, claims: Sequence[Mapping[str, Any]] = ()) -> Attempt:
    """Build an Attempt from a supervisor record (src.experiments.xl_supervisor). Every eligible incumbent payload is
    read and checked against its receipt hash, then verified with `objective(payload_bytes) -> objective or None`
    (the family's independent verifier: None or InvalidSolution = invalid solution, counted as infeasible; any other
    exception = infrastructure). Supervisor errors and stored payloads that are missing or fail their hash become
    infrastructure errors. Claims published through the channel are kept
    raw: a malformed one (record entry "error") is passed on as {"_error": reason} and fails the attempt; `claims`
    adds further claims (e.g. the final output's)."""
    from src.experiments.xl_supervisor import read_payload

    store = record["store_dir"]
    infra = [f"supervisor: {e}" for e in (record.get("errors") or ([record["error"]] if record.get("error") else []))]
    inc, infeasible = [], 0
    cache: dict[str, Any] = {}             # sha256 -> verified objective (or None: infeasible)
    verified_files: dict[str, str] = {}    # stored file -> sha256 of its bytes, checked against size and hash
    for e in record.get("incumbents", []):
        sha, fname = e.get("sha256"), e.get("file")
        # every distinct stored file is checked (size and hash) before its hash's cached objective is reused
        if verified_files.get(fname) != sha:
            try:
                payload = read_payload(store, e)
                if e.get("bytes") is not None and len(payload) != e["bytes"]:
                    raise ValueError(f"stored payload {fname} has {len(payload)} bytes, receipt says {e['bytes']}")
            except (OSError, ValueError, KeyError, TypeError) as exc:   # (b) stored payload missing/corrupt
                infra.append(f"incumbent {e.get('seq')}: {type(exc).__name__}: {exc}"[:200])
                continue
            verified_files[fname] = sha
            if sha not in cache:
                try:
                    cache[sha] = objective(payload)
                except InvalidSolution:   # (a) the solver's payload is an invalid solution: infeasible publication
                    cache[sha] = None
                except Exception as exc:  # noqa: BLE001 - (b) the verifier itself failed: infrastructure, not cached
                    infra.append(f"incumbent {e.get('seq')}: verifier {type(exc).__name__}: {exc}"[:200])
                    verified_files.pop(fname, None)
                    continue
        z = cache[sha]
        if z is None:
            infeasible += 1
        else:
            inc.append((float(e["t"]), z))
    all_claims: list[Mapping[str, Any]] = []
    for e in record.get("claims", []):
        if e.get("error") or not isinstance(e.get("payload"), dict):
            all_claims.append({"_error": e.get("error") or "unparsed", "t": e.get("t"), "seq": e.get("seq")})
        else:
            all_claims.append(dict(e["payload"], t=e.get("t")))
    all_claims += list(claims)
    counts = record.get("counts", {})
    return Attempt(incumbents=inc, method_ok=method_ok, claims=all_claims, certificate=certificate, termination=termination,
                   attempt=attempt, infrastructure=infra,
                   extra={"infeasible_incumbents": infeasible, "late": counts.get("late", 0), "torn": counts.get("torn", 0),
                          "claims_raw": [{k: e.get(k) for k in ("seq", "t", "sha256", "file", "error")} for e in record.get("claims", [])]})
