"""DRAFT XL task prompt and machine-readable view for the k-median family at scale (DecompGym-XL; protocol v4 §3.1,
§9.6). The paper's prompt (b_problem.py, unchanged) expects explicit D, Xf and Xc arrays; at the XL tiers (N = 2103 to
5934) D would be 4-35 million numbers, so the XL data file holds coordinates plus the exact distance rule instead
(67-260 KiB per instance).

Same optimization family, different instance distribution (Codex review 46 #6): the paper's generator draws separate
facility and customer sets and uses unrounded 10 x Euclidean costs (a_gen_data.py); the XL instances are TSPLIB
EUC_2D point sets with facilities = customers (m = n) and costs floor(Euclidean), the convention of the p-median
literature (Garcia, Labbe and Marin 2011; Duran-Mateluna, Ales and Elloumi 2023). The optimization semantics are
unchanged: open exactly k facilities, serve every customer from its cheapest open facility, minimize the total cost.
Published LB1 values in the literature are integer-reported (rounded) Phase-1 bounds, not exact LP* references; the
frozen references are our rigorous LP* intervals.

The certificate channel (supervisor-framed versions, protocol §3.2) is described abstractly; the concrete solver-side
API is part of the infrastructure item §9.3 and not fixed here.
"""
from __future__ import annotations

from typing import Any

KM_VIEW_SPEC: dict[str, Any] = {
    "view": "km-v1",
    "first_stage": "y_i in [0, 1] (facility i open; ids 'y<i>', 0-based, data order), cost 0; one row 'card': sum_i y_i = k",
    "subproblems": "one per customer j, id 'c<j>' (0-based, data order), weight 1",
    "subproblem": "Q_j(y) = min sum_i D_ij x_ij  s.t.  row 'd': sum_i x_ij = 1 (multiplier free);  row 'l<i>': x_ij <= y_i "
                  "(multiplier <= 0);  0 <= x_ij <= 1 (box, implied by 'd' and x >= 0)",
    "theta_box": "[sum_i min(0, D_ij), sum_i max(0, D_ij)] = [0, sum_i D_ij] (default formula)",
    "recourse": "complete for sum y = k, 1 <= k <= m",
    "compact_lp": "identical first-stage rows to the compact LP (classical x_ij <= y_i, sum_i x_ij = 1, sum y = k); the "
                  "radius formulation has auxiliary variables and is not part of the view (its cuts are y-space cuts)",
    "certificate": {
        "format": '{"view": "km-v1", "version": <int>, "cuts": [{"kind": "opt", "terms": [{"s": "c<j>", '
                  '"pi": {"d": <v_j>, "l<i>": <-w_ij>, ...}}, ...]}, ...]}',
        "cut": "a cut sums its terms over DISTINCT customers (weights from the view); a term (v, w) of customer j gives "
               "theta_j >= v + sum_i min(0, D_ij - v + w_i) - sum_i w_i y_i (exact, any finite values)",
        "strong_cut": "for any v, w_i = max(0, v - D_ij) for the facilities with D_ij < v (only those entries are "
                      "needed) gives the cut theta_j >= v - sum_i max(0, v - D_ij) y_i; at an LP point the best v is "
                      "the distance at which the cumulative sum of y over the facilities sorted by D_ij reaches 1",
        "limits": "<= 256 MB, <= 2,000,000 multiplier entries in total (counted as written: one per 'd' and per "
                  "'l<i>'), <= 600 s checker time per attempt over all checkpoints; values <= 1e12 in magnitude, "
                  "tokens <= 64 characters",
        "scoring": "the certificate version current at each checkpoint (60/300/600/900 s) is checked exactly; only "
                   "cuts enter the certified bound, never a reported bound",
    },
}

DATA_SCHEMA = (
    '{"format": "kmedian-xl-v1", "name": str, "k": int, "rounding": "floor" | "none", "scale": number,\n'
    '   "facilities": [[x, y], ...] (m rows), "customers": [[x, y], ...] (n rows), "meta": {...}}'
)

BASE_PROMPT_KM_XL = """You are an expert operations research engineer and Python programmer.

You are given a large k-median facility location problem:
  - Choose exactly k = {k} of the m = {m} candidate facilities to open; every one of the n = {n} customers is served by
    its cheapest open facility.
  - Minimize the total assignment cost: sum over customers j of min over open facilities i of D[i][j].

The data for this instance lives in data.json (read it at run time; it is too large to show in full):
  {schema}
The distance matrix is NOT stored. For facility i and customer j it is defined exactly as
    dx = facilities[i][0] - customers[j][0]
    dy = facilities[i][1] - customers[j][1]
    D[i][j] = r(scale * sqrt(dx * dx + dy * dy))
evaluated in IEEE double precision in exactly this order (scale 1 skips the multiplication), with r = floor when
"rounding" is "floor" and the identity when it is "none". This instance: rounding = "{rounding}", scale = {scale}.
Computing all of D needs m * n = {mn} entries ({mb:.0f} MB as float64); plan your memory (1 CPU, 16 GB).

Preview:
  facilities[0:4] = {fac_preview}
  customers[0:4]  = {cus_preview}
{sites_note}
Lower-bound certificate (scored separately from the solution): publish certificates for the two-stage view "km-v1":
  first stage y_i in [0, 1] with sum_i y_i = k; one subproblem per customer j (id "c<j>"):
    Q_j(y) = min sum_i D[i][j] x_ij  s.t.  row "d": sum_i x_ij = 1;  rows "l<i>": x_ij <= y_i;  0 <= x_ij <= 1.
  A certificate is {{"view": "km-v1", "version": <int>, "cuts": [{{"kind": "opt", "terms": [{{"s": "c<j>", "pi": {{"d": v,
  "l<i>": -w_i, ...}}}}, ...]}}, ...]}}: each term is a dual vector of a subproblem (multiplier of "d" free, of "l<i>"
  <= 0); the evaluator builds the cuts in exact arithmetic and re-solves the LP master over y in [0, 1]^m, sum y = k.
  Limits: at most 2,000,000 multiplier entries per certificate; the version current at each checkpoint is scored.

Task:
Assume gurobipy and NumPy are available; the Python standard library is also allowed. Do not depend on any files other
than data.json. Write a Python program solver.py that:
1) Loads data.json and builds what it needs from the coordinates and the distance rule.
2) Implements Benders decomposition according to the method requirements below.
3) Writes output.json with at least:
    {{
      "status": "optimal | feasible | infeasible | error",
      "method": "benders_decomposition",
      "objective_value": <float>,
      "y": [0/1 list of length m, exactly k ones],
      "assign": [index of the open facility serving each customer (optional)]
    }}
   objective_value must equal sum_j min over open i of D[i][j].

{method_requirements}
"""


def render_prompt_xl(data: dict[str, Any], method_requirements: str = "<method requirements>") -> str:
    """Draft rendering (the XL generation pipeline and the publication API are not wired yet)."""
    m, n = len(data["facilities"]), len(data["customers"])
    same = data["facilities"] == data["customers"]
    note = ("  (Facilities and customers are the same point set: facility i is at customer i's location, so D[i][i] = 0.)\n"
            if same else "")
    return BASE_PROMPT_KM_XL.format(
        k=data["k"], m=m, n=n, schema=DATA_SCHEMA, rounding=data.get("rounding", "floor"), scale=data.get("scale", 1),
        mn=m * n, mb=m * n * 8 / 2**20, fac_preview=data["facilities"][:4], cus_preview=data["customers"][:4],
        sites_note=note, method_requirements=method_requirements)
