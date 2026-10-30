# Closing the LRP lower-bound gaps

25 October 2026 · local branch `codex/extension-pilot`.

All three original 40-customer pilot instances now have numerical optimality certificates over their saved route pools. A new Benders variant finds better solutions and closes its own bounds within the original 20-second allowance. A separate enumeration of all 256 depot patterns per instance confirms the results without using the Benders master or saved cuts.

| Instance | Previous repair solution | Certified optimum | New method time | Previous repair excess cost |
| --- | ---: | ---: | ---: | ---: |
| 40 customers, seed 0 | 116.53296 | 94.68506 | 3.119 s | 23.07% |
| 40 customers, seed 1 | 112.45189 | 86.62911 | 1.179 s | 29.81% |
| 40 customers, seed 2 | 118.73930 | 86.01309 | 2.169 s | 38.05% |

“Excess cost” is `(previous cost − optimum) / optimum`. It uses a different denominator from the pilot's certified gap `(upper − lower) / upper`. Numerical optimality means relative bound gap at most 1e-7, not symbolic exact arithmetic. The observed final bound differences are below 0.000001 in objective units.

## What changed

The earlier methods stalled on four-depot choices. Those depots provide eight vehicles, and 40 customers require eight five-customer routes. The search therefore needs an exact partition of customers into whole supplied routes, with at most two routes from each depot. The generic integer solver did not find these partitions within the allotted time.

A small initial exact-cover probe found feasible partitions for all three stalled choices in under half a second each. Those plans alone reduced the certified gaps to 3.6–5.1%, using the existing lower bounds. Extending the probe to minimize route cost made it possible to finish integer recourse and strengthen the Benders master with certified integer costs.

The new `integer_benders_exact_cover` variant retains LP-first ordering, the same depot master, and the same pointwise integer-cut formula as before. The substantive change is its integer-recourse engine:

1. Select the unserved customer with the fewest currently usable routes.
2. Branch over routes serving that customer; remove routes that overlap customers already served.
3. Enforce each depot's remaining vehicle allowance and prune impossible capacity totals. A route too short to cover the remaining customers with the remaining fleet is inadmissible.
4. Use LP dual values to bound the cheapest possible completion of a partial route selection; explore promising routes first.
5. Return an optimum only after exhausting all remaining possibilities or pruning them using valid bounds. At timeout, retain checked feasible solutions and a conservative lower bound, without adding an integer optimality or infeasibility cut.

The selected fixed-depot problems took around 200,000–230,000 search nodes each at 40 customers. Benders needed three completed integer subproblems for seed 0, one for seed 1, and two for seed 2. Cheap LP cuts and master bounds handled the remaining depot choices. Good solutions alone would not have been sufficient for this conclusion.

## Why the lower bounds are valid

For fixed opening vector y, an LP dual satisfies `beta <= 0` and `sum(alpha[i] for i in route r) + beta[depot(r)] <= cost[r]`. For a partial solution, a lower bound on its completed route cost is:

`cost already selected + sum(alpha over uncovered customers) + sum(beta[j] * remaining_fleet[j])`.

The bound is valid because every completion covers each remaining customer exactly once and uses at most the remaining fleet. Choosing a route increases this expression by that route's reduced cost. No cost from a merely feasible incumbent is substituted for a lower bound.

The implementation conservatively repairs dual residuals: clip beta to nonpositive values and shift alpha down enough to satisfy every explicit route inequality. It subtracts a numerical guard during pruning and when reporting a completed recourse lower bound. The old pointwise cut `theta >= q * (1 − Hamming(y, ybar))` uses this certified q. All routing costs are nonnegative, so the cut remains valid at every other binary opening vector. Exhaustive small-instance tests check these cuts against a separate partition oracle.

## Independent check of the depot search

The `certify` command starts with a verified incumbent and examines every binary opening vector. It does not use the Benders master or its saved cuts. A pattern is ruled out by insufficient fleet, opening cost alone, an independently solved LP bound, or a completed exact-cover search.

| 40-customer seed | Insufficient fleet | Opening-cost bound | LP bound | Exact route searches | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 93 | 134 | 26 | 3 | 256 |
| 1 | 93 | 132 | 30 | 1 | 256 |
| 2 | 93 | 142 | 19 | 2 | 256 |

This audit took an additional 3.76, 1.98, and 2.56 seconds respectively. Audit time is separate from benchmark timing. It independently checks the depot-search logic but **shares the exact-cover engine**, so it is not a wholly independent second optimization implementation. That engine is separately checked against an exhaustive partition oracle on small instances, including deliberately invalid duals, infeasibility, and interrupted search.

The certificate files record every pattern and its reason/bound, LP dual values where used, exact-search results, verified candidate decisions, input hashes, and an auditing-source snapshot. An interrupted enumeration returns unresolved rather than claiming a global certificate. A suboptimal candidate produces a verified counterexample.

## Experiment and validation

The `lrp_pilot_bounds.json` campaign ran 27 method/instance pairs: full MIP, LP-first plus repair, and exact-cover Benders on the nine original 16-, 28-, and 40-customer inputs. Every input and shared witness matches the first scaling campaign; each method receives one thread and the same 20-second total budget. Algorithms execute frozen source copies.

- All 27 outputs pass independent route/coverage/fleet/objective verification; 21 runs prove optimality and six return feasible solutions.
- Exact-cover Benders proves all nine optima. The other methods prove the six smaller cases but remain unresolved at 40 customers. The full MIP's large-case feasible costs are 116.98, 110.62, and 110.88 in the matched comparison.
- The pattern-enumeration audit certifies all nine exact-cover results, with no cross-method bound contradictions.
- Full network-disabled test suite: **743 passed, 22 skipped**. Scoped lint/format checks pass; all 81 historical dataset hashes match. Original paper solvers and results are unchanged.

The backend is HiGHS for LP/master optimization plus a custom Python exact-cover engine for integer recourse. This is a reference-algorithm comparison, not an agent evaluation or reproduction of Gurobi paper timings. The benchmark source snapshot predates the separate enumeration-audit module; each artifact records the code it actually used.

## What this establishes—and its limits

We now know the optima of these three saved 40-customer problems and can measure solver or future-agent quality against them. The earlier long gaps reflected an implementation bottleneck; the cases are much easier for a solver that exploits their exact-cover structure.

All nine scaling route pools are sampled. These certificates apply to those supplied routes, not to every possible tour, capacity-constrained LRP, or VRPTW. Fleet tightness is particularly helpful here. Before selecting a larger benchmark, vary fleet slack and route density, test unseen seeds, and preserve the distinction between optimization over supplied routes and route generation. This result is useful evidence for Sikai's efficiency direction, but a general speedup claim would require broader matched experiments.

At the time of these experiments, no cluster run or provider call was made.
