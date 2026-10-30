# Local LRP calibration pilot

Start with [the sharing overview](../../../docs/LRP-SHARING.md) for saved results, provenance, and limitations.

Run from the repository root, using its `.venv`. No API calls, credentials, Docker, or Gurobi license are needed for this pilot. SciPy supplies HiGHS. The original Gurobi implementations, fixed-81 dataset, paper configuration, and published results are unchanged.

```bash
.venv/bin/python -m pytest tests/test_lrp_pilot.py -q
.venv/bin/python -m src.pilots.lrp.run \
  --config configs/lrp_pilot_smoke.json --output outputs/lrp-smoke-001
.venv/bin/python -m src.pilots.lrp.run \
  --config configs/lrp_pilot_scale.json --output outputs/lrp-scale-001
```

Additional checked-in configurations are `lrp_pilot_paper_inputs.json` (the nine original LRP inputs; 10 seconds per method) and `lrp_pilot_pool_control.json` (a complete 16-customer route pool; 20 seconds per method). Local observations are recorded in `docs/lrp-pilot-results.md`.

`lrp_pilot_lp_first.json` compares the original three methods with `integer_benders_lp_first` on the same nine scaling inputs and 20-second budgets. `lrp_pilot_repair.json` compares LP-first with `integer_benders_lp_first_repair` on that same grid. The original three remain the default methods when `methods` is omitted.

`lrp_pilot_lp_first_repeat.json` repeats the original integer/LP-first comparison on the six smaller inputs. Run it into separate new directories for repeated timings. See `docs/lrp-benders-improvements.md` for the 78-run follow-up and its limits.

`lrp_pilot_bounds.json` compares full MIP, LP-first plus repair, and `integer_benders_exact_cover` on the same nine inputs and 20-second budgets. The exact-cover variant uses HiGHS for the master and LP recourse, and a custom Python search for integer recourse. See `docs/lrp-lower-bounds.md` for results and certificates.

```bash
.venv/bin/python -m src.pilots.lrp.run \
  --config configs/lrp_pilot_bounds.json --output outputs/lrp-bounds-001
.venv/bin/python -m src.pilots.lrp.certify \
  --campaign outputs/lrp-bounds-001 --output outputs/lrp-certificates-001 --seconds 30
```

The second command separately audits each exact-cover result by enumerating all depot patterns (at most 16 depots). It never reads the Benders cuts or calls the master. It shares the exact-cover engine and returns an unresolved result on timeout, or a verified counterexample if the candidate is suboptimal. Audit time is separate from the benchmark's method time.

Each output directory must be new. A run freezes its configuration, data, Python source files, source hashes, package versions, HiGHS native binary hash/version, and Git identity. Workers execute the frozen source, not the mutable checkout. `report.md`, `summary.csv`, and `summary.png` summarize results; `results.json` and per-method directories retain decisions, verification, cut coefficients, bounds, solver statuses, timings, checkpoints, and logs. An interrupted run retains completed cases, but automatic resume is not implemented; use a new directory to rerun.

For a network-disabled run on macOS, prefix the command with:

```bash
sandbox-exec -p '(version 1)(allow default)(deny network*)'
```

This optional wrapper is a local network restriction, not the paper's generated-code sandbox. The pilot runs trusted reference implementations and does not execute generated agent code.

## The model and the three methods

Open depots `y[j]` and select routes `lambda[r]`, both binary. Each customer appears on exactly one selected route. Depot j operates at most `M[j] * y[j]` routes. Minimize depot-opening plus route costs. The result is optimal over the supplied route pool; the pilot does not introduce vehicle capacity, time windows, or new route pricing.

- `monolithic`: HiGHS solves the full depot/route MIP in one model.
- `lp_benders`: a depot-opening master generates cuts from the continuous route-selection LP; fixed-opening binary recourse produces actual incumbents. It may stop with an unresolved integer gap.
- `integer_benders`: adds generic pointwise integer-recourse cuts and exclusions for depot plans with proved-infeasible integer recourse. It can close gaps that LP cuts leave open. This is a deliberately simple exact reference strategy, not a tuned state-of-the-art implementation.

Three optional variants keep the same model and cut validity rules:

- `integer_benders_lp_first`: whenever LP recourse generates a violated cost cut, immediately reoptimize the master. Whole-route recourse is attempted only once the selected plan's LP estimate is consistent. This changes solve order, without adding a new relaxation or cut family. Traces mark each deferred integer solve.
- `integer_benders_lp_first_repair`: also limits the first whole-route attempt at each selected plan to two seconds. If it times out, try opening each currently closed depot individually, in ascending opening-cost order, for at most two seconds each. Close unused depots in each returned solution and independently verify it. These auxiliary solves only supply incumbents; they add no cuts, exclude no choices, and do not change the global lower bound. Finally restart the original recourse solve using the remaining budget. All work shares the original deadline. Repair does not guarantee improvement and can consume time that would otherwise prove optimality.
- `integer_benders_exact_cover`: uses LP-first ordering, then solves fixed-depot integer recourse by a minimum-cost exact-cover search. It branches on the least-covered remaining customer, removes overlapping routes, enforces per-depot fleet limits, and prunes with conservative LP-dual lower bounds. Completed search supplies the same pointwise integer cuts used by the existing integer baseline; proved infeasibility supplies a no-good cut. Interrupted search supplies feasible incumbents but no new integer cut. This exploits the supplied route-enumerated model; it is not a general routing or branch-and-price solver.

All three use the same route data and one solver thread. Every generated instance supplies a verified feasible witness, available to all methods as an external incumbent. It is not a solver warm start. Existing imported data have no automatically constructed witness. Execution order rotates by instance.

### Why the cuts are valid

For open-depot vector `y`, LP recourse is `min c lambda`, with coverage `A lambda = 1`, fleet `B lambda <= M y`, and `lambda >= 0`. An optimal dual `(alpha, beta)` has unrestricted alpha, beta <= 0, and `A^T alpha + B^T beta <= c`. Thus `theta >= sum(alpha) + sum(beta[j] M[j] y[j])` is a valid lower bound on LP and integer recourse. Dual feasibility and primal/dual agreement are checked before admitting the cut.

When LP recourse is infeasible, phase I minimizes nonnegative uncovered-customer slacks. Its dual gives a feasibility cut `sum(alpha) + sum(beta[j] M[j] y[j]) <= 0`. It is necessary for zero-slack coverage. Unlike a hand-chosen big-M penalty, this is a separate feasibility problem.

For a binary vector `ybar`, let `q` be a valid lower bound on its integer recourse, obtained from a completed fixed-opening MIP. Since all recourse costs are nonnegative, `theta >= q * (1 - Hamming(y, ybar))` is valid: at ybar it requires theta >= q; at every other binary vector its RHS is nonpositive. An incumbent value from an unresolved MIP is never substituted for q. If integer recourse is proved infeasible, `Hamming(y, ybar) >= 1` excludes that vector. The current implementation waits for a completed recourse solve before adding integer cuts; tighter problem-specific cuts and partial-solve bounds are future improvements.

With finitely many opening vectors and completed master/recourse solves, these integer cuts provide an exact enumeration-based strategy. Real runs have time/iteration limits and may therefore return only feasible incumbents. Numerical optimality uses relative bound gap <= 1e-7, not symbolic proof. Tests compare all admitted cuts with an independent exhaustive partition oracle on tiny problems, including LP-feasible/integer-infeasible recourse.

HiGHS MIP dual bounds and LP marginals are consumed through SciPy's documented [MILP result fields](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html) and [LP marginal conventions](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ds.html). `threads=1` is forwarded to HiGHS by SciPy; only the warning about that forwarded option is suppressed. Pricing/exactness logic from the paper is not involved here.

## Generator controls

Each case specifies a unique `name`, `depots`, `customers`, `max_stops`, `routes_per_depot`, and a list of `seeds`. Optional `opening_cost_scale` changes opening/routing cost tradeoffs, and `vehicles_per_depot` changes fleet tightness. The default fleet allows spare capacity so location choices are not automatically forced by total fleet capacity.

All single-customer routes and a planted feasible partition are retained. Remaining sampled routes are drawn across route lengths, avoiding the original generator's short-route-first truncation. Exact Held-Karp dynamic programming finds a cheapest tour for each selected customer subset. If the route cap covers every subset up to `max_stops`, metadata say `complete_stop_limited`; otherwise they say `sampled_restricted`. The complete label applies only to this Euclidean stop-limited model. The independent output verifier recounts coverage and fleet from explicit route sequences, checks binary decisions, and recomputes cost.

To solve existing data without regeneration, use a case such as:

```json
{"name": "paper_small_0", "source": "data/bd/lrp/lrp_small_seed000/data.json"}
```

Top-level controls are `schema_version: 1`, `solver_seconds`, `max_iterations`, `methods`, and `cases`. Historical `data/`, `results/`, source, config, and Git directories are protected against use as output destinations.

Imported inputs use the default `seeds: [0]` only; this is a run identifier, not a regeneration seed. The imported filename retains its original instance/seed identity. A reporting failure leaves the manifest marked `report_failed`, even if solver runs already finished.

## Timing and what results can support

`method_seconds` includes input validation, matrix construction, master/subproblem solves, and checkpoint writes before finalization. JSON input loading, interpreter imports, and final serialization are included in `process_wall_seconds` instead. Instance generation has its own timer. Solver subcalls receive the remaining shared wall budget; each method also has a process watchdog with 10 seconds of import/serialization grace. At watchdog termination, any recovered result is labeled as a checkpoint and does not pretend to contain a final method timer. Peak RSS is the worker process high-water mark where available.

Both good incumbent quality and certificate quality matter. A feasible solution can be optimal in hindsight while its own lower bound remains weak. A time-limited monolithic incumbent is not automatically the reference optimum. Cross-method bounds are checked against verified incumbents to detect contradictions. These exploratory single-seed/per-setting measurements do not establish a robust speedup; no hour-long or agent campaign is run by the supplied default configurations.

## Next useful changes

Use the local calibration to choose a pilot difficulty range, then freeze separate evaluation seeds. Add richer geometry/demand/fleet settings if simple size increases mainly stress set partitioning. Profile phase I, LP cuts, and integer recourse before tuning. Once an authorized Gurobi license/runtime is available, add matched Gurobi runs on the same frozen input. Any eventual agent campaign requires its own protocol and runtime checks.
