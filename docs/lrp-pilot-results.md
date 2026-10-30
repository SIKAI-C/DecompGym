# Local LRP pilot: first working results

25 October 2026 · branch `codex/extension-pilot` · local reference solvers only.

Follow-up: `docs/lrp-benders-improvements.md` records 78 additional runs testing LP-first Benders and a feasible-plan repair heuristic. This document preserves the first 69-run calibration.

The pilot works locally. It generates harder location-routing inputs, solves them three ways, checks the returned decisions independently, and saves the evidence needed to inspect each run. Across 69 runs on 23 input settings, every returned solution passed verification. There were 49 numerical optimality certificates, 20 feasible results with unresolved gaps, no worker errors, and no cross-method bound contradictions.

This is the first calibration step toward Sikai's scaling experiments. It does not yet show that decomposition beats a strong monolithic solver on difficult instances, and it does not evaluate agents.

## What we built, in plain language

The model decides which depots to open and which delivery routes to operate. Every customer must be visited exactly once, and an open depot has a limited number of vehicles. The objective adds depot-opening and route costs.

1. **Full MIP (`monolithic`)** makes the depot and route decisions together.
2. **LP-cut Benders (`lp_benders`)** chooses depots first, then uses a version of the routing problem that permits fractions of routes to estimate cost. Separate whole-route solves supply feasible delivery plans. Fractional estimates may leave an optimality gap even when the best delivery plan has been found.
3. **Integer-cut Benders (`integer_benders`)** also learns certified whole-route costs for depot choices. Its extra cuts can close gaps that fractional-route cuts leave open. These are simple pointwise cuts, a correctness reference rather than a highly tuned logic-based Benders algorithm.

The independent checker reconstructs customer visits and depot fleet use from explicit routes and recalculates the objective. It does not reuse the solver's incidence-matrix constraints. Tiny tests additionally enumerate every feasible partition and depot pattern to check optimal values and the validity of all recorded cuts.

## What the experiments found

| Experiment | Inputs / runs | Finding |
| --- | ---: | --- |
| Smoke, 6 and 12 customers | 4 / 12 | Full MIP and integer-cut Benders prove all four optima. LP-only Benders leaves small gaps on both 12-customer inputs. |
| Scale, 16 customers | 3 / 9 | Full MIP and integer-cut Benders prove all three optima. LP-only gaps are 0.50–0.76%. |
| Scale, 28 customers | 3 / 9 | Full MIP and integer-cut Benders again prove all three optima. LP-only gaps are 0.63–1.59%. Integer Benders takes 0.21–7.30 seconds; full MIP takes 0.49–3.89 seconds. Neither consistently wins. |
| Scale, 40 customers | 3 / 9 | All methods reach their 20-second budgets without proving optimality. Full MIP finds better solutions with 20.98–24.88% certified gaps; both Benders variants retain 65.19–67.79% gaps. |
| Original paper LRP inputs | 9 / 27 | All 27 returned objectives exactly match the saved monolithic references. Full MIP and integer Benders prove all nine; LP-only Benders proves seven and leaves gaps of 0.27% and 1.29% on two. |
| Complete route-pool control, 16 customers | 1 / 3 | All methods find cost 42.3175; full MIP and integer Benders prove it. LP-only gap is 0.284%. |

At 40 customers, Benders spends almost the entire budget trying to find a whole-route solution for its first LP-feasible depot choice. Neither Benders variant improves the supplied feasible starting plan on these three cases. The full MIP can change depot choices within its search and finds much cheaper solutions. This is a useful implementation bottleneck, not evidence that decomposition inherently cannot help.

The same 16-customer geography has optimum 43.9336 with 800 sampled routes and 42.3175 with all 10,064 routes of up to four stops. Coordinates, opening costs, fleet limits, and route costs for shared routes were checked identical. This demonstrates why route-pool scope must be explicit: a sampled-pool optimum is an optimum over the routes supplied. The complete pool also leaves an LP integrality gap, so that phenomenon is not confined to the sampled generator.

## How to reproduce and inspect

Use the repository's Python 3.12.11 environment and the commands in `src/pilots/lrp/README.md`. The checked-in configurations reproduce the smoke, scale, original-input, and complete-pool workloads. Always choose a new output directory.

Each campaign writes `report.md`, `summary.csv`, `summary.png`, `results.json`, and `manifest.json`. Per-instance/method directories contain decisions, independently checked feasibility, lower/upper bounds, cuts, LP/MIP status records, runtime, peak memory, and logs. The runner freezes source/configuration/data and checks that workers execute those frozen source files. It records package versions and the HiGHS binary hash.

The original exploratory smoke run predates enforcement of frozen-source execution. It retained source snapshots, and no implementation changes were made during that run. Later campaigns record and verify the actual executed-source hashes. Small runner robustness changes and a single-stop generator fix followed the experiments; the final integration tests exercised that final code. Source snapshots identify the exact experiment versions rather than implying every run used the final commit.

The 69 runs consumed approximately 218 seconds of measured method time in total, excluding instance generation, interpreter startup, and reporting. They ran sequentially with one solver thread. This is exploratory calibration with one timing per instance/method, not a robust timing study. No hour-long runs were needed.

## Validation and limits

- Full network-disabled test suite: **717 passed, 22 skipped**. The 27 added pilot cases include independent exact oracles, integer infeasibility despite LP feasibility, a genuine LP integrality gap, timeout behavior, rejected malformed outputs, and actual subprocess execution.
- Scoped Ruff lint/format checks pass. All 81 historical dataset hashes remain unchanged. The existing paper solvers, dataset, configurations, and published results were preserved.
- Experiments used **SciPy 1.18.1 / HiGHS 1.12.0**, Python 3.12.11, one solver thread, on the local Apple Silicon machine. No Gurobi license was used. These timings cannot be presented as reproductions of the paper's Gurobi results.
- Generated inputs retain a feasible planted partition. The same witness is available externally to every method, but is not installed as a solver warm start. Imported paper inputs do not receive a generated witness.
- This remains the route-enumerated, stop-limited LRP model. It adds neither capacity/time windows nor route pricing. Most scaling pools are sampled; sparse random route pools may produce artificial set-partitioning difficulty. More realistic route generation and held-out instances are necessary before calling this a mature benchmark.
- Integer cuts use valid lower bounds from completed recourse solves. Unfinished recourse incumbents never become exact cuts. Reported optimality is numerical, with relative bound tolerance 1e-7.
- The existing agent sandbox/runtime and Gurobi licensing requirements still need separate work. This pilot does not require those components or Sikai's raw campaign folders.

## Next experiment worth doing

Keep the current three methods as frozen baselines. Test a separately named Benders variant that first strengthens the inexpensive LP master bound before spending its entire budget on a whole-route subproblem. Then consider bounded recourse effort, reusable feasible-route heuristics, and stronger problem-specific cuts. Validate each change against the tiny exhaustive oracle and compare on the same inputs and budgets, including failures.

Before increasing runtime or moving to a cluster, sweep intermediate sizes and route density independently, add realistic geography/fleet settings, and select a calibration range where feasible solutions and meaningful bounds can be obtained locally. Freeze separate evaluation seeds after calibration. A matched Gurobi comparison and agent evaluation are subsequent stages.

These local experiments used no cluster job or provider call.
