# LRP Benders follow-up: solve ordering and feasible-plan repair

25 October 2026. Local follow-up to the first LRP calibration on `codex/extension-pilot`.

Later work: `docs/lrp-lower-bounds.md` closes the three 40-customer gaps using custom exact-cover recourse. This document preserves the earlier LP-first/repair comparison.

The follow-up completed **78 solver runs**: 36 in the four-method LP-first comparison, 18 in the paired repair comparison, and 24 timing-repeat runs. Every returned solution passed independent verification; 54 runs proved optimality numerically and 24 returned feasible solutions with unresolved gaps. No worker errors, process-watchdog terminations, or cross-campaign bound contradictions occurred.

## Changes

The original three algorithms remain available with unchanged default selection. Two separately named variants make the effect of each change measurable:

1. `integer_benders_lp_first` sends every violated LP cost cut back to the master immediately. It starts whole-route recourse only when the current depot choice's LP estimate is consistent with its solved LP. The model and valid cut families are unchanged.
2. `integer_benders_lp_first_repair` also caps the initial whole-route attempt at two seconds. On timeout it tries allowing one additional depot at a time, ordered by opening cost, for at most two seconds each. It closes unused depots in returned solutions, verifies them independently, then restarts recourse for the original depot choice with the remaining budget. These auxiliary solves only produce feasible incumbents: no global lower bound, cut, or exclusion is inferred from them.

Both variants obey the same total per-method deadline, including repair work. Checkpoints and traces distinguish deferred integer calls, repair attempts, and incumbent sources. Two seconds is an exploratory fixed policy, not a tuned parameter.

## First comparison: LP-first

The `lrp_pilot_lp_first.json` campaign reran all three original methods and LP-first on the nine original scaling inputs: three seeds each at 16, 28, and 40 customers. Inputs match the saved first-pilot files byte for byte. All methods receive the same generated witness, one solver thread, and a 20-second budget.

| 28-customer seed | Original integer Benders | LP-first | Full MIP | Whole-route calls: original → LP-first |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.217 s | 0.234 s | 0.486 s | 1 → 1 |
| 1 | 6.898 s | 1.795 s | 3.706 s | 4 → 1 |
| 2 | 4.842 s | 1.964 s | 3.940 s | 3 → 1 |

All entries in this table prove the same instance-specific optimum. LP-first's observed runtime ratios are 3.84× and 2.47× faster than original integer Benders on seeds 1 and 2; it is slightly slower on seed 0. The trace explains the useful cases: it collects the needed LP cuts without repeatedly solving the harder integer problem along the way. All three 16-customer cases also solve, with small overhead from the extra master/LP round.

At 40 customers, LP-first improves the lower bounds from 75.0, 66.0, and 67.0 to approximately 92.11, 83.64, and 82.82. It still cannot improve the supplied feasible witnesses within 20 seconds. Gaps fall from 65.19–67.79% to 57.23–60.18%, while full MIP retains substantially better incumbents and 20.98–24.88% gaps. A smaller gap here initially reflects a stronger lower bound, not a better delivery plan.

Two additional runs of each original-integer/LP-first pair on all six smaller inputs check timing variability. All 24 repeated solves prove the same optima. Across the first comparison and these two repeats, median times are 6.966 → 1.788 seconds (3.90×) for medium seed 1 and 4.861 → 1.965 seconds (2.47×) for medium seed 2. The easy medium seed 0 is 0.218 → 0.235 seconds, and the three small inputs also show minor overhead. Repeats confirm the observed effect on these fixed inputs; they do not establish general performance across unseen instances. Use `lrp_pilot_lp_first_repeat.json` with fresh output directories to repeat this workload.

## Second comparison: repair

The `lrp_pilot_repair.json` campaign pairs LP-first with LP-first plus repair on the same nine inputs and budgets. Both methods prove all six smaller optima. Repair improves the incumbent on all three 40-customer inputs:

| 40-customer seed | LP-first cost | LP-first + repair cost | Full MIP cost | Repair gap | Full MIP gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 215.4265 | 116.5330 | 116.9803 | 20.96% | 20.98% |
| 1 | 195.5475 | 112.4519 | 110.6235 | 25.62% | 23.97% |
| 2 | 207.9857 | 118.7393 | 110.8759 | 30.25% | 24.88% |

Costs are feasible objective values; lower is better. No entry in this table is claimed to be the optimum. Full MIP values come from the adjacent four-method campaign on the identical data, backend, threads, and time budget. LP-first and repair are paired within the repair campaign.

Repair reduces cost by 45.91%, 42.49%, and 42.91% relative to the starting witnesses retained by LP-first. Its best plans open five depots instead of the witness's eight. It competes with full MIP on seed 0 but remains worse on seeds 1 and 2. All repair attempts share the 20-second budget; the improvement is not purchased by increasing time. Lower bounds stay equal to LP-first's bounds, so this second improvement is entirely in the feasible plan.

Several repair subproblems time out after finding a feasible solution. Their incumbents are still useful, but their costs are never promoted to exact recourse cuts or global lower bounds. The untouched master-choice solve is restarted with the remaining time and still unresolved on the large cases.

## Verification and scope

The full network-disabled suite passes **729 tests, with 22 environment/data-dependent skips**. Added coverage checks both variants against exhaustive partition/depot oracles, verifies all admitted cuts on tiny inputs, exercises LP-feasible/integer-infeasible recourse, checks that LP-first really defers integer work, and simulates an initial recourse timeout followed by repair and a return to the original depot choice. A separate test checks that repair changes neither the lower bound nor the cut list and closes unused depots.

The comparison uses local SciPy/HiGHS, not Gurobi or agents. All scaling inputs use sampled route pools. These are exploratory measurements, not statistically established speedups or unrestricted-routing results. Forty-customer optima remain unknown. The original paper solvers, dataset, and results remain untouched. Machine-readable comparisons verify data hashes, shared witnesses, budgets, thread counts, executed-source snapshots, incumbent feasibility/objective, and consistency of all lower bounds with every verified incumbent across campaigns.

Run commands and exact algorithm details are in `src/pilots/lrp/README.md`. Campaign manifests preserve each implementation snapshot, executed-source hashes, configuration, data, package versions, and solver binary identity. The first LP-first campaign intentionally predates the repair implementation; its workers executed their frozen copy while the second variant was developed.

At the time of these experiments, the work was local and had not been shared.

## What to investigate next

Keep these variants explicit and preserve the original baselines. LP-first is a useful simple improvement where early integer solves are redundant; the repair variant addresses poor incumbents when the selected depot plan is difficult. Neither resolves the weak optimality certificate at 40 customers. Next work should target that certificate—better integer-recourse bounds or valid problem-specific cuts—while checking whether sparse sampled route pools are creating artificial difficulty. Test on new calibration seeds before freezing a held-out evaluation set. Larger compute budgets alone would not answer those algorithm and instance-design questions.
