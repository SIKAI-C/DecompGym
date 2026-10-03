# GPT-5.6 Luna results

- Model: `gpt-5.6-luna`, requested explicitly through Codex CLI.
- Dataset: 81 instances across nine problem types; each type has three seeds at each of small, medium, and large. Each reflection setting contains 81 runs, for 162 completed runs total.
- Protocol: at most ten coding attempts per instance, a 60-second deadline for the entire generated solver, one solver thread, and a 1% verified objective-gap threshold. Generation, review, and reflection have separate 900-second deadlines.
- The first 48 completed runs used one worker; the remaining 114 used two. Worker count changes concurrency, while the solver deadline and thread limit stayed fixed.
- All saved outputs were independently audited against canonical data. Success is cumulative by attempt budget; later unused slots retain earlier qualifying success.
- Attempt one has no reflection context in either setting. This single cohort records observed outcomes and repairs; it does not isolate a causal advantage for reflection.

## By problem type

E / V / Q means execution / verification / quality. The first-100% columns give attempt numbers, not percentages. Every problem type has **N = 9 per setting**.

| Group | N per setting | Without: E / V / Q @1 (%) | With: E / V / Q @1 (%) | Without: E / V / Q first 100% | With: E / V / Q first 100% |
| --- | ---: | --- | --- | --- | --- |
| BD · FCTP | 9 | 100.0 / 66.7 / 55.6 | 77.8 / 44.4 / 44.4 | 1 / 2 / 3 | 2 / 4 / 5 |
| BD · KM | 9 | 100.0 / 100.0 / 100.0 | 100.0 / 100.0 / 100.0 | 1 / 1 / 1 | 1 / 1 / 1 |
| BD · LRP | 9 | 100.0 / 55.6 / 55.6 | 100.0 / 22.2 / 22.2 | 1 / 4 / 4 | 1 / 4 / 4 |
| BD · TSSO | 9 | 100.0 / 55.6 / 55.6 | 100.0 / 100.0 / 100.0 | 1 / 3 / 3 | 1 / 1 / 1 |
| BD · UFL | 9 | 100.0 / 88.9 / 88.9 | 100.0 / 88.9 / 88.9 | 1 / 2 / 2 | 1 / 2 / 2 |
| CG · BP | 9 | 100.0 / 77.8 / 77.8 | 100.0 / 100.0 / 100.0 | 1 / 2 / 2 | 1 / 1 / 1 |
| CG · CSP | 9 | 100.0 / 77.8 / 77.8 | 100.0 / 88.9 / 77.8 | 1 / 2 / 2 | 1 / 2 / 2 |
| CG · GAP | 9 | 100.0 / 77.8 / 77.8 | 100.0 / 88.9 / 88.9 | 1 / 5 / 5 | 1 / 2 / 2 |
| CG · SHIFT | 9 | 100.0 / 100.0 / 100.0 | 100.0 / 100.0 / 100.0 | 1 / 1 / 1 | 1 / 1 / 1 |

![Quality success by problem type](../../assets/results/luna_60s/problem_attempt_curves.png)

The per-type figure shows Q for both policies; [attempt_breakdowns.csv](attempt_breakdowns.csv) contains all E/V/Q counts and rates for every round and group.

## Benders decomposition and column generation

| Group | N per setting | Without: E / V / Q @1 (%) | With: E / V / Q @1 (%) | Without: E / V / Q first 100% | With: E / V / Q first 100% |
| --- | ---: | --- | --- | --- | --- |
| Benders decomposition | 45 | 100.0 / 73.3 / 71.1 | 95.6 / 71.1 / 71.1 | 1 / 4 / 4 | 2 / 4 / 5 |
| Column generation | 36 | 100.0 / 83.3 / 83.3 | 100.0 / 94.4 / 91.7 | 1 / 5 / 5 | 1 / 2 / 2 |

![Cumulative E/V/Q by decomposition method](../../assets/results/luna_60s/suite_attempt_curves.png)

## By instance size

Sizes are relative to each problem's generator. Each size contains **N = 27 per setting**.

| Group | N per setting | Without: E / V / Q @1 (%) | With: E / V / Q @1 (%) | Without: E / V / Q first 100% | With: E / V / Q first 100% |
| --- | ---: | --- | --- | --- | --- |
| SMALL | 27 | 100.0 / 81.5 / 81.5 | 100.0 / 88.9 / 85.2 | 1 / 5 / 5 | 1 / 4 / 4 |
| MEDIUM | 27 | 100.0 / 66.7 / 66.7 | 96.3 / 77.8 / 77.8 | 1 / 3 / 3 | 2 / 4 / 5 |
| LARGE | 27 | 100.0 / 85.2 / 81.5 | 96.3 / 77.8 / 77.8 | 1 / 4 / 4 | 2 / 3 / 3 |

![Cumulative E/V/Q by instance size](../../assets/results/luna_60s/size_attempt_curves.png)

## Overall

| Group | N per setting | Without: E / V / Q @1 (%) | With: E / V / Q @1 (%) | Without: E / V / Q first 100% | With: E / V / Q first 100% |
| --- | ---: | --- | --- | --- | --- |
| OVERALL | 81 | 100.0 / 77.8 / 76.5 | 97.5 / 81.5 / 80.2 | 1 / 5 / 5 | 2 / 4 / 5 |

![Overall cumulative E/V/Q](../../assets/results/luna_60s/overall_attempt_curves.png)

- Both settings first attain E/V/Q all at 100% at **attempt 5**. Without reflection, first full E/V/Q occur at 1/5/5; with reflection, at 2/4/5.
- Without reflection used 110 real coding attempts; with reflection used 105 coding attempts plus 24 reflection calls. Reused success slots are not extra coding calls.
- Q requires verification and at most 1% objective gap against the monolithic reference; it does not mean every qualifying result is exactly optimal.
- [instances.csv](instances.csv) contains one row per run, including the first passing round for E, V, and Q. [attempt_curves.csv](attempt_curves.csv) contains overall counts and rates, and [attempt_curves.json](attempt_curves.json) contains all grouped curves. [configuration.json](configuration.json) records the public experiment protocol.
- Two portable teaching trajectories are retained: [one round without reflection](../../outputs/examples/without_reflection/README.md) and [five rounds with reflection](../../outputs/examples/with_reflection/README.md). Full private working outputs remain ignored.

## Run your own comparison

```bash
python -m src.04_run_experiments --model gpt-5.6-luna --workers 2 --solver-threads 1 --solver-timeout 60 --max-attempts 10 --gap-threshold 0.01 --output-root outputs/experiments/my_luna_run
python -m src.05_report --input-root outputs/experiments/my_luna_run
```

- These commands use your Codex authentication and an available model with the requested identifier. Sampling can produce different trajectories and results.
- Your report stays under the experiment directory. Published results here are a selected snapshot; report generation does not update repository documentation.
