# DecompBench

## Experiment Scripts

### Overview and Intended Use
- `src/02_run_exp1.py`: BD + CG, no reflection/history. Each attempt is independent and only sees `data.json` + `prompt.txt`.
- `src/02_run_exp2.py`: BD + CG, with reflection/history. Later attempts see prior `solver.py`, `debug.md`, `summary.json` as read-only context.
- `src/02_run_exp3.py`: BD only, no reflection/history, lazy-constraint requirement appended to `prompt.txt`.
- `src/02_run_exp4.py`: BD only, with reflection/history, lazy-constraint requirement appended to `prompt.txt`.

### Current Logic Notes (Potential Pitfalls)
- `exp1`/`exp3`: no `debug.md` reflections are generated; if you expect reflection content, use `exp2`/`exp4`.
- `exp2`/`exp4`: prior attempts are exposed as read-only context; this can bias later attempts toward earlier mistakes.
- `exp1`/`exp3`/`exp4`: attempt directories are created up front, but `solver.py`, `exec.log`, and `debug.md` are created only when an attempt runs (skipped attempts do not create files).
- `exp3`/`exp4` are BD-only. Using `--suite cg` will fail because their suite is hard-coded to BD.

### Usage

Run a single instance:
```bash
python -m src.02_run_exp1 -i z_artifacts/benchmarks/cg/shift/shift_small_seed004
python -m src.02_run_exp2 -i z_artifacts/benchmarks/cg/shift/shift_small_seed004
python -m src.02_run_exp3 -i z_artifacts/benchmarks/bd/km/km_small_seed000
python -m src.02_run_exp4 -i z_artifacts/benchmarks/bd/km/km_small_seed000
```

Run a full suite (use `--suite bd` or `--suite cg` where supported):
```bash
python -m src.02_run_exp1 --suite cg --run-all
python -m src.02_run_exp2 --suite bd --run-all
python -m src.02_run_exp3 --suite bd --run-all
python -m src.02_run_exp4 --suite bd --run-all
```

Outputs:
- `exp1`: `z_artifacts/output/exp1/...`
- `exp2`: `z_artifacts/output/exp2/...`
- `exp3`: `z_artifacts/output/exp3/...`
- `exp4`: `z_artifacts/output/exp4/...`