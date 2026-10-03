# Without reflection: 1 coding round(s)

- Instance: `ufl_small_seed000` (BD / UFL); model: `gpt-5.6-luna`.
- Protocol: 60 seconds per entire solver execution, one solver thread, at most ten attempt slots, and a 1% verified gap threshold.
- This is a selected real trajectory from the Luna sweep. It contains 1 actual coding attempt(s); later slots in [instance_summary.json](ufl_small_seed000/instance_summary.json) reuse the final success.
- Host paths and local license notices were sanitized; raw CLI event streams and stderr were omitted. Artifact hashes and the retained scope are recorded in [provenance.json](provenance.json).

| Round | Status | E | V | Q | Objective | Gap |
| ---: | --- | --- | --- | --- | ---: | ---: |
| [1](ufl_small_seed000/attempts/attempt_01/summary.json) | ok | ✓ | ✓ | ✓ | 109.113932 | 0.00% |

## Read the trajectory

- Start with [prompt.txt](ufl_small_seed000/prompt.txt) and [data.json](ufl_small_seed000/data.json).
- In each [attempt folder](ufl_small_seed000/attempts/), read `agent_prompt.md` → `solver.py` → `output.json` → `exec.log` → `alg_verifier.md` → `summary.json`.
- A failed reflected attempt adds `debug.md`; the next `agent_prompt.md` embeds previous solver code and reflection notes. Each generation, review, and reflection uses a fresh Codex session.
- Invocation and metrics JSON files record the requested model, scoped-session settings, elapsed time, and token usage. Reused success slots do not invoke Codex again.
- The monolithic reference objective is 109.113932; the canonical reference is in [data/](../../../data/bd/ufl/ufl_small_seed000/gold_monolithic.json). It was used for evaluation and was not supplied in the generation prompt.

## Run a fresh experiment

```bash
python -m src.02_exp_cli --suite bd --problem ufl --instance data/bd/ufl/ufl_small_seed000 --model gpt-5.6-luna --max-attempts 10 --solver-timeout 60 --solver-threads 1
```

- This command runs from the repository root and creates a new ignored output directory. It uses your Codex authentication and solver license; stochastic generation can produce a different trajectory.
