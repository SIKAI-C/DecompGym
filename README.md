# DecompGym

DecompGym is an executable environment for testing whether coding agents can design and implement **decomposition algorithms**. Each task asks an agent to write a standalone `solver.py` that solves an optimization instance with **Benders decomposition** or **column generation**, using Gurobi. The environment runs the solver, checks its solution, has a second model audit whether the code really uses the requested method, and compares the objective with a reference.

This repository holds the benchmark, the harness that ran the paper's experiments, and the code that rebuilds every table and figure of the paper. (The repository and its paths still use the earlier name, DecompBench.)

![Overview: nine problem families, a task specification, a coding agent writing solver.py, and the three evaluation criteria](results/figures/benchmark_overview.png)

## Benchmark

| Method | Families |
| --- | --- |
| Benders decomposition (45 instances) | uncapacitated facility location (UFL), fixed-charge transportation (FCTP), k-median (KM), location routing (LRP), two-stage stochastic production (TSSO) |
| Column generation (36 instances) | bin packing (BP), cutting stock (CSP), generalized assignment (GAP), shift scheduling (SHIFT) |

Each family has three sizes × three seeds, so 81 fixed instances in [data/](data/README.md). Every task prompt includes the instance data, the output contract and a **method contract**: the solver must use the named decomposition, may not solve a separate monolithic model of the full problem, and may report `optimal` only with a proof of global optimality.

An attempt is scored by three criteria in order:

- **E — executable**: `solver.py` finishes within 60 seconds on one solver thread and writes a parseable `output.json`.
- **V — verified**: an independent checker confirms feasibility and recomputes the objective, and a **method judge** (GPT-6 Astra, high effort) reads the source and accepts that the decomposition is genuinely used.
- **Q — quality**: verified, and within 1% of the reference objective.

Q@k is the share of instances solved by at least one of the first k attempts. After a failed attempt, two policies continue from the same first attempt: **plain regeneration** (P) starts again from the task, and **reflection** (R) first asks the agent to diagnose the failure from the code, logs and judge notes, then revises. Each policy gets up to 10 attempts; an agent call may take up to 900 seconds.

The paper evaluates 12 configurations, all at medium effort: five GPT models through Codex, six Claude models through Claude Code, and Gemini 3.8 Flash through Antigravity — 972 paired runs in total.

## Results

| Configuration | Q@1 | Q@2 P | Q@2 R | Q@10 P | Q@10 R |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-6 Astra | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| GPT-6 Sol | 97.5% | 98.8% | 100.0% | 100.0% | 100.0% |
| GPT-6.1 Sol | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| GPT-6 Luna | 77.8% | 88.9% | 87.7% | 95.1% | 98.8% |
| GPT-5.6 Luna | 67.9% | 90.1% | 93.8% | 96.3% | 100.0% |
| Claude Opus 5.5 | 96.3% | 100.0% | 100.0% | 100.0% | 100.0% |
| Claude Sonnet 5.5 | 95.1% | 100.0% | 100.0% | 100.0% | 100.0% |
| Claude Haiku 5.5 | 87.7% | 100.0% | 98.8% | 100.0% | 100.0% |
| Claude Opus 5 | 86.4% | 95.1% | 100.0% | 100.0% | 100.0% |
| Claude Sonnet 5 | 56.8% | 67.9% | 88.9% | 97.5% | 98.8% |
| Claude Fable 5 | 97.5% | 100.0% | 100.0% | 100.0% | 100.0% |
| Gemini 3.8 Flash | 90.1% | 95.1% | 100.0% | 100.0% | 100.0% |
| **All 972 pairs** | **87.8%** | **94.7%** | **97.4%** | **99.1%** | **99.8%** |

- 853 of 972 first attempts meet all three criteria. A single reflection retry solves 94 of the 119 failures, against 67 for plain regeneration.
- The method audit matters: 93 rejected attempts were feasible and within 1% of the reference, so a check of the output alone would have accepted them. Most of them report `optimal` without a proof; Claude Sonnet 5 accounts for 92 of the 96 such claims.

![Quality success Q@k by attempt budget, without (left) and with (right) reflection](results/figures/attempt_quality.png)

All tables, figures and the prompt examples of the paper are in [results/](results/README.md). One complete trajectory, the paper's case study, is in [outputs/examples/](outputs/examples/claude_sonnet_5_csp_medium_seed000/README.md).

## Quick start

Requirements: macOS on Apple silicon (the solver sandbox uses `sandbox-exec`), Python 3.12.11 and Gurobi 11.0.3 with a license. Docker is needed only to run agents; rebuilding the paper's figures also needs `pdflatex` (TikZ, helvet, inconsolata, fontawesome5) and Ghostscript (`gs`).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q                                   # unit tests; no model calls
python src/01_gen.py -s all -m validate -vl simple    # check the 81 instances and their references
```

The reference solvers can also be called from Python:

```python
from src.utils.api import solve, verify
result = solve("bd", "ufl", "data/bd/ufl/ufl_small_seed000/data.json")                       # Benders reference
print(verify("bd", "ufl", "data/bd/ufl/ufl_small_seed000/data.json", result["output"]))
```

[docs/problems.md](docs/problems.md) explains each family: the original model, the master and subproblem or pricing problem, and what is passed back, with links to the monolithic and decomposition implementations in [src/bd/](src/bd/) and [src/cg/](src/cg/).

## Running a model

Agents run inside a Docker image with the pinned CLIs (Codex 0.162.1, Claude Code 2.1.296, Antigravity 1.3.1); each call starts from a fresh workspace and only `solver.py`, `debug.md` or the judge's notes come back. Solvers run natively in a macOS sandbox without network access.

```bash
docker build -f docker/agents/Dockerfile -t decompbench-agents-v2:20261013 .
python src/02_experiment.py docker prepare                        # record the image and sandbox evidence in outputs/runtime/
python src/02_experiment.py docker auth --backend codex --verify  # once per CLI: codex, claude or agy (see below)

python src/02_experiment.py init   --root outputs/experiments/my_run --config configs/paper.json
python src/02_experiment.py resume --root outputs/experiments/my_run      # a new campaign starts paused
python src/02_experiment.py watch  --root outputs/experiments/my_run --all-models
python src/02_experiment.py dashboard --root outputs/experiments/my_run   # http://localhost:8765
python src/03_analysis.py --campaign outputs/experiments/my_run           # audited report in outputs/reports/
```

`docker auth` copies an existing login into a private Docker volume and, with `--verify`, makes one unscored test edit; it does not sign in for you:

- **Codex:** sign in on the host first (`codex login`); `auth` copies `~/.codex/auth.json`.
- **Claude Code:** create a subscription token with `claude setup-token` and paste it at the hidden prompt (or set `CLAUDE_CODE_OAUTH_TOKEN`).
- **Antigravity:** the harness never signs in. It expects an existing Docker volume `decompbench-agy-native-keyring-v1` (label `io.decompbench.purpose=agy-native-auth`) holding a passphrase-protected GNOME keyring in which an Antigravity login has already been sealed; it unlocks that keyring with `DECOMPBENCH_AGY_KEYRING_PASSPHRASE` and restores the login into each fresh container. This repository does not include a tool to create that volume, so running Antigravity models requires setting it up by hand first.

To test other models, copy [configs/paper.json](configs/paper.json) and edit `model_queue`; the attempt budget, time limits, judge and prompts are fixed by the protocol. `init` freezes a copy of `src/`, `docker/`, the data and the config inside the campaign root, and `run`/`watch` only execute that copy, so later code changes never alter a running or finished campaign. `watch` restarts the run after infrastructure failures and pauses on CPU, memory or thermal pressure; Antigravity needs `DECOMPBENCH_AGY_KEYRING_PASSPHRASE` in the launch environment.

## Reproducing the paper

The paper's two campaign folders (the original run and a four-model Claude extension, 1.8 GB) are not in the repository. With them in `outputs/experiments/`:

```bash
python src/03_analysis.py --refresh                     # audit the raw records and rebuild everything
python src/03_analysis.py --paper path/to/paper         # also compare with a paper folder
DECOMPBENCH_PAPER=path/to/paper python -m pytest tests/test_paper_equivalence.py
```

`03_analysis.py` (which needs `pdflatex` and `gs`) writes all 12 tables, 12 figures and 8 prompt examples to `outputs/reports/paper/` and publishes them to [results/](results/README.md). The equivalence tests check that this repository still matches the code that ran the paper: every one of the 4,970 recorded generation, judge and reflection prompts is rebuilt byte for byte, all 2,363 executed solver outputs re-score identically, and every paper artifact is regenerated (text byte for byte, figures with identical rendering).

## Repository layout

```
src/
  01_gen.py           generate and validate the dataset             -> bd/, cg/, utils/
  02_experiment.py    Docker setup, campaigns, dashboard            -> experiments/
  03_analysis.py      every paper table and figure; campaign reports -> results/
  agent/              Docker CLI client, container runtime, all prompt text (prompts/)
  experiments/        campaign controller, paired runner, solver sandbox, verification, dashboard
  results/            audit snapshot, paper tables and figures, failure analysis, prompt examples
  bd/, cg/            the nine families: generator, task prompt, references, verifier
  utils/              base classes, dataset helpers, Python API, shared records
configs/paper.json    the paper's models and protocol
data/                 the 81 instances
docker/agents/        agent image, worker and seccomp profile
results/              published paper tables, figures and prompt examples
outputs/examples/     one complete example trajectory
docs/problems.md      problem formulations
tests/                unit tests and paper-equivalence tests
```

## Citation

*Beyond Formulation: DecompGym, an Executable Environment for LLM Agents Designing Decomposition Algorithms* (paper forthcoming on arXiv).

## Acknowledgment

The workflow of generating code, running it and feeding the outcome back builds on [HeuriGym](https://github.com/cornell-zhang/heurigym) ([paper](https://arxiv.org/abs/2506.07972)).

## License

[MIT](LICENSE). Gurobi and the agent CLIs are external dependencies with their own license and service terms.
