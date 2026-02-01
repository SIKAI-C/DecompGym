# Exp2 (Experiment 2) — Current Pipeline + Refactor Plan (for BD/CG)

This note summarizes what Exp2 currently does and proposes a cleaner refactor into modular components (classes) so we can run controlled ablations later (prompt variants, reflection variants, context variants, etc.).

## 1) What Exp2 is (current behavior)

Exp2 is the baseline orchestration framework for running **Benders Decomposition (BD)** and **Column Generation (CG)** benchmark instances using an LLM agent to iteratively generate `solver.py`, execute it, verify feasibility, check algorithmic compliance, and optionally reflect to improve the next attempt.

### Core goals of Exp2

- Provide a repeatable multi-attempt loop per instance.
- Record everything needed for debugging (`solver.py`, logs, verifier results).
- Support early stopping when a solution is good enough (based on feasibility + gap-to-gold threshold).
- Enable future experiments by swapping policies (context given to the agent, reflection on/off, lazy constraints on/off, etc.).

## 2) Current workflow (single instance)

Below is the current high-level flow of `run_instance()` and `run_attempt()`.

### Per-instance orchestration (`run_instance`)

1. Copy the benchmark instance into an output folder and create an `attempts/attempt_XX` structure.
2. Load the gold objective from `meta.json` and/or gold solution files.
3. For each attempt:
   - If a prior attempt already solved the instance (early stop), mark the attempt as skipped/reused.
   - Otherwise run one attempt:
     - Generate `solver.py` (LLM)
     - Execute `solver.py` -> produce `output.json`
     - Verify feasibility (problem verifier)
     - Check algorithmic compliance (LLM verifier: "did you actually implement BD/CG, not monolithic MIP?")
     - If failed, produce a reflection (`debug.md`) to guide the next attempt
4. Write `instance_summary.json` containing all attempt results.

### Per-attempt orchestration (`run_attempt`)

1. Build the prompt (benchmark prompt + "load data from `../../`" note + possibly previous reflections).
2. Call the LLM (via Aider) to write `solver.py`.
3. Execute `solver.py` and capture stdout/stderr.
4. Verify feasibility using the problem-specific verifier (`d_verifier.py`).
5. Run an LLM-based algorithm verifier to ensure the solver is actually BD/CG (and optionally BD + lazy constraints).
6. Compute gap-to-gold and set attempt status:
   - `ok`, `verify_failed`, `feasible_not_optimal`, `exec_failed`
7. If failed, generate `debug.md` reflection; always write `summary.json`.

## 3) Why refactor (pain points)

### Main issues in the current draft

- Too many responsibilities mixed in one file (prompting, file management, running, verification, reflection, early stop logic, logging).
- Eager file creation: the framework pre-creates files/folders that may never be used.
- Harder to run ablations: policies (context inclusion, reflection modes, lazy constraints, etc.) are implemented as ad-hoc logic rather than cleanly swappable components.
- Not enough observability: we want token/time cost per substep (solver-gen vs reflection vs verifier).

## 4) Refactor proposal: components as classes

Below is the "component decomposition" I propose. This is not final—please treat it as a discussion draft.

### A) FileManager (or ArtifactManager)

**Responsibility:** all filesystem layout + lazy creation.

**Key behaviors**

- Create attempt directories only when needed.
- Create files only when needed:
  - Create `solver.py` only right before the solver-writer step.
  - If reflection mode is OFF, do not create `debug.md`.
  - If LLM verifier is OFF, do not create `alg_verifier.md`.
- Standardize artifact structure.

**Proposed output layout (per attempt)**

- `attempt_XX/`
  - `solver.py` (only if solver-writer runs)
  - `output.json` (only if execution succeeded enough to write it)
  - `exec.log` (only if solver execution ran)
  - `summary.json` (always)
  - `debug.md` (only if reflection runs)
  - `alg_verifier.md` (only if LLM verifier runs)
  - `prompts/` (only if prompt snapshot logging enabled)
    - `base_prompt.txt`
    - `solver_prompt.txt`
    - `reflection_prompt.txt`
    - `alg_verifier_prompt.txt`

### B) BasePromptBuilder

**Responsibility:** build the canonical prompt for the agent from benchmark prompt + framework rules.

**Features / flags**

- `include_lazy_constraints`: bool
  - If True: add BD "must use lazy constraints in Gurobi" requirement.
  - If False: only include load-data instructions.
- Always include "load data from `../../data.json` and `../../prompt.txt`" instruction.

**Additional requirement**

- Save the final effective prompt used in each attempt under `attempt_XX/prompts/solver_prompt.txt`.
  - This makes debugging much easier when results look suspicious.

### C) SolverWriter (LLM coding step)

**Responsibility:** call aider to generate/edit `solver.py`.

**Key configurable experiment modes (context policy)**

1. **No-previous-info mode**
   - Only base prompt. No prior `solver.py`, no `debug.md`.
2. **All-previous-info mode**
   - Provide all previous `solver.py` and `debug.md` as read-only.
3. **Last-attempt mode**
   - Only the last attempt's `solver.py` and `debug.md` as read-only.
4. **All-attempts partial info**
   - Provide all previous `debug.md`, but no previous `solver.py`.
5. **Last-attempt partial info**
   - Provide only last `debug.md`, but no previous `solver.py`.

**Important micro-ablation: solver "seed" strategy**

This might significantly affect:

- Token usage
- Reasoning time
- Final performance/stability

Two policies:

- **Empty seed:** create an empty `solver.py` each attempt and ask the agent to write from scratch.
- **Carry-over seed:** copy previous attempt's `solver.py` into the new attempt and ask the agent to edit it.

This should be a first-class parameter, e.g.:

- `solver_seed_mode { empty, copy_previous }`

### D) OutputVerifier

**Responsibility:** deterministic feasibility verification of `output.json` against `data.json` using `d_verifier.py`.

**Why make it a class?**

- Equivalent importance to solver-writing and reflection modules.
- Clean interface:
  - `verify(data_path, output_path, tol) -> VerifyResult(ok, issues, stats)`
- Easier to:
  - swap tolerances
  - add richer verifier diagnostics
  - unify reporting in summaries

### E) AlgorithmVerifierLLM (LLM verifier)

**Responsibility:** check whether `solver.py` truly implements the required algorithm (not monolithic).

**Modes**

- `cg` (Column Generation)
- `bd` (Benders Decomposition)
- `bd_lazy` (Benders + lazy constraints)

**Additional requirement**

- Save the verifier prompt used in each attempt under `attempt_XX/prompts/alg_verifier_prompt.txt`.

### F) ReflectionEngine

**Responsibility:** write `debug.md` reflection after failures (or stay idle).

**Modes**

1. **Idle:** do nothing (no reflection file).
2. **Normal:** standard reflection based on:
   - `exec.log`
   - `solver.py`
   - feasibility verifier issues
   - LLM algorithm verifier output
3. **Special (research mode):** used to test reflection styles, e.g.
   - force conciseness
   - structured template
   - "top-3 actionable fixes only"
   - separate "bug-fix" vs "optimality-improvement" reflection formats

### G) MetricsTracker (tokens + time)

**Responsibility:** track per-step and per-attempt resource usage.

**Targets**

- Token usage per aider call:
  - solver writing
  - reflection writing
  - LLM algorithm verifier
- Wall-clock time per step:
  - solver generation time
  - solver execution time
  - verification time (deterministic + LLM)
- Aggregate to per-attempt and per-instance totals.

## 5) Token counting & logging (how to implement)

### Option 1: Use aider's `/tokens` command

Aider supports running in-chat commands in the Python scripting API, including `/tokens`.

This is useful for quickly inspecting token usage, but we need a reliable way to capture and parse the output programmatically.

### Option 2 (recommended): Use aider analytics logging to a JSONL file

Aider can log analytics events to a local file:

- `aider --analytics-log filename.jsonl`
- and can log without sending analytics remotely via:
- `aider --analytics-log filename.jsonl --no-analytics`

Also, the options reference documents `--analytics-log` (and its environment variable) as a supported configuration.

**Plan**

- For each attempt (or each substep), set the analytics log file path to:
  - `attempt_XX/metrics/aider_analytics.jsonl`
- Parse JSONL lines to attribute tokens/cost to the corresponding substep (solver vs reflection vs verifier).

**Privacy note**

- Analytics is opt-in, but logging locally without sending is explicitly supported (`--no-analytics`).

## 7) Future experiments enabled by this refactor

Once everything is modular, we can run controlled comparisons:

### A) Prompt / algorithm requirements

- BD vs BD-lazy vs CG
- With/without extra hints (e.g., "use lazy constraints")
- Prompt snapshotting + auditability (per attempt)

### B) Context policy ablations (solver writer)

- no previous info vs last attempt vs all attempts
- debug-only vs solver+debug
- empty seed vs carry-over solver seed

### C) Reflection policy ablations

- reflection off vs on
- long vs concise reflection
- structured reflection templates

### D) Budget-aware routing (later)

- If token/time budget is tight, switch to "last-attempt partial info" mode.
- If stuck, switch to "all-previous-info" mode.

## 8) Suggested deliverables (next version)

1. A clean class-based architecture (as above), with a small orchestrator that wires components together.
2. A single config object (dataclass) that defines all experiment knobs.
3. A standardized attempt artifact layout + prompt snapshots.
4. Token + time logging:
   - per substep
   - per attempt
   - per instance
5. Backward-compatible output summaries (`summary.json`, `instance_summary.json`, `problem_summary.json`) so analysis scripts do not break.
