# Example: Claude Sonnet 5 on cutting stock (csp_medium_seed000)

The case study of the paper, copied from the Claude extension campaign (protocol `a7653c56…`). Both policies share attempt 1.

Attempt 1 cuts the 87 stock pieces of the reference, but reports `optimal` because Gurobi solved the final integer restricted master to optimality. That proves optimality only over the generated patterns, so the method judge rejects the unproven claim ([judge notes](plain/attempts/attempt_01/alg_verifier.md)).

**With reflection**, the agent reads the judge's notes, writes [advice](reflection/attempts/attempt_01/debug.md), and attempt 2 reports `optimal` only when pricing has converged and the integer objective equals the rounded-up LP bound. The judge accepts it.

| Attempt | Status | Judge | Objective |
| ---: | --- | --- | ---: |
| [1](reflection/attempts/attempt_01/) | verify_failed | fail | 87.0 |
| [2](reflection/attempts/attempt_02/) | ok | pass | 87.0 |

**Without reflection**, each attempt starts again from the task and repeats the same status logic:

| Attempt | Status | Judge | Objective |
| ---: | --- | --- | ---: |
| [1](plain/attempts/attempt_01/) | verify_failed | fail | 87.0 |
| [2](plain/attempts/attempt_02/) | verify_failed | fail | 87.0 |
| [3](plain/attempts/attempt_03/) | verify_failed | fail | 87.0 |
| [4](plain/attempts/attempt_04/) | verify_failed | fail | 87.0 |
| [5](plain/attempts/attempt_05/) | verify_failed | fail | 87.0 |
| [6](plain/attempts/attempt_06/) | verify_failed | fail | 87.0 |
| [7](plain/attempts/attempt_07/) | verify_failed | fail | 87.0 |
| [8](plain/attempts/attempt_08/) | verify_failed | fail | 87.0 |
| [9](plain/attempts/attempt_09/) | verify_failed | fail | 87.0 |
| [10](plain/attempts/attempt_10/) | verify_failed | fail | 87.0 |

Each attempt folder holds the exact prompts sent (`agent_prompt.md`, `llm_verifier_prompt.md`, `reflection_prompt.md`), the generated `solver.py`, its `output.json` and `exec.log`, the judge's `alg_verifier.md` and the scored `summary.json`. Call event streams, container records and host paths were left out.
