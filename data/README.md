# Dataset

- 81 instances: nine problem families × three sizes (small, medium, large) × three seeds.
- Benders families (`bd/`): UFL, FCTP, k-median, location routing over supplied routes, and two-stage stochastic production.
- Column-generation families (`cg/`): bin packing, cutting stock, generalized assignment, and shift scheduling.
- Each instance provides `data.json`, `prompt.txt` (the task the agent receives), `meta.json`, `gold_monolithic.json`, and `gold_bd.json` or `gold_cg.json`.
- `prompt.txt` is the strict-decomposition prompt used in the paper; it is rendered from `data.json` by the family's `b_problem.py`, and byte-identical to the prompt the agents received.
- The monolithic objective is the comparison reference (quality = within 1% of it). Decomposition references are separate feasible solutions with iteration and bound traces; restricted integer masters can have a positive gap, and the default bin-packing references have five positive integer gaps.
- `manifest.json` lists instance identities, data hashes, objectives and decomposition gaps; `reference-validation.json` records the feasibility and objective checks of all 162 reference solutions.
- `python src/01_gen.py -s all -m validate -vl simple` validates the shipped instances; generation reuses existing instances unless `--regenerate` is given.
