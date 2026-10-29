# Harder-family dataset (exploration, small scale)

- 54 instances: six problem families × three sizes (small, medium, large) × three seeds. These sizes are all small in absolute terms; a scale-up is still being designed.
- Benders family (`bd/`): logic-based Benders — task assignment to machines with single-machine scheduling, release times and weighted tardiness (`lbbd`).
- Column-generation families (`cg/`): graph coloring (`gcol`), bin packing with conflicts (`bppc`), vehicle routing with time windows (`vrptw`), pickup and delivery with time windows (`pdptw`), and nurse rostering (`nrp`).
- Each instance provides `data.json`, `prompt.txt` (the task the agent receives), `meta.json`, `gold_monolithic.json`, and `gold_bd.json` or `gold_cg.json`.
- Ground truth: 52 instances from the monolithic MIP optimum; BPPC medium seeds 0 and 2 from the branch-and-price optimality certificate (`meta.json` records `ground_truth_source`). Every decomposition reference proves optimality within 60 s and matches the ground truth.
- Generators, references and verifiers live in `src/bd/lbbd/` and `src/cg/{gcol,bppc,vrptw,pdptw,nrp}/`; `python src/01_gen.py --family-set harder` regenerates the cohort (it refuses to write into `data/`).
- This is a copy of the cohort used by `configs/harder_small.json` and `configs/harder_small_all.json` (inventory SHA-256 `4713b3796aed963ddae5e6d1c788506e3f4db97ee350b6c61c7589f03793186b`).
