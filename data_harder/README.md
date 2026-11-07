# Small-scale dataset of configs/harder_small_all.json (reused cohort)

- 54 instances: six problem families × three sizes (small, medium, large) × three seeds. These sizes are all small in absolute terms; a scale-up is still being designed.
- Benders family (`bd/`): logic-based Benders — task assignment to machines with single-machine scheduling, release times and weighted tardiness (`lbbd`).
- Column-generation families (`cg/`): graph coloring (`gcol`), bin packing with conflicts (`bppc`), vehicle routing with time windows (`vrptw`), pickup and delivery with time windows (`pdptw`), and nurse rostering (`nrp`).
- Each instance provides `data.json`, `prompt.txt` (the task the agent receives), `meta.json`, `gold_monolithic.json`, and `gold_bd.json` or `gold_cg.json`.
- Ground truth: 52 instances from the monolithic MIP optimum; BPPC medium seeds 0 and 2 from the branch-and-price optimality certificate (`meta.json` records `ground_truth_source`). Every decomposition reference proves optimality within 60 s and matches the ground truth.
- `lbbd`, `gcol` and `nrp` are final families whose results are reused: their generators, references and verifiers live in `src/bd/lbbd/` and `src/cg/{gcol,nrp}/` (registered in `src/utils/families.py` with their historical admission rule), and `python src/01_gen.py -ir <fresh root> -p gcol -m generate` reproduces them. The `bppc`, `vrptw` and `pdptw` instances here come from earlier generators that the literature-recipe generators replaced (kept in the Git history); their final small-scale instances are in `outputs/litgen_data` (`configs/small_scale.json`).
- This is a copy of the cohort used by `configs/harder_small_all.json` (inventory SHA-256 `4713b3796aed963ddae5e6d1c788506e3f4db97ee350b6c61c7589f03793186b`).
