"""Preregistration panels and task lists of the UFL / stochastic CFL bound-side Stage 1 (protocol v4 sections 6, 14-16),
and the additive freeze step. No optimization: file writing and hashing only.

  --write            writes scripts/phoenix/panels/{ufl,scfl}_stage1_panel.json (status PREREGISTERED: tiers, seeds,
                     instances, arms, thresholds, checker limits, the decision procedure; instance hashes, LP* intervals
                     and code hashes still empty) and the task lists scripts/phoenix/tasks/{ufl,scfl}_*.txt. Refuses to
                     overwrite a FROZEN panel.
  --freeze FAMILY --manifest M --references DIR [--commit SHA]
                     the additive freeze (after the generation job and the Stage 1 reference job): fills
                     references.<name>.instance_sha256 from the generation manifest (made with this very preregistered
                     panel), lp_star_frozen.intervals for the Stage 1 instances of the primary tiers from
                     DIR/<name>__lp_reference.json (each must be rigorous, within max_reference_rel_width, and computed on
                     the manifest's instance), the code hashes and commit, and sets status FROZEN. Every preregistered
                     field is checked against the stored preregistration sha256 first; nothing else changes. Holdout and
                     fallback-tier references are computed in their own jobs and sealed (--seal FAMILY TIER PART
                     --references DIR) before any of their arms is submitted; the runner reads only sealed references.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "scripts" / "phoenix" / "panels"
TASKS = ROOT / "scripts" / "phoenix" / "tasks"
INSTANCE_DIR = "outputs/scaleup/instances"
RAW_SCFL = "outputs/scaleup/raw/scfl/orlib"
CAP_SHA256 = {"cap111": "16b09017bb73164a5490fe9cd71a36da9269c702e8559b542e23a84d4e25774a",
              "cap114": "da4e7d25b11594e1621406e3a1ec8d2dce402d14f576ccaa88b42409afee221b"}
FROZEN_ONLY = ("status", "references", "lp_star_frozen", "code_sha256_at_preparation", "code_commit", "freeze", "preregistration")


def _runner():
    spec = importlib.util.spec_from_file_location("_bdlp_runner", ROOT / "scripts" / "phoenix" / "scaleup_bdlp_stage1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def prereg_sha(panel: dict) -> str:
    body = {k: v for k, v in panel.items() if k not in FROZEN_ONLY}
    return sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


RULES = {
    "eps_b": 1e-05,
    "eps_p": 0.001,
    "naive_quality_gap": 0.005,
    "naive_speed_ratio": 3.0,
    "saved_fraction": 0.25,
    "control_speed_ratio": 2.0,
    "control_saved_seconds": 225.0,
    "primal_gate": False,
    "max_reference_rel_width": 1e-06,
    "stage1_pass": "jointly >= 3 of 4 instances",
    "holdout_pass": "2 of 2 fresh instances, scored only after Stage 1 passed",
    "headline": "final_target for every tier: the engineered arm meets bound_target at T and every bound control misses it",
    "bound_target": "cert_ok and (LP_hi - LB_cert)/max(|LP_hi|, 1) <= eps_b, LP_hi from the frozen rigorous interval",
    "t_bound": "first checkpoint (60/300/600/900 s) meeting bound_target (censored at T)",
    "naive_separation": "naive LP-master bgap_LP - engineered bgap_LP >= 0.005 at T (exact), or t_naive (censored at T) >= 3 t_eng "
                        "AND t_naive - t_eng >= 0.25 T",
    "primal": "reported, not gated (protocol v4 section 14: the primal side is saturated for both families; the bound is the "
              "only headline): z_best = best verified incumbent of any arm at T; near_best = gap <= eps_p; the primal "
              "control and the naive integer Benders are reported against the engineered LP-rounding incumbent. A primal "
              "evaluation that cannot finish within the task allocation leaves only the primal metrics undefined",
    "method_ok": "every arm is fixed evaluator code whose method is settled before the run: the engineered arms are Benders "
                 "LP phases (UFL: subgradient warm-start cuts + purging; SCFL: Jensen cluster-dual seed cuts + core), the "
                 "intended section 14 configurations; eng_plain (same initialization as naive_lp: shared greedy design + "
                 "lower-bound cuts) is the matched-start diagnostic. bound_target therefore needs no separate method audit "
                 "in Stage 1; agent runs keep the audit",
    "bound_claim_refutation": "a solver-reported bound above z_best (relative 1e-6) is a refuted claim: infrastructure (rerun)",
    "publication_volume": "certificate versions received after the per-arm volume cap are recorded (time, size, sha256) but not "
                          "stored and never eligible; the arm is scored on its last stored version and the overflow is reported",
    "pruning": "after the canonical result is persisted, stored certificate files that no checkpoint selected are deleted "
               "(their receipt records stay in events.jsonl)",
    "lagrangian": "diagnostic (itself a decomposition; the method audit decides method_ok)",
    "mip": "reported baseline, not a gate",
    "decision_procedure": "an instance is decided only when every panel arm has a complete result with a valid identity; "
                          "infrastructure in a decision arm makes it pending (rerun), never a beaten control; a tier passes "
                          "Stage 1 with >= 3 of 4 passes (fails once 3 is impossible), then needs its holdout 2 of 2; "
                          "tiers are decided in tier_order, a fallback tier is run and read only after its primary tier "
                          "was rejected; the family keeps the first retained tier",
    "reference_rule": "an interval is usable only if rigorous (LP_lo <= LP_hi, both exact) with relative width <= "
                      "max_reference_rel_width; otherwise the instance stays pending (reference rerun with a longer T), "
                      "never a failure",
    "no_switching": "no change of headline, thresholds, arms, seeds or instances after any Stage 1 result is read",
    "oom_rule": "as k-median: a SIGKILL before T is a resource outcome only with a kernel-confirmed OOM kill of the arm",
    "result_identity": "as k-median plus the family: instance, arm, family, T, panel sha256, frozen instance sha256 and every "
                       "frozen code sha256; references read from the results directory (holdout, fallback tiers) must match "
                       "their reference seal (record sha256, witness hashes, instance hash, family and name), written by "
                       "scaleup_bdlp_panel.py --seal before that part's arms are submitted",
    "fractional_domain": "SCFL: every panel instance satisfies max_i u_i <= its smallest positive scenario total, so the cover "
                         "row is the exact recourse-feasible domain at fractional y (checked at generation and at the freeze)",
    "attempts": "as k-median: fresh attempt directories, exactly one scored attempt per arm (earliest complete, valid identity, "
                "no infrastructure)",
}

CHECKER = {
    "canonical": "src/experiments/xl_certificate.check_versions_supervised (forked checker, one hard per-attempt deadline, "
                 "family format converted to the canonical one inside it); views: xl_certificate.ufl_view (ufl-v1), "
                 "src/bd/scfl/h_view_xl.scfl_view (scfl-v1); no memory_bytes frozen, so every memory failure is infrastructure",
    "budget_seconds_per_attempt": 600,
    "max_bytes": 268435456,
    "max_nonzeros_expanded": 2000000,
    "max_master_nonzeros": 20000000,
    "max_abs_coef": 1e12,
    "token_chars": 64,
    "decimal_exponent": 400,
    "gurobi_mem_gb": 8.0,
    "version_rule": "last complete stored version received strictly before each checkpoint; final version checked first",
}


def _ufl_instances() -> tuple[dict, dict, list]:
    def spec(N, c, seed, tier, part):
        return {"generator": {"recipe": "Koerkel-Ghosh (UflLib), src/bd/ufl/a_data_xl.make_kg", "N": N, "class": c,
                              "seed": seed, "symmetric": True}, "tier": tier, "part": part}
    inst = {}
    tiers = {
        "N4000": {"headline": "final_target", "stage1": ["kgs4000b_g11", "kgs4000b_g12", "kgs4000c_g13", "kgs4000c_g14"],
                  "holdout": ["kgs4000b_g21", "kgs4000c_g22"],
                  "rationale": "protocol v4 section 14: N = 4000 classes b and c (compact LP out of memory; engineered reaches "
                               "1e-5 in 180-240 s, naive LP-master 0.39-0.58% short); fresh seeds (the probes used g1), one seed per "
                               "instance so that no two instances share a connection matrix (D is drawn first from the seed)"},
        "N3000c": {"headline": "final_target", "stage1": ["kgs3000c_g31", "kgs3000c_g32", "kgs3000c_g33", "kgs3000c_g34"],
                   "holdout": ["kgs3000c_g41", "kgs3000c_g42"], "run_if_rejected": "N4000",
                   "rationale": "protocol v4 section 14 fallback: N = 3000 class c (barrier 0.18% short at 900 s)"},
    }
    for tier, t in tiers.items():
        for part in ("stage1", "holdout"):
            for name in t[part]:
                N, c, seed = int(name[3:7]), name[7], int(name.split("_g")[1])
                inst[name] = spec(N, c, seed, tier, part)
    return inst, tiers, ["N4000", "N3000c"]


def _scfl_instances() -> tuple[dict, dict, list]:
    def spec(base, S, seed, tier, part):
        return {"generator": {"recipe": "OR-Library CAP base + Bodur et al. (2017) Sec. 4.4 scenarios, "
                                        "src/bd/scfl/a_data_xl.make_instance", "base": base, "base_sha256": CAP_SHA256[base],
                              "S": S, "seed": seed}, "tier": tier, "part": part}
    tiers = {
        "cap114_S5000": {"headline": "final_target",
                         "stage1": ["cap114_s5000_g11", "cap114_s5000_g12", "cap114_s5000_g13", "cap114_s5000_g14"],
                         "holdout": ["cap114_s5000_g21", "cap114_s5000_g22"],
                         "rationale": "protocol v4 section 14: S = 5000, cap114 preferred (compact LP out of memory; engineered "
                                      "Jensen-seeded reaches 1e-5 in 240 s; naive LP-master 11% short; Jensen control 0.19%)"},
        "cap111_S5000": {"headline": "final_target",
                         "stage1": ["cap111_s5000_g31", "cap111_s5000_g32", "cap111_s5000_g33", "cap111_s5000_g34"],
                         "holdout": ["cap111_s5000_g41", "cap111_s5000_g42"], "run_if_rejected": "cap114_S5000",
                         "rationale": "fallback named by section 14's probes (cap111 S5000: engineered 60 s, naive 13%, "
                                      "Jensen 0.068%, compact LP out of memory)"},
    }
    inst = {}
    for tier, t in tiers.items():
        for part in ("stage1", "holdout"):
            for name in t[part]:
                base, S, seed = name.split("_")[0], int(name.split("_s")[1].split("_")[0]), int(name.split("_g")[1])
                inst[name] = spec(base, S, seed, tier, part)
    return inst, tiers, ["cap114_S5000", "cap111_S5000"]


def build_panel(fam_name: str) -> dict:
    runner = _runner()
    fam = runner.family(fam_name)
    inst, tiers, order = (_ufl_instances if fam_name == "ufl" else _scfl_instances)()
    configs = {a: {"class": f"{cls.__module__}.{cls.__qualname__}", "kwargs": kw} for a, (cls, kw) in fam.arms().items()}
    ref_cls, ref_kw = fam.reference_arm()
    panel = {
        "family": fam_name,
        "protocol": "DecompBench/docs/2026-11-03-scaleup-protocol-v4.md sections 3, 4, 6, 14, 15, 16",
        "T": 900.0,
        "checkpoints": list(runner.CHECKPOINTS),
        "resources": {"cpus": 1, "mem": "16G (Slurm cgroup = process budget)", "gurobi_threads": 1,
                      "gurobi_mem_limit_gb": runner.GUROBI_MEM_GB, "slurm_time_per_arm_task": "01:00:00",
                      "slurm_time_per_reference_task": "03:00:00 (T = 3600 s)"},
        "checker": CHECKER,
        "publication_volume_cap_bytes": runner.PUBLICATION_VOLUME_CAP,
        "rules": RULES,
        "arms": {"engineered": fam.ENG, "naive_bound": fam.NAIVE, "bound_controls": fam.BOUND_CONTROLS,
                 "primal_control": fam.PRIMAL_CONTROL, "primal_naive": fam.PRIMAL_NAIVE, "diagnostics": fam.DIAGNOSTICS,
                 "all": fam.ALL_ARMS, "configs": configs,
                 "reference": {"class": f"{ref_cls.__module__}.{ref_cls.__qualname__}", "kwargs": ref_kw},
                 "trivial_certificate_controls": {
                     "empty": "no cuts",
                     "analytical": ("closed-form radius cuts at the R cheapest cost levels per customer (k-median recipe), "
                                    "src/bd/ufl/i_controls_xl.analytical_cuts") if fam_name == "ufl" else
                                   ("uncapacitated-relaxation cuts (alpha = 0) at the r-th smallest unit cost per customer, "
                                    "r = 1..R, one term per scenario, src/bd/scfl/i_controls_xl.analytical_terms"),
                     "oneshot": "exact cuts at the shared initializer's design" + (
                         "; ils: at each new best design of the ILS control" if fam_name == "ufl" else
                         "; oneshot_saa: at a 10-scenario SAA design; heur: at the best design of the SAA + local-search "
                         "control, in a reserve of max(60 s, 0.1 T)"),
                     "pool": "deterministic no-feedback oracle pool, seed 20261103 (g_bound_xl Pool)",
                     **({"jensen": "cluster-mean extensive-form LPs (G = 10, 20, 40, ...), duals as cuts"} if fam_name == "scfl" else {})}},
        "runtime": {"enforced": {"python": "3.11.6", "gurobi": "13.0.1"},
                    "recorded": ["numpy", "scipy", "host", "slurm ids"]},
        "tiers": tiers,
        "tier_order": order,
        "instances": inst,
        "instance_paths": f"{INSTANCE_DIR}/{fam_name}/<name>.json (generated by --gen; hashes frozen at the freeze)",
        "status": "PREREGISTERED (2026-11-05): tiers, seeds, arms, thresholds and rules frozen before any run; instance "
                  "hashes, LP* intervals and code hashes are added by the additive freeze only",
        "references": {n: {"instance_sha256": None, "tier": s["tier"], "part": s["part"]} for n, s in inst.items()},
        "lp_star_frozen": {"intervals": {}},
        "code_sha256_at_preparation": {},
    }
    panel["preregistration"] = {"sha256": prereg_sha(panel), "excluded_fields": list(FROZEN_ONLY)}
    return panel


def task_lists(panel: dict) -> dict[str, list[str]]:
    fam = panel["family"]
    arms = panel["arms"]["all"]
    path = lambda n: f"{INSTANCE_DIR}/{fam}/{n}.json"   # noqa: E731
    out: dict[str, list[str]] = {}
    gen = f"--gen {fam} {INSTANCE_DIR}/{fam}" + (f" --raw {RAW_SCFL}" if fam == "scfl" else "")
    out[f"{fam}_gen.txt"] = [gen]
    for tier in panel["tier_order"]:
        t = panel["tiers"][tier]
        tag = fam if "run_if_rejected" not in t else f"{fam}_fallback_{tier}"
        out[f"{tag}_stage1_reference.txt"] = [f"--reference {fam} {path(n)}" for n in t["stage1"]]
        out[f"{tag}_stage1_runs.txt"] = [f"--task {fam} {path(n)} {a}" for n in t["stage1"] for a in arms]
        out[f"{tag}_holdout_reference.txt"] = [f"--reference {fam} {path(n)}" for n in t["holdout"]]
        out[f"{tag}_holdout_runs.txt"] = [f"--task {fam} {path(n)} {a}" for n in t["holdout"] for a in arms]
    return out


def write(families=("ufl", "scfl")) -> None:
    PANELS.mkdir(parents=True, exist_ok=True)
    TASKS.mkdir(parents=True, exist_ok=True)
    for fam in families:
        p = PANELS / f"{fam}_stage1_panel.json"
        if p.exists() and json.loads(p.read_text()).get("status", "").startswith("FROZEN"):
            raise SystemExit(f"{p} is FROZEN: refusing to overwrite")
        panel = build_panel(fam)
        p.write_text(json.dumps(panel, indent=1) + "\n")
        for fname, lines in task_lists(panel).items():
            (TASKS / fname).write_text("\n".join(lines) + "\n")
        print(f"wrote {p.relative_to(ROOT)} (preregistration {panel['preregistration']['sha256'][:12]}) and "
              f"{len(task_lists(panel))} task lists")


def check_reference(row: dict, name: str, fam: str, inst_sha: str, refdir: Path, panel_sha: str,
                    code: dict[str, str], max_width: float, runtime: dict | None = None) -> dict:
    """Validation of one reference record before it is frozen or sealed (Codex round 1 #2): family, instance name and
    hash; exact endpoints with lo <= hi and the relative width recomputed exactly; witness files present with their
    recorded hashes; the record produced under the expected panel and the expected code. Returns the interval entry."""
    def bad(why: str):
        raise SystemExit(f"{name}: {why}")
    if row.get("family") != fam or row.get("instance") != name:
        bad("family/instance mismatch")
    if row.get("instance_sha256") != inst_sha:
        bad("reference computed on another instance file")
    iv = row.get("interval") or {}
    try:
        lo, hi = Fraction(str(iv["lo"])), Fraction(str(iv["hi"]))
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        bad(f"interval without exact endpoints: {iv}")
    if lo > hi:
        bad("lo > hi")
    can = row.get("canonical") or {}
    if can.get("cert_ok") is not True or can.get("lb") is None or Fraction(str(can["lb"])) != lo:
        bad("LP_lo is not the canonical certified bound of the reference certificate")
    if (row.get("lp_hi") or {}).get("value") is None or Fraction(str(row["lp_hi"]["value"])) != hi:
        bad("LP_hi is not the exact value of the reference LP point")
    width = (hi - lo) / max(abs(hi), Fraction(1))
    if width > Fraction(str(max_width)):
        bad(f"interval too wide ({float(width):.3g}): rerun the reference with a longer T")
    for fkey, hkey in (("certificate_file", "certificate_sha256"), ("lp_point_file", "lp_point_sha256")):
        f = row.get(fkey)
        if not f or not (refdir / f).is_file() or sha256((refdir / f).read_bytes()) != row.get(hkey):
            bad(f"witness {fkey} missing or hash mismatch")
    w = (row.get("lp_hi") or {}).get("witness_file")
    if fam == "scfl" and (not w or not (refdir / w).is_file() or sha256((refdir / w).read_bytes()) != row["lp_hi"].get("witness_sha256")):
        bad("SCFL LP_hi flow witness missing or hash mismatch")
    ident = row.get("identity") or {}
    for key, want in (runtime or {}).items():   # the panel's runtime pins (Codex round 2 #2)
        if ident.get(key) != want:
            bad(f"reference produced under runtime {key} {ident.get(key)!r} != frozen {want!r}")
    if (ident.get("panel") or {}).get("sha256") != panel_sha:
        bad("reference produced under another panel")
    got = ident.get("code_sha256") or {}
    diff = sorted(f for f, h in code.items() if got.get(f) != h)
    if diff:
        bad(f"reference produced with other code: {diff}")
    return {"lo": str(lo), "hi": str(hi), "rigorous": True, "rel_width": float(width), "lo_float": float(lo), "hi_float": float(hi),
            "instance_sha256": inst_sha, "reference_file_sha256": sha256((refdir / f"{name}__lp_reference.json").read_bytes()),
            "certificate_sha256": row.get("certificate_sha256"), "point_sha256": row.get("lp_point_sha256"),
            "witness_sha256": (row.get("lp_hi") or {}).get("witness_sha256"),
            "lo_provenance": iv.get("lo_provenance"), "hi_provenance": iv.get("hi_provenance")}


def code_hashes(runner, fam: str) -> dict[str, str]:
    files = sorted(set(runner.COMMON_CODE) | set(runner.family(fam).CODE_FILES) | {"scripts/phoenix/scaleup_bdlp_panel.py"})
    return {f: sha256((ROOT / f).read_bytes()) for f in files}


def freeze(fam: str, manifest: Path, refdir: Path, commit: str | None, panel_path: Path | None = None) -> dict:
    runner = _runner()
    p = Path(panel_path) if panel_path is not None else PANELS / f"{fam}_stage1_panel.json"
    raw = p.read_bytes()
    panel = json.loads(raw)
    if not panel.get("status", "").startswith("PREREGISTERED"):
        raise SystemExit(f"{p}: status {panel.get('status')!r} (expected PREREGISTERED)")
    if prereg_sha(panel) != panel["preregistration"]["sha256"]:
        raise SystemExit("preregistered fields changed since the preregistration (sha256 mismatch)")
    man = json.loads(manifest.read_text())
    if man.get("family") != fam or man.get("panel_sha256") != sha256(raw):
        raise SystemExit("the generation manifest was not made with this preregistered panel")
    for name in panel["instances"]:
        r = man["instances"].get(name)
        if not r or r.get("recipe_problems"):
            raise SystemExit(f"{name}: missing from the manifest or recipe problems {r and r.get('recipe_problems')}")
        panel["references"][name]["instance_sha256"] = r["sha256"]
    code = code_hashes(runner, fam)
    intervals = {}
    for tier in panel["tier_order"]:
        t = panel["tiers"][tier]
        if "run_if_rejected" in t:
            continue
        for name in t["stage1"]:
            row = json.loads((refdir / f"{name}__lp_reference.json").read_text())
            intervals[name] = check_reference(row, name, fam, panel["references"][name]["instance_sha256"], refdir,
                                              sha256(raw), code, panel["rules"]["max_reference_rel_width"],
                                              (panel.get("runtime") or {}).get("enforced"))
    panel["code_sha256_at_preparation"] = code
    if commit:
        panel["code_commit"] = commit
    panel["lp_star_frozen"] = {"intervals": intervals, "directory": str(refdir)}
    tasks = {f.name: sha256(f.read_bytes()) for f in sorted(TASKS.glob(f"{fam}_*.txt"))}
    panel["freeze"] = {"manifest_sha256": sha256(manifest.read_bytes()), "preregistered_panel_sha256": sha256(raw),
                       "task_lists_sha256": tasks,
                       "sbatch_sha256": sha256((ROOT / "scripts/phoenix/scaleup_bdlp_stage1.sbatch").read_bytes())}
    panel["status"] = "FROZEN for Stage 1 (additive freeze of the preregistered panel " + sha256(raw)[:12] + ")"
    p.write_text(json.dumps(panel, indent=1) + "\n")
    print(f"froze {p}: {len(intervals)} intervals, {len(code)} code hashes; sha256 {sha256(p.read_bytes())}")
    return panel


def seal(fam: str, refdir: Path, tier: str, part: str, panel_path: Path | None = None) -> dict:
    """Reference seal of a holdout part or a fallback tier's Stage 1 (Codex round 1 #2): validates every reference
    record of that part (as at the freeze, against the FROZEN panel and its code hashes) and writes
    refdir/<fam>_<tier>_<part>_reference_seal.json once (never overwritten). Run it after the part's reference job and
    BEFORE submitting its arm tasks; the summary reads only sealed references."""
    runner = _runner()
    p = Path(panel_path) if panel_path is not None else PANELS / f"{fam}_stage1_panel.json"
    raw = p.read_bytes()
    panel = json.loads(raw)
    if not panel.get("status", "").startswith("FROZEN"):
        raise SystemExit("seal needs the FROZEN panel")
    t = panel["tiers"][tier]
    if part not in ("stage1", "holdout") or (part == "stage1" and "run_if_rejected" not in t):
        raise SystemExit("seal: holdout of any tier, or stage1 of a fallback tier (primary Stage 1 is frozen in the panel)")
    refs = {}
    for name in t[part]:
        row = json.loads((refdir / f"{name}__lp_reference.json").read_text())
        check_reference(row, name, fam, panel["references"][name]["instance_sha256"], refdir, sha256(raw),
                        panel["code_sha256_at_preparation"], panel["rules"]["max_reference_rel_width"],
                        (panel.get("runtime") or {}).get("enforced"))
        refs[name] = sha256((refdir / f"{name}__lp_reference.json").read_bytes())
    out = {"family": fam, "tier": tier, "part": part, "panel_sha256": sha256(raw), "references": refs}
    sp = runner.seal_path(refdir, fam, tier, part)
    with open(sp, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(f"sealed {sp} ({len(refs)} references)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--freeze", metavar="FAMILY")
    ap.add_argument("--manifest")
    ap.add_argument("--references")
    ap.add_argument("--commit", default=None)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--seal", nargs=3, metavar=("FAMILY", "TIER", "PART"))
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    if a.write:
        write()
    if a.freeze:
        freeze(a.freeze, Path(a.manifest), Path(a.references), a.commit, Path(a.panel) if a.panel else None)
    if a.seal:
        seal(a.seal[0], Path(a.references), a.seal[1], a.seal[2], Path(a.panel) if a.panel else None)


if __name__ == "__main__":
    main()
