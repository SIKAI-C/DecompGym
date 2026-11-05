"""Panel and task-list builder for the protocol v4 CG Stage 1 (CLRP, CVRP, BPPC). Builds no model and runs no solver.

Writes, under --out (default outputs/scaleup/cg_v4):
  instances/<family>/<file>     the instance files (literature files copied byte for byte; BPPC recipe instances
                                generated with fixed seeds), each hashed in the panel
  <family>_panel.json           the preregistration panel: tiers (dev / stage1 / holdout), arms, rules, references
                                (instance sha256, z_pre and its source), code hashes at preparation
  tasks/<family>_{dev,stage1,holdout}.txt   one array task per line: "--task <family> <instance path> <arm>"
Status DRAFT until `--freeze EPS_JSON --proposals ... --commit SHA` writes <family>_panel_frozen.json (the draft stays
immutable: the development runs are identity-checked against it); nothing after a freeze may change.

Instance choice (protocol v4 Section 6, "no grandfathering": the controls run on instances that were not used for the
v3.1 decisions; dev = the v3.1 panel instances, used only to calibrate eps_p):
  CVRP  tier X9: Uchoa X with n/K <= 9 (K = ceil(q(N)/Q)), the v3.1 stratum. Its n = 101-200 part is exhausted (11 of
        12 used), so the fresh instances are the six smallest unused ones with n/K <= 9 (n = 194-250), split into
        Stage 1 / holdout by a frozen seeded shuffle; dev = the 11 v3.1 instances.
  BPPC  tiers DA250 (rho 0.6) and UA500 (rho 0.9), Sadykov-Vanderbeck recipe (src/cg/bppc/a_data.arbitrary_instance):
        Stage 1 k = 5..8 (fresh); holdout DA250 k = 11, 12 (fresh seeds of the same recipe); UA500 has no unused
        literature base file (k <= 10), so its holdout uses fresh conflict graphs on the items of k = 1, 2 (replicate
        label g2, generated here with the recipe's edge rule); dev = the v3.1 k = 1-4, 9, 10.
  CLRP  tier P100: Prodhon 100-customer instances. All 18 Prodhon 100-200 instances were used in v3.1, so NO fresh
        Prodhon instance exists: Stage 1 = the four variant-3 instances (v3.1 holdout, never used for development),
        dev = the eight variant 1/2 instances; the holdout must be chosen by the user (--clrp-holdout), else the
        panel stays DRAFT with an empty holdout (open question).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SPLIT_SEED = 20261105
T_F = 900.0
RULES = {
    "headline": "near_best (final quality at T_f)",
    "near_best": "(z - z_best) / max(1, |z_best|) <= eps_p (tier), exact",
    "z_best": "min(z_pre, best verified incumbent of any panel arm at T_f)",
    "naive_quality_gap": 0.005,
    "naive_quality_rule": "z_naive - z_eng >= ceil(0.005 z_best) (integer objectives); no naive incumbent = infinitely worse",
    "naive_speed_ratio": 3.0,
    "saved_fraction": 0.25,
    "naive_speed_rule": "t_naive(near_best) >= 3 t_eng(near_best) and t_naive - t_eng >= 0.25 T (naive never reaching: t = T)",
    "controls": "every decomposition-free control misses near_best (no incumbent = miss)",
    "mip": "reported baseline, not a gate (protocol v4 6.3)",
    "stage1_pass": "jointly >= 3 of 4 instances",
    "holdout_pass": "2 of 2, scored only after Stage 1 passed",
    "eps_p_rule": "frozen per tier before Stage 1 from the development runs (eng and controls on the dev instances): "
                  "the midpoint of the widest interval [g_eng, g_ctrl) that separates the most dev instances; at least "
                  "ceil(3/4 n_dev) must be separated, else the tier fails before Stage 1 (scaleup_cg_stage1.separating_eps)",
    "no_switching": "no change of eps_p, arms, instances, T or code after the freeze",
    "infrastructure": "an arm with a supervisor error, an unexpected exit, an evaluator or stdlib-verifier failure is "
                      "rerun (fresh attempt directory), never scored; a kernel-confirmed OOM kill is an arm outcome",
}
ARMS = {
    "eng": "engineered B&P(&C) reference (d_ref_cg_xl)",
    "naive": "naive CG (plain RMP, exact pricing, 1 column per round, no stabilization / heuristic pricing, integer RMP last 20%)",
    "ls": "decomposition-free control (cvrp: f_ils_xl.CVRPRuinRecreate; clrp: g_lrp_ils_xl.CLRPLocationRoutingILS; "
          "bppc: f_heuristic.BPPCHeuristicControl), seed 0",
    "mip": "compact MIP baseline (b_mip_xl), reported only",
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def seeded_split(names: list[str], n_stage1: int, salt: str) -> tuple[list[str], list[str]]:
    rnd = random.Random(f"{SPLIT_SEED}:{salt}")
    order = list(names)
    rnd.shuffle(order)
    s1 = sorted(order[:n_stage1], key=names.index)
    return s1, sorted(order[n_stage1:], key=names.index)


# ---------------------------------------------------------------- CVRP

CVRP_V31 = ["X-n101-k25", "X-n106-k14", "X-n110-k13", "X-n125-k30", "X-n148-k46", "X-n153-k22", "X-n176-k26",
            "X-n200-k36", "X-n129-k18", "X-n172-k51", "X-n181-k23"]


def cvrp(raw: Path, out: Path) -> dict:
    from src.cg.cvrp.a_data import load_sol, load_vrp
    from src.cg.cvrp.b_mip_xl import route_cost
    src = raw / "cvrp" / "Instances-main" / "CVRP"
    rows = []
    for f in src.glob("X-n*.vrp"):
        inst = load_vrp(f)
        K = math.ceil(int(inst.demand.sum()) / inst.capacity)
        rows.append((inst.n, f.stem, inst.n / K))
    rows.sort()
    fresh = [nm for n, nm, r in rows if r <= 9 and nm not in CVRP_V31][:6]
    s1, hold = seeded_split(fresh, 4, "cvrp")
    refs = {}
    for nm in CVRP_V31 + fresh:
        dst = out / "instances" / "cvrp" / f"{nm}.vrp"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / f"{nm}.vrp", dst)
        inst = load_vrp(dst)
        z_pre, why = None, "no .sol"
        if (src / f"{nm}.sol").exists():
            routes, stated = load_sol(src / f"{nm}.sol")
            got = route_cost(inst, routes)
            z_pre, why = (stated, "CVRPLIB .sol (BKS), plan re-verified") if got == stated else (None, f".sol plan cost {got} != stated {stated}")
        refs[nm] = {"file": f"instances/cvrp/{nm}.vrp", "instance_sha256": sha256_file(dst), "z_pre": z_pre,
                    "z_pre_source": why, "n": inst.n, "part": "dev" if nm in CVRP_V31 else ("stage1" if nm in s1 else "holdout")}
    return {"tiers": {"X9": {"scope": "Uchoa X, n/K <= 9 (v3.1 stratum); fresh part n = 194-250 (n = 101-200 exhausted)",
                             "dev": CVRP_V31, "stage1": s1, "holdout": hold, "eps_p": None}}, "references": refs}


# ---------------------------------------------------------------- BPPC

def bppc_replicate(kind: str, n: int, rho10: int, k: int, rep: str, raw_dir: Path):
    """The recipe instance of class (kind, n, rho10) on the items of base k, with a fresh conflict graph drawn by the
    recipe's edge rule (a_data.random_conflicts) from the seed label '<name>_<rep>'."""
    import numpy as np
    from src.cg.bppc.a_data import BPPCInstance, _rng, arbitrary_instance, random_conflicts
    base = arbitrary_instance(kind, n, rho10, k, raw_dir)
    name = f"{base.name}_{rep}"
    adj = np.zeros((n, n), dtype=bool)
    for i, j in random_conflicts(n, rho10 / 10, _rng(f"bppc-sv-arbitrary:{name}")):
        adj[i, j] = adj[j, i] = True
    return BPPCInstance(name, base.capacity, base.weights.copy(), adj, dict(base.meta, replicate=rep))


# Verified v3.1 bin counts (journey 2026-10-30 sections 51.1 and 51.2: Stage 1 job 14824076 and holdout job 14825400,
# code_0a7b2f9, every incumbent verified; witnesses on Phoenix under results/stage1_bppc_t900_20261031 and
# results/stage2_bppc_holdout_t900_20261031). Merged with bppc_references.json by minimum (Codex review CG-1 #5).
BPPC_V31_VERIFIED = {"da250_r6_k1": 39, "da250_r6_k2": 39, "da250_r6_k3": 39, "da250_r6_k4": 39, "da250_r6_k9": 38,
                     "da250_r6_k10": 40, "ua500_r9_k1": 201, "ua500_r9_k2": 204, "ua500_r9_k3": 204, "ua500_r9_k4": 207,
                     "ua500_r9_k9": 198, "ua500_r9_k10": 204}


def bppc(raw: Path, refs_dir: Path, out: Path) -> dict:
    from src.cg.bppc.a_data import arbitrary_instance, write_bppc
    lit = raw / "bppc" / "Istanze"
    known = json.loads((refs_dir / "bppc_references.json").read_text()).get("instances", {})
    tiers, refs = {}, {}
    spec = {"DA250": ("da", 250, 6, [5, 6, 7, 8], [("k", 11), ("k", 12)]),
            "UA500": ("ua", 500, 9, [5, 6, 7, 8], [("g2", 1), ("g2", 2)])}
    for tname, (kind, n, r, s1k, holdk) in spec.items():
        dev = [f"{kind}{n}_r{r}_k{k}" for k in (1, 2, 3, 4, 9, 10)]
        names = {"dev": [], "stage1": [], "holdout": []}
        jobs = [("dev", k, None) for k in (1, 2, 3, 4, 9, 10)] + [("stage1", k, None) for k in s1k] + \
               [("holdout", k, None if lab == "k" else lab) for lab, k in holdk]
        for part, k, rep in jobs:
            inst = arbitrary_instance(kind, n, r, k, lit) if rep is None else bppc_replicate(kind, n, r, k, rep, lit)
            dst = write_bppc(out / "instances" / "bppc" / f"{inst.name}.txt", inst)
            names[part].append(inst.name)
            z1 = (known.get(inst.name) or {}).get("best_known_this_study")
            z2 = BPPC_V31_VERIFIED.get(inst.name)
            zs = [int(x) for x in (z1, z2) if x is not None]
            z = min(zs) if zs else None
            src = None if z is None else ("v3.1 Stage 1/holdout verified run (journey 51.1-51.2)" if z == z2 else
                                          "bppc_references.json best_known_this_study (verified v3.1 development run)")
            refs[inst.name] = {"file": f"instances/bppc/{inst.name}.txt", "instance_sha256": sha256_file(dst),
                               "z_pre": z, "z_pre_source": src,
                               "n": inst.n, "edges": inst.num_conflicts, "part": part}
        assert names["dev"] == dev
        tiers[tname] = {"scope": f"Sadykov-Vanderbeck {kind}{n}, rho = {r / 10}", **names, "eps_p": None}
    tiers["UA500"]["holdout_note"] = ("fresh conflict graphs (replicate g2) on REUSED weight vectors (the items of k = 1, "
                                      "2, also dev instances): no unused literature base file; fresh conditional on the weights")
    return {"tiers": tiers, "references": refs}


# ---------------------------------------------------------------- CLRP

def clrp(raw: Path, refs_dir: Path, out: Path, holdout: list[str]) -> dict:
    src = raw / "lrp" / "prodhon"
    known = json.loads((refs_dir / "clrp_prodhon_references.json").read_text())
    by_file = {v["file"]: v for v in known.values()}
    dev = ["coord100-5-1", "coord100-5-1b", "coord100-5-2", "coord100-5-2b", "coord100-10-1", "coord100-10-1b",
           "coord100-10-2", "coord100-10-2b"]
    s1 = ["coord100-5-3", "coord100-5-3b", "coord100-10-3", "coord100-10-3b"]
    refs = {}
    for nm in dev + s1 + holdout:
        if not (src / f"{nm}.dat").exists():
            raise SystemExit(f"CLRP instance {nm}.dat not found in {src}")
        dst = out / "instances" / "clrp" / f"{nm}.dat"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / f"{nm}.dat", dst)
        k = by_file.get(f"{nm}.dat")
        refs[nm] = {"file": f"instances/clrp/{nm}.dat", "instance_sha256": sha256_file(dst),
                    "z_pre": int(k["value"]) if k else None, "z_pre_source": k["source"] if k else None,
                    "z_pre_kind": k["kind"] if k else None,
                    "part": "dev" if nm in dev else ("stage1" if nm in s1 else "holdout")}
    return {"tiers": {"P100": {"scope": "Prodhon 100 customers, 5 and 10 depots", "dev": dev, "stage1": s1,
                               "holdout": holdout, "eps_p": None,
                               "freshness": "NOT fresh: every Prodhon 100-200 instance was used in v3.1 (Stage 1 = v3.1 "
                                            "holdout variant 3, never used for development)"}}, "references": refs}


# ---------------------------------------------------------------- panel and tasks

def code_hashes(family: str) -> dict:
    from scripts.phoenix.scaleup_cg_stage1 import code_files  # noqa: E402 (path set above)
    return {f: sha256_file(ROOT / f) for f in code_files(family)}


def write_tasks(family: str, panel: dict, out: Path) -> dict:
    tdir = out / "tasks"
    tdir.mkdir(parents=True, exist_ok=True)
    files = {}
    try:   # task paths are relative to the code checkout (the sbatch runs from CODE_ROOT)
        base = out.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        base = out.resolve().as_posix()
    for part, arms in (("dev", ["eng", "ls"]), ("stage1", list(ARMS)), ("holdout", list(ARMS))):
        lines = [f"--task {family} {base}/{panel['references'][nm]['file']} {a}"
                 for t in panel["tiers"].values() for nm in t.get(part, []) for a in arms]
        p = tdir / f"{family}_{part}.txt"
        p.write_text("\n".join(lines) + ("\n" if lines else ""))
        files[p.name] = {"lines": len(lines), "sha256": sha256_file(p)}
    return files


def build(args) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    built = {"cvrp": lambda: cvrp(args.raw, out), "bppc": lambda: bppc(args.raw, args.refs, out),
             "clrp": lambda: clrp(args.raw, args.refs, out, args.clrp_holdout or [])}
    for fam in args.families:
        body = built[fam]()
        status = "DRAFT (eps_p pending the development calibration; not frozen)"
        if fam == "clrp" and not args.clrp_holdout:
            status += "; CLRP holdout unresolved (user decision)"
        panel = {"family": fam, "status": status, "protocol": "DecompBench/docs/2026-11-03-scaleup-protocol-v4.md sections 4, 6",
                 "T": T_F, "resources": {"cpus": 1, "mem": "16G (Slurm cgroup)", "gurobi_threads": 1,
                                         "gurobi_soft_mem_limit_gb": 12.0, "slurm_time_per_task": "00:40:00"},
                 "rules": RULES, "arms": ARMS, "split_seed": SPLIT_SEED, **body,
                 "code_sha256_at_preparation": code_hashes(fam)}
        (out / f"{fam}_panel.json").write_text(json.dumps(panel, indent=1) + "\n")
        panel["split_problems"] = split_problems(panel)   # must be empty before the freeze (CLRP: holdout undecided)
        panel["tasks"] = write_tasks(fam, panel, out)
        (out / f"{fam}_panel.json").write_text(json.dumps(panel, indent=1) + "\n")
        print(fam, {t: {p: len(v[p]) for p in ("dev", "stage1", "holdout")} for t, v in panel["tiers"].items()}, panel["tasks"])


def split_problems(panel: dict) -> list[str]:
    """Each tier: 4 distinct Stage 1 and 2 distinct holdout instances; dev, Stage 1 and holdout pairwise disjoint by name
    AND by instance sha256 across the whole panel (Codex review CG-1 #1)."""
    p = []
    refs = panel["references"]
    seen: dict[str, str] = {}
    for t, tier in panel["tiers"].items():
        if len(set(tier["stage1"])) != 4 or len(tier["stage1"]) != 4:
            p.append(f"{t}: Stage 1 must hold 4 distinct instances")
        if len(set(tier["holdout"])) != 2 or len(tier["holdout"]) != 2:
            p.append(f"{t}: holdout must hold 2 distinct instances")
        if len(set(tier.get("dev", []))) != len(tier.get("dev", [])):
            p.append(f"{t}: development instances must be distinct")
        for part in ("dev", "stage1", "holdout"):
            for nm in tier.get(part, []):
                if nm not in refs:
                    p.append(f"{t}/{part}: {nm} has no reference entry")
                    continue
                for key in (f"name:{nm}", f"sha:{refs[nm]['instance_sha256']}"):
                    if key in seen and seen[key] != f"{t}/{part}/{nm}":
                        p.append(f"{t}/{part}/{nm} duplicates {seen[key]} ({key.split(':')[0]})")
                    seen.setdefault(key, f"{t}/{part}/{nm}")
    return p


def freeze(args) -> None:
    """Writes <family>_panel_frozen.json next to the (immutable) draft: eps_p per tier from a reviewed file
    {"family": {"tier": "eps"}}, the draft panel's sha256, the hashes of the development proposals
    (scaleup_cg_stage1.py --freeze-eps output, which must contain eps inside its interval), and the code commit.
    Refuses: a tier without eps_p, eps outside the proposal interval, splits that are not distinct/disjoint, or code
    that differs from the draft's (= the code of the development runs) unless --allow-code-change gives a reason."""
    from fractions import Fraction
    eps = json.loads(Path(args.freeze).read_text())
    proposals = {}
    for f in args.proposals or []:
        raw = Path(f).read_bytes()
        d = json.loads(raw)
        proposals[d["panel"]] = {"file": Path(f).name, "sha256": hashlib.sha256(raw).hexdigest(), "content": d}
    for fam in args.families:
        p = args.out / f"{fam}_panel.json"
        raw = p.read_bytes()
        panel = json.loads(raw)
        prop = proposals.get(p.name)
        if prop is None:
            raise SystemExit(f"{fam}: no development proposal for {p.name} (--proposals)")
        if prop["content"].get("panel_sha256") != hashlib.sha256(raw).hexdigest():
            raise SystemExit(f"{fam}: the proposal was computed on another draft panel")
        for t, tier in panel["tiers"].items():
            v = (eps.get(fam) or {}).get(t)
            if v is None:
                raise SystemExit(f"{fam}/{t}: no eps_p in {args.freeze}")
            iv = prop["content"]["tiers"][t]["proposal"].get("interval")
            if not iv or not (Fraction(iv[0]) <= Fraction(str(v)) and (iv[1] is None or Fraction(str(v)) < Fraction(iv[1]))):
                raise SystemExit(f"{fam}/{t}: eps {v} is outside the development interval {iv}")
            tier["eps_p"] = str(v)
            tier["eps_p_interval"] = iv
        bad = split_problems(panel)
        if bad:
            raise SystemExit(f"{fam}: {bad}")
        now = code_hashes(fam)
        changed = sorted(f for f in set(now) | set(panel["code_sha256_at_preparation"])
                         if now.get(f) != panel["code_sha256_at_preparation"].get(f))
        if changed and not args.allow_code_change:
            raise SystemExit(f"{fam}: code changed since the development runs: {changed}")
        panel["code_changes_since_development"] = {"files": changed, "reason": args.allow_code_change} if changed else None
        panel["development"] = {"draft_panel": p.name, "draft_panel_sha256": hashlib.sha256(raw).hexdigest(),
                                "draft_code_sha256": panel["code_sha256_at_preparation"],
                                "proposal": {k: prop[k] for k in ("file", "sha256")}}
        panel["code_sha256_at_preparation"] = now
        panel["code_commit"] = args.commit
        panel["status"] = f"FROZEN for Stage 1 at {args.commit}"
        q = args.out / f"{fam}_panel_frozen.json"
        if q.exists():
            raise SystemExit(f"{q} exists: a frozen panel is never overwritten")
        q.write_text(json.dumps(panel, indent=1) + "\n")
        print(fam, "frozen", q, hashlib.sha256(q.read_bytes()).hexdigest())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", type=Path, default=ROOT.parent / "DecompBench" / "outputs" / "scaleup" / "raw")
    ap.add_argument("--refs", type=Path, default=ROOT.parent / "DecompBench" / "outputs" / "scaleup" / "references")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "scaleup" / "cg_v4")
    ap.add_argument("--families", nargs="+", default=["cvrp", "clrp", "bppc"])
    ap.add_argument("--clrp-holdout", nargs="*", default=None, help="two Prodhon stems for the CLRP holdout (user decision: no fresh Prodhon 100-customer instance exists; the freeze refuses stems already used as dev or Stage 1)")
    ap.add_argument("--freeze", metavar="EPS_JSON")
    ap.add_argument("--commit")
    ap.add_argument("--proposals", nargs="*", help="eps_proposal_<family>.json files from --freeze-eps")
    ap.add_argument("--allow-code-change", metavar="REASON")
    args = ap.parse_args()
    if args.freeze:
        if not args.commit:
            ap.error("--freeze needs --commit")
        freeze(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
