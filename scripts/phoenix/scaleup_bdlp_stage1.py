"""BD-LP bound-side Stage 1 / holdout runner for UFL (Koerkel-Ghosh, N = 3000-4000) and two-stage stochastic CFL
(OR-Library cap11x + Bodur scenarios, S = 5000), protocol v4 sections 3, 6, 14-16. Phoenix only (no optimization on
the laptop except the tiny smoke tests of tests/large_scale/test_bdlp_stage1.py through the serial wrapper).

It generalizes the k-median Stage 1 runner (scripts/phoenix/scaleup_km_stage1.py, frozen; its family-independent,
already reviewed helpers are imported from it unchanged: atomic writes, attempt directories, exit classification,
payload schema, OOM evidence, tier decision, checkpoint labels) to a family adapter (UFLFamily, SCFLFamily below).

Modes (FAMILY in {ufl, scfl}; PANEL defaults to scripts/phoenix/panels/<family>_stage1_panel.json)
  --task FAMILY INSTANCE ARM
        One arm in its own process tree (new session). The supervisor starts its clock just before spawning the
        child, reads length-framed messages from the child's stdout ("BDLPMSG <kind> <nbytes>\\n" + payload:
        incumbent, bound, certificate, result), timestamps each on complete receipt, persists it at once
        (events.jsonl appended and fsynced; certificate versions as immutable files certs/v<seq>.json with their
        sha256) and SIGKILLs the whole process group at T. Then, outside T: (1) incumbents received strictly before T
        are evaluated exactly (family evaluator; the SCFL evaluator is capped at the first and the last
        EVAL_CAP - 1 distinct designs, the skipped count is recorded) and the best one is verified independently;
        (2) the certificate version current at each frozen checkpoint (60/300/600/900 s: the last complete stored
        version received strictly before it) is converted to the canonical format and checked by
        src/experiments/xl_certificate.check_versions_supervised (forked checker, one hard per-attempt deadline,
        final version first, the panel's frozen limits). Every run writes a fresh directory claimed atomically
        (OUT/<instance>__<arm>, a retry OUT/<instance>__<arm>/attempt_<k>); result.json is replaced atomically.
  --reference FAMILY INSTANCE
        Frozen LP* interval: the engineered LP phase (eng configuration, conv_tol 1e-9, a certificate every 60 s) for
        --T seconds in process; LP_lo = the canonical certified bound of its last certificate, LP_hi = the exact
        compact-LP objective of its last LP point (family exact routine: g_bound_xl.exact_master_value). Written to
        OUT/<instance>__lp_reference.json with the certificate and point files and their hashes.
  --gen FAMILY OUTDIR [--raw DIR]
        Generates every instance the panel lists (Stage 1, holdout, fallback tiers) with the pinned family
        generator (src/bd/ufl/a_data_xl.py; src/bd/scfl/a_data_xl.py, whose OR-Library base file must match the
        panel's frozen sha256) and writes OUTDIR/<family>_instances_manifest.json (name, sha256, recipe check).
  --summarize FAMILY
        Per instance and arm: certified bound per checkpoint, bgap_LP against the frozen interval, bound_target,
        t_bound, primal metrics (reported; gated only if the panel's rules.primal_gate is true), and the frozen
        acceptance (panel). Result identity (instance, arm, T, panel sha256, instance sha256, frozen code hashes) is
        enforced; only the panel's arms are read; exactly one attempt per arm is scored (the earliest complete one
        with a valid identity and no infrastructure marker). Tiers are decided in the panel's order; a fallback tier
        is read only after its primary tier was rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_km():
    """The frozen k-median runner as a helper library (read-only use; it is never modified from here)."""
    spec = importlib.util.spec_from_file_location("_km_stage1_helpers", ROOT / "scripts" / "phoenix" / "scaleup_km_stage1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


km = _load_km()
write_json_atomic, claim_attempt, attempt_index = km.write_json_atomic, km.claim_attempt, km.attempt_index
code_commit, oom_kill_count, _read_exact = km.code_commit, km.oom_kill_count, km._read_exact
classify_run, payload_problem, tier_decision = km.classify_run, km.payload_problem, km.tier_decision
final_label, _bgap, arm_kwargs = km.final_label, km._bgap, km.arm_kwargs

MAGIC = b"BDLPMSG"
CHECKPOINTS = (60.0, 300.0, 600.0, 900.0)
MESSAGE_KINDS = ("incumbent", "bound", "certificate", "result")
MAX_HEADER = 256
MAX_MESSAGE_BYTES = 256 * 2**20
CHECKER_BUDGET = 600.0
CERTIFY_BUDGET = 1500.0      # one hard deadline for view build + conversion setup + the 600 s supervised checker
TASK_ALLOCATION = 3600.0     # Slurm time per arm task (scaleup_bdlp_stage1.sbatch)
TASK_SAFETY = 120.0          # kept free at the end of the allocation for the final atomic write
REFERENCE_CHECK_BUDGET = 1800.0
REFERENCE_LP_HI_BUDGET = 1800.0
GUROBI_MEM_GB = 15.5
PUBLICATION_VOLUME_CAP = 4 * 2**30
RUNNER = "scripts/phoenix/scaleup_bdlp_stage1.py"
PANEL_DIR = ROOT / "scripts" / "phoenix" / "panels"
COMMON_CODE = [RUNNER, "scripts/phoenix/scaleup_km_stage1.py", "scripts/phoenix/scaleup_bdlp_stage1.sbatch",
               "scripts/phoenix/scaleup_bdlp_panel.py", "src/experiments/xl_certificate.py", "src/bd/scfl/b_mip_xl.py"]


def default_panel(family: str) -> Path:
    return PANEL_DIR / f"{family}_stage1_panel.json"


def stem(path: str) -> str:
    return Path(path).stem


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- family adapters

class _Converter:
    """Family certificate format -> canonical bytes, run INSIDE the supervised checker. The family payload is validated
    strictly first (`validate`: format, view, exact shapes, index ranges, distinct ids per cut, finite numbers; Codex
    round 1 #4); malformed input rejects the version with that reason (charged to the certificate); MemoryError passes
    to the checker's memory gate."""

    def __init__(self, validate, fn) -> None:
        self.validate, self.fn = validate, fn

    def __call__(self, data: bytes) -> bytes:
        import src.experiments.xl_certificate as xc
        try:
            obj = json.loads(data)
            problem = self.validate(obj)
            if problem:
                raise xc.CertificateRejected(f"malformed: {problem}"[:200])
            return self.fn(data)
        except MemoryError:
            raise
        except xc.CertificateRejected:
            raise
        except (ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:   # (a) malformed
            raise xc.CertificateRejected(f"malformed: {type(exc).__name__}: {exc}"[:200]) from None


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)) and math.isfinite(x)


def validate_ufl_cert(obj, n: int) -> str | None:
    """ufl-cert-c1: {"format", "view": "ufl-v1", "version", "singles": {"j": [int], "v": [num]} (equal lengths),
    "multi": [{"ids": [distinct int], "v": [num]} (equal lengths)]}; customer indices in [0, n)."""
    if not isinstance(obj, dict) or obj.get("format") != "ufl-cert-c1" or obj.get("view") != "ufl-v1":
        return "format/view"
    sg = obj.get("singles")
    if not isinstance(sg, dict) or not isinstance(sg.get("j"), list) or not isinstance(sg.get("v"), list) or len(sg["j"]) != len(sg["v"]):
        return "singles: j and v must be lists of equal length"
    if not all(_is_int(j) and 0 <= j < n for j in sg["j"]) or not all(_is_num(v) for v in sg["v"]):
        return "singles: customer index out of range or non-finite value"
    multi = obj.get("multi", [])
    if not isinstance(multi, list):
        return "multi must be a list"
    for c in multi:
        if not isinstance(c, dict) or not isinstance(c.get("ids"), list) or not isinstance(c.get("v"), list) \
                or len(c["ids"]) != len(c["v"]) or not c["ids"]:
            return "multi cut: ids and v must be nonempty lists of equal length"
        if not all(_is_int(j) and 0 <= j < n for j in c["ids"]) or len(set(c["ids"])) != len(c["ids"]):
            return "multi cut: customer ids out of range or duplicated"
        if not all(_is_num(v) for v in c["v"]):
            return "multi cut: non-finite value"
    return None


def validate_scfl_cert(obj, S: int, I: int, J: int) -> str | None:
    """scfl-cert-c1: {"format", "view": "scfl-v1", "version", "cuts": [{"terms": [[s, v (J numbers), alpha (I
    numbers)], ...]}]}; scenario ids in [0, S), distinct within a cut, nonempty cuts."""
    if not isinstance(obj, dict) or obj.get("format") != "scfl-cert-c1" or obj.get("view") != "scfl-v1":
        return "format/view"
    cuts = obj.get("cuts")
    if not isinstance(cuts, list):
        return "cuts must be a list"
    for c in cuts:
        terms = c.get("terms") if isinstance(c, dict) else None
        if not isinstance(terms, list) or not terms:
            return "a cut needs a nonempty terms list"
        seen = set()
        for t in terms:
            if not isinstance(t, list) or len(t) != 3 or not _is_int(t[0]) or not 0 <= t[0] < S or t[0] in seen:
                return "term: [scenario id in range, distinct within the cut, v, alpha]"
            seen.add(t[0])
            if not isinstance(t[1], list) or len(t[1]) != J or not all(_is_num(x) for x in t[1]):
                return f"term: v must be {J} finite numbers"
            if not isinstance(t[2], list) or len(t[2]) != I or not all(_is_num(x) for x in t[2]):
                return f"term: alpha must be {I} finite numbers"
    return None


class UFLFamily:
    name = "ufl"
    ENG, NAIVE, PRIMAL_CONTROL, PRIMAL_NAIVE = "eng", "naive_lp", "ils", "naive_int"
    BOUND_CONTROLS = ["lp_bar", "lp_dual", "empty", "analytical", "oneshot_greedy", "pool", "ils"]
    DIAGNOSTICS = ["eng_plain", "eng_nostab", "naive_int", "lagr", "mip"]
    ALL_ARMS = ["eng", "eng_plain", "eng_nostab", "naive_lp", "naive_int", "lp_bar", "lp_dual", "empty", "analytical",
                "oneshot_greedy", "pool", "ils", "lagr", "mip"]
    EVAL_CAP = None
    ENG_KW = {"lagr_warm": 0.1, "lagr_cap": 60.0, "purge_age": 5}
    CODE_FILES = ["src/bd/ufl/g_bound_xl.py", "src/bd/ufl/a_data_xl.py", "src/bd/ufl/i_controls_xl.py",
                  "src/bd/ufl/b_ref_bd_xl.py"]

    def arms(self) -> dict:
        from src.bd.ufl import g_bound_xl as g
        from src.bd.ufl.i_controls_xl import UFLAnalytical
        mem = {"mem_limit_gb": GUROBI_MEM_GB}
        return {"eng": (g.UFLEngLP, dict(self.ENG_KW)), "eng_plain": (g.UFLEngLP, {}),
                "eng_nostab": (g.UFLEngLP, {**self.ENG_KW, "stabilize": False}), "naive_lp": (g.UFLNaiveLP, {}),
                "naive_int": (g.UFLNaiveInt, {}), "lp_bar": (g.UFLCompactLP, {"method": 2, "lean": True, **mem}),
                "lp_dual": (g.UFLCompactLP, {"method": 1, "lean": True, **mem}), "empty": (g.UFLStatic, {"kind": "empty"}),
                "analytical": (UFLAnalytical, {}), "oneshot_greedy": (g.UFLStatic, {"kind": "oneshot_greedy"}),
                "pool": (g.UFLPool, {}), "ils": (g.UFLILS, {}), "lagr": (g.UFLLagrangian, {}),
                "mip": (g.UFLCompactMIP, {"lean": True, **mem})}

    def reference_arm(self):
        from src.bd.ufl import g_bound_xl as g
        return g.UFLEngLP, {**self.ENG_KW, "conv_tol": 1e-9, "cert_interval": 60.0}

    def empty_certificate(self) -> bytes:
        from src.bd.ufl.g_bound_xl import cert_bytes
        return cert_bytes(version=1)

    def test_design(self, path: str) -> list[int]:
        return [0]

    def evaluator(self, path: str):
        from src.bd.ufl.g_bound_xl import design_value, load_instance
        inst = load_instance(path)

        class _Ev:
            def objective(self, design):
                return design_value(inst, design)

            def close(self):
                pass
        return _Ev()

    def verify(self, path: str, design, value: float) -> dict:
        """Independent recomputation from the raw file in Python integers (no NumPy, no solver)."""
        obj = json.loads(Path(path).read_text())
        m, n = int(obj["m"]), int(obj["n"])
        idx = sorted({int(i) for i in design})
        if not idx or idx[0] < 0 or idx[-1] >= m:
            return {"ok": False, "issues": ["design out of range or empty"], "objective_value": None}
        rows = [obj["D"][i] for i in idx]
        tot = sum(int(obj["f"][i]) for i in idx) + sum(min(int(r[j]) for r in rows) for j in range(n))
        return {"ok": True, "issues": [], "objective_value": float(tot)}

    def view(self, path: str):
        import src.experiments.xl_certificate as xc
        return xc.ufl_view(path)

    def converter(self, path: str) -> _Converter:
        from src.bd.ufl.g_bound_xl import Sorted, load_instance, to_generic
        inst = load_instance(path)
        srt = Sorted(inst.D)
        return _Converter(lambda obj: validate_ufl_cert(obj, inst.n), lambda data: to_generic(data, inst, srt))

    def lp_hi(self, path: str, point, witness: Path | None = None) -> tuple[Fraction | None, dict]:
        """No witness file needed: the exact value is a pure integer computation from the saved point (no solver)."""
        from src.bd.ufl.g_bound_xl import Sorted, exact_master_value, load_instance
        inst = load_instance(path)
        return exact_master_value(inst, Sorted(inst.D), point)

    def generate(self, spec: dict, outdir: Path, raw: Path | None) -> dict:
        from src.bd.ufl.a_data_xl import check_recipe, make_kg, write_instance
        g = spec["generator"]
        obj = make_kg(int(g["N"]), g["class"], int(g["seed"]), bool(g["symmetric"]))
        problems = check_recipe(obj)
        h = write_instance(obj, outdir / f"{obj['name']}.json")
        return {"name": obj["name"], "sha256": h, "recipe_problems": problems}


class SCFLFamily:
    name = "scfl"
    ENG, NAIVE, PRIMAL_CONTROL, PRIMAL_NAIVE = "eng", "naive_lp", "heur", "naive_int"
    BOUND_CONTROLS = ["lp_bar", "lp_dual", "jensen", "empty", "analytical", "oneshot_greedy", "oneshot_saa", "pool", "heur"]
    DIAGNOSTICS = ["eng_plain", "eng_nostab", "naive_int", "lagr", "mip"]
    ALL_ARMS = ["eng", "eng_plain", "eng_nostab", "naive_lp", "naive_int", "lp_bar", "lp_dual", "jensen", "empty",
                "analytical", "oneshot_greedy", "oneshot_saa", "pool", "heur", "lagr", "mip"]
    EVAL_CAP = 25
    ENG_KW = {"jensen_seed": 50}
    CODE_FILES = ["src/bd/scfl/g_bound_xl.py", "src/bd/scfl/h_view_xl.py", "src/bd/scfl/a_data_xl.py",
                  "src/bd/scfl/i_controls_xl.py", "src/bd/scfl/a_data.py", "src/bd/scfl/c_naive_bd.py",
                  "src/bd/scfl/e_verifier.py", "src/bd/scfl/f_heuristic.py"]

    def arms(self) -> dict:
        from src.bd.scfl import g_bound_xl as g
        from src.bd.scfl.b_mip_xl import SCFLExtensiveMIP
        from src.bd.scfl.c_naive_bd import SCFLNaiveBenders
        from src.bd.scfl.i_controls_xl import SCFLAnalytical, SCFLHeurOneShot
        mem = {"mem_limit_gb": GUROBI_MEM_GB}
        return {"eng": (g.SCFLEngLP, dict(self.ENG_KW)), "eng_plain": (g.SCFLEngLP, {}),
                "eng_nostab": (g.SCFLEngLP, {**self.ENG_KW, "stabilize": False}), "naive_lp": (g.SCFLNaiveLP, {}),
                "naive_int": (SCFLNaiveBenders, {}), "lp_bar": (g.SCFLCompactLP, {"method": 2, "lean": True, **mem}),
                "lp_dual": (g.SCFLCompactLP, {"method": 1, "lean": True, **mem}),
                "jensen": (g.SCFLJensen, {"lean": True, **mem}), "empty": (g.SCFLStatic, {"kind": "empty"}),
                "analytical": (SCFLAnalytical, {}), "oneshot_greedy": (g.SCFLStatic, {"kind": "oneshot_greedy"}),
                "oneshot_saa": (g.SCFLStatic, {"kind": "oneshot_saa"}), "pool": (g.SCFLPool, {}),
                "heur": (SCFLHeurOneShot, {"mode": "saa"}), "lagr": (g.SCFLLagrangian, {}),
                "mip": (SCFLExtensiveMIP, {"strong": True, **mem})}

    def reference_arm(self):
        from src.bd.scfl import g_bound_xl as g
        return g.SCFLEngLP, {**self.ENG_KW, "conv_tol": 1e-9, "cert_interval": 60.0}

    def empty_certificate(self) -> bytes:
        from src.bd.scfl.h_view_xl import cert_bytes
        return cert_bytes([], version=1)

    def test_design(self, path: str) -> list[int]:
        return list(range(len(json.loads(Path(path).read_text())["capacity"])))

    def evaluator(self, path: str):
        from src.bd.scfl.a_data import load_instance
        from src.bd.scfl.b_mip_xl import DesignEvaluator
        ev = DesignEvaluator(load_instance(path))

        class _Ev:
            def objective(self, design):
                try:
                    return ev.objective(design)
                except (ValueError, IndexError):   # an out-of-range design: invalid (counted by the caller)
                    return None

            def close(self):
                ev.close()
        return _Ev()

    def verify(self, path: str, design, value: float) -> dict:
        """The stdlib min-cost-flow verifier src/bd/scfl/e_verifier.py on the design's output file."""
        from src.bd.scfl.a_data import load_instance
        from src.bd.scfl.e_verifier import verify_solution
        from src.bd.scfl.b_mip_xl import design_vector
        inst = load_instance(path)
        out = {"status": "feasible", "method": "stage1", "objective_value": float(value),   # the claimed value is the
               "y": design_vector(inst, design).astype(int).tolist()}                      # evaluator's; re-derived below
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "out.json"
            f.write_text(json.dumps(out))
            vr = verify_solution(path, f)
        return {"ok": vr.get("ok"), "issues": (vr.get("issues") or [])[:10], "objective_value": vr.get("objective_value")}

    def view(self, path: str):
        from src.bd.scfl.h_view_xl import scfl_view
        return scfl_view(path)

    def converter(self, path: str) -> _Converter:
        from src.bd.scfl.a_data import load_instance
        from src.bd.scfl.h_view_xl import to_generic
        inst = load_instance(path)
        cost = inst.cost
        return _Converter(lambda obj: validate_scfl_cert(obj, inst.S, inst.I, inst.J), lambda data: to_generic(data, cost))

    def lp_hi(self, path: str, point, witness: Path | None = None) -> tuple[Fraction | None, dict]:
        from src.bd.scfl.a_data import load_instance
        from src.bd.scfl.g_bound_xl import exact_master_value
        return exact_master_value(path, load_instance(path), point, witness=str(witness) if witness else None,
                                  scenario_seconds=60.0)

    def generate(self, spec: dict, outdir: Path, raw: Path | None) -> dict:
        from src.bd.scfl.a_data_xl import make_instance, read_cap, write_instance
        g = spec["generator"]
        if raw is None:
            raise ValueError("--raw (the OR-Library CAP directory) is required for scfl")
        base_file = raw / f"{g['base']}.txt"
        got = sha256(base_file.read_bytes())
        if got != g["base_sha256"]:
            raise ValueError(f"{base_file}: sha256 {got} != frozen {g['base_sha256']}")
        obj = make_instance(read_cap(base_file), int(g["S"]), int(g["seed"]))
        h = write_instance(obj, outdir / f"{obj['name']}.json")
        return {"name": obj["name"], "sha256": h, "recipe_problems": scfl_domain_problems(obj), "base_sha256": got}


def scfl_domain_problems(obj: dict) -> list[str]:
    """Frozen fractional-domain invariant of an SCFL panel instance (Codex round 1 #5): with strong linking, the
    recourse of scenario s is feasible at a fractional y iff sum_i min(u_i, D_s) y_i >= D_s. If max_i u_i <= D_s for
    every scenario with D_s > 0, then min(u_i, D_s) = u_i and the view's single cover row sum_i u_i y_i >= D_max implies
    every scenario's condition: the master domain is exactly the recourse-feasible set. Checked as max_i u_i <= the
    smallest positive scenario total. A violation would let penalized Benders converge below LP*."""
    cap = [float(u) for u in obj["capacity"]]
    tot = [float(sum(row)) for row in obj["demand"]]
    pos = [d for d in tot if d > 0]
    if pos and max(cap) > min(pos):
        return [f"fractional-domain invariant violated: max capacity {max(cap)} > smallest positive scenario demand {min(pos)}"]
    return []


FAMILIES = {"ufl": UFLFamily, "scfl": SCFLFamily}


def family(name: str):
    if name not in FAMILIES:
        raise SystemExit(f"unknown family {name!r} (ufl or scfl)")
    return FAMILIES[name]()


# ---------------------------------------------------------------- test arms (failure injection; never in a panel)

class _Sleeper:
    """Publishes one incumbent and the empty certificate, then never returns (the supervisor must kill it)."""

    def __init__(self, *, fam, time_limit: float, on_incumbent, on_certificate, **kw) -> None:
        self.fam, self.on_incumbent, self.on_certificate = fam, on_incumbent, on_certificate

    def run(self, data_path: str) -> dict:
        self.on_incumbent(self.fam.test_design(data_path), stage="sleeper")
        self.on_certificate(self.fam.empty_certificate())
        while True:
            time.sleep(3600)


class _Crash:
    def __init__(self, *, fam, time_limit: float, on_incumbent, **kw) -> None:
        self.fam, self.on_incumbent = fam, on_incumbent

    def run(self, data_path: str) -> dict:
        self.on_incumbent(self.fam.test_design(data_path), stage="test")
        raise RuntimeError("injected arm crash")


def arm_table(fam) -> dict:
    return {**fam.arms(), "_test_sleeper": (_Sleeper, {"fam": fam}), "_test_crash": (_Crash, {"fam": fam})}


# ---------------------------------------------------------------- child

def child(fam_name: str, instance: str, arm: str, T: float, start_wall: float) -> None:
    try:   # the arm, not the supervisor, is the kernel's OOM victim inside the job's memory cgroup
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:   # not Linux or not permitted: OOM evidence then decides nothing (infrastructure)
        pass
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)                 # stray prints go to stderr, never into the framed channel
    sys.stdout = sys.stderr
    lock = threading.Lock()

    def write_all(data: bytes) -> None:
        view = memoryview(data)
        while view:
            view = view[out.write(view):]

    def send(kind: str, payload: bytes) -> None:
        with lock:
            write_all(MAGIC + b" " + kind.encode() + b" " + str(len(payload)).encode() + b"\n" + payload)

    def on_inc(design, stage=None):
        send("incumbent", json.dumps({"design": [int(i) for i in design], "stage": stage}).encode())

    def on_bound(b):
        send("bound", json.dumps({"value": float(b)}).encode())

    def on_cert(data: bytes):
        send("certificate", bytes(data))

    fam = family(fam_name)
    cls, kw = arm_table(fam)[arm]
    left = T - (time.time() - start_wall) - 2.0
    kwargs = arm_kwargs(cls, dict(time_limit=max(1.0, left), threads=1, on_incumbent=on_inc, on_bound=on_bound,
                                  on_certificate=on_cert, **kw))
    res = cls(**kwargs).run(instance) or {}
    slim = {k: v for k, v in res.items() if not isinstance(v, (list, dict)) and k not in ("trace", "cut_log")}
    if isinstance(res.get("stats"), dict):
        slim["stats"] = {k: v for k, v in res["stats"].items() if not isinstance(v, (list, dict))}
    for key in ("solves", "simplex_bounds", "heuristic"):
        if res.get(key) is not None:
            slim[key] = res[key]
    send("result", json.dumps(slim, default=str).encode())


# ---------------------------------------------------------------- supervisor (as scaleup_km_stage1.supervise)

def supervise(fam_name: str, instance: str, arm: str, T: float, outdir: Path,
              volume_cap: int = PUBLICATION_VOLUME_CAP) -> tuple[list[dict], dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    if any(outdir.iterdir()):
        raise FileExistsError(f"refusing to reuse the non-empty arm output directory {outdir}")
    certs = outdir / "certs"
    certs.mkdir()
    events = open(outdir / "events.jsonl", "xb")
    records: list[dict] = []
    torn: list[str] = []
    malformed = [0]
    violations: list[str] = []
    terminal: list[int] = []
    reader_errors: list[str] = []
    stored = [0]
    over_volume = [0]
    oom_before = oom_kill_count()
    start_wall = time.time()
    t0 = time.monotonic()
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child", fam_name, instance, arm, "--T", str(T),
                             "--start", repr(start_wall)], stdout=subprocess.PIPE, stderr=open(outdir / "stderr.log", "xb"),
                            start_new_session=True, cwd=str(ROOT))

    def violation(what: str) -> None:
        violations.append(what[:300])

    def read_all() -> None:
        f = proc.stdout
        seq = 0
        while True:
            hdr = f.readline(MAX_HEADER)
            if not hdr:
                break
            if not hdr.endswith(b"\n"):
                if len(hdr) >= MAX_HEADER:
                    violation(f"header longer than {MAX_HEADER} bytes")
                else:
                    torn.append("header")
                break
            parts = hdr.split()
            if len(parts) != 3 or parts[0] != MAGIC or not parts[2].isdigit():
                violation(f"bad header {hdr[:60]!r}")
                break
            kind, nbytes = parts[1].decode(errors="replace"), int(parts[2])
            if kind not in MESSAGE_KINDS:
                violation(f"unknown message kind {kind!r}")
                break
            if nbytes > MAX_MESSAGE_BYTES:
                violation(f"{kind} message of {nbytes} bytes")
                break
            payload = _read_exact(f, nbytes)
            if payload is None:
                torn.append(kind)
                break
            t = time.monotonic() - t0
            seq += 1
            if terminal:
                violation(f"{kind} message after the terminal result")
            rec = {"seq": seq, "t": round(t, 6), "kind": kind, "bytes": nbytes, "sha256": sha256(payload)}
            if kind == "certificate" and stored[0] + nbytes > volume_cap:
                rec["file"], rec["over_volume"] = None, True
                over_volume[0] += 1
            elif kind == "certificate":
                fname = f"v{seq}.json"
                with open(certs / fname, "xb") as fh:
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                rec["file"] = fname
                stored[0] += nbytes
            else:
                problem = payload_problem(kind, payload)
                if problem:
                    violation(f"{kind} message {seq}: {problem}")
                    rec["payload"], rec["valid"] = None, False
                    malformed[0] += 1
                else:
                    rec["payload"] = json.loads(payload)
                if kind == "result":
                    terminal.append(seq)
            events.write(json.dumps(rec).encode() + b"\n")
            events.flush()
            os.fsync(events.fileno())
            records.append(rec)

    def reader() -> None:
        try:
            read_all()
        except BaseException as exc:  # noqa: BLE001 - (b) recorded; the arm's result is infrastructure
            reader_errors.append(f"{type(exc).__name__}: {exc}"[:300])

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    killed = False
    try:
        proc.wait(timeout=max(0.0, T - (time.monotonic() - t0)))
    except subprocess.TimeoutExpired:   # the normal end of an arm: hard kill at T
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:   # benign: the process group already exited
            pass
        killed = True
        proc.wait()
    wall = time.monotonic() - t0
    th.join(timeout=60)
    if th.is_alive():
        reader_errors.append("publication reader did not finish")
    try:
        events.close()
    except OSError as exc:   # (b) infrastructure
        reader_errors.append(f"events.jsonl close: {exc}"[:300])
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    oom_after = oom_kill_count()
    oom = (not killed and proc.returncode == -signal.SIGKILL and oom_before is not None and oom_after is not None
           and oom_after > oom_before)
    run = {"returncode": proc.returncode, "killed_at_T": killed, "wall_seconds": round(wall, 3), "torn_messages": len(torn),
           "torn_kinds": list(torn), "child_peak_rss_mb": round(peak, 1), "reader_errors": list(reader_errors),
           "malformed_messages": malformed[0], "protocol_violations": list(violations), "terminal_result": bool(terminal),
           "certificate_bytes_stored": stored[0], "over_volume_certificates": over_volume[0], "volume_cap_bytes": volume_cap,
           "oom_kill_before": oom_before, "oom_kill_after": oom_after, "oom_confirmed": bool(oom),
           "resource_outcome": "memory_limit (kernel-confirmed OOM kill of the arm)" if oom else None}
    run["infrastructure"] = classify_run(run)
    return records, run


# ---------------------------------------------------------------- evaluation and certification (outside T)

def evaluate_incumbents(fam, instance: str, records: list[dict], T: float) -> dict:
    """Exact values of the incumbents received strictly before T (distinct designs; the family's EVAL_CAP keeps the
    first and the last EVAL_CAP - 1 distinct designs in receipt order, the arms publishing strict improvements by their
    own exact evaluation, so the last ones are the candidates for the best), the best verified independently."""
    first: dict[tuple, float] = {}
    order: list[tuple] = []
    late = malformed = 0
    for r in records:
        if r["kind"] != "incumbent":
            continue
        if r.get("valid") is False or not isinstance(r.get("payload"), dict) or "design" not in r["payload"]:
            malformed += 1
            continue
        if r["t"] >= T:
            late += 1
            continue
        key = tuple(sorted(set(int(i) for i in r["payload"]["design"])))
        if key not in first:
            first[key] = r["t"]
            order.append(key)
    skipped = 0
    chosen = order
    if fam.EVAL_CAP is not None and len(order) > fam.EVAL_CAP:
        chosen = order[:1] + order[-(fam.EVAL_CAP - 1):]
        skipped = len(order) - len(chosen)
    traj, best, best_design, invalid = [], None, None, 0
    t0 = time.monotonic()
    ev = fam.evaluator(instance)
    try:
        for key in chosen:
            v = ev.objective(list(key))
            if v is None:
                invalid += 1
                continue
            if best is None or v < best - 1e-9 * max(1.0, abs(v)):
                best, best_design = v, list(key)
                traj.append((first[key], v))
    finally:
        ev.close()
    eval_s = time.monotonic() - t0
    verified, verification = None, None
    if best_design is not None:
        try:
            vr = fam.verify(instance, best_design, best)
        except Exception as exc:  # noqa: BLE001 - (b) the verifier could not run: infrastructure (arm_metrics)
            verification = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        else:
            verification = vr
            verified = bool(vr.get("ok") and vr.get("objective_value") is not None
                            and abs(vr["objective_value"] - best) <= 1e-6 * max(1.0, abs(best)))
    return {"best": best, "t_first": traj[0][0] if traj else None, "t_best": traj[-1][0] if traj else None,
            "verified": verified, "verification": verification, "invalid": invalid, "late": late, "malformed": malformed,
            "distinct": len(order), "skipped_eval": skipped, "eval_seconds": round(eval_s, 2), "trajectory": traj[-60:],
            "design": best_design}


def checkpoint_versions(records: list[dict], T: float) -> list[tuple[float, dict | None]]:
    certs = [r for r in records if r["kind"] == "certificate" and r["t"] < T and r.get("file")]
    cps = sorted({c for c in CHECKPOINTS if c <= T} | {T})
    out = []
    for c in cps:
        cur = None
        for r in certs:
            if r["t"] < c:
                cur = r
        out.append((c, cur))
    return out


def load_panel(panel_path: str | Path | None, fam_name: str) -> tuple[dict, Path]:
    path = Path(panel_path) if panel_path is not None else default_panel(fam_name)
    return (json.loads(path.read_text()) if path.exists() else {}), path


def frozen_checker(panel: dict, budget: float | None = None) -> dict:
    """The panel's frozen canonical-checker limits (xl_certificate.CheckerLimits defaults where the panel is silent)."""
    import src.experiments.xl_certificate as xc
    ch = panel.get("checker", {})
    base = xc.CheckerLimits()
    b = float(ch.get("budget_seconds_per_attempt", CHECKER_BUDGET)) if budget is None else float(budget)
    limits = xc.CheckerLimits(max_bytes=int(ch.get("max_bytes", base.max_bytes)),
                              max_nonzeros=int(ch.get("max_nonzeros_expanded", base.max_nonzeros)),
                              max_master_nonzeros=int(ch.get("max_master_nonzeros", base.max_master_nonzeros)),
                              max_seconds=b, max_abs_coef=float(ch.get("max_abs_coef", base.max_abs_coef)),
                              max_token_chars=int(ch.get("token_chars", base.max_token_chars)),
                              max_exponent=int(ch.get("decimal_exponent", base.max_exponent)),
                              gurobi_mem_gb=ch.get("gurobi_mem_gb", base.gurobi_mem_gb))
    return {"limits": limits, "budget": b, "memory_bytes": ch.get("memory_bytes"),
            "fit_fraction": float(ch.get("memory_fit_fraction", 0.5))}


def certify(fam, instance: str, records: list[dict], outdir: Path, T: float, panel: dict) -> dict:
    """CANONICAL certification (the scored path): family format -> canonical bytes inside the supervised checker."""
    import src.experiments.xl_certificate as xc
    certs = outdir / "certs"
    cps = checkpoint_versions(records, T)
    out: dict = {"checkpoints": [], "versions": sum(1 for r in records if r["kind"] == "certificate")}
    if not any(cur for _, cur in cps):
        out["checkpoints"] = [{"checkpoint": c, "seq": None, "cert_ok": False, "lb": "0", "lb_float": 0.0,
                               "reason": "absent"} for c, _ in cps]
        return out
    fz = frozen_checker(panel)
    out["checker"] = {"budget": fz["budget"], "memory_bytes": fz["memory_bytes"], "fit_fraction": fz["fit_fraction"]}
    t0 = time.monotonic()
    try:
        view = fam.view(instance)
        convert = fam.converter(instance)
        out["view_seconds"] = round(time.monotonic() - t0, 2)
        recs = []
        for c, cur in cps:
            if cur is None:
                recs.append({"checkpoint": c})
            else:
                recs.append({"checkpoint": c, "file": cur.get("file"), "sha256": cur.get("sha256"), "bytes": cur.get("bytes"),
                             "t": cur["t"], "seq": cur["seq"]})
        res = xc.check_versions_supervised(view, recs, store_dir=certs, limits=fz["limits"], budget_seconds=fz["budget"],
                                           memory_bytes=fz["memory_bytes"], fit_fraction=fz["fit_fraction"], convert=convert)
        canon = []
        for d in res:
            d = dict(d)
            d["lb"] = None if d.get("lb") is None else str(d["lb"])
            canon.append({k: v for k, v in d.items() if k != "stats"} | {"stats": {k: v for k, v in (d.get("stats") or {}).items()
                                                                                    if not isinstance(v, (list, dict))}})
        out["checkpoints"] = canon
        out["canonical"] = True
        del view, convert
    except Exception as exc:  # noqa: BLE001 - (b) the canonical view could not be built: infrastructure (rerun)
        out["canonical"] = False
        out["canonical_error"] = f"{type(exc).__name__}: {exc}"[:300]
        out["checkpoints"] = [{"checkpoint": c, "seq": cur["seq"] if cur else None, "receipt_t": cur["t"] if cur else None,
                               "cert_ok": None, "lb": None, "lb_float": None, "reason": "infrastructure",
                               "infrastructure": "canonical_view_failed",
                               "failure": {"stage": "view", "checkpoint": i, "category": "infrastructure"}}
                              for i, (c, cur) in enumerate(cps)]
    return out


def code_files(fam, panel: dict) -> list[str]:
    return sorted(set(panel.get("code_sha256_at_preparation", {})) | set(COMMON_CODE) | set(fam.CODE_FILES))


def identity(fam, task: dict, panel_path: Path) -> dict:
    import platform
    panel = json.loads(panel_path.read_text()) if panel_path.exists() else {}
    code = {}
    for f in code_files(fam, panel):
        try:
            code[f] = sha256((ROOT / f).read_bytes())
        except OSError:   # recorded as missing (an identity mismatch at summarize)
            code[f] = None
    try:
        import gurobipy
        grb = ".".join(map(str, gurobipy.gurobi.version()))
    except Exception:  # noqa: BLE001 - recorded as unknown
        grb = None
    try:
        import numpy
        import scipy
        npv, spv = numpy.__version__, scipy.__version__
    except Exception:  # noqa: BLE001 - recorded as unknown
        npv = spv = None
    return {"python": sys.version.split()[0], "executable": sys.executable, "host": platform.node(), "gurobi": grb,
            "numpy": npv, "scipy": spv,
            "slurm": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_RESTART_COUNT")},
            "argv": sys.argv, "task": task, "family": fam.name,
            "panel": {"file": panel_path.name if panel_path.exists() else None,
                      "sha256": sha256(panel_path.read_bytes()) if panel_path.exists() else None, "status": panel.get("status")},
            "code_sha256": code}


def run_with_deadline(fn, budget: float, max_bytes: int = 64 * 2**20) -> tuple[str, Any]:
    """scaleup_km_stage1.run_with_deadline, except that the forked child leads its own process group and the WHOLE
    group is SIGKILLed when the call ends: a grandchild (e.g. the supervised checker forked by
    check_versions_supervised) never outlives the deadline or competes for memory with the next step. Returns
    ("done", value), ("error", message), ("deadline", None) or ("ipc_error", message)."""
    import select

    if budget <= 0:
        return "deadline", None
    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(rfd)
        try:
            os.setpgid(0, 0)
            data = json.dumps({"ok": fn()}, default=str).encode()
        except BaseException as exc:  # noqa: BLE001 - reported to the parent
            data = json.dumps({"error": f"{type(exc).__name__}: {exc}"[:300]}).encode()
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(wfd, view):]
        finally:
            os._exit(0)
    os.close(wfd)
    try:
        os.setpgid(pid, pid)       # also from the parent (no race with the child's own call)
    except OSError:                # benign: the child already did it, or already exited
        pass
    deadline = time.monotonic() + budget
    buf, status = bytearray(), "running"
    try:
        os.set_blocking(rfd, False)
        poller = select.poll()
        poller.register(rfd, select.POLLIN | select.POLLHUP | select.POLLERR)
        while status == "running":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "deadline"
                break
            if not poller.poll(max(1, min(int(remaining * 1000) + 1, 250))):
                continue
            try:
                chunk = os.read(rfd, 1 << 16)
            except BlockingIOError:   # benign: no data yet
                continue
            if not chunk:
                status = "done"
                break
            buf += chunk
            if len(buf) > max_bytes:
                status = "ipc_error"
    except Exception as exc:  # noqa: BLE001 - recorded
        status = f"ipc_error: {type(exc).__name__}: {exc}"[:300]
    finally:
        for kill in (lambda: os.killpg(pid, signal.SIGKILL), lambda: os.kill(pid, signal.SIGKILL)):
            try:
                kill()
            except OSError:   # already gone
                pass
        try:
            os.waitpid(pid, 0)
        except OSError:   # already reaped
            pass
        os.close(rfd)
    if status != "done":
        return ("deadline", None) if status == "deadline" else ("ipc_error", status)
    try:
        msg = json.loads(bytes(buf))
    except ValueError:   # the child died before writing a complete result
        return "ipc_error", "incomplete result"
    return ("done", msg["ok"]) if "ok" in msg else ("error", msg.get("error"))


def prune_certificates(records: list[dict], outdir: Path, T: float) -> dict:
    """After the canonical result is persisted: delete the stored certificate files that no checkpoint selected (their
    receipt time, size and sha256 stay in events.jsonl); the selected versions are kept as the audit evidence."""
    keep = {cur["file"] for _, cur in checkpoint_versions(records, T) if cur is not None}
    removed, freed = 0, 0
    for r in records:
        f = r.get("file")
        if r["kind"] == "certificate" and f and f not in keep:
            try:
                path = outdir / "certs" / f
                freed += path.stat().st_size
                path.unlink()
                removed += 1
            except OSError:   # benign: reported counts only
                pass
    return {"kept": sorted(keep), "removed": removed, "freed_bytes": freed}


def run_task(fam_name: str, instance: str, arm: str, T: float, out_root: Path, panel_path: str | Path | None = None,
             allocation: float = TASK_ALLOCATION) -> dict:
    """One arm: supervised run; then (outside T) the CANONICAL certification under one hard deadline
    (CERTIFY_BUDGET, a forked child: view build, conversion setup and the supervised checker), persisted at once
    (result.json, incumbents still pending); then the primal evaluation (exact values of the incumbents and the
    independent verification) under a hard deadline that fits the task allocation, and the final atomic write. A primal
    evaluation that cannot finish leaves the primal metrics undefined ("unavailable") without touching the bound
    result (Codex round 1 #3)."""
    fam = family(fam_name)
    if arm not in arm_table(fam):
        raise SystemExit(f"unknown arm {arm!r} for {fam_name}")
    task_t0 = time.monotonic()
    panel, ppath = load_panel(panel_path, fam_name)
    seal_sha = None
    if sealed_part(panel, stem(instance)) is not None:   # holdout / fallback Stage 1: the sealed reference comes first
        seal_sha = seal_sha256(out_root, fam_name, panel, stem(instance))
        if seal_sha is None:
            raise SystemExit(f"{stem(instance)}: no reference seal {seal_path(out_root, fam_name, *sealed_part(panel, stem(instance)))}"
                             " (run the reference job and scaleup_bdlp_panel.py --seal before the arms)")
    k, outdir = claim_attempt(out_root / f"{stem(instance)}__{arm}")
    cap = int(panel.get("publication_volume_cap_bytes", PUBLICATION_VOLUME_CAP))
    records, run = supervise(fam_name, instance, arm, T, outdir, volume_cap=cap)
    t0 = time.monotonic()
    status, value = run_with_deadline(lambda: certify(fam, instance, records, outdir, T, panel), CERTIFY_BUDGET)
    if status == "done":
        cert = value
    else:   # (b) evaluator side: the bound metrics are undefined (infrastructure, rerun), never a weak bound
        cert = {"canonical": False, "canonical_error": f"certification {status}: {value}"[:300], "checkpoints": [],
                "certify_budget": CERTIFY_BUDGET}
    sb = [(r["t"], r["payload"]["value"]) for r in records if r["kind"] == "bound" and r.get("payload") and r["t"] < T]
    result = next((r["payload"] for r in records if r["kind"] == "result"), None)
    row = {"family": fam_name, "instance": stem(instance), "arm": arm, "T": T, "run": run,
           "incumbents": {"best": None, "unavailable": "pending (primal evaluation not finished)"},
           "solver_bounds": {"final": max((b for _, b in sb), default=None),
                             "at_checkpoints": {str(c): max((b for t, b in sb if t < c), default=None)
                                                for c, _ in checkpoint_versions(records, T)},
                             "label": "solver-reported internal bound (never scored; refuted if above z_best)"},
           "certificates": cert, "arm_result": result,
           "identity": identity(fam, {"instance": str(instance), "instance_sha256": sha256(Path(instance).read_bytes()),
                                      "arm": arm, "T": T, "reference_seal_sha256": seal_sha}, ppath) | {"code_commit": code_commit()},
           "attempt": {"index": k, "dir": outdir.relative_to(out_root).as_posix()},
           "certify_seconds": round(time.monotonic() - t0, 2)}
    write_json_atomic(outdir / "result.json", row)          # the bound result is durable from here on
    if cert.get("canonical") is True:
        row["certificates"]["pruned"] = prune_certificates(records, outdir, T)
    t1 = time.monotonic()
    budget = allocation - (time.monotonic() - task_t0) - TASK_SAFETY
    status, value = run_with_deadline(lambda: evaluate_incumbents(fam, instance, records, T), budget)
    if status == "done":
        row["incumbents"] = value
    else:
        row["incumbents"] = {"best": None, "unavailable": f"primal evaluation {status}: {value}"[:300],
                             "budget_seconds": round(budget, 1)}
    row["primal_seconds"] = round(time.monotonic() - t1, 2)
    row["task_seconds"] = round(time.monotonic() - task_t0, 2)
    write_json_atomic(outdir / "result.json", row)
    return row


# ---------------------------------------------------------------- frozen LP* reference

def reference(fam_name: str, instance: str, T: float, out_root: Path, panel_path: str | Path | None = None) -> dict:
    """Frozen LP* interval (module docstring). The reference arm runs in process for T seconds (it publishes a
    certificate every 60 s; only the last is kept); the canonical check and the exact LP_hi each run in a forked child
    under their own hard deadline (REFERENCE_CHECK_BUDGET, REFERENCE_LP_HI_BUDGET; Codex round 1 #3). The record is
    written atomically with its witness files and their hashes."""
    import src.experiments.xl_certificate as xc
    fam = family(fam_name)
    panel, ppath = load_panel(panel_path, fam_name)
    cls, kw = fam.reference_arm()
    last: list[bytes] = []
    t0 = time.monotonic()
    def keep_last(b: bytes) -> None:
        last[:] = [b]

    res = cls(time_limit=T, threads=1, on_certificate=keep_last, **kw).run(instance)
    run_s = time.monotonic() - t0
    out_root.mkdir(parents=True, exist_ok=True)
    name = stem(instance)
    row: dict[str, Any] = {"family": fam_name, "instance": name, "instance_sha256": sha256(Path(instance).read_bytes()),
                           "label": "LP* reference (not an arm)", "T": T, "run_seconds": round(run_s, 1),
                           "termination": res.get("termination_reason"),
                           "stats": {k: v for k, v in (res.get("stats") or {}).items() if not isinstance(v, (list, dict))}}
    lo = hi = None
    if last:
        cpath = out_root / f"{name}__lp_reference_certificate.json"
        cpath.write_bytes(last[-1])
        row["certificate_file"], row["certificate_sha256"] = cpath.name, sha256(last[-1])

        def check() -> dict:
            view = fam.view(instance)
            gen = fam.converter(instance)(last[-1])
            cr = xc.check_certificate(view, gen, limits=frozen_checker(panel, REFERENCE_CHECK_BUDGET)["limits"])
            return {"cert_ok": bool(cr.cert_ok), "lb": str(cr.lb), "lb_float": float(cr.lb_float), "reason": cr.reason,
                    "infrastructure": cr.infrastructure}

        t1 = time.monotonic()
        status, value = run_with_deadline(check, REFERENCE_CHECK_BUDGET)
        row["canonical"] = value if status == "done" else {"cert_ok": False, "error": f"{status}: {value}"[:300]}
        row["canonical_seconds"] = round(time.monotonic() - t1, 1)
        if status == "done" and value.get("cert_ok"):
            lo = Fraction(value["lb"])
    point = res.get("lp_point")
    if point is not None:
        ppt = out_root / f"{name}__lp_reference_point.json"
        ppt.write_text(json.dumps({"instance": name, "source": "eng LP phase, last master LP point (y)",
                                   "point": [repr(float(v)) for v in point]}))
        row["lp_point_file"], row["lp_point_sha256"] = ppt.name, sha256(ppt.read_bytes())
        wit = out_root / f"{name}__lp_reference_witness.npz" if fam_name == "scfl" else None
        pt = [float(v) for v in point]

        def upper() -> dict:
            val, info = fam.lp_hi(instance, pt, wit)
            return {"value": None if val is None else str(val), **info}

        t2 = time.monotonic()
        status, value = run_with_deadline(upper, REFERENCE_LP_HI_BUDGET)
        row["lp_hi"] = value if status == "done" else {"value": None, "error": f"{status}: {value}"[:300]}
        if status == "done" and value.get("witness_file"):
            row["lp_hi"]["witness_sha256"] = sha256((out_root / value["witness_file"]).read_bytes())
        row["lp_hi_seconds"] = round(time.monotonic() - t2, 1)
        if row["lp_hi"].get("value") is not None:
            hi = Fraction(row["lp_hi"]["value"])
    iv: dict[str, Any] = {"lo": None if lo is None else str(lo), "hi": None if hi is None else str(hi),
                          "lo_provenance": "canonical xl_certificate, eng LP-phase last certificate",
                          "hi_provenance": "exact compact-LP objective of the eng LP-phase last master point (exact repair)"}
    if lo is not None and hi is not None:
        iv.update(rigorous=bool(lo <= hi), rel_width=float((hi - lo) / max(abs(hi), Fraction(1))),
                  lo_float=float(lo), hi_float=float(hi))
    else:
        iv["rigorous"] = False
    row["interval"] = iv
    row["identity"] = identity(fam, {"instance": str(instance), "instance_sha256": row["instance_sha256"], "mode": "reference",
                                     "T": T}, ppath) | {"code_commit": code_commit()}
    row["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    write_json_atomic(out_root / f"{name}__lp_reference.json", row)
    return row


# ---------------------------------------------------------------- instance generation

def gen(fam_name: str, outdir: Path, raw: Path | None, panel_path: str | Path | None = None) -> dict:
    fam = family(fam_name)
    panel, ppath = load_panel(panel_path, fam_name)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, spec in panel["instances"].items():
        r = fam.generate(spec, outdir, raw)
        if r["name"] != name:
            raise ValueError(f"generated {r['name']} for the panel instance {name}")
        rows[name] = r
        print("GEN " + json.dumps(r), flush=True)
    manifest = {"family": fam_name, "panel": ppath.name, "panel_sha256": sha256(ppath.read_bytes()),
                "code_commit": code_commit(), "instances": rows}
    write_json_atomic(outdir / f"{fam_name}_instances_manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------- summary and frozen acceptance

def sealed_part(panel: dict, name: str) -> tuple[str, str] | None:
    """(tier, part) when the instance's LP* reference is read from the results directory under a seal (every holdout,
    and the Stage 1 of a fallback tier), else None (primary Stage 1: frozen in the panel)."""
    spec = (panel.get("instances") or {}).get(name) or {}
    tier, part = spec.get("tier"), spec.get("part")
    if tier is None:
        for t, ts in (panel.get("tiers") or {}).items():
            for pt in ("stage1", "holdout"):
                if name in ts.get(pt, []):
                    tier, part = t, pt
    if tier is None:
        return None
    if part == "holdout" or "run_if_rejected" in (panel.get("tiers") or {}).get(tier, {}):
        return tier, part
    return None


def seal_sha256(out_root: Path, fam_name: str, panel: dict, name: str) -> str | None:
    sp = sealed_part(panel, name)
    if sp is None:
        return None
    f = seal_path(out_root, fam_name, *sp)
    return sha256(f.read_bytes()) if f.is_file() else None


def identity_problems(r: dict, name: str, arm: str, panel: dict, panel_sha: str, seal_sha: str | None = None) -> list[str]:
    """scaleup_km_stage1.identity_problems with the family, the runtime pins and, for a sealed part (holdout, fallback
    Stage 1), the sha256 of the reference seal the arm was started under (Codex round 2 #3)."""
    p = km.identity_problems(r, name, arm, panel, panel_sha)
    if sealed_part(panel, name) is not None:
        got = ((r.get("identity") or {}).get("task") or {}).get("reference_seal_sha256") if isinstance(r.get("identity"), dict) else None
        if seal_sha is None or got != seal_sha:
            p.append(f"arm not started under the current reference seal ({got!r} != {seal_sha!r})")
    if r.get("family") != panel.get("family"):
        p.append(f"family {r.get('family')!r} != panel family {panel.get('family')!r}")
    ident = r.get("identity") if isinstance(r.get("identity"), dict) else {}
    for key, want in ((panel.get("runtime") or {}).get("enforced") or {}).items():
        if ident.get(key) != want:
            p.append(f"runtime {key} {ident.get(key)!r} != frozen {want!r}")
    return p


def verified_ceiling(out_root: Path, name: str, arms: list[str], panel: dict, panel_sha: str,
                     seal_sha: str | None = None) -> float | None:
    """The smallest verified incumbent over EVERY complete attempt with a valid identity of every arm of the instance:
    a deterministic ceiling for the bound-claim refutation used in attempt selection (Codex round 2 #1); a solver
    bound above it is refuted whichever attempt is finally scored."""
    vals = []
    for arm in arms:
        for _, d in km.arm_attempts(out_root, name, arm):
            try:
                r = json.loads((d / "result.json").read_text())
                if not isinstance(r, dict) or identity_problems(r, name, arm, panel, panel_sha, seal_sha):
                    continue
                inc = r.get("incumbents")
                if isinstance(inc, dict) and inc.get("verified") is True and isinstance(inc.get("best"), (int, float)) \
                        and not isinstance(inc.get("best"), bool) and math.isfinite(inc["best"]):
                    vals.append(float(inc["best"]))
            except (OSError, ValueError, KeyError, TypeError, AttributeError):   # malformed or unreadable attempt:
                continue                                    # never a ceiling source; select_attempt reports it (round 3)
    return min(vals, default=None)


def select_attempt(out_root: Path, name: str, arm: str, panel: dict, panel_sha: str, iv: dict, tgt: Fraction | None,
                   ceiling: float | None = None, seal_sha: str | None = None) -> dict:
    seen, valid, clean = [], [], []
    for k, d in km.arm_attempts(out_root, name, arm):
        rel = d.relative_to(out_root).as_posix()
        f = d / "result.json"
        if not f.is_file():
            seen.append({"attempt": rel, "status": "incomplete"})
            continue
        try:
            r = json.loads(f.read_text())
            if not isinstance(r, dict):
                raise ValueError("result.json is not an object")
            problems = identity_problems(r, name, arm, panel, panel_sha, seal_sha)
            infra = [] if problems else arm_metrics(r, iv, tgt, None, panel["rules"]["eps_p"], ceiling).get("infrastructure") or []
        except (OSError, ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError) as exc:   # (b) unreadable
            seen.append({"attempt": rel, "status": "unreadable", "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        seen.append({"attempt": rel, "status": "complete", "identity_problems": problems, "infrastructure": list(infra)})
        if not problems:
            valid.append((rel, r))
            if not infra:
                clean.append((rel, r))
    pick = clean[0] if clean else (valid[0] if valid else None)
    return {"attempt": pick[0] if pick else None, "result": pick[1] if pick else None, "attempts": seen,
            "complete": any(a["status"] != "incomplete" for a in seen),
            "identity_problems": [] if pick else [p for a in seen for p in a.get("identity_problems", [])]
                                                  + [a["error"] for a in seen if a["status"] == "unreadable"]}


def scan_unexpected(out_root: Path, names: list[str], arms: list[str]) -> list[str]:
    expected = {f"{n}__{a}" for n in names for a in arms}
    flagged = []
    if not out_root.is_dir():
        return flagged
    for d in sorted(out_root.iterdir()):
        if not d.is_dir():
            continue
        if d.name not in expected:
            flagged.append(d.name)
            continue
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and sub.name != "certs" and attempt_index(sub.name) is None:
                flagged.append(f"{d.name}/{sub.name}")
    return flagged


def target(iv: dict, eps_b: float) -> Fraction:
    """bound_target threshold: LB_cert >= LP_hi - eps_b * max(|LP_hi|, 1) (the conservative end of the interval)."""
    hi = Fraction(iv["hi"])
    return hi - Fraction(str(eps_b)) * max(abs(hi), Fraction(1))


def arm_metrics(r: dict, iv: dict, tgt: Fraction | None, z_best: float | None, eps_p: float,
                ceiling: float | None = None) -> dict:
    """scaleup_km_stage1.arm_metrics (bound metrics from the canonical checker only; any infrastructure marker makes
    them undefined; Stage 1 arms are evaluator code, so invalid designs, unverified incumbents and content-rejected
    certificates are evaluator failures), plus:
      - a primal evaluation that could not finish (incumbents "unavailable") leaves only the PRIMAL metrics undefined
        (primal_undefined); the bound metrics stand (Codex round 1 #3);
      - publication beyond the frozen volume cap is reported (publication_volume_exhausted): those versions were
        received but not stored and are never eligible; the arm is scored on its last stored version (Codex round 1 #7);
      - the evaluated primal trajectory is labelled partial when the family's evaluation cap skipped designs."""
    inc = r.get("incumbents") or {}
    unavailable = inc.get("unavailable")
    rr = r if not unavailable else dict(r, incumbents={"best": None, "verified": None, "invalid": 0, "malformed": 0})
    out = km.arm_metrics(rr, iv, tgt, z_best, eps_p)
    run = r.get("run") or {}
    out["publication_volume_exhausted"] = bool(run.get("over_volume_certificates"))
    out["primal_partial"] = bool(inc.get("skipped_eval"))
    if unavailable:
        out.update(best=None, gap_best=None, near_best=None, t_best=None, primal_undefined=True, primal_unavailable=unavailable)
    if refuted(out, ceiling):   # a refuted bound claim: infrastructure, every bound field undefined (Codex round 2 #1)
        out["infrastructure"] = list(out.get("infrastructure") or []) + [
            f"refuted solver bound claim {out['solver_bound_T']} > verified incumbent ceiling {ceiling}"]
        out.update(t_bound=None, bound_target_T=None, bgap_LP_T=None, bgap_LP_T_exact=None, cert_ok_T=None,
                   final_label="infrastructure")
    return out


def refuted(m: dict, z_best: float | None) -> bool:
    """A solver-reported bound above the best verified incumbent (beyond a relative 1e-6) refutes the arm's bound
    claim (protocol v3.1 refutation rule; for evaluator arms an evaluator bug): infrastructure (rerun)."""
    b = m.get("solver_bound_T")
    return b is not None and z_best is not None and b > z_best + 1e-6 * max(1.0, abs(z_best))


def decision_arms(fam, rules: dict) -> list[str]:
    arms = [fam.ENG, fam.NAIVE, *fam.BOUND_CONTROLS]
    if rules.get("primal_gate"):
        arms += [a for a in (fam.PRIMAL_CONTROL, fam.PRIMAL_NAIVE) if a not in arms]
    return arms


def accept(fam, per: dict, headline: str, rules: dict, T: float, z_best: float | None = None) -> dict:
    """Frozen acceptance for one instance (protocol v4 section 6; the k-median rules of scaleup_km_stage1.accept with
    the family's arms). An arm needed for the decision with an infrastructure marker makes it undefined.
      1. naive separates: naive bgap_LP - engineered bgap_LP >= naive_quality_gap (exact), or t_naive (censored at T)
         >= naive_speed_ratio t_eng AND t_naive - t_eng >= saved_fraction T;
      2. bound controls: final_target: engineered meets the target at T and every control misses it; time_to_target:
         each control misses the target or is >= control_speed_ratio x slower with >= control_saved_seconds saved, and
         the engineered arm meets the final target;
      3. primal (only if rules.primal_gate): engineered near_best, the primal control misses it, naive integer Benders
         worse by ceil(naive_quality_gap z_best) units. Otherwise the primal side is reported, never gated."""
    eng, naive = per.get(fam.ENG), per.get(fam.NAIVE)
    need = decision_arms(fam, rules)
    missing = [a for a in need if a not in per]
    if missing:
        return {"pass": None, "undefined": "missing arm results: rerun", "missing": missing}
    infra = {a: per[a]["infrastructure"] for a in need if per[a].get("infrastructure")}
    if infra:
        return {"pass": None, "undefined": "infrastructure: rerun the listed arms", "infrastructure": infra}
    why: dict[str, Any] = {}
    gap_rule = Fraction(str(rules["naive_quality_gap"]))
    ne, nn = _bgap(eng), _bgap(naive)
    q_branch = ne is not None and nn is not None and nn - ne >= gap_rule
    t_eng = eng["t_bound"]
    t_naive = naive["t_bound"] if naive["t_bound"] is not None else T
    s_branch = (t_eng is not None and t_naive >= rules["naive_speed_ratio"] * t_eng
                and t_naive - t_eng >= rules["saved_fraction"] * T)
    why["naive_separates"] = bool(q_branch or s_branch)
    why["naive_separation_branch"] = "quality" if q_branch else ("speed (censored at T)" if s_branch and naive["t_bound"] is None
                                                                 else "speed" if s_branch else None)
    ctrl = {}
    for a in fam.BOUND_CONTROLS:
        c = per[a]
        if headline == "time_to_target":
            ok = t_eng is not None and (c["t_bound"] is None or (c["t_bound"] >= rules["control_speed_ratio"] * t_eng
                                                                 and c["t_bound"] - t_eng >= rules["control_saved_seconds"]))
        elif headline == "final_target":
            ok = eng["bound_target_T"] is True and c["bound_target_T"] is False
        else:
            raise ValueError(f"unknown headline {headline!r}")
        ctrl[a] = "separated" if ok else "NOT separated"
    why["bound_controls"] = ctrl
    why["engineered_bound_target"] = eng["bound_target_T"] is True
    pc, pn = per.get(fam.PRIMAL_CONTROL) or {}, per.get(fam.PRIMAL_NAIVE) or {}
    primal = {"engineered_near_best": eng.get("near_best"), "primal_control_near_best": pc.get("near_best"),
              "naive_int_worse": km.primal_naive_worse(eng, pn, rules["naive_quality_gap"], z_best) if pn else None,
              "gated": bool(rules.get("primal_gate"))}
    why["primal"] = primal
    ok = why["naive_separates"] and why["engineered_bound_target"] and all(v == "separated" for v in ctrl.values())
    if rules.get("primal_gate"):
        ok = ok and primal["engineered_near_best"] is True and primal["primal_control_near_best"] is False \
            and primal["naive_int_worse"] is True
    why["pass"] = bool(ok)
    why["diagnostics"] = {a: {"t_bound": per[a].get("t_bound"), "bgap_LP_T": per[a].get("bgap_LP_T"),
                              "final_label": per[a].get("final_label"), "solver_bound_T": per[a].get("solver_bound_T"),
                              "infrastructure": per[a].get("infrastructure") or None}
                          for a in [*fam.DIAGNOSTICS, fam.NAIVE, fam.ENG] if a in per}
    return why


def interval_check(iv: dict | None, max_width: float) -> tuple[str, Fraction | None]:
    """'ready' only for an interval with exact endpoints lo <= hi whose relative width (hi - lo)/max(|hi|, 1),
    recomputed exactly from the endpoints (never read from the record), is at most max_width (Codex round 1 #2);
    otherwise 'pending_reference' (absent), 'undefined_reference' (malformed or lo > hi) or 'reference_too_wide'.
    Never a failure: the instance stays pending (reference rerun)."""
    if not iv or iv.get("lo") is None or iv.get("hi") is None:
        return "pending_reference", None
    try:
        lo, hi = Fraction(str(iv["lo"])), Fraction(str(iv["hi"]))
    except (ValueError, ZeroDivisionError, TypeError):
        return "undefined_reference", None
    if lo > hi:
        return "undefined_reference", None
    width = (hi - lo) / max(abs(hi), Fraction(1))
    if width > Fraction(str(max_width)):
        return "reference_too_wide", width
    return "ready", width


def instance_decision(fam, name: str, spec: dict, panel: dict, out_root: Path, iv: dict | None, panel_sha: str,
                      ref_problems: list[str] | None = None) -> dict:
    rules = panel["rules"]
    state, width = interval_check(iv, rules["max_reference_rel_width"])
    if ref_problems:
        state = "reference_identity_mismatch"
    base = {"headline": spec["headline"], "lp_interval": iv, "reference_state": state,
            "reference_rel_width_exact": None if width is None else float(width)}
    unexpected = [u for u in scan_unexpected(out_root, [name], fam.ALL_ARMS) if u.startswith(f"{name}__")]
    if unexpected:
        base["unexpected"] = unexpected
    if state != "ready":
        return base | {"decision": "pending", "acceptance": {"pass": None, "undefined": state, **({"problems": ref_problems} if ref_problems else {})}}
    tgt = target(iv, rules["eps_b"])
    seal_sha = seal_sha256(out_root, fam.name, panel, name)
    ceiling = verified_ceiling(out_root, name, fam.ALL_ARMS, panel, panel_sha, seal_sha)
    sel = {a: select_attempt(out_root, name, a, panel, panel_sha, iv, tgt, ceiling, seal_sha) for a in fam.ALL_ARMS}
    base["attempts"] = {a: s["attempt"] for a, s in sel.items()}
    retried = {a: s["attempts"] for a, s in sel.items() if len(s["attempts"]) > 1}
    if retried:
        base["attempt_history"] = retried
    arms_res = {a: s["result"] for a, s in sel.items() if s["result"] is not None}
    missing = [a for a in fam.ALL_ARMS if not sel[a]["complete"]]
    ident = {a: sel[a]["identity_problems"] for a in fam.ALL_ARMS if sel[a]["complete"] and sel[a]["result"] is None}
    if missing or ident:   # every panel arm (diagnostics too) is resolved before any decision is read
        why = "; ".join(x for x in ("arms pending" if missing else "",
                                    "invalid results (identity mismatch or unreadable; infrastructure): rerun" if ident else "") if x)
        return base | {"decision": "pending", "acceptance": {"pass": None, "undefined": why, "missing": missing,
                                                             **({"identity": ident} if ident else {})}}
    pre = {a: arm_metrics(r, iv, tgt, None, rules["eps_p"], ceiling) for a, r in arms_res.items()}
    unresolved = [a for a in fam.ALL_ARMS if pre[a].get("primal_undefined")]
    if unresolved and rules.get("primal_gate"):
        return base | {"decision": "pending", "acceptance": {"pass": None, "undefined": "z_best not final: rerun", "arms": unresolved}}
    verified = [r["incumbents"]["best"] for a, r in arms_res.items() if a not in unresolved
                and r["incumbents"].get("best") is not None and r["incumbents"].get("verified") is True]
    z_best = min(verified, default=None)
    per = {a: arm_metrics(r, iv, tgt, z_best, rules["eps_p"], ceiling) for a, r in arms_res.items()}
    acc = accept(fam, per, spec["headline"], rules, panel["T"], z_best=z_best)
    decision = {True: "pass", False: "fail", None: "pending"}[acc.get("pass")]
    return base | {"target": str(tgt), "z_best": z_best, "z_best_final": not unresolved, "refutation_ceiling": ceiling,
                   "arms": per, "acceptance": acc, "decision": decision}


def reference_problems(row: dict | None, name: str, panel: dict, out_root: Path | None = None,
                       seal: dict | None = None) -> list[str]:
    """Identity of a reference interval (Codex round 1 #2). Always: the panel freezes the instance hash and the record
    carries the same hash. For a reference read from the results directory (holdout, fallback tier) also: the
    record's family and instance name, its witness files (certificate, LP point, SCFL flow witness) present with their
    recorded sha256, and the record file's sha256 equal to the one in the reference seal written before any of
    that part's arm ran (scaleup_bdlp_panel.py --seal)."""
    want = ((panel.get("references") or {}).get(name) or {}).get("instance_sha256")
    if not want:
        return ["the panel freezes no instance_sha256 for this instance"]
    if row is None:
        return []
    p = []
    if row.get("instance_sha256") != want:
        p.append(f"reference instance sha256 {row.get('instance_sha256')!r} != frozen {want!r}")
    if out_root is None:
        return p
    if row.get("family") != panel.get("family") or row.get("instance") != name:
        p.append("reference record family/instance mismatch")
    for fkey, hkey in (("certificate_file", "certificate_sha256"), ("lp_point_file", "lp_point_sha256")):
        f = row.get(fkey)
        if not f or not (out_root / f).is_file() or sha256((out_root / f).read_bytes()) != row.get(hkey):
            p.append(f"reference witness {fkey} missing or hash mismatch")
    w = (row.get("lp_hi") or {}).get("witness_file")
    if w is not None and (not (out_root / w).is_file() or sha256((out_root / w).read_bytes()) != row["lp_hi"].get("witness_sha256")):
        p.append("reference LP_hi witness missing or hash mismatch")
    ref_file = out_root / f"{name}__lp_reference.json"
    sealed = ((seal or {}).get("references") or {}).get(name)
    if seal is None:
        p.append("no reference seal for this part (scaleup_bdlp_panel.py --seal before its arm runs)")
    elif seal.get("panel_sha256") != panel.get("_sha256"):
        p.append("reference seal made for another panel")
    elif not ref_file.is_file() or sealed != sha256(ref_file.read_bytes()):
        p.append("reference record differs from the sealed one")
    return p


def seal_path(out_root: Path, fam_name: str, tier: str, part: str) -> Path:
    return out_root / f"{fam_name}_{tier}_{part}_reference_seal.json"


def load_seal(out_root: Path, fam_name: str, tier: str, part: str) -> dict | None:
    f = seal_path(out_root, fam_name, tier, part)
    try:
        return json.loads(f.read_text()) if f.is_file() else None
    except (OSError, ValueError):   # unreadable seal: treated as absent (pending)
        return None


def summarize(fam_name: str, out_root: Path, panel_path: Path) -> dict:
    fam = family(fam_name)
    raw = panel_path.read_bytes()
    panel = json.loads(raw)
    panel_sha = sha256(raw)
    panel["_sha256"] = panel_sha      # in memory only (reference seals name the panel they were made for)
    if panel.get("family") != fam_name:
        raise SystemExit(f"panel family {panel.get('family')!r} != {fam_name!r}")
    if panel.get("arms", {}).get("all") != fam.ALL_ARMS:
        raise SystemExit("panel arm list differs from the runner's family arms")
    tiers = panel["tiers"]
    names = [n for spec in tiers.values() for n in [*spec["stage1"], *spec.get("holdout", [])]]
    summary: dict = {"family": fam_name, "panel": panel_path.name, "panel_sha256": panel_sha, "instances": {}, "tiers": {},
                     "unexpected": scan_unexpected(out_root, names, fam.ALL_ARMS)}
    frozen = panel.get("lp_star_frozen", {}).get("intervals", {})
    for tier in panel["tier_order"]:
        spec = tiers[tier]
        gate_tier = spec.get("run_if_rejected")
        if gate_tier is not None and summary["tiers"].get(gate_tier, {}).get("tier") != "rejected":
            for name in [*spec["stage1"], *spec.get("holdout", [])]:
                summary["instances"][name] = {"tier": tier, "decision": "not_run",
                                              "acceptance": {"pass": None, "undefined": f"fallback: {gate_tier} not rejected"}}
            summary["tiers"][tier] = {"stage1": "not_run", "holdout": "not_run", "tier": "not_needed"}
            continue
        s1 = []
        for name in spec["stage1"]:
            if gate_tier is None:          # primary tier: the interval frozen in the panel
                iv = frozen.get(name)
                ref_p = reference_problems(iv, name, panel)
            else:                          # fallback tier: its own sealed reference job, run before any of its arms
                ref = out_root / f"{name}__lp_reference.json"
                row = json.loads(ref.read_text()) if ref.exists() else None
                iv = row["interval"] if row else None
                ref_p = reference_problems(row, name, panel, out_root, load_seal(out_root, fam_name, tier, "stage1"))
            d = instance_decision(fam, name, spec, panel, out_root, iv, panel_sha, ref_p) | {"tier": tier, "part": "stage1"}
            summary["instances"][name] = d
            s1.append(d["decision"])
        gate = tier_decision(s1, None)
        hold = None
        if gate["stage1"] == "pass":
            hold = []
            for name in spec.get("holdout", []):
                ref = out_root / f"{name}__lp_reference.json"
                row = json.loads(ref.read_text()) if ref.exists() else None
                iv = row["interval"] if row else None
                d = instance_decision(fam, name, spec, panel, out_root, iv, panel_sha,
                                      reference_problems(row, name, panel, out_root, load_seal(out_root, fam_name, tier, "holdout"))) | \
                    {"tier": tier, "part": "holdout"}
                summary["instances"][name] = d
                hold.append(d["decision"])
        else:
            for name in spec.get("holdout", []):
                summary["instances"][name] = {"tier": tier, "part": "holdout", "decision": "not_run",
                                              "acceptance": {"pass": None, "undefined": f"Stage 1 {gate['stage1']}"}}
        summary["tiers"][tier] = tier_decision(s1, hold) | {"stage1_decisions": dict(zip(spec["stage1"], s1)),
                                                            "holdout_decisions": dict(zip(spec.get("holdout", []), hold)) if hold is not None else None}
    retained = [t for t in panel["tier_order"] if summary["tiers"][t]["tier"] == "retained"]
    summary["family_decision"] = retained[0] if retained else (
        "rejected" if all(summary["tiers"][t]["tier"] in ("rejected", "not_needed") for t in panel["tier_order"]) else "pending")
    write_json_atomic(out_root / f"{fam_name}_stage1_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs=3, metavar=("FAMILY", "INSTANCE", "ARM"))
    ap.add_argument("--child", nargs=3, metavar=("FAMILY", "INSTANCE", "ARM"), help=argparse.SUPPRESS)
    ap.add_argument("--start", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--reference", nargs=2, metavar=("FAMILY", "INSTANCE"))
    ap.add_argument("--gen", nargs=2, metavar=("FAMILY", "OUTDIR"))
    ap.add_argument("--raw", default=None, help="OR-Library CAP directory (scfl --gen)")
    ap.add_argument("--summarize", metavar="FAMILY")
    ap.add_argument("--panel", default=None)
    ap.add_argument("--T", type=float, default=900.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.child:
        child(a.child[0], a.child[1], a.child[2], a.T, a.start if a.start is not None else time.time())
        return
    if a.gen:
        gen(a.gen[0], Path(a.gen[1]), Path(a.raw) if a.raw else None, a.panel)
        return
    out = Path(a.out)
    if a.task:
        r = run_task(a.task[0], a.task[1], a.task[2], a.T, out, a.panel)
        print("TASK " + json.dumps({"instance": r["instance"], "arm": r["arm"], "run": r["run"], "best": r["incumbents"]["best"],
                                    "checkpoints": [(c.get("checkpoint"), c.get("cert_ok"), c.get("lb_float"), c.get("reason"))
                                                    for c in r["certificates"].get("checkpoints", [])]}, default=str), flush=True)
    if a.reference:
        r = reference(a.reference[0], a.reference[1], a.T, out, a.panel)
        print("REFERENCE " + json.dumps({k: r.get(k) for k in ("instance", "termination", "interval", "run_seconds")}, default=str),
              flush=True)
    if a.summarize:
        fam_name = a.summarize
        s = summarize(fam_name, out, Path(a.panel) if a.panel else default_panel(fam_name))
        if s["unexpected"]:
            print("UNEXPECTED (flagged, never read)", json.dumps(s["unexpected"]), flush=True)
        for name, d in s["instances"].items():
            print("SUMMARY", name, d.get("decision"), json.dumps(d.get("acceptance"), default=str), flush=True)
        for tier, d in s["tiers"].items():
            print("TIER", tier, json.dumps({k: d[k] for k in ("stage1", "holdout", "tier")}), flush=True)
        print("FAMILY", fam_name, s["family_decision"], flush=True)


if __name__ == "__main__":
    main()
