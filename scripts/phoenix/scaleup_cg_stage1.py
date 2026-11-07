"""Protocol v4 Stage 1 / holdout / development runner for the column-generation scale-up families CLRP, CVRP and BPPC
(primal-scored, T_f = 900 s). Phoenix only for every mode that runs an arm.

Arms (one per Slurm task; 1 thread, 16 GB job cgroup, Gurobi SoftMemLimit 12 GB where Gurobi is used):
    eng     engineered branch-and-price(-and-cut) reference (stabilization, heuristic pricing, primal heuristics)
    naive   naive column generation: plain restricted master, exact pricing, one column per round, no stabilization or
            heuristic pricing, integer restricted master ("price-and-branch" rounding) in the last 20% of the budget
    ls      the family's decomposition-free control (CVRP: SISR-style ruin-and-recreate + local search; CLRP:
            location-allocation + routing ILS; BPPC: conflict-aware FFD + LNS + Capua-style tabu repair)
    mip     the compact MIP: reported baseline, never a gate (protocol v4 Section 6.3)

Modes
  --task FAMILY INSTANCE ARM   one supervised arm. The supervisor (src/experiments/xl_supervisor.Supervisor, unix
                          socket, acknowledged publications) starts its clock right before spawning the arm in a new
                          session; the arm publishes every incumbent through the channel (publish_incumbent) and a
                          rate-limited, capped series of bound claims (publish_claim). The whole process group is
                          SIGKILLed at T whatever it is doing. Then, outside T: every eligible incumbent (received
                          strictly before T) is re-evaluated by the family evaluator through
                          xl_scoring.attempt_from_supervisor, and the best one is checked by the family's stdlib
                          verifier. Each run writes a FRESH directory claimed atomically (OUT/<instance>__<arm>, retries
                          OUT/<instance>__<arm>/attempt_<k>), result.json is written atomically.
  --summarize             per instance of a frozen panel: every panel arm resolved with a valid identity, z_best =
                          min(z_pre, best verified incumbent of any panel arm), xl_scoring.score_attempt per arm
                          (near_best at the tier's frozen eps_p, gap_best, primal integral, matches_or_beats_MIP), and
                          the frozen acceptance rule; tier decision Stage 1 >= 3/4, then holdout 2/2 (scored only after
                          Stage 1 passed).
  --freeze-eps            development calibration (protocol v4 Section 4): from the dev results (eng and ls on the
                          tier's development instances) the near_best eps_p interval that separates the engineered
                          reference from every control on the most instances; a proposal (eps_p and its interval) is
                          written, never applied to the panel automatically. A tier without an interval separating >= 3/4
                          of its dev instances fails before Stage 1.

Acceptance per instance (frozen in the panel's rules; integer objectives everywhere):
    (1) engineered near_best: gap_best(eng) <= eps_p (no refuted claim);
    (2) naive separates: z_naive - z_eng >= ceil(0.005 z_best) (no naive incumbent counts as infinitely worse), or the
        naive first reaches near_best (physical verified attainment, ungated) >= 3x later than eng (gated) with >= 0.25 T
        saved (never reaching: censored at T);
    (3) every control misses near_best (an incumbent-less control misses it);
    the MIP is reported (z_MIP, matches_or_beats_MIP), never a gate.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FAMILIES = ("cvrp", "clrp", "bppc")
PANEL_ARMS = ["eng", "naive", "ls", "mip"]
DEV_ARMS = ["eng", "ls"]
CONTROLS = ["ls"]
SOFT_MEM_GB = 12.0            # Gurobi SoftMemLimit (as in the v3.1 calibration runner), below the 16 GB job cgroup
CHILD_MARGIN = 2.0            # the arm's own time limit ends this many seconds before the supervisor's kill at T
MAX_CLAIMS = 100              # bound claims published per arm run (improving, >= CLAIM_EVERY s apart)
CLAIM_EVERY = 5.0
STAGE1_REQUIRED = 3
RUNNER_FILES = ["scripts/phoenix/scaleup_cg_stage1.py", "scripts/phoenix/scaleup_cg_stage1.sbatch",
                "scripts/phoenix/scaleup_cg_panel.py", "src/experiments/xl_supervisor.py", "src/experiments/xl_scoring.py"]
FAMILY_FILES = {
    "cvrp": ["src/cg/cvrp/a_data.py", "src/cg/cvrp/heuristic.py", "src/cg/cvrp/b_mip_xl.py", "src/cg/cvrp/c_naive_cg.py",
             "src/cg/cvrp/d_ref_cg_xl.py", "src/cg/cvrp/e_verifier.py", "src/cg/cvrp/f_ils_xl.py",
             "src/cg/clrp/a_data.py", "src/cg/clrp/b_mip_xl.py", "src/cg/clrp/c_heuristic.py",
             "src/cg/clrp/d_ref_cg_xl.py", "src/cg/clrp/f_sisr_xl.py"],
    "clrp": ["src/cg/clrp/a_data.py", "src/cg/clrp/b_mip_xl.py", "src/cg/clrp/c_heuristic.py", "src/cg/clrp/d_ref_cg_xl.py",
             "src/cg/clrp/e_verifier.py", "src/cg/clrp/e_naive_cg.py", "src/cg/clrp/f_sisr_xl.py",
             "src/cg/clrp/g_lrp_ils_xl.py", "src/cg/cvrp/f_ils_xl.py", "src/cg/cvrp/a_data.py", "src/cg/cvrp/b_mip_xl.py",
             "src/cg/cvrp/heuristic.py"],
    "bppc": ["src/cg/bppc/a_data.py", "src/cg/bppc/heuristic.py", "src/cg/bppc/pricing.py", "src/cg/bppc/b_mip_xl.py",
             "src/cg/bppc/c_naive_cg.py", "src/cg/bppc/d_ref_cg_xl.py", "src/cg/bppc/e_verifier.py",
             "src/cg/bppc/f_heuristic.py"],
}


def code_files(family: str) -> list[str]:
    return sorted(set(RUNNER_FILES) | set(FAMILY_FILES[family]))


# ---------------------------------------------------------------- arms

def arms(family: str) -> dict:
    common = {"soft_mem_gb": SOFT_MEM_GB}
    if family == "cvrp":
        from src.cg.cvrp.b_mip_xl import CVRPCompactMIP
        from src.cg.cvrp.c_naive_cg import CVRPNaiveCG
        from src.cg.cvrp.d_ref_cg_xl import CVRPColumnGenerationXL
        from src.cg.cvrp.f_ils_xl import CVRPRuinRecreate
        table = {"eng": (CVRPColumnGenerationXL, common), "naive": (CVRPNaiveCG, common),
                 "ls": (CVRPRuinRecreate, {"seed": 0}), "mip": (CVRPCompactMIP, common)}
    elif family == "clrp":
        from src.cg.clrp.b_mip_xl import CLRPCompactMIP
        from src.cg.clrp.d_ref_cg_xl import CLRPColumnGenerationXL
        from src.cg.clrp.e_naive_cg import CLRPNaiveCG
        from src.cg.clrp.g_lrp_ils_xl import CLRPLocationRoutingILS
        table = {"eng": (CLRPColumnGenerationXL, common), "naive": (CLRPNaiveCG, common),
                 "ls": (CLRPLocationRoutingILS, {"seed": 0}), "mip": (CLRPCompactMIP, common)}
    elif family == "bppc":
        from src.cg.bppc.b_mip_xl import BPPCCompactMIP
        from src.cg.bppc.c_naive_cg import BPPCNaiveCG
        from src.cg.bppc.d_ref_cg_xl import BPPCBranchAndPrice
        from src.cg.bppc.f_heuristic import BPPCHeuristicControl
        table = {"eng": (BPPCBranchAndPrice, common), "naive": (BPPCNaiveCG, common),
                 "ls": (BPPCHeuristicControl, {"seed": 0}), "mip": (BPPCCompactMIP, common)}
    else:
        raise ValueError(f"unknown family {family!r}")
    table.update({f"_test_{m}": (_TestArm, {"mode": m, "family": family}) for m in _TestArm.MODES})
    return table


def trivial_solution(family: str, instance: str) -> Any:
    """A feasible plan without any search (test arms only): one route per customer / one bin per item."""
    if family == "cvrp":
        from src.cg.cvrp.a_data import load_vrp
        return [[i] for i in range(1, load_vrp(instance).n + 1)]
    if family == "bppc":
        from src.cg.bppc.a_data import load_bppc
        return [[i] for i in range(1, load_bppc(instance).n + 1)]
    raise ValueError("test arms are defined for cvrp and bppc only")


class _TestArm:
    """Failure-injection arms (never in a panel): each first publishes the trivial plan."""

    MODES = ("ok", "sleep", "crash", "no_result", "sigkill_self", "invalid", "bad_bound")

    def __init__(self, *, time_limit: float, on_incumbent, on_bound, mode: str = "ok", family: str = "cvrp", **kw) -> None:
        self.on_incumbent, self.on_bound, self.mode, self.family = on_incumbent, on_bound, mode, family

    def run(self, data_path: str) -> dict:
        sol = trivial_solution(self.family, data_path)
        self.on_incumbent(sol, stage="test")
        if self.mode == "invalid":
            self.on_incumbent(sol[1:], stage="invalid")      # misses one customer / item
        if self.mode == "bad_bound":
            self.on_bound(float(len(sol) * 10**9))          # a bound above a verified solution: refuted claim
        if self.mode == "crash":
            raise RuntimeError("injected arm crash")
        if self.mode == "no_result":
            os._exit(0)
        if self.mode == "sigkill_self":
            os.kill(os.getpid(), signal.SIGKILL)
        if self.mode == "sleep":
            while True:
                time.sleep(3600)
        return {"termination_reason": "finished", "best_objective": None}


def arm_kwargs(cls, kwargs: dict) -> dict:
    """Keywords accepted by the arm's constructor (soft_mem_gb, seed, on_bound are dropped where not accepted)."""
    params = inspect.signature(cls).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


# ---------------------------------------------------------------- family adapters (evaluator side, outside T)

class Family:
    """Loading, payload evaluation (exact objective or InvalidSolution) and the stdlib verifier of one family."""

    def __init__(self, family: str, instance: str) -> None:
        self.family, self.instance = family, instance
        if family == "cvrp":
            from src.cg.cvrp.a_data import load_vrp
            self.inst = load_vrp(instance)
            self.dist = self.inst.distances()
        elif family == "clrp":
            from src.cg.clrp.a_data import load_prodhon
            self.inst = load_prodhon(instance)
            self.cost = self.inst.travel("ceil")
        elif family == "bppc":
            from src.cg.bppc.a_data import load_bppc
            self.inst = load_bppc(instance)
        else:
            raise ValueError(family)

    @staticmethod
    def solution_of(payload: bytes) -> Any:
        obj = json.loads(payload)
        if not isinstance(obj, dict) or "solution" not in obj:
            raise ValueError("payload must be an object with a 'solution'")
        return obj["solution"]

    def value(self, solution: Any) -> int | None:
        """Exact integer objective of a feasible plan, None for an infeasible or malformed one."""
        try:
            if self.family == "cvrp":
                from src.cg.cvrp.b_mip_xl import route_cost
                if not isinstance(solution, list):
                    return None
                v = route_cost(self.inst, solution, self.dist)
            elif self.family == "clrp":
                from src.cg.clrp.b_mip_xl import route_cost
                if not isinstance(solution, list) or not all(
                        isinstance(r, dict) and isinstance(r.get("depot"), int) and not isinstance(r.get("depot"), bool)
                        and 0 <= r["depot"] < self.inst.m and isinstance(r.get("customers"), list) and r["customers"]
                        and all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c < self.inst.n for c in r["customers"])
                        for r in solution):
                    return None
                v = route_cost(self.inst, self.cost, solution)
            else:
                from src.cg.bppc.b_mip_xl import bin_count
                v = bin_count(self.inst, solution)
        except (TypeError, ValueError, KeyError, IndexError):   # (a) malformed solution object: infeasible publication
            return None
        if v is None:
            return None
        if float(v) != round(float(v)):
            raise ValueError(f"non-integral objective {v}")   # (b) evaluator inconsistency: infrastructure
        return int(round(float(v)))

    def objective(self, payload: bytes) -> int:
        from src.experiments.xl_scoring import InvalidSolution
        try:
            sol = self.solution_of(payload)
        except ValueError as exc:   # (a) not the payload schema: an infeasible publication of the arm
            raise InvalidSolution(str(exc)) from exc
        v = self.value(sol)
        if v is None:
            raise InvalidSolution("infeasible or malformed plan")
        return v

    def verify(self, solution: Any, method: str) -> dict:
        """The family's independent stdlib verifier on the output contract of `solution`."""
        if self.family == "cvrp":
            from src.cg.cvrp.b_mip_xl import cvrp_output as out_fn
            from src.cg.cvrp.e_verifier import verify_solution
            out = out_fn(self.inst, solution, method=method)
        elif self.family == "clrp":
            from src.cg.clrp.b_mip_xl import route_cost
            from src.cg.clrp.e_verifier import verify_solution
            out = {"status": "feasible", "method": method, "objective_value": route_cost(self.inst, self.cost, solution),
                   "routes": solution}
        else:
            from src.cg.bppc.b_mip_xl import bppc_output as out_fn
            from src.cg.bppc.e_verifier import verify_solution
            out = out_fn(self.inst, solution, method=method)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "output.json"
            p.write_text(json.dumps(out))
            res = verify_solution(self.instance, p)
        return {"ok": bool(res.get("ok")), "issues": (res.get("issues") or [])[:10], "objective_value": res.get("objective_value")}


# ---------------------------------------------------------------- small utilities (as in scaleup_km_stage1.py)

def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stem(path: str) -> str:
    return Path(path).stem


def write_json_atomic(path: Path, obj: Any) -> None:
    """Replace `path` atomically (unique temp file in the same directory, fsync, rename, directory fsync)."""
    data = json.dumps(obj, default=str, indent=1).encode()
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            os.fchmod(fh.fileno(), 0o644)
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:   # benign: already renamed or never created
            pass
        raise
    try:
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:   # benign: the directory fsync is a durability hint
        pass


ATTEMPT_PREFIX = "attempt_"


def attempt_index(name: str) -> int | None:
    k = name[len(ATTEMPT_PREFIX):] if name.startswith(ATTEMPT_PREFIX) else ""
    if not (k.isascii() and k.isdigit()) or k[0] == "0" or int(k) < 2:
        return None
    return int(k)


def claim_attempt(base: Path) -> tuple[int, Path]:
    """A fresh, empty directory claimed atomically with mkdir: attempt 1 is `base`, a retry base/attempt_<k>."""
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        base.mkdir()
        return 1, base
    except FileExistsError:   # benign: an earlier attempt owns base
        pass
    k = 2
    while True:
        d = base / f"{ATTEMPT_PREFIX}{k}"
        try:
            d.mkdir()
            return k, d
        except FileExistsError:   # benign: taken by an earlier attempt
            k += 1


def code_commit() -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):   # recorded as unknown
        return None
    return (r.stdout.strip() if r.returncode == 0 else "") or None


def job_memory_cgroup() -> Path | None:
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                return Path("/sys/fs/cgroup" + line[3:])
    except OSError:   # no cgroup v2 information: no OOM evidence
        return None
    return None


def oom_kill_count() -> int | None:
    cg = job_memory_cgroup()
    if cg is None:
        return None
    try:
        for line in (cg / "memory.events").read_text().splitlines():
            key, _, value = line.partition(" ")
            if key == "oom_kill":
                return int(value)
    except (OSError, ValueError):   # unreadable: no evidence
        return None
    return None


# ---------------------------------------------------------------- child (the arm process)

def child(family: str, instance: str, arm: str, T: float, start_wall: float, outdir: str) -> None:
    try:   # the arm, not the supervisor, is the kernel's OOM victim inside the job's memory cgroup
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:   # not Linux or not permitted: OOM evidence then decides nothing (infrastructure)
        pass
    sys.stdout = sys.stderr
    from src.experiments.xl_supervisor import publish_claim, publish_incumbent
    claims = {"n": 0, "last_t": -math.inf, "best": -math.inf, "all": []}
    published = {"incumbents": 0, "not_acknowledged": 0}

    def on_inc(solution, stage=None, **kw):
        ok = publish_incumbent({"solution": solution, "stage": stage}, wait=True)
        published["incumbents"] += 1
        published["not_acknowledged"] += 0 if ok else 1

    def on_bound(b):
        b = float(b)
        claims["all"].append((round(time.time() - start_wall, 3), b))
        now = time.monotonic()
        if (math.isfinite(b) and b > claims["best"] and claims["n"] < MAX_CLAIMS and now - claims["last_t"] >= CLAIM_EVERY):
            publish_claim({"lower_bound": b}, wait=True)
            claims.update(n=claims["n"] + 1, last_t=now, best=b)

    cls, kw = arms(family)[arm]
    left = T - (time.time() - start_wall) - CHILD_MARGIN
    kwargs = arm_kwargs(cls, dict(time_limit=max(1.0, left), threads=1, on_incumbent=on_inc, on_bound=on_bound, **kw))
    res = cls(**kwargs).run(instance)
    slim = {k: v for k, v in res.items() if not isinstance(v, (list, dict)) or k in ("open_depots",)}
    slim["solver_bounds"] = {"count": len(claims["all"]), "final": claims["all"][-1][1] if claims["all"] else None,
                             "claims_published": claims["n"], "label": "solver-reported bounds (diagnostic)"}
    slim["publications"] = published
    write_json_atomic(Path(outdir) / "arm_result.json", slim)


# ---------------------------------------------------------------- supervisor side

def classify_run(run: dict) -> list[str]:
    """Infrastructure problems of one run (arms are evaluator code): a kill at T is expected; otherwise the arm must exit
    0 after writing arm_result.json; a kernel-confirmed OOM kill is an arm resource outcome (its eligible incumbents
    count); every other exit and every supervisor error is infrastructure (rerun)."""
    problems = [f"supervisor: {e}" for e in run.get("supervisor_errors") or []]
    if run.get("killed_at_T"):
        if run.get("returncode") not in (-signal.SIGKILL, 0):
            problems.append(f"arm exited with {run.get('returncode')} while being killed at T")
    elif run.get("oom_confirmed"):
        pass
    else:
        rc = run.get("returncode")
        if rc != 0:
            problems.append(f"arm exited unexpectedly: {'signal ' + str(-rc) if isinstance(rc, int) and rc < 0 else 'code ' + str(rc)}")
        elif not run.get("arm_result"):
            problems.append("arm finished without its arm_result.json")
    if run.get("torn") and not run.get("killed_at_T") and not run.get("oom_confirmed"):
        problems.append(f"{run['torn']} torn publication(s) without a deadline kill")
    return problems


def supervise(family: str, instance: str, arm: str, T: float, outdir: Path) -> tuple[dict, dict]:
    from src.experiments.xl_supervisor import Supervisor
    if any(outdir.iterdir()):
        raise FileExistsError(f"refusing to reuse the non-empty arm directory {outdir}")
    sup = Supervisor(outdir / "publications", T, transport="unix", checkpoints=[T]).start()
    env = dict(os.environ, **sup.env(), PYTHONUNBUFFERED="1")
    oom_before = oom_kill_count()
    log = open(outdir / "arm.log", "xb")
    sup.start_clock()
    t0 = sup.t0
    start_wall = time.time()
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child", family, instance, arm, "--T", str(T),
                             "--start", repr(start_wall), "--outdir", str(outdir)],
                            stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=str(ROOT))
    killed = False
    try:
        proc.wait(timeout=max(0.0, T - (time.monotonic() - t0)))
    except subprocess.TimeoutExpired:   # the normal end of a long arm: hard kill of the whole group at T
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:   # benign: the group already exited
            pass
        killed = True
        proc.wait()
    wall = time.monotonic() - t0
    if not killed:   # the arm exited by itself: make sure no orphan of its group survives
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):   # benign: nothing left
            pass
    log.close()
    record = sup.stop()
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    oom_after = oom_kill_count()
    oom = (not killed and proc.returncode == -signal.SIGKILL and oom_before is not None and oom_after is not None
           and oom_after > oom_before)
    counts = record.get("counts", {})
    run = {"returncode": proc.returncode, "killed_at_T": killed, "wall_seconds": round(wall, 3),
           "child_peak_rss_mb": round(peak, 1), "oom_kill_before": oom_before, "oom_kill_after": oom_after,
           "oom_confirmed": bool(oom), "arm_result": (outdir / "arm_result.json").is_file(),
           "torn": counts.get("torn", 0), "late": counts.get("late", 0), "supervisor_counts": counts,
           "supervisor_errors": list(record.get("errors") or []),
           "resource_outcome": "memory_limit (kernel-confirmed OOM kill of the arm)" if oom else None}
    run["infrastructure"] = classify_run(run)
    return record, run


def evaluate(fam: Family, record: dict, run: dict, arm: str) -> dict:
    """Eligible incumbents re-evaluated (xl_scoring.attempt_from_supervisor), the improving trajectory, the claims, and
    the stdlib verification of the best plan."""
    from src.experiments.xl_scoring import attempt_from_supervisor
    from src.experiments.xl_supervisor import read_payload
    att = attempt_from_supervisor(record, fam.objective, method_ok=True,
                                  termination="killed_at_T" if run.get("killed_at_T") else None)
    traj, best = [], None
    for t, z in sorted(att.incumbents, key=lambda x: x[0]):
        if best is None or z < best:
            best = z
            traj.append([t, z])
    verified, verification = None, None
    if best is not None:
        entry = None
        for e in record.get("incumbents", []):   # the earliest stored payload of the best value
            try:
                payload = read_payload(record["store_dir"], e)
                if fam.objective(payload) == best:
                    entry = payload
                    break
            except Exception:  # noqa: BLE001 - payloads already classified by attempt_from_supervisor
                continue
        if entry is None:
            verification = {"error": "best payload not found"}
        else:
            try:
                verification = fam.verify(Family.solution_of(entry), arm)
                verified = bool(verification["ok"] and verification["objective_value"] is not None
                                and abs(verification["objective_value"] - best) <= 1e-6 * max(1.0, abs(best)))
            except Exception as exc:  # noqa: BLE001 - (b) the verifier could not run: infrastructure
                verification = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    claims = [{k: c.get(k) for k in ("lower_bound", "t", "_error")} for c in att.claims]
    return {"trajectory": traj, "best": best, "t_first": traj[0][0] if traj else None, "t_best": traj[-1][0] if traj else None,
            "verified": verified, "verification": verification, "infeasible": att.extra.get("infeasible_incumbents"),
            "late": att.extra.get("late"), "torn": att.extra.get("torn"), "claims": claims,
            "infrastructure": list(att.infrastructure)}


def identity(task: dict, panel_path: Path | None) -> dict:
    import platform
    panel = json.loads(panel_path.read_text()) if panel_path is not None and panel_path.exists() else {}
    files = sorted(set(panel.get("code_sha256_at_preparation", {})) | set(code_files(task["family"])))
    code = {}
    for f in files:
        try:
            code[f] = sha256((ROOT / f).read_bytes())
        except OSError:   # recorded as missing (reviewed at submission)
            code[f] = None
    try:
        import gurobipy
        grb = ".".join(map(str, gurobipy.gurobi.version()))
    except Exception:  # noqa: BLE001 - recorded as unknown
        grb = None
    versions = {}
    for mod in ("numpy", "scipy"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - recorded as unknown
            versions[mod] = None
    return {"python": sys.version.split()[0], "executable": sys.executable, "host": platform.node(), "gurobi": grb,
            "versions": versions,
            "slurm": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_RESTART_COUNT")},
            "argv": sys.argv, "task": task,
            "panel": {"file": panel_path.name if panel_path is not None and panel_path.exists() else None,
                      "sha256": sha256(panel_path.read_bytes()) if panel_path is not None and panel_path.exists() else None,
                      "status": panel.get("status")},
            "code_sha256": code, "code_commit": code_commit()}


def run_task(family: str, instance: str, arm: str, T: float, out_root: Path, panel_path: Path | None) -> dict:
    task_t0 = time.monotonic()
    k, outdir = claim_attempt(out_root / f"{stem(instance)}__{arm}")
    record, run = supervise(family, instance, arm, T, outdir)
    post0 = time.monotonic()
    fam = Family(family, instance)
    try:
        inc = evaluate(fam, record, run, arm)
    except Exception as exc:  # noqa: BLE001 - (b) evaluator failure: infrastructure (rerun), never a score
        run["infrastructure"].append(f"evaluator: {type(exc).__name__}: {exc}"[:300])
        inc = {"trajectory": [], "best": None, "verified": None, "claims": [], "infrastructure": []}
    if inc.get("best") is not None and inc.get("verified") is not True:
        run["infrastructure"].append(f"stdlib verifier disagrees or failed: {inc.get('verification')}"[:300])
    arm_result = None
    if (outdir / "arm_result.json").is_file():
        try:
            arm_result = json.loads((outdir / "arm_result.json").read_text())
        except (OSError, ValueError) as exc:   # (b) recorded; arm results are diagnostics
            arm_result = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    row = {"family": family, "instance": stem(instance), "arm": arm, "T": T, "run": run, "incumbents": inc,
           "arm_result": arm_result,
           "identity": identity({"family": family, "instance": str(instance), "instance_sha256": sha256(Path(instance).read_bytes()),
                                 "arm": arm, "T": T}, panel_path),
           "attempt": {"index": k, "dir": outdir.relative_to(out_root).as_posix()},
           "post_seconds": round(time.monotonic() - post0, 2), "task_seconds": round(time.monotonic() - task_t0, 2)}
    write_json_atomic(outdir / "result.json", row)
    return row


# ---------------------------------------------------------------- summary (frozen panel)

def identity_problems(r: dict, name: str, arm: str, panel: dict, panel_sha: str) -> list[str]:
    """A result is scored only if produced for this family, instance and arm, at the panel's T, under the summarized
    panel (sha256), on the frozen instance file and with every frozen code hash (and the frozen commit, if any)."""
    p = []
    ident = r.get("identity") if isinstance(r.get("identity"), dict) else {}
    task = ident.get("task") if isinstance(ident.get("task"), dict) else {}
    if r.get("family") != panel.get("family") or task.get("family") != panel.get("family"):
        p.append(f"family {r.get('family')!r} != {panel.get('family')!r}")
    if r.get("instance") != name:
        p.append(f"instance {r.get('instance')!r} != {name!r}")
    if r.get("arm") != arm or task.get("arm") != arm:
        p.append(f"arm {r.get('arm')!r}/{task.get('arm')!r} != {arm!r}")
    T = panel.get("T")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or x != T for x in (r.get("T"), task.get("T"))):
        p.append(f"T {r.get('T')!r}/{task.get('T')!r} != panel T {T!r}")
    got_panel = (ident.get("panel") or {}).get("sha256") if isinstance(ident.get("panel"), dict) else None
    if got_panel != panel_sha:
        p.append(f"panel sha256 {got_panel!r} != summarized panel {panel_sha}")
    want = ((panel.get("references") or {}).get(name) or {}).get("instance_sha256")
    if not want:
        p.append("the panel freezes no instance_sha256 for this instance")
    elif task.get("instance_sha256") != want:
        p.append(f"instance sha256 {task.get('instance_sha256')!r} != frozen {want}")
    frozen = panel.get("code_sha256_at_preparation") or {}
    got = ident.get("code_sha256") if isinstance(ident.get("code_sha256"), dict) else {}
    if not frozen:
        p.append("the panel freezes no code hashes")
    bad = sorted(f for f, h in frozen.items() if got.get(f) != h)
    if bad:
        p.append(f"code sha256 differs from the frozen panel: {bad}")
    if panel.get("code_commit") and ident.get("code_commit") != panel["code_commit"]:
        p.append(f"code commit {ident.get('code_commit')!r} != frozen {panel['code_commit']}")
    return p


def arm_attempts(out_root: Path, name: str, arm: str) -> list[tuple[int, Path]]:
    base = out_root / f"{name}__{arm}"
    if not base.is_dir():
        return []
    found = [(1, base)] + [(attempt_index(d.name), d) for d in base.iterdir() if d.is_dir() and attempt_index(d.name)]
    return sorted(found)


def result_infrastructure(r: dict) -> list[str]:
    return list((r.get("run") or {}).get("infrastructure") or []) + list((r.get("incumbents") or {}).get("infrastructure") or [])


def select_attempt(out_root: Path, name: str, arm: str, panel: dict, panel_sha: str) -> dict:
    """The earliest complete attempt with a valid identity and no infrastructure marker (else the earliest valid one,
    whose markers keep the arm undefined); none if no complete attempt has a valid identity."""
    seen, valid, clean = [], [], []
    for _, d in arm_attempts(out_root, name, arm):
        rel = d.relative_to(out_root).as_posix()
        f = d / "result.json"
        if not f.is_file():
            seen.append({"attempt": rel, "status": "incomplete"})
            continue
        try:
            r = json.loads(f.read_text())
            if not isinstance(r, dict):
                raise ValueError("result.json is not an object")
            problems = identity_problems(r, name, arm, panel, panel_sha)
            infra = [] if problems else result_infrastructure(r)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:   # (b) unreadable: infrastructure
            seen.append({"attempt": rel, "status": "unreadable", "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        seen.append({"attempt": rel, "status": "complete", "identity_problems": problems, "infrastructure": infra})
        if not problems:
            valid.append((rel, r))
            if not infra:
                clean.append((rel, r))
    pick = clean[0] if clean else (valid[0] if valid else None)
    return {"attempt": pick[0] if pick else None, "result": pick[1] if pick else None, "attempts": seen,
            "complete": any(a["status"] != "incomplete" for a in seen),
            "identity_problems": [] if pick else [p for a in seen for p in a.get("identity_problems", [])]
                                                  + [a["error"] for a in seen if a["status"] == "unreadable"]}


def score(r: dict, z_best: int, eps_p: Fraction, T: float, z_mip: int | None, t_mip: float | None, name: str) -> dict:
    from src.experiments.xl_scoring import Attempt, Reference, score_attempt
    inc = r["incumbents"]
    ref = Reference(T_f=float(T), z_best=Fraction(z_best), integral=True, eps_p=eps_p,
                    z_mip=None if z_mip is None else Fraction(z_mip), t_mip=t_mip, headline="near_best", instance=name)
    claims = [({"lower_bound": c["lower_bound"]} if c.get("_error") is None else {"_error": c["_error"]}) for c in inc.get("claims") or []]
    att = Attempt(incumbents=[(float(t), int(z)) for t, z in inc.get("trajectory") or []], method_ok=True, claims=claims,
                  infrastructure=result_infrastructure(r))
    rec = score_attempt(att, ref)
    return {k: rec.get(k) for k in ("feasible", "objective", "gap_best", "gap_best_exact", "primal_integral", "near_best",
                                    "t_near_best", "matches_or_beats_MIP", "t_matches_MIP", "speedup", "speedup_upper",
                                    "refuted_claim", "refutations", "success", "time_to_target", "infrastructure_errors",
                                    "first_incumbent_t")}


def accept(per: dict, rules: dict, T: float, z_best: int) -> dict:
    """The frozen per-instance rule (module docstring); every comparison exact on integers / Fractions."""
    eng = per["eng"]
    out: dict[str, Any] = {}
    out["eng_near_best"] = eng["success"]["near_best"]
    naive = per["naive"]
    unit = math.ceil(Fraction(str(rules["naive_quality_gap"])) * z_best)
    z_e, z_n = eng["objective"], naive["objective"]
    out["naive_quality"] = z_e is not None and (z_n is None or int(z_n) - int(z_e) >= unit)
    out["naive_quality_units"] = unit
    t_e = eng["time_to_target"]["near_best"]
    # physical (ungated) first verified attainment of near_best by the naive (preregistered: a refuted naive claim does
    # not make the naive "slow"; Codex review CG-1); the engineered time is the gated one (eng must succeed anyway)
    t_n = naive["t_near_best"]
    t_n_c = T if t_n is None else t_n
    out["naive_speed"] = (t_e is not None and t_n_c >= rules["naive_speed_ratio"] * max(t_e, 1e-3)
                          and t_n_c - t_e >= rules["saved_fraction"] * T)
    out["naive_separates"] = out["naive_quality"] or out["naive_speed"]
    out["controls_miss"] = {c: per[c]["success"]["near_best"] is False for c in CONTROLS}
    undefined = [a for a, m in per.items() if m["success"]["near_best"] is None and a in ("eng", "naive", *CONTROLS)]
    if undefined:
        return out | {"pass": None, "undefined": f"near_best undefined for {undefined} (infrastructure)"}
    out["pass"] = bool(out["eng_near_best"] and out["naive_separates"] and all(out["controls_miss"].values()))
    reasons = []
    if not out["eng_near_best"]:
        reasons.append("engineered misses near_best")
    if not out["naive_separates"]:
        reasons.append("naive NOT separated")
    reasons += [f"control {c} reaches near_best (NOT separated)" for c, ok in out["controls_miss"].items() if not ok]
    out["reasons"] = reasons
    return out


def instance_decision(name: str, tier: dict, panel: dict, out_root: Path, panel_sha: str) -> dict:
    sel = {a: select_attempt(out_root, name, a, panel, panel_sha) for a in PANEL_ARMS}
    base: dict[str, Any] = {"attempts": {a: s["attempt"] for a, s in sel.items()}}
    missing = [a for a in PANEL_ARMS if not sel[a]["complete"]]
    ident = {a: sel[a]["identity_problems"] for a in PANEL_ARMS if sel[a]["complete"] and sel[a]["result"] is None}
    if missing or ident:
        return base | {"decision": "pending", "acceptance": {"pass": None, "undefined": "arms pending or invalid (rerun)",
                                                             "missing": missing, **({"identity": ident} if ident else {})}}
    res = {a: s["result"] for a, s in sel.items()}
    unresolved = [a for a in PANEL_ARMS if result_infrastructure(res[a])]
    if unresolved:
        return base | {"decision": "pending", "acceptance": {"pass": None, "undefined": "infrastructure: rerun", "arms": unresolved,
                                                             "markers": {a: result_infrastructure(res[a]) for a in unresolved}}}
    z_pre = (panel["references"].get(name) or {}).get("z_pre")
    verified = [int(r["incumbents"]["best"]) for r in res.values()
                if r["incumbents"].get("best") is not None and r["incumbents"].get("verified") is True]
    cands = [int(z_pre)] if z_pre is not None else []
    cands += verified
    if not cands:
        return base | {"decision": "pending", "acceptance": {"pass": None, "undefined": "no z_best (no verified plan, no z_pre)"}}
    z_best = min(cands)
    mip = res["mip"]["incumbents"]
    z_mip = int(mip["best"]) if mip.get("best") is not None and mip.get("verified") is True else None
    t_mip = None
    if z_mip is not None:
        t_mip = next(float(t) for t, z in mip["trajectory"] if int(z) <= z_mip)
    eps_p = Fraction(str(tier["eps_p"]))
    T = float(panel["T"])
    per = {a: score(r, z_best, eps_p, T, z_mip, t_mip, name) for a, r in res.items()}
    acc = accept(per, panel["rules"], T, z_best)
    gaps = {a: per[a]["gap_best"] for a in per}
    ctrl = min((gaps[c] for c in CONTROLS), default=math.inf)
    diag = {"separation_interval": [gaps["eng"], ctrl],
            "nonempty": gaps["eng"] < ctrl, "mip_baseline": {"z_mip": z_mip, "t_mip": t_mip,
                                                             "eng_matches_or_beats_MIP": per["eng"]["matches_or_beats_MIP"]}}
    decision = {True: "pass", False: "fail", None: "pending"}[acc.get("pass")]
    return base | {"z_best": z_best, "z_pre": z_pre, "arms": per, "acceptance": acc, "diagnostics": diag, "decision": decision}


def tier_decision(stage1: list[str], holdout: list[str] | None, n_holdout: int) -> dict:
    n_pass, n_fail = stage1.count("pass"), stage1.count("fail")
    s1 = "pass" if n_pass >= STAGE1_REQUIRED else ("fail" if len(stage1) - n_fail < STAGE1_REQUIRED else "pending")
    if s1 != "pass":
        return {"stage1": s1, "holdout": "not_run", "tier": "rejected" if s1 == "fail" else "pending"}
    if holdout is None:
        return {"stage1": s1, "holdout": "not_run", "tier": "pending"}
    if n_holdout == 0:
        return {"stage1": s1, "holdout": "undeclared", "tier": "pending"}
    h = "fail" if "fail" in holdout else ("pass" if len(holdout) == n_holdout and all(x == "pass" for x in holdout) else "pending")
    return {"stage1": s1, "holdout": h, "tier": {"pass": "retained", "fail": "rejected", "pending": "pending"}[h]}


def summarize(out_root: Path, panel_path: Path) -> dict:
    raw = panel_path.read_bytes()
    panel = json.loads(raw)
    panel_sha = sha256(raw)
    summary: dict[str, Any] = {"panel": panel_path.name, "panel_sha256": panel_sha, "family": panel.get("family"),
                               "status": panel.get("status"), "instances": {}, "tiers": {}}
    if not str(panel.get("status", "")).startswith("FROZEN"):
        summary["warning"] = "panel is not frozen: decisions are provisional and must not be reported"
    expected = {f"{n}__{a}" for t in panel["tiers"].values() for n in t["stage1"] + t.get("holdout", []) for a in PANEL_ARMS}
    summary["unexpected"] = sorted(d.name for d in out_root.iterdir() if d.is_dir() and d.name not in expected) if out_root.is_dir() else []
    for tname, tier in panel["tiers"].items():
        s1 = []
        for name in tier["stage1"]:
            d = instance_decision(name, tier, panel, out_root, panel_sha) | {"tier": tname, "part": "stage1"}
            summary["instances"][name] = d
            s1.append(d["decision"])
        gate = tier_decision(s1, None, len(tier.get("holdout", [])))
        hold = None
        if gate["stage1"] == "pass":
            hold = []
            for name in tier.get("holdout", []):
                d = instance_decision(name, tier, panel, out_root, panel_sha) | {"tier": tname, "part": "holdout"}
                summary["instances"][name] = d
                hold.append(d["decision"])
        else:
            for name in tier.get("holdout", []):
                summary["instances"][name] = {"tier": tname, "part": "holdout", "decision": "not_run"}
        summary["tiers"][tname] = tier_decision(s1, hold, len(tier.get("holdout", []))) | {
            "eps_p": tier["eps_p"], "stage1_decisions": dict(zip(tier["stage1"], s1)),
            "holdout_decisions": dict(zip(tier.get("holdout", []), hold)) if hold is not None else None}
    if out_root.is_dir():
        write_json_atomic(out_root / f"summary_{panel.get('family')}.json", summary)
    return summary


# ---------------------------------------------------------------- development calibration of eps_p

def separating_eps(pairs: list[tuple[Fraction, Fraction]], required: int) -> dict:
    """pairs: (gap_eng, gap_best_control) per dev instance (gap_best_control = the smallest gap of any control; None when
    no control has an incumbent, i.e. +infinity). count(eps) = #{i: g_eng_i <= eps < g_ctrl_i} is
    piecewise constant on [b_k, b_k+1) between the sorted breakpoints; the maximal count is taken, and among its
    intervals the widest one (the first on ties); eps = its midpoint (lower end + 0.005 if unbounded), rounded to 1e-6
    inside the interval. No eps when the maximal count is below `required`."""
    if not pairs:
        return {"eps_p": None, "count": 0, "required": required, "reason": "no development pairs"}
    pts = sorted({g for g, _ in pairs} | {c for _, c in pairs if c is not None})
    best = None
    for k, lo in enumerate(pts):
        hi = pts[k + 1] if k + 1 < len(pts) else None
        cnt = sum(1 for g, c in pairs if g <= lo and (c is None or lo < c))
        width = (hi - lo) if hi is not None else Fraction(10**9)
        if best is None or cnt > best[0] or (cnt == best[0] and width > best[3]):
            best = (cnt, lo, hi, width)
    cnt, lo, hi = best[0], best[1], best[2]
    if cnt < required:
        return {"eps_p": None, "count": cnt, "required": required, "reason": "no interval separates enough dev instances",
                "best_interval": [str(lo), None if hi is None else str(hi)]}
    mid = lo + Fraction(1, 200) if hi is None else (lo + hi) / 2
    q = Fraction(math.floor(mid * 10**6), 10**6)
    if not (lo <= q and (hi is None or q < hi)):
        q = mid
    return {"eps_p": str(float(q)) if Fraction(str(float(q))) == q else str(q), "eps_p_exact": str(q), "count": cnt,
            "required": required, "interval": [str(lo), None if hi is None else str(hi)],
            "interval_float": [float(lo), None if hi is None else float(hi)]}


def freeze_eps(dev_root: Path, panel_path: Path) -> dict:
    """Proposal of eps_p per tier from the development runs (eng and every control on panel['tiers'][t]['dev']).
    Dev results are identity-checked against the panel's family, T, instance hashes and code hashes like Stage 1 (the
    panel used here is the PRE-freeze panel whose code hashes are those of the dev run)."""
    raw = panel_path.read_bytes()
    panel = json.loads(raw)
    panel_sha = sha256(raw)
    out: dict[str, Any] = {"panel": panel_path.name, "panel_sha256": panel_sha, "tiers": {}}
    for tname, tier in panel["tiers"].items():
        rows, pending = {}, []
        for name in tier.get("dev", []):
            sel = {a: select_attempt(dev_root, name, a, panel, panel_sha) for a in DEV_ARMS}
            res = {a: s["result"] for a, s in sel.items()}
            if any(r is None or result_infrastructure(r) for r in res.values()):
                pending.append(name)
                continue
            z_pre = (panel["references"].get(name) or {}).get("z_pre")
            vals = [int(r["incumbents"]["best"]) for r in res.values() if r["incumbents"].get("best") is not None
                    and r["incumbents"].get("verified") is True] + ([int(z_pre)] if z_pre is not None else [])
            if not vals:   # no z_best at all: the instance counts as not separated (kept in the denominator)
                rows[name] = {"z_best": None, "gap_eng": None, "gap_control": None, "note": "no z_best"}
                continue
            z_best = min(vals)
            T = float(panel["T"])
            # the same integrity gate as confirmation: a refuted or malformed claim fails near_best; an engineered run
            # failing it is not separated, a control failing it counts as missing near_best (no gap)
            gated = {a: score(r, z_best, Fraction(0), T, None, None, name)["refuted_claim"] for a, r in res.items()}
            norm = max(abs(z_best), 1)

            def gap(a):
                b = res[a]["incumbents"].get("best")
                return None if b is None or gated[a] else Fraction(int(b) - z_best, norm)
            cg = [gap(c) for c in CONTROLS if gap(c) is not None]
            ctrl = min(cg) if cg else None
            rows[name] = {"z_best": z_best, "gap_eng": gap("eng"), "gap_control": ctrl, "refuted": gated}
        required = math.ceil(0.75 * len(tier.get("dev", [])))
        # an engineered run without incumbent cannot be separated: its instance counts as not separated (left out of
        # the pairs while `required` stays ceil(3/4 n_dev))
        pairs = [(r["gap_eng"], r["gap_control"]) for r in rows.values() if r["gap_eng"] is not None]
        prop = separating_eps(pairs, required) if not pending else \
            {"eps_p": None, "reason": f"dev runs pending or infrastructure: {pending}"}
        out["tiers"][tname] = {"proposal": prop, "dev": {n: {k: (str(v) if isinstance(v, Fraction) else v) for k, v in r.items()}
                                                          for n, r in rows.items()}}
    write_json_atomic(dev_root / f"eps_proposal_{panel.get('family')}.json", out)
    return out


# ---------------------------------------------------------------- pre-launch identity

def prelaunch_problems(family: str, instance: str, T: float, panel_path: Path) -> list[str]:
    """Before an arm starts: the panel's family and T, the instance listed in the panel with the frozen sha256, and every
    frozen code hash equal to the checkout's (the same identity that --summarize / --freeze-eps enforce)."""
    panel = json.loads(panel_path.read_text())
    p = []
    if panel.get("family") != family:
        p.append(f"family {family!r} != panel {panel.get('family')!r}")
    if float(panel.get("T", -1)) != float(T):
        p.append(f"T {T} != panel T {panel.get('T')}")
    ref = (panel.get("references") or {}).get(stem(instance))
    if not ref:
        p.append(f"instance {stem(instance)!r} is not in the panel")
    elif not Path(instance).is_file():
        p.append(f"instance file {instance} is missing")
    elif sha256(Path(instance).read_bytes()) != ref.get("instance_sha256"):
        p.append(f"instance sha256 differs from the panel for {stem(instance)!r}")
    frozen = panel.get("code_sha256_at_preparation") or {}
    if not frozen:
        p.append("the panel freezes no code hashes")
    for f, h in sorted(frozen.items()):
        try:
            got = sha256((ROOT / f).read_bytes())
        except OSError:
            got = None
        if got != h:
            p.append(f"code sha256 differs from the panel: {f}")
    missing = sorted(set(code_files(family)) - set(frozen))
    if missing:
        p.append(f"panel code hashes miss {missing}")
    return p


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", nargs=3, metavar=("FAMILY", "INSTANCE", "ARM"))
    ap.add_argument("--child", nargs=3, metavar=("FAMILY", "INSTANCE", "ARM"))
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--freeze-eps", action="store_true")
    ap.add_argument("--panel", type=Path)
    ap.add_argument("--T", type=float, default=900.0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--start", type=float)
    ap.add_argument("--outdir")
    args = ap.parse_args()
    if args.child:
        child(*args.child, args.T, args.start, args.outdir)
        return
    if args.task:
        family, instance, arm = args.task
        known_arms = PANEL_ARMS + [f"_test_{m}" for m in _TestArm.MODES]
        if family not in FAMILIES or arm not in known_arms:   # validated without importing any arm implementation
            raise SystemExit(f"unknown family/arm {family}/{arm}")
        if args.panel is None:
            raise SystemExit("--task needs --panel (identity is checked before the arm starts)")
        try:
            problems = prelaunch_problems(family, instance, args.T, args.panel)
        except (OSError, ValueError) as exc:   # unreadable panel or instance: infrastructure, reported like a mismatch
            problems = [f"prelaunch check failed: {type(exc).__name__}: {exc}"[:300]]
        if problems:   # Codex review CG-1 #2: never spend a job on a result that summarize would reject
            print(json.dumps({"abort": "prelaunch identity check failed (infrastructure)", "problems": problems}))
            raise SystemExit(3)
        row = run_task(family, str(Path(instance).resolve()) if not Path(instance).is_absolute() else instance, arm,
                       args.T, args.out, args.panel)
        inc = row["incumbents"]
        print(json.dumps({"instance": row["instance"], "arm": arm, "best": inc.get("best"), "t_best": inc.get("t_best"),
                          "verified": inc.get("verified"), "infrastructure": row["run"]["infrastructure"]}))
        return
    if args.summarize:
        print(json.dumps(summarize(args.out, args.panel), indent=1, default=str))
        return
    if args.freeze_eps:
        print(json.dumps(freeze_eps(args.out, args.panel), indent=1, default=str))
        return
    ap.error("one of --task, --summarize, --freeze-eps")


if __name__ == "__main__":
    main()
