"""Bound-side probes for the remaining Benders scale-up candidates (UFL, FCTP, stochastic CFL) under protocol v4
(development only; not calibration data). Phoenix only: no optimization on the laptop.

Modes
  --task FAMILY INSTANCE ARM[,ARM...]
        Each arm runs in its own process tree (new session), one after the other. The supervisor starts its clock
        just before spawning the child, reads length-framed messages from the child's stdout ("BDMSG <kind>
        <nbytes>\\n" + payload: incumbent, bound, certificate, result), timestamps each on complete receipt,
        persists it at once (events.jsonl appended and fsynced; certificate versions as immutable files
        certs/v<seq>.bin with their sha256) and SIGKILLs the process group at T. After all arms (outside T):
        incumbents received strictly before T are evaluated exactly (best one verified independently); the
        certificate version current at each checkpoint (the last complete version received strictly before it) is
        expanded to the canonical format and checked by src/experiments/xl_certificate.check_certificate (one view
        per task; final version first; cache by hash; development budget). Checkpoints: the frozen 60/300/600/900 s
        (<= T) plus a development grid (5, 10, 15, 30, 120, 180, 240, 450, 750 s); the frozen ones are flagged.
        Arms with an LP point get the exact LP_hi of that point (family routine).
  --reference FAMILY INSTANCE   "reference" (not an arm): the engineered LP phase run in process to a tight
        convergence tolerance for --T seconds; its final certificate certified canonically (LP_lo) and the exact
        value of its final LP point (LP_hi).
  --summarize FAMILY            per instance: the LP* interval [max certified LB over arms and references, min
        exact LP_hi], and per arm the certified bound at each checkpoint, bgap_LP at T and t_bound for eps_b in
        {1e-5, 1e-4} (bgap_LP = (LP_hi - LB)/max(|LP_hi|, 1)), plus primal values.
  --gen ...                     instance generation (see gen()).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MAGIC = b"BDMSG"
FROZEN = (60.0, 300.0, 600.0, 900.0)
DEV = (5.0, 10.0, 15.0, 30.0, 120.0, 180.0, 240.0, 450.0, 750.0)
GUROBI_MEM_GB = 15.5
DEV_CHECK_BUDGET = 2400.0


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stem(p: str) -> str:
    return Path(p).stem


# ---------------------------------------------------------------- families

class UFLFamily:
    name = "ufl"

    def arms(self) -> dict:
        from src.bd.ufl import g_bound_xl as g
        lp = {"mem_limit_gb": GUROBI_MEM_GB}
        return {"eng": (g.UFLEngLP, {}), "eng_nostab": (g.UFLEngLP, {"stabilize": False}), "naive_lp": (g.UFLNaiveLP, {}),
                "eng_lw": (g.UFLEngLP, {"lagr_warm": 0.1, "lagr_cap": 60.0, "purge_age": 5}),
                "eng_cap": (g.UFLEngLP, {"purge_age": 5, "max_add": 500}),
                "lp_bar": (g.UFLCompactLP, {"method": 2, **lp}), "lp_dual": (g.UFLCompactLP, {"method": 1, **lp}),
                "empty": (g.UFLStatic, {"kind": "empty"}), "oneshot_greedy": (g.UFLStatic, {"kind": "oneshot_greedy"}),
                "pool": (g.UFLPool, {}), "lagr": (g.UFLLagrangian, {}), "ils": (g.UFLILS, {}),
                "naive_int": (g.UFLNaiveInt, {}), "mip": (g.UFLCompactMIP, lp), "eng_primal": (g.UFLEngPrimal, {})}

    def reference_arm(self):
        from src.bd.ufl import g_bound_xl as g
        return g.UFLEngLP, {"conv_tol": 1e-11, "cert_interval": 1e9}

    def setup(self, path: str) -> None:
        from src.bd.ufl import g_bound_xl as g
        self.g = g
        self.inst = g.load_instance(path)
        self.srt = g.Sorted(self.inst.D)

    def evaluate(self, design) -> float | None:
        return self.g.design_value(self.inst, design)

    def verify(self, path: str, design, value: float) -> bool:
        obj = json.loads(Path(path).read_text())       # independent: Python ints from the raw file
        idx = sorted({int(i) for i in design})
        tot = sum(int(obj["f"][i]) for i in idx) + sum(min(int(obj["D"][i][j]) for i in idx) for j in range(int(obj["n"])))
        return abs(tot - value) <= 1e-6 * max(1.0, abs(value))

    def generic(self, data: bytes, version) -> bytes:
        return self.g.to_generic(data, self.inst, self.srt, version)

    def view(self, path: str):
        import src.experiments.xl_certificate as xc
        return xc.ufl_view(path)

    def lp_hi(self, path: str, point) -> tuple[Fraction | None, dict]:
        return self.g.exact_master_value(self.inst, self.srt, point)

    def info(self) -> dict:
        return {"m": self.inst.m, "n": self.inst.n}


class SCFLFamily:
    name = "scfl"

    def arms(self) -> dict:
        from src.bd.scfl import g_bound_xl as g
        from src.bd.scfl.b_mip_xl import SCFLExtensiveMIP
        from src.bd.scfl.c_naive_bd import SCFLNaiveBenders
        from src.bd.scfl.d_ref_bd_xl import SCFLBendersXL
        from src.bd.scfl.f_heuristic import SCFLHeuristicControl
        lp = {"mem_limit_gb": GUROBI_MEM_GB}
        return {"eng": (g.SCFLEngLP, {}), "eng_nostab": (g.SCFLEngLP, {"stabilize": False}), "naive_lp": (g.SCFLNaiveLP, {}),
                "eng_js": (g.SCFLEngLP, {"jensen_seed": 50}),
                "lp_bar": (g.SCFLCompactLP, {"method": 2, **lp}), "lp_dual": (g.SCFLCompactLP, {"method": 1, **lp}),
                "jensen": (g.SCFLJensen, lp), "empty": (g.SCFLStatic, {"kind": "empty"}),
                "oneshot_greedy": (g.SCFLStatic, {"kind": "oneshot_greedy"}), "oneshot_saa": (g.SCFLStatic, {"kind": "oneshot_saa"}),
                "pool": (g.SCFLPool, {}), "lagr": (g.SCFLLagrangian, {}),
                "eng_primal": (SCFLBendersXL, {"_nocert": True}), "naive_int": (SCFLNaiveBenders, {"_nocert": True}),
                "heur_ls": (SCFLHeuristicControl, {"_nocert": True, "mode": "ls"}),
                "heur_saa": (SCFLHeuristicControl, {"_nocert": True, "mode": "saa"}),
                "mip": (SCFLExtensiveMIP, {"_nocert": True, "strong": True, **lp})}

    def reference_arm(self):
        from src.bd.scfl import g_bound_xl as g
        return g.SCFLEngLP, {"conv_tol": 1e-11, "cert_interval": 1e9, "jensen_seed": 50}

    def setup(self, path: str) -> None:
        from src.bd.scfl.a_data import load_instance
        from src.bd.scfl.b_mip_xl import DesignEvaluator
        self.inst = load_instance(path)
        self.ev = DesignEvaluator(self.inst)

    def evaluate(self, design) -> float | None:
        return self.ev.objective(design)

    def verify(self, path: str, design, value: float) -> bool:
        import tempfile
        from src.bd.scfl.b_mip_xl import scfl_output
        from src.bd.scfl.e_verifier import verify_solution
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(scfl_output(self.inst, design, method="probe"), fh)
        v = verify_solution(path, fh.name)
        return bool(v["ok"] and abs(v["objective_value"] - value) <= 1e-6 * max(1.0, abs(value)))

    def generic(self, data: bytes, version) -> bytes:
        from src.bd.scfl.h_view_xl import to_generic
        return to_generic(data, self.inst.cost, version)

    def view(self, path: str):
        from src.bd.scfl.h_view_xl import scfl_view
        return scfl_view(path)

    def lp_hi(self, path: str, point) -> tuple[Fraction | None, dict]:
        from src.bd.scfl.g_bound_xl import exact_master_value
        return exact_master_value(path, self.inst, point)

    def info(self) -> dict:
        return {"I": self.inst.I, "J": self.inst.J, "S": self.inst.S}


class FCTPFamily:
    name = "fctp"

    def arms(self) -> dict:
        from src.bd.fctp import g_bound_xl as g
        lp = {"mem_limit_gb": GUROBI_MEM_GB}
        return {"lp_bar": (g.FCTPCompactLP, {"method": 2, **lp}), "lp_dual": (g.FCTPCompactLP, {"method": 1, **lp}),
                "lp_bar_x": (g.FCTPCompactLP, {"method": 2, "crossover": -1, **lp}),
                "balinski": (g.FCTPCompactLP, {"method": 1, "balinski": True, **lp}),
                "balinski_bar": (g.FCTPCompactLP, {"method": 2, "crossover": -1, "balinski": True, **lp})}

    def reference_arm(self):
        from src.bd.fctp import g_bound_xl as g
        return g.FCTPCompactLP, {"method": 1, "balinski": True}

    def setup(self, path: str) -> None:
        from src.bd.fctp.a_data_xl import load_fctp
        self.inst = load_fctp(path)

    def evaluate(self, design):
        return None

    def verify(self, path, design, value) -> bool:
        return False

    def generic(self, data: bytes, version) -> bytes:
        from src.bd.fctp.g_bound_xl import to_generic
        return to_generic(data, self.inst.tcost, version)

    def view(self, path: str):
        import src.experiments.xl_certificate as xc
        return xc.fctp_view(path)

    def lp_hi(self, path, point):
        return None, {"reason": "FCTP LP_hi comes from the arm (exact Balinski flows)"}

    def info(self) -> dict:
        return {"m": self.inst.m, "n": self.inst.n}


FAMILIES = {"ufl": UFLFamily, "scfl": SCFLFamily, "fctp": FCTPFamily}


# ---------------------------------------------------------------- child

def child(family: str, instance: str, arm: str, T: float, start_wall: float) -> None:
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    lock = threading.Lock()

    def send(kind: str, payload: bytes) -> None:
        with lock:
            out.write(MAGIC + b" " + kind.encode() + b" " + str(len(payload)).encode() + b"\n" + payload)

    def on_inc(design, stage=None):
        send("incumbent", json.dumps({"design": [int(i) for i in design], "stage": stage}).encode())

    def on_bound(b):
        send("bound", json.dumps({"value": float(b)}).encode())

    def on_cert(data: bytes):
        send("certificate", data)

    cls, kw = FAMILIES[family]().arms()[arm]
    kw = dict(kw)
    nocert = kw.pop("_nocert", False)
    left = T - (time.time() - start_wall) - 2.0
    args = dict(time_limit=max(1.0, left), threads=1, on_incumbent=on_inc, on_bound=on_bound, **kw)
    if not nocert:
        args["on_certificate"] = on_cert
    res = cls(**args).run(instance)
    slim = {}
    for k, v in (res or {}).items():
        if k in ("trace", "cut_log", "u_snapshots"):
            continue
        if k == "lp_point" and v is not None:
            slim[k] = [float(x) for x in v]
        elif k == "stats" and isinstance(v, dict):
            slim[k] = {a: b for a, b in v.items() if not isinstance(b, (list, dict))}
        elif not isinstance(v, (list, dict)) or k in ("solves", "simplex_bounds", "bound_trace", "design"):
            slim[k] = v
    send("result", json.dumps(slim, default=str).encode())


# ---------------------------------------------------------------- supervisor

def _read_exact(f, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def supervise(family: str, instance: str, arm: str, T: float, outdir: Path) -> tuple[list[dict], dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    certs = outdir / "certs"
    certs.mkdir(exist_ok=True)
    events = open(outdir / "events.jsonl", "ab")
    records: list[dict] = []
    torn = [0]
    start_wall = time.time()
    t0 = time.monotonic()
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child", family, instance, arm, "--T", str(T),
                             "--start", repr(start_wall)], stdout=subprocess.PIPE, stderr=open(outdir / "stderr.log", "wb"),
                            start_new_session=True, cwd=str(ROOT))

    def reader() -> None:
        f = proc.stdout
        seq = 0
        while True:
            hdr = f.readline()
            if not hdr:
                break
            parts = hdr.split()
            if len(parts) != 3 or parts[0] != MAGIC or not parts[2].isdigit():
                torn[0] += 1
                break
            kind, nbytes = parts[1].decode(errors="replace"), int(parts[2])
            payload = _read_exact(f, nbytes)
            if payload is None:
                torn[0] += 1
                break
            t = time.monotonic() - t0
            seq += 1
            rec = {"seq": seq, "t": round(t, 6), "kind": kind, "bytes": nbytes, "sha256": sha256(payload)}
            if kind == "certificate":
                fname = f"v{seq}.bin"
                with open(certs / fname, "wb") as fh:
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                rec["file"] = fname
            else:
                try:
                    rec["payload"] = json.loads(payload)
                except ValueError:
                    rec["payload"] = None
            events.write(json.dumps(rec).encode() + b"\n")
            events.flush()
            os.fsync(events.fileno())
            records.append(rec)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    killed = False
    try:
        proc.wait(timeout=max(0.0, T - (time.monotonic() - t0)))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        killed = True
        proc.wait()
    wall = time.monotonic() - t0
    th.join(timeout=60)
    events.close()
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    return records, {"returncode": proc.returncode, "killed_at_T": killed, "wall_seconds": round(wall, 3),
                     "torn_messages": torn[0], "children_peak_rss_mb": round(peak, 1)}


# ---------------------------------------------------------------- post-processing (outside T)

def checkpoints(T: float) -> list[float]:
    return sorted({c for c in FROZEN + DEV if c <= T} | {T})


def versions_at(records: list[dict], T: float) -> list[tuple[float, dict | None]]:
    certs = [r for r in records if r["kind"] == "certificate" and r["t"] < T]
    out = []
    for c in checkpoints(T):
        cur = None
        for r in certs:
            if r["t"] < c:
                cur = r
        out.append((c, cur))
    return out


def evaluate_incumbents(fam, path: str, records: list[dict], T: float, max_eval: int = 60) -> dict:
    incs = [r for r in records if r["kind"] == "incumbent" and r.get("payload")]
    late = sum(1 for r in incs if r["t"] >= T)
    incs = [r for r in incs if r["t"] < T]
    seen, uniq = set(), []
    for r in incs:
        key = tuple(sorted(r["payload"]["design"]))
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    skipped = 0
    if len(uniq) > max_eval:
        skipped = len(uniq) - max_eval
        uniq = uniq[:5] + uniq[-(max_eval - 5):]
    traj, best, best_design, invalid = [], None, None, 0
    t0 = time.monotonic()
    for r in uniq:
        v = fam.evaluate(r["payload"]["design"])
        if v is None:
            invalid += 1
            continue
        if best is None or v < best - 1e-9 * max(1.0, abs(v)):
            best, best_design = v, r["payload"]["design"]
            traj.append((r["t"], v, r["payload"].get("stage")))
    eval_s = time.monotonic() - t0
    verified = None
    if best_design is not None:
        try:
            verified = fam.verify(path, best_design, best)
        except Exception as exc:  # noqa: BLE001
            verified = f"error: {type(exc).__name__}: {exc}"[:200]
    return {"best": best, "t_first": traj[0][0] if traj else None, "t_best": traj[-1][0] if traj else None,
            "verified": verified, "invalid": invalid, "late": late, "distinct": len(seen), "skipped_eval": skipped,
            "trajectory": traj[-40:], "design": best_design, "eval_seconds": round(eval_s, 2)}


def certify_arm(fam, view, xc, outdir: Path, records: list[dict], T: float, budget: float) -> dict:
    certs = outdir / "certs"
    cps = versions_at(records, T)
    out: dict = {"versions": sum(1 for r in records if r["kind"] == "certificate"), "checkpoints": []}
    limits = xc.CheckerLimits(gurobi_mem_gb=12.0)
    cache: dict[str, dict] = {}
    used = 0.0
    order = list(range(len(cps)))
    order = [order[-1]] + order[:-1]
    res_by_idx: dict[int, dict] = {}
    for idx in order:
        c, cur = cps[idx]
        base = {"checkpoint": c, "frozen": c in FROZEN or c == T}
        if cur is None:
            res_by_idx[idx] = {**base, "seq": None, "cert_ok": False, "lb": None, "lb_float": None, "reason": "absent"}
            continue
        base.update(seq=cur["seq"], receipt_t=cur["t"])
        if str(cur["seq"]) in cache:
            res_by_idx[idx] = {**base, **cache[str(cur["seq"])], "cached": True}
            continue
        if used >= budget:
            res_by_idx[idx] = {**base, "cert_ok": False, "lb": None, "lb_float": None, "reason": "dev_budget_exhausted"}
            continue
        t0 = time.monotonic()
        data = (certs / cur["file"]).read_bytes()
        if sha256(data) != cur["sha256"]:
            r = {"cert_ok": False, "lb": None, "lb_float": None, "reason": "receipt_hash_mismatch"}
        else:
            try:
                g = fam.generic(data, cur["seq"])
                (certs / f"v{cur['seq']}.generic.json").write_bytes(g)
                conv = time.monotonic() - t0
                cr = xc.check_certificate(view, g, limits=limits)
                st = {k: v for k, v in (cr.stats or {}).items() if not isinstance(v, (list, dict))}
                r = {"cert_ok": bool(cr.cert_ok), "lb": str(cr.lb), "lb_float": float(cr.lb_float), "reason": cr.reason,
                     "conversion_seconds": round(conv, 2), "generic_bytes": len(g), "stats": st}
            except Exception as exc:  # noqa: BLE001 - recorded, not fatal
                r = {"cert_ok": False, "lb": None, "lb_float": None, "reason": f"error: {type(exc).__name__}: {exc}"[:300]}
        r["check_seconds"] = round(time.monotonic() - t0, 2)
        used += time.monotonic() - t0
        cache[str(cur["seq"])] = r
        res_by_idx[idx] = {**base, **r}
    out["checkpoints"] = [res_by_idx[i] for i in range(len(cps))]
    out["checker_seconds"] = round(used, 1)
    return out


def run_task(family: str, instance: str, arms: list[str], T: float, out_root: Path) -> None:
    fam = FAMILIES[family]()
    runs = {}
    for arm in arms:
        outdir = out_root / f"{stem(instance)}__{arm}"
        if (outdir / "result.json").exists():
            print(f"SKIP {outdir.name} (result exists)", flush=True)
            continue
        if outdir.exists():
            import shutil
            shutil.rmtree(outdir)
        records, run = supervise(family, instance, arm, T, outdir)
        runs[arm] = (outdir, records, run)
        print("RAN " + json.dumps({"arm": arm, **run}), flush=True)
    if not runs:
        return
    t0 = time.monotonic()
    fam.setup(instance)
    setup_s = time.monotonic() - t0
    view = xc = None
    view_s = None
    if any(any(r["kind"] == "certificate" for r in recs) for _, recs, _ in runs.values()):
        import src.experiments.xl_certificate as xc
        t1 = time.monotonic()
        try:
            view = fam.view(instance)
        except MemoryError as exc:
            view = None
            print(f"VIEW MemoryError {exc}", flush=True)
        view_s = time.monotonic() - t1
    for arm, (outdir, records, run) in runs.items():
        t2 = time.monotonic()
        inc = evaluate_incumbents(fam, instance, records, T)
        sb = [(r["t"], r["payload"]["value"]) for r in records if r["kind"] == "bound" and r.get("payload") and r["t"] < T]
        result = next((r["payload"] for r in records if r["kind"] == "result"), None)
        cert = certify_arm(fam, view, xc, outdir, records, T, DEV_CHECK_BUDGET) if view is not None else {"checkpoints": [], "versions": 0}
        lp_hi = None
        if result and result.get("lp_point") is not None:
            try:
                val, info = fam.lp_hi(instance, result["lp_point"])
                lp_hi = {"value": str(val) if val is not None else None, "float": float(val) if val is not None else None, **info}
            except Exception as exc:  # noqa: BLE001
                lp_hi = {"value": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
        if result and result.get("lp_hi_exact"):
            lp_hi = {"value": result["lp_hi_exact"], "float": float(Fraction(result["lp_hi_exact"])), "source": "arm"}
        row = {"family": family, "instance": stem(instance), "arm": arm, "T": T, "info": fam.info(), "run": run,
               "incumbents": inc,
               "solver_bounds": {"final": max((b for _, b in sb), default=None), "trace": sb[-30:], "label": "solver-reported (never scored)"},
               "certificates": cert, "lp_hi": lp_hi, "arm_result": result, "post_seconds": round(time.monotonic() - t2, 2),
               "setup_seconds": round(setup_s, 2), "view_seconds": round(view_s, 2) if view_s is not None else None}
        (outdir / "result.json").write_text(json.dumps(row, default=str, indent=1))
        print("TASK " + json.dumps({"arm": arm, "run": run, "best": inc["best"], "verified": inc["verified"],
                                    "lp_hi": (lp_hi or {}).get("float"),
                                    "cps": [(c["checkpoint"], c.get("cert_ok"), c.get("lb_float"), c.get("reason")) for c in cert["checkpoints"]],
                                    "result": {k: v for k, v in (result or {}).items() if k not in ("lp_point", "design", "simplex_bounds", "bound_trace", "solves")}},
                                   default=str), flush=True)


# ---------------------------------------------------------------- reference

def reference(family: str, instance: str, T: float, out_root: Path) -> dict:
    fam = FAMILIES[family]()
    cls, kw = fam.reference_arm()
    got: list[bytes] = []
    t0 = time.monotonic()
    res = cls(time_limit=T, threads=1, on_certificate=got.append, **kw).run(instance)
    run_s = time.monotonic() - t0
    out_root.mkdir(parents=True, exist_ok=True)
    fam.setup(instance)
    row = {"family": family, "instance": stem(instance), "label": "reference (not an arm)", "T": T, "run_seconds": round(run_s, 1),
           "termination": res.get("termination_reason"), "stats": {k: v for k, v in (res.get("stats") or {}).items() if not isinstance(v, (list, dict))},
           "versions": len(got)}
    if got:
        import src.experiments.xl_certificate as xc
        g = fam.generic(got[-1], "reference")
        (out_root / f"{stem(instance)}__reference_certificate.json").write_bytes(g)
        t1 = time.monotonic()
        view = fam.view(instance)
        row["view_seconds"] = round(time.monotonic() - t1, 1)
        t2 = time.monotonic()
        cr = xc.check_certificate(view, g, limits=xc.CheckerLimits(gurobi_mem_gb=12.0, max_seconds=3600.0))
        row["lp_lo"] = {"value": str(cr.lb), "float": float(cr.lb_float), "cert_ok": bool(cr.cert_ok), "reason": cr.reason,
                        "check_seconds": round(time.monotonic() - t2, 1), "certificate_sha256": sha256(g)}
    point = res.get("lp_point")
    if point is not None:
        val, info = fam.lp_hi(instance, point)
        row["lp_hi"] = {"value": str(val) if val is not None else None, "float": float(val) if val is not None else None, **info}
    elif res.get("lp_hi_exact"):
        row["lp_hi"] = {"value": res["lp_hi_exact"], "float": float(Fraction(res["lp_hi_exact"])), "source": "arm"}
    if row.get("lp_lo") and row.get("lp_hi", {}).get("value"):
        lo, hi = Fraction(row["lp_lo"]["value"]), Fraction(row["lp_hi"]["value"])
        row["interval"] = {"lo": str(lo), "hi": str(hi), "rigorous": lo <= hi, "rel_width": float((hi - lo) / max(abs(hi), 1))}
    row["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    (out_root / f"{stem(instance)}__reference.json").write_text(json.dumps(row, default=str, indent=1))
    return row


# ---------------------------------------------------------------- summary

def summarize(family: str, out_root: Path, eps_list=(1e-5, 1e-4)) -> dict:
    by_inst: dict[str, dict] = {}
    for f in sorted(out_root.glob("*__*/result.json")):
        r = json.loads(f.read_text())
        if r.get("family") != family:
            continue
        by_inst.setdefault(r["instance"], {})[r["arm"]] = r
    refs = {}
    for f in out_root.glob("*__reference.json"):
        r = json.loads(f.read_text())
        if r.get("family") == family:
            refs[r["instance"]] = r
    summary = {}
    for name, arms in sorted(by_inst.items()):
        los, his = [], []
        for arm, r in arms.items():
            for c in r["certificates"].get("checkpoints", []):
                if c.get("cert_ok") and c.get("lb"):
                    los.append((Fraction(c["lb"]), arm))
            if r.get("lp_hi") and r["lp_hi"].get("value"):
                his.append((Fraction(r["lp_hi"]["value"]), arm))
        ref = refs.get(name)
        if ref:
            if ref.get("lp_lo", {}).get("cert_ok"):
                los.append((Fraction(ref["lp_lo"]["value"]), "reference"))
            if ref.get("lp_hi", {}).get("value"):
                his.append((Fraction(ref["lp_hi"]["value"]), "reference"))
        lo = max(los) if los else None
        hi = min(his) if his else None
        iv = {"lo": float(lo[0]) if lo else None, "lo_from": lo[1] if lo else None, "hi": float(hi[0]) if hi else None,
              "hi_from": hi[1] if hi else None}
        if lo and hi:
            iv["rel_width"] = float((hi[0] - lo[0]) / max(abs(hi[0]), 1))
        z_best = min((r["incumbents"]["best"] for r in arms.values() if r["incumbents"].get("best") is not None), default=None)
        per = {}
        for arm, r in sorted(arms.items()):
            cps = r["certificates"].get("checkpoints", [])
            final = cps[-1] if cps else None
            e = {"term": (r.get("arm_result") or {}).get("termination_reason"), "killed": r["run"]["killed_at_T"],
                 "rss_mb": r["run"]["children_peak_rss_mb"], "best": r["incumbents"].get("best"),
                 "t_best": r["incumbents"].get("t_best"), "verified": r["incumbents"].get("verified"),
                 "solver_bound": r["solver_bounds"].get("final"),
                 "cert_T": final.get("lb_float") if final else None, "cert_ok_T": final.get("cert_ok") if final else None,
                 "cert_reason_T": final.get("reason") if final else None}
            if hi and final and final.get("cert_ok") and final.get("lb"):
                e["bgap_T"] = float((hi[0] - Fraction(final["lb"])) / max(abs(hi[0]), 1))
            for eps in eps_list:
                if hi:
                    tgt = hi[0] - Fraction(eps) * max(abs(hi[0]), 1)
                    hits = [c["checkpoint"] for c in cps if c.get("cert_ok") and c.get("lb") and Fraction(c["lb"]) >= tgt]
                    e[f"t_bound_{eps:g}"] = min(hits) if hits else None
                if r["solver_bounds"].get("trace") and hi:
                    tgt = float(hi[0]) * (1 - eps) if hi[0] > 0 else float(hi[0]) - eps
                    e[f"solver_t_{eps:g}"] = next((t for t, b in r["solver_bounds"]["trace"] if b >= tgt), None)
            if z_best is not None and e["best"] is not None:
                e["gap_best"] = (e["best"] - z_best) / max(1.0, abs(z_best))
            e["curve"] = [(c["checkpoint"], c.get("lb_float")) for c in cps if c.get("cert_ok")]
            per[arm] = e
        summary[name] = {"interval": iv, "z_best": z_best, "arms": per, "info": next(iter(arms.values()))["info"]}
    (out_root / f"summary_{family}.json").write_text(json.dumps(summary, default=str, indent=1))
    for name, s in summary.items():
        print(f"== {name} {s['info']} LP* in [{s['interval'].get('lo')}, {s['interval'].get('hi')}] "
              f"rel_width={s['interval'].get('rel_width')} z_best={s['z_best']}")
        for arm, e in s["arms"].items():
            print(f"   {arm:15s} term={str(e['term'])[:22]:22s} bgap_T={e.get('bgap_T')} t1e-5={e.get('t_bound_1e-05')} "
                  f"t1e-4={e.get('t_bound_0.0001')} best={e['best']} gap_best={e.get('gap_best')} rss={e['rss_mb']}")
    return summary


# ---------------------------------------------------------------- instance generation

def gen(args: list[str]) -> None:
    """ufl N CLASS SEED OUTDIR | scfl BASE_TXT S SEED OUTDIR | fctp M N SEED BASE_FCTP OUTDIR"""
    kind = args[0]
    if kind == "ufl":
        from src.bd.ufl.g_bound_xl import make_kg, write_instance
        N, cls, seed, out = int(args[1]), args[2], int(args[3]), Path(args[4])
        out.mkdir(parents=True, exist_ok=True)
        obj = make_kg(N, cls, seed)
        h = write_instance(obj, out / f"{obj['name']}.json")
        print(json.dumps({"name": obj["name"], "sha256": h}))
    elif kind == "scfl":
        from src.bd.scfl.a_data import make_stochastic, read_orlib_cap, write_instance
        base, S, seed, out = args[1], int(args[2]), int(args[3]), Path(args[4])
        out.mkdir(parents=True, exist_ok=True)
        inst = make_stochastic(read_orlib_cap(base), S, seed)
        h = write_instance(inst, out / f"{inst.name}.json")
        print(json.dumps({"name": inst.name, "sha256": h, "max_demand": inst.max_demand}))
    elif kind == "fctp":
        from src.bd.fctp.g_bound_xl import make_glover_like
        m, n, seed, base, out = int(args[1]), int(args[2]), int(args[3]), args[4], Path(args[5])
        out.mkdir(parents=True, exist_ok=True)
        text = make_glover_like(m, n, seed, base)
        name = f"gl{m}x{n}_{Path(base).stem}_g{seed}"
        (out / f"{name}.FCTP").write_text(text)
        print(json.dumps({"name": name, "sha256": sha256(text.encode())}))
    else:
        raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs=3, metavar=("FAMILY", "INSTANCE", "ARMS"))
    ap.add_argument("--child", nargs=3, metavar=("FAMILY", "INSTANCE", "ARM"), help=argparse.SUPPRESS)
    ap.add_argument("--start", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--reference", nargs=2, metavar=("FAMILY", "INSTANCE"))
    ap.add_argument("--summarize", metavar="FAMILY")
    ap.add_argument("--gen", nargs="+")
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.child:
        child(a.child[0], a.child[1], a.child[2], a.T, a.start if a.start is not None else time.time())
        return
    if a.gen:
        gen(a.gen)
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.task:
        run_task(a.task[0], a.task[1], a.task[2].split(","), a.T, out)
    if a.reference:
        r = reference(a.reference[0], a.reference[1], a.T, out)
        print("REFERENCE " + json.dumps(r, default=str), flush=True)
    if a.summarize:
        summarize(a.summarize, out)


if __name__ == "__main__":
    main()
