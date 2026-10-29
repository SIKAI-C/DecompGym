"""Screen a candidate harder family: monolithic, reference and shortcut solvers on 3 sizes x 3 seeds.

Every solve runs sequentially on one thread. Each objective is independently re-verified, and a
shortcut fails an instance when its verified objective is more than 1% from the monolithic optimum
(the benchmark's quality rule) or it produces no plan within the time limit.
Run as: python -m src.experiments.harder_screen --family lbbd --root outputs/harder/lbbd
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def lbbd_setup():
    from src.bd.lbbd.a_gen_data import generate_lbbd_instance
    from src.bd.lbbd.a_gold_bd import LBBDBenders
    from src.bd.lbbd.a_gold_monolithic import LBBDBaseline
    from src.bd.lbbd.d_verifier import verify_solution

    def mono(path, tl):
        res = LBBDBaseline(params={"TimeLimit": tl}).run(path)
        return res, {"solution": {"assignments": res["master_metadata"]["assignments"]}}

    def solver(**kw):
        def run(path, tl):
            res = LBBDBenders(time_limit=tl, **kw).run(path)
            return res, {"solution": {"assignments": res["assignments"]}}
        return run

    def reference(path, tl):
        res = LBBDBenders(time_limit=tl).run_branch_and_check(path)
        return res, {"solution": {"assignments": res["assignments"]}}

    return generate_lbbd_instance, verify_solution, mono, {"reference": reference, "S2_outer_loop_strengthened": solver(), "S1_nogood_no_relax": solver(strong_cuts=False, relaxation=False), "S1b_nogood_with_relax": solver(strong_cuts=False)}


def sslp_setup():
    from src.bd.sslp.a_gen_data import generate_sslp_instance
    from src.bd.sslp.a_gold_bd import SSLPIntegerLShaped
    from src.bd.sslp.a_gold_monolithic import SSLPBaseline
    from src.bd.sslp.d_verifier import verify_solution

    def mono(path, tl):
        res = SSLPBaseline(params={"TimeLimit": tl}).run(path)
        meta = res["master_metadata"]
        return res, {"solution": {"open_sites": meta["open_sites"], "scenarios": meta["scenarios"]}}

    def solver(**kw):
        def run(path, tl):
            res = SSLPIntegerLShaped(time_limit=tl, **kw).run(path)
            return res, {"solution": {"open_sites": res["open_sites"], "scenarios": res["scenarios"]}}
        return run

    return generate_sslp_instance, verify_solution, mono, {"reference": solver(), "S1_lp_lshaped_then_integer_eval": solver(integer_cuts=False)}


def vrptw_setup():
    from src.cg.vrptw.a_gen_data import generate_vrptw_instance
    from src.cg.vrptw.a_gold_cg import BranchAndPrice, RootColumnGeneration
    from src.cg.vrptw.a_gold_monolithic import VRPTWBaseline
    from src.cg.vrptw.d_verifier import verify_solution

    def mono(path, tl):
        res = VRPTWBaseline(params={"TimeLimit": tl}).run(path)
        return res, {"solution": {"routes": res["master_metadata"]["routes"]}}

    def price_and_branch(path, tl):
        res = RootColumnGeneration(time_limit=tl).run(path)
        return res, {"solution": {"routes": res["routes"]}}

    def reference(path, tl):
        res = BranchAndPrice(time_limit=tl).run(path)
        return res, {"solution": {"routes": res["routes"]}}

    return generate_vrptw_instance, verify_solution, mono, {"reference": reference, "S1_root_cg_then_integer_rmp": price_and_branch}


def partition_setup(family):
    def setup():
        import importlib
        gen = getattr(importlib.import_module(f"src.cg.{family}.a_gen_data"), f"generate_{family}_instance")
        verify = importlib.import_module(f"src.cg.{family}.d_verifier").verify_solution
        mono_cls = {"gcol": "GraphColoringBaseline", "bppc": "BPPCBaseline"}[family]
        bnp_cls = {"gcol": "GraphColoringBranchAndPrice", "bppc": "BPPCBranchAndPrice"}[family]
        key = {"gcol": "color_classes", "bppc": "bins"}[family]
        Mono = getattr(importlib.import_module(f"src.cg.{family}.a_gold_monolithic"), mono_cls)
        BnP = getattr(importlib.import_module(f"src.cg.{family}.a_gold_cg"), bnp_cls)

        def mono(path, tl):
            res = Mono(params={"TimeLimit": tl}).run(path)
            return res, {"solution": {key: res["groups"]}}

        def solver(root_only):
            def run(path, tl):
                res = BnP(time_limit=tl, root_only=root_only).run(path)
                return res, {"solution": {key: res[key]}}
            return run

        return gen, verify, mono, {"reference": solver(False), "S1_root_cg_then_integer_rmp": solver(True)}
    return setup


def pdptw_setup():
    from src.cg.pdptw.a_gen_data import generate_pdptw_instance
    from src.cg.pdptw.a_gold_cg import PDPTWBranchAndPrice, PDPTWRootColumnGeneration
    from src.cg.pdptw.a_gold_monolithic import PDPTWBaseline
    from src.cg.pdptw.d_verifier import verify_solution

    def wrap(cls):
        def run(path, tl):
            res = cls(time_limit=tl).run(path) if cls is not PDPTWBaseline else cls(params={"TimeLimit": tl}).run(path)
            return res, {"solution": {"routes": res["routes"]}}
        return run

    return generate_pdptw_instance, verify_solution, wrap(PDPTWBaseline), {"reference": wrap(PDPTWBranchAndPrice), "S1_root_cg_then_integer_rmp": wrap(PDPTWRootColumnGeneration)}


def nrp_setup():
    from src.cg.nrp.a_gen_data import generate_nrp_instance
    from src.cg.nrp.a_gold_cg import NRPBranchAndPrice
    from src.cg.nrp.a_gold_monolithic import NRPBaseline
    from src.cg.nrp.d_verifier import verify_solution

    def mono(path, tl):
        res = NRPBaseline(params={"TimeLimit": tl}).run(path)
        return res, {"solution": {"rosters": res["rosters"]}}

    def solver(root_only):
        def run(path, tl):
            res = NRPBranchAndPrice(time_limit=tl, root_only=root_only).run(path)
            return res, {"solution": {"rosters": res["rosters"]}}
        return run

    return generate_nrp_instance, verify_solution, mono, {"reference": solver(False), "S1_root_cg_then_integer_rmp": solver(True)}


SETUPS = {"nrp": nrp_setup, "pdptw": pdptw_setup, "lbbd": lbbd_setup, "sslp": sslp_setup, "vrptw": vrptw_setup, "gcol": partition_setup("gcol"), "bppc": partition_setup("bppc")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(SETUPS), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--sizes", default="small,medium,large")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    gen, verify, mono, solvers = SETUPS[args.family]()
    root = Path(args.root)
    rows = []
    for size in args.sizes.split(","):
        for seed in range(args.seeds):
            inst_dir = root / f"{args.family}_{size}_seed{seed:03d}"
            gen(seed, size, inst_dir)
            path = str(inst_dir / "data.json")
            row = {"instance": inst_dir.name}
            t = time.perf_counter()
            try:
                res, out = mono(path, args.time_limit)
                check = verify(path, out)
                opt = check["objective_value"]
                row["monolithic"] = {"obj": opt, "bound": res["global_lb"], "seconds": round(time.perf_counter() - t, 2), "verified": check["ok"]}
            except Exception as exc:
                # No proved optimum within the limit: the instance fails the 60-second reference gate.
                opt = None
                row["monolithic"] = {"error": f"{type(exc).__name__}: {exc}", "seconds": round(time.perf_counter() - t, 2)}
            for name, run in solvers.items():
                t = time.perf_counter()
                try:
                    res, out = run(path, args.time_limit)
                    check = verify(path, out)
                    obj = check["objective_value"]
                    gap = abs(obj - opt) / max(1.0, abs(opt)) if check["ok"] and opt is not None else None
                    row[name] = {"obj": obj, "lb": res.get("global_lb"), "gap_to_opt": gap, "pass": bool(check["ok"] and gap is not None and gap <= 0.01),
                                 "termination": res.get("termination_reason"), "iters": res.get("iters"), "seconds": round(time.perf_counter() - t, 2), "verified": check["ok"]}
                except Exception as exc:
                    row[name] = {"error": f"{type(exc).__name__}: {exc}", "pass": False, "seconds": round(time.perf_counter() - t, 2)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    (root / "screen.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
