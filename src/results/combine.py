"""Combine audited campaign reports into one paper dataset, and the frozen model styles."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from matplotlib.colors import to_rgb

from src.results.policy import PAPER_MODELS, run_identity


STYLE_PATH = Path(__file__).with_name("styles.json")
STYLE = json.loads(STYLE_PATH.read_text())
MODELS = STYLE["models"]
METRICS = {"execution": ("E", "Execution"), "verification": ("V", "Verified decomposition"), "quality": ("Q", "Verified solution quality")}
MODES = {"plain": "Without reflection", "reflection": "With reflection"}
PROBLEMS = [("bd", p) for p in ("ufl", "fctp", "km", "lrp", "tsso")] + [("cg", p) for p in ("bp", "csp", "gap", "shift")]
SCIENTIFIC_FIELDS = ("prompt_protocol", "prompt_policies", "algorithm_judge_prompt_sha256", "execution_environment", "workers", "max_attempts", "solver_timeout", "solver_threads", "agent_timeout", "solver_cooldown_seconds", "judge_model", "judge_effort", "gap_threshold", "shared_first_attempt", "stop_after_success", "reflection_policy", "resource_guard", "agent_execution", "solver_execution", "wrong_target_reflection_failure")
CASE_FIELDS = ("suite", "problem", "instance", "size", "seed")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_style(model, *, grayscale=False):
    """Every model plot uses this frozen lookup; filtering never changes styles."""
    if model not in MODELS:
        raise ValueError(f"No frozen style for {model}; append it to styles.json")
    style = MODELS[model]
    color = style["color"]
    if grayscale:
        luminance = sum(c * w for c, w in zip(to_rgb(color), (0.2126, 0.7152, 0.0722), strict=True))
        color = (luminance,) * 3
    return {"color": color, "marker": style["marker"], "linestyle": style["linestyle"], "linewidth": 1.8, "markersize": 6.4, "markerfacecolor": "white", "markeredgewidth": 1.3}


def validate_compatibility(manifests):
    first = manifests[0]
    for manifest in manifests:
        if not run_identity(manifest)["paper_run"]:
            raise ValueError("Only the paper campaigns (results/policy.py PAPER_RUNS) may enter these figures")
        if manifest["instances"] != first["instances"]:
            raise ValueError("Case inventory, inputs, references or rendered prompts differ")
        for key in SCIENTIFIC_FIELDS:
            if manifest.get(key) != first.get(key):
                raise ValueError(f"Scientific settings differ: {key}")
        for key in ("reference_policy", "runtime_manifest_sha256"):
            if manifest["experiment_plan"].get(key) != first["experiment_plan"].get(key):
                raise ValueError(f"Reference/runtime policy differs: {key}")
    for name in ("src/agent/paired_runner.py", "src/agent/prompt_builder.py", "src/agent/output_verifier.py", "src/agent/native_solver.py"):
        hashes = [m["controller_sha256"].get(name) for m in manifests]
        if not all(hashes) or len(set(hashes)) != 1:
            raise ValueError(f"Scoring or prompt-construction source differs: {name}")


def validate_pairs_and_curves(report):
    """Check displayed aggregates against audited case outcomes, not status counters."""
    complete = [p for p in report["pairs"] if p["state"] == "complete" and p["model"] in PAPER_MODELS]
    budget = report["configuration"]["max_attempts"]
    for pair in complete:
        for mode in MODES:
            policy = pair["policies"][mode]
            if policy["state"] != "complete" or policy["audit_issues"]:
                raise ValueError("A complete pair contains an unaudited policy")
            previous = {m: False for m in METRICS}
            for k in range(1, budget + 1):
                values = policy["attempt_metrics"][str(k)]
                if not values["quality"] <= values["verification"] <= values["execution"]:
                    raise ValueError("Q <= V <= E violated")
                if any(previous[m] > values[m] for m in METRICS):
                    raise ValueError("Cumulative outcomes decreased")
                previous = values
        if pair["policies"]["plain"]["attempt_metrics"]["1"] != pair["policies"]["reflection"]["attempt_metrics"]["1"]:
            raise ValueError("Shared first-attempt outcomes differ")
    seen = set()
    for row in report["paired_curves"]:
        if row["model"] not in PAPER_MODELS:
            continue
        key = tuple(row[k] for k in ("model", "dimension", "group", "budget", "metric"))
        if key in seen:
            raise ValueError("Duplicate curve point")
        seen.add(key)
        cases = [p for p in complete if p["model"] == row["model"] and (row["dimension"] == "overall" or (p["suite"] + "/" + p["problem"] if row["dimension"] == "problem" else p[row["dimension"]]) == row["group"])]
        if len(cases) != row["paired_cases"] or not cases:
            raise ValueError("Curve denominator differs from audited paired cases")
        for mode in MODES:
            count = sum(p["policies"][mode]["attempt_metrics"][str(row["budget"])][row["metric"]] for p in cases)
            if count != row[mode + "_successes"] or not math.isclose(count / len(cases), row[mode + "_rate"], abs_tol=1e-12):
                raise ValueError("Curve count/rate differs from audited case outcomes")
    models = {p["model"] for p in complete}
    for model in models:
        cases = [p for p in complete if p["model"] == model]
        groups = {("overall", "all")} | {("suite", p["suite"]) for p in cases} | {("problem", p["suite"] + "/" + p["problem"]) for p in cases} | {("size", p["size"]) for p in cases}
        expected = {(model, dimension, group, k, metric) for dimension, group in groups for k in range(1, budget + 1) for metric in METRICS}
        if expected != {key for key in seen if key[0] == model}:
            raise ValueError("Missing or unexpected curve points")


def combine_reports(reports, manifests):
    validate_compatibility(manifests)
    rows, coverage, pairs = [], [], []
    seen_models, seen_cases = set(), set()
    expected_cases = {tuple(c[k] for k in CASE_FIELDS) for c in manifests[0]["instances"]}
    for report, manifest in zip(reports, manifests, strict=True):
        if report["cohort_manifest"] != manifest:
            raise ValueError("Audited report does not match its current frozen manifest")
        if report["audit_errors"] or report["snapshot"]["read_errors"]:
            raise ValueError("Resolve campaign audit/read errors before plotting")
        validate_pairs_and_curves(report)
        protocol = manifest["protocol_id"]
        eligible = [p for p in report["pairs"] if p["model"] in PAPER_MODELS]
        for model in dict.fromkeys(p["model"] for p in eligible):
            if model in seen_models:
                raise ValueError("A model occurs in multiple cohorts; choose a comparison explicitly")
            if model not in MODELS:
                raise ValueError(f"Append an explicit frozen style for {model}")
            seen_models.add(model)
            all_cases = [p for p in eligible if p["model"] == model]
            done = [p for p in all_cases if p["state"] == "complete"]
            keys = {tuple(p[k] for k in CASE_FIELDS) for p in done}
            if {tuple(p[k] for k in CASE_FIELDS) for p in all_cases} != expected_cases:
                raise ValueError("Model case inventory differs from the common inventory")
            coverage.append({"model": model, "label": MODELS[model]["label"], "completed_pairs": len(done), "planned_pairs": len(expected_cases), "primary": keys == expected_cases, "pair_states": dict(Counter(p["state"] for p in all_cases)), "cohort_protocol": protocol, "input_root": report["input_root"]})
        for pair in eligible:
            key = (pair["model"], *[pair[k] for k in CASE_FIELDS])
            if key in seen_cases:
                raise ValueError("Duplicate model/case pair")
            seen_cases.add(key)
            pairs.append(pair | {"cohort_protocol": protocol, "input_root": report["input_root"]})
        rows.extend(r | {"cohort_protocol": protocol, "input_root": report["input_root"]} for r in report["paired_curves"] if r["model"] in PAPER_MODELS)
    coverage.sort(key=lambda c: MODELS[c["model"]]["order"])
    primary = [c["model"] for c in coverage if c["primary"]]
    if not primary:
        raise ValueError("No model has complete audited coverage of the common inventory")
    rows = [r | {"primary_model": r["model"] in primary} for r in rows]
    return {"coverage": coverage, "primary_models": primary, "rows": rows, "pairs": pairs, "max_attempts": manifests[0]["max_attempts"], "cases": manifests[0]["instances"]}


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def points(data, model, metric="quality", dimension="overall", group="all"):
    return sorted((r for r in data["rows"] if r["model"] == model and r["metric"] == metric and r["dimension"] == dimension and r["group"] == group), key=lambda r: r["budget"])
