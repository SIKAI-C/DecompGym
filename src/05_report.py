"""Audit experiment coverage and export reproducible evaluation tables and charts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from src.agent.experiment_runner import SUITE_PROBLEMS
from src.agent.output_verifier import OutputVerifier
from src.reporting_policy import MODEL_EXCLUSIONS, require_reportable_artifact, require_reportable_model, require_reportable_report
from src.agent.policy_completion import TERMINAL_KIND, is_terminal_failure, TerminalCompletionError, load_policy_summary
from src.utils.json_utils import safe_load_json


MODE_FLAGS = {"plain": False, "reflection": True}
MODE_LABELS = {"plain": "Without reflection", "reflection": "With reflection"}
# ColorBrewer Paired, four classes; blind[4 - 3] = 1 in the embedded catalog.
COLORS = {"plain": "#a6cee3", "reflection": "#1f78b4"}
HATCHES = {"plain": "", "reflection": "//"}
# ColorBrewer Dark2, three classes; blind[3 - 3] = 1. Distinct markers and line styles also identify metrics.
METRIC_STYLES = {"execution": ("#1b9e77", "o", "-", 8), "verification": ("#d95f02", "s", "--", 6), "quality": ("#7570b3", "^", ":", 4)}
# Dark blue and green from the colorblind-safe ColorBrewer Paired four-class palette.
MODE_CURVE_STYLES = {"plain": ("#1f78b4", "o", "-"), "reflection": ("#33a02c", "s", "--")}


def expected_cases(data_root: Path, config: dict | None = None) -> set[tuple[str, str, str, str]]:
    manifest = json.loads((data_root / "manifest.json").read_text())
    scope = (config or {}).get("scope", {})
    modes = scope.get("modes", list(MODE_FLAGS))
    suite = scope.get("suite", "all")
    limit = scope.get("instances_per_problem")
    groups = defaultdict(list)
    for record in manifest["instances"]:
        if suite == "all" or record["suite"] == suite:
            groups[(record["suite"], record["problem"])].append(record)
    selected = []
    for records in groups.values():
        records.sort(key=lambda record: (next((i for i, size in enumerate(["small", "medium", "large"]) if f"_{size}_" in record["instance"]), 3), record["instance"]))
        selected.extend(records[:limit] if limit is not None else records)
    cases = set()
    for record in selected:
        if record["suite"] not in SUITE_PROBLEMS or record["problem"] not in SUITE_PROBLEMS[record["suite"]]:
            raise ValueError(f"Unknown dataset problem: {record['suite']}/{record['problem']}")
        if not record["complete"]:
            raise ValueError(f"Incomplete dataset instance: {record['instance']}")
        for mode in modes:
            if mode not in MODE_FLAGS:
                raise ValueError(f"Unknown reflection setting: {mode}")
            cases.add((mode, record["suite"], record["problem"], record["instance"]))
    return cases


def attempt_scores(attempts: list[dict], gold: float, threshold: float, budget: int) -> dict[str, Any]:
    selected = [attempt for attempt in attempts if attempt["attempt"] <= budget]
    verified = [attempt for attempt in selected if attempt.get("verify_ok")]
    gaps = []
    for attempt in verified:
        objective = attempt.get("objective")
        if objective is not None and math.isfinite(float(objective)):
            gaps.append(abs(float(objective) - gold) / max(1.0, abs(gold)))
    return {"execution": any(attempt.get("exec_ok") for attempt in selected),
            "verification": bool(verified), "quality": bool(gaps) and min(gaps) <= threshold,
            "best_verified_gap": min(gaps) if gaps else None}


def audit_instance(case: tuple[str, str, str, str], path: Path, summary: dict, canonical: Path, config: dict) -> list[str]:
    mode, suite, problem, name = case
    errors = []
    try:
        summary = load_policy_summary(path.parent, summary)
    except TerminalCompletionError as error:
        return [str(error)]
    terminal = is_terminal_failure(summary)
    flags = MODE_FLAGS[mode]
    if any(summary.get(key) != value for key, value in [("instance", name), ("suite", suite), ("problem", problem),
                                                        ("reflection", flags)]):
        errors.append("instance identity or experiment mode differs")
    if summary.get("max_attempts") != config["max_attempts"] or summary.get("gap_threshold") != config["gap_threshold"]:
        errors.append("attempt budget or gap threshold differs")
    attempts = summary["attempts"]
    last_attempt = summary["terminal_attempt"] if terminal else config["max_attempts"]
    if [attempt.get("attempt") for attempt in attempts] != list(range(1, last_attempt + 1)):
        errors.append("attempt sequence is incomplete or duplicated")
    data = path.parent / "data.json"
    if not data.exists() or hashlib.sha256(data.read_bytes()).digest() != hashlib.sha256((canonical / "data.json").read_bytes()).digest():
        errors.append("runtime input differs from the canonical instance")
    gold = OutputVerifier.load_gold_objective(canonical, suite)
    recorded_gold = summary.get("gold_objective")
    if recorded_gold is None and terminal:
        recorded_gold = attempts[0].get("gold_objective")
    if gold is None or recorded_gold is None or not math.isclose(float(recorded_gold), gold, rel_tol=1e-10, abs_tol=1e-8):
        errors.append("reference objective differs")
    successful = None
    for attempt in attempts:
        index = attempt.get("attempt")
        if attempt.get("reused_from_previous"):
            if successful is None or attempt.get("first_success_attempt") != successful["attempt"]:
                errors.append(f"attempt {index}: reused without an earlier qualifying success")
            elif any(attempt.get(key) != successful.get(key) for key in ["objective", "gap_to_gold", "exec_ok", "verify_ok"]):
                errors.append(f"attempt {index}: reused result differs from its success")
            continue
        directory = path.parent / "attempts" / f"attempt_{index:02d}"
        saved = safe_load_json(directory / "summary.json")
        if saved != attempt:
            errors.append(f"attempt {index}: attempt and instance summaries differ")
        invocation = safe_load_json(directory / "agent_invocation.json")
        native_adjudication = terminal and summary.get("backend") == "agy-cli"
        valid_invocation = (invocation and invocation.get("fresh_conversation") and invocation.get("resume") is False and invocation.get("sandbox")) if native_adjudication else (invocation and invocation.get("ephemeral") and invocation.get("ignore_user_config") and invocation.get("project_doc_max_bytes") == 0)
        if not valid_invocation:
            errors.append(f"attempt {index}: missing fresh-session invocation evidence")
        if attempt.get("verify_ok"):
            if not attempt.get("exec_ok"):
                errors.append(f"attempt {index}: verification recorded without successful execution")
            checked = OutputVerifier.verify_output(suite, problem, data, directory / "output.json", 1e-6)
            if not checked.get("ok"):
                errors.append(f"attempt {index}: saved output fails independent verification: {checked.get('issues')}")
            elif not math.isclose(float(attempt["objective"]), float(checked["objective_value"]), rel_tol=1e-6, abs_tol=1e-6):
                errors.append(f"attempt {index}: objective differs from recomputed value")
            if attempt.get("llm_verdict") != "pass":
                errors.append(f"attempt {index}: missing accepted algorithm review")
            review = directory / "alg_verifier.md"
            if not review.exists() or not review.read_text().strip() or review.read_text().strip().splitlines()[0].upper() != "VERDICT: PASS":
                errors.append(f"attempt {index}: saved algorithm review does not accept the method")
            if gold is not None and attempt.get("gap_to_gold") is not None:
                gap = abs(float(attempt["objective"]) - gold) / max(1.0, abs(gold))
                if not math.isclose(gap, attempt["gap_to_gold"], rel_tol=1e-7, abs_tol=1e-8):
                    errors.append(f"attempt {index}: recorded objective gap differs")
                if gap <= config["gap_threshold"]:
                    successful = attempt
    return errors


def aggregate(instances: list[dict]) -> dict:
    metrics = {}
    for budget in instances[0]["metrics"]:
        values = [instance["metrics"][budget] for instance in instances]
        gaps = [value["best_verified_gap"] for value in values if value["best_verified_gap"] is not None]
        metrics[budget] = {key: sum(value[key] for value in values) / len(values) for key in ["execution", "verification", "quality"]}
        metrics[budget].update(mean_verified_gap=statistics.mean(gaps) if gaps else None,
                               median_verified_gap=statistics.median(gaps) if gaps else None,
                               verified_gap_count=len(gaps))
    return {"num_instances": len(instances), "metrics": metrics}


def cumulative_attempt_curves(instances: list[dict], max_attempts: int) -> list[dict]:
    """Count instances passing at least once by each budget, with separate denominators per mode."""
    groups = defaultdict(list)
    for instance in instances:
        groups[instance["mode"]].append(instance)
    curves = []
    for mode, group in sorted(groups.items()):
        points = []
        for budget in range(1, max_attempts + 1):
            counts = {key: sum(instance["attempt_metrics"][str(budget)][key] for instance in group) for key in METRIC_STYLES}
            points.append({"budget": budget, "counts": counts, **{key: count / len(group) for key, count in counts.items()}})
        first_full = {key: next((point["budget"] for point in points if point["counts"][key] == len(group)), None) for key in METRIC_STYLES}
        curves.append({"mode": mode, "num_instances": len(group), "points": points, "first_full_round": first_full,
                       "first_all_full_round": max(first_full.values()) if all(value is not None for value in first_full.values()) else None})
    return curves


def attempt_breakdowns(instances: list[dict], max_attempts: int) -> dict[str, list[dict]]:
    """Preserve the full E/V/Q curves within each method, problem family, and size."""
    breakdowns = {}
    for dimension in ["suite", "problem", "size"]:
        groups = defaultdict(list)
        for instance in instances:
            key = f"{instance['suite']}/{instance['problem']}" if dimension == "problem" else instance[dimension]
            groups[key].append(instance)
        keys = [size for size in ["small", "medium", "large", "unknown"] if size in groups] if dimension == "size" else sorted(groups)
        breakdowns[dimension] = [{"group": key, "curves": cumulative_attempt_curves(groups[key], max_attempts)} for key in keys]
    return breakdowns


def build_report(root: Path, data_root: Path) -> dict:
    if (root / "campaign.json").is_file():
        from src.campaign_report import build_campaign_report
        return build_campaign_report(root)
    require_reportable_artifact(root)
    config = json.loads((root / "run_config.json").read_text())
    require_reportable_model(config.get("model"))
    budgets = sorted({budget for budget in [1, 5, 10, config["max_attempts"]] if budget <= config["max_attempts"]})
    expected = expected_cases(data_root, config)
    completed = {}
    partial = []
    errors = []
    for path in sorted(root.glob("*/*/*/*/instance_summary.json")):
        relative = path.relative_to(root).parts
        if ".interrupted_" in relative[3]:
            continue
        case = tuple(relative[:4])
        summary = safe_load_json(path)
        if summary:
            require_reportable_model(summary.get("model"))
        if case not in expected:
            continue
        try:
            summary = load_policy_summary(path.parent, summary or {})
        except TerminalCompletionError as error:
            errors.append({"case": "/".join(case), "issues": [str(error)]})
            continue
        terminal = is_terminal_failure(summary)
        if not summary or (len(summary.get("attempts", [])) != config["max_attempts"] and not terminal):
            partial.append("/".join(case))
            continue
        canonical = data_root / case[1] / case[2] / case[3]
        issues = audit_instance(case, path, summary, canonical, config)
        if issues:
            errors.append({"case": "/".join(case), "issues": issues})
            continue
        gold = float(summary.get("gold_objective", summary["attempts"][0].get("gold_objective")))
        attempts = summary["attempts"]
        real = [attempt for attempt in attempts if not attempt.get("reused_from_previous")]
        tokens = Counter()
        durations = Counter()
        for attempt in real:
            for step in ["generation", "algorithm_verification", "reflection"]:
                metrics = attempt.get("metrics", {}).get(step) or {}
                tokens.update({key: value for key, value in metrics.get("usage", {}).items() if isinstance(value, int)})
                durations[step] += metrics.get("elapsed_seconds", 0)
        attempt_metrics = {str(budget): attempt_scores(attempts, gold, config["gap_threshold"], budget) for budget in range(1, config["max_attempts"] + 1)}
        completed[case] = {"mode": case[0], "suite": case[1], "problem": case[2], "instance": case[3],
                           "size": next((size for size in ["small", "medium", "large"] if f"_{size}_" in case[3]), "unknown"),
                           "gold_objective": gold, "actual_attempts": len(real), "reused_slots": len(attempts) - len(real),
                           "completion_kind": summary.get("completion_kind"), "terminal_reason": summary.get("terminal_reason"),
                           "unexecuted_attempt_slots": summary.get("unexecuted_attempt_slots", 0),
                           "attempt_statuses": dict(Counter(attempt["status"] for attempt in real)),
                           "token_usage": dict(tokens), "step_seconds": dict(durations),
                           "metrics": {str(budget): attempt_metrics[str(budget)] for budget in budgets}, "attempt_metrics": attempt_metrics}
    problem_groups = defaultdict(list)
    suite_groups = defaultdict(list)
    for instance in completed.values():
        problem_groups[(instance["mode"], instance["suite"], instance["problem"])].append(instance)
        suite_groups[(instance["mode"], instance["suite"])].append(instance)
    rows = [{"mode": mode, "suite": suite, "problem": problem, **aggregate(instances)} for (mode, suite, problem), instances in sorted(problem_groups.items())]
    overall = [{"mode": mode, "suite": suite, **aggregate(instances)} for (mode, suite), instances in sorted(suite_groups.items())]
    missing = expected - set(completed)
    return {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "reporting_exclusions": list(MODEL_EXCLUSIONS.values()),
            "configuration": {key: config[key] for key in ["backend", "model", "max_attempts", "agent_timeout", "solver_timeout", "gap_threshold", "python", "platform"]} | {"workers": config.get("workers"), "solver_threads": config.get("solver_threads", 0), "solver_cooldown_seconds": config.get("solver_cooldown_seconds", 0), "scope": config.get("scope", {})},
            "coverage": {"expected_data_instances": len({case[1:] for case in expected}),
                         "expected_instance_runs": len(expected), "completed_instance_runs": len(completed),
                         "partial_instance_runs": partial, "missing_instance_runs": ["/".join(case) for case in sorted(missing)],
                         "complete": not missing and not errors},
            "audit_errors": errors, "overall": overall, "rows": rows, "instances": list(completed.values()),
            "attempt_curves": cumulative_attempt_curves(list(completed.values()), config["max_attempts"]),
            "attempt_breakdowns": attempt_breakdowns(list(completed.values()), config["max_attempts"])}


def percentages(row: dict, key: str) -> str:
    return " / ".join(f"{metrics[key] * 100:.1f}%" for metrics in row["metrics"].values())


def markdown_table(rows: list[dict], detailed: bool = False) -> str:
    if not rows:
        return "No completed instance runs yet."
    budget_label = "/".join(rows[0]["metrics"])
    last_budget = next(reversed(rows[0]["metrics"]))
    lines = ["| Mode | Suite | " + ("Problem | " if detailed else "") + f"N | solveE@{budget_label} | solveV@{budget_label} | solveQ@{budget_label} | Mean verified gap@{last_budget} |",
             "| --- | --- | " + ("--- | " if detailed else "") + "---: | --- | --- | --- | ---: |"]
    for row in rows:
        gap = row["metrics"][last_budget]["mean_verified_gap"]
        lines.append(f"| {MODE_LABELS[row['mode']]} | {row['suite'].upper()} | " + (f"{row['problem'].upper()} | " if detailed else "") + f"{row['num_instances']} | {percentages(row, 'execution')} | {percentages(row, 'verification')} | {percentages(row, 'quality')} | " + (f"{gap * 100:.3f}%" if gap is not None else "Unknown") + " |")
    return "\n".join(lines)


def export_attempt_curves(report: dict, destination: Path) -> None:
    """Export every observed attempt budget and the first budget attaining 100%."""
    require_reportable_report(report)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at_utc": report["generated_at_utc"], "configuration": report["configuration"], "coverage": report["coverage"],
               "definition": "Fraction of audited completed instances passing at least once by attempt k; later reused slots retain earlier successes.",
               "attempt_curves": report["attempt_curves"], "attempt_breakdowns": report["attempt_breakdowns"]}
    (destination / "attempt_curves.json").write_text(json.dumps(payload, indent=2))
    with (destination / "attempt_curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["mode", "num_instances", "budget", "execution_count", "verification_count", "quality_count", "execution", "verification", "quality"])
        writer.writeheader()
        for curve in report["attempt_curves"]:
            for point in curve["points"]:
                writer.writerow({"mode": curve["mode"], "num_instances": curve["num_instances"], "budget": point["budget"],
                                 **{key + "_count": count for key, count in point["counts"].items()}, **{key: point[key] for key in METRIC_STYLES}})
    with (destination / "attempt_breakdowns.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dimension", "group", "mode", "num_instances", "budget", "execution_count", "verification_count", "quality_count", "execution", "verification", "quality"])
        writer.writeheader()
        for dimension, groups in report["attempt_breakdowns"].items():
            for group in groups:
                for curve in group["curves"]:
                    for point in curve["points"]:
                        writer.writerow({"dimension": dimension, "group": group["group"], "mode": curve["mode"], "num_instances": curve["num_instances"], "budget": point["budget"],
                                         **{key + "_count": count for key, count in point["counts"].items()}, **{key: point[key] for key in METRIC_STYLES}})
    with (destination / "instances.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["mode", "suite", "problem", "size", "instance", "gold_objective", "actual_attempts", "reused_slots", "first_execution_round", "first_verification_round", "first_quality_round", "best_verified_gap"])
        writer.writeheader()
        for instance in report["instances"]:
            rounds = {f"first_{key}_round": next((int(budget) for budget, metrics in instance["attempt_metrics"].items() if metrics[key]), None) for key in METRIC_STYLES}
            writer.writerow({key: instance[key] for key in ["mode", "suite", "problem", "size", "instance", "gold_objective", "actual_attempts", "reused_slots"]} | rounds | {"best_verified_gap": instance["attempt_metrics"][str(report["configuration"]["max_attempts"])]["best_verified_gap"]})


def plot_attempt_curves(report: dict, destination: Path, *, grayscale: bool = False) -> None:
    require_reportable_report(report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    curves = report["attempt_curves"]
    if not curves:
        return
    configuration = report["configuration"]
    max_attempts = configuration["max_attempts"]
    labels = {"execution": "E: execution", "verification": "V: verification", "quality": f"Q: verified gap ≤ {configuration['gap_threshold'] * 100:g}%"}
    minimum_rate = min(point[key] * 100 for curve in curves for point in curve["points"] for key in METRIC_STYLES)
    lower_limit = min(90, math.floor(minimum_rate / 10) * 10)
    fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 5.8), sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.23, top=0.77, wspace=0.12)
    for ax, curve in zip(axes[0], curves):
        budgets = [point["budget"] for point in curve["points"]]
        for key, (color, marker, style, size) in METRIC_STYLES.items():
            values = [point[key] * 100 for point in curve["points"]]
            ax.plot(budgets, values, color="#333333" if grayscale else color, marker=marker, linestyle=style, markersize=size,
                    markerfacecolor="white", markeredgewidth=1.4, linewidth=2.2, label=labels[key], zorder=3)
        ax.set_title(f"{MODE_LABELS[curve['mode']]} (n = {curve['num_instances']})", fontsize=12, pad=12)
        ax.set_xlabel("Attempt budget k", fontsize=11, labelpad=9)
        ax.set_xticks(range(1, max_attempts + 1))
        ax.set_xlim(0.75, max_attempts + 0.25)
        ax.set_ylim(lower_limit - 0.5, 103)
        ax.set_yticks(range(lower_limit, 101, 10))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.grid(color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#9ca3af")
        full_round = curve["first_all_full_round"]
        if full_round is not None:
            ax.axvline(full_round, color="#6b7280", linestyle="--", linewidth=1, zorder=2)
            ax.text(full_round, lower_limit + 0.16 * (100 - lower_limit), f"All E/V/Q = 100%\nat k = {full_round}", fontsize=10, ha="center", va="center",
                    color="#374151", bbox={"facecolor": "white", "edgecolor": "none", "pad": 4})
        rounds = "   ·   ".join(f"{key[0].upper()}: {curve['first_full_round'][key] if curve['first_full_round'][key] is not None else 'not reached'}" for key in METRIC_STYLES)
        ax.text(0.5, -0.25, "First round at 100% — " + rounds, transform=ax.transAxes, fontsize=10.5, ha="center", color="#374151")
    axes[0, 0].set_ylabel("Instances passing by attempt k (%)", fontsize=11, labelpad=9)
    coverage = report["coverage"]
    qualifier = "" if coverage["complete"] else f" · partial ({coverage['completed_instance_runs']}/{coverage['expected_instance_runs']} runs)"
    fig.suptitle("Cumulative solve rates" + qualifier, fontsize=17, y=0.975)
    model = configuration.get("model") or "CLI default model"
    fig.text(0.5, 0.916, f"{model}  ·  All problem families  ·  {configuration['solver_timeout']:g} s per generated solver", ha="center", fontsize=10.5, color="#4b5563")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=3, frameon=False, fontsize=10.5)
    fig.text(0.5, 0.045, "Cumulative success: a case counts once it passes; later unused attempt slots retain that success.", ha="center", fontsize=9, color="#4b5563")
    destination.mkdir(parents=True, exist_ok=True)
    name = "overall_attempt_curves" + ("_grayscale" if grayscale else "")
    for suffix in ["svg", "png", "pdf"]:
        fig.savefig(destination / f"{name}.{suffix}", dpi=200)
    plt.close(fig)


def plot_attempt_breakdowns(report: dict, destination: Path) -> None:
    require_reportable_report(report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    configuration = report["configuration"]
    max_attempts = configuration["max_attempts"]
    modes = [mode for mode in MODE_FLAGS if any(curve["mode"] == mode for curve in report["attempt_curves"])]
    if not modes:
        return
    destination.mkdir(parents=True, exist_ok=True)
    quality_label = f"Q: verified gap ≤ {configuration['gap_threshold'] * 100:g}%"
    metric_labels = {"execution": "E: execution", "verification": "V: verification", "quality": quality_label}
    qualifier = "" if report["coverage"]["complete"] else " · partial completed runs"
    for dimension, title in [("suite", "Benders decomposition and column generation"), ("size", "Performance by instance size")]:
        groups = report["attempt_breakdowns"][dimension]
        fig, axes = plt.subplots(len(groups), len(modes), figsize=(6 * len(modes), 3.6 * len(groups) + 1), squeeze=False, sharey=True)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.85, wspace=0.13, hspace=0.5)
        for row, group in enumerate(groups):
            curves = {curve["mode"]: curve for curve in group["curves"]}
            for column, mode in enumerate(modes):
                ax = axes[row, column]
                if mode not in curves:
                    ax.set_visible(False)
                    continue
                curve = curves[mode]
                budgets = [point["budget"] for point in curve["points"]]
                for key, (color, marker, style, size) in METRIC_STYLES.items():
                    ax.plot(budgets, [point[key] * 100 for point in curve["points"]], color=color, marker=marker, linestyle=style, markersize=size,
                            markerfacecolor="white", markeredgewidth=1.2, linewidth=2, label=metric_labels[key])
                group_label = {"bd": "Benders decomposition", "cg": "Column generation"}.get(group["group"], group["group"].capitalize())
                ax.set_title(f"{group_label} · {MODE_LABELS[mode]} (n = {curve['num_instances']})", fontsize=11)
                ax.set_xticks(range(1, max_attempts + 1))
                ax.set_xlim(0.75, max_attempts + 0.25)
                ax.set_ylim(-3, 105)
                ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
                ax.grid(color="#e5e7eb", linewidth=0.8)
                ax.set_axisbelow(True)
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
                if column == 0:
                    ax.set_ylabel("Cumulative passing instances (%)", fontsize=10)
                rounds = " · ".join(f"{key[0].upper()}: {curve['first_full_round'][key] if curve['first_full_round'][key] is not None else 'not reached'}" for key in METRIC_STYLES)
                ax.text(0.5, -0.21, "First 100% — " + rounds, transform=ax.transAxes, ha="center", fontsize=9.5, color="#374151")
        fig.suptitle(title + qualifier, fontsize=16, y=0.98)
        fig.text(0.5, 0.935, f"{configuration.get('model') or 'CLI default model'} · cumulative E/V/Q across all families in each group", ha="center", fontsize=10, color="#4b5563")
        handles, labels = next(ax.get_legend_handles_labels() for ax in axes.flat if ax.get_visible())
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=3, frameon=False, fontsize=10)
        fig.supxlabel("Attempt budget k", y=0.025, fontsize=11)
        for suffix in ["svg", "png", "pdf"]:
            fig.savefig(destination / f"{dimension}_attempt_curves.{suffix}", dpi=180)
        plt.close(fig)

    groups = report["attempt_breakdowns"]["problem"]
    rows = math.ceil(len(groups) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3 * rows + 1), squeeze=False, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.86, wspace=0.16, hspace=0.48)
    for ax, group in zip(axes.flat, groups):
        counts = []
        rounds = []
        for curve in group["curves"]:
            mode = curve["mode"]
            color, marker, style = MODE_CURVE_STYLES[mode]
            ax.plot([point["budget"] for point in curve["points"]], [point["quality"] * 100 for point in curve["points"]],
                    color=color, marker=marker, linestyle=style, markersize=5, markerfacecolor="white", linewidth=2, label=MODE_LABELS[mode])
            counts.append(str(curve["num_instances"]))
            rounds.append(f"{'Without' if mode == 'plain' else 'With'}: {curve['first_full_round']['quality'] or 'not reached'}")
        ax.set_title(group["group"].upper().replace("/", " · ") + f" (n = {'/'.join(dict.fromkeys(counts))} per setting)", fontsize=11)
        ax.set_xticks([budget for budget in [1, 3, 5, 7, 10] if budget <= max_attempts])
        ax.set_xlim(0.75, max_attempts + 0.25)
        ax.set_ylim(-3, 105)
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.grid(color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.text(0.5, -0.22, "Q first 100% — " + " · ".join(rounds), transform=ax.transAxes, ha="center", fontsize=9, color="#374151")
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative quality success (%)", fontsize=10)
    fig.suptitle("Performance by problem type" + qualifier, fontsize=16, y=0.985)
    fig.text(0.5, 0.945, f"{configuration.get('model') or 'CLI default model'} · {quality_label} · E/V/Q counts for every round accompany this figure", ha="center", fontsize=10, color="#4b5563")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False, fontsize=10)
    fig.supxlabel("Attempt budget k", y=0.025, fontsize=11)
    for suffix in ["svg", "png", "pdf"]:
        fig.savefig(destination / f"problem_attempt_curves.{suffix}", dpi=180)
    plt.close(fig)


def plot_report(report: dict, destination: Path) -> None:
    require_reportable_report(report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = {(row["mode"], row["suite"], row["problem"]): row for row in report["rows"]}
    plot_budget = str(min(10, report["configuration"]["max_attempts"]))
    labels = {"bd": ["ufl", "fctp", "km", "lrp", "tsso"], "cg": ["bp", "csp", "gap", "shift"]}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), layout="constrained")
    for suite_index, suite in enumerate(["bd", "cg"]):
        modes = list(MODE_FLAGS)
        problems = labels[suite]
        width = 0.8 / len(modes)
        for column, metric in enumerate(["execution", "verification", "quality"]):
            ax = axes[suite_index, column]
            for index, mode in enumerate(modes):
                values = [rows[(mode, suite, problem)]["metrics"][plot_budget][metric] * 100 if (mode, suite, problem) in rows else math.nan for problem in problems]
                xs = [x - 0.4 + width * (index + 0.5) for x in range(len(problems))]
                ax.bar(xs, values, width=width, color=COLORS[mode], edgecolor="#333333", linewidth=0.5, hatch=HATCHES[mode])
            ax.set_xticks(range(len(problems)), [problem.upper() for problem in problems])
            ax.set_ylim(0, 105)
            ax.set_ylabel("Completed-instance yield (%)")
            ax.set_title(f"{suite.upper()} · solve{metric[0].upper()}@{plot_budget}")
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            if not any(key[1] == suite for key in rows):
                ax.text(0.5, 0.5, "Pending", transform=ax.transAxes, ha="center", va="center", color="#666666")
    coverage = report["coverage"]
    qualifier = "" if coverage["complete"] else f" (partial: {coverage['completed_instance_runs']}/{coverage['expected_instance_runs']} runs)"
    fig.suptitle("Codex decomposition experiments" + qualifier)
    handles = [Patch(facecolor=COLORS[mode], edgecolor="#333333", hatch=HATCHES[mode], label=MODE_LABELS[mode]) for mode in MODE_FLAGS]
    axes[0, 1].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.14), ncol=4, fontsize=8)
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "solve_rates.svg")
    fig.savefig(destination / "solve_rates.png", dpi=180)
    plt.close(fig)
    plot_attempt_curves(report, destination)
    plot_attempt_breakdowns(report, destination)


def export_report(report: dict, destination: Path, charts: bool = True) -> None:
    if report.get("layout") == "native_campaign":
        from src.campaign_report import export_campaign_report
        export_campaign_report(report, destination)
        return
    require_reportable_report(report)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evaluation.json").write_text(json.dumps(report, indent=2))
    export_attempt_curves(report, destination)
    fields = ["mode", "suite", "problem", "num_instances", "budget", "execution", "verification", "quality", "mean_verified_gap", "median_verified_gap", "verified_gap_count"]
    with (destination / "evaluation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            for budget, metrics in row["metrics"].items():
                writer.writerow({key: row[key] for key in ["mode", "suite", "problem", "num_instances"]} | {"budget": budget} | metrics)
    coverage = report["coverage"]
    state = "Complete" if coverage["complete"] else "Partial"
    protocol = report["configuration"]
    text = f"- **Coverage:** {state}; {coverage['completed_instance_runs']}/{coverage['expected_instance_runs']} instance runs.\n- **Protocol:** {protocol['max_attempts']} attempt slots per instance, early-stop reuse, and a {protocol['gap_threshold'] * 100:g}% objective-gap threshold; with and without reflection.\n- **Execution budget:** {protocol['solver_timeout']} seconds for the entire generated solver per attempt, with {protocol['solver_threads']} solver thread(s); model calls have a separate {protocol['agent_timeout']}-second deadline. Cooling pauses are outside the scored execution time.\n- **Backend:** locally authenticated Codex CLI; requested model: {report['configuration']['model'] or 'CLI default (model identifier not recorded)'}.\n- **Verification:** saved outputs are rechecked against canonical data; interrupted backups and partial runs are excluded from yield denominators.\n- **Files:** evaluation.json includes coverage, audit details, per-instance metrics, token usage, and step times; evaluation.csv contains the per-problem table.\n- **Interpretation:** token totals and step times include actual attempts only; time sums are work totals, not the parallel sweep's wall-clock duration.\n\n" + markdown_table(report["overall"]) + "\n\n" + markdown_table(report["rows"], detailed=True) + "\n"
    if report["audit_errors"]:
        text += "\n- **Audit errors:** inspect evaluation.json before interpreting these results.\n"
    text += "\n- **Attempt curves:** attempt_curves.csv/json contain E/V/Q counts and rates for every round, with the first round reaching 100%.\n\n| Mode | N | E first 100% | V first 100% | Q first 100% | All E/V/Q first 100% |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
    for curve in report["attempt_curves"]:
        values = [curve["first_full_round"][key] for key in METRIC_STYLES] + [curve["first_all_full_round"]]
        text += f"| {MODE_LABELS[curve['mode']]} | {curve['num_instances']} | " + " | ".join(str(value) if value is not None else "Not reached" for value in values) + " |\n"
    if charts:
        text += "\n![Cumulative E/V/Q success by attempt budget](figures/overall_attempt_curves.svg)\n"
    (destination / "report.md").write_text(text)
    if charts:
        plot_report(report, destination / "figures")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/experiments/default_run"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=None, help="Derived report destination; native campaigns default to timestamped outputs/reports, legacy runs to <input-root>/report.")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        args.output = (Path("outputs/reports") / (args.input_root.name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))) if (args.input_root / "campaign.json").is_file() else args.input_root / "report"
    report = build_report(args.input_root, args.data_root)
    if not report["coverage"]["complete"] and not args.allow_partial:
        raise SystemExit(f"Incomplete or unaudited matrix: {report['coverage']['completed_instance_runs']}/{report['coverage']['expected_instance_runs']}; use --allow-partial only for a labeled interim report.")
    export_report(report, args.output, charts=not args.no_charts)
    print(json.dumps({key: value for key, value in report["coverage"].items() if not isinstance(value, list)}, indent=2))
    print(f"Report written to {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
