"""Candidate-level failure anatomy for the DecompBench paper cohort.

Reads the saved per-attempt summaries of the 972 audited pairs (read-only) and the manual
method-judge labels in src/results/failure_labels.json. Each physical candidate is assigned the
first gate it fails; the shared first candidate is counted once. Outputs three figures, the
failure tables and a JSON count file. It never launches an experiment, solver, or model call.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from src.results.paper_figures import FAMILIES, STYLE, TABLE_BUDGETS, digest, label, save, setup


LABELS_PATH = Path(__file__).with_name("failure_labels.json")

# First failed gate, in evaluation order. Success is neutral; failure hues pass the dataviz
# validator (adjacent CVD dE >= 24.7); the low-contrast yellow carries direct count labels.
CATEGORIES = (
    ("success", "Quality success", "#d9d9d6"),
    ("gap", "Feasible, gap > 1%", "#eda100"),
    ("method", "Method rejected", "#6250d6"),
    ("infeasible", "Infeasible or inconsistent", "#eb6834"),
    ("no_objective", "Ran, no objective", "#2a78d6"),
    ("no_run", "Not delivered or crashed", "#52514e"),
)
JUDGE_LABELS = ("certificate_overclaim", "inactive_decomposition", "solution_bypass")


def first_failed_gate(summary):
    status = summary["status"]
    if status == "agent_failed" or status in ("exec_failed", "timeout"):
        return "no_run"
    if status == "objective_not_available":
        return "no_objective"
    if status == "verify_failed":
        return "method" if summary.get("feasibility_ok") else "infeasible"
    if status == "feasible_not_optimal":
        return "gap"
    assert status == "ok" and summary["verify_ok"] and summary["llm_verdict"] == "pass", summary["status"]
    return "success"


def candidates(data):
    """Yield one record per physical candidate committed to a scored policy trajectory."""
    models = set(data["primary_models"])
    for pair in data["pairs"]:
        if pair["model"] not in models:
            continue
        root = Path(pair["input_root"])
        for mode in ("plain", "reflection"):
            policy = pair["policies"][mode]
            for attempt in range(1, policy["actual_attempts"] + 1):
                if mode == "reflection" and attempt == 1:
                    continue  # shared first candidate, already counted under plain
                summary = json.loads((root / policy["path"] / "attempts" / f"attempt_{attempt:02d}" / "summary.json").read_text())
                assert summary["attempt"] == attempt
                yield {"model": pair["model"], "suite": pair["suite"], "problem": pair["problem"], "instance": pair["instance"], "mode": mode, "attempt": attempt,
                       "gate": first_failed_gate(summary), "judge_fail": summary.get("llm_verdict") == "fail",
                       "numeric_pass": bool(summary.get("feasibility_ok")) and summary.get("gap_to_gold") is not None and summary["gap_to_gold"] <= .01,
                       "quality": bool(policy["attempt_metrics"][str(attempt)]["quality"]) and summary["status"] == "ok"}


def analyse(data):
    records = list(candidates(data))
    labels = json.loads(LABELS_PATH.read_text())["labels"]
    by_key = {(x["model"], x["mode"], x["instance"], x["attempt"]): x["label"] for x in labels}
    judged = {(r["model"], r["mode"], r["instance"], r["attempt"]) for r in records if r["judge_fail"]}
    assert judged == set(by_key), "manual labels must cover exactly the observed judge failures"
    for r in records:
        r["judge_label"] = by_key.get((r["model"], r["mode"], r["instance"], r["attempt"]))
    first = [r for r in records if r["attempt"] == 1]
    assert len(first) == 972
    groups = {
        "first_bd": [r for r in first if r["suite"] == "bd"],
        "first_cg": [r for r in first if r["suite"] == "cg"],
        "retry_plain": [r for r in records if r["attempt"] > 1 and r["mode"] == "plain"],
        "retry_reflection": [r for r in records if r["attempt"] > 1 and r["mode"] == "reflection"],
    }
    counts = {name: {"n": len(rows), **{key: sum(r["gate"] == key for r in rows) for key, _, _ in CATEGORIES}} for name, rows in groups.items()}
    judge = {name: dict(Counter(r["judge_label"] for r in rows if r["judge_label"])) for name, rows in groups.items()}
    # Candidates that only the source-level method judge rejects: feasible output within 1% of the reference.
    judge_only = {name: sum(r["judge_fail"] and r["numeric_pass"] for r in rows) for name, rows in groups.items()}
    judge_only_by_model = dict(Counter(r["model"] for r in records if r["judge_fail"] and r["numeric_pass"]))
    # Does the candidate that follows a method rejection pass the method audit? The shared first candidate
    # precedes candidate two of both policies, so it is followed once per policy here.
    trajectories = {}
    for r in records:
        trajectories.setdefault((r["model"], r["instance"], r["mode"]), {})[r["attempt"]] = r
    for (model, instance, mode), t in trajectories.items():
        if mode == "reflection" and 2 in t:
            t[1] = trajectories[(model, instance, "plain")][1]
    after_rejection = {mode: Counter("pass" if not t[a + 1]["judge_fail"] else "fail" for (_, _, m), t in trajectories.items() if m == mode for a in t if t[a]["judge_fail"] and a + 1 in t) for mode in ("plain", "reflection")}
    judge_by_model = {}
    for r in records:
        if r["judge_label"]:
            judge_by_model.setdefault(r["model"], Counter())[r["judge_label"]] += 1
    first_matrix = {m: {p: sum(r["gate"] == "success" for r in first if r["model"] == m and f"{r['suite']}/{r['problem']}" == p) for p in FAMILIES} for m in data["primary_models"]}
    return records, {"physical_candidates": len(records), "groups": counts, "judge_labels": judge, "judge_labels_total": dict(Counter(by_key.values())), "judge_only_rejections": judge_only, "next_candidate_after_rejection": {m: dict(c) for m, c in after_rejection.items()}, "judge_only_rejections_by_model": judge_only_by_model,
                     "judge_labels_by_model": {m: dict(c) for m, c in judge_by_model.items()}, "first_attempt_family_successes": first_matrix}


def anatomy_figure(summary, output, chunk, name):
    """One stacked-bar panel per figure, so each can sit next to the paragraph that discusses it."""
    fig, ax = plt.subplots(figsize=(8, 2.25))
    fig.subplots_adjust(left=.235, right=.965, top=.70, bottom=.25)
    for y, (key, title) in enumerate(chunk):
        c = summary["groups"][key]
        left = 0.0
        for cat, _, color in CATEGORIES:
            share = c[cat] / c["n"]
            ax.barh(y, share, left=left, height=.62, color=color, edgecolor="white", linewidth=1.2)
            if c[cat] and share >= .027:
                ax.text(left + share / 2, y, str(c[cat]), ha="center", va="center", fontsize=8.6, color="#0b0b0b" if cat in ("success", "gap") else "white")
            left += share
    ax.set_yticks(range(len(chunk)), [t.format(n=summary["groups"][k]["n"]) for k, t in chunk], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of attempts (labels are counts)")
    handles = [matplotlib.patches.Patch(color=color, label=label_) for _, label_, color in CATEGORIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995), ncol=3, frameon=False, fontsize=9.2, handlelength=1.4, columnspacing=1.4)
    save(fig, output, name)


FIRST_ROWS = (("first_bd", "Benders (540)"), ("first_cg", "Column gen. (432)"))
RETRY_ROWS = (("retry_plain", "Without reflection ({n})"), ("retry_reflection", "With reflection ({n})"))


def first_family_figure(summary, data, output):
    models = data["primary_models"]
    values = np.array([[summary["first_attempt_family_successes"][m][p] for p in FAMILIES] for m in models])
    cmap = LinearSegmentedColormap.from_list("frozen_YlGnBu", STYLE["palette"]["heatmap"]["colors"])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.subplots_adjust(left=.19, right=.91, top=.83, bottom=.06)
    ax.imshow(values, vmin=0, vmax=9, cmap=cmap, aspect="auto")
    for (i, j), v in np.ndenumerate(values):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v >= 6 else "black", fontsize=9.5)
    totals = values.sum(axis=0)
    ax.set_xticks(range(9), [p.split("/")[1].upper() for p in FAMILIES], fontsize=9.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(models)), [label(m) for m in models], fontsize=9.5)
    for j, t in enumerate(totals):
        ax.text(j, len(models) - .35, f"{t}", ha="center", va="top", fontsize=8.6, color="#52514e")
    ax.text(-.6, len(models) - .35, "Total /108", ha="right", va="top", fontsize=8.6, color="#52514e")
    rows = values.sum(axis=1)
    for i, t in enumerate(rows):
        ax.text(8.65, i, f"{t}/81", ha="left", va="center", fontsize=8.6, color="#52514e")
    ax.axvline(4.5, color="white", lw=2.5)
    ax.text(2, -1.55, "Benders decomposition (5 families)", ha="center", fontsize=10, weight="bold")
    ax.text(6.5, -1.55, "Column generation (4 families)", ha="center", fontsize=10, weight="bold")
    ax.set_ylim(len(models) + .45, -.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    save(fig, output, "family_first")


def table(summary, output):
    names = {"certificate_overclaim": "Reports \\code{optimal} without a proof of optimality", "inactive_decomposition": "Decomposition loop never runs or cannot affect the master", "solution_bypass": "Heuristic or enumeration fallback produces the solution"}
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "Judge finding & First BD & First CG & Retries P & Retries R \\\\", "\\midrule"]
    for key in JUDGE_LABELS:
        vals = [summary["judge_labels"][g].get(key, 0) for g in ("first_bd", "first_cg", "retry_plain", "retry_reflection")]
        lines.append(f"{names[key]} & " + " & ".join(map(str, vals)) + " \\\\")
    totals = [sum(summary["judge_labels"][g].values()) for g in ("first_bd", "first_cg", "retry_plain", "retry_reflection")]
    lines += ["\\midrule", "All method rejections & " + " & ".join(map(str, totals)) + " \\\\", "\\bottomrule", "\\end{tabular}"]
    (output / "tables" / "judge_findings.tex").write_text("\n".join(lines) + "\n")


def status_sensitivity(records, data, output):
    """Q@k if unproven-optimality rejections with a feasible, within-1% output had passed.

    A trajectory stops at its first success, so under the relaxed rule it would stop at the first attempt that
    either succeeded or was rejected only for an unproven optimal status; attempts up to that point are unchanged.
    """
    traj = {}
    for r in records:
        traj.setdefault((r["model"], r["instance"], r["mode"]), {})[r["attempt"]] = r
    def attempts(model, instance, mode):
        t = dict(traj.get((model, instance, mode), {}))
        t[1] = traj[(model, instance, "plain")][1]  # shared first attempt
        return t
    def solved(t, k, relaxed):
        return any(a <= k and (r["gate"] == "success" or (relaxed and r["judge_label"] == "certificate_overclaim" and r["numeric_pass"])) for a, r in t.items())
    budgets = TABLE_BUDGETS
    rows = {}
    for m in data["primary_models"]:
        instances = sorted({i for (mm, i, _) in traj if mm == m})
        assert len(instances) == 81
        rows[m] = [[sum(solved(attempts(m, i, mode), k, relaxed) for i in instances) for k, mode in budgets] for relaxed in (False, True)]
    lines = ["\\begin{tabular}{lccccccc}", "\\toprule", "Configuration & Q@1 & Q@2 P & Q@2 R & Q@3 P & Q@3 R & Q@10 P & Q@10 R \\\\", "\\midrule"]
    def cell(a, b):
        return f"{a}" if a == b else f"{a} $\\to$ \\textbf{{{b}}}"
    for m, (strict, relaxed) in rows.items():
        lines.append(f"{label(m)} & " + " & ".join(cell(a, b) for a, b in zip(strict, relaxed)) + " \\\\")
    tot = [[sum(rows[m][x][j] for m in rows) for j in range(len(budgets))] for x in (0, 1)]
    lines += ["\\midrule", "All configurations (of 972) & " + " & ".join(cell(a, b) for a, b in zip(*tot)) + " \\\\", "\\bottomrule", "\\end{tabular}"]
    (output / "tables" / "status_sensitivity.tex").write_text("\n".join(lines) + "\n")
    return {label(m): {"strict": v[0], "relaxed": v[1]} for m, v in rows.items()} | {"total": {"strict": tot[0], "relaxed": tot[1]}}


def lines_table(path, columns, header, rows, totals=None):
    """Write a booktabs table; `totals` rows follow a midrule."""
    text = ["\\begin{tabular}{" + columns + "}", "\\toprule", header + " \\\\", "\\midrule"]
    text += [" & ".join(map(str, row)) + " \\\\" for row in rows]
    if totals:
        text += ["\\midrule"] + [" & ".join(map(str, row)) + " \\\\" for row in totals]
    text += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(text) + "\n")


def trajectories(records):
    """Attempts per (model, instance, policy); the shared first attempt also starts the reflection policy."""
    traj = {}
    for r in records:
        traj.setdefault((r["model"], r["instance"], r["mode"]), {})[r["attempt"]] = r
    for (model, instance, mode), t in traj.items():
        if mode == "reflection" and 2 in t:
            t[1] = traj[(model, instance, "plain")][1]
    return traj


def after_rejection_table(records, out):
    """What the attempt after a method rejection does, per policy."""
    counts = {mode: Counter() for mode in ("plain", "reflection")}
    for (_, _, mode), t in trajectories(records).items():
        for a in t:
            if t[a]["judge_fail"] and a + 1 in t:
                counts[mode]["followed"] += 1
                counts[mode]["passes_audit"] += not t[a + 1]["judge_fail"]
                counts[mode]["success"] += t[a + 1]["gate"] == "success"
    rows = [[title, counts["plain"][key], counts["reflection"][key]] for key, title in (("followed", "Method rejections followed by another attempt"), ("passes_audit", "Next attempt passes the method audit"), ("success", "Next attempt meets all three criteria"))]
    lines_table(out / "tables" / "after_rejection.tex", "lrr", " & P & R", rows)
    return {mode: dict(c) for mode, c in counts.items()}


def retry_failures_table(records, out):
    """First failed check of each unsuccessful retry, per policy; objective gaps split by family."""
    failed = {mode: [r for r in records if r["attempt"] > 1 and r["mode"] == mode and r["gate"] != "success"] for mode in ("plain", "reflection")}
    def count(mode, gate, problems=None, exclude=()):
        return sum(r["gate"] == gate and (problems is None or r["problem"] in problems) and r["problem"] not in exclude for r in failed[mode])
    rows = [["Feasible, gap $>$ 1\\%", *(count(m, "gap") for m in failed)],
            ["\\quad FCTP", *(count(m, "gap", {"fctp"}) for m in failed)],
            ["\\quad TSSO", *(count(m, "gap", {"tsso"}) for m in failed)],
            ["\\quad Other families", *(count(m, "gap", exclude={"fctp", "tsso"}) for m in failed)]]
    rows += [[title, *(count(m, gate) for m in failed)] for gate, title in (("method", "Method rejected"), ("no_objective", "Ran, no objective"), ("infeasible", "Infeasible or inconsistent"), ("no_run", "Not delivered or crashed"))]
    lines_table(out / "tables" / "retry_failures.tex", "lrr", "First check failed & P & R", rows, [["All unsuccessful retries", *(len(failed[m]) for m in failed)]])
    return {mode: len(rows) for mode, rows in failed.items()}


def success_status_table(records, data, out):
    """The status reported by each successful physical attempt (optimal or feasible), by method."""
    paths = {(p["model"], p["instance"], mode): Path(p["input_root"]) / p["policies"][mode]["path"] for p in data["pairs"] for mode in ("plain", "reflection")}
    counts = Counter()
    for r in records:
        if r["gate"] == "success":
            output = json.loads((paths[(r["model"], r["instance"], r["mode"])] / "attempts" / f"attempt_{r['attempt']:02d}" / "output.json").read_text())
            counts[(output["status"], r["suite"])] += 1
    assert set(status for status, _ in counts) <= {"optimal", "feasible"}
    rows = [[f"\\code{{{status}}}", counts[(status, "bd")], counts[(status, "cg")], counts[(status, "bd")] + counts[(status, "cg")]] for status in ("optimal", "feasible")]
    totals = [sum(row[i] for row in rows) for i in (1, 2, 3)]
    lines_table(out / "tables" / "success_status.tex", "lrrr", "Reported status & Benders & Column generation & All", rows, [["All successful attempts", *(f"{n:,}".replace(",", "{,}") for n in totals)]])
    return {f"{status}/{suite}": n for (status, suite), n in counts.items()}


NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")


def unproven_claims_table(records, data, out):
    """Unproven optimality claims (judge label certificate_overclaim) per configuration and attempt type."""
    counts = {m: [0, 0, 0] for m in data["primary_models"]}
    for r in records:
        if r["judge_label"] == "certificate_overclaim":
            counts[r["model"]][0 if r["attempt"] == 1 else 1 if r["mode"] == "plain" else 2] += 1
    order = list(data["primary_models"])
    claimed = sorted((m for m in order if sum(counts[m])), key=lambda m: (-sum(counts[m]), order.index(m)))
    rows = [[label(m), *counts[m], sum(counts[m])] for m in claimed]
    rest = len(order) - len(claimed)
    if rest:
        rows.append([f"Other {NUMBER_WORDS[rest]} configurations", 0, 0, 0, 0])
    totals = [sum(counts[m][i] for m in order) for i in range(3)]
    lines_table(out / "tables" / "unproven_claims.tex", "lrrrr", "Configuration & First attempt & Retries P & Retries R & Total", rows, [["All configurations", *totals, sum(totals)]])
    return {m: counts[m] for m in claimed}


def build(out):
    """Write the failure figures and tables into out/figures and out/tables; return the counts."""
    data = json.loads((out / "combined.json").read_text())
    records, summary = analyse(data)
    # Cross-check against the audited first-attempt gate counts used in the paper tables.
    first = summary["groups"]["first_bd"], summary["groups"]["first_cg"]
    assert sum(g["success"] for g in first) == 853
    assert sum(g["success"] + g["gap"] for g in first) == 876
    assert sum(g["n"] - g["no_run"] for g in first) == 962
    expected = sum(p["policies"][m]["actual_attempts"] for p in data["pairs"] if p["model"] in data["primary_models"] for m in ("plain", "reflection")) - 972
    assert len(records) == expected
    summary["sources"] = {"combined.json": digest(out / "combined.json"), "failure_labels.json": digest(LABELS_PATH), "failures.py": digest(__file__)}
    setup()
    anatomy_figure(summary, out, FIRST_ROWS, "failure_first")
    anatomy_figure(summary, out, RETRY_ROWS, "failure_retry")
    first_family_figure(summary, data, out)
    table(summary, out)
    summary["status_sensitivity"] = status_sensitivity(records, data, out)
    summary["after_rejection"] = after_rejection_table(records, out)
    summary["unsuccessful_retries"] = retry_failures_table(records, out)
    summary["success_status"] = success_status_table(records, data, out)
    summary["unproven_claims"] = unproven_claims_table(records, data, out)
    (out / "qa" / "failure_anatomy.json").write_text(json.dumps(summary, indent=1) + "\n")
    return records, summary
