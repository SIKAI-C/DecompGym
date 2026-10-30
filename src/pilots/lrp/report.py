"""Summarize all pilot runs, retaining timeouts and unclosed certificates."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def report(folder, rows):
    folder = Path(folder)
    fields = [
        "instance",
        "method",
        "depots",
        "customers",
        "routes",
        "route_pool",
        "status",
        "termination_reason",
        "upper_bound",
        "lower_bound",
        "relative_gap",
        "method_seconds",
        "process_wall_seconds",
        "generation_seconds",
        "peak_rss_mb",
    ]
    with (folder / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    methods = sorted({r["method"] for r in rows})
    lines = [
        "# Local LRP pilot results",
        "",
        "These are new SciPy/HiGHS implementations on supplied route pools. They are not Gurobi paper timings, agent results, or unrestricted routing optima.",
        "",
        "Each method receives the same input and per-instance budget. Generated instances provide all methods the same feasible witness (recorded externally; no solver warm start). One solver thread is configured. Execution order rotates by instance.",
        "",
        "Method time includes input validation, matrix construction, and solves. Process time additionally includes interpreter startup/imports and serialization. Instance generation is reported separately. No time-to-target speedup is inferred from end-of-run timings.",
        "",
        "| Method | Runs | Proved optimal within tolerance | Verified feasible outputs | Errors |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        values = [r for r in rows if r["method"] == method]
        lines.append(
            f"| {method} | {len(values)} | {sum(r['status'] == 'optimal' for r in values)} | {sum(bool((r.get('independent_verification') or {}).get('ok')) for r in values)} | {sum(r['status'] == 'error' for r in values)} |"
        )
    lines += [
        "",
        "A feasible output may still have a large unresolved optimality gap. The LP-cut variant can stop because no new LP cuts are available; its integer incumbent is not automatically optimal.",
        "",
        "| Instance | Method | Objective | Lower bound | Gap | Seconds | Termination |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]

    def number(value):
        return "-" if value is None else f"{value:.4f}"

    for row in sorted(rows, key=lambda r: (r["instance"], r["method"])):
        gap = "-" if row["relative_gap"] is None else f"{100 * row['relative_gap']:.2f}%"
        lines.append(
            f"| {row['instance']} | {row['method']} | {number(row['upper_bound'])} | {number(row['lower_bound'])} | {gap} | {number(row.get('method_seconds'))} | {row['termination_reason']} |"
        )
    # Cross-method consistency can catch errors that checking one solver cannot.
    contradictions = []
    for identifier in sorted({r["instance"] for r in rows}):
        values = [r for r in rows if r["instance"] == identifier and r["status"] != "error"]
        uppers = [
            r["upper_bound"]
            for r in values
            if r["upper_bound"] is not None and (r.get("independent_verification") or {}).get("ok")
        ]
        lowers = [r["lower_bound"] for r in values if r["lower_bound"] is not None]
        if uppers and lowers and max(lowers) > min(uppers) + 1e-6 * max(1, abs(min(uppers))):
            contradictions.append(identifier)
    (folder / "cross_method_checks.json").write_text(
        json.dumps({"bound_contradictions": contradictions}, indent=2) + "\n"
    )
    lines += [
        "",
        f"Cross-method bound contradictions: {len(contradictions)}.",
        "",
        "Inspect `summary.csv`, the frozen configuration/manifest, and each method directory for decisions, independent verification, cuts, LP/MIP statuses, and bound traces. Sampled pools are deliberately labeled restricted. All numerical optimality claims use a relative bound tolerance of 1e-7.",
        "",
        "The integer variant uses generic pointwise integer-recourse cuts. It is an exact finite-pattern strategy when all required solves finish, not a claim of state-of-the-art logic-based Benders performance.",
        "",
    ]
    if rows:
        plot(folder, rows)
        lines += ["![Final optimality gaps and method runtimes](summary.png)", ""]
    (folder / "report.md").write_text("\n".join(lines))
    if contradictions:
        raise RuntimeError(f"Cross-method bounds contradict verified incumbents: {contradictions}")


def plot(folder, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sizes = {r["instance"]: r["customers"] for r in rows}
    ids = sorted(sizes, key=lambda i: (sizes[i], i))
    methods = sorted({r["method"] for r in rows})
    fig, axes = plt.subplots(2, 1, figsize=(max(9, len(ids) * 0.9), 7), sharex=True)
    colors = {
        "monolithic": "#234e70",
        "lp_benders": "#d77a27",
        "integer_benders": "#287d63",
        "integer_benders_lp_first": "#8754a1",
        "integer_benders_lp_first_repair": "#ad3355",
        "integer_benders_exact_cover": "#705122",
    }
    labels = {
        "monolithic": "Full MIP",
        "lp_benders": "Benders: LP cuts",
        "integer_benders": "Benders: integer cuts",
        "integer_benders_lp_first": "Benders: LP-first",
        "integer_benders_lp_first_repair": "Benders: LP-first + repair",
        "integer_benders_exact_cover": "Benders: exact-cover recourse",
    }
    width = 0.8 / len(methods)
    for index, method in enumerate(methods):
        mapping = {r["instance"]: r for r in rows if r["method"] == method}
        x = np.arange(len(ids)) + (index - (len(methods) - 1) / 2) * width
        gaps = [
            100 * mapping[i]["relative_gap"] if mapping.get(i, {}).get("relative_gap") is not None else np.nan
            for i in ids
        ]
        times = [mapping.get(i, {}).get("method_seconds") or np.nan for i in ids]
        axes[0].bar(x, gaps, width, label=labels[method], color=colors[method])
        axes[1].bar(x, times, width, color=colors[method])
    axes[0].set_ylabel("Final certified gap (%)")
    axes[0].legend(frameon=False, ncols=min(2, len(methods)))
    axes[1].set_ylabel("Method time (seconds)")
    axes[1].set_xticks(np.arange(len(ids)), ids, rotation=35, ha="right")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("LRP local calibration • SciPy/HiGHS • one solver thread", fontweight="bold")
    fig.text(
        0.01,
        0.01,
        "Zero gap means bounds meet within tolerance. Missing bars mean unavailable values; inspect the table. No agent runs.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(folder / "summary.png", dpi=160)
    plt.close(fig)
