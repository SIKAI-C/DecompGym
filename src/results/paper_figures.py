"""Paper figures and tables from the combined audit snapshot (src/results/snapshot.py).

Reads saved results only; it never launches an experiment, solver or model call.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from src.results.combine import MODELS, STYLE, STYLE_PATH, model_style, points, write_csv
from src.results.policy import PAPER_MODELS

MODES = ("plain", "reflection")
TITLES = ("Without reflection", "With reflection")
FAMILIES = ("bd/ufl", "bd/fctp", "bd/km", "bd/lrp", "bd/tsso", "cg/bp", "cg/csp", "cg/gap", "cg/shift")
# ColorBrewer Dark2, three classes, colorblind rating 1. Model colors stay frozen.
SIZE_COLORS = ("#1b9e77", "#d95f02", "#7570b3")
# ColorBrewer Paired, four classes, colorblind rating 1.
STAGE_COLORS = ("#a6cee3", "#1f78b4", "#b2df8a", "#33a02c")
FIGURE_SIZES = {"paired": (8, 5), "matrix": (8, 5.4), "single": (8, 5), "stages": (8, 9)}
# Budget columns of the result tables: Q@1 is shared, then each policy at budgets 2, 3 and 10.
TABLE_BUDGETS = ((1, "plain"), (2, "plain"), (2, "reflection"), (3, "plain"), (3, "reflection"), (10, "plain"), (10, "reflection"))
PLOT_PRESENTATION = {"attempt_quality_ymin": .4, "stage_success_ymin": .5, "size_quality_ymin": .85, "reflection_gain_budgets": [2, 3], "main_table_unit": "percent (one decimal)", "family_matrix_unit": "successes / 9"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(model):
    return MODELS[model]["label"].replace(" (M)", "")


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#777777", "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, output, name):
    # Fixed MediaBox dimensions are part of the paper convention; never tight-crop.
    # Check the actual rendered labels before exporting, including rotated y labels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Axis tight bounds include only ticks actually drawn inside the view interval.
    extents = [(f"axes {i}", ax.get_tightbbox(renderer)) for i, ax in enumerate(fig.axes)]
    extents.extend(("figure text or legend", artist.get_window_extent(renderer)) for artist in [*fig.texts, *fig.legends] if artist.get_visible())
    for label, bounds in extents:
        if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(bounds.x1, bounds.y1):
            raise ValueError(f"{name}: {label} outside canvas: {bounds.bounds}")
    fig.savefig(output / "figures" / f"{name}.pdf")
    fig.savefig(output / "figures" / f"{name}.png", dpi=180)
    plt.close(fig)


def legend(fig, models, y=.995):
    handles = [Line2D([0], [0], **model_style(m), label=label(m)) for m in models]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, y), ncol=4, frameon=False, fontsize=9.3, handlelength=2.6, columnspacing=1.25)


def axes_pair(models=None, size="paired", bottom=.13):
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES[size], sharey=True)
    fig.subplots_adjust(left=.15, right=.985, bottom=bottom, top=.77 if models else .91, wspace=.16)
    if models:
        legend(fig, models)
    for ax, title in zip(axes, TITLES):
        ax.set_title(title)
        ax.grid(axis="y", color="#e6e6e6", lw=.6)
        ax.set_axisbelow(True)
    return fig, axes


def success_curves(data, output, metric="quality"):
    models = data["primary_models"]
    fig, axes = axes_pair(models)
    for ax, mode in zip(axes, MODES):
        for m in models:
            rows = points(data, m, metric)
            st = model_style(m) | {"markersize": 5.3, "linewidth": 1.3, "markeredgewidth": .9}
            ax.plot([r["budget"] for r in rows], [r[mode + "_rate"] for r in rows], **st)
        ax.set(xlim=(.8, 10.25), ylim=(PLOT_PRESENTATION["attempt_quality_ymin"], 1.025), yticks=np.arange(.4, 1.01, .1), xticks=(1, 3, 5, 7, 10), xlabel="Attempt budget")
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_ylabel("Quality success, Q@k (%)", labelpad=9)
    save(fig, output, "attempt_quality")


def gains(data, output):
    models = data["primary_models"]
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["single"], sharey=True, sharex=True)
    fig.subplots_adjust(left=.24, right=.96, bottom=.12, top=.9, wspace=.15)
    for ax, k in zip(axes, PLOT_PRESENTATION["reflection_gain_budgets"]):
        for i, m in enumerate(models):
            r = points(data, m)[k - 1]
            delta = 100 * (r["reflection_rate"] - r["plain_rate"])
            st = model_style(m)
            ax.plot([0, delta], [i, i], color=st["color"], lw=1.5)
            ax.plot(delta, i, linestyle="none", marker=st["marker"], color=st["color"], ms=6)
            ax.text(28.3, i, f"{r['reflection_successes'] - r['plain_successes']:+d}/81", fontsize=9.5, va="center", ha="right")
        ax.axvline(0, color="#999999", lw=.7)
        ax.set(title=f"Budget k = {k}", xlabel="Gain (percentage points)", xlim=(-2.5, 29), xticks=(0, 10, 20))
        ax.grid(axis="x", color="#eeeeee")
    axes[0].set_yticks(range(len(models)), [label(m) for m in models], fontsize=10)
    axes[0].invert_yaxis()
    save(fig, output, "reflection_gain")


def size_curves(data, output):
    fig, axes = axes_pair()
    for ax, mode in zip(axes, MODES):
        for size, color, marker in zip(("small", "medium", "large"), SIZE_COLORS, ("o", "s", "^")):
            rates = []
            for k in range(1, 11):
                rows = [points(data, m, dimension="size", group=size)[k-1] for m in data["primary_models"]]
                rates.append(sum(r[mode+"_successes"] for r in rows)/sum(r["paired_cases"] for r in rows))
            ax.plot(range(1, 11), rates, color=color, marker=marker, mfc="white", label=size.title(), lw=1.6)
        ax.set(xlim=(.8, 10.25), ylim=(PLOT_PRESENTATION["size_quality_ymin"], 1.008), yticks=(.85, .9, .95, 1), xticks=(1, 3, 5, 7, 10), xlabel="Attempt budget")
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.legend(loc="lower right", frameon=False)
    axes[0].set_ylabel("Quality success (%)", labelpad=9)
    save(fig, output, "size_quality")


def family_matrix(data, output):
    models = data["primary_models"]
    cmap = LinearSegmentedColormap.from_list("frozen_YlGnBu", STYLE["palette"]["heatmap"]["colors"])
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["matrix"], sharey=True)
    fig.subplots_adjust(left=.23, right=.985, top=.85, bottom=.12, wspace=.08)
    fig.text(.60, .965, "Quality successes per family: out of 9 instances", ha="center", fontsize=11)
    for ax, mode, title in zip(axes, MODES, TITLES):
        values = np.array([[points(data, m, dimension="problem", group=p)[2][mode+"_successes"] for p in FAMILIES] for m in models])
        ax.imshow(values, vmin=0, vmax=9, cmap=cmap, aspect="auto")
        for (i, j), v in np.ndenumerate(values):
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v >= 6 else "black", fontsize=10)
        ax.set_title(title + " (k = 3)")
        ax.set_xticks(range(9), [p.split("/")[1].upper() for p in FAMILIES], rotation=45, ha="right", fontsize=9.5)
        ax.set_yticks(range(len(models)), [label(m) for m in models], fontsize=10)
        ax.axvline(4.5, color="white", lw=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    save(fig, output, "family_quality")


def stages(data, output):
    models = data["primary_models"]
    fig, axes = plt.subplots(3, 2, figsize=FIGURE_SIZES["stages"], sharex=True, sharey=True)
    fig.subplots_adjust(left=.15, right=.985, top=.86, bottom=.07, hspace=.28, wspace=.16)
    legend(fig, models)
    for row, (metric, title) in enumerate((("execution", "Execution success, E (%)"), ("verification", "Verified success, V (%)"), ("quality", "Quality success, Q (%)"))):
        for col, mode in enumerate(MODES):
            ax = axes[row, col]
            for m in models:
                rs = points(data, m, metric)
                ax.plot([r["budget"] for r in rs], [r[mode+"_rate"] for r in rs], **(model_style(m) | {"markersize": 4.5, "linewidth": 1.1, "markeredgewidth": .7}))
            ax.set(xlim=(.8, 10.25), ylim=(PLOT_PRESENTATION["stage_success_ymin"], 1.025), xticks=(1, 3, 5, 7, 10), yticks=(.5, .6, .7, .8, .9, 1))
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.grid(axis="y", color="#eeeeee")
            if col == 0:
                ax.set_ylabel(title, fontsize=10, labelpad=9)
            if row == 0:
                ax.set_title(TITLES[col])
            if row == 2:
                ax.set_xlabel("Attempt budget")
    save(fig, output, "stage_success")


def resource_curve(data, resources, output, field, name, xlabel):
    rows = resources["rows"]
    models = [m for m in data["primary_models"] if field != "tokens_to_quality" or all(r["agent_accounted_tokens"] is not None for r in rows if r["model"] == m)]
    fig, axes = axes_pair(models)
    xmin, xmax = (5, 3000) if field != "tokens_to_quality" else (10000, 5000000)
    for ax, mode in zip(axes, MODES):
        for m in models:
            own = [r for r in rows if r["model"] == m and r["mode"] == mode]
            vals = sorted(r[field] for r in own if r["quality"] and r[field] is not None)
            x = [xmin, *[max(v, xmin) for v in vals], xmax]
            y = [0, *[i/81 for i in range(1, len(vals)+1)], len(vals)/81]
            ax.step(x, y, where="post", **(model_style(m) | {"markersize": 4.3, "markevery": 12, "linewidth": 1.25, "markeredgewidth": .8}))
        ax.set(xscale="log", xlim=(xmin, xmax), ylim=(0, 1.04), xlabel=xlabel)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_ylabel("Quality success (%)", labelpad=9)
    save(fig, output, name)


def cost_breakdown(data, resources, output):
    models = data["primary_models"]
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["matrix"], sharex=True, sharey=True)
    fig.subplots_adjust(left=.23, right=.98, bottom=.12, top=.86, wspace=.12)
    fields = ("generation_seconds", "reflection_seconds", "judge_seconds", "solver_seconds")
    for ax, mode, title in zip(axes, MODES, TITLES):
        left = np.zeros(len(models))
        for field, color, stage in zip(fields, STAGE_COLORS, ("Generation", "Reflection", "Judge", "Solver")):
            vals = [next(r for r in resources["summary"] if r["model"] == m and r["mode"] == mode and r["dimension"] == "overall")["mean_"+field] for m in models]
            ax.barh(range(len(models)), vals, left=left, color=color, edgecolor="white", lw=.4, label=stage)
            left += np.array(vals)
        ax.set(title=title, xlabel="Mean active stage time per instance (s)")
        ax.grid(axis="x", color="#eeeeee")
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(models)), [label(m) for m in models], fontsize=10)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=10)
    save(fig, output, "cost_breakdown")


def tex_table(path, columns, header, rows):
    text = [r"\begin{tabular}{" + columns + "}", r"\toprule", header + r" \\", r"\midrule"]
    text.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    text += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(text)+"\n")


def tables(data, resources, output):
    dest = output/"tables"
    models = data["primary_models"]
    rows=[];costs=[]
    for m in models:
        ps = points(data, m)
        rows.append([label(m), ps[0]["plain_successes"], ps[1]["plain_successes"], ps[1]["reflection_successes"], ps[2]["plain_successes"], ps[2]["reflection_successes"], ps[9]["plain_successes"], ps[9]["reflection_successes"]])
        rs = [next(r for r in resources["summary"] if r["model"] == m and r["mode"] == mode and r["dimension"] == "overall") for mode in MODES]
        toks=["--" if r["mean_agent_accounted_tokens"] is None else f"{r['mean_agent_accounted_tokens']/1000:.1f}" for r in rs]
        costs.append([label(m), *[f"{r['mean_agent_seconds']:.1f}" for r in rs], *toks])
    total = [sum(int(r[j]) for r in rows) for j in range(1, 8)]
    percent_rows = [[r[0], *[f"{100*n/81:.1f}\\%" for n in r[1:]]] for r in rows]
    percent_rows.append([r"\textbf{Overall}", *[f"{100*n/972:.1f}\\%" for n in total]])
    tex_table(dest/"main_results.tex", "lrrrrrrr", r"Configuration & Q@1 & Q@2 P & Q@2 R & Q@3 P & Q@3 R & Q@10 P & Q@10 R", percent_rows)
    tex_table(dest/"resource_costs.tex", "lrrrr", r"Configuration & Time P & Time R & Tokens P & Tokens R", costs)
    sizes=[]
    for size in ("small", "medium", "large"):
        own=[r for r in data["rows"] if r["model"] in models and r["dimension"]=="size" and r["group"]==size and r["metric"]=="quality"]
        values=[sum(r[mode+"_successes"] for r in own if r["budget"]==k) for k, mode in ((1,"plain"),(3,"plain"),(3,"reflection"),(10,"plain"),(10,"reflection"))]
        sizes.append([size.title(),*values])
    tex_table(dest/"size_results.tex", "lrrrrr", r"Size & Q@1 & Q@3 P & Q@3 R & Q@10 P & Q@10 R", sizes)
    counts=[]
    for metric, title in (("execution", "E: executable"), ("verification", "V: verified"), ("quality", "Q: quality")):
        own=[r for r in data["rows"] if r["model"] in models and r["dimension"]=="overall" and r["metric"]==metric]
        values=[sum(r[mode+"_successes"] for r in own if r["budget"]==k) for k, mode in TABLE_BUDGETS]
        counts.append([title,*values])
    tex_table(dest/"staged_counts.tex", "lrrrrrrr", r"Criterion & @1 & @2 P & @2 R & @3 P & @3 R & @10 P & @10 R", counts)
    dimensions = {
        "bd/ufl": ("Sites / customers", lambda d: (d["m"], d["n"])),
        "bd/fctp": ("Origins / destinations", lambda d: (d["m"], d["n"])),
        "bd/km": ("Sites / customers / $k$", lambda d: (d["m"], d["n"], d["k"])),
        "bd/lrp": ("Depots / customers", lambda d: (d["m"], d["n"])),
        "bd/tsso": ("Products / periods / scenarios", lambda d: (d["n_products"], d["n_periods"], d["n_scenarios"])),
        "cg/bp": ("Items / bin capacity", lambda d: (len(d["items"]), d["bin_capacity"])),
        "cg/csp": ("Types / stock length", lambda d: (len(d["item_types"]), d["stock_size"])),
        "cg/gap": ("Machines / jobs", lambda d: (len(d["machines"]), len(d["jobs"]))),
        "cg/shift": ("Time slots / max. shift", lambda d: (len(d["time_slots"]), d["shift_max_length"])),
    }
    source = Path(data["coverage"][0]["input_root"]) / "dataset"
    sizes = []
    for family, (dim, extract) in dimensions.items():
        cells=[]
        for size in ("small", "medium", "large"):
            instances=sorted((source/family).glob(f"*_{size}_seed*/data.json"))
            assert len(instances)==3
            values=[extract(json.loads(p.read_text())) for p in instances]
            cells.append(" / ".join(str(min(v)) if min(v)==max(v) else f"{min(v)}--{max(v)}" for v in zip(*values)))
        sizes.append([family.split("/")[1].upper(), dim, *cells])
    tex_table(dest/"instance_sizes.tex", "lllll", r"Family & Dimensions & Small & Medium & Large", sizes)
    gemini=[]
    for mode in MODES:
        own=[r for r in resources["rows"] if r["model"]=="gemini-3.8-flash-medium" and r["mode"]==mode]
        fields=("agent_output_tokens", "agent_reasoning_tokens", "agent_cache_read_tokens", "agent_provider_reported_total_tokens")
        gemini.append(["Plain" if mode=="plain" else "Reflection", *[f"{sum(r[k] for r in own)/len(own)/1000:.1f}" for k in fields]])
    tex_table(dest/"gemini_usage.tex", "lrrrr", r"Policy & Output & Thinking & Cache read & Provider total", gemini)
    write_csv(dest/"curves.csv", [r for r in data["rows"] if r["model"] in models])
    write_csv(dest/"resource_summary.csv", [r for r in resources["summary"] if r["model"] in models])
    write_csv(dest/"case_resources.csv", [r for r in resources["rows"] if r["model"] in models])
    first=[]
    for p in data["pairs"]:
        if p["model"] in models:
            for mode in MODES:
                first.append({k:p[k] for k in ("model","suite","problem","instance","size","seed","cohort_protocol")} | {"mode":mode,**{metric:next((k for k in range(1,11) if p["policies"][mode]["attempt_metrics"][str(k)][metric]),None) for metric in ("execution","verification","quality")}})
    write_csv(dest/"first_success.csv", first)


FIGURES = ("attempt_quality", "reflection_gain", "size_quality", "family_quality", "stage_success", "agent_time", "agent_tokens", "cost_breakdown")
TABLES = ("main_results", "resource_costs", "size_results", "staged_counts", "instance_sizes", "gemini_usage")


def check_dataset(data):
    models=data["primary_models"]
    assert models==list(PAPER_MODELS), models
    assert len(data["cases"])==81
    pairs=[p for p in data["pairs"] if p["model"] in models]
    assert len(pairs)==972 and all(p["state"]=="complete" for p in pairs)
    assert Counter(p["model"] for p in pairs)==Counter({m:81 for m in models})
    return models, pairs


def build(out):
    """Write the paper's result figures and tables into out/figures and out/tables."""
    data=json.loads((out/"combined.json").read_text());resources=json.loads((out/"resources.json").read_text())
    models, pairs = check_dataset(data)
    for folder in ("figures","tables","qa"):(out/folder).mkdir(exist_ok=True)
    setup();tables(data,resources,out)
    success_curves(data,out);gains(data,out);size_curves(data,out);family_matrix(data,out);stages(data,out)
    resource_curve(data,resources,out,"time_to_quality_agent_seconds","agent_time","Cumulative agent time (seconds)")
    resource_curve(data,resources,out,"tokens_to_quality","agent_tokens","Cumulative context + output tokens")
    cost_breakdown(data,resources,out)
    manifest={"generated_at":datetime.now(timezone.utc).isoformat(),"primary_models":models,"pairs":972,"instances":81,"protocols":sorted({p["cohort_protocol"] for p in pairs}),"style_id":STYLE["style_id"],"style_sha256":digest(STYLE_PATH),"figure_sizes_inches":FIGURE_SIZES,"presentation":PLOT_PRESENTATION,"script_sha256":digest(__file__),"sources":{str(p.relative_to(out)):digest(p) for p in (out/"combined.json",out/"resources.json",out/"source_original/evaluation.json",out/"source_extension/evaluation.json")}}
    (out/"paper_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest
