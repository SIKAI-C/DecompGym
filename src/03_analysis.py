from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def list_instance_dirs(root: Path) -> list[Path]:
    """Find subfolders containing instance_summary.json."""
    if (root / "instance_summary.json").exists():
        return [root]
    insts: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "instance_summary.json").exists():
            insts.append(child)
    return insts


def load_attempts(instance_dir: Path) -> tuple[list[dict[str, Any]], float | None]:
    """
    Load attempts and gold objective directly from instance_summary.json.
    """
    inst_sum = instance_dir / "instance_summary.json"
    if not inst_sum.exists():
        return [], None
    try:
        data = json.loads(inst_sum.read_text())
    except Exception:
        return [], None

    attempts = data.get("attempts") or []
    gold = data.get("gold_objective")
    return attempts, gold


def infer_difficulty(instance_dir: Path) -> str | None:
    # Prefer meta.json if present
    meta_path = instance_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            diff = meta.get("difficulty")
            if isinstance(diff, str):
                return diff
        except Exception:
            pass
    # Fallback to name pattern
    name = instance_dir.name.lower()
    for level in ("easy", "medium", "hard"):
        if level in name:
            return level
    return None


def cumulative_metrics(attempts: list[dict[str, Any]], quality_tol: float, gold_obj: float | None) -> dict[str, Any]:
    n = len(attempts)
    solveE: list[int] = []
    solveV: list[int] = []
    solveQ: list[int] = []
    quality: list[float | None] = []

    best_gap: float | None = None
    e_seen = False
    v_seen = False
    q_seen = False

    for i in range(n):
        att = attempts[i]
        exec_ok = bool(att.get("exec_ok"))
        verify_ok = bool(att.get("verify_ok"))
        gap = next((att[key] for key in ("gap_to_gold", "gap", "best_gap") if att.get(key) is not None), None)
        try:
            gap_val = float(gap) if gap is not None else None
        except Exception:
            gap_val = None
        # If verifier passed but gap is missing, try to compute it or at least
        # mark the attempt as feasible so counts stay consistent with verify_ok.
        if verify_ok and gap_val is None:
            obj = att.get("objective")
            try:
                if obj is not None and gold_obj is not None:
                    gap_val = abs(float(obj) - float(gold_obj)) / max(1.0, abs(float(gold_obj)))
                else:
                    gap_val = None
            except Exception:
                gap_val = None

        if exec_ok:
            e_seen = True
        if verify_ok:
            v_seen = True
        if verify_ok and gap_val is not None and gap_val <= quality_tol:
            q_seen = True

        # Only update quality/best_gap with feasible solutions
        if verify_ok and gap_val is not None:
            best_gap = gap_val if best_gap is None else min(best_gap, gap_val)

        solveE.append(1 if e_seen else 0)
        solveV.append(1 if v_seen else 0)
        solveQ.append(1 if q_seen else 0)
        quality.append(best_gap)

    return {
        "solveE": solveE,
        "solveV": solveV,
        "solveQ": solveQ,
        "quality": quality,
        "best_gap": best_gap,
    }


def pad_series(series: list[Any], target_len: int) -> list[Any]:
    if not series:
        return [None] * target_len
    if len(series) >= target_len:
        return series[:target_len]
    last = series[-1]
    return series + [last] * (target_len - len(series))


def analyze_instance(inst_dir: Path, quality_tol: float, write_outputs: bool) -> dict[str, Any]:
    attempts, gold_obj = load_attempts(inst_dir)
    attempts = sorted(attempts, key=lambda x: int(x.get("attempt", 0)))
    metrics = cumulative_metrics(attempts, quality_tol, gold_obj)
    inst_name = inst_dir.name
    difficulty = infer_difficulty(inst_dir)
    instance_report = {
        "instance": inst_name,
        "path": str(inst_dir),
        "num_attempts": len(attempts),
        "gold_objective": gold_obj,
        "difficulty": difficulty,
        "solveE_at_i": metrics["solveE"],
        "solveV_at_i": metrics["solveV"],
        "solveQ_at_i": metrics["solveQ"],
        "quality_at_i": metrics["quality"],
        "best_gap": metrics["best_gap"],
        "quality_tol": quality_tol,
    }
    if write_outputs:
        analysis_dir = inst_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True, parents=True)
        (analysis_dir / "instance_analysis.json").write_text(json.dumps(instance_report, indent=2))
        plot_instance(instance_report, analysis_dir)
    return instance_report


def aggregate_by_difficulty(instances: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for inst in instances:
        diff = inst.get("difficulty")
        if diff in buckets:
            buckets[diff].append(inst)

    agg: dict[str, Any] = {}
    for diff, insts in buckets.items():
        if not insts:
            agg[diff] = None
            continue
        max_len = max(len(inst["solveE_at_i"]) for inst in insts)
        solveE_avg: list[float] = []
        solveV_avg: list[float] = []
        solveQ_avg: list[float] = []
        quality_avg: list[float | None] = []
        feasible_counts: list[int] = []
        total = len(insts)

        for i in range(max_len):
            e_vals: list[int] = []
            v_vals: list[int] = []
            q_vals: list[int] = []
            g_vals: list[float | None] = []
            for inst in insts:
                e_series = pad_series(inst["solveE_at_i"], max_len)
                v_series = pad_series(inst["solveV_at_i"], max_len)
                q_series = pad_series(inst["solveQ_at_i"], max_len)
                g_series = pad_series(inst["quality_at_i"], max_len)
                e_vals.append(e_series[i])
                v_vals.append(v_series[i])
                q_vals.append(q_series[i])
                g_vals.append(g_series[i])

            solveE_avg.append(sum(e_vals) / len(e_vals))
            solveV_avg.append(sum(v_vals) / len(v_vals))
            solveQ_avg.append(sum(q_vals) / len(q_vals))
            gaps = [g for g in g_vals if g is not None]
            quality_avg.append(sum(gaps) / len(gaps) if gaps else None)
            feasible_counts.append(len(gaps))  # gaps counted only when feasible & gap known

        agg[diff] = {
            "num_instances": total,
            "solveE_at_i": solveE_avg,
            "solveV_at_i": solveV_avg,
            "solveQ_at_i": solveQ_avg,
            "quality_at_i": quality_avg,
            "feasible_counts": feasible_counts,
        }
    return agg


def plot_series(ax, series: list[float], label: str, marker: str = "o"):
    xs = list(range(1, len(series) + 1))
    ax.plot(xs, series, marker=marker, label=label)


def plot_instance(inst: dict[str, Any], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_series(ax, inst["solveE_at_i"], "solveE@i")
    plot_series(ax, inst["solveV_at_i"], "solveV@i")
    plot_series(ax, inst["solveQ_at_i"], "solveQ@i")
    ax.set_xlabel("Attempt")
    ax.set_ylabel("Solved (0/1)")
    ax.set_title(f"Solve curves: {inst.get('instance', '')}")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / f"{inst.get('instance', 'instance')}_solve.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    qual = inst["quality_at_i"]
    xs = list(range(1, len(qual) + 1))
    ax.plot(xs, qual, marker="x")
    ax.set_xlabel("Attempt")
    ax.set_ylabel("Best gap")
    ax.set_title(f"Quality@i: {inst.get('instance', '')}")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"{inst.get('instance', 'instance')}_quality.png", bbox_inches="tight")
    plt.close(fig)


def plot_run_by_difficulty(agg: dict[str, Any], out_dir: Path) -> None:
    diffs = ["easy", "medium", "hard"]

    # Solve rates
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, diff in zip(axes, diffs):
        data = agg.get(diff)
        ax.set_title(f"{diff.capitalize()} solve rates")
        if not data:
            ax.text(0.5, 0.5, "No instances", ha="center", va="center")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            continue
        plot_series(ax, data["solveE_at_i"], "solveE@i")
        plot_series(ax, data["solveV_at_i"], "solveV@i")
        plot_series(ax, data["solveQ_at_i"], "solveQ@i")
        ax.set_xlabel("Attempt")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Rate")
    fig.suptitle("Aggregate solve rates by difficulty")
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_solve_by_hardness.png", bbox_inches="tight")
    plt.close(fig)

    # Quality (average gap) with feasible counts annotated
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, diff in zip(axes, diffs):
        data = agg.get(diff)
        ax.set_title(f"{diff.capitalize()} quality@i")
        if not data:
            ax.text(0.5, 0.5, "No instances", ha="center", va="center")
            ax.grid(True, alpha=0.3)
            continue
        qual = data["quality_at_i"]
        xs = list(range(1, len(qual) + 1))
        ax.plot(xs, qual, marker="x", label="avg best gap (feasible only)")
        counts = data["feasible_counts"]
        total = data["num_instances"]
        for x, q, c in zip(xs, qual, counts):
            if q is None:
                continue
            ax.annotate(
                f"{c}/{total}",
                (x, q),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                color="gray",
            )
        ax.set_xlabel("Attempt")
        ax.set_ylabel("Best gap (avg of feasibles)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Aggregate quality by difficulty (feasible-only averages)")
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_quality_by_hardness.png", bbox_inches="tight")
    plt.close(fig)


def aggregate_overall(instances: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not instances:
        return None
    max_len = max(len(inst["solveE_at_i"]) for inst in instances)
    solveE_avg: list[float] = []
    solveV_avg: list[float] = []
    solveQ_avg: list[float] = []
    quality_avg: list[float | None] = []
    feasible_counts: list[int] = []
    total = len(instances)

    for i in range(max_len):
        e_vals: list[int] = []
        v_vals: list[int] = []
        q_vals: list[int] = []
        g_vals: list[float | None] = []
        for inst in instances:
            e_series = pad_series(inst["solveE_at_i"], max_len)
            v_series = pad_series(inst["solveV_at_i"], max_len)
            q_series = pad_series(inst["solveQ_at_i"], max_len)
            g_series = pad_series(inst["quality_at_i"], max_len)
            e_vals.append(e_series[i])
            v_vals.append(v_series[i])
            q_vals.append(q_series[i])
            g_vals.append(g_series[i])

        solveE_avg.append(sum(e_vals) / len(e_vals))
        solveV_avg.append(sum(v_vals) / len(v_vals))
        solveQ_avg.append(sum(q_vals) / len(q_vals))
        gaps = [g for g in g_vals if g is not None]
        quality_avg.append(sum(gaps) / len(gaps) if gaps else None)
        feasible_counts.append(len(gaps))

    return {
        "num_instances": total,
        "solveE_at_i": solveE_avg,
        "solveV_at_i": solveV_avg,
        "solveQ_at_i": solveQ_avg,
        "quality_at_i": quality_avg,
        "feasible_counts": feasible_counts,
    }


REQUIRED_ANALYSIS_FILES = [
    "run_analysis.json",
    "aggregate_solve_by_hardness.png",
    "aggregate_quality_by_hardness.png",
    "aggregate_overall_by_hardness.png",
]


def is_analysis_complete(run_dir: Path) -> bool:
    analysis_dir = run_dir / "analysis"
    if not analysis_dir.exists():
        return False
    return all((analysis_dir / name).exists() for name in REQUIRED_ANALYSIS_FILES)


def find_run_dirs(output_root: Path) -> list[Path]:
    """Discover run folders by locating instance_summary.json files."""
    runs: set[Path] = set()
    for inst_sum in output_root.rglob("instance_summary.json"):
        inst_dir = inst_sum.parent
        run_dir = inst_dir.parent
        runs.add(run_dir)
    return sorted(runs)


def analyze_run(root: Path, quality_tol: float, force: bool = True) -> bool:
    """
    Run analysis for a given run folder.
    Returns True if analysis was performed, False otherwise.
    """
    if not root.exists():
        return False
    if not force and is_analysis_complete(root):
        return False

    inst_dirs = list_instance_dirs(root)
    if not inst_dirs:
        return False

    multiple_instances = len(inst_dirs) > 1 or root == inst_dirs[0].parent

    instance_reports: list[dict[str, Any]] = []
    for inst in inst_dirs:
        write_outputs = not multiple_instances
        instance_reports.append(analyze_instance(inst, quality_tol, write_outputs=write_outputs))

    analysis_dir = root / "analysis"
    analysis_dir.mkdir(exist_ok=True, parents=True)
    agg = aggregate_by_difficulty(instance_reports)
    agg_overall = aggregate_overall(instance_reports)
    agg_payload = {
        "run_path": str(root),
        "quality_tol": quality_tol,
        "aggregate": agg,
        "aggregate_overall": agg_overall,
        "instances": instance_reports,
    }
    (analysis_dir / "run_analysis.json").write_text(json.dumps(agg_payload, indent=2))
    plot_run_by_difficulty(agg, analysis_dir)
    plot_run_overall(agg_overall, analysis_dir)
    return True


def plot_run_overall(agg: dict[str, Any] | None, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Solve rates combined
    ax = axes[0]
    ax.set_title("Overall solve rates")
    if not agg:
        ax.text(0.5, 0.5, "No instances", ha="center", va="center")
        ax.set_ylim(0, 1.05)
    else:
        plot_series(ax, agg["solveE_at_i"], "solveE@i")
        plot_series(ax, agg["solveV_at_i"], "solveV@i")
        plot_series(ax, agg["solveQ_at_i"], "solveQ@i")
        ax.set_ylim(0, 1.05)
    ax.set_xlabel("Attempt")
    ax.set_ylabel("Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Quality combined
    ax = axes[1]
    ax.set_title("Overall quality@i")
    if not agg:
        ax.text(0.5, 0.5, "No instances", ha="center", va="center")
    else:
        qual = agg["quality_at_i"]
        xs = list(range(1, len(qual) + 1))
        ax.plot(xs, qual, marker="x", label="avg best gap (feasible only)")
        counts = agg["feasible_counts"]
        total = agg["num_instances"]
        for x, q, c in zip(xs, qual, counts):
            if q is None:
                continue
            ax.annotate(
                f"{c}/{total}",
                (x, q),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                color="gray",
            )
    ax.set_xlabel("Attempt")
    ax.set_ylabel("Best gap (avg of feasibles)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Aggregate across all hardness levels")
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_overall_by_hardness.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize run/instance outputs into analysis metrics.")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        type=str,
        help="Path to a run folder or a single instance folder. If omitted, scan outputs/experiments.",
    )
    parser.add_argument("--quality-tol", type=float, default=1e-4, help="Gap threshold for solveQ.")
    args = parser.parse_args()

    if args.path:
        run_dir = Path(args.path).resolve()
        if not analyze_run(run_dir, args.quality_tol, force=True):
            raise RuntimeError(f"No instance_summary.json found or analysis failed under {run_dir}")
    else:
        output_root = Path("outputs") / "experiments"
        if not output_root.exists():
            raise RuntimeError(f"Output root not found at {output_root}")
        run_dirs = find_run_dirs(output_root)
        if not run_dirs:
            raise RuntimeError(f"No run folders with instance_summary.json found under {output_root}")
        ran_any = False
        for run_dir in run_dirs:
            ran_any = analyze_run(run_dir, args.quality_tol, force=False) or ran_any
        if not ran_any:
            print("All analyses already complete; nothing to do.")


if __name__ == "__main__":
    main()
