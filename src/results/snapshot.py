"""Audit the two paper campaigns read-only and save the combined dataset the paper figures and tables use."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.results.campaign_report import build_campaign_report, export_campaign_report
from src.results.combine import combine_reports
from src.results.costs import combine_costs, summaries
from src.results.policy import PAPER_MODELS, PAPER_RUNS
from src.results.time_analysis import build_time_analysis, write_analysis
from src.results.token_analysis import build_token_analysis


# Each paper campaign's audit is exported under this label inside the output folder.
LABELS = {"fixed81_strict_decomposition_20261013_r1": "source_original", "fixed81_strict_claude_extension_20261014_r1": "source_extension"}


def check_cached(report, root):
    """A saved audit is reused only while the campaign manifest and its frozen dataset are unchanged."""
    manifest = json.loads((root / "campaign.json").read_text())
    if report["cohort_manifest"] != manifest:
        raise ValueError(f"The saved audit of {root.name} no longer matches its campaign.json; rerun with --refresh")
    for case in manifest["instances"]:
        directory = root / "dataset" / case["suite"] / case["problem"] / case["instance"]
        for name, expected in case["sha256"].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"{directory / name} changed since the saved audit; rerun with --refresh")


def build_snapshot(experiments: Path, output: Path, *, refresh: bool):
    """With refresh, re-audit the raw campaigns; otherwise reuse the saved audits in `output`."""
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, protocol in PAPER_RUNS.items():
        root = (experiments / name).resolve()
        label = LABELS[name]
        if refresh:
            print("Auditing", name, flush=True)
            report = build_campaign_report(root, models=PAPER_MODELS)
            export_campaign_report(report, output / label)
        else:
            report = json.loads((output / label / "evaluation.json").read_text())
            check_cached(report, root)
            # The saved audit records where the campaign was; read it from where it is now.
            report["input_root"] = str(root)
        if report["cohort_manifest"]["protocol_id"] != protocol:
            raise ValueError(f"{name} is not the frozen paper campaign")
        reports.append(report)
    data = combine_reports(reports, [r["cohort_manifest"] for r in reports])
    time = build_time_analysis(reports)
    tokens = build_token_analysis(reports)
    resources = combine_costs(time, tokens)
    write_analysis(time, output / "time")
    (output / "tokens").mkdir(exist_ok=True)
    for name, content in (("combined.json", data), ("tokens/token_analysis.json", tokens), ("resources.json", {"rows": resources, "summary": summaries(resources), "methods": tokens["methods"]})):
        (output / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    summary = {"models": len(data["primary_models"]), "complete_pairs": sum(c["completed_pairs"] for c in data["coverage"]),
               "read_errors": sum(len(r["snapshot"]["read_errors"]) for r in reports), "audit_errors": sum(len(r["audit_errors"]) for r in reports)}
    print(json.dumps(summary), flush=True)
    return summary
