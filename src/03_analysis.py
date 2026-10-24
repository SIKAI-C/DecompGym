"""Reproduce every figure, table and prompt example of the paper from the saved experiment records.

    python src/03_analysis.py                          # audit the raw campaign records, then build everything
    python src/03_analysis.py --reuse-audit            # quick rebuild from the last saved audit (not published)
    python src/03_analysis.py --paper ../paper         # also compare with (or, with --update, copy into) a paper folder
    python src/03_analysis.py --campaign outputs/experiments/my_run   # audit any one campaign and export its report

Only the overview diagram (Figure 1) is drawn in TikZ and needs pdflatex; without it that one figure is skipped.
Ghostscript (gs), if present, adds a PNG of the diagram and lets --paper compare PDFs by their rendering.

Everything is written to outputs/reports/paper/ (with the audit snapshot and QA files); the small artifacts and a
summary page are also published to the tracked results/ folder. Nothing here calls a model or runs a solver.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
# Fixed timestamps keep the generated PDFs byte-reproducible.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

from src.results import failures, overview, paper_figures, prompt_excerpts, summary, verify  # noqa: E402
from src.results.campaign_report import build_campaign_report, export_campaign_report  # noqa: E402
from src.results.snapshot import build_snapshot  # noqa: E402


ARTIFACTS = {"figures": "*.pdf", "tables": "*.tex", "prompts": "*.txt"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rendered_difference(left, right):
    """Share of pixels that differ when both PDFs are rendered at 100 dpi with Ghostscript."""
    import matplotlib.image as image
    import numpy as np
    with tempfile.TemporaryDirectory() as directory:
        pixels = []
        for index, path in enumerate((left, right)):
            png = Path(directory) / f"{index}.png"
            subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", "-r100", f"-sOutputFile={png}", str(path)], check=True)
            pixels.append(image.imread(png))
    if pixels[0].shape != pixels[1].shape:
        return 1.0
    return float((np.abs(pixels[0] - pixels[1]).max(axis=2) > 0).mean())


def compare(out, paper):
    """Compare generated artifacts with a paper folder: text byte for byte, PDFs by their rendering."""
    report = {}
    for folder, pattern in ARTIFACTS.items():
        for path in sorted((out / folder).glob(pattern)):
            target = paper / folder / path.name
            if not target.exists():
                state = "missing in paper"
            elif path.read_bytes() == target.read_bytes():
                state = "same"
            elif path.suffix == ".pdf" and shutil.which("gs"):
                share = rendered_difference(path, target)
                # LaTeX builds can differ in a few anti-aliased edge pixels between TeX installations.
                state = "same rendering" if share == 0 else f"rendering within anti-aliasing ({share:.4%} of pixels)" if share < 1e-3 else f"rendering differs ({share:.2%} of pixels)"
            else:
                state = "differs"
            report[f"{folder}/{path.name}"] = state
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", type=Path, default=REPOSITORY / "outputs/experiments", help="folder containing the two paper campaign roots")
    parser.add_argument("--output", type=Path, help="defaults to outputs/reports/paper, or outputs/reports/<campaign> with --campaign")
    parser.add_argument("--reuse-audit", action="store_true", help="rebuild from the last saved audit instead of re-reading the raw records; for quick iteration only, never published to results/")
    parser.add_argument("--results", type=Path, default=REPOSITORY / "results", help="tracked folder for the published figures, tables and summary")
    parser.add_argument("--paper", type=Path, help="paper folder (with figures/, tables/, prompts/) to compare against")
    parser.add_argument("--update", action="store_true", help="with --paper, copy the generated artifacts into the paper folder")
    parser.add_argument("--campaign", type=Path, help="instead of the paper, audit this campaign root and export its report")
    parser.add_argument("--model", action="append", dest="models", help="with --campaign, limit the report to these models")
    args = parser.parse_args(argv)
    if args.campaign:
        root = args.campaign.resolve()
        destination = args.output or REPOSITORY / "outputs/reports" / root.name
        export_campaign_report(build_campaign_report(root, models=args.models), destination)
        print(destination / "report.md")
        return
    out = (args.output or REPOSITORY / "outputs/reports/paper").resolve()
    build_snapshot(args.experiments.resolve(), out, refresh=not args.reuse_audit)
    manifest = paper_figures.build(out)
    failures.build(out)
    checks = verify.verify(out)
    # Only the overview diagram (Figure 1) is a TikZ drawing; everything else is plain Python.
    if shutil.which("pdflatex"):
        overview.build(out)
    else:
        print("pdflatex not found: skipping the overview diagram (Figure 1); all other artifacts are built", flush=True)
    prompt_excerpts.write(args.experiments.resolve(), out)
    artifacts = {f"{folder}/{p.name}": digest(p) for folder, pattern in ARTIFACTS.items() for p in sorted((out / folder).glob(pattern))}
    provenance = manifest | {"verification": {k: checks[k] for k in ("status", "physical_policy_summaries_read", "candidate_budget_states_checked", "printed_percentage_cells_checked")},
                             "analysis_modules": {p.name: digest(p) for p in sorted((REPOSITORY / "src/results").iterdir()) if p.is_file()}, "artifacts": artifacts,
                             "generated_at": datetime.now(timezone.utc).isoformat()}
    (out / "results_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if args.reuse_audit:
        print("Built from a saved audit: results/ was not updated (run without --reuse-audit to publish)", flush=True)
    else:
        summary.publish(out, args.results.resolve())
    print(json.dumps({"output": str(out), "artifacts": len(artifacts), "verification": checks["status"]}, indent=2))
    if args.paper:
        paper = args.paper.resolve()
        if args.update:
            for name in artifacts:
                (paper / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(out / name, paper / name)
            shutil.copyfile(out / "results_provenance.json", paper / "results_provenance.json")
        comparison = compare(out, paper)
        print(json.dumps({"paper": str(paper), "identical_bytes": sum(state == "same" for state in comparison.values()), "other": {name: state for name, state in comparison.items() if state != "same"}}, indent=2))


if __name__ == "__main__":
    main()
