"""Compile the benchmark overview figure (overview.tex, TikZ) into figures/benchmark_overview.pdf."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile


SOURCE = Path(__file__).with_name("overview.tex")


def build(out: Path) -> Path:
    if shutil.which("pdflatex") is None:
        raise RuntimeError("The overview figure needs a LaTeX installation with pdflatex, TikZ, helvet, inconsolata and fontawesome5")
    target = out / "figures" / "benchmark_overview.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="overview-") as directory:
        shutil.copyfile(SOURCE, Path(directory) / "overview.tex")
        # A fixed date keeps the PDF reproducible.
        environment = os.environ | {"SOURCE_DATE_EPOCH": "0", "FORCE_SOURCE_DATE": "1"}
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "overview.tex"], cwd=directory, env=environment, check=True, capture_output=True)
        shutil.copyfile(Path(directory) / "overview.pdf", target)
    if shutil.which("gs"):
        # A PNG for the README; the paper uses the PDF.
        subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", "-r200", f"-sOutputFile={target.with_suffix('.png')}", str(target)], check=True)
    return target
