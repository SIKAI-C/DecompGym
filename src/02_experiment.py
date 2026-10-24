"""Run the benchmark: agents edit solver.py inside Docker, solvers run natively in a macOS sandbox.

One-time setup on a machine:
    python src/02_experiment.py docker prepare                      # record image and sandbox evidence in outputs/runtime/
    python src/02_experiment.py docker auth --backend codex --verify  # private CLI login (codex, claude or agy)

A campaign (one frozen copy of the code, data and config):
    python src/02_experiment.py check-config --config configs/paper.json
    python src/02_experiment.py init   --root outputs/experiments/my_run --config configs/paper.json
    python src/02_experiment.py resume --root outputs/experiments/my_run     # init leaves every campaign paused
    python src/02_experiment.py watch  --root outputs/experiments/my_run --all-models
    python src/02_experiment.py status --root outputs/experiments/my_run
    python src/02_experiment.py dashboard --root outputs/experiments/my_run  # http://localhost:8765
    python src/03_analysis.py --campaign outputs/experiments/my_run          # audited report in outputs/reports/

`run` executes the pairs once; `watch` supervises `run`, restarts it after infrastructure failures and enforces the
resource guard. Both execute the campaign's frozen code in <root>/controller, never the working copy of src/.
Antigravity (agy) needs DECOMPBENCH_AGY_KEYRING_PASSPHRASE in the launch environment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from src.experiments import campaign, dashboard_server, docker_setup, plan  # noqa: E402


def frozen(command, rest):
    """Run `run` or `watch` with the campaign's frozen controller code."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=Path, required=True)
    known, others = parser.parse_known_args(rest)
    root = known.root.resolve()
    controller = root / "controller"
    # The paper campaigns froze the older layout, whose controller module is src/native_campaign.py.
    module = next((name for name, path in (("src.experiments.campaign", "src/experiments/campaign.py"), ("src.native_campaign", "src/native_campaign.py")) if (controller / path).is_file()), None)
    if module is None:
        raise SystemExit(f"{controller} is not a campaign controller; run init first")
    return subprocess.call([sys.executable, "-u", "-m", module, command, "--root", str(root), *others], cwd=controller)


COMMANDS = {
    "docker": lambda rest: docker_setup.main(rest),
    "check-config": lambda rest: plan.main(rest),
    "init": lambda rest: campaign.main(["init", *rest]),
    "resume": lambda rest: campaign.main(["resume", *rest]),
    "status": lambda rest: campaign.main(["status", *rest]),
    "run": lambda rest: frozen("run", rest),
    "watch": lambda rest: frozen("watch", rest),
    "dashboard": lambda rest: dashboard_server.main(rest),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="options of the command; see `<command> --help`")
    args = parser.parse_args(argv)
    raise SystemExit(COMMANDS[args.command](args.arguments) or 0)


if __name__ == "__main__":
    main()
