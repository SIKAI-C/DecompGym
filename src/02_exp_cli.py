"""Run decomposition experiments using a locally authenticated Codex CLI."""
import argparse

from src.agent.experiment_runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Codex-driven decomposition experiments.")
    parser.add_argument("--model", "-m", default=None, help="Codex model; omitted uses the CLI default.")
    parser.add_argument("--note", "-n", default=None)
    parser.add_argument("--suite", "-s", choices=["bd", "cg"], default="bd")
    parser.add_argument("--problem", "-p", nargs="*", default=None)
    parser.add_argument("--instance", "-i", default=None)
    parser.add_argument("--benchmark-root", "-br", default="data")
    parser.add_argument("--max-attempts", "-ma", type=int, default=10)
    parser.add_argument("--solver-timeout", "-sot", type=int, default=60, help="Deadline in seconds for the entire generated solver execution.")
    parser.add_argument("--solver-threads", type=int, default=1, help="Thread limit for each local solver process.")
    parser.add_argument("--solver-cooldown-seconds", type=float, default=0, help="Pause after solver executions lasting at least five seconds; outside the execution budget.")
    parser.add_argument("--agent-timeout", type=int, default=900)
    parser.add_argument("--tol", "-t", type=float, default=1e-6)
    parser.add_argument("--gap-threshold", "-gt", type=float, default=0.01)
    parser.add_argument("--reflection", "-rfl", action="store_true")
    parser.add_argument("--codex-executable", default="codex")
    return parser.parse_args()


def main() -> None:
    ExperimentRunner(**vars(parse_args())).run()


if __name__ == "__main__":
    main()
