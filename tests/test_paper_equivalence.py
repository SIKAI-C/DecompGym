"""The cleaned code reproduces the paper campaigns: same prompts, same settings, same scores.

These checks read the two paper campaign roots and run only when they are available, e.g.
    DECOMPBENCH_EXPERIMENTS=/path/to/outputs/experiments pytest tests/test_paper_equivalence.py
No model is called and no candidate solver is executed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import pytest

from src.agent.docker_client import DockerCliClient
from src.agent.prompts.builder import PromptBuilder
from src.agent.prompts.judge import algorithm_review_prompt
from src.agent.prompts.policy import record_workspace_prompt
from src.agent.prompts.snapshot import PROMPT_PROTOCOL, judge_prompt_hashes, render_case_prompt
from src.experiments.paired_runner import AttemptResult, PairedRunner, previous_attempt_files
from src.experiments.plan import load_plan
from src.experiments.verification import OutputVerifier
from src.results.policy import PAPER_MODELS, PAPER_RUNS


REPOSITORY = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path(os.environ.get("DECOMPBENCH_EXPERIMENTS", REPOSITORY / "outputs/experiments"))
ROOTS = [EXPERIMENTS / name for name in PAPER_RUNS]
pytestmark = pytest.mark.skipif(not all((root / "campaign.json").is_file() for root in ROOTS), reason="paper campaign records are not available")


def manifest(root):
    return json.loads((root / "campaign.json").read_text())


def policies():
    """(root, specification, case, mode, policy directory) for every paper pair."""
    for root in ROOTS:
        m = manifest(root)
        for specification in m["model_queue"]:
            if specification["model"] not in PAPER_MODELS:
                continue
            for case in m["instances"]:
                for mode in ("plain", "reflection"):
                    yield root, specification, case, mode, root / "runs" / specification["model"] / mode / case["suite"] / case["problem"] / case["instance"]


def attempts(directory):
    for path in sorted((directory / "attempts").glob("attempt_*")):
        if (path / "summary.json").is_file():
            yield path, json.loads((path / "summary.json").read_text())


@pytest.mark.parametrize("root", ROOTS, ids=lambda r: r.name)
def test_problem_prompts_and_judge_prompts_match_each_campaign(root):
    m = manifest(root)
    for case in m["instances"]:
        data = json.loads((root / "dataset" / case["suite"] / case["problem"] / case["instance"] / "data.json").read_text())
        rendered = render_case_prompt(case["suite"], case["problem"], data)
        assert hashlib.sha256(rendered.encode()).hexdigest() == case["prompt_snapshot"]["effective_sha256"] == case["sha256"]["prompt.txt"], case["instance"]
        assert (REPOSITORY / "data" / case["suite"] / case["problem"] / case["instance"] / "prompt.txt").read_text() == rendered
    assert m["prompt_protocol"] == PROMPT_PROTOCOL
    assert m["algorithm_judge_prompt_sha256"] == judge_prompt_hashes()


@pytest.mark.parametrize("root", ROOTS, ids=lambda r: r.name)
def test_paper_config_matches_each_campaign(root):
    m = manifest(root)
    config = load_plan(REPOSITORY / "configs/paper.json")
    for key, value in config["execution"].items():
        assert m[key] == value, key
    assert m["prompt_policies"] == config["prompt_policies"] and m["reflection_policy"] == config["reflection_policy"]
    queue = {item["model"]: item for item in config["model_queue"]}
    for item in m["model_queue"]:
        if item["model"] in queue:
            assert item == queue[item["model"]]


def test_every_recorded_prompt_is_rebuilt_byte_for_byte():
    """Generation, method-judge and reflection prompts of all 972 pairs, rebuilt from the saved files."""
    judge = DockerCliClient("gpt-6-astra", backend="codex-cli", image_id="sha256:equivalence", runtime=None, evidence_root=None)
    scorer = PairedRunner(client=None, judge_client=None, solver_executor=None)
    checked = 0
    with tempfile.TemporaryDirectory() as scratch:
        def effective(client, target, prompt, context):
            return record_workspace_prompt(Path("/work"), target, client.compose_prompt(target, prompt, context), Path(scratch), "check", old_absolute_note=client.backend == "agy-cli")
        for root, specification, case, mode, directory in policies():
            client = DockerCliClient(specification["model"], backend=specification["backend"], image_id="sha256:equivalence", runtime=None, evidence_root=None)
            generation = PromptBuilder.build_generation_prompt((directory / "prompt.txt").read_text(), solver_timeout=60, solver_threads=1)
            for attempt_dir, summary in attempts(directory):
                index = summary["attempt"]
                if summary["status"] == "skipped_previous_solved":
                    continue
                context = [directory / "data.json"]
                if mode == "reflection" and index > 1:
                    context += previous_attempt_files(attempt_dir.parent, index)
                expected = {"agent_prompt.md": effective(client, "solver.py", generation, context)}
                if (attempt_dir / "llm_verifier_prompt.md").is_file():
                    expected["llm_verifier_prompt.md"] = effective(judge, "alg_verifier.md", algorithm_review_prompt(case["suite"]), [attempt_dir / "solver.py"])
                if mode == "reflection" and (attempt_dir / "reflection_prompt.md").is_file():
                    result = AttemptResult(**{k: v for k, v in summary.items() if k in AttemptResult.__dataclass_fields__})
                    reflection = PromptBuilder.build_reflection_prompt(generation, objective_quality_failed=scorer.objective_quality_failed(result))
                    files = [directory / "data.json"] + [attempt_dir / name for name in ("solver.py", "exec.log", "alg_verifier.md", "output.json")]
                    expected["reflection_prompt.md"] = effective(client, "debug.md", reflection, files)
                for name, text in expected.items():
                    assert (attempt_dir / name).read_text() == text, f"{attempt_dir}/{name}"
                    checked += 1
    assert checked == 4970  # every generation, judge and reflection prompt of the 972 pairs


def test_every_recorded_output_is_rescored_identically():
    """Re-verify each executed candidate with the cleaned verifier and recompute its status and gap."""
    checked = 0
    for root, specification, case, mode, directory in policies():
        canonical = root / "dataset" / case["suite"] / case["problem"] / case["instance"] / "data.json"
        for attempt_dir, summary in attempts(directory):
            if summary["status"] == "skipped_previous_solved" or not summary["exec_ok"]:
                continue
            output = json.loads((attempt_dir / "output.json").read_text())
            objective, _, _ = OutputVerifier.extract_objective(output)
            verification = OutputVerifier.verify_output(case["suite"], case["problem"], canonical, attempt_dir / "output.json", 1e-6)
            if verification.get("objective_value") is not None:
                objective = float(verification["objective_value"])
            assert bool(verification["ok"]) == summary["feasibility_ok"], attempt_dir
            verify_ok = bool(verification["ok"]) and summary["llm_verdict"] == "pass"
            assert verify_ok == summary["verify_ok"], attempt_dir
            status, gap = OutputVerifier.assess_status(exec_ok=True, output=output, verify_ok=verify_ok, objective=objective, gold_obj=summary["gold_objective"], gap_threshold=0.01, execution_timed_out=summary["metrics"].get("execution_timed_out", False))
            assert (status, gap, objective) == (summary["status"], summary["gap_to_gold"], summary["objective"]), attempt_dir
            checked += 1
    assert checked == 2363  # every executed candidate


@pytest.mark.skipif(not os.environ.get("DECOMPBENCH_PAPER"), reason="set DECOMPBENCH_PAPER to the paper folder to rebuild and compare every artifact (a few minutes)")
def test_every_paper_table_figure_and_prompt_example_is_reproduced(tmp_path):
    import importlib
    analysis = importlib.import_module("src.03_analysis")
    analysis.main(["--experiments", str(EXPERIMENTS), "--output", str(tmp_path / "paper"), "--results", str(tmp_path / "results"), "--refresh"])
    comparison = analysis.compare(tmp_path / "paper", Path(os.environ["DECOMPBENCH_PAPER"]))
    assert len(comparison) == 32  # 12 figures, 12 tables, 8 prompt examples
    assert all(state == "same" or state == "same rendering" or state.startswith("rendering within anti-aliasing") for state in comparison.values()), comparison
