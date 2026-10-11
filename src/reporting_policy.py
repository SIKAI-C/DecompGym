"""Current reporting eligibility; raw experiments and execution stay unchanged."""
from __future__ import annotations

import json
from pathlib import Path
import re


MODEL_EXCLUSIONS = {
    "gpt-5.6-sol": {
        "model": "gpt-5.6-sol",
        "name": "GPT-5.6 Sol",
        "effective_date": "2026-10-08",
        "status": "excluded_pending_reproducibility_review",
        "reason": "Unresolved Docker/native pass-rate discrepancy; excluded from reported results and figures at the user's request.",
        "docker_observation": {"passed": 52, "evaluated": 62, "planned": 81, "complete": False},
        "native_observation": {"passed": 79, "evaluated": 81, "planned": 81, "complete": True},
        "reinstatement": "Investigate the matched-case protocol differences or perform a controlled rerun, then obtain an explicit decision to include the model again.",
    }
}


def model_exclusion(model):
    normalized = re.sub(r"[\s_]+", "-", str(model or "").strip().lower())
    return MODEL_EXCLUSIONS.get(normalized)


def require_reportable_model(model):
    exclusion = model_exclusion(model)
    if exclusion:
        raise ValueError(f"{exclusion['name']} is excluded from reported results and figures since {exclusion['effective_date']}: {exclusion['reason']} Raw results are retained for investigation.")


def artifact_exclusion(path: Path):
    """Recognize native model folders and legacy summaries/configurations."""
    path = path.resolve()
    for component in path.parts:
        if exclusion := model_exclusion(component):
            return exclusion
    for directory in (path, *path.parents):
        for name in ("instance_summary.json", "identity.json", "run_config.json"):
            metadata_path = directory / name
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text())
                if exclusion := model_exclusion(metadata.get("model")):
                    return exclusion
    return None


def require_reportable_artifact(path: Path):
    if exclusion := artifact_exclusion(path):
        require_reportable_model(exclusion["model"])


def require_reportable_report(report):
    require_reportable_model(report.get("configuration", {}).get("model"))
    for instance in report.get("instances", []):
        require_reportable_model(instance.get("model"))
