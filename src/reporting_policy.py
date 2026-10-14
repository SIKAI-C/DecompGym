"""Current reporting eligibility; raw experiments and execution stay unchanged."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


MODEL_EXCLUSIONS = {
    "gpt-5.6-sol": {
        "model": "gpt-5.6-sol",
        "name": "GPT-5.6 Sol",
        "effective_date": "2026-10-08",
        "status": "historical_cohorts_excluded",
        "reason": "Historical Docker/native Sol results remain excluded. The explicitly authorized fresh Docker cohort is eligible under the October 13 decision.",
        "docker_observation": {"passed": 52, "evaluated": 62, "planned": 81, "complete": False},
        "native_observation": {"passed": 79, "evaluated": 81, "planned": 81, "complete": True},
        "reinstatement": "No investigation of the historical discrepancy is required; inclusion is limited to the explicitly authorized fresh cohort.",
    }
}


AUTHORIZED_FRESH_PROTOCOL_IDS = {"e6a01d3d1cf966de676323a7d9f9e5b344552ee678281b46feb92823b4a65c73", "ac9ff895ad775edc40ca785a82d2055add0188e2cd20653ea9e916273711ca1a"}
AUTHORIZED_FRESH_PLAN = "fixed81_qualitative_20261012"
AUTHORIZED_FRESH_PLANS = {AUTHORIZED_FRESH_PLAN, "fixed81_strict_decomposition_all_models_20261013"}
AUTHORIZED_INVENTORY_SHA256 = "8e8cda5f0a4d37eea74fb85f75530e91c7a94a8ec4632dd31b6e3eea88f7cb34"


def cohort_eligibility(manifest: dict | None) -> dict:
    """Use frozen scientific identity, never a suggestive directory name."""
    manifest = manifest or {}
    plan = manifest.get("experiment_plan", {})
    authorized = (manifest.get("protocol_id") in AUTHORIZED_FRESH_PROTOCOL_IDS
                  and hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != "protocol_id"}, sort_keys=True).encode()).hexdigest() == manifest.get("protocol_id")
                  and plan.get("name") in AUTHORIZED_FRESH_PLANS
                  and plan.get("dataset_inventory_sha256") == AUTHORIZED_INVENTORY_SHA256
                  and plan.get("reuse_generated_context") is False
                  and manifest.get("agent_execution") == "docker"
                  and manifest.get("solver_execution") == "native-macos-seatbelt"
                  and manifest.get("historical_reference_in_fresh_totals") is False
                  and plan.get("reporting", {}).get("mix_legacy_results_into_fresh_totals") is False)
    return {"id": manifest.get("protocol_id"), "plan": plan.get("name"),
            "authorized_fresh_cohort": authorized, "historical_sol_eligible": False,
            "decision_date": "2026-10-13", "manifest": manifest}


def model_exclusion(model, cohort: dict | None = None):
    normalized = re.sub(r"[\s_]+", "-", str(model or "").strip().lower())
    if normalized == "gpt-5.6-sol" and cohort_eligibility(cohort)["authorized_fresh_cohort"]:
        return None
    return MODEL_EXCLUSIONS.get(normalized)


def require_reportable_model(model, cohort: dict | None = None):
    exclusion = model_exclusion(model, cohort)
    if exclusion:
        raise ValueError(f"{exclusion['name']} is excluded from reported results and figures since {exclusion['effective_date']}: {exclusion['reason']} Raw results are retained for investigation.")


def artifact_exclusion(path: Path):
    """Recognize native model folders and legacy summaries/configurations."""
    path = path.resolve()
    cohort = next((json.loads((directory / "campaign.json").read_text()) for directory in (path, *path.parents) if (directory / "campaign.json").is_file()), None)
    for component in path.parts:
        if exclusion := model_exclusion(component, cohort):
            return exclusion
    for directory in (path, *path.parents):
        for name in ("instance_summary.json", "identity.json", "run_config.json"):
            metadata_path = directory / name
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text())
                if exclusion := model_exclusion(metadata.get("model"), cohort):
                    return exclusion
    return None


def require_reportable_artifact(path: Path):
    if exclusion := artifact_exclusion(path):
        require_reportable_model(exclusion["model"])


def require_reportable_report(report):
    cohort = report.get("cohort_manifest")
    require_reportable_model(report.get("configuration", {}).get("model"), cohort)
    for key in ("instances", "pairs", "calls", "paired_curves"):
        for instance in report.get(key, []):
            require_reportable_model(instance.get("model"), cohort)
