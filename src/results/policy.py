"""Which models and experiment runs the paper reports. Raw experiment records are never changed."""
from __future__ import annotations


# The twelve configurations of the paper, in the order of its tables.
PAPER_MODELS = (
    "gpt-6-astra", "gpt-6-sol", "gpt-6.1-sol", "gpt-6-luna", "gpt-5.6-luna",
    "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "gemini-3.8-flash-medium",
)

# The two campaign roots behind the paper (under outputs/experiments/) and their frozen protocol IDs.
# The second root added four Claude models a day later under the same instances, prompts, scoring and limits.
PAPER_RUNS = {
    "fixed81_strict_decomposition_20261013_r1": "ac9ff895ad775edc40ca785a82d2055add0188e2cd20653ea9e916273711ca1a",
    "fixed81_strict_claude_extension_20261014_r1": "a7653c5686cddb4fbd2b2073ccb324b9dced7f8a2afc0ae72bdf7b083776ebee",
}
PAPER_INVENTORY_SHA256 = "8e8cda5f0a4d37eea74fb85f75530e91c7a94a8ec4632dd31b6e3eea88f7cb34"


def run_identity(manifest: dict | None) -> dict:
    """Describe a campaign by its frozen protocol, never by a suggestive directory name."""
    manifest = manifest or {}
    plan = manifest.get("experiment_plan", {})
    return {"id": manifest.get("protocol_id"), "plan": plan.get("name"), "paper_run": manifest.get("protocol_id") in PAPER_RUNS.values()}


def reported(model, models=None) -> bool:
    """Report every model of a campaign unless a selection (such as PAPER_MODELS) is given."""
    return models is None or model in models
