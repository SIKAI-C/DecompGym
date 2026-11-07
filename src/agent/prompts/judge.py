"""Prompt for the method judge that reviews whether a solver really uses the requested decomposition.

Extension points (no config uses them; the defaults give the paper's text byte for byte):
- `sources`: the delivered files under review, solver.py first (a multi-file delivery adds its C++ sources);
- `target` (AuditTarget): what the decomposition must produce and which code the judge traces to it; PAPER_AUDIT_TARGET
  is the reported solution written to output.json, a large-scale target may name published incumbents and certified
  bounds (a bound-only attempt then needs no primal solution);
- `extensions`: AuditExtension paragraphs inserted before the closing format instruction, for example the review of
  certificate-producing code paths. An extension that states a method rule (`author=True`) must also be in the author's
  Method requirements (the paired runner checks this); judge_policy names target and extensions after the reviewed
  algorithm_judge id.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.agent.prompts.method import DECOMPOSITION_POLICY, METHOD_NAMES, method_requirements


ALGORITHM_JUDGE = "strict-decomposition-source-review-v2"  # src.agent.prompts.snapshot.PROMPT_PROTOCOL["algorithm_judge"]
SOLVER_SOURCES = ("solver.py",)


@dataclass(frozen=True)
class AuditTarget:
    id: str
    produced: str  # what the decomposition must substantively produce
    traced: str  # the code the judge traces that result to


PAPER_AUDIT_TARGET = AuditTarget("reported-solution-output-json-v1", "the reported solution", "the code that writes output.json")


@dataclass(frozen=True)
class AuditExtension:
    id: str
    paragraphs: tuple[str, ...]
    author: bool = False  # a method rule: also stated in the author's and the reflection's Method requirements


def judge_policy(extensions=(), target=PAPER_AUDIT_TARGET) -> str:
    """The judge id of a review with this target and these extensions (the reviewed id for the paper's)."""
    return "+".join((ALGORITHM_JUDGE, *([target.id] if target != PAPER_AUDIT_TARGET else []), *(extension.id for extension in extensions)))


def algorithm_review_prompt(suite: str, contract: str = DECOMPOSITION_POLICY, *, sources=SOLVER_SOURCES, extensions=(), target=PAPER_AUDIT_TARGET) -> str:
    """The judge applies the same contract id as the campaign's instance prompts (prompt_protocol.decomposition_contract)."""
    sources = tuple(sources)
    if not sources or sources[0] != "solver.py" or len(set(sources)) != len(sources):
        raise ValueError("The method judge reviews solver.py first and each source once")
    single = sources == SOLVER_SOURCES
    subject = "solver.py" if single else "source files (" + ", ".join(sources) + "; the entry point is solver.py)"
    verbs = ("it implements", "uses") if single else ("they implement", "use")
    paragraphs = [
        f"Inspect the supplied {subject} and determine whether {verbs[0]} {METHOD_NAMES[suite]} and substantively {verbs[1]} that method to produce {target.produced}. Apply the following method contract, which is also supplied to the solver author:",
        method_requirements(suite, contract),
        f"Trace the entry point, relevant reachable call paths, information passed back to the master, and {target.traced}. Method names, comments, unused decomposition functions, or loops that do not feed subproblem information back into the master are insufficient. For an initialization or final solve, identify which decisions are fixed, which variables or columns are optimized, and where its result is used; an LP or MIP call by itself is not a violation.",
        "Return FAIL if a reachable solution-producing path uses a prohibited separate full-original-model solve, even if decomposition also runs. For conditional fallbacks, cite the triggering conditions and role from source without assuming they ran. Unused helpers or provably unreachable branches alone are not a reason for FAIL. A genuine decomposition can converge or stop without adding a cut or column in a particular instance, and may return its initialized feasible incumbent.",
        "This is an algorithm-use assessment, not a full mathematical proof of every cut, bound, or pricing formulation. Execution, feasibility, output consistency, and objective quality are checked separately. An incorrect numerical result alone is not a reason for algorithm FAIL; determine whether the requested decomposition is substantively implemented and used. Do not require a particular cut construction, initialization heuristic, pricing algorithm, or number of iterations.",
    ]
    for extension in extensions:
        if not extension.paragraphs or not all(isinstance(text, str) and text.strip() for text in extension.paragraphs):
            raise ValueError(f"Audit extension {extension.id!r} has no text")
        paragraphs += list(extension.paragraphs)
    reviewed = "solver.py" if single else "the source files"
    paragraphs.append(f"Do not execute or modify {reviewed}. Do not claim to have observed runtime behavior when reviewing source only. Write alg_verifier.md with exactly a first line VERDICT: PASS or VERDICT: FAIL, followed by NOTES: and concise code evidence supporting the verdict.")
    return "\n\n".join(paragraphs)


def parse_verdict(notes: str):
    """The judge's first line decides: "pass", "fail", or None for a malformed verdict (case and spacing ignored)."""
    notes = notes.strip()
    return {"VERDICT: PASS": "pass", "VERDICT: FAIL": "fail"}.get(notes.splitlines()[0].strip().upper() if notes else "")
