"""Behavioral evaluation harness for the AI Resume Analyzer.

The default command is a LIVE EVALUATION through the production ``run_agent``
path and local Ollama. ``--deterministic-only`` runs only cases that production
``run_agent`` can complete without model generation and labels the rest
LIVE-ONLY.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent_loop import run_agent


SYNTHETIC_RESUME = """Alex Rivera

EDUCATION
Bachelor of Science in Information Systems
Example University

EXPERIENCE
Example Corporation
Software Engineering Intern
Developed internal web applications.
Maintained database records.
Provided technical support.
Created internal documentation.

PROJECT EXPERIENCE
CAPSTONE PROJECT
Example Delivery Platform
Built the platform using CapstoneFramework and Python.
Implemented user authentication and order tracking.

OTHER PROJECT
Example Inventory Tool
Built the inventory tool using OtherProjectFramework.

SKILLS
Programming: Python, JavaScript
Frameworks: GlobalSkillFramework
Databases: PostgreSQL
"""

REFUSAL_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\bno evidence\b",
    r"\b(?:resume|cv)\b.{0,50}\b(?:does not|doesn't|doesn't appear to)\b.{0,30}\b(?:state|mention|list|provide|specify|indicate|say|show|contain)\b",
    r"\b(?:does not|doesn't|cannot|can't|could not|couldn't|unable to)\b.{0,50}\b(?:confirm|find|verify|provide|identify|determine)\b",
    r"\bnot (?:stated|mentioned|listed|provided|specified|available|included|found)\b",
    r"\bno information\b",
    r"\bnot enough information\b",
    r"\bnot in (?:the )?(?:resume|cv)\b",
    r"\bdoes not say\b|\bdoesn't say\b",
))

UNCERTAINTY_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\bno evidence\b",
    r"\b(?:does not|doesn't) establish\b",
    r"\bnot established\b",
    r"\b(?:cannot|can't|could not|couldn't|unable to) (?:verify|confirm|establish)\b",
    r"\bnot (?:stated|specified|shown|documented|linked|associated|established)\b",
    r"\bnot (?:used|applied|implemented|part of|associated|linked)\b",
    r"\b(?:project|capstone)\b.{0,60}\bnot (?:used|applied|implemented|associated|linked)\b",
    r"\b(?:mentions?|listed|appears?)\b.{0,80}\bseparately\b",
    r"\bseparately\b.{0,80}\b(?:mentioned|listed|appear)\b",
    r"\bonly (?:appears|listed|mentioned) in (?:the )?skills\b",
    r"\bdoes not say\b|\bdoesn't say\b",
))

CONTRAST_SPLIT = re.compile(
    r"(?<=[.!?;])\s+|,\s+|\b(?:but|however|although|whereas)\b",
    re.IGNORECASE,
)
NEGATION_CUE = re.compile(
    r"\b(?:not|never|no|doesn't|does not|isn't|is not|wasn't|was not|without|cannot|can't)\b",
    re.IGNORECASE,
)
USE_VERB = r"(?:use|uses|used|using)"


def _normalize(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _contains_any(text: str, phrases: list[str]) -> bool:
    normalized = _normalize(text)
    return any(_normalize(phrase) in normalized for phrase in phrases)


def _clauses(text: str) -> list[str]:
    return [part.strip() for part in CONTRAST_SPLIT.split(text) if part.strip()]


def _has_refusal(text: str) -> bool:
    return any(pattern.search(text) for pattern in REFUSAL_PATTERNS)


def _has_uncertainty(text: str) -> bool:
    return any(pattern.search(text) for pattern in UNCERTAINTY_PATTERNS)


def _has_positive_project_attribution(
    text: str,
    technology: str,
    project: str | list[str],
) -> bool:
    technology_pattern = re.escape(technology)
    projects = project if isinstance(project, list) else [project]
    for clause in _clauses(text):
        if NEGATION_CUE.search(clause):
            continue
        for project_name in projects:
            project_pattern = re.escape(project_name)
            relation_patterns = (
                re.compile(
                    rf"\b{USE_VERB}\b.{{0,100}}\b{technology_pattern}\b.{{0,100}}\b(?:in|within|for|on|as part of)\b.{{0,80}}\b{project_pattern}\b",
                    re.IGNORECASE,
                ),
                re.compile(
                    rf"\b{technology_pattern}\b.{{0,60}}\b(?:was|is|were|has been|had been)\s+used\b.{{0,80}}\b(?:in|within|for|on)\b.{{0,60}}\b{project_pattern}\b",
                    re.IGNORECASE,
                ),
                re.compile(
                    rf"\b{project_pattern}\b.{{0,80}}\b{USE_VERB}\b.{{0,80}}\b{technology_pattern}\b",
                    re.IGNORECASE,
                ),
            )
            if any(pattern.search(clause) for pattern in relation_patterns):
                return True
    return False


def _has_positive_employer_claim(text: str, entity: str) -> bool:
    entity_pattern = re.escape(entity)
    relation_patterns = (
        re.compile(
            rf"\b(?:work|works|worked|employ|employs|employed)\b.{{0,50}}\b(?:at|for|by)\b.{{0,50}}\b{entity_pattern}\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b{entity_pattern}\b.{{0,50}}\b(?:is|was|were)\b.{{0,20}}\b(?:an?\s+)?(?:employer|company)\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:employer|company)\b.{{0,40}}\b(?:is|was|were)\b.{{0,20}}\b{entity_pattern}\b",
            re.IGNORECASE,
        ),
    )
    for clause in _clauses(text):
        if NEGATION_CUE.search(clause):
            continue
        if any(pattern.search(clause) for pattern in relation_patterns):
            return True
    return False


def _has_positive_favorite_claim(text: str) -> bool:
    patterns = (
        re.compile(r"\bfavorite\b.{0,80}\b(?:is|was|:|equals)\b", re.IGNORECASE),
        re.compile(r"\b\w[\w+#.-]*\b.{0,50}\b(?:is|was)\b.{0,30}\bfavorite\b", re.IGNORECASE),
    )
    return any(
        not NEGATION_CUE.search(clause)
        and any(pattern.search(clause) for pattern in patterns)
        for clause in _clauses(text)
    )


def _has_positive_salary_claim(text: str) -> bool:
    patterns = (
        re.compile(r"\b(?:salary|pay|compensation|income)\b.{0,40}(?:\$\s*)?\d", re.IGNORECASE),
        re.compile(r"(?:\$\s*\d[\d,.]*|\b\d[\d,.]*\s*(?:per\s+year|annually|monthly))", re.IGNORECASE),
    )
    return any(
        not NEGATION_CUE.search(clause)
        and any(pattern.search(clause) for pattern in patterns)
        for clause in _clauses(text)
    )


def _load_cases() -> list[dict]:
    dataset_path = Path(__file__).with_name("questions.json")
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("questions.json must contain a non-empty 'cases' list")

    seen_ids = set()
    allowed_behaviors = {"answer", "refuse", "relationship_not_established"}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every evaluation case must be a JSON object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise ValueError(f"Evaluation case ids must be unique non-empty strings: {case_id!r}")
        seen_ids.add(case_id)
        if case.get("expected_behavior") not in allowed_behaviors:
            raise ValueError(f"Unsupported expected_behavior for {case_id!r}")
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"Evaluation case {case_id!r} has an empty question")
    return cases


def _case_failures(case: dict, outcome: dict) -> tuple[list[str], list[str], list[str]]:
    """Return behavior, unsupported-claim, and relationship failures separately."""
    answer = str(outcome.get("answer", ""))
    expected = case["expected_behavior"]
    assertions = case.get("assertions", {})
    behavior_failures = []
    unsupported_failures = []
    relationship_failures = []

    if outcome.get("status") != "final":
        behavior_failures.append(f"agent finished with status {outcome.get('status')!r}")
    if not answer.strip():
        behavior_failures.append("no answer text was produced")

    if expected == "answer":
        for phrase in assertions.get("must_contain_all", []):
            if not _contains_any(answer, [phrase]):
                behavior_failures.append(f"missing required evidence {phrase!r}")
        for field in ("must_contain_any", "must_contain_any_context"):
            phrases = assertions.get(field, [])
            if phrases and not _contains_any(answer, phrases):
                behavior_failures.append(f"missing all accepted {field.replace('_', ' ')} markers")

    elif expected == "refuse":
        if not _has_refusal(answer):
            behavior_failures.append("answer did not express an evidence-based refusal")

    elif expected == "relationship_not_established":
        if not _has_uncertainty(answer):
            behavior_failures.append("answer did not communicate that the relationship is unverified")
        relationship_check = case.get("relationship_check", {})
        if relationship_check.get("kind") == "technology_project":
            if _has_positive_project_attribution(
                answer,
                relationship_check["technology"],
                relationship_check["project"],
            ):
                relationship_failures.append("answer positively claimed the unsupported project relationship")
        elif relationship_check.get("kind") == "entity_employer":
            if _has_positive_employer_claim(answer, relationship_check["entity"]):
                relationship_failures.append("answer positively claimed the project was an employer")

    for relation in assertions.get("must_not_attribute_to_project", []):
        if _has_positive_project_attribution(answer, relation["technology"], relation["project"]):
            unsupported_failures.append(
                f"answer attributed {relation['technology']!r} to {relation['project']!r} without evidence"
            )
            relationship_failures.append(
                f"answer positively claimed {relation['technology']!r} was used in {relation['project']!r}"
            )

    for entity in assertions.get("must_not_claim_employer", []):
        if _has_positive_employer_claim(answer, entity):
            unsupported_failures.append(f"answer claimed {entity!r} as an employer")
            relationship_failures.append(f"answer positively claimed employment at {entity!r}")

    if assertions.get("must_not_claim_favorite") and _has_positive_favorite_claim(answer):
        unsupported_failures.append("answer invented a favorite programming language")
    if assertions.get("must_not_claim_salary") and _has_positive_salary_claim(answer):
        unsupported_failures.append("answer invented a salary")

    return behavior_failures, unsupported_failures, relationship_failures


def _is_timeout(outcome: dict) -> bool:
    if outcome.get("status") not in {"error", "exception"}:
        return False
    details = str(outcome.get("answer", ""))
    return bool(re.search(r"\b(?:timeout|timed out|request timeout)\b", details, re.IGNORECASE))


def _actual_answer(outcome: dict) -> str:
    if outcome.get("status") in {"error", "exception"}:
        return "<no answer produced>"
    answer = str(outcome.get("answer", "")).strip()
    return answer if answer else "<no answer produced>"


def _print_case(case: dict, status: str, outcome: dict | None, failures: list[str]) -> None:
    if outcome is None:
        answer = "<no answer produced>"
        grounding = "n/a"
        iterations = "n/a"
        tools = "n/a"
    else:
        answer = _actual_answer(outcome)
        grounding = (
            "Yes" if outcome.get("grounding_verified") else "No"
        ) if "grounding_verified" in outcome else "n/a"
        iterations = str(outcome.get("iterations", "n/a"))
        tools = str(outcome.get("tools_executed", "n/a"))

    print(f"[{status}] {case['id']}")
    print(f"  ID: {case['id']}")
    print(f"  Category: {case['category']}")
    print(f"  Question: {case['question']}")
    print(f"  Status: {status}")
    print(f"  Expected behavior: {case['expected_behavior']}")
    print(f"  Actual answer: {answer}")
    print(f"  Grounding verified: {grounding}")
    print(f"  Iterations: {iterations}")
    print(f"  Tool calls: {tools}")
    print(f"  Failure reason: {'; '.join(failures) if failures else 'n/a'}")


def _is_unsupported_metric_case(case: dict) -> bool:
    assertions = case.get("assertions", {})
    return (
        case["expected_behavior"] == "refuse"
        or bool(assertions.get("must_not_attribute_to_project"))
        or bool(assertions.get("must_not_claim_employer"))
        or bool(assertions.get("must_not_claim_favorite"))
        or bool(assertions.get("must_not_claim_salary"))
    )


def _is_relationship_metric_case(case: dict) -> bool:
    return (
        case["expected_behavior"] == "relationship_not_established"
        or bool(case.get("relationship_check"))
        or bool(case.get("assertions", {}).get("must_not_attribute_to_project"))
    )


def run_evaluation(deterministic_only: bool = False) -> int:
    cases = _load_cases()
    live_only = 0
    passed = 0
    failed = 0
    timeouts = 0
    completed_outcomes = []
    category_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"passed": 0, "failed": 0, "timeouts": 0, "live_only": 0}
    )

    if deterministic_only:
        print("DETERMINISTIC-ONLY EVALUATION")
    else:
        print("LIVE EVALUATION: local Ollama")
    print(f"Synthetic resume: embedded fictional fixture ({len(SYNTHETIC_RESUME)} chars)")
    print(f"Cases loaded: {len(cases)}\n")

    for case in cases:
        category = case["category"]
        if deterministic_only and case.get("requires_live_llm", True):
            live_only += 1
            category_counts[category]["live_only"] += 1
            _print_case(case, "LIVE-ONLY", None, ["requires model decision/generation"])
            continue

        captured_output = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured_output):
                outcome = run_agent(case["question"], SYNTHETIC_RESUME)
        except Exception as error:
            outcome = {
                "status": "exception",
                "answer": f"{type(error).__name__}: {error}",
                "iterations": 0,
                "tools_executed": 0,
            }

        if _is_timeout(outcome):
            timeouts += 1
            category_counts[category]["timeouts"] += 1
            _print_case(
                case,
                "TIMEOUT",
                outcome,
                ["Ollama/request timeout; answer correctness was not evaluated"],
            )
            continue

        behavior_failures, unsupported_failures, relationship_failures = _case_failures(case, outcome)
        failures = behavior_failures + unsupported_failures + relationship_failures
        completed_outcomes.append({
            "case": case,
            "outcome": outcome,
            "behavior_ok": not behavior_failures,
            "unsupported_ok": not unsupported_failures,
            "relationship_ok": not relationship_failures,
            "failures": failures,
        })

        if failures:
            failed += 1
            category_counts[category]["failed"] += 1
            status = "FAIL"
        else:
            passed += 1
            category_counts[category]["passed"] += 1
            status = "PASS"
        _print_case(case, status, outcome, failures)

    evaluated = len(completed_outcomes)
    behavior_correct = sum(item["behavior_ok"] for item in completed_outcomes)
    grounding_known = [
        item for item in completed_outcomes
        if "grounding_verified" in item["outcome"]
    ]
    grounding_true = sum(bool(item["outcome"].get("grounding_verified")) for item in grounding_known)

    unsupported_cases = [
        item for item in completed_outcomes
        if _is_unsupported_metric_case(item["case"])
    ]
    unsupported_safe = sum(item["unsupported_ok"] for item in unsupported_cases)

    relationship_cases = [
        item for item in completed_outcomes
        if _is_relationship_metric_case(item["case"])
    ]
    relationship_safe = sum(item["relationship_ok"] for item in relationship_cases)

    if deterministic_only:
        print("\n# Deterministic Results")
        print(f"Evaluated: {evaluated}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Live-only: {live_only}")
        print(f"Timeouts: {timeouts}")
    else:
        model_dependent = sum(case.get("requires_live_llm", True) for case in cases)
        print("\n# Live Evaluation Results")
        print(f"Live cases: {model_dependent}")
        print(f"Cases evaluated: {evaluated}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Timeouts: {timeouts}")

    print(f"Answer correctness: {behavior_correct}/{evaluated}")
    print(f"Grounding verified: {grounding_true}/{len(grounding_known)}")
    print(f"Unsupported-claim protection: {unsupported_safe}/{len(unsupported_cases)}")
    print(f"Relationship safety: {relationship_safe}/{len(relationship_cases)}")

    print("\nBy category:")
    for category in sorted(category_counts):
        counts = category_counts[category]
        print(
            f"{category}: {counts['passed']} passed, {counts['failed']} failed, "
            f"{counts['timeouts']} timeouts, {counts['live_only']} live-only"
        )

    return 1 if failed or timeouts else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="run only cases that production run_agent can complete without Ollama",
    )
    arguments = parser.parse_args()
    try:
        return run_evaluation(deterministic_only=arguments.deterministic_only)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Evaluation could not start: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
