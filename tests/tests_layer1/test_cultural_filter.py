"""
Comprehensive test suite for output_validator.py
=================================================

Tests:
- PASS cases
- PII detection
- Toxicity blocking
- Policy/security violations
- Semantic failure
- Relevance failure
- Claim verification integration
- Multiple failures
- Empty inputs
- Config overrides
- Audit-friendly structured output

Run:
    python test_output_validator.py
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------
# Add project root
# ---------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
from pprint import pprint
import json
import uuid

from src.layer3.output_validator import (
    run_layer3,
    ValidatorConfig,
)

# ---------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------

TEST_CASES = [

    # ================================================================
    # ALLOW / SAFE CASES
    # ================================================================

    {
        "id": "ALLOW-001",
        "text": "Paris is the capital of France.",
        "query": "What is the capital of France?",
        "expected_failed": [],
        "description": "Clean factual answer",
    },

    {
        "id": "ALLOW-002",
        "text": "Machine learning is a subset of AI that learns from data.",
        "query": "Explain machine learning",
        "expected_failed": [],
        "description": "Educational safe response",
    },

    # ================================================================
    # PII DETECTION
    # ================================================================

    {
        "id": "PII-001",
        "text": "My email is test@example.com",
        "query": "contact info",
        "expected_failed": ["privacy"],
        "description": "Email detection",
    },

    {
        "id": "PII-002",
        "text": "Call me at +213555123456",
        "query": "phone number",
        "expected_failed": ["privacy"],
        "description": "Phone detection",
    },

    # ================================================================
    # TOXICITY
    # ================================================================

    {
        "id": "TOX-001",
        "text": "I hate you, you are useless and stupid",
        "query": "say something",
        "config": ValidatorConfig(toxicity_threshold=0.0),
        "expected_failed": ["toxicity"],
        "description": "Forced toxicity failure",
    },

    # ================================================================
    # POLICY / SECURITY
    # ================================================================

    {
        "id": "POL-001",
        "text": "DROP TABLE users;",
        "query": "database help",
        "expected_failed": ["policy"],
        "description": "SQL injection pattern",
    },

    {
        "id": "POL-002",
        "text": "rm -rf /",
        "query": "system cleanup",
        "expected_failed": ["policy"],
        "description": "Dangerous system command",
    },

    {
        "id": "POL-003",
        "text": "<script>alert(1)</script>",
        "query": "web content",
        "expected_failed": ["policy"],
        "description": "XSS detection",
    },

    # ================================================================
    # SEMANTIC FAILURE
    # ================================================================

    {
        "id": "SEM-001",
        "text": "Bananas grow in tropical regions.",
        "query": "Explain quantum physics",
        "config": ValidatorConfig(min_semantic_score=0.9),
        "expected_failed": ["semantic"],
        "description": "Off-topic response",
    },

    # ================================================================
    # RELEVANCE FAILURE
    # ================================================================

    {
        "id": "REL-001",
        "text": "Cooking pasta step by step.",
        "query": "Explain neural networks",
        "config": ValidatorConfig(min_relevance_score=0.9),
        "expected_failed": ["relevance"],
        "description": "Irrelevant answer",
    },

    # ================================================================
    # MULTIPLE FAILURES
    # ================================================================

    {
        "id": "MULTI-001",
        "text": "DROP TABLE users; email test@test.com I hate you",
        "query": "Explain databases",
        "config": ValidatorConfig(
            toxicity_threshold=0.0,
            min_relevance_score=0.9,
            min_semantic_score=0.9,
        ),
        "expected_failed": ["privacy", "toxicity", "policy"],
        "description": "Multiple violations",
    },

    # ================================================================
    # EMPTY INPUT
    # ================================================================

    {
        "id": "EMPTY-001",
        "text": "",
        "query": "",
        "expected_failed": [],
        "description": "Empty safe input",
    },

    # ================================================================
    # CLAIM VERIFICATION EDGE
    # ================================================================

    {
        "id": "CLAIM-001",
        "text": "The capital of France is Berlin.",
        "query": "What is capital of France?",
        "expected_claim_pass_exists": True,
        "description": "Wrong factual claim",
    },
]

# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def run_case(case):
    config = case.get("config", ValidatorConfig())

    result = run_layer3(
        raw_output=case["text"],
        original_query=case["query"],
        config=config,
    )

    failed = set(result.failed_checks)

    expected = set(case.get("expected_failed", []))

    passed = failed == expected

    output = {
        "test_id": case["id"],
        "description": case["description"],
        "input": case["text"],
        "query": case["query"],
        "expected_failed": list(expected),
        "actual_failed": list(failed),
        "passed": passed,
        "toxicity": result.toxicity_score,
        "pii": result.pii_detected,
        "claim_passed": (
            result.claim_verification.passed
            if result.claim_verification
            else None
        ),
        "trace_id": f"trace_{uuid.uuid4().hex[:8]}",
    }

    return output


# ---------------------------------------------------------------------
# Extra diagnostic test
# ---------------------------------------------------------------------

def test_config_override():
    print("\n" + "=" * 70)
    print("CONFIG OVERRIDE TEST")
    print("=" * 70)

    result = run_layer3(
        raw_output="Random irrelevant text",
        original_query="Explain AI",
        config=ValidatorConfig(min_relevance_score=0.99),
    )

    pprint(result.__dict__)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("=" * 70)
    print("OUTPUT VALIDATOR TEST SUITE")
    print("=" * 70)

    passed = 0
    failed = 0
    results = []

    for case in TEST_CASES:

        result = run_case(case)
        results.append(result)

        if result["passed"]:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(
            f"[{status}] "
            f"{result['test_id']} | "
            f"Expected={result['expected_failed']} | "
            f"Actual={result['actual_failed']}"
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Passed : {passed}")
    print(f"Failed : {failed}")
    print(f"Total  : {len(TEST_CASES)}")

    # Save audit log
    output_path = Path("audit_output_validator.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved audit log → {output_path.resolve()}")

    # Extra diagnostic
    test_config_override()


if __name__ == "__main__":
    main()