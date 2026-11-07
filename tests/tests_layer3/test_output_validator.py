"""
Manual test runner for output_validator.py
(No pytest required)
"""

from src.layer3.output_validator import (
    run_layer3,
    ValidatorConfig,
    Layer3Output,
)

# -------------------------------------------------
# Simple assertion helper
# -------------------------------------------------
def check(name, condition):
    if condition:
        print(f"[PASS] {name}")
        return True
    else:
        print(f"[FAIL] {name}")
        return False


# -------------------------------------------------
# TESTS
# -------------------------------------------------

def test_happy_path():
    config = ValidatorConfig()

    result = run_layer3(
        raw_output="Paris is the capital of France.",
        original_query="What is the capital of France?",
        config=config,
    )

    ok1 = check("happy_path type", isinstance(result, Layer3Output))
    ok2 = check("happy_path raw_output", bool(result.raw_output))
    ok3 = check("happy_path pii false", result.pii_detected is False)

    return ok1 and ok2 and ok3


def test_pii_detection_email():
    result = run_layer3(
        raw_output="Contact me at test@example.com",
        original_query="Give contact info",
    )

    ok1 = check("PII email detected", result.pii_detected is True)
    ok2 = check("PII email in failed_checks", "privacy" in result.failed_checks)

    return ok1 and ok2


def test_pii_detection_phone():
    result = run_layer3(
        raw_output="Call me at +213555123456",
        original_query="How to contact?",
    )

    ok1 = check("PII phone detected", result.pii_detected is True)
    ok2 = check("PII phone in failed_checks", "privacy" in result.failed_checks)

    return ok1 and ok2


def test_toxicity_trigger():
    toxic_text = "I hate you, you are worthless and stupid"

    result = run_layer3(
        raw_output=toxic_text,
        original_query="say something",
        config=ValidatorConfig(toxicity_threshold=0.0),
    )

    return check("toxicity triggered", "toxicity" in result.failed_checks)


def test_policy_sql_injection():
    result = run_layer3(
        raw_output="DROP TABLE users;",
        original_query="show database info",
    )

    return check("SQL injection blocked", "policy" in result.failed_checks)


def test_policy_command_injection():
    result = run_layer3(
        raw_output="rm -rf /",
        original_query="system cleanup",
    )

    return check("command injection blocked", "policy" in result.failed_checks)


def test_semantic_failure():
    result = run_layer3(
        raw_output="Bananas are yellow and monkeys like trees.",
        original_query="Explain quantum physics",
        config=ValidatorConfig(min_semantic_score=0.9),
    )

    return check("semantic failure", "semantic" in result.failed_checks)


def test_relevance_failure():
    result = run_layer3(
        raw_output="This is about cooking pasta.",
        original_query="Explain neural networks",
        config=ValidatorConfig(min_relevance_score=0.9),
    )

    return check("relevance failure", "relevance" in result.failed_checks)


def test_claim_verification():
    result = run_layer3(
        raw_output="The capital of France is Berlin.",
        original_query="What is capital of France?",
    )

    ok1 = check("claim_verification exists", result.claim_verification is not None)
    ok2 = check("claim_verification has passed", hasattr(result.claim_verification, "passed"))

    return ok1 and ok2


def test_empty_input():
    result = run_layer3("", "")

    ok1 = check("empty input raw_output", result.raw_output == "")
    ok2 = check("empty input failed_checks list", isinstance(result.failed_checks, list))

    return ok1 and ok2


def test_multiple_failures():
    result = run_layer3(
        raw_output="DROP TABLE users; email test@test.com I hate you",
        original_query="Explain databases",
        config=ValidatorConfig(
            toxicity_threshold=0.0,
            min_relevance_score=0.9,
            min_semantic_score=0.9,
        ),
    )

    return check("multiple failures >= 2", len(result.failed_checks) >= 2)


def test_custom_config():
    config = ValidatorConfig(
        toxicity_threshold=0.1,
        min_relevance_score=0.8,
    )

    result = run_layer3(
        raw_output="Random unrelated text",
        original_query="Explain AI",
        config=config,
    )

    return check("custom config works", isinstance(result.failed_checks, list))


# -------------------------------------------------
# RUNNER
# -------------------------------------------------

def main():
    tests = [
        test_happy_path,
        test_pii_detection_email,
        test_pii_detection_phone,
        test_toxicity_trigger,
        test_policy_sql_injection,
        test_policy_command_injection,
        test_semantic_failure,
        test_relevance_failure,
        test_claim_verification,
        test_empty_input,
        test_multiple_failures,
        test_custom_config,
    ]

    passed = 0
    failed = 0

    print("\n==============================")
    print(" OUTPUT VALIDATOR TEST RUNNER")
    print("==============================\n")

    for t in tests:
        try:
            print(f"\nRunning {t.__name__} ...")
            result = t()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] {t.__name__} crashed -> {e}")

    print("\n==============================")
    print(" SUMMARY")
    print("==============================")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total : {len(tests)}")


if __name__ == "__main__":
    main()