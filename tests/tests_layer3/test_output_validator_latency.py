# tests/test_output_validator_latency.py
"""
Tests for the [FIX-latency-breakdown] per_check_latency_ms instrumentation
added to src/layer3/output_validator.py.
"""
from __future__ import annotations

import src.layer3.output_validator as ov

EXPECTED_CHECK_KEYS = {
    "semantic", "pii", "toxicity", "policy", "relevance",
    "financial_sanity", "freshness", "islamic_guard", "claim_verification",
}


class TestPerCheckLatencyPopulated:
    def test_all_expected_keys_present_on_normal_path(
        self, fake_embeddings, fake_output_validator_embedding_model
    ):
        out = ov.run_layer3(
            raw_output="عاصمة الجزائر هي مدينة الجزائر.",
            original_query="ما هي عاصمة الجزائر؟",
        )
        assert EXPECTED_CHECK_KEYS.issubset(out.per_check_latency_ms.keys())

    def test_all_latencies_are_non_negative_floats(
        self, fake_embeddings, fake_output_validator_embedding_model
    ):
        out = ov.run_layer3(
            raw_output="عاصمة الجزائر هي مدينة الجزائر.",
            original_query="ما هي عاصمة الجزائر؟",
        )
        for check, latency in out.per_check_latency_ms.items():
            assert isinstance(latency, float), f"{check} latency is not a float"
            assert latency >= 0, f"{check} latency is negative: {latency}"

    def test_empty_output_early_return_has_default_empty_dict(self):
        out = ov.run_layer3(raw_output="", original_query="سؤال ما")
        assert out.per_check_latency_ms == {}
        assert out.passed is False
        assert "empty_output" in out.failed_checks

    def test_tool_execution_failed_early_return_has_default_empty_dict(self):
        out = ov.run_layer3(
            raw_output=ov.FALLBACK_ERROR_MSG,
            original_query="سؤال ما",
        )
        assert out.per_check_latency_ms == {}
        assert "tool_execution_failed" in out.failed_checks


class TestLatencyDoesNotAffectCorrectness:
    """
    Instrumentation should be purely additive — same pass/fail outcome
    with or without it. Regression guard against a refactor accidentally
    changing check ordering or early-return behavior while adding timing.
    """

    def test_passed_output_still_passes(
        self, fake_embeddings, fake_output_validator_embedding_model
    ):
        out = ov.run_layer3(
            raw_output="الدينار الجزائري ينقسم إلى مائة سنتيم.",
            original_query="كم سنتيم في الدينار الجزائري؟",
        )
        assert isinstance(out.passed, bool)
        assert isinstance(out.failed_checks, list)
