# tests/test_benchmark_metrics.py
"""
Tests for src/layer3/evaluation/benchmark.py.

These tests exercise the metrics math and dataset loading directly (pure
Python + a fake predict_fn) — they don't need real embeddings, so they
also serve as a fast, always-green regression guard for the harness logic
itself, independent of model accuracy.
"""
from __future__ import annotations

import pytest

from src.layer3.benchmark import (
    CheckMetrics,
    evaluate_binary_check,
    load_claim_baselines,
)


class TestCheckMetricsMath:
    def test_perfect_classifier(self):
        m = CheckMetrics(check_name="dummy", tp=10, fp=0, tn=10, fn=0)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.accuracy == 1.0
        assert m.false_positive_rate == 0.0

    def test_all_wrong_classifier(self):
        m = CheckMetrics(check_name="dummy", tp=0, fp=10, tn=0, fn=10)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 is None  # 2pr/(p+r) undefined when p=r=0
        assert m.accuracy == 0.0
        assert m.false_positive_rate == 1.0

    def test_no_positive_predictions_precision_is_none(self):
        m = CheckMetrics(check_name="dummy", tp=0, fp=0, tn=10, fn=5)
        assert m.precision is None  # tp+fp == 0
        assert m.recall == 0.0

    def test_empty_metrics_all_none(self):
        m = CheckMetrics(check_name="dummy")
        assert m.n == 0
        assert m.precision is None
        assert m.recall is None
        assert m.f1 is None
        assert m.accuracy is None

    def test_as_dict_is_json_serializable_shape(self):
        import json
        m = CheckMetrics(check_name="dummy", tp=3, fp=1, tn=4, fn=2)
        d = m.as_dict()
        json.dumps(d)  # must not raise
        assert d["check_name"] == "dummy"
        assert d["n"] == 10


class TestEvaluateBinaryCheck:
    def test_confusion_matrix_correct(self):
        examples = [
            {"id": "a", "expected_label": "unsafe"},  # predict True -> TP
            {"id": "b", "expected_label": "safe"},     # predict True -> FP
            {"id": "c", "expected_label": "safe"},      # predict False -> TN
            {"id": "d", "expected_label": "unsafe"},    # predict False -> FN
        ]
        predictions = {"a": True, "b": True, "c": False, "d": False}

        def predict_fn(ex):
            return predictions[ex["id"]]

        m = evaluate_binary_check("dummy_check", examples, predict_fn)
        assert (m.tp, m.fp, m.tn, m.fn) == (1, 1, 1, 1)
        assert m.n == 4

    def test_latencies_recorded_per_example(self):
        examples = [{"id": "a", "expected_label": "safe"}]

        def predict_fn(ex):
            return False

        m = evaluate_binary_check("dummy_check", examples, predict_fn)
        assert len(m.latencies_ms) == 1
        assert m.latencies_ms[0] >= 0

    def test_per_example_records_correctness_flag(self):
        examples = [
            {"id": "a", "expected_label": "unsafe"},
            {"id": "b", "expected_label": "safe"},
        ]

        def predict_fn(ex):
            return ex["id"] == "a"  # gets both right

        m = evaluate_binary_check("dummy_check", examples, predict_fn)
        assert all(row["correct"] for row in m.per_example)


class TestClaimBaselinesDataset:
    """
    Regression guard for the n>=2 -> n>=20 safe-baseline expansion the
    review specifically asked for.
    """

    def test_dataset_loads(self):
        examples = load_claim_baselines()
        assert len(examples) > 0

    def test_safe_baseline_count_meets_target(self):
        examples = load_claim_baselines()
        safe = [e for e in examples if e["expected_label"] == "safe"]
        assert len(safe) >= 20, (
            f"expected >=20 safe claim-verification baselines per the "
            f"review's false-positive-rate target, got {len(safe)}"
        )

    def test_dataset_has_both_labels(self):
        examples = load_claim_baselines()
        labels = {e["expected_label"] for e in examples}
        assert labels == {"safe", "unsafe"}

    def test_every_example_has_required_fields(self):
        required = {"id", "query", "output", "expected_label"}
        for ex in load_claim_baselines():
            missing = required - ex.keys()
            assert not missing, f"example {ex.get('id')} missing fields {missing}"

    def test_ids_are_unique(self):
        examples = load_claim_baselines()
        ids = [e["id"] for e in examples]
        assert len(ids) == len(set(ids))
