# src/layer3/benchmark.py
"""
Layer 3 benchmark harness.

Addresses the peer-review gaps around Step E5 / pre-Layer 4 evaluation:

  1. Per-check precision / recall / F1 (instead of one aggregate F1 + the
     claim-verification outlier). ``evaluate_claim_verification`` reports a
     full 2x2 confusion matrix and P/R/F1 for the claim_verification check.
     The same ``evaluate_binary_check`` primitive generalizes to the other
     five checks (semantic, PII, toxicity, policy, relevance) once a
     labeled dataset in the same shape exists for them — see
     ``TODO_OTHER_CHECKS`` below for the exact format expected.

  2. Per-check latency breakdown, using the ``per_check_latency_ms`` field
     now emitted by ``run_layer3`` (see output_validator.py) instead of only
     the aggregate elapsed_ms. This lets us confirm/refute whether
     claim_verification is also the Layer 3 latency bottleneck.

  3. Threshold-sensitivity sweeps for the thresholds the paper calls out as
     fixed-and-unjustified: theta_sem (ValidatorConfig.min_semantic_score),
     EVIDENCE_MATCH_THRESHOLD, and CONTRADICTION_THRESHOLD (both in
     claim_verifier.py).

  4. The claim-verification benchmark set has been expanded from n=2 safe
     baselines to n=22 safe + n=8 unsafe (see
     claim_verification_baselines.json) so precision / false-positive rate
     for that check is no longer essentially unmeasurable.

Usage

    python -m src.layer3.evaluation.benchmark

or import the functions directly:

    from src.layer3.evaluation.benchmark import (
        evaluate_claim_verification, sweep_claim_thresholds, sweep_theta_sem,
    )
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_CLAIM_BASELINES_PATH = _HERE / "claim_verification_baselines.json"



# Generic binary-check evaluation primitive


@dataclass
class CheckMetrics:
    check_name:     str
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    latencies_ms:   List[float] = field(default_factory=list)
    per_example:    List[dict]  = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> Optional[float]:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    @property
    def recall(self) -> Optional[float]:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if not p or not r or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> Optional[float]:
        return (self.tp + self.tn) / self.n if self.n else None

    @property
    def false_positive_rate(self) -> Optional[float]:
        denom = self.fp + self.tn
        return self.fp / denom if denom else None

    def latency_summary(self) -> Dict[str, float]:
        if not self.latencies_ms:
            return {}
        s = sorted(self.latencies_ms)
        return {
            "mean_ms":   round(statistics.mean(s), 2),
            "p50_ms":    round(statistics.median(s), 2),
            "p95_ms":    round(s[min(len(s) - 1, int(0.95 * len(s)))], 2),
            "max_ms":    round(max(s), 2),
            "n":         len(s),
        }

    def as_dict(self) -> dict:
        return {
            "check_name":  self.check_name,
            "n":           self.n,
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "precision":   _r(self.precision),
            "recall":      _r(self.recall),
            "f1":          _r(self.f1),
            "accuracy":    _r(self.accuracy),
            "false_positive_rate": _r(self.false_positive_rate),
            "latency":     self.latency_summary(),
        }


def _r(x: Optional[float], nd: int = 4) -> Optional[float]:
    return round(x, nd) if x is not None else None


def evaluate_binary_check(
    check_name: str,
    examples: List[dict],
    predict_fn: Callable[[dict], bool],
    *,
    positive_label: str = "unsafe",
    label_key: str = "expected_label",
) -> CheckMetrics:
    """
    Generic evaluator: `predict_fn(example) -> bool` should return True when
    the check FIRES / FAILS the example (i.e. flags it as unsafe/violating).
    `example[label_key]` is the ground-truth label; `positive_label` denotes
    "the check should fire" (e.g. "unsafe", "toxic", "pii_present").

    This primitive is check-agnostic on purpose: to fill in the remaining
    five rows of the per-check P/R/F1 table, build a labeled dataset in the
    same {id, ..., expected_label} shape for semantic/pii/toxicity/policy/
    relevance and call this with the matching predict_fn (e.g.
    `lambda ex: not _semantic_consistency(ex["output"], ex["query"])`).
    """
    m = CheckMetrics(check_name=check_name)
    for ex in examples:
        expected_positive = ex[label_key] == positive_label
        t0 = time.monotonic()
        predicted_positive = predict_fn(ex)
        elapsed_ms = (time.monotonic() - t0) * 1000
        m.latencies_ms.append(elapsed_ms)

        if predicted_positive and expected_positive:
            m.tp += 1
        elif predicted_positive and not expected_positive:
            m.fp += 1
        elif not predicted_positive and not expected_positive:
            m.tn += 1
        else:
            m.fn += 1

        m.per_example.append({
            "id": ex.get("id"),
            "expected": ex[label_key],
            "predicted": "unsafe" if predicted_positive else "safe",
            "correct": predicted_positive == expected_positive,
            "latency_ms": round(elapsed_ms, 2),
        })
    return m



# Claim verification: concrete evaluation using the expanded baseline set


def load_claim_baselines(path: Path = _CLAIM_BASELINES_PATH) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["examples"]


def evaluate_claim_verification(
    examples: Optional[List[dict]] = None,
    *,
    require_tool_evidence: bool = False,
) -> CheckMetrics:
    """
    Runs run_claim_verification over the (expanded) benchmark set and
    reports full P/R/F1 + latency, instead of only the 50%-accuracy
    aggregate the paper reports.
    """
    from src.layer3.claim_verifier import run_claim_verification

    examples = examples if examples is not None else load_claim_baselines()

    def predict_fn(ex: dict) -> bool:
        result = run_claim_verification(
            ex["output"], ex["query"], ex.get("evidence_passages") or None,
            require_tool_evidence=require_tool_evidence,
        )
        # "fires" (predicted unsafe) when the check does NOT pass
        return not result.passed

    return evaluate_binary_check("claim_verification", examples, predict_fn)



# The other eight checks: semantic, pii, toxicity, policy, relevance,
# financial_sanity, freshness, islamic_guard.
#
# Fills in TODO_OTHER_CHECKS (below): each check now has a labeled
# {id, query, output, expected_label} baseline file (built from the
# effectiveness benchmark data set) plus a predict_fn wired directly to the
# real Layer 3 check function, so evaluate_binary_check produces the same
# P/R/F1 + latency table for every check, not just claim_verification.
#
# pii, policy, financial_sanity, freshness, and islamic_guard are pure
# rule/regex checks and need no model. semantic, relevance, and (via
# evaluate_claim_verification above) claim_verification are embedding-backed;
# toxicity is classifier-backed. In environments without model access, tests
# stub these via the fixtures in tests/conftest.py — evaluate_check() itself
# makes no assumption about whether the model is real or faked.


_CHECK_BASELINE_FILES: Dict[str, str] = {
    "semantic":         "semantic_baselines.json",
    "pii":              "pii_baselines.json",
    "toxicity":         "toxicity_baselines.json",
    "policy":           "policy_baselines.json",
    "relevance":        "relevance_baselines.json",
    "financial_sanity": "financial_sanity_baselines.json",
    "freshness":        "freshness_baselines.json",
    "islamic_guard":    "islamic_guard_baselines.json",
}


def load_check_baselines(check_name: str) -> List[dict]:
    """Load the {id, query, output, expected_label} baseline set for one of
    the eight non-claim checks (see _CHECK_BASELINE_FILES)."""
    if check_name not in _CHECK_BASELINE_FILES:
        raise ValueError(
            f"no baseline file registered for check {check_name!r}; "
            f"known checks: {sorted(_CHECK_BASELINE_FILES)}"
        )
    path = _HERE / _CHECK_BASELINE_FILES[check_name]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["examples"]


def _predict_semantic(ex: dict) -> bool:
    from src.layer3.output_validator import _semantic_consistency
    return not _semantic_consistency(ex["output"], ex["query"])


def _predict_pii(ex: dict) -> bool:
    from src.layer3.output_validator import _detect_pii
    fired, _ = _detect_pii(ex["output"])
    return fired


def _predict_toxicity(ex: dict) -> bool:
    from src.layer3.output_validator import _toxicity_score, DEFAULT_CONFIG
    return _toxicity_score(ex["output"], DEFAULT_CONFIG) >= DEFAULT_CONFIG.toxicity_threshold


def _predict_policy(ex: dict) -> bool:
    from src.layer3.output_validator import _policy_contradiction
    fired, _ = _policy_contradiction(ex["output"])
    return fired


def _predict_relevance(ex: dict) -> bool:
    from src.layer3.output_validator import _relevance_score, DEFAULT_CONFIG
    return _relevance_score(ex["output"], ex["query"]) < DEFAULT_CONFIG.min_relevance_score


def _predict_financial_sanity(ex: dict) -> bool:
    from src.layer3.financial_data_postprocessor import postprocess_financial_data
    _, result = postprocess_financial_data(
        tool_output=ex["output"], original_query=ex["query"], jurisdiction="Algeria",
    )
    return result.hard_fail


def _predict_freshness(ex: dict) -> bool:
    from src.layer3.web_search_freshness_validator import validate_web_search_freshness
    result = validate_web_search_freshness(query=ex["query"], result_text=ex["output"])
    # Only the hard/actionable signals count as "fires" — bare
    # freshness_unknown_* soft-warns are informational and would make every
    # dateless answer register as a false positive.
    return any(
        fc.startswith("freshness_stale") for fc in result.failed_checks
    ) or result.freshness_status == "stale"


def _predict_islamic_guard(ex: dict) -> bool:
    from src.layer3.islamic_content_guard import (
        check_islamic_content, IslamicFabricationError,
    )
    try:
        result = check_islamic_content(raw_output=ex["output"], original_query=ex["query"])
    except IslamicFabricationError:
        return True
    return bool(result.failed_checks) or result.islamic_no_evidence


_CHECK_PREDICT_FNS: Dict[str, Callable[[dict], bool]] = {
    "semantic":         _predict_semantic,
    "pii":              _predict_pii,
    "toxicity":         _predict_toxicity,
    "policy":           _predict_policy,
    "relevance":        _predict_relevance,
    "financial_sanity": _predict_financial_sanity,
    "freshness":        _predict_freshness,
    "islamic_guard":    _predict_islamic_guard,
}


def evaluate_check(check_name: str, examples: Optional[List[dict]] = None) -> CheckMetrics:
    """
    Run the effectiveness benchmark for any one of the eight non-claim
    checks. Equivalent in shape/output to evaluate_claim_verification, so all
    nine checks can be reported through the same CheckMetrics/as_dict shape.
    """
    examples = examples if examples is not None else load_check_baselines(check_name)
    predict_fn = _CHECK_PREDICT_FNS[check_name]
    return evaluate_binary_check(check_name, examples, predict_fn)


def evaluate_all_checks(*, include_claim: bool = True) -> Dict[str, CheckMetrics]:
    """
    Run the full nine-row Layer 3 effectiveness table: the eight rule/model
    checks plus claim_verification. This is what "layer 3 effectiveness"
    means end-to-end — one P/R/F1/latency row per check, all measured against
    real baseline data instead of the single aggregate figure the original
    paper reported.
    """
    results: Dict[str, CheckMetrics] = {
        name: evaluate_check(name) for name in _CHECK_PREDICT_FNS
    }
    if include_claim:
        results["claim_verification"] = evaluate_claim_verification()
    return results



# Threshold sensitivity sweeps


def sweep_claim_thresholds(
    param_name: str,
    values: List[float],
    examples: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Sweep EVIDENCE_MATCH_THRESHOLD or CONTRADICTION_THRESHOLD in
    claim_verifier.py across *values*, re-running the claim-verification
    benchmark at each value. Restores the original value afterwards.

    param_name must be "EVIDENCE_MATCH_THRESHOLD" or "CONTRADICTION_THRESHOLD".
    """
    import src.layer3.claim_verifier as cv

    if not hasattr(cv, param_name):
        raise ValueError(f"claim_verifier has no threshold named {param_name!r}")

    examples = examples if examples is not None else load_claim_baselines()
    original = getattr(cv, param_name)
    results: List[dict] = []
    try:
        for v in values:
            setattr(cv, param_name, v)
            metrics = evaluate_claim_verification(examples)
            results.append({
                param_name: v,
                "precision": _r(metrics.precision),
                "recall":    _r(metrics.recall),
                "f1":        _r(metrics.f1),
                "false_positive_rate": _r(metrics.false_positive_rate),
            })
    finally:
        setattr(cv, param_name, original)
    return results


def sweep_theta_sem(
    values: List[float],
    examples: List[dict],
) -> List[dict]:
    """
    Sweep ValidatorConfig.min_semantic_score (theta_sem) across *values*
    against a labeled (query, output, expected_label) dataset where
    expected_label is "safe" when output is genuinely on-topic for query
    and "unsafe" when it is an off-topic / irrelevant response.

    NOTE: no such dataset ships in this repo yet — pass one in with the
    same {id, query, output, expected_label} shape as the claim baselines.
    This function is provided so the sweep can be run the moment that
    dataset exists, per the reviewer's request for a theta_sem sensitivity
    analysis.
    """
    from src.layer3.output_validator import _semantic_consistency, ValidatorConfig

    results: List[dict] = []
    for v in values:
        cfg = ValidatorConfig(min_semantic_score=v)

        def predict_fn(ex: dict, _cfg=cfg) -> bool:
            # "fires" (unsafe/flagged) when semantic consistency FAILS
            return not _semantic_consistency(ex["output"], ex["query"], _cfg)

        metrics = evaluate_binary_check("semantic", examples, predict_fn)
        results.append({
            "theta_sem": v,
            "precision": _r(metrics.precision),
            "recall":    _r(metrics.recall),
            "f1":        _r(metrics.f1),
            "false_positive_rate": _r(metrics.false_positive_rate),
        })
    return results



# CLI entry point


# TODO_OTHER_CHECKS: DONE. All eight remaining checks (semantic, pii,
# toxicity, policy, relevance, financial_sanity, freshness, islamic_guard)
# now have baseline JSON files in this same {id, query, output,
# expected_label} shape (see _CHECK_BASELINE_FILES) and are wired up through
# evaluate_check() / evaluate_all_checks() above, using the same
# evaluate_binary_check() primitive as claim_verification. The full 9-row
# per-check P/R/F1 table is available via `evaluate_all_checks()` or
# `python -m src.layer3.benchmark`.

def main() -> None:
    print("=== Layer 3 effectiveness — full 9-check P/R/F1/latency table ===")
    all_results = evaluate_all_checks()
    for check_name, metrics in all_results.items():
        print(f"\n--- {check_name} ---")
        print(json.dumps(metrics.as_dict(), indent=2, ensure_ascii=False))

    print("\n=== Threshold sensitivity: EVIDENCE_MATCH_THRESHOLD ===")
    for row in sweep_claim_thresholds(
        "EVIDENCE_MATCH_THRESHOLD", [0.45, 0.55, 0.65, 0.75, 0.85]
    ):
        print(row)

    print("\n=== Threshold sensitivity: CONTRADICTION_THRESHOLD ===")
    for row in sweep_claim_thresholds(
        "CONTRADICTION_THRESHOLD", [0.55, 0.65, 0.72, 0.80, 0.90]
    ):
        print(row)


if __name__ == "__main__":
    main()

