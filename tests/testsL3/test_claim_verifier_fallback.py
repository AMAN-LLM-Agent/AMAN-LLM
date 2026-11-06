# tests/test_claim_verifier_fallback.py
"""
Tests for the [FIX-claim-kb-fallback] curated fact-snippet fallback added to
src/layer3/claim_verifier.py.

Covers:
  - true, curated-domain claims pass verification when no evidence_passages
    are supplied (the core fix — previously these were false-flagged as
    claim_unsupported purely for lack of any evidence anchor)
  - off-topic / non-curated claims still correctly fail (the fallback must
    not become a rubber stamp)
  - the fallback is INERT whenever the caller supplies real evidence_passages
    (it must never override or suppress genuine tool evidence)
  - the fallback is capped and fails safe (never raises) on embedding errors
"""
from __future__ import annotations

import pytest

import src.layer3.claim_verifier as cv


# ---------------------------------------------------------------------------
# End-to-end: run_claim_verification with the fallback engaged
# ---------------------------------------------------------------------------

class TestCuratedFallbackEndToEnd:
    def test_true_curated_claim_passes_with_no_evidence(self, fake_embeddings):
        """
        A true, static fact from a curated domain (currency) should pass
        verification even though the caller supplied no evidence_passages —
        this is the whole point of the fallback.
        """
        result = cv.run_claim_verification(
            "الدينار الجزائري ينقسم إلى مائة سنتيم.",
            "كم سنتيم في الدينار الجزائري؟",
            evidence_passages=None,
        )
        assert result.passed is True
        assert "claim_unsupported:1" not in result.failed_checks
        assert result.claims[0].evidence_snippet is not None

    def test_offtopic_claim_still_fails_with_no_evidence(self, fake_embeddings):
        """
        A hallucinated claim unrelated to both the query and the curated KB
        must still be flagged — the fallback must not paper over genuine
        hallucinations just because *some* curated snippet happens to exist.
        """
        result = cv.run_claim_verification(
            "السماء زرقاء اللون في النهار الصافي جداً.",
            "أخبرني شيئاً عن مدينة الجزائر العاصمة من فضلك.",
            evidence_passages=None,
        )
        assert result.passed is False
        assert any(c.startswith("claim_unsupported") for c in result.failed_checks)
        assert result.claims[0].evidence_snippet is None

    @pytest.mark.parametrize(
        "output,query",
        [
            ("الدينار الجزائري ينقسم إلى مائة سنتيم.", "كم سنتيم في الدينار الجزائري؟"),
            ("عاصمة الجزائر هي مدينة الجزائر.", "ما هي عاصمة الجزائر؟"),
            ("استقلت الجزائر في 5 يوليو 1962.", "متى استقلت الجزائر؟"),
            ("الكيلومتر الواحد يساوي ألف متر.", "كم متراً في الكيلومتر؟"),
        ],
    )
    def test_curated_domain_examples_do_not_error(self, fake_embeddings, output, query):
        """Smoke test across all four curated domains — must not raise."""
        result = cv.run_claim_verification(output, query, evidence_passages=None)
        assert result.verification_error is False


# ---------------------------------------------------------------------------
# The fallback must be inert when real evidence is supplied
# ---------------------------------------------------------------------------

class TestCuratedFallbackOnlyAppliesWithNoEvidence:
    def test_not_invoked_when_evidence_passages_supplied(self, fake_embeddings, monkeypatch):
        """
        [FIX-claim-kb-fallback] must never override or suppress genuine
        tool evidence. Verify _retrieve_curated_evidence is simply not
        called when the caller provides evidence_passages.
        """
        calls = []
        original = cv._retrieve_curated_evidence

        def spy(claims_text):
            calls.append(claims_text)
            return original(claims_text)

        monkeypatch.setattr(cv, "_retrieve_curated_evidence", spy)

        cv.run_claim_verification(
            "الدينار الجزائري ينقسم إلى مائة سنتيم.",
            "كم سنتيم في الدينار الجزائري؟",
            evidence_passages=["نص دليل حقيقي من نتائج البحث."],
        )
        assert calls == []

    def test_invoked_when_evidence_passages_is_none(self, fake_embeddings, monkeypatch):
        calls = []
        original = cv._retrieve_curated_evidence

        def spy(claims_text):
            calls.append(claims_text)
            return original(claims_text)

        monkeypatch.setattr(cv, "_retrieve_curated_evidence", spy)

        cv.run_claim_verification(
            "الدينار الجزائري ينقسم إلى مائة سنتيم.",
            "كم سنتيم في الدينار الجزائري؟",
            evidence_passages=None,
        )
        assert len(calls) == 1

    def test_invoked_when_evidence_passages_is_empty_list(self, fake_embeddings, monkeypatch):
        """An explicitly empty list is falsy too — should also trigger fallback."""
        calls = []
        original = cv._retrieve_curated_evidence

        def spy(claims_text):
            calls.append(claims_text)
            return original(claims_text)

        monkeypatch.setattr(cv, "_retrieve_curated_evidence", spy)

        cv.run_claim_verification(
            "الدينار الجزائري ينقسم إلى مائة سنتيم.",
            "كم سنتيم في الدينار الجزائري؟",
            evidence_passages=[],
        )
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# _retrieve_curated_evidence unit behavior
# ---------------------------------------------------------------------------

class TestRetrieveCuratedEvidenceUnit:
    def test_empty_claims_returns_empty(self, fake_embeddings):
        assert cv._retrieve_curated_evidence([]) == []

    def test_returns_capped_number_of_snippets(self, fake_embeddings):
        # A claim that overlaps several currency/geography snippets at once
        claims = [
            "الدينار الجزائري ينقسم إلى مائة سنتيم مثل اليورو والدولار والجنيه "
            "وعاصمة الجزائر مدينة الجزائر والكيلومتر يساوي ألف متر."
        ]
        snippets = cv._retrieve_curated_evidence(claims)
        assert len(snippets) <= cv._MAX_CURATED_SNIPPETS

    def test_fails_safe_on_embedding_error(self, fake_embeddings, monkeypatch):
        def boom(_texts):
            raise RuntimeError("embedding backend unavailable")

        monkeypatch.setattr(cv, "_embed", boom)
        # Must not raise — best-effort fallback only.
        assert cv._retrieve_curated_evidence(["الدينار الجزائري ينقسم إلى مائة سنتيم."]) == []

    def test_unrelated_claim_returns_no_snippets(self, fake_embeddings):
        result = cv._retrieve_curated_evidence(
            ["السماء زرقاء اللون في النهار الصافي جداً."]
        )
        assert result == []


# ---------------------------------------------------------------------------
# KB structural sanity
# ---------------------------------------------------------------------------

class TestCuratedKnowledgeBaseStructure:
    def test_index_is_flattened_correctly(self):
        total_snippets = sum(len(v) for v in cv._CURATED_FACT_SNIPPETS.values())
        assert len(cv._CURATED_SNIPPET_INDEX) == total_snippets

    def test_no_duplicate_snippets_across_domains(self):
        texts = [s for s, _domain in cv._CURATED_SNIPPET_INDEX]
        assert len(texts) == len(set(texts))

    def test_kb_excludes_time_sensitive_domains(self):
        """
        Guard rail: the fallback is documented as intentionally excluding
        anything that changes over time (officeholders, live prices). This
        test just checks none of the obviously time-sensitive marker words
        leaked into the static KB.
        """
        banned_markers = ["رئيس", "وزير", "سعر السوق", "اليوم"]
        for snippet, _domain in cv._CURATED_SNIPPET_INDEX:
            for marker in banned_markers:
                assert marker not in snippet, (
                    f"time-sensitive marker {marker!r} found in curated "
                    f"snippet {snippet!r} — this KB must stay static"
                )
