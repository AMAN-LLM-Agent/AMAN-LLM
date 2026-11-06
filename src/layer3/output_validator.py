# src/layer3/output_validator.py
"""
AMAN-LLM
Layer 3 — Output Validator Orchestrator

PURPOSE
    Applies a sequence of post-generation checks to the text produced by the
    LLM Agent Core (after any tool execution) and decides whether the output
    may be released, must be blocked, or must be passed to Layer 4 with
    additional flags. Every decision is recorded through the Audit Logger.

ARCHITECTURAL ROLE
    Pipeline position:
        ... -> Layer 2 (Policy Engine Orchestrator, tool execution)
            -> Layer 3 (this module)
            -> Layer 4 (Constitutional Self-Critique)
    This module orchestrates the individual output checks. Several checks are
    delegated to sibling modules (claim_verifier, islamic_content_guard,
    financial_data_postprocessor, web_search_freshness_validator); the
    remaining checks are implemented here.

INPUTS
    raw_output          Text produced by the LLM Agent Core.
    original_query      The user query that triggered the output.
    config              ValidatorConfig (thresholds and model identifiers).
    request_id          Optional trace identifier (UUID4 generated if absent).
    evidence_passages   Optional passages retrieved by tools, used as
                        grounding evidence by the claim and Islamic checks.
    query_domain        Optional pipeline-level domain label, forwarded to the
                        freshness validator.

OUTPUTS
    Layer3Output        Dataclass carrying the pass/fail decision, the list of
                        failed check identifiers, per-check diagnostics and
                        flags consumed by Layer 4.
    Audit record        One "layer3" event emitted through log_event().

CHECKS (in execution order)
    1.  Semantic consistency between query and output (whole text, then
        clause-level topic-drift detection).
    2.  PII detection (regular-expression based).
    3.  Toxicity screening (multilingual classifier plus an Arabic insult
        lexicon).
    4.  Policy/security pattern matching (SQL, shell, script-injection and
        instruction-override patterns).
    5.  Query-output relevance scoring.
    5b. Financial data sanity (delegated, optional module).
    5c. Web-search freshness validation (delegated, optional module).
    6.  Islamic content guard (delegated; may hard-block fabricated citations).
    7.  Claim verification (delegated), followed by content-type-dependent
        downgrading of selected claim-related failure codes.

IMPLEMENTATION NOTES
    * Heavy dependencies (transformers, sentence-transformers, scikit-learn)
      are optional. When unavailable the module degrades as follows:
      toxicity classifier -> score 0.0 (the insult lexicon remains active);
      embeddings -> TF-IDF cosine -> word-level Jaccard similarity.
    * Checks 1-5 operate on the first ``max_input_length`` characters of the
      output; the delegated checks receive the full output text.
    * The public entry point run_layer3() never raises; an unexpected
      internal error yields passed=False with failed_checks=["internal_error"]
      (fail-closed behaviour).
    * The detection mechanisms are heuristic (patterns, lexicons and
      embedding similarity). They detect selected categories of problems and
      are not claimed to be exhaustive.
    * Threshold values in ValidatorConfig were calibrated empirically against
      the project's evaluation sets; see the accompanying paper for the
      evaluation protocol.

REPRODUCIBILITY
    Model identifiers and thresholds are fixed in ValidatorConfig
    (DEFAULT_CONFIG). Models are loaded lazily and cached per process.
    Toxicity scores are cached by a hash of the input text.

REVISION NOTES
    FIX-74d36501  Escalation of ``require_tool_evidence`` for security-related
                  queries is gated on query intent (information request vs.
                  condemnation/opposition stance), not on entity presence.
    FIX-cd57ac80  ``claim_contradiction`` is removed from the failure list for
                  content types tutorial, definition, comparison, recipe and
                  how_to when no evidence passages exist (the contradiction
                  checker is intended for news/data-style queries).
    FIX-334cf5f1  ``contains_unverified_religious_content`` is set once per
                  call and is not cleared afterwards, independent of
                  ``query_domain``.
    FIX-b31cf026  A HARD FAIL on ``financial_price_sanity`` sets
                  ``financial_price_correction_required=True`` on the output;
                  Layer 4 is expected to honour this flag.
    FIX-freshness The freshness validator receives ``content_type`` and
                  ``query_domain`` so that freshness checks can be restricted
                  to freshness-sensitive domains.
"""
from __future__ import annotations

import functools
import hashlib
import logging
import re
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Optional third-party dependencies
# Each import is guarded so that the validator remains importable when a
# dependency is absent; the affected check falls back as described in the
# module documentation.

try:
    from transformers import pipeline as hf_pipeline
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    logging.warning("transformers not available — toxicity scoring disabled.")

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logging.warning(
        "sentence-transformers not available — falling back to TF-IDF similarity."
    )

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available — falling back to word-overlap similarity.")

from src.audit_logging.audit_logger import log_event
from .claim_verifier import ClaimVerificationResult, run_claim_verification

from .islamic_content_guard import (
    IslamicContentResult,
    IslamicFabricationError,
    check_islamic_content,
)

# The freshness validator is an optional component; check 5c is skipped when
# it cannot be imported.
try:
    from .web_search_freshness_validator import (
        validate_web_search_freshness,
        parse_search_meta,
        FreshnessResult as _FreshnessResult,
    )
    _FRESHNESS_VALIDATOR_AVAILABLE = True
except ImportError:
    _FRESHNESS_VALIDATOR_AVAILABLE = False

logger = logging.getLogger(__name__)

if not _FRESHNESS_VALIDATOR_AVAILABLE:
    logger.debug("web_search_freshness_validator not available — P2-4 check skipped")

# Generic Arabic error message ("Sorry, an unexpected error occurred"). An
# output equal to this string is treated as the result of a failed tool
# execution (see "tool_execution_failed" in _run_layer3_impl).
FALLBACK_ERROR_MSG = "عذراً، حدث خطأ غير متوقع"



# Configuration


@dataclass(frozen=True)
class ValidatorConfig:
    """
    Immutable configuration for the Output Validator Orchestrator.

    Attributes
    
    toxicity_threshold:
        Classifier score at or above which the ``toxicity`` check fails.
    min_relevance_score:
        Minimum query-output similarity for the ``relevance`` check.
    min_semantic_score:
        Minimum query-output similarity for the ``semantic`` check.
    min_claim_confidence:
        Minimum claim-verification confidence. Kept aligned with the
        low-confidence threshold used inside claim_verifier.
    max_input_length:
        Number of leading characters of the output examined by checks 1-5.
    embedding_model, toxicity_model:
        Hugging Face model identifiers.
    toxicity_cache_size:
        Maximum number of cached toxicity scores.
    require_tool_evidence:
        If True, claims lacking tool evidence are treated as unsupported.
    enable_islamic_guard:
        Enables the Islamic content guard (check 6).
    """
    # Calibration note (toxicity_threshold): on benign Arabic technical and
    # scientific text the multilingual classifier produced baseline scores of
    # roughly 0.05-0.35 in the project's safe-set evaluation. A threshold of
    # 0.08 produced a 70% false-positive rate on that set, so the threshold
    # was raised above the observed benign ceiling (~0.36).
    toxicity_threshold:   float = 0.65
    min_relevance_score:  float = 0.3
    # Calibration note (min_semantic_score): both this threshold and
    # min_relevance_score are applied to the output of the same function,
    # _semantic_consistency_score. They were previously 0.25 and 0.3; outputs
    # scoring between the two failed ``relevance`` but passed ``semantic``.
    # The thresholds are now identical.
    min_semantic_score:  float = 0.3
    # Calibration note (min_claim_confidence): aligned with claim_verifier's
    # LOW_CONFIDENCE_THRESHOLD (0.45). A higher value here previously acted
    # as a second, stricter confidence gate on top of claim_verifier's own
    # decision and affected grounding-exempt claims when no evidence passages
    # were available.
    min_claim_confidence: float = 0.45
    max_input_length:     int   = 10_000
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    toxicity_model:      str = "unitary/multilingual-toxic-xlm-roberta"
    toxicity_cache_size: int = 256
    require_tool_evidence: bool = False
    enable_islamic_guard: bool = True


# Default configuration used by all entry points unless overridden.
DEFAULT_CONFIG = ValidatorConfig()


# PII patterns (check 2)
#
# Regular expressions for selected categories of personal identifiers
# (Algerian and international phone numbers, national-ID-length digit
# strings, bank-account-like numbers, e-mail addresses, IPv4 addresses and
# US-SSN-like numbers). The patterns are syntactic: they do not validate
# checksums or semantics and may match non-PII numeric strings.


_RAW_PII_PATTERNS: Dict[str, str] = {
    "national_id":  r"(?<!\d)\d{10,12}(?!\d)",
    "phone_dz":     r"(?:(?:\+|00)213|0)[5-7]\d{8}",
    "phone_intl":   r"(?:\+\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,6}",
    "bank_account": r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
    "email":        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "ipv4":         r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "ssn_like":     r"\b\d{3}-\d{2}-\d{4}\b",
}

PII_PATTERNS: Dict[str, re.Pattern] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in _RAW_PII_PATTERNS.items()
}


# Policy / security patterns (check 4)
#
# Patterns for selected content categories in model output: SQL data-
# manipulation statements, shell commands, code-execution calls, script
# injection and natural-language instruction-override phrases. A match
# indicates the presence of the pattern only; it does not establish intent.


_POLICY_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bDROP\s+TABLE\b",                      re.IGNORECASE),
    re.compile(r"\bDELETE\s+FROM\b",                     re.IGNORECASE),
    re.compile(r"\bTRUNCATE\s+TABLE\b",                  re.IGNORECASE),
    re.compile(r"\bINSERT\s+INTO\b.*\bVALUES\b",         re.IGNORECASE),
    re.compile(r"\bUNION\s+SELECT\b",                    re.IGNORECASE),
    re.compile(r"--\s*$",                                re.MULTILINE),
    re.compile(r";\s*DROP",                              re.IGNORECASE),
    re.compile(r"\brm\s+-[rRfF]{1,3}\b"),
    re.compile(r"\bchmod\s+[0-7]{3,4}\b"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bchown\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+if="),
    re.compile(r"\bwget\b|\bcurl\b.*\bsh\b"),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bexec\s*\("),
    re.compile(r"\bos\.system\s*\("),
    re.compile(r"\bsubprocess\s*\."),
    re.compile(r"<\s*script",                            re.IGNORECASE),
    re.compile(r"javascript\s*:",                        re.IGNORECASE),
    re.compile(
        r"\b(ignore|disregard|forget|override)"
        r"(?:\s+(?:all|the|any|every|those|these|your|my|our|its|previous|above|prior|earlier))*"
        r"\s+(instructions?|prompts?|rules?|guidelines?|constraints?|restrictions?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bsystem\s+prompt\b",                   re.IGNORECASE),
    re.compile(r"\bjailbreak\b",                         re.IGNORECASE),
    re.compile(r"\bdan\s+mode\b",                        re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(if\s+you\s+(are|were)|an?\s+unrestricted)", re.IGNORECASE),
    re.compile(r"\bpretend\s+(you\s+are|to\s+be)\b",     re.IGNORECASE),
]

# Zero-width and bidirectional control characters removed during
# input sanitisation (_sanitise_input).
_ZERO_WIDTH_RE = re.compile(
    r"[\u200B\u200C\u200D\u200E\u200F\u00AD\uFEFF\u2060]"
)


# Lazy-loaded models
# Models are loaded on first use and cached for the lifetime of the process.


@functools.lru_cache(maxsize=1)
def _get_toxicity_classifier(model_name: str):
    """Return a cached text-classification pipeline, or None if unavailable."""
    if not _HF_AVAILABLE:
        return None
    try:
        return hf_pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            truncation=True,
            max_length=512,
        )
    except Exception as exc:
        logger.error("Failed to load toxicity model '%s': %s", model_name, exc)
        return None


@functools.lru_cache(maxsize=1)
def _get_embedding_model(model_name: str):
    """Return a cached SentenceTransformer in eval mode, or None if unavailable."""
    if not _ST_AVAILABLE:
        return None
    try:
        m = SentenceTransformer(model_name)
        m.eval()
        return m
    except Exception as exc:
        logger.error("Failed to load embedding model '%s': %s", model_name, exc)
        return None



# Thread-safe toxicity result cache
# Maps a hash of the input text to its toxicity score; access is guarded by
# a lock. Eviction is first-in-first-out once toxicity_cache_size is reached.


_toxicity_cache: Dict[str, float] = {}
_toxicity_cache_lock = threading.Lock()


def _cache_key(text: str) -> str:
    """Return an MD5 digest of *text* used solely as a cache key (not for security)."""
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()



# Output dataclass


@dataclass
class Layer3Output:
    """
    Result of Layer 3 validation, consumed by Layer 4.

    Attributes
    
    passed:
        True if no check identifier is present in ``failed_checks``.
    raw_output:
        Output text after sanitisation (and, where applicable, financial
        post-processing or replacement by a fixed message).
    failed_checks:
        Identifiers of failed checks (e.g. "semantic", "privacy",
        "toxicity", "policy", "relevance", "claim_*", "freshness_*").
    pii_detected, toxicity_score:
        Results of checks 2 and 3.
    claim_verification:
        Result object returned by the claim verifier, if it was run.
    request_id:
        Trace identifier shared with the audit record.
    islamic_result, islamic_no_evidence, contains_unverified_religious_content:
        Islamic content guard outputs. The last flag is set once per call.
    financial_postprocessor_flags, financial_price_correction_required:
        Financial sanity outputs. When the latter is True, Layer 4 is
        expected to correct the price or state that it could not be verified
        rather than append a disclaimer only.
    content_type:
        Heuristic label: recipe, tutorial, definition, comparison,
        arithmetic or factual.
    freshness_checks, web_search_retrieved_at, web_search_data_type,
    freshness_status:
        Web-search freshness outputs.
    per_check_latency_ms:
        Wall-clock latency per check in milliseconds. Keys correspond to the
        executed checks: semantic, pii, toxicity, policy, relevance,
        financial_sanity, freshness, islamic_guard, claim_verification.
    """
    passed:             bool
    raw_output:         str
    failed_checks:      List[str]                          = field(default_factory=list)
    pii_detected:       bool                               = False
    toxicity_score:     float                              = 0.0
    claim_verification: Optional[ClaimVerificationResult]  = None
    request_id:         str                                = field(default_factory=lambda: str(uuid.uuid4()))

    islamic_result:                        Optional[IslamicContentResult] = None
    islamic_no_evidence:                   bool = False
    contains_unverified_religious_content: bool = False

    financial_postprocessor_flags: List[str] = field(default_factory=list)
    # Set when the financial sanity check hard-fails. Layer 4 is expected to
    # correct the price or state that it could not be verified, instead of
    # issuing a disclaimer while retaining the implausible figure.
    financial_price_correction_required: bool = False

    content_type: str = "factual"

    freshness_checks:          List[str]        = field(default_factory=list)
    web_search_retrieved_at:   Optional[str]    = None
    web_search_data_type:      str              = "generic"
    freshness_status:          str              = "unknown"

    # Per-check latency (ms), recorded in addition to the aggregate elapsed
    # time so that slow requests can be attributed to a specific check.
    per_check_latency_ms: Dict[str, float] = field(default_factory=dict)



# Security-query intent classification (FIX-74d36501)
#
# Queries that mention a terrorism-related entity are treated as sensitive
# information requests, which raises the evidence requirement for claim
# verification. An earlier entity-presence rule also escalated queries that
# merely express opposition (e.g. "أنا ضد داعش وضد الإرهاب"), for which no
# tool evidence exists or is required. The rule below therefore distinguishes
# information-seeking queries from condemnation/opposition statements.
# The regular expressions are keyword lists (Arabic, English, French) and
# cover only the listed terms.


# Terrorism / security-related entity terms.
_TERRORISM_ENTITIES_RE = re.compile(
    r"\b(داعش|ISIS|ISIL|إرهاب|إرهابي|تطرف|القاعدة|بوكو\s*حرام"
    r"|طالبان|جهاد|الدولة\s*الإسلامية|terrorism|terrorist|extremis[tm]"
    r"|jihadis[tm]|al[- ]qa[ei]da|boko\s*haram|al[- ]shabaab)\b",
    re.IGNORECASE | re.UNICODE,
)

# Condemnation / opposition stance markers (Arabic, English, French).
# A query containing an entity term and one of these markers, but no
# information-request marker, is treated as a stance statement.
_CONDEMNATION_STANCE_RE = re.compile(
    r"\b(ضد|أرفض|ندين|نرفض|نشجب|معارض|أعارض|أشجب|لا\s+أؤيد|أدين"
    r"|against|oppose|condemn|reject|denounce|anti[- ]|abhor|detest"
    r"|contre|condamne|oppose[rz]|rejette)\b",
    re.IGNORECASE | re.UNICODE,
)

# Information-request markers: interrogatives and topic words indicating that
# the query asks for facts about the entity.
_INFO_REQUEST_RE = re.compile(
    r"\b(من هم|ما هو|كيف|متى|لماذا|أين|شرح|اشرح|هل صحيح|أخبرني عن"
    r"|who (are|is)|what (is|are)|how (do|does|did)|when did|why did"
    r"|explain|tell me about|history of|tactics|operations|funding"
    r"|تمويل|عمليات|تكتيكات|أساليب|أيديولوجية)\b",
    re.IGNORECASE | re.UNICODE,
)


def _is_security_information_request(query: str) -> bool:
    """
    Decide whether *query* is a security-related information request.

    Returns False if no security entity term is present, or if the query
    contains a condemnation marker and no information-request marker.
    Returns True otherwise, including the ambiguous case (entity present,
    neither marker present), which is treated conservatively as an
    information request.
    """
    if not _TERRORISM_ENTITIES_RE.search(query):
        return False  # no security entity → definitely not a security info request

    has_info_request = bool(_INFO_REQUEST_RE.search(query))
    has_condemnation = bool(_CONDEMNATION_STANCE_RE.search(query))

    if has_condemnation and not has_info_request:
        # Stance statement only: do not escalate.
        logger.debug(
            "_is_security_information_request: query contains security entity "
            "but is a condemnation stance — not escalating require_tool_evidence. "
            "query=%r", query[:120]
        )
        return False

    # Explicit information request, or ambiguous (entity present, neither
    # marker matched): treated as an information request (conservative).
    return True



# Public entry point


def _run_layer3_impl(
    raw_output: str,
    original_query: str,
    config: ValidatorConfig = DEFAULT_CONFIG,
    request_id: Optional[str] = None,
    evidence_passages: Optional[List[str]] = None,
    *,
    query_domain: Optional[str] = None,
) -> Layer3Output:
    """
    Run all Layer 3 checks on *raw_output* (implementation body).

    Applications should call run_layer3(), which wraps this function and
    guarantees that no exception propagates. This function may raise on
    unexpected internal errors.

    Parameters
    
    raw_output:
        Text produced by the LLM Agent Core.
    original_query:
        The original user query.
    config:
        Thresholds and model identifiers.
    request_id:
        Caller-supplied trace identifier; a UUID4 is generated if omitted.
    evidence_passages:
        Passages obtained from tool results, used as grounding evidence.
    query_domain:
        Pipeline-level domain label (e.g. 'recipe', 'technical', 'fiqh',
        'factual'), forwarded to the freshness validator so that it can skip
        freshness checks for timeless domains.

    Returns
    
    Layer3Output
        Always populated; ``passed`` is True only if ``failed_checks`` is
        empty.

    Early exits
    
    * Empty output after sanitisation -> failed_checks=["empty_output"].
    * Output equal to FALLBACK_ERROR_MSG ->
      failed_checks=["tool_execution_failed"].
    * Short apology text without safety-related vocabulary is treated as a
      capability-restriction message; it is replaced by a fixed message and
      returned with passed=True without running further checks.
    """
    start_ts   = time.monotonic()
    request_id = request_id or str(uuid.uuid4())
    failed: List[str] = []
    # Per-check latency in milliseconds; reported on Layer3Output and in the
    # audit record.
    per_check_latency_ms: Dict[str, float] = {}

    # Normalise inputs: None/non-string handling, zero-width character
    # removal and whitespace stripping.
    raw_output     = _sanitise_input(raw_output)
    original_query = _sanitise_input(original_query)

    if not raw_output:
        failed.append("empty_output")
        _emit_audit_log(
            request_id=request_id, passed=False, failed=failed,
            pii_found=False, pii_types=[], tox_score=0.0,
            sem_score=0.0, rel_score=0.0, policy_matches=[],
            claim_result=None,
            elapsed_ms=int((time.monotonic() - start_ts) * 1000),
            raw_output=raw_output, original_query=original_query,
            islamic_result=None,
        )
        return Layer3Output(
            passed=False, raw_output=raw_output,
            failed_checks=failed, request_id=request_id,
        )

    if raw_output.strip() == FALLBACK_ERROR_MSG:
        logger.info(
            "run_layer3 | request_id=%s | tool_execution_failed — skipping "
            "semantic/relevance checks", request_id,
        )
        failed.append("tool_execution_failed")
        elapsed_ms = int((time.monotonic() - start_ts) * 1000)
        _emit_audit_log(
            request_id=request_id, passed=False, failed=failed,
            pii_found=False, pii_types=[], tox_score=0.0,
            sem_score=None, rel_score=None, policy_matches=[],
            claim_result=None, elapsed_ms=elapsed_ms,
            raw_output=raw_output, original_query=original_query,
            islamic_result=None,
        )
        return Layer3Output(
            passed=False, raw_output=raw_output,
            failed_checks=failed, request_id=request_id,
        )

    # Capability-restriction stub detection.
    # Short apology messages (< _TOOL_BLOCK_STUB_MAX_LEN characters) that use
    # the apology vocabulary below are treated as "tool not available for
    # this role" messages. A genuine safety refusal can be equally short and
    # use the same vocabulary, so the stub interpretation additionally
    # requires the absence of the safety/policy markers; texts containing a
    # marker proceed through the normal check pipeline.
    _TOOL_BLOCK_STUB_MAX_LEN = 40
    _TOOL_BLOCK_INDICATORS = ("أعتذر", "عذرا", "لا يمكنني")
    _SAFETY_REFUSAL_MARKERS = (
        "سياس", "الأمان", "أمان", "محظور", "خطر", "ضار", "قانون",
        "غير مسموح", "غير أخلاقي", "يتعارض",
    )
    _looks_like_capability_stub = (
        len(raw_output.strip()) < _TOOL_BLOCK_STUB_MAX_LEN
        and any(ind in raw_output for ind in _TOOL_BLOCK_INDICATORS)
        and not any(marker in raw_output for marker in _SAFETY_REFUSAL_MARKERS)
    )
    if _looks_like_capability_stub:
        # Fixed Arabic message: "This function requires additional
        # permissions not available to your current account. You may contact
        # the administrator to obtain the appropriate access."
        capability_msg = (
            "عذراً، هذه الوظيفة تتطلب صلاحيات إضافية غير متاحة لحسابك الحالي. "
            "يمكنك التواصل مع المسؤول للحصول على الوصول المناسب."
        )
        logger.info(
            "run_layer3 | request_id=%s | tool_block_stub detected (len=%d)",
            request_id, len(raw_output.strip()),
        )
        elapsed_ms = int((time.monotonic() - start_ts) * 1000)
        return Layer3Output(
            passed=True, raw_output=capability_msg,
            failed_checks=[], request_id=request_id,
            content_type="factual",
        )

    # Checks 1-5 operate on the leading max_input_length characters only.
    truncated = raw_output[: config.max_input_length]

    # ── Check 1 — Semantic consistency ──────────────────────────────────
    _t0 = time.monotonic()
    sem_score = _semantic_consistency_score(truncated, original_query, config)
    if sem_score < config.min_semantic_score:
        failed.append("semantic")
    else:
        # Clause-level topic-drift check. Whole-text similarity can remain
        # above threshold when a correct on-topic clause is combined with an
        # unrelated one (e.g. "Linux نظام تشغيل، واليوم حضرت كسكس." — "Linux
        # is an operating system, and today I made couscous"). Each clause
        # is therefore scored independently, and a mixed output is flagged
        # even if the whole-text score passed.
        drift, drift_clause = _detect_topic_drift(truncated, original_query, config)
        if drift:
            failed.append("semantic")
            logger.warning(
                "run_layer3 | request_id=%s | topic drift detected in "
                "clause=%r (whole-text sem_score=%.2f passed threshold)",
                request_id, drift_clause, sem_score,
            )
    per_check_latency_ms["semantic"] = round((time.monotonic() - _t0) * 1000, 2)

    # ── Check 2 — PII detection ─────────────────────────────────────────
    _t0 = time.monotonic()
    pii_found, pii_types = _detect_pii(truncated)
    if pii_found:
        failed.append("privacy")
    per_check_latency_ms["pii"] = round((time.monotonic() - _t0) * 1000, 2)

    # ── Check 3 — Toxicity screening ────────────────────────────────────
    _t0 = time.monotonic()
    tox_score = _toxicity_score(truncated, config)
    lexicon_hit, lexicon_term = _contains_arabic_insult(truncated)
    # Lexicon signal alongside the classifier. In the project's evaluation,
    # the classifier scored mild Algerian-dialect insults (e.g. "يا غبي
    # اسكت", "روح بعيد يا فاشل") at 0.01-0.36, overlapping the score range
    # observed on benign Arabic text, so no single threshold separated the
    # two. A lexicon of insult vocabulary and put-down patterns is therefore
    # used as an independent signal; the check fails if either signal fires.
    if tox_score >= config.toxicity_threshold or lexicon_hit:
        failed.append("toxicity")
        if lexicon_hit:
            logger.warning(
                "run_layer3 | request_id=%s | toxicity flagged via insult "
                "lexicon (term=%r), classifier_score=%.3f",
                request_id, lexicon_term, tox_score,
            )
    per_check_latency_ms["toxicity"] = round((time.monotonic() - _t0) * 1000, 2)

    # ── Check 4 — Policy / security violations ───────────────────────────
    _t0 = time.monotonic()
    policy_hit, policy_matches = _policy_contradiction(truncated)
    if policy_hit:
        failed.append("policy")
    per_check_latency_ms["policy"] = round((time.monotonic() - _t0) * 1000, 2)

    # ── Check 5 — Relevance scoring ──────────────────────────────────────
    _t0 = time.monotonic()
    rel_score = _relevance_score(truncated, original_query, config)
    if rel_score < config.min_relevance_score:
        failed.append("relevance")
    per_check_latency_ms["relevance"] = round((time.monotonic() - _t0) * 1000, 2)

    # ── [P1-4] Check 5b — Financial data sanity ──────────────────────────
    # Optional check, delegated to financial_data_postprocessor. When the
    # query concerns a commodity or an everyday staple good, the output is
    # post-processed (jurisdiction fixed to "Algeria"); a hard failure adds
    # "financial_price_sanity" to the failure list and sets the correction
    # flag for Layer 4. Import or runtime errors skip the check.
    _fp_commodity = None
    _fp_result: Optional["FinancialPostProcessResult"] = None
    _financial_price_correction_required = False
    _t0 = time.monotonic()
    try:
        from src.layer3.financial_data_postprocessor import (
            postprocess_financial_data,
            _detect_commodity,
            _detect_staple_good,
        )
        _fp_commodity = _detect_commodity(original_query)
        # Everyday-goods queries (e.g. bread, milk, a car, a phone) contain
        # no forex/metal commodity keyword; a separate staple-good detector
        # is therefore consulted so that these queries are also processed.
        _fp_is_staple = False
        if not _fp_commodity:
            _fp_staple = _detect_staple_good(original_query)
            if _fp_staple:
                _fp_commodity = _fp_staple
                _fp_is_staple = True
        if _fp_commodity:
            raw_output, _fp_result = postprocess_financial_data(
                tool_output=raw_output,
                original_query=original_query,
                jurisdiction="Algeria",
            )
            truncated = raw_output[: config.max_input_length]
            if _fp_result.hard_fail:
                failed.append("financial_price_sanity")
                # Signal to Layer 4 that the value must be corrected rather
                # than only accompanied by a disclaimer.
                _financial_price_correction_required = True
                logger.error(
                    "run_layer3 | request_id=%s | financial_price_sanity HARD FAIL"
                    " — financial_price_correction_required=True: %s",
                    request_id, _fp_result.flags,
                )
            elif _fp_result.flags:
                logger.warning(
                    "run_layer3 | request_id=%s | financial_postprocessor flags: %s",
                    request_id, _fp_result.flags,
                )
    except ImportError:
        logger.debug("financial_data_postprocessor not available — skipping P1-4 check")
    except Exception as _fp_exc:
        logger.warning("P1-4 financial postprocessor failed: %s", _fp_exc)
    per_check_latency_ms["financial_sanity"] = round((time.monotonic() - _t0) * 1000, 2)

    _t0 = time.monotonic()
    # ── [P2-4] Check 5c — Web search freshness validation ────────────────
    # content_type and query_domain are passed to the freshness validator so
    # that it can restrict checks to freshness-sensitive domains (a generic
    # "unknown freshness" warning is not meaningful for recipe, arithmetic,
    # tutorial or definition queries).

    _freshness_result: Optional["_FreshnessResult"] = None
    _freshness_checks: List[str] = []
    _ws_retrieved_at:  Optional[str] = None
    _ws_data_type:     str = "generic"
    _security_freshness_escalation: bool = False

    # The content type is classified here because the freshness validator
    # requires it; the same value is reused below for the Islamic guard and
    # the claim-related downgrades.
    _early_content_type = _classify_content_type(original_query, truncated)

    if _FRESHNESS_VALIDATOR_AVAILABLE:
        try:
            candidate_texts = (evidence_passages or []) + [raw_output]
            meta: Optional[dict] = None
            for cand in candidate_texts:
                meta = parse_search_meta(cand)
                if meta:
                    break

            if meta:
                _ws_data_type    = meta["data_type"]
                _ws_retrieved_at = meta["retrieved_at"]
                is_stale         = meta["stale"]
                age_h            = meta["age_h"]

                if is_stale:
                    _freshness_checks.append(f"freshness_stale_{_ws_data_type}")
                    logger.warning(
                        "run_layer3 | request_id=%s | [P2-4] stale web_search "
                        "(header): data_type=%s retrieved_at=%s age_h=%s",
                        request_id, _ws_data_type, _ws_retrieved_at,
                        f"{age_h:.1f}" if age_h is not None else "unknown",
                    )

            else:
                search_text = (
                    " ".join(evidence_passages)
                    if evidence_passages
                    else raw_output
                )
                # Forward content_type and query_domain so that timeless
                # domains can be skipped by the validator.
                _freshness_result = validate_web_search_freshness(
                    query=original_query,
                    result_text=search_text,
                    content_type=_early_content_type,
                    query_domain=query_domain,
                )
                _freshness_checks  = _freshness_result.failed_checks
                _ws_data_type      = _freshness_result.data_type
                _ws_retrieved_at   = (
                    _freshness_result.retrieved_at.strftime("%Y-%m-%dT%H:%M:%SZ")
                    if _freshness_result.retrieved_at
                    else None
                )
                if _freshness_checks:
                    logger.warning(
                        "run_layer3 | request_id=%s | [P2-4] freshness "
                        "data_type=%s | age_h=%s | checks=%s",
                        request_id, _ws_data_type,
                        _freshness_result.staleness_h, _freshness_checks,
                    )

            _security_freshness_escalation = False
            for fc in _freshness_checks:
                if fc == "freshness_unknown_security":
                    _security_freshness_escalation = True
                    logger.warning(
                        "run_layer3 | request_id=%s | [P2-4] SECURITY FRESHNESS "
                        "ESCALATION — require_tool_evidence will be forced True",
                        request_id,
                    )
                    # The finding is also recorded as a failure: an answer
                    # of unknown freshness on a security/political topic is
                    # not released silently.
                    if fc not in failed:
                        failed.append(fc)
                elif fc.startswith("freshness_stale_") or fc.startswith("freshness_unknown_"):
                    # All stale/unknown freshness findings are propagated to
                    # the failure list. (An earlier revision propagated only
                    # "freshness_stale_price"; other findings were logged but
                    # did not affect the decision.)
                    if fc not in failed:
                        failed.append(fc)
                        logger.warning(
                            "run_layer3 | request_id=%s | [P2-4] freshness check "
                            "failed: %s", request_id, fc,
                        )
                else:
                    logger.warning(
                        "run_layer3 | request_id=%s | [P2-4] SOFT WARN: %s",
                        request_id, fc,
                    )

        except Exception as _fresh_exc:
            logger.warning(
                "run_layer3 | request_id=%s | [P2-4] freshness check error: %s",
                request_id, _fresh_exc,
            )

    _freshness_status: str = "unknown"
    if _freshness_result is not None:
        _freshness_status = _freshness_result.freshness_status
    elif _freshness_checks:
        if any("stale" in fc for fc in _freshness_checks):
            _freshness_status = "stale"
        elif any("unknown" in fc for fc in _freshness_checks):
            _freshness_status = "unknown"
        else:
            _freshness_status = "fresh"
    elif _ws_retrieved_at is not None:
        _freshness_status = "fresh"

    per_check_latency_ms["freshness"] = round((time.monotonic() - _t0) * 1000, 2)

    # ── [P2-2] Content-type classification ──────────────────────────────
    content_type = _early_content_type  # computed above

    # ── [P0-2] Islamic content guard (check 6) ───────────────────────────
    # Delegated to islamic_content_guard. Failed-check identifiers are merged
    # into the failure list. IslamicFabricationError is treated as a hard
    # block: processing stops and a fixed Arabic refusal message ("I cannot
    # cite this verse without a reliable source. Please refer to an
    # authenticated Quran copy.") replaces the output.
    islamic_result: Optional[IslamicContentResult] = None
    islamic_no_evidence = False
    contains_unverified_religious_content = False

    _t0 = time.monotonic()
    if config.enable_islamic_guard:
        try:
            islamic_result = check_islamic_content(
                raw_output=raw_output,
                original_query=original_query,
                evidence_passages=evidence_passages,
                content_type=content_type,
            )

            for fc in islamic_result.failed_checks:
                if fc not in failed:
                    failed.append(fc)

            islamic_no_evidence = islamic_result.islamic_no_evidence

            if islamic_no_evidence:
                # Set once per call and not cleared later, independent of
                # query_domain (FIX-334cf5f1).
                contains_unverified_religious_content = True
                logger.warning(
                    "run_layer3 | request_id=%s | Islamic fiqh answer without "
                    "evidence — OPT-10 suppressed in L4 (permanent this cycle).",
                    request_id,
                )

        except IslamicFabricationError as fab_exc:
            logger.error(
                "run_layer3 | request_id=%s | HARD BLOCK — IslamicFabricationError: %s",
                request_id, fab_exc,
            )
            fabrication_check = "islamic_fabricated_citation:hard_block"
            if fabrication_check not in failed:
                failed.append(fabrication_check)

            islamic_result = IslamicContentResult(
                passed=False,
                failed_checks=[fabrication_check],
                warnings=[],
                islamic_no_evidence=False,
                fabricated_citations=[str(fab_exc)[:200]],
                hajj_errors=[],
            )
            per_check_latency_ms["islamic_guard"] = round((time.monotonic() - _t0) * 1000, 2)
            elapsed_ms = int((time.monotonic() - start_ts) * 1000)
            _emit_audit_log(
                request_id=request_id, passed=False, failed=failed,
                pii_found=pii_found if "pii_found" in dir() else False,
                pii_types=pii_types if "pii_types" in dir() else [],
                tox_score=tox_score if "tox_score" in dir() else 0.0,
                sem_score=sem_score if "sem_score" in dir() else None,
                rel_score=rel_score if "rel_score" in dir() else None,
                policy_matches=policy_matches if "policy_matches" in dir() else [],
                claim_result=None, elapsed_ms=elapsed_ms,
                raw_output=raw_output, original_query=original_query,
                islamic_result=islamic_result,
                fp_commodity=None, fp_result=None,
                freshness_checks=_freshness_checks,
                web_search_retrieved_at=_ws_retrieved_at,
                web_search_data_type=_ws_data_type,
            )
            return Layer3Output(
                passed=False,
                raw_output=(
                    "لا يمكنني الاستشهاد بهذه الآية دون مصدر موثوق. "
                    "يُرجى الرجوع إلى مصحف معتمد."
                ),
                failed_checks=failed,
                pii_detected=False, toxicity_score=0.0,
                claim_verification=None, request_id=request_id,
                islamic_result=islamic_result,
                islamic_no_evidence=False,
                contains_unverified_religious_content=True,
                content_type=content_type,
                freshness_checks=_freshness_checks,
                web_search_retrieved_at=_ws_retrieved_at,
                web_search_data_type=_ws_data_type,
                freshness_status=_freshness_status,
                per_check_latency_ms=per_check_latency_ms,
            )

    per_check_latency_ms["islamic_guard"] = round((time.monotonic() - _t0) * 1000, 2)

    # ── Check 7 — Claim verification ──────────────────────────────────────
    # The evidence requirement is raised (require_tool_evidence=True) for:
    # explicit configuration, Islamic answers without evidence, security
    # information requests (intent-aware, FIX-74d36501) and security-
    # freshness escalation. For grounding-exempt content types without
    # evidence passages it is lowered again below.
    _query_is_sensitive_security = _is_security_information_request(original_query)
    if _query_is_sensitive_security and not evidence_passages:
        logger.warning(
            "run_layer3 | request_id=%s | security INFO REQUEST detected — "
            "require_tool_evidence upgraded to True",
            request_id,
        )

    effective_require_tool_evidence = (
        config.require_tool_evidence or islamic_no_evidence
        or _query_is_sensitive_security
        or _security_freshness_escalation
    )
    if effective_require_tool_evidence and not config.require_tool_evidence:
        effective_config = ValidatorConfig(
            **{
                f.name: getattr(config, f.name)
                for f in config.__dataclass_fields__.values()  # type: ignore[attr-defined]
                if f.name != "require_tool_evidence"
            },
            require_tool_evidence=True,
        )
    else:
        effective_config = config

    NON_FACTUAL_TYPES = {"recipe", "tutorial", "how_to", "creative", "definition",
                         "greeting", "comparison", "arithmetic"}

    if content_type in _GROUNDING_EXEMPT_CONTENT_TYPES and not evidence_passages:
        logger.debug(
            "run_layer3 | request_id=%s | content_type=%s — "
            "claim_unsupported downgraded (not a factual query type)",
            request_id, content_type,
        )
        effective_require_tool_evidence = False
        effective_config = ValidatorConfig(
            **{
                f.name: getattr(effective_config, f.name)
                for f in effective_config.__dataclass_fields__.values()  # type: ignore[attr-defined]
                if f.name != "require_tool_evidence"
            },
            require_tool_evidence=False,
        )

    _t0 = time.monotonic()
    claim_result = _run_claim_verification_safe(
        raw_output, original_query, effective_config,
        evidence_passages, require_tool_evidence=effective_require_tool_evidence,
    )
    per_check_latency_ms["claim_verification"] = round((time.monotonic() - _t0) * 1000, 2)
    if claim_result.verification_error:
        if "claim_model_failure" not in failed:
            failed.append("claim_model_failure")
    elif not claim_result.passed:
        for c in claim_result.failed_checks:
            if c not in failed:
                failed.append(c)

    # ── [P2-2] Content-type-dependent downgrade of "claim_unsupported" ──
    # Only "claim_unsupported*" identifiers are removed; "claim_low_confidence"
    # is retained for all content types.
    if content_type in NON_FACTUAL_TYPES:
        pre_downgrade = failed[:]
        failed = [
            f for f in failed
            if not f.startswith("claim_unsupported")
            # claim_low_confidence is kept — low confidence is a signal
            # regardless of content type (e.g. self_contradiction on definition)
        ]
        if failed != pre_downgrade:
            logger.debug(
                "run_layer3 | request_id=%s | content_type=%s — "
                "downgraded claim_unsupported only (claim_low_confidence retained)",
                request_id, content_type,
            )

    # ── [FIX-cd57ac80] Removal of "claim_contradiction" for instructional ──
    # and definitional content types when no evidence passages exist. The
    # contradiction check targets news/data-style queries; on sequential
    # instructions or comparisons (e.g. encoder vs. decoder) it produced
    # false positives in the project's evaluation.
    _CONTRADICTION_SOFT_WARN_TYPES = {"tutorial", "definition", "comparison", "recipe",
                                      "how_to"}
    if content_type in _CONTRADICTION_SOFT_WARN_TYPES and not evidence_passages:
        pre_contra = failed[:]
        failed = [f for f in failed if not f.startswith("claim_contradiction")]
        if failed != pre_contra:
            downgraded = [f for f in pre_contra if f.startswith("claim_contradiction")]
            logger.warning(
                "run_layer3 | request_id=%s | content_type=%s — "
                "claim_contradiction downgraded to SOFT_WARN: %s "
                "(no tool evidence; instructional/definitional content — "
                "contradiction checker is not reliable here)",
                request_id, content_type, downgraded,
            )

    # ── Removal of "claim_contradiction" for procedural types without evidence ─
    if content_type in {"recipe", "tutorial", "how_to"} and not evidence_passages:
        pre_contra = failed[:]
        failed = [f for f in failed if not f.startswith("claim_contradiction")]
        if failed != pre_contra:
            logger.debug(
                "run_layer3 | request_id=%s | content_type=%s — "
                "suppressed claim_contradiction (no tool evidence, procedural)",
                request_id, content_type,
            )

    # ── Relevance failure for non-factual content types ──────────────────
    # The "relevance" identifier is replaced by "warn_relevance_dialect_mismatch".
    if content_type in NON_FACTUAL_TYPES and "relevance" in failed:
        failed = [f for f in failed if f != "relevance"]
        failed.append("warn_relevance_dialect_mismatch")
        logger.warning(
            "run_layer3 | request_id=%s | content_type=%s — "
            "relevance FAIL downgraded to SOFT_WARN",
            request_id, content_type,
        )

    # Final decision: the output passes only if no failure identifier remains.
    passed     = len(failed) == 0
    elapsed_ms = int((time.monotonic() - start_ts) * 1000)

    _emit_audit_log(
        request_id=request_id, passed=passed, failed=failed,
        pii_found=pii_found, pii_types=pii_types, tox_score=tox_score,
        sem_score=sem_score, rel_score=rel_score,
        policy_matches=policy_matches, claim_result=claim_result,
        elapsed_ms=elapsed_ms, raw_output=raw_output,
        original_query=original_query, islamic_result=islamic_result,
        fp_commodity=_fp_commodity, fp_result=_fp_result,
        freshness_checks=_freshness_checks,
        web_search_retrieved_at=_ws_retrieved_at,
        web_search_data_type=_ws_data_type,
        per_check_latency_ms=per_check_latency_ms,
    )

    return Layer3Output(
        passed=passed, raw_output=raw_output,
        failed_checks=failed, pii_detected=pii_found,
        toxicity_score=round(tox_score, 4),
        claim_verification=claim_result, request_id=request_id,
        islamic_result=islamic_result,
        islamic_no_evidence=islamic_no_evidence,
        # Set once per call; see the Islamic content guard section.
        contains_unverified_religious_content=contains_unverified_religious_content,
        financial_postprocessor_flags=_fp_result.flags if _fp_result else [],
        financial_price_correction_required=_financial_price_correction_required,
        content_type=content_type,
        freshness_checks=_freshness_checks,
        web_search_retrieved_at=_ws_retrieved_at,
        web_search_data_type=_ws_data_type,
        freshness_status=_freshness_status,
        per_check_latency_ms=per_check_latency_ms,
    )


def run_layer3(
    raw_output: str,
    original_query: str,
    config: ValidatorConfig = DEFAULT_CONFIG,
    request_id: Optional[str] = None,
    evidence_passages: Optional[List[str]] = None,
    *,
    query_domain: Optional[str] = None,
) -> Layer3Output:
    """
    Public entry point of Layer 3 (Output Validator Orchestrator).

    Delegates to _run_layer3_impl() and returns its Layer3Output. If any
    exception escapes the implementation, the error is logged and a
    Layer3Output with ``passed=False`` and ``failed_checks=["internal_error"]``
    is returned instead, so the function never raises. This fail-closed
    behaviour ensures that an output whose checks could not be completed is
    not recorded as having passed them.

    Parameters and return value are as documented for _run_layer3_impl().
    """
    try:
        return _run_layer3_impl(
            raw_output, original_query, config, request_id,
            evidence_passages, query_domain=query_domain,
        )
    except Exception as exc:
        rid = request_id or "unknown"
        logger.error(
            "run_layer3 | request_id=%s | unhandled exception — failing "
            "CLOSED (passed=False): %s", rid, exc,
        )
        return Layer3Output(
            passed=False, raw_output=raw_output,
            failed_checks=["internal_error"], request_id=rid,
        )



# Audit log helper


def _emit_audit_log(
    *,
    request_id: str,
    passed: bool,
    failed: List[str],
    pii_found: bool,
    pii_types: List[str],
    tox_score: float,
    sem_score: Optional[float],
    rel_score: Optional[float],
    policy_matches: List[str],
    claim_result: Optional[ClaimVerificationResult],
    elapsed_ms: int,
    raw_output: str,
    original_query: str,
    islamic_result: Optional[IslamicContentResult],
    fp_commodity: Optional[str] = None,
    fp_result: Optional["FinancialPostProcessResult"] = None,
    freshness_checks:          Optional[List[str]] = None,
    web_search_retrieved_at:   Optional[str]       = None,
    web_search_data_type:      str                 = "generic",
    per_check_latency_ms:      Optional[Dict[str, float]] = None,
) -> None:
    """
    Emit one "layer3" event to the Audit Logger (log_event, risk_level="high").

    The record contains the decision (PASS/BLOCK), failed-check identifiers,
    scores, summaries of the claim and Islamic-guard results, financial and
    freshness diagnostics, latencies, and the lengths of the output and
    query. The output and query texts themselves are not written to the
    record.
    """
    audit_status = "PASS" if passed else "BLOCK"

    claim_audit: Optional[dict] = None
    if claim_result is not None:
        claim_audit = {
            "passed":               claim_result.passed,
            "confidence_score":     claim_result.confidence_score,
            "contradictions_found": claim_result.contradictions_found,
            "unsupported_claims":   claim_result.unsupported_claims,
            "failed_checks":        claim_result.failed_checks,
            "verification_error":   claim_result.verification_error,
        }

    islamic_audit: Optional[dict] = None
    if islamic_result is not None:
        islamic_audit = {
            "passed":               islamic_result.passed,
            "failed_checks":        islamic_result.failed_checks,
            "warnings":             islamic_result.warnings,
            "islamic_no_evidence":  islamic_result.islamic_no_evidence,
            "fabricated_citations": islamic_result.fabricated_citations,
            "hajj_errors":          islamic_result.hajj_errors,
        }

    log_event(
        "layer3",
        audit_status,
        {
            "request_id":        request_id,
            "passed":            passed,
            "failed_checks":     failed,
            "pii_detected":      pii_found,
            "pii_types":         pii_types,
            "toxicity_score":    round(tox_score, 4),
            "semantic_score":    round(sem_score, 4) if sem_score is not None else None,
            "relevance_score":   round(rel_score, 4) if rel_score is not None else None,
            "policy_violations": policy_matches,
            "claim_verification": claim_audit,
            "islamic_content":   islamic_audit,
            "financial_postprocessor": {
                "commodity":             fp_commodity,
                "flags":                 fp_result.flags if fp_result else [],
                "stale":                 fp_result.stale if fp_result else None,
                "filtered_snippets":     fp_result.jurisdiction_filtered_snippets
                                         if fp_result else 0,
            },
            "freshness_checks":        freshness_checks or [],
            "web_search_retrieved_at": web_search_retrieved_at,
            "web_search_data_type":    web_search_data_type,
            "elapsed_ms":    elapsed_ms,
            # Per-check latency in milliseconds.
            "per_check_latency_ms": per_check_latency_ms or {},
            "output_length": len(raw_output),
            "query_length":  len(original_query),
        },
        risk_level="high",
    )



# Internal helpers


def _sanitise_input(text: str) -> str:
    """
    Return *text* as a stripped string without zero-width characters.

    None and values that cannot be converted to ``str`` yield "".
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    text = _ZERO_WIDTH_RE.sub("", text)
    return text.strip()


def _normalize_for_policy(text: str) -> str:
    """Return the NFKD-normalised, lower-cased form of *text* for policy matching."""
    return unicodedata.normalize("NFKD", text).lower()



# Arabic / dialectal insult lexicon (check 3, complementary to the classifier)
#
# Single-word insults in Modern Standard Arabic and Algerian dialect. The
# list is restricted to vocabulary that is unlikely to occur in neutral
# educational or historical text, to limit false positives. It is a finite
# list and does not cover all insulting expressions.

_ARABIC_INSULT_WORDS: List[str] = [
    "غبي", "غبية", "أغبى", "اغبى", "حيوان", "حقير", "حقيرة", "فاشل", "فاشلة",
    "أحمق", "احمق", "حمقاء", "تافه", "تافهة", "عديم الفائدة", "عديمة الفائدة",
    "عديم الذكاء", "عديم الفهم", "مقرف", "مقرفة", "متخلف", "متخلفة",
    "مزعج جداً", "مزعج جدا", "حمار", "حثالة", "بليد", "بليدة", "أبله", "ابله",
    "معتوه", "معتوهة", "قذر", "وسخ", "نذل", "نذلة",
]
_ARABIC_INSULT_WORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _ARABIC_INSULT_WORDS) + r")\b",
    re.UNICODE,
)

# Dialectal put-down *patterns*, where the phrase structure rather than a
# single keyword is indicative:
# "ما عندك حتى X" (you don't even have X), "ما تفهمش والو" / "ما تنفع
# لوالو" (you understand/are worth nothing), "روح بعيد/تعلم" dismissals,
# and "انت/أنت + كارثة" (calling a person, not an event, "a disaster").
_ARABIC_PUTDOWN_PATTERNS: List[re.Pattern] = [
    re.compile(r"ما\s+عندك\s+حتى\s+\S+", re.UNICODE),
    re.compile(r"ما\s+تفهمش?\s+(?:ل)?والو", re.UNICODE),
    re.compile(r"ما\s+تنفع\s+(?:ل)?والو", re.UNICODE),
    re.compile(r"روح\s+(?:بعيد|تعلم)", re.UNICODE),
    re.compile(r"(?:انت|إنت|أنت)\s+كارثة", re.UNICODE),
    re.compile(r"جاهل\s+بزاف", re.UNICODE),
]


def _contains_arabic_insult(text: str) -> Tuple[bool, str]:
    """
    Search *text* for lexicon insults and put-down patterns.

    Returns (True, matched_text) for the first match (single-word lexicon
    first, then patterns), otherwise (False, "").
    """
    m = _ARABIC_INSULT_WORD_RE.search(text)
    if m:
        return True, m.group(0)
    for pat in _ARABIC_PUTDOWN_PATTERNS:
        m = pat.search(text)
        if m:
            return True, m.group(0)
    return False, ""


def _detect_pii(text: str) -> Tuple[bool, List[str]]:
    """
    Apply every pattern in PII_PATTERNS to *text*.

    Returns (any_match, names_of_matching_patterns). Matched values are not
    returned, so they do not reach the logs.
    """
    if not text:
        return False, []
    matched = [name for name, pat in PII_PATTERNS.items() if pat.search(text)]
    return bool(matched), matched


def _toxicity_score(text: str, config: ValidatorConfig = DEFAULT_CONFIG) -> float:
    """
    Return the maximum classifier label score over 512-character chunks of *text*.

    Returns 0.0 for empty input, when the classifier is unavailable, or when
    inference fails (the failure is logged). Results are cached by text hash.
    """
    if not text:
        return 0.0
    key = _cache_key(text)
    with _toxicity_cache_lock:
        if key in _toxicity_cache:
            return _toxicity_cache[key]
    classifier = _get_toxicity_classifier(config.toxicity_model)
    if classifier is None:
        return 0.0
    chunk_size = 512
    chunks = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)] or [text]
    max_score = 0.0
    try:
        for chunk in chunks:
            results = classifier(chunk)
            label_scores = results[0] if results and isinstance(results[0], list) else results
            chunk_score  = max(
                (r["score"] for r in label_scores if isinstance(r, dict) and "score" in r),
                default=0.0,
            )
            max_score = max(max_score, chunk_score)
    except Exception as exc:
        logger.warning("Toxicity inference failed: %s", exc)
        max_score = 0.0
    with _toxicity_cache_lock:
        if len(_toxicity_cache) >= config.toxicity_cache_size:
            del _toxicity_cache[next(iter(_toxicity_cache))]
        _toxicity_cache[key] = max_score
    return max_score


def _semantic_consistency_score(
    output: str,
    query: str,
    config: ValidatorConfig = DEFAULT_CONFIG,
) -> float:
    """
    Return a similarity score in [0, 1] between *query* and *output*.

    Backends are tried in order: multilingual sentence-embedding cosine
    similarity, TF-IDF (1-2-gram) cosine similarity, and word-level Jaccard
    similarity. Returns 0.0 if either text is empty. Scores from different
    backends are not calibrated to one another.
    """
    if not output or not query:
        return 0.0
    model = _get_embedding_model(config.embedding_model)
    if model is not None:
        try:
            embs  = model.encode(
                [query, output],
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            score = float(st_util.cos_sim(embs[0], embs[1]).item())
            return max(0.0, min(1.0, score))
        except Exception as exc:
            logger.debug("SentenceTransformer similarity failed: %s", exc)
    if _SKLEARN_AVAILABLE:
        try:
            vec   = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
            tfidf = vec.fit_transform([query, output])
            score = float(sk_cosine(tfidf[0], tfidf[1])[0, 0])
            return max(0.0, min(1.0, score))
        except Exception as exc:
            logger.debug("TF-IDF similarity failed: %s", exc)
    return _jaccard_similarity(query, output)


def _semantic_consistency(
    output: str,
    query: str,
    config: ValidatorConfig = DEFAULT_CONFIG,
) -> bool:
    """Return True if the similarity score reaches config.min_semantic_score."""
    return _semantic_consistency_score(output, query, config) >= config.min_semantic_score


# Clause delimiters (Arabic/Latin comma and semicolon) used for topic-drift
# detection. Periods are deliberately not used, so that a single sentence of
# the form "X, and unrelated Y" is split without fragmenting ordinary
# multi-sentence answers.
_CLAUSE_SPLIT_RE = re.compile(r"[،,؛;]")
_LEADING_CONNECTOR_RE = re.compile(r"^(?:و|أو|ثم|كما)\s*", re.UNICODE)


def _detect_topic_drift(
    output: str,
    query: str,
    config: ValidatorConfig = DEFAULT_CONFIG,
) -> Tuple[bool, Optional[str]]:
    """
    Detect an output that combines an on-topic clause with an unrelated one.

    Returns (drift_detected, offending_clause). The output is split into
    clauses of at least 8 characters; at least two are required. Drift is
    reported only for a mixed case: the best clause scores >= 0.45 and the
    worst clause scores < 0.08 against the query. Uniformly low scores are
    handled by the whole-text semantic check, and uniformly high scores
    (e.g. a multi-step procedure) are not flagged.
    """
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(output) if len(c.strip()) >= 8]
    if len(clauses) < 2:
        return False, None

    effective_query = query
    if _is_dialectal_query(query):
        effective_query = _extract_content_query(query)

    scored = []
    for c in clauses:
        c_stripped = _LEADING_CONNECTOR_RE.sub("", c)
        s = _semantic_consistency_score(c_stripped, effective_query, config)
        scored.append((c, s))

    lo_clause, lo_score = min(scored, key=lambda x: x[1])
    hi_score = max(s for _, s in scored)

    # Limitation: the thresholds were calibrated on scores observed in one
    # evaluation run with the same embedding-model family (on-topic clauses
    # ~0.8, off-topic clauses ~0.09). A terse, correct procedural clause
    # (e.g. "استعمل الاختصار Ctrl+S") was also observed scoring ~0.1 against
    # its own query, close to the off-topic range. The lower threshold
    # (0.08) is kept low to reduce such false positives. The values should
    # be re-validated on the safe set if the model or data change.
    if hi_score >= 0.45 and lo_score < 0.08:
        return True, lo_clause
    return False, None


# Darija marker words, in Latin transliteration and in Arabic script. They
# are used only to decide whether dialectal filler words should be removed
# before relevance scoring; the lists are not exhaustive.
_DARIJA_LATIN_RE = re.compile(
    r"\b(wach|kifah|ndiru|khoya|rani|rahi|ntuma|Ana|hna|huma|bah|wah|ya3ni"
    r"|taw|daba|machi|bach|ila|mliha|bzaf|3lah|3ndek|sahbi|weyn|fin|bezzaf"
    r"|chhal|kima|nta|nti|ntos|kima|walo|3la|fih|fiha|fihom)\b",
    re.IGNORECASE,
)
_DARIJA_AR_RE = re.compile(
    r"\b(واش|كيفاش|نديرو|راني|راهي|نتوما|هنا|هنايا|بصح|ياك|تاع|ديال"
    r"|علاش|فين|كيما|والو|بزاف|شحال|درك|دابا|مشي|باش|ماشي|كاين|كانت)\b",
    re.UNICODE,
)


def _is_dialectal_query(query: str) -> bool:
    """Return True if *query* contains a listed Algerian-dialect (Darija) token."""
    return bool(_DARIJA_LATIN_RE.search(query) or _DARIJA_AR_RE.search(query))


def _extract_content_query(query: str) -> str:
    """
    Remove listed dialectal filler words and surrounding punctuation from *query*.

    The original query is returned if fewer than 5 characters would remain.
    """
    stripped = re.sub(
        r"\b(wach|kifah|ndiru|khoya|rani|rahi|ntuma|ya3ni|taw|daba"
        r"|machi|bach|sahbi|weyn|fin|bah|wah|واش|كيفاش|نديرو|راني"
        r"|راهي|نتوما|يعني|تاع|ديال|ياك|بصح|كاين|درك|دابا)\b",
        " ",
        query,
        flags=re.IGNORECASE | re.UNICODE,
    )
    stripped = re.sub(r"\s+", " ", stripped).strip(" ,،?؟!")
    return stripped if len(stripped) >= 5 else query


def _relevance_score(
    output: str,
    query: str,
    config: ValidatorConfig = DEFAULT_CONFIG,
) -> float:
    """
    Return the query-output relevance score used by check 5.

    Returns 1.0 for an empty query and 0.0 for an empty output. For
    dialectal queries, dialectal filler words are removed before scoring.
    """
    if not query:
        return 1.0
    if not output:
        return 0.0
    effective_query = query
    if _is_dialectal_query(query):
        content_query = _extract_content_query(query)
        if content_query != query:
            logger.debug(
                "_relevance_score | dialect detected — using content query %r",
                content_query[:80],
            )
        effective_query = content_query
    return _semantic_consistency_score(output, effective_query, config)


def _policy_contradiction(text: str) -> Tuple[bool, List[str]]:
    """
    Match *text* (and its NFKD-lowercased form) against _POLICY_PATTERNS.

    Returns (any_match, matched_substrings), one entry per matching pattern.
    The function name is historical: it detects pattern matches, not logical
    contradictions with a policy.
    """
    if not text:
        return False, []
    normalized = _normalize_for_policy(text)
    matches: List[str] = []
    for pattern in _POLICY_PATTERNS:
        m = pattern.search(text) or pattern.search(normalized)
        if m:
            matches.append(m.group(0))
    return bool(matches), matches


def _run_claim_verification_safe(
    raw_output: str,
    original_query: str,
    config: ValidatorConfig,
    evidence_passages: Optional[List[str]] = None,
    *,
    require_tool_evidence: bool = False,
) -> ClaimVerificationResult:
    """
    Run the claim verifier with error containment and no-evidence downgrading.

    If the verifier raises or returns None, a failed result with
    ``claim_model_failure`` and ``verification_error=True`` is returned.
    When neither the configuration nor the caller supplies evidence, the
    failure codes starting with "claim_unsupported" or
    "claim_low_confidence" are removed from the result; any remaining
    failures determine ``passed``.
    """
    try:
        result = run_claim_verification(
            raw_output, original_query, evidence_passages,
            require_tool_evidence=require_tool_evidence,
        )
        if result is None:
            raise ValueError("run_claim_verification returned None")
        _ = (
            result.passed, result.confidence_score,
            result.contradictions_found, result.unsupported_claims,
            result.failed_checks,
        )
        if (
            not config.require_tool_evidence
            and evidence_passages is None
            and not result.verification_error
        ):
            hard_failures = [
                c for c in result.failed_checks
                if not c.startswith("claim_unsupported")
                and not c.startswith("claim_low_confidence")
            ]
            if len(hard_failures) != len(result.failed_checks):
                downgraded = [c for c in result.failed_checks if c not in hard_failures]
                logger.warning(
                    "Claim checks downgraded to warning (no tool evidence): %s",
                    downgraded,
                )
                result = ClaimVerificationResult(
                    passed=(len(hard_failures) == 0),
                    failed_checks=hard_failures,
                    claims=result.claims,
                    confidence_score=result.confidence_score,
                    contradictions_found=result.contradictions_found,
                    unsupported_claims=result.unsupported_claims,
                    verification_error=False,
                )
        # No additional confidence gate is applied here. An earlier revision
        # re-applied config.min_claim_confidence to the mean confidence over
        # all claims (including grounding-exempt ones), which reversed the
        # downgrade above and bypassed claim_verifier's own decision.
        # claim_verifier now decides claim_low_confidence itself, using only
        # non-exempt claims.
        return result
    except Exception as exc:
        logger.error("Claim verification failed: %s", exc)
        return ClaimVerificationResult(
            passed=False, failed_checks=["claim_model_failure"],
            confidence_score=0.0, contradictions_found=0,
            unsupported_claims=0, verification_error=True,
        )


# ── Content-type classifier ───────────────────────────────────────────────────
# Keyword-based heuristics (Arabic and English) assigning a coarse content type
# to a query/output pair. The label controls which claim-related failures are
# downgraded in _run_layer3_impl. The patterns are lexical and may misclassify.

_CONTENT_TYPE_RECIPE = re.compile(
    r"\b(recipe|ingredient|cook|bake|stir|boil|fry|oven|tablespoon|teaspoon"
    r"|وصفة|مكوّنات|طبخ|اطبخ|مقلاة|فرن|ملعقة)\b",
    re.IGNORECASE | re.UNICODE,
)
_CONTENT_TYPE_TUTORIAL = re.compile(
    r"\b(tutorial|how[-\s]to|step[- ]by[- ]step|guide|install|configure"
    r"|دليل|خطوات|كيفية|تثبيت|إعداد|شرح)\b",
    re.IGNORECASE | re.UNICODE,
)
_CONTENT_TYPE_DEFINITION = re.compile(
    r"\b(what is|what are|explain|define|definition|overview|introduction"
    r"|ما هو|ما هي|اشرح|تعريف|مقدمة|نظرة عامة)\b",
    re.IGNORECASE | re.UNICODE,
)
_CONTENT_TYPE_COMPARISON = re.compile(
    r"\b(compare|comparison|difference|versus|vs\.?|pros? and cons?"
    r"|قارن|مقارنة|الفرق|مقابل|مزايا وعيوب)\b",
    re.IGNORECASE | re.UNICODE,
)
_CONTENT_TYPE_ARITHMETIC = re.compile(
    r"(?:"
    r"\b(calculate|compute|how much|how many|convert|eval|evaluate"
    r"|احسب|كم يساوي|ما ناتج|حوّل|اجمع|اطرح|اضرب|اقسم)\b"
    r"|\d[\d\s\+\-\*\/\^\.]+(?:=\s*\?|(?:\s*=\s*\d))?"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def _classify_content_type(query: str, output: str) -> str:
    """
    Classify a query/output pair as recipe, tutorial, definition, comparison,
    arithmetic or factual.

    The query and the first 500 characters of the output are searched with
    the patterns above; the first match in that priority order determines
    the label, and "factual" is the default.
    """
    combined = f"{query} {output[:500]}"
    if _CONTENT_TYPE_RECIPE.search(combined):
        return "recipe"
    if _CONTENT_TYPE_TUTORIAL.search(combined):
        return "tutorial"
    if _CONTENT_TYPE_DEFINITION.search(combined):
        return "definition"
    if _CONTENT_TYPE_COMPARISON.search(combined):
        return "comparison"
    if _CONTENT_TYPE_ARITHMETIC.search(combined):
        return "arithmetic"
    return "factual"


# Content types for which missing tool evidence does not by itself make claims
# "unsupported" (see _run_layer3_impl).
_GROUNDING_EXEMPT_CONTENT_TYPES = {"recipe", "tutorial", "definition", "comparison"}


def _jaccard_similarity(a: str, b: str) -> float:
    """
    Return the Jaccard index of the lower-cased whitespace-separated token
    sets of *a* and *b* (the last-resort similarity backend).

    Two empty inputs yield 1.0; one empty input yields 0.0.
    """
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
