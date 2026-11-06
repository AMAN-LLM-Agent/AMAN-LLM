# src/layer3/claim_verifier.py
"""
AMAN-LLM
Layer 3 — Output Validator Orchestrator
Sub-module: Claim Verifier

PURPOSE
    Checks the factual claims contained in an LLM Agent Core response
    against a pool of evidence passages, using multilingual sentence
    embeddings (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
    The module flags claims that (a) have no sufficiently similar evidence,
    (b) appear to contradict other claims or the supplied evidence, or
    (c) state a small set of well-known static numeric facts incorrectly.

ARCHITECTURAL ROLE
    Layer 3 sub-module. It is called by the Output Validator Orchestrator
    after the LLM Agent Core has produced a response. It is a heuristic
    detector: it provides an additional runtime check on response content
    and reduces the risk that unsupported or self-contradictory statements
    are passed on unflagged. It does not establish the truth of a claim,
    and its decisions are bounded by the quality of the evidence supplied.

INPUTS
    output_text        Response text produced by the LLM Agent Core.
    query              The user query associated with the response. It is
                       always inserted as the first evidence anchor.
    evidence_passages  Optional list of retrieved passages (for example,
                       web-search results). May be None or empty.
    require_tool_evidence
                       Keyword-only flag. When True, claims are additionally
                       tested for contradiction against the supplied
                       evidence passages (strict mode).

OUTPUTS
    ClaimVerificationResult, containing the overall pass/fail decision, a
    list of failed-check labels, per-claim ClaimResult records, aggregate
    counts, a mean confidence score, and a `verification_error` flag.
    This module does not itself write audit records; it only emits
    diagnostics through the standard `logging` module.

PIPELINE (run_claim_verification)
    1. Split the response into candidate claim sentences (_extract_claims).
    2. Build a de-duplicated evidence list (query first, then caller-supplied
       passages, then — only if none were supplied — curated snippets).
    3. Embed claims and evidence; compute the cosine-similarity matrix.
    4. Per claim: apply the grounding-exemption taxonomy, test evidence
       support against EVIDENCE_MATCH_THRESHOLD, and run contradiction
       detection (evidence-level in strict mode, inter-claim otherwise).
    5. Apply the known-numeric-fact check to the whole response.
    6. Aggregate into failed-check labels and a ClaimVerificationResult.

IMPLEMENTATION NOTES
    * Failure handling: if the embedding model fails during step 3, the
      function returns passed=False with verification_error=True and the
      label "claim_model_failure". Callers must treat this as a failure,
      not as a pass.
    * Contradiction detection is embedding-based. A claim is negated by a
      rule-based rewrite (_negate); the claim is flagged when the embedding
      of its negation is similar to another claim (or evidence passage)
      above CONTRADICTION_THRESHOLD. Each embedding call is bounded by
      CONTRADICTION_PAIR_TIMEOUT_S via a ThreadPoolExecutor.
    * The inter-claim contradiction loop considers at most
      MAX_CLAIMS_FOR_CONTRADICTION partner claims to bound its quadratic cost.
    * Structural exemptions (sequential steps, temporally scoped claims,
      comparative/contrastive claims) are excluded from inter-claim
      contradiction checks, and definitional / procedural / normative
      sentences are exempt from evidence grounding (see _claim_taxonomy).
    * Language coverage: English, French, Modern Standard Arabic, and
      selected Arabic dialect markers (Egyptian, Gulf, Levantine, Moroccan
      Darija). Pattern lists are heuristic and not exhaustive.
    * The static numeric-fact table and the curated evidence snippets are
      deliberately small. They are not a general knowledge base.
    * Cache sizing: lru_cache(maxsize=2048) on _embed_single_cached holds
      roughly 3 MB for MiniLM-L12 (384-dim x float32 x 2048). If a larger
      embedding model is substituted, reduce maxsize proportionally or use a
      byte-budgeted cache.

REPRODUCIBILITY
    * All thresholds and caps are module-level constants defined below
      ("Thresholds" section). The values are those used by the
      implementation evaluated in the accompanying study; they were chosen
      empirically and are not derived analytically.
    * Embeddings are computed in eval mode, under torch.no_grad(), with
      L2-normalised outputs, so cosine similarity equals the dot product.
    * Model weights are downloaded by `sentence_transformers` on first use;
      the model name is fixed in `_MODEL_NAME`.
    * No random sampling is performed in this module.

REVISION HISTORY (condensed from in-code change notes)
    * Arabic Unicode normalization applied before claim de-duplication.
    * Model failures are reported explicitly (verification_error=True,
      passed=False) rather than passing silently.
    * CONTRADICTION_PAIR_TIMEOUT_S is enforced through a ThreadPoolExecutor.
    * Arabic semicolon (؛) added as a claim delimiter.
    * Arabic negation handling extended to MSA and dialect markers.
    * Inter-claim contradiction loop capped at MAX_CLAIMS_FOR_CONTRADICTION.
    * Dialect-specific non-factual filler patterns added.
    * Length and diversity bonuses removed from confidence scoring.
    * Structural exemptions (sequential / temporal / comparative) added to the
      contradiction loop.
"""
from __future__ import annotations

import concurrent.futures
import logging
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


# Failure sentinel
#
# Intended to represent a hard verification failure (model crash, out-of-
# memory, etc.). Any consumer that receives it must treat it as a FAIL, not
# a pass. See "Potential implementation issue" notes accompanying this file:
# the sentinel is currently defined but not referenced in this module, which
# reports hard failures through ClaimVerificationResult.verification_error.

VERIFICATION_FAILED_SENTINEL = object()


# Embedding model
#
# The model is loaded lazily on first use and shared process-wide. Loading is
# protected by double-checked locking so concurrent callers trigger at most
# one load.


_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_EMBED_MODEL: Optional[SentenceTransformer] = None
_MODEL_LOCK = threading.Lock()


def _get_model() -> SentenceTransformer:
    """
    Return the shared SentenceTransformer instance, loading it on first call.

    Thread-safe (double-checked locking). The model is placed in eval mode.
    A load failure is logged and the exception is re-raised so that the
    caller (run_claim_verification) can convert it into a
    verification_error result.
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        with _MODEL_LOCK:
            if _EMBED_MODEL is None:
                try:
                    m = SentenceTransformer(_MODEL_NAME)
                    m.eval()
                    _EMBED_MODEL = m
                except Exception as exc:
                    logger.error("Failed to load embedding model '%s': %s", _MODEL_NAME, exc)
                    raise
    return _EMBED_MODEL



# Thresholds and limits
#
# All values were selected empirically during development. They are
# calibrated for the multilingual MiniLM-L12 embedding space and should be
# re-evaluated if the embedding model is changed.


# Minimum claim-to-evidence cosine similarity for a factual claim to count as
# "supported". Raised from 0.55 during development because 0.55 admitted too
# many weak matches in the multilingual setting.
EVIDENCE_MATCH_THRESHOLD = 0.65
# Minimum cosine similarity between a negated claim and another claim (or an
# evidence passage) for the claim to be flagged as contradictory. Lowered from
# 0.85 during development because 0.85 was not reached in practice.
CONTRADICTION_THRESHOLD  = 0.72
# Mean confidence over evidence-graded claims below which the response is
# flagged "claim_low_confidence".
LOW_CONFIDENCE_THRESHOLD = 0.45

# Maximum number of characters of the response that are analysed.
MAX_INPUT_CHARS  = 12_000
# Minimum sentence length (characters) for a sentence to be a candidate claim.
MIN_CLAIM_LEN    = 20

# Upper bound on the number of partner claims examined by the inter-claim
# contradiction loop; limits quadratic growth on long outputs.
MAX_CLAIMS_FOR_CONTRADICTION = 30

# Per-call time budget (seconds) for the negated-claim embedding used in
# contradiction checks. Enforced with
# _CONTRA_EXECUTOR.submit(...).result(timeout=...). On timeout the pair (or
# evidence check) is skipped and the event is logged at DEBUG level.
CONTRADICTION_PAIR_TIMEOUT_S = 0.5

# Shared worker pool used to run time-bounded contradiction embeddings.
_CONTRA_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="contra_embed",
)



# Result data structures


@dataclass
class ClaimResult:
    """
    Verification outcome for a single extracted claim sentence.

    Attributes
   
    claim              The claim text as extracted from the response.
    supported          True if the claim is grounding-exempt or its best
                       evidence similarity reaches EVIDENCE_MATCH_THRESHOLD.
    confidence         Best claim-to-evidence cosine similarity clipped to
                       [0, 1]; multiplied by 0.6 when the claim is
                       unsupported.
    evidence_snippet   First 200 characters of the best-matching evidence
                       passage; set only for supported, non-exempt claims.
    contradiction      True if a contradiction was detected for this claim.
    contradiction_with First 120 characters of the conflicting claim or
                       evidence passage, when a contradiction was detected.
    grounding_exempt   True if the claim was accepted through the
                       definitional / procedural / normative exemption
                       instead of being matched against evidence. For such
                       claims `confidence` is only a query-vs-answer
                       similarity (a plausibility proxy that is naturally
                       low for short, correct answers); it does not measure
                       evidence strength and is excluded from the
                       low-confidence gate.
    """
    claim: str
    supported: bool
    confidence: float
    evidence_snippet: Optional[str] = None
    contradiction: bool = False
    contradiction_with: Optional[str] = None
    grounding_exempt: bool = False


@dataclass
class ClaimVerificationResult:
    """
    Aggregate outcome of claim verification for one response.

    Attributes
   
    passed               True when no failed check was recorded.
    failed_checks        Labels of failed checks, e.g. "claim_unsupported:N",
                         "claim_contradiction:N",
                         "claim_contradiction:numeric_fact:<details>",
                         "claim_low_confidence:<score>",
                         "claim_model_failure".
    claims               Per-claim ClaimResult records.
    confidence_score     Mean per-claim confidence (all claims, including
                         grounding-exempt ones). Defaults to 1.0 when no
                         claims were evaluated.
    contradictions_found Number of contradicting claims plus numeric-fact
                         contradictions.
    unsupported_claims   Number of claims with supported == False.
    verification_error   True when a hard model failure occurred. Callers
                         (the Output Validator Orchestrator) must treat this
                         as a FAIL.
    """
    passed: bool
    failed_checks: List[str] = field(default_factory=list)
    claims: List[ClaimResult] = field(default_factory=list)
    confidence_score: float = 1.0
    contradictions_found: int = 0
    unsupported_claims: int = 0
    verification_error: bool = False



# Arabic Unicode normalization


_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")


def _normalize_arabic(text: str) -> str:
    """
    Normalize Arabic orthography so that spelling variants compare equal.

    Used for claim de-duplication and for the known-numeric-fact check. It is
    not applied to the text that is embedded or to the heuristic pattern
    lists, except where stated.

    Steps:
      1. Unicode NFC normalization.
      2. Removal of diacritics (tashkeel).
      3. Alef variants (أ إ آ ٱ) -> bare alef (ا).
      4. Taa marbuta (ة) -> haa (ه); a common MSA/dialect spelling mismatch.
      5. Alef maqsura (ى) -> yaa (ي).
      6. Hamza on waw (ؤ) -> waw (و).
      7. Hamza on yaa (ئ) -> yaa (ي).
    """
    text = unicodedata.normalize("NFC", text)
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    return text



# Non-factual / filler heuristics
#
# Sentences that begin with a greeting, acknowledgement, or assistant
# boilerplate are not treated as claims. Patterns are anchored at the start
# of the sentence and cover English, French, MSA, and selected Egyptian,
# Gulf, Levantine, and Moroccan Darija openers.


_NON_FACTUAL_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in [
        # English greetings / fillers
        r"^\s*(hello|hi|hey|good\s+(morning|afternoon|evening)|greetings)\b",
        r"^\s*(sure|of course|certainly|absolutely|great|ok|okay|thanks|thank you)\b",
        r"^\s*i (hope|think|believe|would|can|will|am here)",
        r"^\s*(feel free|please (let|feel|don't hesitate))",
        r"^\s*(in (this|the following)|below (is|are|you('ll| will)))",
        r"^\s*(as (an|a) (AI|language model|assistant))",
        # Arabic MSA greetings / fillers
        r"^\s*(مرحبا|أهلا|السلام|صباح|مساء|شكرا|بالطبع|بكل سرور)",
        r"^\s*(هل يمكنني|يمكنني|دعني|تفضل|بالتأكيد|من فضلك)",
        # Egyptian dialect fillers
        r"^\s*(إيه الأخبار|عامل إيه|إزيك|أهلاً وسهلاً|تمام|معلش|يلا)",
        # Gulf dialect fillers
        r"^\s*(كيف الحال|شلونك|هلا|والله|ما شاء الله|تفضل)",
        # Levantine dialect fillers
        r"^\s*(كيفك|شو الأخبار|منيح|ولله|يعطيك العافية|مرحبتين)",
        # Moroccan Darija fillers
        r"^\s*(لاباس|كيداير|واخا|بزاف|مزيان|صحيح)",
        # French greetings
        r"^\s*(bonjour|bonsoir|salut|merci|bien sûr|certainement|d'accord)\b",
    ]
]


# Known static numeric facts
#
# Motivation: a sentence such as "الساعة فيها 90 دقيقة" (the hour has 90
# minutes) is structurally almost identical to the true statement "الساعة
# تساوي ستين دقيقة" (the hour equals sixty minutes). Embedding similarity
# between the two is high because the sentence shape matches, so a single
# incorrect number in an otherwise well-formed, on-topic sentence is not
# reliably detected by embedding-based grounding alone.
#
# Mechanism: a small explicit table of static (non-time-varying) counting
# facts. Each entry is keyed by two co-occurring anchor patterns (to limit
# false matches on unrelated sentences) and a known-correct integer. If both
# anchors occur and a different number is stated near the unit word, the
# response is flagged with a numeric-fact contradiction. This is a
# deterministic rule over a deliberately small table, not a general
# arithmetic or fact checker.


# Arabic number words -> integer value. Keys are in *normalized* form
# (alef variants -> ا, ة -> ه) because they are matched against
# _normalize_arabic() output inside _number_near().
_AR_NUM_WORD_MAP: Dict[str, int] = {
    "صفر": 0, "واحد": 1, "واحده": 1, "اثنان": 2, "اثنين": 2, "اثنتان": 2,
    "ثلاثه": 3, "ثلاث": 3, "اربعه": 4, "اربع": 4, "خمسه": 5, "خمس": 5,
    "سته": 6, "ست": 6, "سبعه": 7, "سبع": 7, "ثمانيه": 8, "ثمان": 8,
    "تسعه": 9, "تسع": 9, "عشره": 10, "عشر": 10,
    "احد عشر": 11, "احدى عشر": 11, "اثنا عشر": 12, "اثني عشر": 12,
    "عشرون": 20, "عشرين": 20, "ثلاثون": 30, "ثلاثين": 30,
    "اربعون": 40, "اربعين": 40, "خمسون": 50, "خمسين": 50,
    "ستون": 60, "ستين": 60, "سبعون": 70, "سبعين": 70,
    "ثمانون": 80, "ثمانين": 80, "تسعون": 90, "تسعين": 90,
    "مائه": 100, "مئه": 100,
}
# Alternation of all number words, longest first so multi-word forms match
# before their single-word prefixes.
_AR_NUM_WORD_RE = re.compile(
    r"\b(?:" + "|".join(sorted((re.escape(k) for k in _AR_NUM_WORD_MAP), key=len, reverse=True)) + r")\b",
    re.UNICODE,
)


def _number_near(text: str, anchor_match_end: int, window: int = 18) -> Optional[int]:
    """
    Return the integer stated near a unit word, or None if none is found.

    A window of *window* characters on each side of *anchor_match_end* is
    searched. A digit sequence takes precedence over an Arabic number word.
    *text* is expected to be already normalized with _normalize_arabic().
    """
    span = text[max(0, anchor_match_end - window): anchor_match_end + window]
    digit_m = re.search(r"\d+", span)
    if digit_m:
        return int(digit_m.group(0))
    word_m = _AR_NUM_WORD_RE.search(span)
    if word_m:
        return _AR_NUM_WORD_MAP.get(word_m.group(0))
    return None


# Each entry: (unit-word pattern, co-occurring anchor pattern required in the
# same text, correct integer value, human-readable label).
# Patterns are written against the *normalized* spelling produced by
# _normalize_arabic() (أ/إ/آ/ٱ -> ا, ة -> ه, ئ -> ي), not standard
# orthography, because _check_known_numeric_facts() applies them to
# already-normalized text.
_KNOWN_NUMERIC_FACTS: List[Tuple[re.Pattern, re.Pattern, int, str]] = [
    (re.compile(r"دقيقه"), re.compile(r"ساعه"), 60, "دقائق الساعة"),
    (re.compile(r"ثانيه"), re.compile(r"دقيقه"), 60, "ثواني الدقيقة"),
    (re.compile(r"يوم|ايام"), re.compile(r"اسبوع"), 7, "أيام الأسبوع"),
    (re.compile(r"شهر|اشهر"), re.compile(r"سنه"), 12, "أشهر السنة"),
    (re.compile(r"قاره|قارات"), re.compile(r"عالم|ارض"), 7, "قارات العالم"),
    (re.compile(r"ولايه|ولايات"), re.compile(r"جزاير"), 58, "ولايات الجزائر"),
]


def _check_known_numeric_facts(text: str) -> List[str]:
    """
    Return one finding string per well-known static count stated incorrectly.

    Findings have the form "<label>=<stated>(expected <correct>)". An empty
    list means no table entry was violated. Each violation is also logged at
    WARNING level.
    """
    findings: List[str] = []
    norm = _normalize_arabic(text)
    for unit_re, anchor_re, correct, label in _KNOWN_NUMERIC_FACTS:
        unit_m = unit_re.search(norm)
        if not unit_m or not anchor_re.search(norm):
            continue
        stated = _number_near(norm, unit_m.end())
        if stated is not None and stated != correct:
            findings.append(f"{label}={stated}(expected {correct})")
            logger.warning(
                "[numeric-fact] contradiction: %s stated=%s expected=%s",
                label, stated, correct,
            )
    return findings


# Negation-word detectors (English, Arabic incl. selected dialect markers,
# French). Used only to decide whether the two claims of a pair differ in
# polarity before the topical-similarity pre-filter is applied.
_NEGATION_WORDS_EN = re.compile(
    r"\b(not|never|no|cannot|can't|won't|isn't|aren't|wasn't|weren't"
    r"|doesn't|don't|didn't|neither|nor)\b",
    re.IGNORECASE,
)
_NEGATION_WORDS_AR = re.compile(
    r"\b(لا|ليس|ليست|لم|لن|لما|غير|بدون|عدم|ما|مش|ميش|مو|مب|ماش)\b",
    re.UNICODE,
)
_NEGATION_WORDS_FR = re.compile(r"\b(ne|pas|non|jamais|aucun|sans|ni)\b", re.IGNORECASE)


# Structural-context exemption patterns
#
# Motivation: the embedding-based inter-claim contradiction check can fire on
# pairs that are not contradictory: contrastive statements (encoder vs.
# decoder), sequential steps (learn mathematics before libraries), temporally
# scoped statements (daytime vs. nighttime temperature), and explicit
# comparisons. These structures are detected with marker lists, and matching
# claim pairs are excluded from the inter-claim contradiction check.
# The marker lists are heuristic and can both miss structures and match
# sentences that are not of the intended kind.


# Sequential-step markers: Arabic ordinals and connectives commonly used in
# instructional text. A claim that begins with one of these is treated as a
# step in a sequence rather than a standalone assertion.
_SEQUENTIAL_PREFIXES_AR = re.compile(
    r"^\s*("
    r"أولا|ثانيا|ثالثا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا"  # ordinals
    r"|أولاً|ثانياً|ثالثاً|رابعاً|خامساً"                            # with tanwin
    r"|أول|ثاني|ثالث|رابع|خامس"                                     # short forms
    r"|ثم|بعد ذلك|بعدها|في البداية|في النهاية|أخيرا|أخيراً"         # sequencing adverbs
    r"|الخطوة|المرحلة|في المرحلة|في الخطوة"                          # step/phase nouns
    r")\b",
    re.UNICODE,
)
_SEQUENTIAL_PREFIXES_EN = re.compile(
    r"^\s*(first|second|third|fourth|fifth|"
    r"step \d|stage \d|\d+[.)]\s|"
    r"then|next|after that|finally|lastly|subsequently|before|afterward)\b",
    re.IGNORECASE,
)

# Temporal-scope markers: a claim with an explicit time reference
# (morning/evening, day/night, season, past/future, ...) is assumed to refer
# to a different time window than another temporally marked claim, so the
# pair is not checked for contradiction.
_TEMPORAL_MARKERS_AR = re.compile(
    r"\b(صباح|مساء|نهار|ليل|ليلا|نهارا|صباحا|مساءً|صباحاً"
    r"|في الصباح|في المساء|في الليل|في النهار"
    r"|شتاء|صيف|ربيع|خريف|في الشتاء|في الصيف"
    r"|اليوم|أمس|غدا|غداً|المستقبل|الماضي|الحاضر"
    r"|في السابق|في الوقت الحاضر|في المستقبل"
    r"|درجة الحرارة الدنيا|درجة الحرارة العليا)\b",
    re.UNICODE,
)
_TEMPORAL_MARKERS_EN = re.compile(
    r"\b(morning|evening|night|day|daytime|nighttime|"
    r"summer|winter|spring|autumn|fall|"
    r"today|yesterday|tomorrow|past|present|future|"
    r"maximum temperature|minimum temperature|high of|low of)\b",
    re.IGNORECASE,
)

# Comparison-structure markers: when a sentence explicitly contrasts two
# entities, differing statements about each are intentional and are not
# treated as contradictory.
_COMPARISON_MARKERS_AR = re.compile(
    r"\b(مقارنة|بينما|في حين|على عكس|على النقيض|عكس ذلك"
    r"|من ناحية أخرى|من جهة أخرى|في المقابل|أما|بالمقارنة"
    r"|يختلف|تختلف|يتميز|تتميز|يتعارض|تتعارض)\b",
    re.UNICODE,
)
_COMPARISON_MARKERS_EN = re.compile(
    r"\b(whereas|while|unlike|in contrast|on the other hand|"
    r"however|conversely|differs|on the contrary|"
    r"vs\.?|versus|compared to|as opposed to)\b",
    re.IGNORECASE,
)


def _is_sequential_claim(text: str) -> bool:
    """Return True if *text* opens with a sequential-step marker."""
    return bool(
        _SEQUENTIAL_PREFIXES_AR.match(text)
        or _SEQUENTIAL_PREFIXES_EN.match(text)
    )


def _has_temporal_scope(text: str) -> bool:
    """Return True if *text* contains an explicit temporal reference."""
    return bool(
        _TEMPORAL_MARKERS_AR.search(text)
        or _TEMPORAL_MARKERS_EN.search(text)
    )


def _is_comparative_claim(text: str) -> bool:
    """Return True if *text* contains an explicit comparison/contrast marker."""
    return bool(
        _COMPARISON_MARKERS_AR.search(text)
        or _COMPARISON_MARKERS_EN.search(text)
    )


def _structural_exemption(claim_a: str, claim_b: str) -> bool:
    """
    Return True when the pair (claim_a, claim_b) is exempt from the
    inter-claim contradiction check because of its structural context.

    Exemption classes:
      1. Sequential:  either claim opens with a numbered/ordinal step marker.
      2. Temporal:    both claims carry temporal-scope markers, i.e. they
                      are assumed to describe different time windows.
      3. Comparative: either claim contains an explicit contrast or
                      comparison marker; a stated difference is then
                      intentional.
    """
    # 1. Sequential steps are not treated as contradictions.
    if _is_sequential_claim(claim_a) or _is_sequential_claim(claim_b):
        return True

    # 2. When both claims have temporal markers they are assumed to refer to
    #    different time windows (e.g. night-time vs. daytime temperature).
    if _has_temporal_scope(claim_a) and _has_temporal_scope(claim_b):
        return True

    # 3. Explicit contrast / comparison language makes the difference
    #    intentional rather than contradictory.
    if _is_comparative_claim(claim_a) or _is_comparative_claim(claim_b):
        return True

    return False




# Claim taxonomy — grounding exemption
#
# Rationale: only claims that assert verifiable facts (statistics, named
# events, citations, attributions) are meaningfully checked against retrieved
# evidence passages. Definitional, procedural, and normative statements are
# generally not verifiable against such passages and, if graded against
# them, would inflate the "claim_unsupported" count. These three categories
# are therefore exempt from the evidence-grounding requirement.
#
# Exempt categories:
#   DEFINITIONAL — explains what something is or how it works
#   PROCEDURAL   — step-by-step instructions, recipes, tutorials
#   NORMATIVE    — opinions, recommendations, evaluations
#
# Exempt claims still receive a ClaimResult and still take part in
# contradiction detection; only the evidence-support test is bypassed.


_DEFINITIONAL_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in [
        # English definitional markers
        r"\b(is|are|refers? to|defined? as|means?|known as|called|consists? of"
        r"|composed? of|made (up )?of|characterized by|described as"
        r"|represents?|denotes?|indicates?)\b",
        r"\b(the (purpose|goal|aim|function|role|idea|concept|principle|mechanism)"
        r" of)\b",
        r"\b(in (other )?words|that is( to say)?|i\.e\.?|e\.g\.?|for example"
        r"|for instance|such as|namely|to (illustrate|explain|clarify))\b",
        r"\b(architecture|component|layer|module|mechanism|algorithm|framework"
        r"|process|technique|approach|method|paradigm)\b.{0,40}\b(is|are|works?)\b",
        # Arabic definitional markers
        r"\b(هو|هي|هم|يعني|تعني|يُعرَّف|تُعرَّف|يُشار إليه|يتكون من"
        r"|يتألف من|يتميز بـ?|يُعبِّر عن|يدل على|المقصود|المعنى"
        r"|بمعنى آخر|أي أن|أي|مثلاً|على سبيل المثال)\b",
        r"\b(بنية|مكوّن|طبقة|وحدة|آلية|خوارزمية|إطار|عملية|تقنية|نهج|أسلوب"
        r"|مفهوم|مبدأ)\b.{0,40}\b(هو|هي|يعمل|تعمل)\b",
    ]
]

_PROCEDURAL_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in [
        # English procedural markers
        r"^\s*(first|second|third|step \d|then|next|after( that)?|finally"
        r"|begin by|start by|proceed to|add|mix|stir|heat|cook|bake|pour"
        r"|place|insert|click|open|select|press|run|execute|install|configure"
        r"|enter|type|choose|download|upload|import|export|save|close)\b",
        r"\b(to (do|create|make|build|install|configure|set up|use|run|get"
        r"|achieve|complete|perform|apply|start|begin|finish))\b",
        r"\b(follow (these|the) (steps?|instructions?|guide))\b",
        r"\b(you (need to|should|must|can|have to|will))\b",
        # Arabic procedural markers
        r"^\s*(أولاً|ثانياً|ثالثاً|ثم|بعد ذلك|أضف|اخلط|سخِّن|اطبخ|ضع"
        r"|انقر|افتح|اختر|اضغط|شغِّل|نفِّذ|ثبِّت|اضبط|أدخل|اكتب|احفظ"
        r"|أغلق|استورد|صدِّر)\b",
        r"\b(للقيام بـ?|لإنشاء|لتشغيل|لاستخدام|لتثبيت|لإعداد|للحصول على)\b",
        r"\b(تحتاج إلى|يجب عليك|يمكنك|عليك أن)\b",
    ]
]

_NORMATIVE_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in [
        # English normative / evaluative markers
        r"\b(we (recommend|suggest|advise|encourage|urge|condemn|support|oppose"
        r"|believe|think|consider|endorse|reject))\b",
        r"\b(it is (important|essential|necessary|recommended|advisable|best"
        r"|better|worse|good|bad|wrong|right|crucial|vital|key|useful"
        r"|beneficial|harmful|appropriate|inappropriate))\b",
        r"\b(should|ought to|must|need to|have to)\b.{0,30}\b(in order to"
        r"|to ensure|to prevent|to avoid|to achieve|to maintain)\b",
        r"\b(the (best|worst|most (efficient|effective|common|popular|recommended"
        r"|widely used|important)))\b",
        # Arabic normative markers
        r"\b(نوصي|ننصح|نقترح|نحثّ|ندعو|نرفض|نؤيد|نعارض|نشجب|نُدين)\b",
        r"\b(من المهم|من الضروري|من الأفضل|يُنصح بـ?|يجدر بـ?|ينبغي أن)\b",
        r"\b(الأفضل|الأنسب|الأكثر فعالية|الأكثر شيوعاً|الأكثر استخداماً)\b",
    ]
]

# Factual anchors: a sentence containing any of these is classified as
# factual even if it also contains definitional language, so that it remains
# subject to the grounding check.
_FACTUAL_ANCHORS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in [
        # Statistics / quantities with units
        r"\b\d[\d,\.]+\s*(%|percent|million|billion|trillion|km|kg|°C|°F"
        r"|euros?|dollars?|pounds?|دولار|يورو|مليون|مليار)\b",
        # Named citations / sources
        r"\b(according to|cited by|published (by|in)|reported by|source:?"
        r"|reference:?|وفقاً لـ?|حسب|استناداً إلى|كما أفاد)\b",
        # Named dates
        r"\b(in \d{4}|on \d{1,2}[/-]\d{1,2}|في (عام|سنة) \d{4})\b",
        # Named persons / organizations with attribution verbs
        r"\b(said|stated|announced|confirmed|denied|claimed|argued|wrote"
        r"|قال|أعلن|أكد|نفى|ادّعى|أفاد)\b",
    ]
]


def _claim_taxonomy(sentence: str) -> str:
    """
    Classify *sentence* with a rule-based, regular-expression taxonomy.

    Returns one of:
      'factual'      — statistics, named events, citations, attributions
      'definitional' — explanations of what or how something is
      'procedural'   — step-by-step instructions, recipes
      'normative'    — opinions, recommendations, evaluations

    Only 'factual' claims require evidence grounding; the other three are
    exempt (see _is_grounding_exempt).

    Decision order (first match wins):
      1. Factual anchor present -> 'factual' (overrides all other signals)
      2. Procedural pattern     -> 'procedural'
      3. Normative pattern      -> 'normative'
      4. Definitional pattern   -> 'definitional'
      5. Default                -> 'factual'
    """
    # 1. A factual anchor overrides every other signal.
    for pat in _FACTUAL_ANCHORS:
        if pat.search(sentence):
            return "factual"

    # 2-4. Exempt categories, checked in priority order.
    for pat in _PROCEDURAL_PATTERNS:
        if pat.search(sentence):
            return "procedural"

    for pat in _NORMATIVE_PATTERNS:
        if pat.search(sentence):
            return "normative"

    for pat in _DEFINITIONAL_PATTERNS:
        if pat.search(sentence):
            return "definitional"

    return "factual"


def _is_grounding_exempt(sentence: str) -> bool:
    """
    Return True when *sentence* does not require evidence-passage grounding.

    Exempt: definitional, procedural, normative.
    Not exempt: factual (statistics, citations, named events, attributions).
    """
    return _claim_taxonomy(sentence) != "factual"


# Text helpers


def _is_non_factual(sentence: str) -> bool:
    """Return True if *sentence* matches a greeting/filler pattern."""
    return any(pat.search(sentence) for pat in _NON_FACTUAL_PATTERNS)


def _contains_negation(text: str) -> bool:
    """Return True if *text* contains an English, Arabic, or French negation word."""
    return bool(
        _NEGATION_WORDS_EN.search(text)
        or _NEGATION_WORDS_AR.search(text)
        or _NEGATION_WORDS_FR.search(text)
    )


def _contains_arabic(text: str) -> bool:
    """Return True if *text* contains at least one character in an Arabic Unicode block."""
    return bool(re.search(
        r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]",
        text,
    ))


def _extract_claims(text: str) -> List[str]:
    """
    Split *text* into candidate claim sentences.

    Processing:
      * Truncate to MAX_INPUT_CHARS and normalise line endings.
      * Remove markdown bold/italic markers.
      * Split on terminal punctuation (. ! ? ؟), Arabic comma/semicolon
        (، ؛), newlines, and list-item markers.
      * Discard fragments shorter than MIN_CLAIM_LEN characters and
        fragments matching a greeting/filler pattern.
      * De-duplicate; for Arabic text the comparison key is passed through
        _normalize_arabic() so orthographic variants of the same sentence
        do not produce duplicate claims.

    Returns the surviving sentences in their original order and spelling.
    """
    if not text:
        return []

    text = text[:MAX_INPUT_CHARS]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip markdown bold/italic markers
    text = re.sub(r"[*_]{1,2}(.+?)[*_]{1,2}", r"\1", text)

    raw_parts = re.split(
        r"(?<=[.!?؟])[\s]+"          # after terminal punctuation + whitespace
        r"|(?<=[،؛])\s+"             # after Arabic comma / semicolon
        r"|\n+"                      # newlines / blank lines
        r"|\s*\n\s*[-\u2022\u2013\u2014]\s+"    # list bullets
        r"|\s*\n\s*\d+[.)]\s+",     # numbered items "1. " / "1) "
        text.strip(),
    )

    seen: set = set()
    claims: List[str] = []

    for part in raw_parts:
        sentence = part.strip()
        if len(sentence) < MIN_CLAIM_LEN:
            continue
        if _is_non_factual(sentence):
            continue

        # Normalize before de-duplication so orthographic variants collapse.
        if _contains_arabic(sentence):
            norm_key = _normalize_arabic(re.sub(r"\s+", " ", sentence.lower()))
        else:
            norm_key = re.sub(r"\s+", " ", sentence.lower())

        if norm_key in seen:
            continue
        seen.add(norm_key)
        claims.append(sentence)

    return claims


def _embed(texts: List[str]) -> torch.Tensor:
    """
    Encode *texts* with the shared model.

    Returns an L2-normalised 2-D tensor of shape (len(texts), embedding_dim),
    so that cosine similarity between rows equals their dot product.
    Gradient tracking is disabled.
    """
    model = _get_model()
    with torch.no_grad():
        return model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


@lru_cache(maxsize=2048)
def _embed_single_cached(text: str) -> torch.Tensor:
    """
    Return the cached embedding of a single text (negated claims in the
    contradiction checks).

    Cache budget: 2048 x 384-dim x float32 is approximately 3 MB, which is
    appropriate for MiniLM-L12. For a larger model (e.g. 768-dim mpnet-base),
    reduce maxsize proportionally.
    """
    return _embed([text])[0]


def _negate(sentence: str) -> str:
    """
    Produce a rule-based negated rewrite of *sentence* for contradiction tests.

    The first matching rule (English, then Arabic, then French; first
    occurrence only) is applied. Coverage: English auxiliaries and modals;
    Arabic copula, modal, existence, future, and common past-tense forms;
    Egyptian/Gulf/Levantine dialect negators (removed to yield the
    affirmative); and French auxiliaries. If no rule matches, a
    language-appropriate prefix ("هذا غير صحيح: " or "not true: ") is added.

    The rewrite is approximate and may be ungrammatical; it is used only to
    create an embedding that is close to a statement contradicting the
    original.
    """
    en_map: List[Tuple[str, str]] = [
        (r"\bisn't\b",   "is"),
        (r"\baren't\b",  "are"),
        (r"\bwasn't\b",  "was"),
        (r"\bweren't\b", "were"),
        (r"\bcan't\b",   "can"),
        (r"\bwon't\b",   "will"),
        (r"\bdon't\b",   "do"),
        (r"\bdoesn't\b", "does"),
        (r"\bdidn't\b",  "did"),
        (r"\bis\b",      "is not"),
        (r"\bare\b",     "are not"),
        (r"\bwas\b",     "was not"),
        (r"\bwere\b",    "were not"),
        (r"\bwill\b",    "will not"),
        (r"\bcan\b",     "cannot"),
        (r"\bhas\b",     "has not"),
        (r"\bhave\b",    "have not"),
        (r"\bhad\b",     "had not"),
        (r"\bshould\b",  "should not"),
        (r"\bmust\b",    "must not"),
    ]
    ar_map: List[Tuple[str, str]] = [
        # Copula / to-be
        (r"\bيكون\b",   "لا يكون"),
        (r"\bتكون\b",   "لا تكون"),
        (r"\bأكون\b",   "لا أكون"),
        (r"\bنكون\b",   "لا نكون"),
        (r"\bهو\b",     "ليس هو"),
        (r"\bهي\b",     "ليست هي"),
        (r"\bهم\b",     "ليسوا هم"),
        (r"\bهن\b",     "لسن هن"),
        (r"\bكان\b",    "لم يكن"),
        (r"\bكانت\b",   "لم تكن"),
        (r"\bكانوا\b",  "لم يكونوا"),
        # Modal verbs
        (r"\bيمكن\b",   "لا يمكن"),
        (r"\bيجب\b",    "لا يجب"),
        (r"\bيجوز\b",   "لا يجوز"),
        (r"\bينبغي\b",  "لا ينبغي"),
        (r"\bيحتاج\b",  "لا يحتاج"),
        (r"\bيريد\b",   "لا يريد"),
        (r"\bتريد\b",   "لا تريد"),
        (r"\bأريد\b",   "لا أريد"),
        # Existence
        (r"\bيوجد\b",   "لا يوجد"),
        (r"\bتوجد\b",   "لا توجد"),
        (r"\bوجد\b",    "لم يوجد"),
        (r"\bموجود\b",  "غير موجود"),
        # Future auxiliary
        (r"\bسيكون\b",  "لن يكون"),
        (r"\bسيتم\b",   "لن يتم"),
        (r"\bسوف\b",    "لن"),
        # Past tense common forms
        (r"\bأصبح\b",   "لم يصبح"),
        (r"\bأصبحت\b",  "لم تصبح"),
        (r"\bحدث\b",    "لم يحدث"),
        (r"\bوصل\b",    "لم يصل"),
        (r"\bنجح\b",    "لم ينجح"),
        (r"\bفشل\b",    "لم يفشل"),
        # Dialect negation reversal — strip the marker to produce affirmative
        (r"\bمش\b",     ""),
        (r"\bمو\b",     ""),
        (r"\bمب\b",     ""),
    ]
    fr_map: List[Tuple[str, str]] = [
        (r"\best\b",   "n'est pas"),
        (r"\bsont\b",  "ne sont pas"),
        (r"\bétait\b", "n'était pas"),
        (r"\ba\b",     "n'a pas"),
        (r"\bpeut\b",  "ne peut pas"),
        (r"\bdoit\b",  "ne doit pas"),
    ]

    result = sentence
    for pattern, replacement in en_map + ar_map + fr_map:
        replaced = re.sub(pattern, replacement, result, count=1, flags=re.IGNORECASE)
        if replaced != result:
            return replaced

    if _contains_arabic(sentence):
        return "هذا غير صحيح: " + sentence
    return "not true: " + sentence



# Curated-evidence fallback
#
# Rationale: during development, claim verification showed low accuracy when
# no evidence passages were supplied. In that case the only evidence anchor
# is the query itself, so a correct statement about a stable, well-known fact
# (a currency subunit, a historical date, a static geographic fact) can be
# scored as unsupported.
#
# Scope: this is NOT a general-purpose knowledge base. It is a small,
# conservative set of static, non-time-sensitive facts in a few domains
# (currency, calendar/date facts, geography, units). It does not cover
# current officeholders, prices, or any information that changes over time;
# those require genuine retrieval (the require_tool_evidence path with
# web-search evidence).
#
# Usage: curated snippets are used only when the caller supplied no
# evidence_passages, and they supplement (never replace) the evidence pool.


_CURATED_FACT_SNIPPETS: Dict[str, List[str]] = {
    "currency": [
        "الدينار الجزائري ينقسم إلى مائة سنتيم.",
        "الدينار الجزائري هو العملة الرسمية للجزائر.",
        "اليورو ينقسم إلى مائة سنت، والدولار الأمريكي ينقسم إلى مائة سنت.",
        "الجنيه الإسترليني ينقسم إلى مائة بنس.",
    ],
    "dates_calendar": [
        "استقلت الجزائر في 5 يوليو 1962.",
        "السنة الميلادية تتكون من 365 يوماً، و366 يوماً في السنة الكبيسة.",
        "الشهر الهجري يبدأ برؤية الهلال ويتراوح بين 29 و30 يوماً.",
        "الأسبوع يتكون من سبعة أيام.",
    ],
    "geography": [
        "عاصمة الجزائر هي مدينة الجزائر.",
        "الجزائر أكبر دولة في أفريقيا من حيث المساحة.",
        "نهر النيل هو أطول نهر في أفريقيا.",
        "جبال الأطلس تمتد عبر المغرب والجزائر وتونس.",
    ],
    "units": [
        "الكيلومتر الواحد يساوي ألف متر.",
        "الكيلوغرام الواحد يساوي ألف غرام.",
        "الساعة الواحدة تساوي ستين دقيقة، والدقيقة تساوي ستين ثانية.",
    ],
}

# Flattened (snippet, domain) pairs, built once at import time.
_CURATED_SNIPPET_INDEX: List[Tuple[str, str]] = [
    (snippet, domain)
    for domain, snippets in _CURATED_FACT_SNIPPETS.items()
    for snippet in snippets
]

# Minimum claim-to-snippet cosine similarity for a curated snippet to be used.
# The table is small and static rather than a general retriever, so the bar
# is kept high to avoid injecting an unrelated snippet that merely shares
# vocabulary with a claim.
_CURATED_SNIPPET_MATCH_THRESHOLD = 0.55
# Maximum number of curated snippets injected, to keep the evidence set small
# and auditable.
_MAX_CURATED_SNIPPETS = 4


def _retrieve_curated_evidence(claims_text: List[str]) -> List[str]:
    """
    Return up to _MAX_CURATED_SNIPPETS curated snippets that are similar to
    at least one claim in *claims_text*.

    Snippets are ranked by their best similarity to any claim and kept while
    that similarity is >= _CURATED_SNIPPET_MATCH_THRESHOLD. Best-effort: an
    embedding failure yields an empty list rather than an error.
    """
    if not claims_text or not _CURATED_SNIPPET_INDEX:
        return []
    try:
        snippet_texts = [s for s, _domain in _CURATED_SNIPPET_INDEX]
        claim_embs    = _embed(claims_text)
        snippet_embs  = _embed(snippet_texts)
        sims = util.cos_sim(claim_embs, snippet_embs)  # (n_claims, n_snippets)
        best_per_snippet = sims.max(dim=0).values  # closest claim per snippet
        picked: List[str] = []
        # Rank snippets by their best match to any claim, take the top hits
        # above threshold.
        order = best_per_snippet.argsort(descending=True).tolist()
        for idx in order:
            if len(picked) >= _MAX_CURATED_SNIPPETS:
                break
            score = float(best_per_snippet[idx].item())
            if score < _CURATED_SNIPPET_MATCH_THRESHOLD:
                break
            picked.append(snippet_texts[idx])
        if picked:
            logger.debug(
                "[FIX-claim-kb-fallback] injected %d curated fact snippet(s) "
                "as supplementary evidence (no evidence_passages supplied)",
                len(picked),
            )
        return picked
    except Exception as exc:
        logger.debug("Curated fact-snippet retrieval failed (non-fatal): %s", exc)
        return []



# Core verification pipeline


def run_claim_verification(
    output_text: str,
    query: str,
    evidence_passages: Optional[List[str]] = None,
    *,
    require_tool_evidence: bool = False,
) -> ClaimVerificationResult:
    """
    Verify the factual claims in *output_text* against the available evidence.

    Parameters
   
    output_text : str
        LLM Agent Core response to be checked.
    query : str
        The associated user query. It is always the first evidence anchor.
        When *evidence_passages* is None or empty, the query (plus any
        curated snippets) is the only evidence, so hallucination detection
        is limited in that mode.
    evidence_passages : list of str, optional
        Retrieved passages (e.g. web-search results) against which claims
        are compared.
    require_tool_evidence : bool, keyword-only
        Strict mode, set by the caller (web_search_grounding_checker) when
        the answer was produced from a web_search tool call. See below.

    Strict mode (require_tool_evidence=True, with evidence_passages)
  
    Each non-exempt claim is tested for contradiction against the retrieved
    evidence in addition to the inter-claim check:

      * A claim whose negation is similar (>= CONTRADICTION_THRESHOLD) to a
        retrieved evidence passage is marked as a contradiction
        ("claim_contradiction").
      * A claim that is absent from the evidence but not contradicted is
        marked "claim_unsupported". The intent is that such a claim is
        surfaced as a lower-severity finding; escalation is controlled by
        the Output Validator Orchestrator (config.require_tool_evidence).

    The distinction between contradiction and mere absence is intended to
    avoid false contradictions on correct answers (observed in development
    trace c2070c15) while still surfacing unsupported factual claims.

    When require_tool_evidence=False (default), only the inter-claim
    contradiction check runs; unsupported claims are reported as usual.

    Returns

    ClaimVerificationResult
        ``verification_error=True`` signals a hard model failure. Callers
        (the Output Validator Orchestrator) must treat it as a FAIL.
        An empty response, or one with no extractable claims, returns
        ``passed=True``.
    """
    output_text = (output_text or "").strip()
    query       = (query or "").strip()

    if not output_text:
        return ClaimVerificationResult(passed=True)

    claims_text = _extract_claims(output_text)
    if not claims_text:
        return ClaimVerificationResult(passed=True)

    # Curated-evidence fallback: used only when the caller supplied no
    # evidence at all. It is a lightweight mitigation for the weakness
    # described above (_CURATED_FACT_SNIPPETS); it is not a substitute for
    # genuine retrieval or a curated Arabic knowledge base, and it never
    # overrides caller-supplied evidence.
    _curated_evidence: List[str] = []
    if not evidence_passages:
        _curated_evidence = _retrieve_curated_evidence(claims_text)

    # Build the de-duplicated evidence list (query always first).
    evidence: List[str] = []
    seen_ev: set = set()
    for passage in ([query] + (evidence_passages or []) + _curated_evidence):
        norm = passage.strip()
        if norm and norm not in seen_ev:
            evidence.append(norm)
            seen_ev.add(norm)

    # ── Batch encode ───────────────────────────────────────────────────────
    # A failure here is reported explicitly as verification_error=True /
    # passed=False rather than being treated as a pass.
    try:
        claim_embs    = _embed(claims_text)
        evidence_embs = _embed(evidence)
    except Exception as exc:
        logger.error("Embedding failed during claim verification: %s", exc)
        return ClaimVerificationResult(
            passed=False,
            failed_checks=["claim_model_failure"],
            confidence_score=0.0,
            verification_error=True,
        )

    sim_matrix = util.cos_sim(claim_embs, evidence_embs)   # (n_claims, n_evidence)

    # ── Strict mode: negated-evidence embeddings ───────────────────────────
    # In strict mode, contradiction with retrieved text is distinguished
    # from mere absence. A contradiction requires that (a) the claim is
    # topically close to some evidence passage and (b) the negated claim is
    # similar to that same passage. Claims that are absent but not
    # contradicted are reported only as 'unsupported'.
    evidence_neg_embs: Optional[torch.Tensor] = None
    if require_tool_evidence and evidence_passages:
        try:
            evidence_neg_texts = [_negate(e) for e in evidence]
            evidence_neg_embs  = _embed(evidence_neg_texts)
        except Exception as exc:
            logger.warning(
                "Failed to embed negated evidence for strict mode: %s — "
                "falling back to standard contradiction check.", exc
            )
            evidence_neg_embs = None

    # ── Per-claim verification ─────────────────────────────────────────────
    claim_results: List[ClaimResult] = []
    confidence_sum = 0.0

    # Partner claims for the inter-claim loop (capped for cost control).
    claims_for_contra = claims_text[:MAX_CLAIMS_FOR_CONTRADICTION]

    for i, claim in enumerate(claims_text):
        sims      = sim_matrix[i]
        best_sim  = sims.max().item()
        best_idx  = int(sims.argmax())

        # Grounding exemption: definitional, procedural, and normative claims
        # still receive a ClaimResult (and still take part in contradiction
        # detection), but supported=True is forced so that they never count
        # toward the unsupported total.
        grounding_exempt = _is_grounding_exempt(claim)
        if grounding_exempt:
            supported = True          # never counts as unsupported
        else:
            supported = best_sim >= EVIDENCE_MATCH_THRESHOLD

        # Attach an evidence snippet only for grounded factual claims.
        evidence_snippet: Optional[str] = (
            evidence[best_idx][:200]
            if (supported and not grounding_exempt)
            else None
        )

        # Confidence is the clipped best cosine similarity; unsupported claims
        # are down-weighted by 0.6. No length or diversity bonus is applied.
        base_confidence = min(max(best_sim, 0.0), 1.0)
        if not supported:
            base_confidence *= 0.6
        confidence     = min(1.0, base_confidence)
        confidence_sum += confidence

        # ── Contradiction detection ──────────────────────────────────────
        contradiction      = False
        contradiction_with: Optional[str] = None
        claim_has_negation = _contains_negation(claim)

        # Strict mode: first test the claim against the retrieved evidence.
        # If negate(claim) is similar to an evidence passage, the claim is
        # taken to contradict that passage (hard claim_contradiction). A claim
        # that is merely absent from the evidence (low best_sim) is handled as
        # claim_unsupported in the aggregate step below.
        if require_tool_evidence and evidence_neg_embs is not None and not grounding_exempt:
            try:
                # The negation of the claim should NOT be similar to any
                # evidence passage. If negate(claim) ~ evidence[j], the claim
                # contradicts evidence[j].
                neg_claim_text = _negate(claim)
                future         = _CONTRA_EXECUTOR.submit(_embed_single_cached, neg_claim_text)
                neg_claim_emb  = future.result(timeout=CONTRADICTION_PAIR_TIMEOUT_S)

                ev_contra_sims = util.cos_sim(
                    neg_claim_emb.unsqueeze(0), evidence_embs
                )[0]  # (n_evidence,)

                ev_contra_best     = float(ev_contra_sims.max().item())
                ev_contra_best_idx = int(ev_contra_sims.argmax())

                if ev_contra_best >= CONTRADICTION_THRESHOLD:
                    # The negation of this claim closely matches retrieved
                    # evidence, i.e. the claim conflicts with the retrieved text.
                    contradiction      = True
                    contradiction_with = evidence[ev_contra_best_idx][:120]
                    logger.warning(
                        "[c2070c15] claim_contradicts_evidence: "
                        "claim=%r sim=%.2f evidence=%r",
                        claim[:80], ev_contra_best,
                        contradiction_with[:60],
                    )
            except concurrent.futures.TimeoutError:
                logger.debug("Evidence contradiction check timed out for claim %d", i)
            except Exception as exc:
                logger.debug("Evidence contradiction check failed for claim %d: %s", i, exc)

        # Inter-claim contradiction loop (runs in all modes unless a
        # contradiction with evidence was already found).
        if not contradiction:
            for j, other_claim in enumerate(claims_for_contra):
                if j == i:
                    continue

                # Skip structurally exempt pairs (sequential steps,
                # temporally scoped pairs, comparison clauses).
                if _structural_exemption(claim, other_claim):
                    continue

                other_has_negation = _contains_negation(other_claim)

                if claim_has_negation == other_has_negation:
                    # Same polarity: skip topically distant pairs early.
                    claim_other_sim = util.cos_sim(
                        claim_embs[i].unsqueeze(0),
                        claim_embs[j].unsqueeze(0),
                    ).item()
                    if claim_other_sim < 0.60:
                        continue

                # Time-bounded embedding of the negated claim; on timeout the
                # pair is skipped (see CONTRADICTION_PAIR_TIMEOUT_S).
                try:
                    neg_text = _negate(claim)
                    future   = _CONTRA_EXECUTOR.submit(_embed_single_cached, neg_text)
                    neg_emb  = future.result(timeout=CONTRADICTION_PAIR_TIMEOUT_S)
                    contra_sim = util.cos_sim(
                        neg_emb.unsqueeze(0), claim_embs[j].unsqueeze(0)
                    ).item()
                except concurrent.futures.TimeoutError:
                    logger.debug(
                        "Contradiction check timed out (>%.1fs) for claim %d vs %d",
                        CONTRADICTION_PAIR_TIMEOUT_S, i, j,
                    )
                    continue
                except Exception as exc:
                    logger.debug("Contradiction check failed for claim %d vs %d: %s", i, j, exc)
                    continue

                if contra_sim >= CONTRADICTION_THRESHOLD:
                    contradiction      = True
                    contradiction_with = other_claim[:120]
                    break

        claim_results.append(
            ClaimResult(
                claim=claim,
                supported=supported,
                confidence=round(confidence, 4),
                evidence_snippet=evidence_snippet,
                contradiction=contradiction,
                contradiction_with=contradiction_with,
                grounding_exempt=grounding_exempt,
            )
        )

    # ── Aggregate ──────────────────────────────────────────────────────────
    n                    = len(claim_results)
    mean_confidence      = confidence_sum / n if n else 0.0
    contradictions_found = sum(1 for c in claim_results if c.contradiction)
    unsupported_claims   = sum(1 for c in claim_results if not c.supported)

    # Low-confidence gate: judged only over claims whose confidence reflects
    # real evidence matching. For grounding-exempt claims, "confidence" is
    # merely answer-vs-query cosine similarity, which is naturally low for
    # short, well-formed, correct answers (a question and its answer are
    # lexically different). Gating on it produced false positives during
    # development. If every claim is exempt there is no evidence-graded
    # signal, so the low-confidence check is skipped for that response; the
    # unsupported and contradiction checks are unaffected.
    grounded_results = [c for c in claim_results if not c.grounding_exempt]
    grounded_confidence = (
        sum(c.confidence for c in grounded_results) / len(grounded_results)
        if grounded_results else None
    )

    # Known-numeric-fact check over the full response; see the table
    # definition for why embedding similarity alone does not detect this
    # class of error.
    numeric_fact_findings = _check_known_numeric_facts(output_text)
    if numeric_fact_findings:
        contradictions_found += len(numeric_fact_findings)

    failed: List[str] = []
    if unsupported_claims > 0:
        failed.append(f"claim_unsupported:{unsupported_claims}")
    if contradictions_found > 0:
        failed.append(f"claim_contradiction:{contradictions_found}")
    if numeric_fact_findings:
        failed.append(f"claim_contradiction:numeric_fact:{';'.join(numeric_fact_findings)}")
    if grounded_confidence is not None and grounded_confidence < LOW_CONFIDENCE_THRESHOLD:
        failed.append(f"claim_low_confidence:{grounded_confidence:.2f}")

    return ClaimVerificationResult(
        passed=len(failed) == 0,
        failed_checks=failed,
        claims=claim_results,
        confidence_score=round(mean_confidence, 4),
        contradictions_found=contradictions_found,
        unsupported_claims=unsupported_claims,
        verification_error=False,
    )
