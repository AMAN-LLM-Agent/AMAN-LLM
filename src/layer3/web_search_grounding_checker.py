# src/layer3/web_search_grounding_checker.py
"""

AMAN-LLM
Layer 3 — Output Validator Orchestrator
Sub-component: Web Search Grounding Checker (check code prefix: ``grounding_``)


PURPOSE
    Heuristic post-retrieval check that estimates whether an answer produced
    after a ``web_search`` tool call is supported by the retrieved snippets.
    The module targets two failure modes observed in development traces:

    1. Retrieval noise in list-type answers (trace 36f248b4).
       A generic HTTP client ("Requests") was listed alongside TensorFlow and
       PyTorch in an answer about top Python AI libraries. Each list item is
       therefore compared against the retrieved snippets, and items with
       insufficient lexical support are reported as noise.

    2. Under-grounded news answers (trace 02a369ac).
       An AI-news query yielded a generic thematic summary containing no named
       entities, dates, or source attributions even though retrieval had
       succeeded. The module detects such answers and proposes a refined query.

    A third change forwards the retrieved snippets to the claim verifier
    (trace c2070c15); see EVIDENCE FORWARDING below.

ARCHITECTURAL ROLE
    Layer 3 (Output Validator Orchestrator). The checker runs after a
    web_search invocation in the tool-execution step of the LLM Agent Core and
    its result is consumed by the Output Validator Orchestrator. It does not
    modify the answer text; it only reports findings and recommendations.

INPUTS
    query               original user query (str)
    answer              LLM-synthesised answer or raw tool output (str)
    retrieved_snippets  snippet strings returned by web_search (List[str])
    tool_used           name of the tool that produced the snippets (str)

OUTPUTS
    ``GroundingResult`` with the pass/fail status, check codes, an optional
    re-query recommendation, list items split into evidenced and noise sets,
    simple grounding statistics, and the evidence passages forwarded to claim
    verification.

EVIDENCE FORWARDING (trace c2070c15)
    In an earlier configuration the claim verifier did not receive the
    retrieved evidence and contradicted a correct web_search answer (a claim
    about President Tebboune). ``check_web_search_grounding`` now returns the
    snippets as ``evidence_passages`` and sets ``require_tool_evidence=True``
    when ``tool_used == "web_search"`` and snippets are non-empty. In that
    mode the claim verifier is expected to anchor each claim to the evidence:

      * claim contradicts the retrieved text      -> claim_contradiction (FAIL)
      * claim absent from evidence, not contradicted -> claim_unsupported (WARN)

    The anchoring itself is implemented in the claim verifier, not here.

INTEGRATION POINTS
    * Tool-execution node (Cell 10): call after a successful web_search,
      before ``state["tool_output"]`` is set.
    * Output validator, Check 5d: consume ``GroundingResult.failed_checks``
      via Layer3Output.
    * Output validator, Check 6 (claim verification): consume
      ``GroundingResult.evidence_passages`` and
      ``GroundingResult.require_tool_evidence``.

IMPLEMENTATION NOTES
    * All checks are lexical or regular-expression heuristics. They estimate
      surface-level support and do not establish factual correctness.
    * Item-evidence matching uses token overlap, not embedding similarity
      (see ``_item_in_evidence``).
    * Dependencies: Python standard library only.

REPRODUCIBILITY
    The module is deterministic: identical inputs yield identical outputs.
    All thresholds are module-level constants defined below.

"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# Thresholds


# Minimum token-overlap ratio (fraction of an item's tokens found in a single
# snippet) for a list item to be treated as evidenced. Used by
# ``_item_in_evidence`` via ``filter_list_items``.
_ITEM_EVIDENCE_THRESHOLD = 0.45

# Minimum fraction of sentences that must contain a named-entity-like token
# (name, acronym, year/date, or URL). Answers below this value are flagged as
# under-grounded for news queries.
_MIN_GROUNDED_SENTENCE_FRACTION = 0.30

# Minimum number of distinct named-entity-like tokens required in a news
# answer (and, for non-list/non-news answers, used in the retrieval-overlap
# check).
_MIN_NAMED_ENTITIES_ABS = 1

# Maximum number of characters kept from each snippet when evidence passages
# are forwarded to the claim verifier. Bounds the input length of the
# downstream embedding model.
_SNIPPET_CONTEXT_CHARS = 600


# Named-entity / specificity heuristics


# Approximates "specific" content without an NER model. A match is any of:
#   * a sequence of two or more capitalised English words,
#   * an acronym of 2-10 capital letters (e.g. GPT, NATO),
#   * any Arabic word of at least three characters,
#   * a year in 2000-2099 or a d/m/yyyy-style date,
#   * an http(s) URL,
#   * an Arabic source-attribution phrase followed by an Arabic word.
# Note: the Arabic alternative matches any Arabic word of >= 3 characters, so
# for Arabic text this heuristic is permissive (see the section
# "Potential implementation issue — not modified" in the accompanying notes).
_NAMED_ENTITY_RE = re.compile(
    r"("
    # Capitalised English proper noun (2+ words or a single known acronym)
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+"
    r"|[A-Z]{2,10}"                                  # acronyms: GPT, AI, NATO
    # Arabic proper noun introduced by defined article + context
    r"|[\u0600-\u06FF]{3,}"                          # any Arabic word ≥3 chars
    # Years / dates
    r"|\b20\d{2}\b"                                  # 2000–2099
    r"|\b\d{1,2}/\d{1,2}/20\d{2}\b"
    # URLs / source links
    r"|https?://\S+"
    r"|(?:وفقاً لـ?|حسب|أفادت|أعلن|نشرت)\s+[\u0600-\u06FF]+"
    r")",
    re.UNICODE,
)

# Phrases (English and Arabic) typical of non-specific, thematic answers.
# Two or more hits in a news answer contribute to the under-grounded verdict.
_GENERIC_FILLER_RE = re.compile(
    r"\b(technology advances?|social impact|important developments?"
    r"|various aspects?|many fields?|growing interest"
    r"|تطورات التكنولوجيا|التأثير الاجتماعي|مجالات متعددة"
    r"|اهتمام متزايد|تطورات مهمة|جوانب متنوعة)\b",
    re.IGNORECASE | re.UNICODE,
)

# Detects queries that request a ranked or enumerated set of items
# ("top", "best", "list" and Arabic equivalents).
_LIST_QUERY_RE = re.compile(
    r"\b(top|best|list|أفضل|قائمة|أهم|أبرز|أشهر|أكثر\s+استخداماً"
    r"|أكثر\s+شيوعاً|تعرّف\s+على|تعرف\s+على)\b",
    re.IGNORECASE | re.UNICODE,
)

# Detects queries that ask for recent events ("latest", "news", "today" and
# Arabic equivalents).
_NEWS_QUERY_RE = re.compile(
    r"\b(أحدث|آخر|اليوم|الآن|أخبار|latest|news|recent|today|breaking"
    r"|جديد|جديدة|حديث|حديثة)\b",
    re.IGNORECASE | re.UNICODE,
)


# Dataclass


@dataclass
class GroundingResult:
    """
    Result of the web_search grounding check (Layer 3).

    Attributes
    
    passed : bool
        True when no grounding check failed.
    failed_checks : List[str]
        Check codes consumed by Layer3Output (e.g.
        ``grounding_under_grounded_news``, ``grounding_list_noise:<n>_items``,
        ``grounding_list_noise_high``, ``grounding_no_retrieval_overlap``).
    should_requery : bool
        True when re-running web_search with ``refined_query`` is recommended
        before the answer is synthesised.
    refined_query : str
        Suggested replacement query; empty when no re-query is recommended.
    filtered_items : List[str]
        For list-type queries: items with lexical support in the snippets.
    removed_items : List[str]
        For list-type queries: items without such support (candidate noise).
        The answer text itself is not altered by this module.
    named_entity_count : int
        Number of distinct named-entity-like tokens found in the answer.
    grounded_fraction : float
        Fraction of sentences containing at least one named-entity-like token.
    evidence_passages : List[str]
        Snippets (truncated to ``_SNIPPET_CONTEXT_CHARS``) forwarded to the
        claim verifier. Populated only when ``tool_used == "web_search"``.
    require_tool_evidence : bool
        When True, the claim verifier should run in strict mode and anchor
        each claim against ``evidence_passages`` (trace c2070c15). Set when
        ``tool_used == "web_search"`` and snippets are non-empty.
    """
    passed:               bool       = True
    failed_checks:        List[str]  = field(default_factory=list)
    should_requery:       bool       = False
    refined_query:        str        = ""
    filtered_items:       List[str]  = field(default_factory=list)
    removed_items:        List[str]  = field(default_factory=list)
    named_entity_count:   int        = 0
    grounded_fraction:    float      = 1.0
    # FIX c2070c15: evidence forwarding to claim_verifier
    evidence_passages:    List[str]  = field(default_factory=list)
    require_tool_evidence: bool      = False



# Helpers


def _count_named_entities(text: str) -> int:
    """Return the number of distinct named-entity-like tokens in *text*,
    as matched by ``_NAMED_ENTITY_RE`` (tokens shorter than 2 characters are
    ignored)."""
    matches = _NAMED_ENTITY_RE.findall(text)
    return len(set(m.strip() for m in matches if len(m.strip()) >= 2))


def _sentence_grounded_fraction(text: str) -> float:
    """
    Return the fraction of sentences in *text* that contain at least one
    named-entity-like token or date.

    Sentences are obtained by splitting on terminal punctuation (including the
    Arabic question mark) and newlines; fragments of 15 characters or fewer
    are discarded. If no sentence remains, 1.0 is returned so that very short
    answers are not penalised.
    """
    # Split on terminal punctuation + newlines
    sentences = [s.strip() for s in re.split(r"[.!?؟\n]+", text) if len(s.strip()) > 15]
    if not sentences:
        return 1.0  # short text — don't penalise
    grounded = sum(1 for s in sentences if _NAMED_ENTITY_RE.search(s))
    return grounded / len(sentences)


def _is_list_query(query: str) -> bool:
    """Return True if *query* matches the list-request pattern."""
    return bool(_LIST_QUERY_RE.search(query))


def _is_news_query(query: str) -> bool:
    """Return True if *query* matches the recent-events pattern."""
    return bool(_NEWS_QUERY_RE.search(query))


def _extract_list_items(answer: str) -> List[str]:
    """
    Extract bulleted or numbered list items from *answer*, one per line.

    Recognised line prefixes:
    - Markdown/ASCII bullets: ``-``, ``•``, ``*`` followed by text
    - Numbered items: ``1.``, ``1)``, ``(1)``
    - Arabic ordinal prefixes: أولاً / ثانياً / ... followed by text

    Items shorter than three characters are discarded.
    """
    items: List[str] = []
    lines = answer.split("\n")
    for line in lines:
        line = line.strip()
        # Markdown / ASCII bullet
        m = re.match(r"^[-•*]\s+(.+)", line)
        if m:
            items.append(m.group(1).strip())
            continue
        # Numbered: "1. ", "1) ", "(1) "
        m = re.match(r"^(?:\(?\d+[.)]\s*)(.+)", line)
        if m:
            items.append(m.group(1).strip())
            continue
        # Arabic ordinal prefix
        m = re.match(r"^(?:أولاً|ثانياً|ثالثاً|رابعاً|خامساً|أولا|ثانيا)[:\s]+(.+)", line)
        if m:
            items.append(m.group(1).strip())
            continue
    return [i for i in items if len(i) >= 3]


def _item_in_evidence(item: str, snippets: List[str], threshold: float) -> bool:
    """
    Return True if *item* has lexical support in at least one snippet.

    Method: the item and each snippet are lower-cased and reduced to sets of
    Latin/Arabic tokens of at least three characters. The item counts as
    evidenced when, for some single snippet, the fraction of the item's tokens
    present in that snippet is >= *threshold*.

    Token overlap is used instead of embedding similarity because only a
    coarse filter is required: distinct names such as "Requests" and
    "TensorFlow" share no tokens. Items with no tokens of the required
    length cannot be assessed and are accepted.
    """
    item_tokens = set(re.findall(r"[A-Za-z\u0600-\u06FF]{3,}", item.lower()))
    if not item_tokens:
        return True  # can't judge — let it through

    for snippet in snippets:
        snippet_tokens = set(re.findall(r"[A-Za-z\u0600-\u06FF]{3,}", snippet.lower()))
        if not snippet_tokens:
            continue
        overlap = len(item_tokens & snippet_tokens) / len(item_tokens)
        if overlap >= threshold:
            return True
    return False


def _refine_query_for_news(original_query: str) -> str:
    """
    Build a more specific query for an under-grounded news answer.

    Appends a recency marker (if none of the listed year/period terms is
    already present) and a request for names and dates, in Arabic when the
    query contains Arabic characters and in English otherwise. The aim is to
    bias the search engine towards dated, specific results rather than
    evergreen summaries.
    """
    # Strip any trailing punctuation
    q = original_query.strip().rstrip("؟?")

    # Add a recency signal if not already present
    if not re.search(r"\b(2025|2026|اليوم|هذا الأسبوع|هذا الشهر|هذا العام)\b", q):
        q = q + " 2026"

    # Append a "with names and dates" instruction in the query language
    if re.search(r"[\u0600-\u06FF]", q):
        q = q + " مع أسماء وتواريخ"
    else:
        q = q + " specific named events"

    return q


def _refine_query_for_list(original_query: str) -> str:
    """
    Build a more specific query for a list answer containing retrieval noise
    by appending a request for specific, reliable items (Arabic or English,
    matching the query language).
    """
    q = original_query.strip().rstrip("؟?")
    if re.search(r"[\u0600-\u06FF]", q):
        q = q + " مع أسماء محددة وموثوقة"
    else:
        q = q + " authoritative specific list"
    return q



# Item-level relevance filter


def filter_list_items(
    answer: str,
    retrieved_snippets: List[str],
    threshold: float = _ITEM_EVIDENCE_THRESHOLD,
) -> Tuple[List[str], List[str]]:
    """
    Partition the list items of *answer* into evidenced and noise sets.

    Parameters
    
    answer : str
        LLM-generated answer text.
    retrieved_snippets : List[str]
        Raw snippets returned by web_search.
    threshold : float
        Token-overlap ratio required for an item to be treated as evidenced
        (see ``_item_in_evidence``).

    Returns
    
    Tuple[List[str], List[str]]
        ``(evidenced_items, noise_items)``. Both lists are empty when the
        answer contains no recognisable list items. Each noise item is also
        logged at WARNING level.
    """
    items = _extract_list_items(answer)
    if not items:
        return [], []

    evidenced, noise = [], []
    for item in items:
        if _item_in_evidence(item, retrieved_snippets, threshold):
            evidenced.append(item)
        else:
            noise.append(item)
            logger.warning(
                "[P2-7] list_item_noise: '%s' has no evidence in retrieved snippets",
                item[:80],
            )
    return evidenced, noise



# Main entry point


def check_web_search_grounding(
    query: str,
    answer: str,
    retrieved_snippets: List[str],
    *,
    strict_list_filter: bool = True,
    tool_used: str = "",
) -> GroundingResult:
    """
    Run the grounding checks on an answer derived from a web_search call.

    Parameters
    
    query : str
        Original user query.
    answer : str
        LLM-synthesised answer, or the raw tool output before synthesis.
        Call once per web_search round-trip.
    retrieved_snippets : List[str]
        Snippet strings returned by web_search.
    strict_list_filter : bool
        Intended to select between removing noise items (True) and only
        annotating them (False). In the current implementation this
        parameter is accepted but not read.
    tool_used : str
        Name of the tool that produced the snippets. When it equals
        ``"web_search"`` and snippets are non-empty, the result carries
        ``evidence_passages`` and ``require_tool_evidence=True`` (trace
        c2070c15).

    Returns
    
    GroundingResult
        ``GroundingResult(passed=True)`` is returned immediately when *answer*
        or *retrieved_snippets* is empty.

    Checks
   
    A. Under-grounded news answer (news queries only): fails when the answer
       has fewer than ``_MIN_NAMED_ENTITIES_ABS`` named-entity-like tokens, a
       grounded-sentence fraction below ``_MIN_GROUNDED_SENTENCE_FRACTION``,
       or at least two generic-filler matches. Recommends a re-query.
    B. Retrieval noise (list queries only): fails when at least one list item
       lacks lexical support in the snippets; recommends a re-query when at
       least half of the items lack support.
    C. No retrieval overlap (queries that are neither news nor list): fails
       when the answer covers less than 5% of the combined snippet vocabulary
       (tokens of four or more characters) and also has fewer than
       ``_MIN_NAMED_ENTITIES_ABS`` named-entity-like tokens. Recommends
       re-running the original query.

    Implementation note for Check C: the overlap ratio is computed as
    ``|answer_tokens ∩ snippet_tokens| / |snippet_tokens|``, i.e. normalised
    by the size of the combined snippet vocabulary, not of the answer.
    """
    if not answer or not retrieved_snippets:
        return GroundingResult(passed=True)

    failed_checks: List[str] = []
    should_requery  = False
    refined_query   = ""
    filtered_items: List[str] = []
    removed_items:  List[str] = []

    entity_count      = _count_named_entities(answer)
    grounded_fraction = _sentence_grounded_fraction(answer)
    is_news           = _is_news_query(query)
    is_list           = _is_list_query(query)

    # ── Check A: under-grounded answer (news / recent events) ─────────────
    if is_news:
        generic_hits = len(_GENERIC_FILLER_RE.findall(answer))
        under_grounded = (
            entity_count < _MIN_NAMED_ENTITIES_ABS
            or grounded_fraction < _MIN_GROUNDED_SENTENCE_FRACTION
        )
        if under_grounded or generic_hits >= 2:
            failed_checks.append("grounding_under_grounded_news")
            should_requery = True
            refined_query  = _refine_query_for_news(query)
            logger.warning(
                "[P2-7] under_grounded_news: query=%r | entity_count=%d | "
                "grounded_frac=%.2f | generic_hits=%d | refined_query=%r",
                query[:80], entity_count, grounded_fraction,
                generic_hits, refined_query[:80],
            )

    #  Check B: retrieval noise in list answers 
    if is_list:
        evidenced, noise = filter_list_items(answer, retrieved_snippets)
        filtered_items = evidenced
        removed_items  = noise

        if noise:
            failed_checks.append(f"grounding_list_noise:{len(noise)}_items")
            logger.warning(
                "[P2-7] list_noise: query=%r | noise_items=%s | "
                "evidenced_items=%s",
                query[:80],
                [n[:60] for n in noise],
                [e[:60] for e in evidenced],
            )

            # Recommend a re-query when at least half of the items are noise.
            total = len(evidenced) + len(noise)
            if total > 0 and (len(noise) / total) >= 0.5:
                should_requery = True
                refined_query  = _refine_query_for_list(query)
                failed_checks.append("grounding_list_noise_high")

    #  Check C: answer not distinguishable from a tool-free response 
    if not is_news and not is_list:
        combined_snippet_tokens = set(
            re.findall(r"[A-Za-z\u0600-\u06FF]{4,}", " ".join(retrieved_snippets).lower())
        )
        answer_tokens = set(
            re.findall(r"[A-Za-z\u0600-\u06FF]{4,}", answer.lower())
        )
        if combined_snippet_tokens:
            overlap = len(answer_tokens & combined_snippet_tokens) / len(combined_snippet_tokens)
            if overlap < 0.05 and entity_count < _MIN_NAMED_ENTITIES_ABS:
                failed_checks.append("grounding_no_retrieval_overlap")
                should_requery = True
                refined_query  = query
                logger.warning(
                    "[P2-7] no_retrieval_overlap: query=%r | overlap=%.3f | "
                    "entity_count=%d",
                    query[:80], overlap, entity_count,
                )

    passed = len(failed_checks) == 0

    #  Evidence forwarding to the claim verifier (trace c2070c15) 
    # For web_search answers the retrieved snippets serve as the reference
    # text. They are forwarded so that the claim verifier can anchor each
    # claim against retrieved content rather than against the query alone,
    # which avoids contradicting correct search results.
    forwarded_evidence: List[str] = []
    require_tool_evidence = False

    if tool_used == "web_search" and retrieved_snippets:
        # Truncate each snippet to _SNIPPET_CONTEXT_CHARS to stay within
        # the embedding model's context window.
        forwarded_evidence = [s[:_SNIPPET_CONTEXT_CHARS] for s in retrieved_snippets]
        require_tool_evidence = True
        logger.debug(
            "[P2-7] evidence_forwarding: forwarding %d snippet(s) to claim_verifier "
            "(require_tool_evidence=True)",
            len(forwarded_evidence),
        )

    return GroundingResult(
        passed=passed,
        failed_checks=failed_checks,
        should_requery=should_requery,
        refined_query=refined_query,
        filtered_items=filtered_items,
        removed_items=removed_items,
        named_entity_count=entity_count,
        grounded_fraction=round(grounded_fraction, 3),
        evidence_passages=forwarded_evidence,
        require_tool_evidence=require_tool_evidence,
    )
